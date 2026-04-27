

import os
import json
import logging
from typing import Tuple, Optional

from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

# Sonnet 4.6 - most recent Sonnet, smart enough for coaching prompts, cheaper than Opus
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048


_client = None


def get_client() -> Anthropic:
    """Lazy-init the Anthropic client. Raises clear error if key is missing."""
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )
        _client = Anthropic(api_key=key)
    return _client


def call_claude(system_prompt: str, user_prompt: str) -> Tuple[str, dict]:
    """
    Calls Claude with the given prompts and returns (raw_text, parsed_json).

    The prompt builder asks Claude for a strict JSON response. We parse it here.
    If Claude returns text that isn't valid JSON, we raise so the caller can
    fall back gracefully (or log the bad output for debugging).

    Returns:
        (raw_text: the full text Claude returned,
         parsed:   the dict from json.loads, or None if parse failed)

    Raises:
        RuntimeError on API connection / auth errors (the caller should catch
        and show the user a friendly message).
    """
    client = get_client()

    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except APIConnectionError as e:
        raise RuntimeError(f"Could not reach Anthropic API: {e}") from e
    except RateLimitError as e:
        raise RuntimeError(f"Rate limited by Anthropic API: {e}") from e
    except APIError as e:
        # Includes 401 (bad key), 400 (bad request), etc.
        raise RuntimeError(f"Anthropic API error: {e}") from e

    # Pull text from the first content block. Claude sometimes returns multiple
    # blocks if it uses tools, but for our prompt it's just one text block.
    raw_text = ""
    for block in msg.content:
        if hasattr(block, "text"):
            raw_text += block.text

    raw_text = raw_text.strip()

    # The prompt asks for strict JSON. Sometimes models still wrap it in
    # markdown fences anyway. Strip those if present before parsing.
    cleaned = raw_text
    if cleaned.startswith("```"):
        # Remove first fence line
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    parsed: Optional[dict] = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Claude returned non-JSON response: %s", raw_text[:300])

    # Token usage for cost monitoring
    if hasattr(msg, "usage"):
        logger.info(
            "Claude call complete - input tokens: %d, output tokens: %d",
            msg.usage.input_tokens, msg.usage.output_tokens,
        )

    return raw_text, parsed
