"""
Claude API client.

Wraps the Anthropic API in a clean function so the rest of the app
doesn't need to know about retries, JSON parsing, or error shapes.

Reads ANTHROPIC_API_KEY from environment. Never hardcoded.
"""

import os
import re
import json
import logging
from typing import Tuple, Optional

from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

# Sonnet 4.6 - most recent Sonnet, smart enough for coaching prompts, cheaper than Opus
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096  # weekly plan with 7 reasoning blocks can run long


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
    # markdown fences. Strip those robustly: handles ```json, ``` , and
    # partially-formed fences. Also handles cases where Claude prepends
    # text before the JSON block.
    cleaned = _extract_json(raw_text)

    parsed: Optional[dict] = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Claude returned non-JSON response: %s", raw_text[:500])

    # Token usage for cost monitoring
    if hasattr(msg, "usage"):
        logger.info(
            "Claude call complete - input tokens: %d, output tokens: %d",
            msg.usage.input_tokens, msg.usage.output_tokens,
        )

    return raw_text, parsed


def _extract_json(text: str) -> str:
    """Pull a JSON object out of model output, handling all the ways Claude
    might wrap it: ```json fences, plain ``` fences, leading prose before
    the JSON, etc. Returns the best candidate string for json.loads().

    Strategy:
      1. If wrapped in fences (``` or ```json), extract the fenced content.
      2. Otherwise, find the first '{' and matching final '}' and return that.
      3. Fallback: return the original text and let the caller see the parse fail.
    """
    if not text:
        return text
    s = text.strip()

    # Case 1: fenced. Match ```json ... ``` or ``` ... ```. Tolerates
    # missing closing fence (truncated responses).
    fence_match = re.match(
        r"^```(?:json|JSON)?\s*\n(.*?)(?:\n```|\Z)",
        s, flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Case 2: prose then JSON. Find first '{' and slice from there to the
    # last '}' (inclusive). Loose but works for our prompt format.
    first_brace = s.find("{")
    last_brace  = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return s[first_brace:last_brace + 1].strip()

    return s