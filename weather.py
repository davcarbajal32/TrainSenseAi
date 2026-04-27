import logging
import requests
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 5  # seconds


def geocode_city(city: str) -> Optional[Tuple[float, float, str]]:
    """Look up (lat, lon) for a city string. Returns None on any failure
    (network error, no match, malformed response). Caller should handle None
    by leaving the profile's lat/lon as null - Claude prompts still work."""
    if not city or not city.strip():
        return None

    try:
        r = requests.get(
            GEOCODE_URL,
            params={"name": city.strip(), "count": 1, "format": "json"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("Geocoding failed for %r: %s", city, e)
        return None

    results = data.get("results") or []
    if not results:
        return None

    top = results[0]
    try:
        lat = float(top["latitude"])
        lon = float(top["longitude"])
    except (KeyError, TypeError, ValueError):
        return None

    # Build a resolved display name like "Corona, California, US"
    parts = [top.get("name")]
    if top.get("admin1"): parts.append(top["admin1"])
    if top.get("country_code"): parts.append(top["country_code"])
    resolved = ", ".join(p for p in parts if p)

    return (lat, lon, resolved)


def get_today_weather(lat: float, lon: float) -> Optional[dict]:
    """Fetch today's forecast at (lat, lon). Returns a dict with the fields
    the prompt builder uses, or None on failure. Errors are logged, not raised
    - weather is not critical to recommendations."""
    if lat is None or lon is None:
        return None

    try:
        r = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,precipitation,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "temperature_unit": "fahrenheit",
                "forecast_days": 1,
                "timezone": "auto",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("Weather lookup failed: %s", e)
        return None

    current = data.get("current") or {}
    daily   = data.get("daily") or {}

    temp_now = current.get("temperature_2m")
    precip_now = current.get("precipitation", 0) or 0
    code = current.get("weather_code")

    # Open-Meteo weather codes: https://open-meteo.com/en/docs
    code_phrase = _weather_code_phrase(code)

    summary_parts = []
    if temp_now is not None:
        summary_parts.append(f"{round(temp_now)}F")
    if code_phrase:
        summary_parts.append(code_phrase)
    if precip_now > 0:
        summary_parts.append(f"{precip_now:.1f}mm precip currently")

    return {
        "summary":      ", ".join(summary_parts) if summary_parts else "weather data limited",
        "temp_f":       temp_now,
        "raining":      precip_now > 0,
        "high_f":       (daily.get("temperature_2m_max") or [None])[0],
        "low_f":        (daily.get("temperature_2m_min") or [None])[0],
        "precip_mm":    (daily.get("precipitation_sum") or [None])[0],
    }


def _weather_code_phrase(code) -> str:
    """Translate Open-Meteo WMO code to a short phrase."""
    if code is None:
        return ""
    table = {
        0: "clear",
        1: "mostly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "freezing fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
        56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain",
        66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow",
        77: "snow grains",
        80: "light showers", 81: "showers", 82: "violent showers",
        85: "snow showers", 86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with hail", 99: "severe thunderstorm",
    }
    return table.get(int(code), "")
