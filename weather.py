"""
weather.py - Live weather via Open-Meteo (100% free, no API key).

Two steps:
  1) geocode(city)          -> resolve city name to latitude/longitude
  2) get_weather(lat, lon)  -> current conditions + today's forecast

Requires internet. No key or signup needed.
"""

from __future__ import annotations
import requests

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes -> human-readable description
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def geocode(city: str):
    """Resolve a city name to a location dict, or None if not found."""
    try:
        r = requests.get(
            GEO_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
    except requests.exceptions.RequestException:
        return None
    if not results:
        return None
    g = results[0]
    return {
        "name": g.get("name", city),
        "country": g.get("country", ""),
        "admin1": g.get("admin1", ""),
        "lat": g["latitude"],
        "lon": g["longitude"],
    }


def get_weather(lat: float, lon: float):
    """Fetch current + daily forecast for a coordinate. None on failure."""
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                   "weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,"
                 "precipitation_probability_max",
        "timezone": "auto",
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def weather_summary(city: str):
    """
    Build a human-readable summary of the current weather for a city.

    Returns (ok: bool, text: str). On success, `text` contains location plus
    current conditions and today's forecast, formatted for the LLM to consume.
    """
    loc = geocode(city)
    if not loc:
        return False, f"Could not find a place called '{city}'. Please check the spelling."

    data = get_weather(loc["lat"], loc["lon"])
    if not data or "current" not in data:
        return False, "Weather service returned no data. Please try again in a moment."

    cur = data.get("current", {})
    daily = data.get("daily", {})
    unit = data.get("current_units", {})

    place = loc["name"]
    if loc.get("admin1"):
        place += f", {loc['admin1']}"
    if loc.get("country"):
        place += f", {loc['country']}"

    desc = _WMO.get(cur.get("weather_code"), "unknown")
    tunit = unit.get("temperature_2m", "°C")

    lines = [f"Live weather for {place}:"]
    lines.append(f"- Condition: {desc}")
    if "temperature_2m" in cur:
        lines.append(f"- Temperature: {cur['temperature_2m']}{tunit}")
    if "apparent_temperature" in cur:
        lines.append(f"- Feels like: {cur['apparent_temperature']}{tunit}")
    if "relative_humidity_2m" in cur:
        lines.append(f"- Humidity: {cur['relative_humidity_2m']}%")
    if "wind_speed_10m" in cur:
        lines.append(
            f"- Wind: {cur['wind_speed_10m']} {unit.get('wind_speed_10m', 'km/h')}"
        )

    # Today's high/low and rain chance
    try:
        hi = daily["temperature_2m_max"][0]
        lo = daily["temperature_2m_min"][0]
        lines.append(f"- Today's high/low: {hi}{tunit} / {lo}{tunit}")
        rain = daily.get("precipitation_probability_max", [None])[0]
        if rain is not None:
            lines.append(f"- Chance of rain today: {rain}%")
    except (KeyError, IndexError, TypeError):
        pass

    return True, "\n".join(lines)


def is_weather_query(prompt: str) -> bool:
    """Cheap keyword check: does this prompt look like a weather question?"""
    p = prompt.lower()
    keys = ["weather", "mausam", "forecast", "temperature outside", "how hot",
            "how cold", "raining", "barish", "climate today", "kitni garmi",
            "kitni thand", "will it rain"]
    return any(k in p for k in keys)
