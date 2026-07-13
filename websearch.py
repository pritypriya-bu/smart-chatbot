"""
websearch.py - Live internet search with Tavily and a DuckDuckGo fallback.

Priority:
  1) Tavily      -> best quality (free tier, needs a key from https://tavily.com)
  2) DuckDuckGo  -> no key required (via the `ddgs` library); used when Tavily
                    isn't configured.

Both require an internet connection.
"""

from __future__ import annotations
import requests

TAVILY_URL = "https://api.tavily.com/search"

# The DuckDuckGo client library was renamed from `duckduckgo_search` to `ddgs`.
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def _tavily(query, api_key, max_results=5):
    """Query Tavily and return a formatted results string, or None on failure."""
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": True,
        "search_depth": "basic",
    }
    try:
        r = requests.post(TAVILY_URL, json=payload, timeout=25)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException:
        return None

    lines = []
    if data.get("answer"):
        lines.append(f"Quick answer: {data['answer']}")
        lines.append("")
    for i, res in enumerate(data.get("results", []), 1):
        title = res.get("title", "")
        url = res.get("url", "")
        content = (res.get("content", "") or "")[:400]
        lines.append(f"[{i}] {title}\n{content}\nSource: {url}\n")
    return "\n".join(lines).strip() or None


def _ddg(query, max_results=5):
    """Query DuckDuckGo and return formatted results, or None on failure."""
    if DDGS is None:
        return None
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return None
    if not results:
        return None
    lines = []
    for i, res in enumerate(results, 1):
        title = res.get("title", "")
        url = res.get("href") or res.get("url", "")
        body = (res.get("body", "") or "")[:400]
        lines.append(f"[{i}] {title}\n{body}\nSource: {url}\n")
    return "\n".join(lines).strip() or None


def web_search(query, tavily_key="", max_results=5):
    """
    Search the web using the best available backend.

    Returns (ok: bool, text: str, source: str). Tavily is tried first when a
    key is provided; otherwise DuckDuckGo is used.
    """
    if tavily_key:
        out = _tavily(query, tavily_key, max_results)
        if out:
            return True, out, "Tavily"
    out = _ddg(query, max_results)
    if out:
        return True, out, "DuckDuckGo"
    # Nothing found or both backends unavailable
    if not tavily_key and DDGS is None:
        return False, (
            "Web search needs either a Tavily key (in the sidebar) or "
            "the ddgs library. Install it with:  pip install ddgs"
        ), ""
    return False, "Web search returned no results (network or service issue).", ""


def is_search_query(prompt: str) -> bool:
    """Cheap keyword check: does this prompt need live web data?"""
    p = prompt.lower()
    keys = [
        "latest", "today's news", "news today", "breaking news", "current news",
        "match score", "live score", "score of", "who won", "cricket score",
        "stock price", "share price", "price of", "current price",
        "trending", "recent news", "aaj ki news", "taaza khabar", "kaun jeeta",
        "live match", "election result", "kya hua", "who is the current",
        "latest news", "search for", "google karo", "internet se",
        # Live sporting events / tournaments
        "fifa", "world cup", "olympics", "ipl", "champions league",
        "semifinal", "quarterfinal", "final match", "final matches",
        "matches", "fixture", "fixtures", "schedule", "results of",
        # News / current-affairs style questions
        "list down", "list the", "who is playing", "when is", "when was",
        "yesterday", "tomorrow", "this week", "this month", "this year",
    ]
    return any(k in p for k in keys)
