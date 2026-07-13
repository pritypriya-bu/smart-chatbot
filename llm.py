"""
llm.py - Free LLM backends for Smart Chatbot.

All backends are free to use. Ollama is the default (100% local, no API key
required). Groq and Gemini are also supported via their free tiers.

Supported providers:
  - "ollama"  -> local, free, no key           (RECOMMENDED)
  - "groq"    -> free tier, needs GROQ_API_KEY  (very fast)
  - "gemini"  -> free tier, needs GEMINI_API_KEY
"""

from __future__ import annotations
import os
import json
import requests


class LLMError(Exception):
    """Raised when an LLM call fails."""
    pass


# ----------------------------------------------------------------------------
# OLLAMA  (local, free, no API key)
# Install: https://ollama.com  then run:  ollama pull qwen2.5-coder
# ----------------------------------------------------------------------------
def _ollama_chat(messages, model, host, temperature=0.2, timeout=300):
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise LLMError(
            "Could not connect to Ollama. Is it running?\n"
            "  1) Install from https://ollama.com\n"
            "  2) Run:  ollama pull qwen2.5-coder\n"
            "  3) Ollama should run automatically in the background."
        )
    except requests.exceptions.Timeout:
        raise LLMError("Ollama timed out. Try a smaller model (e.g. llama3.2).")
    except requests.exceptions.HTTPError as e:
        raise LLMError(f"Ollama HTTP error: {e}. Is the model '{model}' pulled?")

    data = r.json()
    return data.get("message", {}).get("content", "").strip()


def ollama_models(host):
    """Return the list of locally installed Ollama models."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ----------------------------------------------------------------------------
# GROQ  (free tier, very fast - needs GROQ_API_KEY)
# Get a key: https://console.groq.com/keys
# ----------------------------------------------------------------------------
def _groq_chat(messages, model, api_key, temperature=0.2, timeout=120):
    if not api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys"
        )
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise LLMError(f"Groq error: {e} - {r.text[:300]}")
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Groq request failed: {e}")
    return r.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------------------
# GEMINI  (free tier - needs GEMINI_API_KEY)
# Get a key: https://aistudio.google.com/apikey
# ----------------------------------------------------------------------------
def _gemini_chat(messages, model, api_key, temperature=0.2, timeout=120):
    if not api_key:
        raise LLMError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey"
        )
    # OpenAI-compatible endpoint
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise LLMError(f"Gemini error: {e} - {r.text[:300]}")
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Gemini request failed: {e}")
    return r.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------------------
# Unified entry point
# ----------------------------------------------------------------------------
class LLM:
    """Unified interface across free LLM providers."""

    def __init__(self, provider="ollama", model=None,
                 ollama_host="http://localhost:11434",
                 groq_key=None, gemini_key=None):
        self.provider = provider
        self.ollama_host = ollama_host
        self.groq_key = groq_key or os.getenv("GROQ_API_KEY", "")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY", "")

        defaults = {
            "ollama": "qwen2.5-coder",
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-2.0-flash",
        }
        self.model = model or defaults.get(provider, "qwen2.5-coder")

    def chat(self, messages, temperature=0.2):
        """Send a full message list and return the assistant reply text."""
        if self.provider == "ollama":
            return _ollama_chat(messages, self.model, self.ollama_host, temperature)
        if self.provider == "groq":
            return _groq_chat(messages, self.model, self.groq_key, temperature)
        if self.provider == "gemini":
            return _gemini_chat(messages, self.model, self.gemini_key, temperature)
        raise LLMError(f"Unknown provider: {self.provider}")

    def ask(self, system, user, temperature=0.2):
        """Convenience helper: send one system + one user message."""
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        return self.chat(msgs, temperature)

    def ask_json(self, system, user, temperature=0.0):
        """Ask the LLM for JSON output and parse it safely."""
        raw = self.ask(system, user, temperature)
        return _extract_json(raw)


def _extract_json(text):
    """Extract the first valid JSON object from an LLM response."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                text = p
                break
    # Fall back to the outermost { ... } span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
