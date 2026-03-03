# ollama_router.py
# Routes all @copilot, @lucidia, and @blackboxprogramming mentions directly to
# the local Ollama instance.  No external AI provider is used.

import re
import urllib.request
import urllib.error
import json
from typing import Optional

# Trigger mentions that unconditionally route to Ollama
OLLAMA_TRIGGERS = re.compile(
    r"@(copilot|lucidia|blackboxprogramming)\b",
    re.IGNORECASE,
)

# Default Ollama endpoint (local hardware, private network)
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


def contains_ollama_trigger(text: str) -> bool:
    """Return True if the text contains any mention that must go to Ollama."""
    return bool(OLLAMA_TRIGGERS.search(text))


def _strip_triggers(text: str) -> str:
    """Remove @mention prefixes before forwarding the prompt."""
    return OLLAMA_TRIGGERS.sub("", text).strip()


def query_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    stream: bool = False,
) -> str:
    """
    Send a prompt directly to the local Ollama instance and return the response.

    Raises ``ConnectionError`` if Ollama is unreachable so callers know
    immediately that no external fallback will be attempted.
    """
    url = f"{base_url}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": stream}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
            return body.get("response", "")
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Ollama is not reachable at {base_url}. "
            "Ensure the Ollama service is running on your local machine. "
            f"Original error: {exc}"
        ) from exc


def route(text: str, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL) -> Optional[str]:
    """
    Inspect *text* for @copilot / @lucidia / @blackboxprogramming mentions.

    * If a trigger is found  → strip the mention and send to Ollama exclusively.
    * If no trigger is found → return ``None`` (caller may handle normally).

    No external AI provider is ever contacted.
    """
    if not contains_ollama_trigger(text):
        return None
    clean_prompt = _strip_triggers(text)
    return query_ollama(clean_prompt, model=model, base_url=base_url)
