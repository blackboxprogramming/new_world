# Placeholder for substrate arbitrator logic
# Implements Equation 13: Base-switching optimization for multi-substrate selection
#
# Ollama routing: any request that contains @copilot, @lucidia, or
# @blackboxprogramming is sent exclusively to the local Ollama instance.
# No external AI provider is contacted.

from ..ollama_router import route as ollama_route


def arbitrate(request_text: str, **kwargs) -> str:
    """
    Top-level entry point for substrate arbitration.

    If the request mentions @copilot, @lucidia, or @blackboxprogramming the
    query is forwarded **directly** to the local Ollama service.  No cloud or
    third-party AI provider is involved.
    """
    response = ollama_route(request_text, **kwargs)
    if response is not None:
        return response
    # No trigger found – proceed with normal substrate selection logic
    return _default_arbitration(request_text)


def _default_arbitration(request_text: str) -> str:
    """Stub for future multi-substrate selection logic (Equation 13)."""
    return ""
