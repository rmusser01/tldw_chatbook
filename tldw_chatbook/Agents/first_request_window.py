"""The context window a Console agent's first request is planned against.

TASK-34100.5 AC#6 (new-entry-exit-handoff-01). The first-request planner
sized the agent preamble (operating prompt plus the fenced protocol for every
runtime tool, ~20.8 KB) against ``get_model_token_limit`` -- for a model no
catalog knows, the 32,000-token application fallback. A llama.cpp server
started with ``-c 4096`` then refused the very first "hi" (4,730 tokens),
while the Console's own send preflight had used the window attached to the
send's resolution. The planner now takes that same window. A window nobody
verified (a provider or application fallback) is planned as the smallest
common local window, with the reply reservation capped to a quarter of it,
so an unknown server gets a plain request instead of an overflow; a verified
window (catalog, server metadata, user override) is used exactly.
"""

from __future__ import annotations

from typing import Any

#: The window an unverified guess is planned as: llama.cpp's and Ollama's
#: common small default, so a guessed window can never license a preamble
#: that overflows a small local server.
UNVERIFIED_PLANNING_WINDOW_TOKENS = 4096
#: An unverified planning window reserves at most this share for the reply.
UNVERIFIED_RESERVATION_DIVISOR = 4


def planning_window(window: Any, response_reserve_tokens: int) -> tuple[int, int] | None:
    """Return ``(context_limit, response_reserve)`` to plan a first request with.

    Args:
        window: The send's ``ContextWindowResolution`` (``tokens``,
            ``verified``), or anything else for "not supplied".
        response_reserve_tokens: The configured reply reservation.

    Returns:
        The window and reservation to plan against, or ``None`` when no usable
        window was supplied (the caller keeps its own resolution).
    """
    tokens = getattr(window, "tokens", None)
    if type(tokens) is not int or tokens <= 0:
        return None
    if getattr(window, "verified", False) is True:
        return tokens, response_reserve_tokens
    tokens = min(tokens, UNVERIFIED_PLANNING_WINDOW_TOKENS)
    return tokens, min(
        response_reserve_tokens, max(1, tokens // UNVERIFIED_RESERVATION_DIVISOR)
    )
