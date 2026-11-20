"""Console's request capacity rule: the reply reservation against the window.

TASK-34100.2 review round 2 (R2-4): moved out of ``console_prepared_request``
so readiness can apply the send preflight's exact rule without importing the
request builder (the Agents runtime, ``LLM_API_Calls``, ``Chat_Functions`` --
about 120 modules) on the UI thread at first paint. ``console_prepared_request``
re-exports every name here, so existing callers are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Chat.console_history_budget import DEFAULT_RESPONSE_RESERVATION

MINIMUM_SAFETY_MARGIN_TOKENS = 512
#: TASK-34100.2 AC#1: an *estimated* window never lets the reply reservation
#: take more than its input share (``estimated_window_reply_cap``), and never
#: caps it below this fraction of the window.
ESTIMATED_WINDOW_RESERVATION_DIVISOR = 4
#: The smallest *estimated* window that leaves any input (pass 3 review round
#: 1, R1-6): at or below 682 tokens the 512-token safety margin plus the
#: quarter-window reservation fill it whatever the configured reply. AC#1's
#: "never pre-blocked" holds above this floor; every window the resolver
#: estimates (``token_counter.PROVIDER_CONTEXT_WINDOWS``, the 32,000-token
#: application fallback) is far above it -- a test pins both.
ESTIMATED_WINDOW_INPUT_FLOOR_TOKENS = 683
LimitSource = Literal["detected", "estimated", "provider_input_cap", "user_override", "unknown"]


@dataclass(frozen=True, slots=True)
class ConsoleRequestCapacity:
    """Resolved provider/model limits and response reservation."""

    context_window_tokens: int | None
    provider_input_cap_tokens: int | None
    provider_output_cap_tokens: int | None
    requested_response_tokens: int
    effective_response_tokens: int
    safety_margin_tokens: int
    effective_input_ceiling_tokens: int | None
    limit_source: LimitSource
    safety_verified: bool


def request_safety_margin_tokens(window_tokens: int) -> int:
    """Tokens a request keeps free of a ``window_tokens`` context window.

    The Console gateway refuses a request over ``window - reply reserve -
    this margin`` (``resolve_request_capacity``); the first-request planner
    plans within the same ceiling (``first_request_window.planning_window``)
    so it never picks a tool surface the gateway then refuses
    (TASK-34100.5.3 review round 2, V2-2).
    """
    return max(MINIMUM_SAFETY_MARGIN_TOKENS, window_tokens // 50)


def estimated_window_reply_cap(window_tokens: int) -> int:
    """The largest reply an *estimated* context window may reserve.

    TASK-34100.2 AC#1: a guessed window must never be pre-blocked to zero
    input by a reservation as large as the guess (the first-run OpenAI
    default: a 4,096 window under a 4,096 reservation). Half the window, after
    the safety margin, stays for input; a configured reply that already
    leaves that much is never cut (review round 1, F5: a flat ``window // 4``
    cut every unlisted Gemini id's 8,192-token reply to 7,680).

    Args:
        window_tokens: The estimated context window.

    Returns:
        The reservation ceiling, at least a quarter of the window. It leaves
        input only for a window of at least
        ``ESTIMATED_WINDOW_INPUT_FLOOR_TOKENS``; below that the safety margin
        alone fills it.
    """
    margin = request_safety_margin_tokens(window_tokens)
    return max(
        1,
        window_tokens // ESTIMATED_WINDOW_RESERVATION_DIVISOR,
        window_tokens - margin - window_tokens // 2,
    )


def resolve_request_capacity(
    *,
    context_window_tokens: int | None,
    provider_input_cap_tokens: int | None = None,
    provider_output_cap_tokens: int | None = None,
    requested_response_tokens: int | None = None,
    context_window_override_tokens: int | None = None,
    context_window_verified: bool = True,
) -> ConsoleRequestCapacity:
    """Resolve input capacity without silently reserving only half a window."""

    for name, value in (
        ("context_window_tokens", context_window_tokens),
        ("provider_input_cap_tokens", provider_input_cap_tokens),
        ("provider_output_cap_tokens", provider_output_cap_tokens),
        ("context_window_override_tokens", context_window_override_tokens),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer when supplied.")

    requested = (
        DEFAULT_RESPONSE_RESERVATION
        if requested_response_tokens is None
        else requested_response_tokens
    )
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("requested_response_tokens must be a positive integer.")
    effective_response = (
        min(requested, provider_output_cap_tokens)
        if provider_output_cap_tokens is not None
        else requested
    )
    effective_window = context_window_override_tokens or context_window_tokens
    source: LimitSource
    if context_window_override_tokens is not None:
        source = "user_override"
    elif context_window_tokens is not None:
        source = "detected" if context_window_verified else "estimated"
    elif provider_input_cap_tokens is not None:
        source = "provider_input_cap"
    else:
        source = "unknown"
    if source == "estimated" and effective_window is not None:
        # TASK-34100.2 AC#1: see ``estimated_window_reply_cap``.
        effective_response = min(
            effective_response, estimated_window_reply_cap(effective_window)
        )
    margin = (
        request_safety_margin_tokens(effective_window)
        if effective_window is not None
        else 0
    )
    candidates: list[int] = []
    if effective_window is not None:
        candidates.append(effective_window - effective_response - margin)
    if provider_input_cap_tokens is not None:
        candidates.append(provider_input_cap_tokens)
    ceiling = min(candidates) if candidates else None
    return ConsoleRequestCapacity(
        context_window_tokens=effective_window,
        provider_input_cap_tokens=provider_input_cap_tokens,
        provider_output_cap_tokens=provider_output_cap_tokens,
        requested_response_tokens=requested,
        effective_response_tokens=effective_response,
        safety_margin_tokens=margin,
        effective_input_ceiling_tokens=max(0, ceiling) if ceiling is not None else None,
        limit_source=source,
        safety_verified=(source == "detected" and effective_window is not None),
    )
