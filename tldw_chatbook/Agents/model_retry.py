"""Decide whether a failed model request is worth trying again.

`run_agent_loop` used to turn any `call_model` exception into a terminal error,
so a single 429 or dropped connection discarded the whole run along with every
tool result already in it.

Classification is by exception TYPE, never by matching the message text. An auth
failure, a malformed request or a content-policy refusal cannot succeed on a
second attempt: retrying spends the user's money and their wall budget to reach
the same place. Anything unrecognised is treated as terminal for the same
reason -- failing closed here costs one run, failing open costs a retry storm
against a provider that is telling us to stop.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)

#: Provider errors that cannot be fixed by asking again.
_TERMINAL_TYPES: tuple[type[Exception], ...] = (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
)

#: Transport failures. These are the ordinary shape of a dropped or slow
#: connection and are exactly what a retry exists for.
_TRANSIENT_TYPES: tuple[type[Exception], ...] = (
    ConnectionError,        # covers ConnectionResetError
    TimeoutError,
)

#: 5xx means the provider failed to serve a request it accepted; 429 means it
#: accepted it and wants us to wait. Both are worth another attempt. 4xx other
#: than 429 is our request being wrong, which a retry cannot fix.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try before giving up."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0


def is_transient_model_error(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying.

    Order matters: the terminal provider types are checked before the generic
    `ChatAPIError` status test, because several of them carry 5xx defaults
    (`ChatConfigurationError` is a 500) and would otherwise read as retryable.
    """
    if isinstance(exc, _TERMINAL_TYPES):
        return False
    if isinstance(exc, ChatRateLimitError):
        return True
    if isinstance(exc, ChatProviderError):
        return _status_of(exc) in _RETRYABLE_STATUS
    if isinstance(exc, ChatAPIError):
        return _status_of(exc) in _RETRYABLE_STATUS
    return isinstance(exc, _TRANSIENT_TYPES)


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _usable_retry_after(exc: BaseException | None) -> float | None:
    """A provider-supplied delay, if it is one we can actually use."""
    if exc is None:
        return None
    raw = getattr(exc, "retry_after", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value <= 0 or math.isnan(value) or math.isinf(value):
        return None
    return value


def retry_delay_seconds(
    attempt: int,
    exc: BaseException | None,
    policy: RetryPolicy,
    jitter: Callable[[], float] = random.random,
) -> float:
    """How long to wait before attempt ``attempt`` + 1.

    A usable ``Retry-After`` wins over the computed backoff -- the provider
    knows when it will serve us -- but is still capped, so a hostile or broken
    value cannot park the run.

    Jitter is applied to the backoff rather than being optional: without it
    every client that failed at the same moment retries at the same moment and
    re-creates the stampede that caused the rate limit.
    """
    supplied = _usable_retry_after(exc)
    if supplied is not None:
        return min(supplied, policy.max_delay)

    exponential = policy.base_delay * (2 ** max(0, attempt - 1))
    capped = min(exponential, policy.max_delay)
    # Full jitter over [capped/2, capped]: still backs off, but spreads.
    return capped * (0.5 + 0.5 * _bounded_unit(jitter))


def _bounded_unit(jitter: Callable[[], float]) -> float:
    try:
        value = float(jitter())
    except Exception:  # noqa: BLE001 -- a bad jitter source must not fail a run
        return 1.0
    if math.isnan(value):
        return 1.0
    return min(1.0, max(0.0, value))
