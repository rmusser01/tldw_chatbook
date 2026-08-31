"""Pick the next provider when the primary one will not recover.

ADR-110. Retry (TASK-25901) covers a provider that will be back in seconds;
this covers one that will not -- it is out of credit, or it has failed every
retry. The two compose: the chain is only consulted after retries are exhausted
or on a credit/quota-terminal class.

Resolution is separated from switching so the interesting decisions -- which
candidates are usable, in what order, and which failures earn a fallback at all
-- are testable without a provider, a network, or a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from tldw_chatbook.Chat.Chat_Deps import ChatAPIError

from .native_tools import provider_supports_native_tools

#: Money/quota is gone. Retrying cannot fix it and neither can waiting, so the
#: chain is consulted immediately rather than after the retry budget.
#: 402 is the explicit "payment required"; 403 in practice carries quota
#: exhaustion for several providers alongside genuine permission errors --
#: falling back on it costs one wasted attempt at worst, and the alternative is
#: stalling a run on a plan that has run out.
_CREDIT_TERMINAL_STATUS = frozenset({402, 403})


@dataclass(frozen=True)
class FallbackCandidate:
    """One entry in a resolved chain."""

    provider: str
    #: Whether the TARGET speaks native tool-calls. Carried here because the
    #: switch needs it to know which way to project the history (ADR-110).
    native: bool
    ready: bool
    skip_reason: str = ""


def is_credit_terminal(exc: BaseException) -> bool:
    """Whether ``exc`` means this provider is out of money or quota.

    Deliberately narrow. A 429 is retry's job, and 401/400 are the user's to
    fix -- handing those to another provider hides a problem the user needs to
    see rather than solving one.
    """
    if not isinstance(exc, ChatAPIError):
        return False
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _CREDIT_TERMINAL_STATUS


def resolve_fallback_chain(
    configured: Iterable[object] | None,
    primary: str,
    is_ready: Callable[[str], bool],
) -> list[FallbackCandidate]:
    """Resolve the configured chain into ordered, readiness-tagged candidates.

    Unready candidates are RETAINED in the result rather than filtered out, so
    the caller can report that a configured provider was skipped and why. A
    user who lists a provider they never set up should learn that, not silently
    get a shorter chain than they believe they configured (AC#3).

    Args:
        configured: The user's ordered provider list, as configured.
        primary: The provider that just failed; never a fallback for itself.
        is_ready: Readiness probe, injected so this stays testable and so a
            broken probe cannot take the run down.

    Returns:
        Candidates in configured order, duplicates collapsed.
    """
    if not configured:
        return []

    seen: set[str] = set()
    candidates: list[FallbackCandidate] = []
    primary_key = str(primary or "").strip().lower()

    for raw in configured:
        if not isinstance(raw, str):
            continue
        provider = raw.strip()
        if not provider:
            continue
        key = provider.lower()
        if key == primary_key or key in seen:
            continue
        seen.add(key)

        try:
            ready = bool(is_ready(provider))
            reason = "" if ready else "not configured"
        except Exception as exc:  # noqa: BLE001 -- a probe must not end a run
            ready = False
            reason = f"readiness check failed ({type(exc).__name__})"

        candidates.append(
            FallbackCandidate(
                provider=provider,
                native=provider_supports_native_tools(provider),
                ready=ready,
                skip_reason=reason,
            )
        )
    return candidates
