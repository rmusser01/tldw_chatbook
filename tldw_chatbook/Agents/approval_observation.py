"""Content-free, optional approval observations. Never an authorization input."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Literal

SCOPES = frozenset(
    {
        "approve_once",
        "approve_session",
        "allow_matching",
        "always_allow",
        "raw_shell_session",
        "deny",
    }
)
_OUTCOMES = {
    "received": {"received"},
    "settled": {"accepted", "timeout", "cancelled", "revoked"},
    "grant": {"applied", "not_applied", "failed", "unknown"},
    "dispatch_started": {"starting"},
    "backend_started": {"running"},
    "tool_completed": {
        "success",
        "failure",
        "blocked",
        "timeout",
        "cancelled",
        "unknown",
    },
    "model_wait": {"waiting"},
}
_ERRORS = frozenset(
    {
        "",
        "writer_failed",
        "writer_unavailable",
        "authority_unavailable",
        "scope_unavailable",
    }
)


@dataclass(frozen=True)
class ApprovalObservationIdentity:
    session_id: str
    run_id: str
    round_id: str
    revision: int
    call_key: str = ""


@dataclass(frozen=True)
class ApprovalObservation:
    identity: ApprovalObservationIdentity
    kind: str
    outcome: str
    actual_scope: str | None = None
    error_code: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _OUTCOMES or self.outcome not in _OUTCOMES[self.kind]:
            raise ValueError("Invalid approval observation")
        if self.actual_scope is not None and self.actual_scope not in SCOPES:
            raise ValueError("Invalid approval scope")
        if self.error_code not in _ERRORS:
            raise ValueError("Invalid approval observation error code")


@dataclass(frozen=True)
class ApprovalObservationContext:
    identity: ApprovalObservationIdentity
    sink: Callable[[ApprovalObservation], None]

    def publish(self, kind: str, outcome: str, **kwargs) -> None:
        try:
            self.sink(ApprovalObservation(self.identity, kind, outcome, **kwargs))
        except Exception:
            # A display consumer must not affect verdicts, writes or release.
            pass


_CURRENT: ContextVar[tuple[ApprovalObservationContext, ...]] = ContextVar(
    "approval_observation", default=()
)
_CONTEXT_LOCK = Lock()


@contextmanager
def approval_observation_scope(
    identity: ApprovalObservationIdentity, sink: Callable[[ApprovalObservation], None]
) -> Iterator[None]:
    with approval_contexts_scope((ApprovalObservationContext(identity, sink),)):
        yield


@contextmanager
def approval_contexts_scope(
    contexts: tuple[ApprovalObservationContext, ...],
) -> Iterator[None]:
    token = _CURRENT.set(contexts)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_approval_observations() -> tuple[ApprovalObservationContext, ...]:
    """Copy only display attribution across a worker/coroutine boundary."""
    return _CURRENT.get()


def publish_grant_application(
    outcome: Literal["applied", "not_applied", "failed"],
    *,
    actual_scope: str | None = None,
    error_code: str = "",
) -> None:
    """Publish an actual writer fact, safely, without a permission return value."""
    for context in _CURRENT.get():
        context.publish(
            "grant", outcome, actual_scope=actual_scope, error_code=error_code
        )


def remember_approval_contexts(owner: object, run_id: str, decisions: Mapping) -> None:
    """Retain optional metadata separately from a provider's unchanged stamps."""
    with _CONTEXT_LOCK:
        runs = getattr(owner, "_approval_observation_runs", None)
        if runs is None:
            runs = {}
            setattr(owner, "_approval_observation_runs", runs)
        runs.pop(run_id, None)
        contexts = getattr(decisions, "observation_contexts", {})
        if contexts:
            runs[run_id] = (
                dict(contexts),
                {
                    name: tuple(keys)
                    if len(keys) == 1
                    and keys[0] in getattr(decisions, "observation_legacy_keys", ())
                    else ()
                    for name, keys in getattr(
                        decisions, "observation_aliases", {}
                    ).items()
                },
            )


def provider_approval_contexts(
    owner: object, tool_name: str
) -> tuple[ApprovalObservationContext, ...]:
    """Prefer exact call identity; name fallback must identify one verdict group."""
    from .run_context import current_run_id, current_tool_call_id

    with _CONTEXT_LOCK:
        contexts, aliases = getattr(owner, "_approval_observation_runs", {}).get(
            current_run_id(), ({}, {})
        )
        call = current_tool_call_id()
        if call and call in contexts:
            return (contexts[call],)
        keys = aliases.get(tool_name, ())
        if len(keys) == 1 and keys[0] in contexts:
            return (contexts[keys[0]],)
    return ()


def decision_approval_contexts(
    decisions: Mapping, keys: tuple[str, ...]
) -> tuple[ApprovalObservationContext, ...]:
    contexts = getattr(decisions, "observation_contexts", {})
    return tuple(contexts[key] for key in dict.fromkeys(keys) if key in contexts)


def observe_grant_errors(function):
    """Preserve an owner's exception while observing only a controlled failure."""
    from functools import wraps

    @wraps(function)
    def observed(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            publish_grant_application("failed", error_code="writer_failed")
            raise

    return observed


def forget_approval_contexts(owner: object, run_id: str) -> None:
    """Release only retired display attribution, never provider stamps."""
    with _CONTEXT_LOCK:
        getattr(owner, "_approval_observation_runs", {}).pop(run_id, None)


@contextmanager
def provider_observation_scope(
    owner: object, run_id: str, *, clear: bool = False
) -> Iterator[None]:
    """Mirror nested stamp lifetime using only the separate display slice."""
    with _CONTEXT_LOCK:
        runs = getattr(owner, "_approval_observation_runs", {})
        saved = runs.get(run_id)
        if clear:
            runs.pop(run_id, None)
    try:
        yield
    finally:
        with _CONTEXT_LOCK:
            runs = getattr(owner, "_approval_observation_runs", {})
            runs.pop(run_id, None)
            if saved is not None:
                runs[run_id] = saved
