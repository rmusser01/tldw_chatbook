from __future__ import annotations

from collections.abc import Iterable
from inspect import getattr_static
from typing import Any, Callable

from .engine import PolicyEngine
from .registry import CAPABILITY_REGISTRY
from .source_state import normalize_runtime_source_state
from .types import CapabilityEntry, PolicyDecision, PolicyDeniedError, RuntimeSourceState


class ServicePolicyEnforcer:
    """Shared runtime-policy hard-stop seam for service-level operations."""

    def __init__(
        self,
        *,
        state_provider: Callable[[], RuntimeSourceState | None],
        engine: PolicyEngine | None = None,
    ) -> None:
        self._state_provider = state_provider
        self._engine = engine or PolicyEngine(CAPABILITY_REGISTRY)

    @classmethod
    def from_runtime_policy_context(cls, context: Any) -> "ServicePolicyEnforcer":
        return cls(
            state_provider=lambda: context.state,
        )

    def current_state(self) -> RuntimeSourceState | None:
        state = self._state_provider()
        if isinstance(state, RuntimeSourceState):
            return state
        return None

    def require_allowed(
        self,
        *,
        action_id: str,
        runtime_state_override: RuntimeSourceState | None = None,
    ) -> None:
        state = (
            runtime_state_override
            if isinstance(runtime_state_override, RuntimeSourceState)
            else self.current_state()
        )
        if state is None:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code="authority_denied",
                user_message="Runtime policy state is unavailable.",
                effective_source="unknown",
                authority_owner="unknown",
            )
        decision = self._engine.evaluate(
            action_id=action_id,
            state=state,
        )
        if decision.allowed:
            return
        raise PolicyDeniedError(
            action_id=action_id,
            reason_code=decision.reason_code or "authority_denied",
            user_message=decision.user_message,
            effective_source=decision.effective_source,
            authority_owner=decision.authority_owner,
        )


# Used only by the Console's two metadata listings with an exact local override.
# The normal enforcer and engine still evaluate the live entries.
_CONSOLE_LOCAL_LISTING_REGISTRY = CAPABILITY_REGISTRY
_CONSOLE_LOCAL_LISTING_METHODS = tuple(
    (
        owner, name, getattr_static(owner, "__getattribute__"), descriptor, function,
        function.__code__, function.__defaults__, function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
    )
    for owner, name in (
        (ServicePolicyEnforcer, "require_allowed"),
        (PolicyEngine, "evaluate"),
        (PolicyEngine, "_deny"),
        (RuntimeSourceState, "__init__"),
        (RuntimeSourceState, "normalized_for_policy"),
        (PolicyDecision, "__init__"),
        (PolicyDeniedError, "__init__"),
    )
    for descriptor in (vars(owner)[name],)
    for function in (
        descriptor.__func__ if type(descriptor) is staticmethod else descriptor,
    )
)
_CONSOLE_NORMALIZER_CODE = normalize_runtime_source_state.__code__


def _console_local_listing_source(enforcer, actions):
    """Observe the exact finite policy chain and entries used by these listings."""
    from . import engine, registry, source_state, types

    if type(enforcer) is not ServicePolicyEnforcer:
        return None
    evaluator = getattr_static(enforcer, "_engine", None)
    if type(evaluator) is not PolicyEngine:
        return None
    entries = getattr_static(evaluator, "_registry", None)
    if (
        type(entries) is not dict
        or entries is not _CONSOLE_LOCAL_LISTING_REGISTRY
        or registry.CAPABILITY_REGISTRY is not entries
        or engine.RuntimeSourceState is not RuntimeSourceState
        or engine.PolicyDecision is not PolicyDecision
        or types.RuntimeSourceState is not RuntimeSourceState
        or types.PolicyDeniedError is not PolicyDeniedError
        or source_state.normalize_runtime_source_state is not normalize_runtime_source_state
        or normalize_runtime_source_state.__code__ is not _CONSOLE_NORMALIZER_CODE
    ):
        return None
    for owner, name, getattribute, descriptor, function, code, defaults, keywords, items in (
        _CONSOLE_LOCAL_LISTING_METHODS
    ):
        receiver = (
            enforcer if owner is ServicePolicyEnforcer else
            evaluator if owner is PolicyEngine else owner
        )
        if (
            getattr_static(owner, "__getattribute__") is not getattribute
            or getattr_static(receiver, name, None) is not descriptor
            or vars(owner).get(name) is not descriptor
            or function.__code__ is not code
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not keywords
            or tuple((keywords or {}).items()) != items
        ):
            return None
    selected = []
    for action in actions:
        entry = entries.get(action)
        if entry is None:
            selected.append((None,))
            continue
        if (
            type(entry) is not CapabilityEntry
            or type(entry.enabled) is not bool
            or type(entry.required_source) is not str
            or type(entry.authority_owner) is not str
        ):
            return None
        selected.append((entry, entry.enabled, entry.required_source, entry.authority_owner))
    return (enforcer, evaluator, entries, tuple(selected))


def classify_backend_exception(error: Exception) -> str | None:
    # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
    from tldw_chatbook.tldw_api.exceptions import (
        APIConnectionError,
        APIResponseError,
        AuthenticationError,
    )

    if isinstance(error, AuthenticationError):
        if _contains_session_invalid_signal(
            [str(error), getattr(error, "response_data", None)]
        ):
            return "server_session_invalid"
        return "server_auth_required"

    if isinstance(error, APIConnectionError):
        return "server_unreachable"

    if isinstance(error, APIResponseError):
        if error.status_code == 401:
            if _contains_session_invalid_signal([str(error), error.response_data]):
                return "server_session_invalid"
            return "server_auth_required"

        if error.status_code == 403:
            return "authority_denied"

    return None


def _contains_session_invalid_signal(values: Iterable[object]) -> bool:
    needles = (
        "session invalid",
        "invalid session",
        "session expired",
        "token expired",
        "auth session invalid",
    )
    for value in values:
        if isinstance(value, dict):
            if _contains_session_invalid_signal(value.values()):
                return True
            continue
        if isinstance(value, (list, tuple, set)):
            if _contains_session_invalid_signal(value):
                return True
            continue
        haystack = str(value).lower()
        if any(needle in haystack for needle in needles):
            return True
    return False
