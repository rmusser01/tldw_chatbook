from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from .engine import PolicyEngine
from .registry import CAPABILITY_REGISTRY
from .types import PolicyDeniedError, RuntimeSourceState


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
            state_provider=lambda: getattr(context, "state", None),
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
        state = runtime_state_override if isinstance(runtime_state_override, RuntimeSourceState) else self.current_state()
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


def classify_backend_exception(error: Exception) -> str | None:
    # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
    from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError, AuthenticationError

    if isinstance(error, AuthenticationError):
        if _contains_session_invalid_signal([str(error), getattr(error, "response_data", None)]):
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
    needles = ("session invalid", "invalid session", "session expired", "token expired", "auth session invalid")
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
