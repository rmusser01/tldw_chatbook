from __future__ import annotations

from collections.abc import Iterable
import importlib
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
        scope_type: str | None = None,
    ) -> None:
        state = self.current_state()
        effective_state = runtime_state_override or state
        if effective_state is None:
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
            runtime_state_override=runtime_state_override,
            scope_type=scope_type,
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
    if _is_backend_exception(error, "AuthenticationError"):
        if _contains_session_invalid_signal([str(error), getattr(error, "response_data", None)]):
            return "server_session_invalid"
        return "server_auth_required"

    if _is_backend_exception(error, "APIConnectionError"):
        return "server_unreachable"

    if _is_backend_exception(error, "APIResponseError"):
        status_code = getattr(error, "status_code", None)
        response_data = getattr(error, "response_data", None)
        if status_code == 401:
            if _contains_session_invalid_signal([str(error), response_data]):
                return "server_session_invalid"
            return "server_auth_required"

        if status_code == 403:
            return "authority_denied"

    return None


def _load_backend_exception_classes(*, module_importer=importlib.import_module) -> dict[str, type[Exception]]:
    try:
        module = module_importer("tldw_chatbook.tldw_api.exceptions")
    except (AttributeError, ImportError, ModuleNotFoundError, TypeError):
        return {}

    resolved: dict[str, type[Exception]] = {}
    for class_name in ("AuthenticationError", "APIConnectionError", "APIResponseError"):
        candidate = getattr(module, class_name, None)
        if isinstance(candidate, type) and issubclass(candidate, Exception):
            resolved[class_name] = candidate
    return resolved


def _is_backend_exception(error: Exception, class_name: str) -> bool:
    backend_classes = _load_backend_exception_classes()
    backend_class = backend_classes.get(class_name)
    if backend_class is not None and isinstance(error, backend_class):
        return True

    error_type = type(error)
    return (
        error_type.__name__ == class_name
        and error_type.__module__ == "tldw_chatbook.tldw_api.exceptions"
    )


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
