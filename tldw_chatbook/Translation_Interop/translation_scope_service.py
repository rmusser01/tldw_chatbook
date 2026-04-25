"""Source-aware routing for server-owned translation actions."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .server_translation_service import ServerTranslationService


class TranslationBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "translation.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server text translation is unavailable in local/offline mode.",
        "affected_action_ids": ["translation.text.launch.server"],
    }
]


class TranslationScopeService:
    """Route text translation through the active server; local translation is intentionally absent."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: TranslationBackend | str | None) -> TranslationBackend:
        if mode is None:
            return TranslationBackend.SERVER
        if isinstance(mode, TranslationBackend):
            return mode
        try:
            return TranslationBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid translation backend: {mode}") from exc

    def _require_server_service(self, mode: TranslationBackend) -> Any:
        if mode == TranslationBackend.LOCAL:
            raise ValueError("Server translation is server-only; switch to server mode to use it.")
        if self.server_service is None:
            raise ValueError("Server translation backend is unavailable.")
        return self.server_service

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @classmethod
    def _rewrite_backend(cls, payload: Any, backend: str) -> Any:
        if isinstance(payload, list):
            return [cls._rewrite_backend(item, backend) for item in payload]
        if isinstance(payload, dict):
            record = {key: cls._rewrite_backend(value, backend) for key, value in payload.items()}
            if record.get("backend") == "server":
                record["backend"] = backend
            return record
        return payload

    def list_unsupported_capabilities(
        self,
        *,
        mode: TranslationBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == TranslationBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def translate_text(
        self,
        request_data: Any,
        *,
        mode: TranslationBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("translation.text.launch.server")
        result = await self._maybe_await(service.translate_text(request_data))
        normalized = ServerTranslationService._normalize_response(result)
        return self._rewrite_backend(normalized, normalized_mode.value)
