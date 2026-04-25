"""
Mode-aware routing for local and server-backed character/persona catalog access.
"""

from __future__ import annotations

import inspect
from typing import Any


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "character.persona.profiles.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_scope_missing",
        "user_message": (
            "Local persona profile CRUD is still handled by older CCP/local chat paths and is not wrapped by "
            "the source-aware character/persona scope yet."
        ),
        "affected_action_ids": [
            "character.persona.create.local",
            "character.persona.delete.local",
            "character.persona.detail.local",
            "character.persona.list.local",
            "character.persona.update.local",
        ],
    },
    {
        "operation_id": "character.sessions.execution.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_scope_missing",
        "user_message": (
            "Local character greetings, presets, settings, and lorebook diagnostics still use legacy local CCP "
            "flows instead of this source-aware scope."
        ),
        "affected_action_ids": [
            "character.sessions.detail.local",
            "character.sessions.launch.local",
            "character.sessions.observe.local",
            "character.sessions.update.local",
        ],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "character.messages.mutation.server",
        "source": "server",
        "supported": False,
        "reason_code": "chatbook_contract_missing",
        "user_message": (
            "Server character-message endpoints exist, but Chatbook does not yet wrap message mutation through "
            "the source-aware character/persona scope."
        ),
        "affected_action_ids": ["character.sessions.detail.server", "character.sessions.update.server"],
    }
]


class CharacterPersonaScopeService:
    """Route character and persona catalog reads to the selected backend."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: str | None) -> str:
        normalized_mode = "local" if mode is None else str(mode).strip().lower()
        if normalized_mode not in {"local", "server"}:
            raise ValueError(f"Invalid character/persona mode: {mode!r}. Expected 'local' or 'server'.")
        return normalized_mode

    def _backend(self, mode: str | None):
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == "server":
            if self.server_service is None:
                raise ValueError("Server character/persona backend is unavailable.")
            return self.server_service

        if self.local_service is None:
            raise ValueError("Local character/persona backend is unavailable.")
        return self.local_service

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _persona_action_id(mode: str, action: str) -> str:
        return f"character.persona.{action}.{mode}"

    @staticmethod
    def _session_action_id(mode: str, action: str = "launch") -> str:
        return f"character.sessions.{action}.{mode}"

    def list_unsupported_capabilities(self, *, mode: str | None = None) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == "local":
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _invoke_backend_method(
        self,
        backend: Any,
        method_names: tuple[str, ...],
        *args: Any,
        missing_message: str,
        **kwargs: Any,
    ) -> Any:
        for method_name in method_names:
            method = getattr(backend, method_name, None)
            if callable(method):
                return await self._maybe_await(method(*args, **kwargs))
        raise ValueError(missing_message)

    async def list_characters(self, mode: str = "local", limit: int = 100, offset: int = 0) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "list"))
        backend = self._backend(normalized_mode)
        if normalized_mode == "local" and not hasattr(backend, "list_characters"):
            if not hasattr(backend, "list_character_cards"):
                raise ValueError(
                    "Local character backend does not provide list_characters() or list_character_cards()."
                )
            return await self._maybe_await(backend.list_character_cards(limit=limit, offset=offset))
        if not hasattr(backend, "list_characters"):
            raise ValueError("Character backend does not provide list_characters().")
        return await self._maybe_await(backend.list_characters(limit=limit, offset=offset))

    async def list_persona_profiles(
        self,
        mode: str = "local",
        active_only: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "list"))
        backend = self._backend(normalized_mode)
        if normalized_mode == "local" and not hasattr(backend, "list_persona_profiles"):
            raise ValueError("Local persona profiles are not available yet.")
        if not hasattr(backend, "list_persona_profiles"):
            raise ValueError("Character/persona backend does not provide list_persona_profiles().")
        return await self._maybe_await(
            backend.list_persona_profiles(
                active_only=active_only,
                include_deleted=include_deleted,
                limit=limit,
                offset=offset,
            )
        )

    async def get_persona_profile(self, persona_id: str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "detail"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local persona profiles are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide get_persona_profile()."
        )
        return await self._invoke_backend_method(
            backend,
            ("get_persona_profile", "fetch_persona_by_id"),
            persona_id,
            missing_message=missing_message,
        )

    async def create_persona_profile(self, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "create"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local persona profiles are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide create_persona_profile()."
        )
        return await self._invoke_backend_method(
            backend,
            ("create_persona_profile", "create_persona"),
            request_data,
            missing_message=missing_message,
        )

    async def update_persona_profile(
        self,
        persona_id: str,
        request_data: Any,
        expected_version: int | None = None,
        mode: str = "local",
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "update"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local persona profiles are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide update_persona_profile()."
        )
        kwargs = {}
        if expected_version is not None:
            kwargs["expected_version"] = expected_version
        return await self._invoke_backend_method(
            backend,
            ("update_persona_profile", "update_persona"),
            persona_id,
            request_data,
            missing_message=missing_message,
            **kwargs,
        )

    async def list_chat_greetings(self, chat_id: str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local chat greetings are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide list_chat_greetings()."
        )
        return await self._invoke_backend_method(
            backend,
            ("list_chat_greetings", "list_greetings"),
            chat_id,
            missing_message=missing_message,
        )

    async def select_chat_greeting(self, chat_id: str, index: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local chat greetings are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide select_chat_greeting()."
        )
        return await self._invoke_backend_method(
            backend,
            ("select_chat_greeting", "select_greeting"),
            chat_id,
            index,
            missing_message=missing_message,
        )

    async def list_chat_presets(self, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "list"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local chat presets are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide list_chat_presets()."
        )
        return await self._invoke_backend_method(
            backend,
            ("list_chat_presets", "list_presets"),
            missing_message=missing_message,
        )

    async def create_chat_preset(self, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "create"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local chat presets are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide create_chat_preset()."
        )
        return await self._invoke_backend_method(
            backend,
            ("create_chat_preset", "create_preset"),
            request_data,
            missing_message=missing_message,
        )

    async def update_chat_preset(self, preset_id: str, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "update"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local chat presets are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide update_chat_preset()."
        )
        return await self._invoke_backend_method(
            backend,
            ("update_chat_preset", "update_preset"),
            preset_id,
            request_data,
            missing_message=missing_message,
        )

    async def delete_chat_preset(self, preset_id: str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._persona_action_id(normalized_mode, "delete"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local chat presets are not available yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide delete_chat_preset()."
        )
        return await self._invoke_backend_method(
            backend,
            ("delete_chat_preset", "delete_preset"),
            preset_id,
            missing_message=missing_message,
        )

    async def create_character_chat_session(self, request_data: Any, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "create"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat session creation is not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide create_character_chat_session()."
        )
        return await self._invoke_backend_method(
            backend,
            ("create_character_chat_session", "create_chat_session"),
            request_data,
            missing_message=missing_message,
            **kwargs,
        )

    async def list_character_chat_sessions(self, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "list"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat session listing is not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide list_character_chat_sessions()."
        )
        return await self._invoke_backend_method(
            backend,
            ("list_character_chat_sessions", "list_chat_sessions"),
            missing_message=missing_message,
            **kwargs,
        )

    async def get_character_chat_session(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "detail"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat session detail is not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide get_character_chat_session()."
        )
        return await self._invoke_backend_method(
            backend,
            ("get_character_chat_session", "get_chat_session"),
            chat_id,
            missing_message=missing_message,
            **kwargs,
        )

    async def update_character_chat_session(
        self,
        chat_id: str,
        request_data: Any,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "update"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat session updates are not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide update_character_chat_session()."
        )
        return await self._invoke_backend_method(
            backend,
            ("update_character_chat_session", "update_chat_session"),
            chat_id,
            request_data,
            missing_message=missing_message,
            **kwargs,
        )

    async def delete_character_chat_session(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "delete"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat session deletion is not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide delete_character_chat_session()."
        )
        return await self._invoke_backend_method(
            backend,
            ("delete_character_chat_session", "delete_chat_session"),
            chat_id,
            missing_message=missing_message,
            **kwargs,
        )

    async def restore_character_chat_session(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "restore"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat session restore is not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide restore_character_chat_session()."
        )
        return await self._invoke_backend_method(
            backend,
            ("restore_character_chat_session", "restore_chat_session"),
            chat_id,
            missing_message=missing_message,
            **kwargs,
        )

    async def get_chat_settings(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "detail"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat settings are not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide get_chat_settings()."
        )
        return await self._invoke_backend_method(
            backend,
            ("get_chat_settings",),
            chat_id,
            missing_message=missing_message,
            **kwargs,
        )

    async def update_chat_settings(self, chat_id: str, request_data: Any, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "update"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat settings updates are not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide update_chat_settings()."
        )
        return await self._invoke_backend_method(
            backend,
            ("update_chat_settings",),
            chat_id,
            request_data,
            missing_message=missing_message,
            **kwargs,
        )

    async def export_chat_history(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "export"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character chat export is not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide export_chat_history()."
        )
        return await self._invoke_backend_method(
            backend,
            ("export_chat_history", "export_chat_session"),
            chat_id,
            missing_message=missing_message,
            **kwargs,
        )

    async def get_author_note_info(self, chat_id: str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "detail"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character author-note info is not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide get_author_note_info()."
        )
        return await self._invoke_backend_method(
            backend,
            ("get_author_note_info",),
            chat_id,
            missing_message=missing_message,
        )

    async def export_lorebook_diagnostics(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._session_action_id(normalized_mode, "observe"))
        backend = self._backend(normalized_mode)
        missing_message = (
            "Local character lorebook diagnostics are not available through this scope service yet."
            if normalized_mode == "local"
            else "Character/persona backend does not provide export_lorebook_diagnostics()."
        )
        return await self._invoke_backend_method(
            backend,
            ("export_lorebook_diagnostics",),
            chat_id,
            missing_message=missing_message,
            **kwargs,
        )
