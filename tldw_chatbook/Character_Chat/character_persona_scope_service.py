"""
Mode-aware routing for local and server-backed character/persona catalog access.
"""

from __future__ import annotations

import inspect
from typing import Any


class CharacterPersonaScopeService:
    """Route character and persona catalog reads to the selected backend."""

    _ACTION_IDS = {
        "session_create": "character.sessions.create",
        "session_list": "character.sessions.list",
        "session_detail": "character.sessions.detail",
        "session_update": "character.sessions.update",
        "session_delete": "character.sessions.delete",
        "message_create": "character.messages.create",
        "message_list": "character.messages.list",
        "message_detail": "character.messages.detail",
        "message_update": "character.messages.update",
        "message_delete": "character.messages.delete",
        "memory_create": "character.memory.create",
        "memory_list": "character.memory.list",
        "memory_update": "character.memory.update",
        "memory_delete": "character.memory.delete",
        "memory_launch": "character.memory.launch",
    }

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: str | None) -> str:
        normalized_mode = "local" if mode is None else str(mode)
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

    def _enforce_policy(self, mode: str, action: str) -> None:
        if self.policy_enforcer is None:
            return
        action_prefix = self._ACTION_IDS.get(action)
        if action_prefix is None:
            return
        self.policy_enforcer.require_allowed(action_id=f"{action_prefix}.{mode}")

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

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
        backend = self._backend(mode)
        if mode in {None, "local"} and not hasattr(backend, "list_characters"):
            if not hasattr(backend, "list_character_cards"):
                raise ValueError("Local character backend does not provide list_characters() or list_character_cards().")
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
        backend = self._backend(mode)
        if mode in {None, "local"} and not hasattr(backend, "list_persona_profiles"):
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
        backend = self._backend(mode)
        missing_message = (
            "Local persona profiles are not available yet."
            if mode in {None, "local"}
            else "Character/persona backend does not provide get_persona_profile()."
        )
        return await self._invoke_backend_method(
            backend,
            ("get_persona_profile", "fetch_persona_by_id"),
            persona_id,
            missing_message=missing_message,
        )

    async def create_persona_profile(self, request_data: Any, mode: str = "local") -> Any:
        backend = self._backend(mode)
        missing_message = (
            "Local persona profiles are not available yet."
            if mode in {None, "local"}
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
        backend = self._backend(mode)
        missing_message = (
            "Local persona profiles are not available yet."
            if mode in {None, "local"}
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
        backend = self._backend(mode)
        missing_message = (
            "Local chat greetings are not available yet."
            if mode in {None, "local"}
            else "Character/persona backend does not provide list_chat_greetings()."
        )
        return await self._invoke_backend_method(
            backend,
            ("list_chat_greetings", "list_greetings"),
            chat_id,
            missing_message=missing_message,
        )

    async def select_chat_greeting(self, chat_id: str, index: int, mode: str = "local") -> Any:
        backend = self._backend(mode)
        missing_message = (
            "Local chat greetings are not available yet."
            if mode in {None, "local"}
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
        backend = self._backend(mode)
        missing_message = (
            "Local chat presets are not available yet."
            if mode in {None, "local"}
            else "Character/persona backend does not provide list_chat_presets()."
        )
        return await self._invoke_backend_method(
            backend,
            ("list_chat_presets", "list_presets"),
            missing_message=missing_message,
        )

    async def create_chat_preset(self, request_data: Any, mode: str = "local") -> Any:
        backend = self._backend(mode)
        missing_message = (
            "Local chat presets are not available yet."
            if mode in {None, "local"}
            else "Character/persona backend does not provide create_chat_preset()."
        )
        return await self._invoke_backend_method(
            backend,
            ("create_chat_preset", "create_preset"),
            request_data,
            missing_message=missing_message,
        )

    async def update_chat_preset(self, preset_id: str, request_data: Any, mode: str = "local") -> Any:
        backend = self._backend(mode)
        missing_message = (
            "Local chat presets are not available yet."
            if mode in {None, "local"}
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
        backend = self._backend(mode)
        missing_message = (
            "Local chat presets are not available yet."
            if mode in {None, "local"}
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
        self._enforce_policy(normalized_mode, "session_create")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("create_character_chat_session",),
            request_data,
            missing_message="Character/persona backend does not provide create_character_chat_session().",
            **kwargs,
        )

    async def list_character_chat_sessions(self, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "session_list")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("list_character_chat_sessions",),
            missing_message="Character/persona backend does not provide list_character_chat_sessions().",
            **kwargs,
        )

    async def get_character_chat_session(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "session_detail")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("get_character_chat_session",),
            chat_id,
            missing_message="Character/persona backend does not provide get_character_chat_session().",
            **kwargs,
        )

    async def update_character_chat_session(
        self,
        chat_id: str,
        request_data: Any,
        *,
        expected_version: int,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "session_update")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("update_character_chat_session",),
            chat_id,
            request_data,
            expected_version=expected_version,
            missing_message="Character/persona backend does not provide update_character_chat_session().",
            **kwargs,
        )

    async def delete_character_chat_session(
        self,
        chat_id: str,
        *,
        expected_version: int | None = None,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "session_delete")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("delete_character_chat_session",),
            chat_id,
            expected_version=expected_version,
            missing_message="Character/persona backend does not provide delete_character_chat_session().",
            **kwargs,
        )

    async def restore_character_chat_session(
        self,
        chat_id: str,
        *,
        expected_version: int | None = None,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "session_update")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("restore_character_chat_session",),
            chat_id,
            expected_version=expected_version,
            missing_message="Character/persona backend does not provide restore_character_chat_session().",
            **kwargs,
        )

    async def get_character_chat_settings(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "session_detail")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("get_character_chat_settings",),
            chat_id,
            missing_message="Character/persona backend does not provide get_character_chat_settings().",
            **kwargs,
        )

    async def update_character_chat_settings(
        self,
        chat_id: str,
        request_data: Any,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "session_update")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("update_character_chat_settings",),
            chat_id,
            request_data,
            missing_message="Character/persona backend does not provide update_character_chat_settings().",
            **kwargs,
        )

    async def create_character_chat_message(
        self,
        chat_id: str,
        request_data: Any,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "message_create")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("create_character_chat_message",),
            chat_id,
            request_data,
            missing_message="Character/persona backend does not provide create_character_chat_message().",
            **kwargs,
        )

    async def list_character_chat_messages(self, chat_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "message_list")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("list_character_chat_messages",),
            chat_id,
            missing_message="Character/persona backend does not provide list_character_chat_messages().",
            **kwargs,
        )

    async def get_character_chat_message(self, message_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "message_detail")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("get_character_chat_message",),
            message_id,
            missing_message="Character/persona backend does not provide get_character_chat_message().",
            **kwargs,
        )

    async def update_character_chat_message(
        self,
        message_id: str,
        request_data: Any,
        *,
        expected_version: int,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "message_update")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("update_character_chat_message",),
            message_id,
            request_data,
            expected_version=expected_version,
            missing_message="Character/persona backend does not provide update_character_chat_message().",
            **kwargs,
        )

    async def delete_character_chat_message(
        self,
        message_id: str,
        *,
        expected_version: int,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "message_delete")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("delete_character_chat_message",),
            message_id,
            expected_version=expected_version,
            missing_message="Character/persona backend does not provide delete_character_chat_message().",
            **kwargs,
        )

    async def search_character_chat_messages(
        self,
        chat_id: str,
        query: str,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "message_list")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("search_character_chat_messages",),
            chat_id,
            query,
            missing_message="Character/persona backend does not provide search_character_chat_messages().",
            **kwargs,
        )

    async def list_character_memories(self, character_id: str, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "memory_list")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("list_character_memories",),
            character_id,
            missing_message="Character/persona backend does not provide list_character_memories().",
            **kwargs,
        )

    async def create_character_memory(self, character_id: str, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "memory_create")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("create_character_memory",),
            character_id,
            request_data,
            missing_message="Character/persona backend does not provide create_character_memory().",
        )

    async def update_character_memory(
        self,
        character_id: str,
        memory_id: str,
        request_data: Any,
        mode: str = "local",
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "memory_update")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("update_character_memory",),
            character_id,
            memory_id,
            request_data,
            missing_message="Character/persona backend does not provide update_character_memory().",
        )

    async def delete_character_memory(self, character_id: str, memory_id: str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "memory_delete")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("delete_character_memory",),
            character_id,
            memory_id,
            missing_message="Character/persona backend does not provide delete_character_memory().",
        )

    async def archive_character_memory(
        self,
        character_id: str,
        memory_id: str,
        request_data: Any,
        mode: str = "local",
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "memory_update")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("archive_character_memory",),
            character_id,
            memory_id,
            request_data,
            missing_message="Character/persona backend does not provide archive_character_memory().",
        )

    async def extract_character_memories(self, character_id: str, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "memory_launch")
        return await self._invoke_backend_method(
            self._backend(normalized_mode),
            ("extract_character_memories",),
            character_id,
            request_data,
            missing_message="Character/persona backend does not provide extract_character_memories().",
        )
