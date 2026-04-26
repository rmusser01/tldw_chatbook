"""
Thin service helpers for server-backed character and persona catalog access.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    PersonaProfileCreate,
    PersonaProfileUpdate,
    PresetCreate,
    PresetUpdate,
    TLDWAPIClient,
)


class ServerCharacterPersonaService:
    """Thin wrapper around the shared TLDW API client for character/persona reads."""

    def __init__(self, client: Optional[TLDWAPIClient], *, policy_enforcer: Any | None = None):
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerCharacterPersonaService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server character and persona operations.")
        return self.client

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None)
                    or "Server character/persona action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _persona_action_id(action: str) -> str:
        return f"character.persona.{action}.server"

    @staticmethod
    def _archetype_action_id(action: str) -> str:
        return f"character.archetypes.{action}.server"

    @staticmethod
    def _session_action_id(action: str = "launch") -> str:
        return f"character.sessions.{action}.server"

    @staticmethod
    def _message_action_id(action: str) -> str:
        return f"character.messages.{action}.server"

    async def list_characters(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        self._enforce(self._persona_action_id("list"))
        client = self._require_client()
        return await client.list_characters(limit=limit, offset=offset)

    async def search_characters(self, query: str, limit: int = 10) -> Any:
        self._enforce(self._persona_action_id("list"))
        client = self._require_client()
        return await client.search_characters(query, limit=limit)

    async def get_character(self, character_id: int) -> Any:
        self._enforce(self._persona_action_id("detail"))
        client = self._require_client()
        return await client.get_character(character_id)

    async def create_character(self, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("create"))
        client = self._require_client()
        return await client.create_character(request_data)

    async def update_character(self, character_id: int, request_data: Any, expected_version: int) -> Any:
        self._enforce(self._persona_action_id("update"))
        client = self._require_client()
        return await client.update_character(character_id, request_data, expected_version)

    async def delete_character(self, character_id: int, expected_version: int) -> Any:
        self._enforce(self._persona_action_id("delete"))
        client = self._require_client()
        return await client.delete_character(character_id, expected_version)

    async def restore_character(self, character_id: int, expected_version: int) -> Any:
        self._enforce(self._persona_action_id("update"))
        client = self._require_client()
        return await client.restore_character(character_id, expected_version)

    async def list_persona_profiles(
        self,
        active_only: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        self._enforce(self._persona_action_id("list"))
        client = self._require_client()
        return await client.list_persona_profiles(
            active_only=active_only,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )

    async def get_persona_profile(self, persona_id: str) -> Any:
        self._enforce(self._persona_action_id("detail"))
        client = self._require_client()
        return await client.get_persona_profile(persona_id)

    async def create_persona_profile(self, request_data: PersonaProfileCreate) -> Any:
        self._enforce(self._persona_action_id("create"))
        client = self._require_client()
        return await client.create_persona_profile(request_data)

    async def update_persona_profile(
        self,
        persona_id: str,
        request_data: PersonaProfileUpdate,
        expected_version: Optional[int] = None,
    ) -> Any:
        self._enforce(self._persona_action_id("update"))
        client = self._require_client()
        return await client.update_persona_profile(
            persona_id,
            request_data,
            expected_version=expected_version,
        )

    async def delete_persona_profile(self, persona_id: str, expected_version: Optional[int] = None) -> Any:
        self._enforce(self._persona_action_id("delete"))
        client = self._require_client()
        return await client.delete_persona_profile(persona_id, expected_version=expected_version)

    async def restore_persona_profile(self, persona_id: str, expected_version: int) -> Any:
        self._enforce(self._persona_action_id("update"))
        client = self._require_client()
        return await client.restore_persona_profile(persona_id, expected_version=expected_version)

    async def list_persona_archetypes(self) -> Any:
        self._enforce(self._archetype_action_id("list"))
        client = self._require_client()
        return await client.list_persona_archetypes()

    async def get_persona_archetype(self, key: str) -> Any:
        self._enforce(self._archetype_action_id("detail"))
        client = self._require_client()
        return await client.get_persona_archetype(key)

    async def preview_persona_archetype(self, key: str) -> Any:
        self._enforce(self._archetype_action_id("preview"))
        client = self._require_client()
        return await client.preview_persona_archetype(key)

    async def list_persona_exemplars(self, persona_id: str, **kwargs: Any) -> Any:
        self._enforce(self._persona_action_id("list"))
        client = self._require_client()
        return await client.list_persona_exemplars(persona_id, **kwargs)

    async def get_persona_exemplar(self, persona_id: str, exemplar_id: str) -> Any:
        self._enforce(self._persona_action_id("detail"))
        client = self._require_client()
        return await client.get_persona_exemplar(persona_id, exemplar_id)

    async def create_persona_exemplar(self, persona_id: str, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("create"))
        client = self._require_client()
        return await client.create_persona_exemplar(persona_id, request_data)

    async def import_persona_exemplars(self, persona_id: str, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("create"))
        client = self._require_client()
        return await client.import_persona_exemplars(persona_id, request_data)

    async def update_persona_exemplar(self, persona_id: str, exemplar_id: str, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("update"))
        client = self._require_client()
        return await client.update_persona_exemplar(persona_id, exemplar_id, request_data)

    async def review_persona_exemplar(self, persona_id: str, exemplar_id: str, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("update"))
        client = self._require_client()
        return await client.review_persona_exemplar(persona_id, exemplar_id, request_data)

    async def delete_persona_exemplar(self, persona_id: str, exemplar_id: str) -> Any:
        self._enforce(self._persona_action_id("delete"))
        client = self._require_client()
        return await client.delete_persona_exemplar(persona_id, exemplar_id)

    async def get_character_exemplar(self, character_id: int, exemplar_id: str) -> Any:
        self._enforce(self._persona_action_id("detail"))
        client = self._require_client()
        return await client.get_character_exemplar(character_id, exemplar_id)

    async def create_character_exemplar(self, character_id: int, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("create"))
        client = self._require_client()
        return await client.create_character_exemplar(character_id, request_data)

    async def update_character_exemplar(self, character_id: int, exemplar_id: str, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("update"))
        client = self._require_client()
        return await client.update_character_exemplar(character_id, exemplar_id, request_data)

    async def delete_character_exemplar(self, character_id: int, exemplar_id: str) -> Any:
        self._enforce(self._persona_action_id("delete"))
        client = self._require_client()
        return await client.delete_character_exemplar(character_id, exemplar_id)

    async def search_character_exemplars(self, character_id: int, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("list"))
        client = self._require_client()
        return await client.search_character_exemplars(character_id, request_data)

    async def select_character_exemplars_debug(self, character_id: int, request_data: Any) -> Any:
        self._enforce(self._persona_action_id("list"))
        client = self._require_client()
        return await client.select_character_exemplars_debug(character_id, request_data)

    async def list_chat_greetings(self, chat_id: str) -> Any:
        self._enforce(self._session_action_id())
        client = self._require_client()
        return await client.list_greetings(chat_id)

    async def select_chat_greeting(self, chat_id: str, index: int) -> Any:
        self._enforce(self._session_action_id())
        client = self._require_client()
        return await client.select_greeting(chat_id, index)

    async def list_chat_presets(self) -> Any:
        self._enforce(self._persona_action_id("list"))
        client = self._require_client()
        return await client.list_presets()

    async def create_chat_preset(self, request_data: PresetCreate) -> Any:
        self._enforce(self._persona_action_id("create"))
        client = self._require_client()
        return await client.create_preset(request_data)

    async def update_chat_preset(self, preset_id: str, request_data: PresetUpdate) -> Any:
        self._enforce(self._persona_action_id("update"))
        client = self._require_client()
        return await client.update_preset(preset_id, request_data)

    async def delete_chat_preset(self, preset_id: str) -> Any:
        self._enforce(self._persona_action_id("delete"))
        client = self._require_client()
        return await client.delete_preset(preset_id)

    async def create_character_chat_session(self, request_data: Any, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("create"))
        client = self._require_client()
        return await client.create_character_chat_session(request_data, **kwargs)

    async def list_character_chat_sessions(self, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("list"))
        client = self._require_client()
        return await client.list_character_chat_sessions(**kwargs)

    async def get_character_chat_session(self, chat_id: str, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("detail"))
        client = self._require_client()
        return await client.get_character_chat_session(chat_id, **kwargs)

    async def update_character_chat_session(self, chat_id: str, request_data: Any, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("update"))
        client = self._require_client()
        return await client.update_character_chat_session(chat_id, request_data, **kwargs)

    async def delete_character_chat_session(self, chat_id: str, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("delete"))
        client = self._require_client()
        return await client.delete_character_chat_session(chat_id, **kwargs)

    async def restore_character_chat_session(self, chat_id: str, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("restore"))
        client = self._require_client()
        return await client.restore_character_chat_session(chat_id, **kwargs)

    async def list_character_messages(self, chat_id: str, **kwargs: Any) -> Any:
        self._enforce(self._message_action_id("list"))
        client = self._require_client()
        return await client.list_character_messages(chat_id, **kwargs)

    async def get_character_message(self, message_id: str, **kwargs: Any) -> Any:
        self._enforce(self._message_action_id("detail"))
        client = self._require_client()
        return await client.get_character_message(message_id, **kwargs)

    async def create_character_message(self, chat_id: str, request_data: Any, **kwargs: Any) -> Any:
        self._enforce(self._message_action_id("create"))
        client = self._require_client()
        return await client.create_character_message(chat_id, request_data, **kwargs)

    async def update_character_message(self, message_id: str, request_data: Any, **kwargs: Any) -> Any:
        self._enforce(self._message_action_id("update"))
        client = self._require_client()
        return await client.update_character_message(message_id, request_data, **kwargs)

    async def delete_character_message(self, message_id: str, **kwargs: Any) -> Any:
        self._enforce(self._message_action_id("delete"))
        client = self._require_client()
        return await client.delete_character_message(message_id, **kwargs)

    async def search_character_messages(self, chat_id: str, query: str, **kwargs: Any) -> Any:
        self._enforce(self._message_action_id("list"))
        client = self._require_client()
        return await client.search_character_messages(chat_id, query, **kwargs)

    async def get_chat_settings(self, chat_id: str, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("detail"))
        client = self._require_client()
        return await client.get_chat_settings(chat_id, **kwargs)

    async def update_chat_settings(self, chat_id: str, request_data: Any, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("update"))
        client = self._require_client()
        return await client.update_chat_settings(chat_id, request_data, **kwargs)

    async def export_chat_history(self, chat_id: str, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("export"))
        client = self._require_client()
        return await client.export_chat_history(chat_id, **kwargs)

    async def get_author_note_info(self, chat_id: str) -> Any:
        self._enforce(self._session_action_id("detail"))
        client = self._require_client()
        return await client.get_author_note_info(chat_id)

    async def export_lorebook_diagnostics(self, chat_id: str, **kwargs: Any) -> Any:
        self._enforce(self._session_action_id("observe"))
        client = self._require_client()
        return await client.export_lorebook_diagnostics(chat_id, **kwargs)
