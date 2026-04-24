"""
Thin service helpers for server-backed character and persona catalog access.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    CharacterCreateRequest,
    CharacterChatMessageCreate,
    CharacterChatMessageUpdate,
    CharacterChatSessionCreate,
    CharacterChatSessionUpdate,
    CharacterChatSettingsUpdate,
    CharacterMemoryArchiveRequest,
    CharacterMemoryCreate,
    CharacterMemoryExtractRequest,
    CharacterMemoryUpdate,
    CharacterUpdateRequest,
    PersonaProfileCreate,
    PersonaProfileUpdate,
    PresetCreate,
    PresetUpdate,
    TLDWAPIClient,
)


class ServerCharacterPersonaService:
    """Thin wrapper around the shared TLDW API client for character/persona reads."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerCharacterPersonaService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server character and persona operations.")
        return self.client

    @staticmethod
    def _request_model(request_data: Any, model_cls: Any, **dump_kwargs: Any) -> Any:
        if isinstance(request_data, model_cls):
            return request_data
        if hasattr(request_data, "model_dump"):
            return model_cls.model_validate(request_data.model_dump(**dump_kwargs))
        return model_cls.model_validate(dict(request_data))

    async def list_characters(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        client = self._require_client()
        return await client.list_characters(limit=limit, offset=offset)

    async def get_character(self, character_id: int) -> Any:
        client = self._require_client()
        return await client.get_character(character_id)

    async def create_character(self, request_data: CharacterCreateRequest | Mapping[str, Any]) -> Any:
        client = self._require_client()
        request = self._request_model(request_data, CharacterCreateRequest, exclude_none=True, mode="json")
        return await client.create_character(request)

    async def update_character(
        self,
        character_id: int,
        request_data: CharacterUpdateRequest | Mapping[str, Any],
        expected_version: int,
    ) -> Any:
        client = self._require_client()
        request = self._request_model(
            request_data,
            CharacterUpdateRequest,
            exclude_unset=True,
            exclude_none=True,
            mode="json",
        )
        return await client.update_character(character_id, request, expected_version)

    async def delete_character(self, character_id: int, expected_version: int) -> Any:
        client = self._require_client()
        return await client.delete_character(character_id, expected_version)

    async def restore_character(self, character_id: int, expected_version: int) -> Any:
        client = self._require_client()
        return await client.restore_character(character_id, expected_version)

    async def list_persona_profiles(
        self,
        active_only: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        client = self._require_client()
        return await client.list_persona_profiles(
            active_only=active_only,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )

    async def get_persona_profile(self, persona_id: str) -> Any:
        client = self._require_client()
        return await client.get_persona_profile(persona_id)

    async def create_persona_profile(self, request_data: PersonaProfileCreate) -> Any:
        client = self._require_client()
        return await client.create_persona_profile(request_data)

    async def update_persona_profile(
        self,
        persona_id: str,
        request_data: PersonaProfileUpdate,
        expected_version: Optional[int] = None,
    ) -> Any:
        client = self._require_client()
        return await client.update_persona_profile(
            persona_id,
            request_data,
            expected_version=expected_version,
        )

    async def list_chat_greetings(self, chat_id: str) -> Any:
        client = self._require_client()
        return await client.list_greetings(chat_id)

    async def select_chat_greeting(self, chat_id: str, index: int) -> Any:
        client = self._require_client()
        return await client.select_greeting(chat_id, index)

    async def list_chat_presets(self) -> Any:
        client = self._require_client()
        return await client.list_presets()

    async def create_chat_preset(self, request_data: PresetCreate) -> Any:
        client = self._require_client()
        return await client.create_preset(request_data)

    async def update_chat_preset(self, preset_id: str, request_data: PresetUpdate) -> Any:
        client = self._require_client()
        return await client.update_preset(preset_id, request_data)

    async def delete_chat_preset(self, preset_id: str) -> Any:
        client = self._require_client()
        return await client.delete_preset(preset_id)

    async def create_character_chat_session(
        self,
        request_data: CharacterChatSessionCreate | Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterChatSessionCreate)
            else CharacterChatSessionCreate.model_validate(dict(request_data))
        )
        return await client.create_character_chat_session(request, **kwargs)

    async def list_character_chat_sessions(self, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.list_character_chat_sessions(**kwargs)

    async def get_character_chat_session(self, chat_id: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.get_character_chat_session(chat_id, **kwargs)

    async def update_character_chat_session(
        self,
        chat_id: str,
        request_data: CharacterChatSessionUpdate | Mapping[str, Any],
        *,
        expected_version: int,
        **kwargs: Any,
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterChatSessionUpdate)
            else CharacterChatSessionUpdate.model_validate(dict(request_data))
        )
        return await client.update_character_chat_session(
            chat_id,
            request,
            expected_version=expected_version,
            **kwargs,
        )

    async def delete_character_chat_session(self, chat_id: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.delete_character_chat_session(chat_id, **kwargs)

    async def restore_character_chat_session(self, chat_id: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.restore_character_chat_session(chat_id, **kwargs)

    async def get_character_chat_settings(self, chat_id: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.get_character_chat_settings(chat_id, **kwargs)

    async def update_character_chat_settings(
        self,
        chat_id: str,
        request_data: CharacterChatSettingsUpdate | Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterChatSettingsUpdate)
            else CharacterChatSettingsUpdate.model_validate(dict(request_data))
        )
        return await client.update_character_chat_settings(chat_id, request, **kwargs)

    async def create_character_chat_message(
        self,
        chat_id: str,
        request_data: CharacterChatMessageCreate | Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterChatMessageCreate)
            else CharacterChatMessageCreate.model_validate(dict(request_data))
        )
        return await client.create_character_chat_message(chat_id, request, **kwargs)

    async def list_character_chat_messages(self, chat_id: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.list_character_chat_messages(chat_id, **kwargs)

    async def get_character_chat_message(self, message_id: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.get_character_chat_message(message_id, **kwargs)

    async def update_character_chat_message(
        self,
        message_id: str,
        request_data: CharacterChatMessageUpdate | Mapping[str, Any],
        *,
        expected_version: int,
        **kwargs: Any,
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterChatMessageUpdate)
            else CharacterChatMessageUpdate.model_validate(dict(request_data))
        )
        return await client.update_character_chat_message(
            message_id,
            request,
            expected_version=expected_version,
            **kwargs,
        )

    async def delete_character_chat_message(self, message_id: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.delete_character_chat_message(message_id, **kwargs)

    async def search_character_chat_messages(self, chat_id: str, query: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.search_character_chat_messages(chat_id, query, **kwargs)

    async def list_character_memories(self, character_id: str, **kwargs: Any) -> Any:
        client = self._require_client()
        return await client.list_character_memories(character_id, **kwargs)

    async def create_character_memory(
        self,
        character_id: str,
        request_data: CharacterMemoryCreate | Mapping[str, Any],
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterMemoryCreate)
            else CharacterMemoryCreate.model_validate(dict(request_data))
        )
        return await client.create_character_memory(character_id, request)

    async def update_character_memory(
        self,
        character_id: str,
        memory_id: str,
        request_data: CharacterMemoryUpdate | Mapping[str, Any],
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterMemoryUpdate)
            else CharacterMemoryUpdate.model_validate(dict(request_data))
        )
        return await client.update_character_memory(character_id, memory_id, request)

    async def delete_character_memory(self, character_id: str, memory_id: str) -> Any:
        client = self._require_client()
        return await client.delete_character_memory(character_id, memory_id)

    async def archive_character_memory(
        self,
        character_id: str,
        memory_id: str,
        request_data: CharacterMemoryArchiveRequest | Mapping[str, Any],
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterMemoryArchiveRequest)
            else CharacterMemoryArchiveRequest.model_validate(dict(request_data))
        )
        return await client.archive_character_memory(character_id, memory_id, request)

    async def extract_character_memories(
        self,
        character_id: str,
        request_data: CharacterMemoryExtractRequest | Mapping[str, Any],
    ) -> Any:
        client = self._require_client()
        request = (
            request_data
            if isinstance(request_data, CharacterMemoryExtractRequest)
            else CharacterMemoryExtractRequest.model_validate(dict(request_data))
        )
        return await client.extract_character_memories(character_id, request)
