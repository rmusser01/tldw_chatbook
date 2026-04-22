"""
Thin service helpers for server-backed character and persona catalog access.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import (
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
        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server character and persona operations.")
        return self.client

    async def list_characters(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        client = self._require_client()
        return await client.list_characters(limit=limit, offset=offset)

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
