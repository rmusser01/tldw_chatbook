"""
Thin service helpers for server-backed character and persona catalog access.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import TLDWAPIClient


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

    async def list_characters(self) -> Any:
        client = self._require_client()
        return await client.list_characters()

    async def list_persona_profiles(
        self,
        active_only: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        client = self._require_client()
        return await client.list_persona_profiles()
