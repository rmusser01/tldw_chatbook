"""Server-backed research search and provider surface helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    PaperSearchDetailRequest,
    PaperSearchIngestRequest,
    PaperSearchRequest,
    TLDWAPIClient,
    WebSearchRequest,
)


class ServerResearchSearchService:
    """Wrap server research search endpoints with plain dict payloads."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerResearchSearchService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server research search operations.")
        return self.client

    @staticmethod
    def _dump_model(value: Any) -> Any:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return value.model_dump(mode="json")
        return value

    async def websearch(self, **payload: Any) -> dict[str, Any]:
        request = WebSearchRequest(**payload)
        return dict(self._dump_model(await self._require_client().run_research_websearch(request)))

    async def paper_search(self, *, endpoint: str, **params: Any) -> dict[str, Any]:
        request = PaperSearchRequest(endpoint=endpoint, params=params)
        return dict(self._dump_model(await self._require_client().run_paper_search(request)))

    async def paper_detail(self, *, endpoint: str, **params: Any) -> dict[str, Any]:
        request = PaperSearchDetailRequest(endpoint=endpoint, params=params)
        return dict(self._dump_model(await self._require_client().get_paper_search_detail(request)))

    async def paper_ingest(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        request = PaperSearchIngestRequest(endpoint=endpoint, params=params, payload=payload)
        return dict(self._dump_model(await self._require_client().run_paper_search_ingest(request)))
