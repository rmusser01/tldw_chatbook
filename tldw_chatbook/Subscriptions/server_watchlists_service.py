"""Thin server-backed watchlists service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from pydantic import AnyUrl, BaseModel, ConfigDict, Field

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import SourceCreateRequest, TLDWAPIClient
from .watchlist_normalizers import normalize_server_watchlist_source

_UNSET = object()


class _ExtendedSourceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    url: AnyUrl | None = None
    source_type: str | None = None
    active: bool | None = None
    tags: list[str] | None = None
    settings: dict[str, Any] | None = None


class ServerWatchlistsService:
    """Thin wrapper around server-backed watchlist endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerWatchlistsService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server watchlist operations.")
        return self.client

    @staticmethod
    def _coerce_items(payload: Any) -> list[dict[str, Any]]:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if isinstance(payload, Mapping):
            raw_items = payload.get("items", [])
            if isinstance(raw_items, list):
                return [dict(item) for item in raw_items]
            return []
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        return []

    @staticmethod
    def _reject_forum_source_type(source_type: Any) -> None:
        if source_type is _UNSET:
            return
        if str(source_type or "").strip().lower() == "forum":
            raise ValueError("Forum sources are not supported in the first slice.")

    @staticmethod
    def _with_optional_fields(**fields: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in fields.items():
            if value is _UNSET:
                continue
            payload[key] = value
        return payload

    async def list_sources(
        self,
        *,
        q: str | None = None,
        tags: list[str] | None = None,
        page: int = 1,
        size: int = 50,
    ) -> dict[str, Any]:
        response = await self._require_client().list_watchlist_sources(q=q, tags=tags, page=page, size=size)
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        items = [normalize_server_watchlist_source(item) for item in self._coerce_items(payload)]
        total = payload.get("total", len(items)) if isinstance(payload, Mapping) else len(items)
        return {"items": items, "total": total, "page": page, "size": size}

    async def get_source_detail(self, source_id: int) -> dict[str, Any]:
        response = await self._require_client().list_watchlist_sources(page=1, size=200)
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        for item in self._coerce_items(payload):
            if int(item.get("id")) == int(source_id):
                return normalize_server_watchlist_source(item)
        raise ValueError(f"Server watchlist source {source_id} was not found.")

    async def create_source(
        self,
        *,
        name: str,
        url: str,
        source_type: str,
        active: bool = True,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        self._reject_forum_source_type(source_type)
        payload = SourceCreateRequest(
            name=name,
            url=url,
            source_type=str(source_type).strip().lower(),
            active=active,
            tags=tags,
        )
        response = await self._require_client().create_watchlist_source(payload)
        return normalize_server_watchlist_source(response)

    async def update_source(
        self,
        source_id: int,
        *,
        name: Any = _UNSET,
        url: Any = _UNSET,
        source_type: Any = _UNSET,
        active: Any = _UNSET,
        tags: Any = _UNSET,
        existing_settings: Any = None,
    ) -> dict[str, Any]:
        self._reject_forum_source_type(source_type)
        payload = _ExtendedSourceUpdateRequest(
            **self._with_optional_fields(
                name=name,
                url=url,
                source_type=str(source_type).strip().lower() if source_type is not _UNSET else _UNSET,
                active=active,
                tags=tags,
                settings=dict(existing_settings) if isinstance(existing_settings, Mapping) else existing_settings,
            )
        )
        response = await self._require_client().update_watchlist_source(int(source_id), payload)
        return normalize_server_watchlist_source(response)

    async def delete_source(self, source_id: int) -> dict[str, Any]:
        response = await self._require_client().delete_watchlist_source(int(source_id))
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, Mapping):
            return dict(response)
        return {"deleted": True, "source_id": int(source_id)}


__all__ = ["ServerWatchlistsService"]
