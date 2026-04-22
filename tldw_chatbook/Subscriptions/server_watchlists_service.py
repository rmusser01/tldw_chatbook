"""Thin server-backed watchlists service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import SourceCreateRequest, SourceUpdateRequest, TLDWAPIClient
from .watchlist_normalizers import normalize_server_watchlist_source

_UNSET = object()


class _ExtendedSourceUpdateRequest(SourceUpdateRequest):
    model_config = ConfigDict(extra="forbid")
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

    @staticmethod
    def _payload_to_mapping(payload: Any) -> dict[str, Any]:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    @staticmethod
    def _filtered_normalized_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            try:
                normalized_items.append(normalize_server_watchlist_source(item))
            except ValueError:
                continue
        return normalized_items

    async def list_sources(
        self,
        *,
        q: str | None = None,
        tags: list[str] | None = None,
        page: int = 1,
        size: int = 50,
    ) -> dict[str, Any]:
        response = await self._require_client().list_watchlist_sources(q=q, tags=tags, page=page, size=size)
        payload = self._payload_to_mapping(response)
        items = self._filtered_normalized_items(self._coerce_items(payload))
        total = len(items)
        return {"items": items, "total": total, "page": page, "size": size}

    async def get_source_detail(self, source_id: int) -> dict[str, Any]:
        page = 1
        size = 200
        while True:
            response = await self._require_client().list_watchlist_sources(q=None, tags=None, page=page, size=size)
            payload = self._payload_to_mapping(response)
            items = self._coerce_items(payload)
            for item in items:
                if int(item.get("id")) == int(source_id):
                    return normalize_server_watchlist_source(item)
            total = payload.get("total") if isinstance(payload, Mapping) else None
            if not items:
                break
            if isinstance(total, int) and page * size >= total:
                break
            if len(items) < size and total in (None, "", 0):
                break
            page += 1
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
        normalized_source_type = (
            str(source_type).strip().lower() if source_type is not _UNSET else _UNSET
        )
        validated = SourceUpdateRequest(
            **self._with_optional_fields(
                name=name,
                url=url,
                source_type=normalized_source_type,
                active=active,
                tags=tags,
            )
        )
        payload_dict = validated.model_dump(exclude_none=True, mode="json")
        if isinstance(existing_settings, Mapping):
            payload_dict["settings"] = dict(existing_settings)
        elif existing_settings is not None:
            payload_dict["settings"] = existing_settings
        payload = _ExtendedSourceUpdateRequest(**payload_dict)
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
