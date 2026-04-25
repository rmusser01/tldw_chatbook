"""Thin server-backed watchlists source service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import (
    SourceCreateRequest,
    SourceUpdateRequest,
    TLDWAPIClient,
)
from .watchlist_normalizers import (
    normalize_server_delete_response,
    normalize_server_watchlist_source,
)


_UNSET = object()


class ServerWatchlistsService:
    """First-slice server watchlist source CRUD service."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerWatchlistsService":
        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server watchlist operations.")
        return self.client

    async def list_sources(
        self,
        *,
        q: str | None = None,
        tags: list[str] | None = None,
        source_type: str | None = None,
        active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        response = await self._require_client().list_watchlist_sources(
            q=q,
            tags=tags,
            source_type=source_type,
            active=active,
            limit=limit,
            offset=offset,
        )
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        return [normalize_server_watchlist_source(item) for item in list(payload.get("items", []))]

    async def get_source(self, source_id: Any) -> dict[str, Any]:
        response = await self._require_client().get_watchlist_source(int(source_id))
        return normalize_server_watchlist_source(response)

    async def create_source(
        self,
        *,
        name: str,
        url: str,
        source_type: str,
        active: bool = True,
        tags: list[str] | None = None,
        settings: Mapping[str, Any] | None = None,
        group_ids: Any = _UNSET,
    ) -> dict[str, Any]:
        if group_ids is not _UNSET:
            raise ValueError("Server watchlist group editing is deferred in this slice.")
        request = SourceCreateRequest(
            name=name,
            url=url,
            source_type=self._validate_source_type(source_type),
            active=active,
            tags=list(tags or []),
            settings=dict(settings or {}),
        )
        response = await self._require_client().create_watchlist_source(request)
        return normalize_server_watchlist_source(response)

    async def update_source(
        self,
        source_id: Any,
        *,
        name: Any = _UNSET,
        url: Any = _UNSET,
        source_type: Any = _UNSET,
        active: Any = _UNSET,
        tags: Any = _UNSET,
        settings: Any = _UNSET,
        existing_settings: Mapping[str, Any] | None = None,
        group_ids: Any = _UNSET,
    ) -> dict[str, Any]:
        if group_ids is not _UNSET:
            raise ValueError("Server watchlist group editing is deferred in this slice.")
        payload: dict[str, Any] = {}
        if name is not _UNSET:
            payload["name"] = name
        if url is not _UNSET:
            payload["url"] = url
        if source_type is not _UNSET:
            payload["source_type"] = self._validate_source_type(source_type)
        if active is not _UNSET:
            payload["active"] = bool(active)
        if tags is not _UNSET:
            payload["tags"] = list(tags or [])
        if settings is not _UNSET:
            payload["settings"] = dict(settings or {})
        elif existing_settings is not None:
            payload["settings"] = dict(existing_settings)

        request = SourceUpdateRequest(**payload)
        response = await self._require_client().update_watchlist_source(int(source_id), request)
        return normalize_server_watchlist_source(response)

    async def delete_source(self, source_id: Any) -> dict[str, Any]:
        response = await self._require_client().delete_watchlist_source(int(source_id))
        return normalize_server_delete_response(response, source_id=source_id)

    @staticmethod
    def _validate_source_type(source_type: Any) -> str:
        normalized = str(source_type or "").strip()
        if normalized == "forum":
            raise ValueError("Forum sources are not supported in the first watchlists slice.")
        if normalized not in {"rss", "site"}:
            raise ValueError("Only rss and site watchlist sources are supported in this slice.")
        return normalized
