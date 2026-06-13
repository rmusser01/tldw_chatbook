"""Thin server-backed Web Clipper service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..tldw_api import (
    TLDWAPIClient,
    WebClipperEnrichmentPayload,
    WebClipperSaveRequest,
)


class ServerWebClipperService:
    """Thin wrapper around the server Web Clipper endpoints."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerWebClipperService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
        )

    @classmethod
    def from_server_context_provider(cls, provider: Any) -> "ServerWebClipperService":
        return cls(client=None, client_provider=provider)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server Web Clipper operations.")

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True, mode="json")
        raise TypeError("Expected a mapping or Pydantic model payload.")

    async def save_clip(self, **payload: Any) -> Any:
        request_data = WebClipperSaveRequest(**payload)
        result = await self._require_client().save_web_clip(request_data)
        return self._as_dict(result)

    async def get_clip_status(self, clip_id: str) -> Any:
        result = await self._require_client().get_web_clip_status(str(clip_id))
        return self._as_dict(result)

    async def persist_enrichment(self, target_clip_id: str, **payload: Any) -> Any:
        request_data = WebClipperEnrichmentPayload(**payload)
        result = await self._require_client().persist_web_clip_enrichment(str(target_clip_id), request_data)
        return self._as_dict(result)
