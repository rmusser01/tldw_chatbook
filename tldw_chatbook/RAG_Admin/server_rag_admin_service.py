"""Thin server-backed retrieval-admin service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import (
    ChunkingTemplateApplyRequest,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateUpdateRequest,
    TLDWAPIClient,
)


class ServerRAGAdminService:
    """Thin wrapper around server chunking-template and collection admin endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerRAGAdminService":
        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server retrieval-admin operations.")
        return self.client

    def _dump_model(self, value: Any) -> Any:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [self._dump_model(item) for item in value]
        return value

    async def list_templates(
        self,
        *,
        include_builtin: bool = True,
        include_custom: bool = True,
        tags: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        response = await self._require_client().list_chunking_templates(
            include_builtin=include_builtin,
            include_custom=include_custom,
            tags=list(tags) if tags is not None else None,
            user_id=user_id,
        )
        payload = self._dump_model(response)
        return list(payload.get("templates", []))

    async def get_template(self, template_name: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_chunking_template(template_name))

    async def create_template(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        template: Mapping[str, Any],
        tags: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        request = ChunkingTemplateCreateRequest(
            name=name,
            description=description,
            template=dict(template),
            tags=list(tags or []),
            user_id=user_id,
        )
        return self._dump_model(await self._require_client().create_chunking_template(request))

    async def update_template(
        self,
        template_name: str,
        *,
        description: Optional[str] = None,
        template: Optional[Mapping[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        request = ChunkingTemplateUpdateRequest(
            description=description,
            template=dict(template) if template is not None else None,
            tags=list(tags) if tags is not None else None,
        )
        return self._dump_model(
            await self._require_client().update_chunking_template(template_name, request)
        )

    async def delete_template(self, template_name: str, *, hard_delete: bool = False) -> None:
        await self._require_client().delete_chunking_template(template_name, hard_delete=hard_delete)

    async def apply_template(
        self,
        template_name: str,
        *,
        text: str,
        override_options: Optional[Mapping[str, Any]] = None,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        request = ChunkingTemplateApplyRequest(
            template_name=template_name,
            text=text,
            override_options=dict(override_options) if override_options is not None else None,
        )
        return self._dump_model(
            await self._require_client().apply_chunking_template(
                request,
                include_metadata=include_metadata,
            )
        )

    async def get_template_diagnostics(self) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_chunking_template_diagnostics())

    async def list_collections(self) -> list[dict[str, Any]]:
        return self._dump_model(await self._require_client().list_embedding_collections())

    async def get_collection_detail(self, collection_name: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_embedding_collection_stats(collection_name))

    async def delete_collection(self, collection_name: str) -> None:
        await self._require_client().delete_embedding_collection(collection_name)
