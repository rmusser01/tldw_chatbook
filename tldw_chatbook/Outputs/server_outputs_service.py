"""Thin server-backed outputs/templates/artifacts service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..tldw_api import (
    OutputCreateRequest,
    OutputsPurgeRequest,
    OutputTemplateCreateRequest,
    OutputTemplatePreviewRequest,
    OutputTemplateUpdateRequest,
    OutputUpdateRequest,
    TLDWAPIClient,
)


class ServerOutputsService:
    """Thin wrapper around server outputs and output-template endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerOutputsService":
        from ..runtime_policy.bootstrap import build_runtime_api_client_from_config

        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server outputs operations.")
        return self.client

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True, mode="json")
        raise TypeError("Expected a mapping or Pydantic model payload.")

    async def list_output_templates(
        self,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        result = await self._require_client().list_output_templates(q=q, limit=limit, offset=offset)
        return self._as_dict(result)

    async def create_output_template(self, **payload: Any) -> dict[str, Any]:
        request_data = OutputTemplateCreateRequest(**payload)
        result = await self._require_client().create_output_template(request_data)
        return self._as_dict(result)

    async def get_output_template(self, template_id: int) -> dict[str, Any]:
        result = await self._require_client().get_output_template(int(template_id))
        return self._as_dict(result)

    async def update_output_template(self, template_id: int, **payload: Any) -> dict[str, Any]:
        request_data = OutputTemplateUpdateRequest(**payload)
        result = await self._require_client().update_output_template(int(template_id), request_data)
        return self._as_dict(result)

    async def delete_output_template(self, template_id: int) -> dict[str, Any]:
        return self._as_dict(await self._require_client().delete_output_template(int(template_id)))

    async def preview_output_template(self, template_id: int, **payload: Any) -> dict[str, Any]:
        request_data = OutputTemplatePreviewRequest(template_id=int(template_id), **payload)
        result = await self._require_client().preview_output_template(int(template_id), request_data)
        return self._as_dict(result)

    async def list_outputs(self, **payload: Any) -> dict[str, Any]:
        result = await self._require_client().list_outputs(**payload)
        return self._as_dict(result)

    async def list_deleted_outputs(self, *, page: int = 1, size: int = 50) -> dict[str, Any]:
        result = await self._require_client().list_deleted_outputs(page=page, size=size)
        return self._as_dict(result)

    async def create_output(self, **payload: Any) -> dict[str, Any]:
        request_data = OutputCreateRequest(**payload)
        result = await self._require_client().create_output(request_data)
        return self._as_dict(result)

    async def get_output(self, output_id: int) -> dict[str, Any]:
        result = await self._require_client().get_output(int(output_id))
        return self._as_dict(result)

    async def update_output(self, output_id: int, **payload: Any) -> dict[str, Any]:
        request_data = OutputUpdateRequest(**payload)
        result = await self._require_client().update_output(int(output_id), request_data)
        return self._as_dict(result)

    async def delete_output(self, output_id: int, *, hard: bool = False, delete_file: bool = False) -> dict[str, Any]:
        return self._as_dict(
            await self._require_client().delete_output(
                int(output_id),
                hard=hard,
                delete_file=delete_file,
            )
        )

    async def download_output(self, output_id: int) -> bytes:
        return await self._require_client().download_output(int(output_id))

    async def download_output_by_name(self, title: str, *, format: str | None = None) -> bytes:
        return await self._require_client().download_output_by_name(str(title), format=format)

    async def purge_outputs(self, **payload: Any) -> dict[str, Any]:
        request_data = OutputsPurgeRequest(**payload)
        result = await self._require_client().purge_outputs(request_data)
        return self._as_dict(result)
