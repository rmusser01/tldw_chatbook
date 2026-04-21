"""Thin server-backed media-reading service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    IngestionSourcePatchRequest,
    MediaSearchRequest,
    ReadingProgressUpdate,
    ReadingUpdateRequest,
    TLDWAPIClient,
)


class ServerMediaReadingService:
    """Thin wrapper around server-backed media-reading endpoints."""

    _SUPPORTED_METADATA_FIELDS = {"status", "favorite", "tags", "notes", "title"}

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerMediaReadingService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server media operations.")
        return self.client

    async def search_media(
        self,
        *,
        query: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        **filters: Any,
    ) -> Any:
        client = self._require_client()
        params = {key: value for key, value in filters.items() if value is not None}
        params.update({"q": query, "limit": limit, "offset": offset})
        return await client.list_reading_items(**{key: value for key, value in params.items() if value is not None})

    async def get_media_detail(self, media_id: Any) -> Any:
        return await self._require_client().get_reading_item(int(media_id))

    async def update_media_metadata(self, media_id: Any, **metadata: Any) -> Any:
        unsupported = sorted(
            key for key, value in metadata.items()
            if value is not None and key not in self._SUPPORTED_METADATA_FIELDS
        )
        if unsupported:
            unsupported_text = ", ".join(unsupported)
            raise ValueError(f"Unsupported server media metadata fields: {unsupported_text}")

        payload = {key: value for key, value in metadata.items() if value is not None}
        request_data = ReadingUpdateRequest(**payload)
        return await self._require_client().update_reading_item(int(media_id), request_data)

    async def delete_media(self, media_id: Any) -> Any:
        return await self._require_client().delete_reading_item(int(media_id), hard=False)

    async def undelete_media(self, media_id: Any) -> Any:
        raise ValueError("Server media undelete is not available yet.")

    async def get_reading_progress(self, media_id: Any) -> Any:
        return await self._require_client().get_reading_progress(int(media_id))

    async def update_reading_progress(self, media_id: Any, progress_data: Mapping[str, Any]) -> Any:
        payload = dict(progress_data)
        if "percent_complete" in payload and "percentage" not in payload:
            payload["percentage"] = payload.pop("percent_complete")
        request_data = ReadingProgressUpdate(**payload)
        return await self._require_client().update_reading_progress(int(media_id), request_data)

    async def delete_reading_progress(self, media_id: Any) -> Any:
        return await self._require_client().delete_reading_progress(int(media_id))

    async def list_ingestion_sources(self) -> Any:
        return await self._require_client().list_ingestion_sources()

    async def get_ingestion_source(self, source_id: Any) -> Any:
        return await self._require_client().get_ingestion_source(int(source_id))

    async def patch_ingestion_source(self, source_id: Any, **changes: Any) -> Any:
        request_data = IngestionSourcePatchRequest(**changes)
        return await self._require_client().patch_ingestion_source(int(source_id), request_data)

    async def list_ingestion_source_items(self, source_id: Any) -> Any:
        return await self._require_client().list_ingestion_source_items(int(source_id))

    async def trigger_ingestion_source_sync(self, source_id: Any) -> Any:
        return await self._require_client().trigger_ingestion_source_sync(int(source_id))

    async def upload_ingestion_source_archive(self, source_id: Any, archive_path: str) -> Any:
        return await self._require_client().upload_ingestion_source_archive(int(source_id), archive_path)

    async def list_document_versions(self, media_id: Any, *, include_deleted: bool = False) -> Any:
        raise ValueError("Server document versions are not available yet.")

    async def save_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        raise ValueError("Server document versions are not available yet.")

    async def overwrite_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        raise ValueError("Server document versions are not available yet.")

    async def delete_analysis_version(self, version_uuid: str) -> Any:
        raise ValueError("Server document versions are not available yet.")
