"""Thin server-backed media-reading service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..tldw_api import (
    IngestWebContentRequest,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    MediaSearchRequest,
    MediaIngestJobSubmitRequest,
    ReadingArchiveCreateRequest,
    ReadingHighlightCreateRequest,
    ReadingHighlightUpdateRequest,
    ReadingNoteLinkCreateRequest,
    ReadingProgressUpdate,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchUpdateRequest,
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
        from ..runtime_policy.bootstrap import build_runtime_api_client_from_config

        return cls(client=build_runtime_api_client_from_config(app_config))

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

    async def create_reading_highlight(
        self,
        item_id: Any,
        *,
        quote: str,
        start_offset: int | None = None,
        end_offset: int | None = None,
        color: str | None = None,
        note: str | None = None,
        anchor_strategy: str = "fuzzy_quote",
    ) -> Any:
        normalized_item_id = int(item_id)
        request_data = ReadingHighlightCreateRequest(
            item_id=normalized_item_id,
            quote=quote,
            start_offset=start_offset,
            end_offset=end_offset,
            color=color,
            note=note,
            anchor_strategy=anchor_strategy,
        )
        return await self._require_client().create_reading_highlight(normalized_item_id, request_data)

    async def list_reading_highlights(self, item_id: Any) -> Any:
        return await self._require_client().list_reading_highlights(int(item_id))

    async def update_reading_highlight(self, highlight_id: Any, **changes: Any) -> Any:
        request_data = ReadingHighlightUpdateRequest(**changes)
        return await self._require_client().update_reading_highlight(int(highlight_id), request_data)

    async def delete_reading_highlight(self, highlight_id: Any) -> Any:
        return await self._require_client().delete_reading_highlight(int(highlight_id))

    async def list_ingestion_sources(self) -> Any:
        return await self._require_client().list_ingestion_sources()

    async def create_ingestion_source(
        self,
        *,
        source_type: str,
        sink_type: str,
        policy: str = "canonical",
        enabled: bool = True,
        schedule_enabled: bool = False,
        schedule: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        request_data = IngestionSourceCreateRequest(
            source_type=source_type,
            sink_type=sink_type,
            policy=policy,
            enabled=enabled,
            schedule_enabled=schedule_enabled,
            schedule=dict(schedule or {}),
            config=dict(config or {}),
        )
        return await self._require_client().create_ingestion_source(request_data)

    async def get_ingestion_source(self, source_id: Any) -> Any:
        return await self._require_client().get_ingestion_source(int(source_id))

    async def patch_ingestion_source(self, source_id: Any, **changes: Any) -> Any:
        request_data = IngestionSourcePatchRequest(**changes)
        return await self._require_client().patch_ingestion_source(int(source_id), request_data)

    async def list_ingestion_source_items(self, source_id: Any) -> Any:
        return await self._require_client().list_ingestion_source_items(int(source_id))

    async def reattach_ingestion_source_item(self, source_id: Any, item_id: Any) -> Any:
        return await self._require_client().reattach_ingestion_source_item(int(source_id), int(item_id))

    async def trigger_ingestion_source_sync(self, source_id: Any) -> Any:
        return await self._require_client().trigger_ingestion_source_sync(int(source_id))

    async def upload_ingestion_source_archive(self, source_id: Any, archive_path: str) -> Any:
        return await self._require_client().upload_ingestion_source_archive(int(source_id), archive_path)

    async def create_reading_saved_search(
        self,
        *,
        name: str,
        query: Mapping[str, Any] | None = None,
        sort: str | None = None,
    ) -> Any:
        request_data = ReadingSavedSearchCreateRequest(
            name=name,
            query=dict(query or {}),
            sort=sort,
        )
        return await self._require_client().create_reading_saved_search(request_data)

    async def list_reading_saved_searches(self, *, limit: int = 50, offset: int = 0) -> Any:
        return await self._require_client().list_reading_saved_searches(limit=limit, offset=offset)

    async def update_reading_saved_search(self, search_id: Any, **changes: Any) -> Any:
        request_data = ReadingSavedSearchUpdateRequest(
            **{key: value for key, value in changes.items() if value is not None}
        )
        return await self._require_client().update_reading_saved_search(int(search_id), request_data)

    async def delete_reading_saved_search(self, search_id: Any) -> Any:
        return await self._require_client().delete_reading_saved_search(int(search_id))

    async def link_reading_item_note(self, item_id: Any, *, note_id: str) -> Any:
        request_data = ReadingNoteLinkCreateRequest(note_id=note_id)
        return await self._require_client().link_reading_item_note(int(item_id), request_data)

    async def list_reading_item_note_links(self, item_id: Any) -> Any:
        return await self._require_client().list_reading_item_note_links(int(item_id))

    async def unlink_reading_item_note(self, item_id: Any, note_id: str) -> Any:
        return await self._require_client().unlink_reading_item_note(int(item_id), str(note_id))

    async def import_reading_items(
        self,
        file_path: str,
        *,
        source: str = "auto",
        merge_tags: bool = True,
    ) -> Any:
        return await self._require_client().import_reading_items(
            file_path,
            source=source,
            merge_tags=merge_tags,
        )

    async def list_reading_import_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        return await self._require_client().list_reading_import_jobs(
            status=status,
            limit=limit,
            offset=offset,
        )

    async def get_reading_import_job(self, job_id: Any) -> Any:
        return await self._require_client().get_reading_import_job(int(job_id))

    async def create_reading_archive(
        self,
        item_id: Any,
        *,
        format: str = "html",
        source: str = "auto",
        title: str | None = None,
        retention_days: int | None = None,
        retention_until: str | None = None,
    ) -> Any:
        request_data = ReadingArchiveCreateRequest(
            format=format,
            source=source,
            title=title,
            retention_days=retention_days,
            retention_until=retention_until,
        )
        return await self._require_client().create_reading_archive(int(item_id), request_data)

    async def submit_media_ingest_jobs(
        self,
        *,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        request_data = MediaIngestJobSubmitRequest(
            media_type=media_type,
            urls=urls,
            **{key: value for key, value in options.items() if value is not None},
        )
        return await self._require_client().submit_media_ingest_jobs(
            request_data,
            file_paths=file_paths,
        )

    async def get_media_ingest_job(self, job_id: Any) -> Any:
        return await self._require_client().get_media_ingest_job(int(job_id))

    async def list_media_ingest_jobs(self, *, batch_id: str, limit: int = 100) -> Any:
        return await self._require_client().list_media_ingest_jobs(batch_id=batch_id, limit=limit)

    async def cancel_media_ingest_job(self, job_id: Any, *, reason: str | None = None) -> Any:
        return await self._require_client().cancel_media_ingest_job(int(job_id), reason=reason)

    async def cancel_media_ingest_jobs_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> Any:
        return await self._require_client().cancel_media_ingest_jobs_batch(
            batch_id=batch_id,
            session_id=session_id,
            reason=reason,
        )

    async def stream_media_ingest_job_events(
        self,
        *,
        batch_id: str | None = None,
        after_id: int = 0,
    ) -> Any:
        async for event in self._require_client().stream_media_ingest_job_events(
            batch_id=batch_id,
            after_id=after_id,
        ):
            if hasattr(event, "model_dump"):
                yield event.model_dump(exclude_none=True, mode="json")
            else:
                yield event

    async def ingest_web_content(self, *, urls: list[str], **options: Any) -> Any:
        request_data = IngestWebContentRequest(
            urls=list(urls),
            **{key: value for key, value in options.items() if value is not None},
        )
        return await self._require_client().ingest_web_content(request_data)

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
