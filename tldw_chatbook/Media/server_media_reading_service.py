"""Thin server-backed media-reading service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..tldw_api import (
    DocumentAnnotationCreateRequest,
    DocumentAnnotationSyncRequest,
    DocumentAnnotationUpdateRequest,
    DocumentInsightsRequest,
    DocumentVersionAdvancedUpsertRequest,
    DocumentVersionCreateRequest,
    DocumentVersionMetadataPatchRequest,
    DocumentVersionRollbackRequest,
    IngestWebContentRequest,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    ItemsBulkRequest,
    MediaSearchRequest,
    MediaIngestJobSubmitRequest,
    MediaKeywordsUpdateRequest,
    MediaUpdateRequest,
    ProcessAudioRequest,
    ProcessCodeRequest,
    ProcessDocumentRequest,
    ProcessEbookRequest,
    ProcessEmailRequest,
    ProcessMediaWikiRequest,
    ProcessPDFRequest,
    ProcessVideoRequest,
    ReadingArchiveCreateRequest,
    ReadingDigestScheduleCreateRequest,
    ReadingDigestScheduleUpdateRequest,
    ReadingExportRequest,
    ReadingHighlightCreateRequest,
    ReadingHighlightUpdateRequest,
    ReadingNoteLinkCreateRequest,
    ReadingProgressUpdate,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchUpdateRequest,
    ReadingSummarizeRequest,
    ReadingTTSRequest,
    ReadingUpdateRequest,
    ReprocessMediaRequest,
    TLDWAPIClient,
    WebScrapingRequest,
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

    async def list_media_keywords(
        self,
        *,
        query: str | None = None,
        limit: int = 100,
    ) -> Any:
        return await self._require_client().list_media_keywords(query=query, limit=limit)

    async def list_backing_media_items(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> Any:
        return await self._require_client().list_media_items(
            page=page,
            results_per_page=results_per_page,
            include_keywords=include_keywords,
        )

    async def search_backing_media_items(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
        **filters: Any,
    ) -> Any:
        request_data = MediaSearchRequest(**{key: value for key, value in filters.items() if value is not None})
        return await self._require_client().search_media_items(
            request_data,
            page=page,
            results_per_page=results_per_page,
        )

    async def list_media_trash(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> Any:
        return await self._require_client().list_media_trash(
            page=page,
            results_per_page=results_per_page,
            include_keywords=include_keywords,
        )

    async def empty_media_trash(self) -> Any:
        return await self._require_client().empty_media_trash()

    async def search_media_metadata(self, **filters: Any) -> Any:
        return await self._require_client().search_media_metadata(
            **{key: value for key, value in filters.items() if value is not None}
        )

    async def get_media_by_identifier(self, **identifiers: Any) -> Any:
        return await self._require_client().get_media_by_identifier(
            **{key: value for key, value in identifiers.items() if value is not None}
        )

    async def get_media_transcription_models(self) -> Any:
        return await self._require_client().get_media_transcription_models()

    async def reprocess_media(self, media_id: Any, **options: Any) -> Any:
        request_data = ReprocessMediaRequest(**{key: value for key, value in options.items() if value is not None})
        return await self._require_client().reprocess_media(int(media_id), request_data)

    async def process_video(self, request_data: ProcessVideoRequest, file_paths: list[str] | None = None) -> Any:
        return await self._require_client().process_video(request_data, file_paths=file_paths)

    async def process_audio(self, request_data: ProcessAudioRequest, file_paths: list[str] | None = None) -> Any:
        return await self._require_client().process_audio(request_data, file_paths=file_paths)

    async def process_pdf(self, request_data: ProcessPDFRequest, file_paths: list[str] | None = None) -> Any:
        return await self._require_client().process_pdf(request_data, file_paths=file_paths)

    async def process_ebook(self, request_data: ProcessEbookRequest, file_paths: list[str] | None = None) -> Any:
        return await self._require_client().process_ebook(request_data, file_paths=file_paths)

    async def process_document(self, request_data: ProcessDocumentRequest, file_paths: list[str] | None = None) -> Any:
        return await self._require_client().process_document(request_data, file_paths=file_paths)

    async def process_code(self, request_data: ProcessCodeRequest, file_paths: list[str] | None = None) -> Any:
        return await self._require_client().process_code(request_data, file_paths=file_paths)

    async def process_email(self, request_data: ProcessEmailRequest, file_paths: list[str] | None = None) -> Any:
        return await self._require_client().process_email(request_data, file_paths=file_paths)

    async def process_mediawiki_dump(self, request_data: ProcessMediaWikiRequest, dump_file_path: str) -> Any:
        async for page in self._require_client().process_mediawiki_dump(request_data, dump_file_path):
            yield page.model_dump(exclude_none=True, mode="json") if hasattr(page, "model_dump") else page

    async def ingest_mediawiki_dump(self, request_data: ProcessMediaWikiRequest, dump_file_path: str) -> Any:
        async for event in self._require_client().ingest_mediawiki_dump(request_data, dump_file_path):
            yield event.model_dump(exclude_none=True, mode="json") if hasattr(event, "model_dump") else event

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

    async def bulk_update_reading_items(
        self,
        *,
        item_ids: list[int],
        action: str,
        status: str | None = None,
        favorite: bool | None = None,
        tags: list[str] | None = None,
        hard: bool = False,
    ) -> Any:
        request_data = ItemsBulkRequest(
            item_ids=[int(item_id) for item_id in item_ids],
            action=action,
            status=status,
            favorite=favorite,
            tags=tags,
            hard=hard,
        )
        return await self._require_client().bulk_update_reading_items(request_data)

    async def get_media_item(
        self,
        media_id: Any,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> Any:
        return await self._require_client().get_media_item(
            int(media_id),
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )

    async def update_media_item(self, media_id: Any, **changes: Any) -> Any:
        if changes.get("keywords") is not None:
            raise ValueError("Use update_media_keywords for server media keyword changes.")
        payload = {key: value for key, value in changes.items() if value is not None}
        request_data = MediaUpdateRequest(**payload)
        return await self._require_client().update_media_item(int(media_id), request_data)

    async def trash_media_item(self, media_id: Any) -> Any:
        return await self._require_client().trash_media_item(int(media_id))

    async def restore_media_item(
        self,
        media_id: Any,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> Any:
        return await self._require_client().restore_media_item(
            int(media_id),
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )

    async def permanently_delete_media_item(self, media_id: Any) -> Any:
        return await self._require_client().permanently_delete_media_item(int(media_id))

    async def update_media_keywords(
        self,
        media_id: Any,
        *,
        keywords: list[str],
        mode: str = "add",
    ) -> Any:
        request_data = MediaKeywordsUpdateRequest(keywords=keywords, mode=mode)
        return await self._require_client().update_media_keywords(int(media_id), request_data)

    async def download_media_file(self, media_id: Any, *, file_type: str = "original") -> bytes:
        return await self._require_client().download_media_file(int(media_id), file_type=file_type)

    async def get_media_navigation(
        self,
        media_id: Any,
        *,
        include_generated_fallback: bool = False,
        max_depth: int = 4,
        max_nodes: int = 500,
        parent_id: str | None = None,
    ) -> Any:
        return await self._require_client().get_media_navigation(
            int(media_id),
            include_generated_fallback=include_generated_fallback,
            max_depth=max_depth,
            max_nodes=max_nodes,
            parent_id=parent_id,
        )

    async def get_media_navigation_content(
        self,
        media_id: Any,
        node_id: str,
        *,
        content_format: str = "auto",
        include_alternates: bool = False,
    ) -> Any:
        return await self._require_client().get_media_navigation_content(
            int(media_id),
            str(node_id),
            content_format=content_format,
            include_alternates=include_alternates,
        )

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

    async def delete_ingestion_source(self, source_id: Any) -> Any:
        raise ValueError("Server ingestion source delete is not available yet.")

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

    async def export_reading_items(self, **filters: Any) -> bytes:
        request_data = ReadingExportRequest(
            **{key: value for key, value in filters.items() if value is not None}
        )
        return await self._require_client().export_reading_items(request_data)

    async def summarize_reading_item(self, item_id: Any, **options: Any) -> Any:
        request_data = ReadingSummarizeRequest(
            **{key: value for key, value in options.items() if value is not None}
        )
        return await self._require_client().summarize_reading_item(int(item_id), request_data)

    async def tts_reading_item(self, item_id: Any, **options: Any) -> bytes:
        request_data = ReadingTTSRequest(
            **{key: value for key, value in options.items() if value is not None}
        )
        return await self._require_client().tts_reading_item(int(item_id), request_data)

    async def create_reading_digest_schedule(
        self,
        *,
        cron: str,
        name: str | None = None,
        timezone: str | None = None,
        enabled: bool = True,
        require_online: bool = False,
        format: str = "md",
        template_id: int | None = None,
        template_name: str | None = None,
        retention_days: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Any:
        request_data = ReadingDigestScheduleCreateRequest(
            name=name,
            cron=cron,
            timezone=timezone,
            enabled=enabled,
            require_online=require_online,
            format=format,
            template_id=template_id,
            template_name=template_name,
            retention_days=retention_days,
            filters=filters,
        )
        return await self._require_client().create_reading_digest_schedule(request_data)

    async def list_reading_digest_schedules(self, *, limit: int = 50, offset: int = 0) -> Any:
        return await self._require_client().list_reading_digest_schedules(limit=limit, offset=offset)

    async def get_reading_digest_schedule(self, schedule_id: str) -> Any:
        return await self._require_client().get_reading_digest_schedule(str(schedule_id))

    async def update_reading_digest_schedule(self, schedule_id: str, **changes: Any) -> Any:
        request_data = ReadingDigestScheduleUpdateRequest(
            **{key: value for key, value in changes.items() if value is not None}
        )
        return await self._require_client().update_reading_digest_schedule(str(schedule_id), request_data)

    async def delete_reading_digest_schedule(self, schedule_id: str) -> Any:
        return await self._require_client().delete_reading_digest_schedule(str(schedule_id))

    async def list_reading_digest_outputs(
        self,
        *,
        schedule_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        return await self._require_client().list_reading_digest_outputs(
            schedule_id=schedule_id,
            limit=limit,
            offset=offset,
        )

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

    async def process_web_scraping(self, request_data: WebScrapingRequest) -> Any:
        response = await self._require_client().process_web_scraping(request_data)
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def get_document_outline(self, media_id: Any) -> Any:
        response = await self._require_client().get_document_outline(int(media_id))
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def get_document_figures(self, media_id: Any, *, min_size: int = 50) -> Any:
        response = await self._require_client().get_document_figures(int(media_id), min_size=int(min_size))
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def list_document_annotations(self, media_id: Any) -> Any:
        response = await self._require_client().list_document_annotations(int(media_id))
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def create_document_annotation(
        self,
        media_id: Any,
        *,
        location: str,
        text: str,
        color: str = "yellow",
        note: str | None = None,
        annotation_type: str = "highlight",
        chapter_title: str | None = None,
        percentage: float | None = None,
    ) -> Any:
        request_data = DocumentAnnotationCreateRequest(
            location=location,
            text=text,
            color=color,
            note=note,
            annotation_type=annotation_type,
            chapter_title=chapter_title,
            percentage=percentage,
        )
        response = await self._require_client().create_document_annotation(int(media_id), request_data)
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def update_document_annotation(self, media_id: Any, annotation_id: str, **changes: Any) -> Any:
        request_data = DocumentAnnotationUpdateRequest(
            **{key: value for key, value in changes.items() if value is not None}
        )
        response = await self._require_client().update_document_annotation(
            int(media_id),
            annotation_id,
            request_data,
        )
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def delete_document_annotation(self, media_id: Any, annotation_id: str) -> Any:
        return await self._require_client().delete_document_annotation(int(media_id), annotation_id)

    async def sync_document_annotations(
        self,
        media_id: Any,
        *,
        annotations: list[Mapping[str, Any]],
        client_ids: list[str] | None = None,
    ) -> Any:
        request_data = DocumentAnnotationSyncRequest(
            annotations=[DocumentAnnotationCreateRequest(**dict(annotation)) for annotation in annotations],
            client_ids=client_ids,
        )
        response = await self._require_client().sync_document_annotations(int(media_id), request_data)
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def generate_document_insights(
        self,
        media_id: Any,
        *,
        categories: list[str] | None = None,
        model: str | None = None,
        max_content_length: int | None = 5000,
        force: bool | None = False,
    ) -> Any:
        request_data = DocumentInsightsRequest(
            categories=categories,
            model=model,
            max_content_length=max_content_length,
            force=force,
        )
        response = await self._require_client().generate_document_insights(int(media_id), request_data)
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def get_document_references(
        self,
        media_id: Any,
        *,
        enrich: bool = False,
        reference_index: int | None = None,
        offset: int = 0,
        limit: int = 20,
        parse_cap: int | None = None,
        search: str | None = None,
    ) -> Any:
        response = await self._require_client().get_document_references(
            int(media_id),
            enrich=enrich,
            reference_index=reference_index,
            offset=offset,
            limit=limit,
            parse_cap=parse_cap,
            search=search,
        )
        return response.model_dump(exclude_none=True, mode="json") if hasattr(response, "model_dump") else response

    async def list_document_versions(self, media_id: Any, *, include_deleted: bool = False) -> Any:
        if include_deleted:
            raise ValueError("Server deleted document version listing is not available yet.")
        versions = await self._require_client().list_media_document_versions(
            int(media_id),
            include_content=False,
            limit=100,
            page=1,
        )
        return [
            version.model_dump(exclude_none=True, mode="json")
            if hasattr(version, "model_dump")
            else dict(version)
            for version in versions
        ]

    async def get_analysis_version(
        self,
        media_id: Any,
        *,
        version_number: Any,
        include_content: bool = True,
    ) -> Any:
        version = await self._require_client().get_media_document_version(
            int(media_id),
            int(version_number),
            include_content=include_content,
        )
        return version.model_dump(exclude_none=True, mode="json") if hasattr(version, "model_dump") else version

    async def save_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        request_data = DocumentVersionCreateRequest(
            content=content,
            prompt=prompt or "",
            analysis_content=analysis_content,
        )
        return await self._require_client().create_media_document_version(int(media_id), request_data)

    async def overwrite_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        return await self.save_analysis_version(
            media_id,
            content=content,
            analysis_content=analysis_content,
            prompt=prompt,
        )

    async def delete_analysis_version(
        self,
        version_uuid: str,
        *,
        media_id: Any | None = None,
        version_number: Any | None = None,
    ) -> Any:
        if media_id is None or version_number is None:
            raise ValueError("Server document version delete requires media_id and version_number.")
        return await self._require_client().delete_media_document_version(
            int(media_id),
            int(version_number),
        )

    async def rollback_analysis_version(self, media_id: Any, *, version_number: Any) -> Any:
        request_data = DocumentVersionRollbackRequest(version_number=int(version_number))
        return await self._require_client().rollback_media_document_version(int(media_id), request_data)

    async def patch_latest_version_metadata(
        self,
        media_id: Any,
        *,
        safe_metadata: Mapping[str, Any],
        merge: bool = True,
        new_version: bool = False,
    ) -> Any:
        request_data = DocumentVersionMetadataPatchRequest(
            safe_metadata=dict(safe_metadata),
            merge=merge,
            new_version=new_version,
        )
        return await self._require_client().patch_media_document_metadata(int(media_id), request_data)

    async def update_analysis_version_metadata(
        self,
        media_id: Any,
        *,
        version_number: Any,
        safe_metadata: Mapping[str, Any],
        merge: bool = True,
    ) -> Any:
        request_data = DocumentVersionMetadataPatchRequest(
            safe_metadata=dict(safe_metadata),
            merge=merge,
        )
        return await self._require_client().update_media_document_version_metadata(
            int(media_id),
            int(version_number),
            request_data,
        )

    async def upsert_analysis_version(
        self,
        media_id: Any,
        *,
        content: str | None = None,
        prompt: str | None = None,
        analysis_content: str | None = None,
        safe_metadata: Mapping[str, Any] | None = None,
        merge: bool = True,
        new_version: bool = True,
    ) -> Any:
        request_data = DocumentVersionAdvancedUpsertRequest(
            content=content,
            prompt=prompt,
            analysis_content=analysis_content,
            safe_metadata=dict(safe_metadata) if safe_metadata is not None else None,
            merge=merge,
            new_version=new_version,
        )
        return await self._require_client().upsert_media_document_version(int(media_id), request_data)
