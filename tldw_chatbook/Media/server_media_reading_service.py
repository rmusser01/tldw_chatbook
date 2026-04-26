"""Thin server-backed media-reading service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    AddMediaRequest,
    DocumentAnnotationCreate,
    DocumentAnnotationSyncRequest,
    DocumentAnnotationUpdate,
    DocumentInsightsRequest,
    FileArtifactsPurgeRequest,
    FileCreateOptions,
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    IngestWebContentRequest,
    IngestWebContentResponse,
    ItemsBulkRequest,
    MediaIngestSubmitRequest,
    MediaItemUpdateRequest,
    MediaKeywordsUpdateRequest,
    MediaSearchRequest,
    MediaVersionCreateRequest,
    BatchMediaProcessResponse,
    ProcessAudioRequest,
    ProcessCodeRequest,
    ProcessDocumentRequest,
    ProcessEbookRequest,
    ProcessEmailsRequest,
    ProcessMediaWikiRequest,
    ProcessPDFRequest,
    ProcessPlaintextRequest,
    ProcessVideoRequest,
    ProcessWebScrapingRequest,
    ReadingArchiveCreateRequest,
    ReadingDigestScheduleCreateRequest,
    ReadingDigestScheduleUpdateRequest,
    ReadingHighlightCreateRequest,
    ReadingHighlightUpdateRequest,
    ReadingProgressUpdate,
    ReadingSaveRequest,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchUpdateRequest,
    ReadingSummarizeRequest,
    ReadingTTSRequest,
    ReadingUpdateRequest,
    ReprocessMediaRequest,
    TLDWAPIClient,
)


class ServerMediaReadingService:
    """Thin wrapper around server-backed media-reading endpoints."""

    _SUPPORTED_METADATA_FIELDS = {"status", "favorite", "tags", "notes", "title"}

    def __init__(self, client: Optional[TLDWAPIClient], *, policy_enforcer: Any | None = None):
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerMediaReadingService":
        from ..runtime_policy.bootstrap import build_runtime_api_client_from_config

        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server media operations.")
        return self.client

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server media action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _reading_action_id(action: str) -> str:
        return f"media.reading.{action}.server"

    @staticmethod
    def _saved_search_action_id(action: str) -> str:
        return f"media.reading.saved_searches.{action}.server"

    @staticmethod
    def _note_link_action_id(action: str) -> str:
        return f"media.reading.note_links.{action}.server"

    @staticmethod
    def _reading_progress_action_id(action: str) -> str:
        return f"media.reading_progress.{action}.server"

    @staticmethod
    def _reading_import_job_action_id(action: str) -> str:
        return f"media.reading_import_jobs.{action}.server"

    @staticmethod
    def _reading_digest_schedule_action_id(action: str) -> str:
        return f"media.reading.digest_schedules.{action}.server"

    @staticmethod
    def _reading_digest_output_action_id(action: str) -> str:
        return f"media.reading.digest_outputs.{action}.server"

    @staticmethod
    def _web_content_ingest_action_id(action: str) -> str:
        return f"media.web_content_ingest.{action}.server"

    @staticmethod
    def _processing_action_id(kind: str, action: str) -> str:
        return f"media.processing.{kind}.{action}.server"

    @staticmethod
    def _transcription_models_action_id(action: str) -> str:
        return f"media.transcription_models.{action}.server"

    @staticmethod
    def _media_item_action_id(action: str) -> str:
        return f"media.items.{action}.server"

    @staticmethod
    def _media_item_subresource_action_id(subresource: str, action: str) -> str:
        return f"media.items.{subresource}.{action}.server"

    @staticmethod
    def _media_add_action_id(action: str) -> str:
        return f"media.add.{action}.server"

    @staticmethod
    def _file_artifact_action_id(action: str) -> str:
        return f"media.file_artifacts.{action}.server"

    @staticmethod
    def _reference_image_action_id(action: str) -> str:
        return f"media.reference_images.{action}.server"

    @staticmethod
    def _navigation_action_id(action: str) -> str:
        return f"media.navigation.{action}.server"

    @staticmethod
    def _ingestion_source_action_id(action: str) -> str:
        return f"media.ingestion_sources.{action}.server"

    @staticmethod
    def _ingestion_job_action_id(action: str) -> str:
        return f"media.ingestion_jobs.{action}.server"

    @staticmethod
    def _ingestion_source_item_action_id(action: str) -> str:
        return f"media.ingestion_source_items.{action}.server"

    async def list_media_items(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> Any:
        self._enforce(self._media_item_action_id("list"))
        return await self._require_client().list_media_items(
            page=page,
            results_per_page=results_per_page,
            include_keywords=include_keywords,
        )

    async def add_media(
        self,
        *,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        self._enforce(self._media_add_action_id("create"))
        payload = {
            "media_type": media_type,
            "urls": urls,
            **options,
        }
        request_data = AddMediaRequest(**{key: value for key, value in payload.items() if value is not None})
        return await self._require_client().add_media(request_data, file_paths=file_paths)

    async def create_file_artifact(
        self,
        *,
        request_data: FileCreateRequest | None = None,
        file_type: str | None = None,
        payload: Mapping[str, Any] | None = None,
        title: str | None = None,
        export: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | FileCreateOptions | None = None,
    ) -> Any:
        self._enforce(self._file_artifact_action_id("create"))
        if request_data is None:
            if not file_type:
                raise ValueError("file_type is required when request_data is not provided.")
            if isinstance(options, FileCreateOptions):
                options_obj = options
            else:
                options_obj = FileCreateOptions(**dict(options or {"persist": True}))
            request_data = FileCreateRequest(
                file_type=file_type,  # type: ignore[arg-type]
                payload=dict(payload or {}),
                title=title,
                export=dict(export) if export is not None else None,  # type: ignore[arg-type]
                options=options_obj,
            )
        return await self._require_client().create_file_artifact(request_data)

    async def list_reference_images(self) -> Any:
        self._enforce(self._reference_image_action_id("list"))
        return await self._require_client().list_reference_images()

    async def get_file_artifact(self, file_id: Any) -> Any:
        self._enforce(self._file_artifact_action_id("detail"))
        return await self._require_client().get_file_artifact(int(file_id))

    async def export_file_artifact(self, file_id: Any, *, format: str) -> Any:
        self._enforce(self._file_artifact_action_id("export"))
        return await self._require_client().export_file_artifact(int(file_id), format=format)

    async def delete_file_artifact(self, file_id: Any, *, hard: bool = False, delete_file: bool = False) -> Any:
        self._enforce(self._file_artifact_action_id("delete"))
        return await self._require_client().delete_file_artifact(int(file_id), hard=hard, delete_file=delete_file)

    async def purge_file_artifacts(
        self,
        request_data: FileArtifactsPurgeRequest | None = None,
        *,
        delete_files: bool = False,
        soft_deleted_grace_days: int = 30,
        include_retention: bool = True,
    ) -> Any:
        self._enforce(self._file_artifact_action_id("purge"))
        payload = request_data or FileArtifactsPurgeRequest(
            delete_files=delete_files,
            soft_deleted_grace_days=soft_deleted_grace_days,
            include_retention=include_retention,
        )
        return await self._require_client().purge_file_artifacts(payload)

    async def list_media_keywords(self, *, query: str | None = None, limit: int = 100) -> Any:
        self._enforce(self._media_item_subresource_action_id("keywords", "list"))
        return await self._require_client().list_media_keywords(query=query, limit=limit)

    async def list_media_trash(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> Any:
        self._enforce(self._media_item_subresource_action_id("trash", "list"))
        return await self._require_client().list_media_trash(
            page=page,
            results_per_page=results_per_page,
            include_keywords=include_keywords,
        )

    async def empty_media_trash(self) -> Any:
        self._enforce(self._media_item_subresource_action_id("trash", "delete"))
        return await self._require_client().empty_media_trash()

    async def get_media_item(
        self,
        media_id: Any,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> Any:
        self._enforce(self._media_item_action_id("detail"))
        return await self._require_client().get_media_item(
            int(media_id),
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )

    async def update_media_item(self, media_id: Any, **fields: Any) -> Any:
        self._enforce(self._media_item_action_id("update"))
        request_data = MediaItemUpdateRequest(**{key: value for key, value in fields.items() if value is not None})
        return await self._require_client().update_media_item(int(media_id), request_data)

    async def delete_media_item(self, media_id: Any) -> Any:
        self._enforce(self._media_item_action_id("delete"))
        return await self._require_client().delete_media_item(int(media_id))

    async def restore_media_item(
        self,
        media_id: Any,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> Any:
        self._enforce(self._media_item_action_id("restore"))
        return await self._require_client().restore_media_item(
            int(media_id),
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )

    async def permanently_delete_media_item(self, media_id: Any) -> Any:
        self._enforce(self._media_item_subresource_action_id("permanent", "delete"))
        return await self._require_client().permanently_delete_media_item(int(media_id))

    async def update_media_keywords(
        self,
        media_id: Any,
        *,
        keywords: list[str],
        mode: str = "add",
    ) -> Any:
        self._enforce(self._media_item_subresource_action_id("keywords", "update"))
        request_data = MediaKeywordsUpdateRequest(keywords=keywords, mode=mode)
        return await self._require_client().update_media_keywords(int(media_id), request_data)

    async def search_media_metadata(self, **kwargs: Any) -> Any:
        self._enforce(self._media_item_subresource_action_id("metadata_search", "list"))
        return await self._require_client().search_media_metadata(
            **{key: value for key, value in kwargs.items() if value is not None}
        )

    async def get_media_by_identifier(self, **kwargs: Any) -> Any:
        self._enforce(self._media_item_subresource_action_id("identifier_lookup", "detail"))
        return await self._require_client().get_media_by_identifier(
            **{key: value for key, value in kwargs.items() if value is not None}
        )

    async def process_mediawiki_dump(self, *, dump_file_path: str, **options: Any):
        self._enforce(self._processing_action_id("mediawiki", "process"))
        request_data = ProcessMediaWikiRequest(**{key: value for key, value in options.items() if value is not None})
        async for item in self._require_client().process_mediawiki_dump(request_data, dump_file_path):
            yield item

    async def ingest_mediawiki_dump(self, *, dump_file_path: str, **options: Any):
        self._enforce(self._processing_action_id("mediawiki", "import"))
        request_data = ProcessMediaWikiRequest(**{key: value for key, value in options.items() if value is not None})
        async for item in self._require_client().ingest_mediawiki_dump(request_data, dump_file_path):
            yield item

    async def download_media_file(self, media_id: Any, *, file_type: str = "original") -> Any:
        self._enforce(self._media_item_subresource_action_id("file", "detail"))
        return await self._require_client().download_media_file(int(media_id), file_type=file_type)

    async def check_media_file(self, media_id: Any, *, file_type: str = "original") -> Any:
        self._enforce(self._media_item_subresource_action_id("file", "detail"))
        return await self._require_client().check_media_file(int(media_id), file_type=file_type)

    async def search_media(
        self,
        *,
        query: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        **filters: Any,
    ) -> Any:
        self._enforce(self._reading_action_id("list"))
        client = self._require_client()
        params = {key: value for key, value in filters.items() if value is not None}
        params.update({"q": query, "limit": limit, "offset": offset})
        return await client.list_reading_items(**{key: value for key, value in params.items() if value is not None})

    async def get_media_detail(self, media_id: Any) -> Any:
        self._enforce(self._reading_action_id("detail"))
        return await self._require_client().get_reading_item(int(media_id))

    async def update_media_metadata(self, media_id: Any, **metadata: Any) -> Any:
        self._enforce(self._reading_action_id("update"))
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
        self._enforce(self._reading_action_id("delete"))
        return await self._require_client().delete_reading_item(int(media_id), hard=False)

    async def undelete_media(self, media_id: Any) -> Any:
        self._enforce(self._reading_action_id("update"))
        raise ValueError("Server media undelete is not available yet.")

    async def save_reading_item(
        self,
        *,
        url: str,
        title: str | None = None,
        tags: list[str] | None = None,
        status: str | None = "saved",
        archive_mode: str = "use_default",
        favorite: bool = False,
        summary: str | None = None,
        notes: str | None = None,
        content: str | None = None,
    ) -> Any:
        self._enforce(self._reading_action_id("create"))
        request_data = ReadingSaveRequest(
            url=url,
            title=title,
            tags=tags or [],
            status=status,
            archive_mode=archive_mode,  # type: ignore[arg-type]
            favorite=favorite,
            summary=summary,
            notes=notes,
            content=content,
        )
        return await self._require_client().save_reading_item(request_data)

    async def create_saved_search(
        self,
        *,
        name: str,
        query: Mapping[str, Any] | None = None,
        sort: str | None = None,
    ) -> Any:
        self._enforce(self._saved_search_action_id("create"))
        request_data = ReadingSavedSearchCreateRequest(
            name=name,
            query=dict(query or {}),
            sort=sort,
        )
        return await self._require_client().create_reading_saved_search(request_data)

    async def list_saved_searches(self, *, limit: int = 50, offset: int = 0) -> Any:
        self._enforce(self._saved_search_action_id("list"))
        return await self._require_client().list_reading_saved_searches(limit=limit, offset=offset)

    async def update_saved_search(
        self,
        search_id: Any,
        *,
        name: str | None = None,
        query: Mapping[str, Any] | None = None,
        sort: str | None = None,
    ) -> Any:
        self._enforce(self._saved_search_action_id("update"))
        request_data = ReadingSavedSearchUpdateRequest(
            name=name,
            query=dict(query) if query is not None else None,
            sort=sort,
        )
        return await self._require_client().update_reading_saved_search(int(search_id), request_data)

    async def delete_saved_search(self, search_id: Any) -> Any:
        self._enforce(self._saved_search_action_id("delete"))
        return await self._require_client().delete_reading_saved_search(int(search_id))

    async def link_note(self, item_id: Any, note_id: str) -> Any:
        self._enforce(self._note_link_action_id("create"))
        return await self._require_client().link_note_to_reading_item(int(item_id), note_id)

    async def list_note_links(self, item_id: Any) -> Any:
        self._enforce(self._note_link_action_id("list"))
        return await self._require_client().list_reading_item_note_links(int(item_id))

    async def unlink_note(self, item_id: Any, note_id: str) -> Any:
        self._enforce(self._note_link_action_id("delete"))
        return await self._require_client().unlink_note_from_reading_item(int(item_id), note_id)

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
        self._enforce(self._reading_action_id("bulk_update"))
        request_data = ItemsBulkRequest(
            item_ids=item_ids,
            action=action,  # type: ignore[arg-type]
            status=status,
            favorite=favorite,
            tags=tags,
            hard=hard,
        )
        return await self._require_client().bulk_update_reading_items(request_data)

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
        self._enforce(self._reading_action_id("archive"))
        request_data = ReadingArchiveCreateRequest(
            format=format,  # type: ignore[arg-type]
            source=source,  # type: ignore[arg-type]
            title=title,
            retention_days=retention_days,
            retention_until=retention_until,
        )
        return await self._require_client().create_reading_archive(int(item_id), request_data)

    async def summarize_reading_item(
        self,
        item_id: Any,
        *,
        provider: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        recursive: bool = False,
        chunked: bool = False,
    ) -> Any:
        self._enforce(self._reading_action_id("summarize"))
        request_data = ReadingSummarizeRequest(
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            recursive=recursive,
            chunked=chunked,
        )
        return await self._require_client().summarize_reading_item(int(item_id), request_data)

    async def tts_reading_item(
        self,
        item_id: Any,
        *,
        model: str,
        voice: str = "af_heart",
        response_format: str = "mp3",
        stream: bool = True,
        speed: float | None = None,
        max_chars: int | None = None,
        text_source: str | None = None,
    ) -> Any:
        self._enforce(self._reading_action_id("tts"))
        request_data = ReadingTTSRequest(
            model=model,
            voice=voice,
            response_format=response_format,  # type: ignore[arg-type]
            stream=stream,
            speed=speed,
            max_chars=max_chars,
            text_source=text_source,  # type: ignore[arg-type]
        )
        return await self._require_client().tts_reading_item(int(item_id), request_data)

    async def import_reading_items(
        self,
        import_path: str,
        *,
        source: str = "auto",
        merge_tags: bool = True,
    ) -> Any:
        self._enforce(self._reading_action_id("import"))
        return await self._require_client().import_reading_items(
            import_path,
            source=source,
            merge_tags=merge_tags,
        )

    async def export_reading_items(
        self,
        *,
        status: list[str] | None = None,
        tags: list[str] | None = None,
        favorite: bool | None = None,
        q: str | None = None,
        domain: str | None = None,
        page: int = 1,
        size: int = 1000,
        include_metadata: bool = True,
        include_clean_html: bool = False,
        include_text: bool = False,
        include_highlights: bool = False,
        include_notes: bool = True,
        format: str = "jsonl",
    ) -> Any:
        self._enforce(self._reading_action_id("export"))
        return await self._require_client().export_reading_items(
            status=status,
            tags=tags,
            favorite=favorite,
            q=q,
            domain=domain,
            page=page,
            size=size,
            include_metadata=include_metadata,
            include_clean_html=include_clean_html,
            include_text=include_text,
            include_highlights=include_highlights,
            include_notes=include_notes,
            format=format,
        )

    async def list_reading_import_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        self._enforce(self._reading_import_job_action_id("list"))
        return await self._require_client().list_reading_import_jobs(status=status, limit=limit, offset=offset)

    async def get_reading_import_job(self, job_id: Any) -> Any:
        self._enforce(self._reading_import_job_action_id("detail"))
        return await self._require_client().get_reading_import_job(int(job_id))

    async def create_reading_digest_schedule(
        self,
        *,
        name: str | None = None,
        cron: str,
        timezone: str | None = None,
        enabled: bool = True,
        require_online: bool = False,
        format: str = "md",
        template_id: int | None = None,
        template_name: str | None = None,
        retention_days: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Any:
        self._enforce(self._reading_digest_schedule_action_id("create"))
        request_data = ReadingDigestScheduleCreateRequest(
            name=name,
            cron=cron,
            timezone=timezone,
            enabled=enabled,
            require_online=require_online,
            format=format,  # type: ignore[arg-type]
            template_id=template_id,
            template_name=template_name,
            retention_days=retention_days,
            filters=dict(filters) if filters is not None else None,
        )
        return await self._require_client().create_reading_digest_schedule(request_data)

    async def list_reading_digest_schedules(self, *, limit: int = 50, offset: int = 0) -> Any:
        self._enforce(self._reading_digest_schedule_action_id("list"))
        return await self._require_client().list_reading_digest_schedules(limit=limit, offset=offset)

    async def get_reading_digest_schedule(self, schedule_id: str) -> Any:
        self._enforce(self._reading_digest_schedule_action_id("detail"))
        return await self._require_client().get_reading_digest_schedule(schedule_id)

    async def update_reading_digest_schedule(
        self,
        schedule_id: str,
        **changes: Any,
    ) -> Any:
        self._enforce(self._reading_digest_schedule_action_id("update"))
        payload = dict(changes)
        if payload.get("filters") is not None:
            payload["filters"] = dict(payload["filters"])
        request_data = ReadingDigestScheduleUpdateRequest(**payload)
        return await self._require_client().update_reading_digest_schedule(schedule_id, request_data)

    async def delete_reading_digest_schedule(self, schedule_id: str) -> Any:
        self._enforce(self._reading_digest_schedule_action_id("delete"))
        return await self._require_client().delete_reading_digest_schedule(schedule_id)

    async def list_reading_digest_outputs(
        self,
        *,
        schedule_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        self._enforce(self._reading_digest_output_action_id("list"))
        return await self._require_client().list_reading_digest_outputs(
            schedule_id=schedule_id,
            limit=limit,
            offset=offset,
        )

    async def ingest_web_content(
        self,
        *,
        urls: list[str],
        titles: list[str] | None = None,
        authors: list[str] | None = None,
        keywords: list[str] | None = None,
        scrape_method: str = "individual",
        url_level: int | None = 2,
        max_pages: int | None = None,
        max_depth: int | None = 3,
        custom_prompt: str | None = None,
        system_prompt: str | None = None,
        perform_translation: bool = False,
        translation_language: str = "en",
        timestamp_option: bool = True,
        overwrite_existing: bool = False,
        perform_analysis: bool = True,
        perform_rolling_summarization: bool = False,
        api_name: str | None = None,
        api_key: str | None = None,
        perform_chunking: bool = True,
        chunk_method: str | None = None,
        use_adaptive_chunking: bool = False,
        use_multi_level_chunking: bool = False,
        chunk_language: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        hierarchical_chunking: bool | None = False,
        hierarchical_template: Mapping[str, Any] | None = None,
        use_cookies: bool = False,
        cookies: str | None = None,
        perform_confabulation_check_of_analysis: bool = False,
        custom_chapter_pattern: str | None = None,
        crawl_strategy: str | None = None,
        include_external: bool | None = None,
        score_threshold: float | None = None,
    ) -> Any:
        self._enforce(self._web_content_ingest_action_id("launch"))
        request_data = IngestWebContentRequest(
            urls=urls,
            titles=titles,
            authors=authors,
            keywords=keywords,
            scrape_method=scrape_method,  # type: ignore[arg-type]
            url_level=url_level,
            max_pages=max_pages,
            max_depth=max_depth,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            perform_translation=perform_translation,
            translation_language=translation_language,
            timestamp_option=timestamp_option,
            overwrite_existing=overwrite_existing,
            perform_analysis=perform_analysis,
            perform_rolling_summarization=perform_rolling_summarization,
            api_name=api_name,
            api_key=api_key,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            chunk_language=chunk_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            hierarchical_chunking=hierarchical_chunking,
            hierarchical_template=dict(hierarchical_template) if hierarchical_template is not None else None,
            use_cookies=use_cookies,
            cookies=cookies,
            perform_confabulation_check_of_analysis=perform_confabulation_check_of_analysis,
            custom_chapter_pattern=custom_chapter_pattern,
            crawl_strategy=crawl_strategy,
            include_external=include_external,
            score_threshold=score_threshold,
        )
        return IngestWebContentResponse.model_validate(
            await self._require_client().ingest_web_content(request_data)
        )

    async def _process_no_db_media(
        self,
        *,
        kind: str,
        client_method_name: str,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> BatchMediaProcessResponse:
        self._enforce(self._processing_action_id(kind, "process"))
        client_method = getattr(self._require_client(), client_method_name)
        return BatchMediaProcessResponse.model_validate(
            await client_method(request_data, file_paths=file_paths)
        )

    async def process_video(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> BatchMediaProcessResponse:
        return await self._process_no_db_media(
            kind="video",
            client_method_name="process_video",
            request_data=ProcessVideoRequest(urls=urls, **options),
            file_paths=file_paths,
        )

    async def process_audio(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> BatchMediaProcessResponse:
        return await self._process_no_db_media(
            kind="audio",
            client_method_name="process_audio",
            request_data=ProcessAudioRequest(urls=urls, **options),
            file_paths=file_paths,
        )

    async def process_pdf(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> BatchMediaProcessResponse:
        return await self._process_no_db_media(
            kind="pdf",
            client_method_name="process_pdf",
            request_data=ProcessPDFRequest(urls=urls, **options),
            file_paths=file_paths,
        )

    async def process_ebook(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> BatchMediaProcessResponse:
        return await self._process_no_db_media(
            kind="ebook",
            client_method_name="process_ebook",
            request_data=ProcessEbookRequest(urls=urls, **options),
            file_paths=file_paths,
        )

    async def process_document(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> BatchMediaProcessResponse:
        return await self._process_no_db_media(
            kind="document",
            client_method_name="process_document",
            request_data=ProcessDocumentRequest(urls=urls, **options),
            file_paths=file_paths,
        )

    async def process_plaintext(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> BatchMediaProcessResponse:
        return await self._process_no_db_media(
            kind="plaintext",
            client_method_name="process_plaintext",
            request_data=ProcessPlaintextRequest(urls=urls, **options),
            file_paths=file_paths,
        )

    async def process_code(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        perform_chunking: bool = True,
        chunk_method: str | None = "code",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> BatchMediaProcessResponse:
        self._enforce(self._processing_action_id("code", "process"))
        request_data = ProcessCodeRequest(
            urls=urls,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,  # type: ignore[arg-type]
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return BatchMediaProcessResponse.model_validate(
            await self._require_client().process_code(request_data, file_paths=file_paths)
        )

    async def process_emails(
        self,
        *,
        file_paths: list[str] | None = None,
        title: str | None = None,
        author: str | None = None,
        keywords: list[str] | None = None,
        custom_prompt: str | None = None,
        system_prompt: str | None = None,
        perform_analysis: bool = True,
        perform_chunking: bool = True,
        chunk_method: str | None = "sentences",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        ingest_attachments: bool = False,
        max_depth: int = 2,
        accept_archives: bool = False,
        accept_mbox: bool = False,
        accept_pst: bool = False,
    ) -> BatchMediaProcessResponse:
        self._enforce(self._processing_action_id("emails", "process"))
        request_data = ProcessEmailsRequest(
            title=title,
            author=author,
            keywords=keywords,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            perform_analysis=perform_analysis,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,  # type: ignore[arg-type]
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ingest_attachments=ingest_attachments,
            max_depth=max_depth,
            accept_archives=accept_archives,
            accept_mbox=accept_mbox,
            accept_pst=accept_pst,
        )
        return BatchMediaProcessResponse.model_validate(
            await self._require_client().process_emails(request_data, file_paths=file_paths)
        )

    async def process_web_scraping(
        self,
        *,
        scrape_method: str,
        url_input: str,
        url_level: int | None = None,
        max_pages: int | None = None,
        max_depth: int = 3,
        summarize_checkbox: bool = False,
        custom_prompt: str | None = None,
        api_name: str | None = None,
        keywords: str | None = "default,no_keyword_set",
        custom_titles: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        custom_cookies: list[dict[str, Any]] | None = None,
        mode: str = "persist",
        user_agent: str | None = None,
        custom_headers: dict[str, str] | None = None,
        crawl_strategy: str | None = None,
        include_external: bool | None = None,
        score_threshold: float | None = None,
    ) -> IngestWebContentResponse:
        self._enforce(self._processing_action_id("web_scraping", "process"))
        request_data = ProcessWebScrapingRequest(
            scrape_method=scrape_method,  # type: ignore[arg-type]
            url_input=url_input,
            url_level=url_level,
            max_pages=max_pages,
            max_depth=max_depth,
            summarize_checkbox=summarize_checkbox,
            custom_prompt=custom_prompt,
            api_name=api_name,
            keywords=keywords,
            custom_titles=custom_titles,
            system_prompt=system_prompt,
            temperature=temperature,
            custom_cookies=custom_cookies,
            mode=mode,
            user_agent=user_agent,
            custom_headers=custom_headers,
            crawl_strategy=crawl_strategy,
            include_external=include_external,
            score_threshold=score_threshold,
        )
        return IngestWebContentResponse.model_validate(
            await self._require_client().process_web_scraping(request_data)
        )

    async def get_transcription_models(self) -> dict[str, Any]:
        self._enforce(self._transcription_models_action_id("list"))
        return await self._require_client().get_transcription_models()

    async def get_reading_progress(self, media_id: Any) -> Any:
        self._enforce(self._reading_progress_action_id("detail"))
        return await self._require_client().get_reading_progress(int(media_id))

    async def update_reading_progress(self, media_id: Any, progress_data: Mapping[str, Any]) -> Any:
        self._enforce(self._reading_progress_action_id("update"))
        payload = dict(progress_data)
        if "percent_complete" in payload and "percentage" not in payload:
            payload["percentage"] = payload.pop("percent_complete")
        request_data = ReadingProgressUpdate(**payload)
        return await self._require_client().update_reading_progress(int(media_id), request_data)

    async def delete_reading_progress(self, media_id: Any) -> Any:
        self._enforce(self._reading_progress_action_id("update"))
        return await self._require_client().delete_reading_progress(int(media_id))

    async def submit_ingest_jobs(
        self,
        *,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        keywords: list[str] | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        perform_chunking: bool = True,
        generate_embeddings: bool = False,
        force_regenerate_embeddings: bool = False,
    ) -> Any:
        self._enforce(self._ingestion_job_action_id("launch"))
        request_data = MediaIngestSubmitRequest(
            media_type=media_type,
            urls=urls,
            keywords=keywords,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_chunking=perform_chunking,
            generate_embeddings=generate_embeddings,
            force_regenerate_embeddings=force_regenerate_embeddings,
        )
        return await self._require_client().submit_media_ingest_jobs(request_data, file_paths=file_paths)

    async def get_ingest_job(self, job_id: Any) -> Any:
        self._enforce(self._ingestion_job_action_id("detail"))
        return await self._require_client().get_media_ingest_job(int(job_id))

    async def list_ingest_jobs(self, batch_id: str, *, limit: int = 100) -> Any:
        self._enforce(self._ingestion_job_action_id("list"))
        return await self._require_client().list_media_ingest_jobs(batch_id, limit=limit)

    def stream_ingest_job_events(self, *, batch_id: str | None = None, after_id: int = 0) -> Any:
        self._enforce(self._ingestion_job_action_id("observe"))
        return self._require_client().stream_media_ingest_job_events(batch_id=batch_id, after_id=after_id)

    async def cancel_ingest_job(self, job_id: Any, *, reason: str | None = None) -> Any:
        self._enforce(self._ingestion_job_action_id("cancel"))
        return await self._require_client().cancel_media_ingest_job(int(job_id), reason=reason)

    async def cancel_ingest_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> Any:
        self._enforce(self._ingestion_job_action_id("cancel"))
        return await self._require_client().cancel_media_ingest_batch(
            batch_id=batch_id,
            session_id=session_id,
            reason=reason,
        )

    async def reprocess_media(
        self,
        media_id: Any,
        *,
        perform_chunking: bool = True,
        generate_embeddings: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        force_regenerate_embeddings: bool = False,
        **options: Any,
    ) -> Any:
        self._enforce(self._ingestion_job_action_id("launch"))
        request_data = ReprocessMediaRequest(
            perform_chunking=perform_chunking,
            generate_embeddings=generate_embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_regenerate_embeddings=force_regenerate_embeddings,
            **options,
        )
        return await self._require_client().reprocess_media(int(media_id), request_data)

    async def create_highlight(
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
        self._enforce(self._reading_action_id("update"))
        normalized_item_id = int(item_id)
        request_data = ReadingHighlightCreateRequest(
            item_id=normalized_item_id,
            quote=quote,
            start_offset=start_offset,
            end_offset=end_offset,
            color=color,
            note=note,
            anchor_strategy=anchor_strategy,  # type: ignore[arg-type]
        )
        return await self._require_client().create_reading_highlight(normalized_item_id, request_data)

    async def list_highlights(self, item_id: Any) -> Any:
        self._enforce(self._reading_action_id("detail"))
        return await self._require_client().list_reading_highlights(int(item_id))

    async def update_highlight(
        self,
        highlight_id: Any,
        *,
        color: str | None = None,
        note: str | None = None,
        state: str | None = None,
    ) -> Any:
        self._enforce(self._reading_action_id("update"))
        request_data = ReadingHighlightUpdateRequest(
            color=color,
            note=note,
            state=state,  # type: ignore[arg-type]
        )
        return await self._require_client().update_reading_highlight(int(highlight_id), request_data)

    async def delete_highlight(self, highlight_id: Any) -> Any:
        self._enforce(self._reading_action_id("delete"))
        return await self._require_client().delete_reading_highlight(int(highlight_id))

    async def list_annotations(self, media_id: Any) -> Any:
        self._enforce(self._reading_action_id("detail"))
        return await self._require_client().list_document_annotations(int(media_id))

    async def create_annotation(
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
        self._enforce(self._reading_action_id("update"))
        request_data = DocumentAnnotationCreate(
            location=location,
            text=text,
            color=color,  # type: ignore[arg-type]
            note=note,
            annotation_type=annotation_type,  # type: ignore[arg-type]
            chapter_title=chapter_title,
            percentage=percentage,
        )
        return await self._require_client().create_document_annotation(int(media_id), request_data)

    async def update_annotation(
        self,
        media_id: Any,
        annotation_id: str,
        *,
        text: str | None = None,
        color: str | None = None,
        note: str | None = None,
    ) -> Any:
        self._enforce(self._reading_action_id("update"))
        request_data = DocumentAnnotationUpdate(
            text=text,
            color=color,  # type: ignore[arg-type]
            note=note,
        )
        return await self._require_client().update_document_annotation(int(media_id), annotation_id, request_data)

    async def delete_annotation(self, media_id: Any, annotation_id: str) -> Any:
        self._enforce(self._reading_action_id("delete"))
        return await self._require_client().delete_document_annotation(int(media_id), annotation_id)

    async def sync_annotations(
        self,
        media_id: Any,
        *,
        annotations: list[Mapping[str, Any]],
        client_ids: list[str] | None = None,
    ) -> Any:
        self._enforce(self._reading_action_id("update"))
        request_data = DocumentAnnotationSyncRequest(
            annotations=[DocumentAnnotationCreate(**dict(annotation)) for annotation in annotations],
            client_ids=client_ids,
        )
        return await self._require_client().sync_document_annotations(int(media_id), request_data)

    async def get_document_outline(self, media_id: Any) -> Any:
        self._enforce(self._reading_action_id("detail"))
        return await self._require_client().get_document_outline(int(media_id))

    async def get_document_figures(self, media_id: Any, *, min_size: int = 50) -> Any:
        self._enforce(self._reading_action_id("detail"))
        return await self._require_client().get_document_figures(int(media_id), min_size=min_size)

    async def get_document_references(
        self,
        media_id: Any,
        *,
        enrich: bool = False,
        reference_index: int | None = None,
        offset: int = 0,
        limit: int = 50,
        parse_cap: int | None = None,
        search: str | None = None,
    ) -> Any:
        self._enforce(self._reading_action_id("detail"))
        return await self._require_client().get_document_references(
            int(media_id),
            enrich=enrich,
            reference_index=reference_index,
            offset=offset,
            limit=limit,
            parse_cap=parse_cap,
            search=search,
        )

    async def get_media_navigation(
        self,
        media_id: Any,
        *,
        include_generated_fallback: bool = False,
        max_depth: int = 4,
        max_nodes: int = 500,
        parent_id: str | None = None,
    ) -> Any:
        self._enforce(self._navigation_action_id("detail"))
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
        format: str = "auto",
        include_alternates: bool = False,
    ) -> Any:
        self._enforce(self._navigation_action_id("detail"))
        return await self._require_client().get_media_navigation_content(
            int(media_id),
            node_id,
            format=format,
            include_alternates=include_alternates,
        )

    async def generate_document_insights(
        self,
        media_id: Any,
        *,
        categories: list[str] | None = None,
        model: str | None = None,
        max_content_length: int | None = 5000,
        force: bool | None = False,
    ) -> Any:
        self._enforce(self._reading_action_id("detail"))
        request_data = DocumentInsightsRequest(
            categories=categories,  # type: ignore[arg-type]
            model=model,
            max_content_length=max_content_length,
            force=force,
        )
        return await self._require_client().generate_document_insights(int(media_id), request_data)

    async def list_ingestion_sources(self) -> Any:
        self._enforce(self._ingestion_source_action_id("list"))
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
        self._enforce(self._ingestion_source_action_id("create"))
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
        self._enforce(self._ingestion_source_action_id("detail"))
        return await self._require_client().get_ingestion_source(int(source_id))

    async def patch_ingestion_source(self, source_id: Any, **changes: Any) -> Any:
        self._enforce(self._ingestion_source_action_id("update"))
        request_data = IngestionSourcePatchRequest(**changes)
        return await self._require_client().patch_ingestion_source(int(source_id), request_data)

    async def delete_ingestion_source(self, source_id: Any) -> Any:
        self._enforce(self._ingestion_source_action_id("delete"))
        raise NotImplementedError("Server ingestion source deletion is not exposed by tldw_server.")

    async def list_ingestion_source_items(self, source_id: Any) -> Any:
        self._enforce(self._ingestion_job_action_id("observe"))
        return await self._require_client().list_ingestion_source_items(int(source_id))

    async def trigger_ingestion_source_sync(self, source_id: Any) -> Any:
        self._enforce(self._ingestion_job_action_id("launch"))
        return await self._require_client().trigger_ingestion_source_sync(int(source_id))

    async def upload_ingestion_source_archive(self, source_id: Any, archive_path: str) -> Any:
        self._enforce(self._ingestion_job_action_id("launch"))
        return await self._require_client().upload_ingestion_source_archive(int(source_id), archive_path)

    async def reattach_ingestion_source_item(self, source_id: Any, item_id: Any) -> Any:
        self._enforce(self._ingestion_source_item_action_id("reattach"))
        return await self._require_client().reattach_ingestion_source_item(int(source_id), int(item_id))

    async def list_document_versions(
        self,
        media_id: Any,
        *,
        include_deleted: bool = False,
        include_content: bool = False,
        limit: int = 10,
        page: int = 1,
    ) -> Any:
        self._enforce(self._reading_action_id("detail"))
        if include_deleted:
            raise ValueError("Server deleted document versions are not available through the current API.")
        return await self._require_client().list_media_versions(
            int(media_id),
            include_content=include_content,
            limit=limit,
            page=page,
        )

    async def get_document_version(
        self,
        media_id: Any,
        version_number: Any,
        *,
        include_content: bool = True,
    ) -> Any:
        self._enforce(self._reading_action_id("detail"))
        return await self._require_client().get_media_version(
            int(media_id),
            int(version_number),
            include_content=include_content,
        )

    async def save_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
        safe_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        self._enforce(self._reading_action_id("update"))
        request_data = MediaVersionCreateRequest(
            content=content,
            prompt=prompt or "",
            analysis_content=analysis_content,
            safe_metadata=dict(safe_metadata) if safe_metadata is not None else None,
        )
        return await self._require_client().create_media_version(int(media_id), request_data)

    async def overwrite_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
        safe_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        return await self.save_analysis_version(
            media_id,
            content=content,
            analysis_content=analysis_content,
            prompt=prompt,
            safe_metadata=safe_metadata,
        )

    async def delete_document_version(self, media_id: Any, version_number: Any) -> Any:
        self._enforce(self._reading_action_id("delete"))
        return await self._require_client().delete_media_version(int(media_id), int(version_number))

    async def delete_analysis_version(self, version_uuid: str) -> Any:
        self._enforce(self._reading_action_id("delete"))
        raise ValueError("Server document version deletion requires media_id and version_number.")
