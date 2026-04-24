"""Scope-aware seam for local and server media-reading flows."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from .media_reading_normalizers import (
    normalize_ingestion_source,
    normalize_ingestion_source_item,
    normalize_local_media_row,
    normalize_media_ingest_batch_cancel,
    normalize_media_ingest_job,
    normalize_media_ingest_job_cancel,
    normalize_media_ingest_job_list,
    normalize_media_ingest_job_submission,
    normalize_media_ingest_job_stream_event,
    normalize_reading_archive,
    normalize_reading_digest_output,
    normalize_reading_digest_schedule,
    normalize_reading_highlight,
    normalize_reading_import_job,
    normalize_reading_items_bulk_update,
    normalize_reading_note_link,
    normalize_reading_progress,
    normalize_reading_saved_search,
    normalize_reading_summary,
    normalize_server_reading_item,
)

ALLOWED_LOCAL_CREATE_SOURCE_TYPES = ("local_directory", "archive_snapshot", "git_repository")
ALLOWED_SERVER_CREATE_SOURCE_TYPES = ("archive_snapshot", "git_repository")


class MediaReadingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


@dataclass(frozen=True)
class ReadItLaterContextCapability:
    available: bool
    aggregate_only: bool
    reason: str | None = None


class MediaReadingScopeService:
    """Route media actions to the active local/server backend and normalize outputs."""

    def __init__(self, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: MediaReadingBackend | str | None) -> MediaReadingBackend:
        if mode is None:
            return MediaReadingBackend.LOCAL
        if isinstance(mode, MediaReadingBackend):
            return mode
        try:
            return MediaReadingBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid media backend: {mode}") from exc

    def get_read_it_later_context_capability(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_type_slug: str | None = None,
    ) -> ReadItLaterContextCapability:
        normalized_mode = self._normalize_mode(mode)
        normalized_type = str(media_type_slug or "all-media").strip().lower() or "all-media"

        if normalized_mode == MediaReadingBackend.LOCAL:
            return ReadItLaterContextCapability(available=True, aggregate_only=False, reason=None)

        if normalized_type == "all-media":
            return ReadItLaterContextCapability(available=True, aggregate_only=True, reason=None)

        return ReadItLaterContextCapability(
            available=False,
            aggregate_only=True,
            reason="Read-it-later is only available in server mode from All Media.",
        )

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _as_mapping_payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return dict(model_dump(mode="json"))
        return {}

    @staticmethod
    def _reading_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.{action}.{mode.value}"

    @staticmethod
    def _reading_progress_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_progress.{action}.{mode.value}"

    @staticmethod
    def _reading_highlight_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_highlights.{action}.{mode.value}"

    @staticmethod
    def _reading_list_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"collections.reading_list.{action}.{mode.value}"

    @staticmethod
    def _reading_saved_search_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_saved_searches.{action}.{mode.value}"

    @staticmethod
    def _reading_note_link_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_note_links.{action}.{mode.value}"

    @staticmethod
    def _reading_import_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_import.{action}.{mode.value}"

    @staticmethod
    def _reading_archive_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_archives.{action}.{mode.value}"

    @staticmethod
    def _reading_export_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_export.{action}.{mode.value}"

    @staticmethod
    def _reading_summary_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_summaries.{action}.{mode.value}"

    @staticmethod
    def _reading_tts_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_tts.{action}.{mode.value}"

    @staticmethod
    def _reading_digest_schedule_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_digest_schedules.{action}.{mode.value}"

    @staticmethod
    def _reading_digest_output_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_digest_outputs.{action}.{mode.value}"

    @staticmethod
    def _ingestion_source_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.ingestion_sources.{action}.{mode.value}"

    @staticmethod
    def _ingestion_job_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.ingestion_jobs.{action}.{mode.value}"

    @staticmethod
    def _web_content_ingest_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.web_content_ingest.{action}.{mode.value}"

    @staticmethod
    def _document_outline_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.document_outline.{action}.{mode.value}"

    @staticmethod
    def _document_figures_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.document_figures.{action}.{mode.value}"

    @staticmethod
    def _document_insights_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.document_insights.{action}.{mode.value}"

    @staticmethod
    def _document_references_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.document_references.{action}.{mode.value}"

    @staticmethod
    def _document_annotation_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.document_annotations.{action}.{mode.value}"

    @staticmethod
    def _document_navigation_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.document_navigation.{action}.{mode.value}"

    @staticmethod
    def _document_navigation_content_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.document_navigation_content.{action}.{mode.value}"

    @staticmethod
    def _media_item_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.items.{action}.{mode.value}"

    @staticmethod
    def _unified_item_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.unified_items.{action}.{mode.value}"

    @staticmethod
    def _media_item_keywords_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.items.keywords.{action}.{mode.value}"

    @staticmethod
    def _media_item_trash_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.items.trash.{action}.{mode.value}"

    @staticmethod
    def _media_item_metadata_search_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.items.metadata_search.{action}.{mode.value}"

    @staticmethod
    def _media_item_identifier_lookup_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.items.identifier_lookup.{action}.{mode.value}"

    @staticmethod
    def _media_item_reprocess_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.items.reprocess.{action}.{mode.value}"

    @staticmethod
    def _media_processing_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.processing.{action}.{mode.value}"

    @staticmethod
    def _media_processing_models_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.processing_models.{action}.{mode.value}"

    @staticmethod
    def _media_item_file_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.items.file.{action}.{mode.value}"

    @staticmethod
    def _require_server_document_workspace(mode: MediaReadingBackend) -> None:
        if mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local document workspace is not available yet.")

    @staticmethod
    def _require_server_media_item_lifecycle(mode: MediaReadingBackend) -> None:
        if mode == MediaReadingBackend.LOCAL:
            raise ValueError("Server media item lifecycle requires server mode.")

    @staticmethod
    def _require_server_unified_items(mode: MediaReadingBackend) -> None:
        if mode == MediaReadingBackend.LOCAL:
            raise ValueError("Server unified items require server mode.")

    @staticmethod
    def _require_server_media_processing(mode: MediaReadingBackend) -> None:
        if mode == MediaReadingBackend.LOCAL:
            raise ValueError("Server media processing requires server mode.")

    def _service_for_mode(self, mode: MediaReadingBackend) -> Any:
        if mode == MediaReadingBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local media backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server media backend is unavailable.")
        return self.server_service

    @staticmethod
    def _raise_local_ingestion_jobs_unavailable() -> None:
        raise ValueError("Local media ingest jobs are not available yet.")

    @staticmethod
    def _raise_local_web_content_ingest_unavailable() -> None:
        raise ValueError("Local web-content ingest is not available yet.")

    @staticmethod
    def _raise_local_mediawiki_dump_unavailable() -> None:
        raise ValueError("Local MediaWiki dump processing is not available yet.")

    @staticmethod
    def _raise_local_reading_import_unavailable() -> None:
        raise ValueError("Local reading import jobs are not available yet.")

    @staticmethod
    def _raise_local_reading_archives_unavailable() -> None:
        raise ValueError("Local reading archives are not available yet.")

    @staticmethod
    def _raise_local_reading_summaries_unavailable() -> None:
        raise ValueError("Local reading summaries are not available yet.")

    @staticmethod
    def _raise_local_reading_tts_unavailable() -> None:
        raise ValueError("Local reading TTS is not available yet.")

    @staticmethod
    def _raise_local_reading_bulk_update_unavailable() -> None:
        raise ValueError("Bulk reading item mutation requires server mode.")

    @staticmethod
    def _raise_local_reading_digest_schedules_unavailable() -> None:
        raise ValueError("Local reading digest schedules are not available yet.")

    @staticmethod
    def _raise_local_reading_digest_outputs_unavailable() -> None:
        raise ValueError("Local reading digest outputs are not available yet.")

    @staticmethod
    def _validate_server_create_source_type(source_type: str) -> str:
        normalized_source_type = str(source_type or "").strip()
        if normalized_source_type not in ALLOWED_SERVER_CREATE_SOURCE_TYPES:
            allowed_types = ", ".join(ALLOWED_SERVER_CREATE_SOURCE_TYPES)
            raise ValueError(
                f"Unsupported server ingestion source type: {normalized_source_type}. "
                f"Allowed types: {allowed_types}."
            )
        return normalized_source_type

    @staticmethod
    def _validate_local_create_source_type(source_type: str) -> str:
        normalized_source_type = str(source_type or "").strip()
        if normalized_source_type not in ALLOWED_LOCAL_CREATE_SOURCE_TYPES:
            allowed_types = ", ".join(ALLOWED_LOCAL_CREATE_SOURCE_TYPES)
            raise ValueError(
                f"Unsupported local ingestion source type: {normalized_source_type}. "
                f"Allowed types: {allowed_types}."
            )
        return normalized_source_type

    def _normalize_media_record(
        self,
        mode: MediaReadingBackend,
        record: Mapping[str, Any],
        *,
        reading_progress: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if mode == MediaReadingBackend.LOCAL:
            return normalize_local_media_row(record, reading_progress=reading_progress)
        return normalize_server_reading_item(record, reading_progress=reading_progress)

    def _resolve_backing_media_id(
        self,
        *,
        record: Optional[Mapping[str, Any]] = None,
        media_id: Any = None,
    ) -> Any:
        if isinstance(record, Mapping):
            backing_media_id = record.get("backing_media_id")
            if backing_media_id not in (None, ""):
                return backing_media_id
            raise ValueError("record['backing_media_id'] is required for reading progress operations.")
        if media_id not in (None, ""):
            return media_id
        raise ValueError("A media record or media_id is required for reading progress operations.")

    def _resolve_highlight_item_id(
        self,
        *,
        mode: MediaReadingBackend,
        record: Optional[Mapping[str, Any]] = None,
        item_id: Any = None,
        media_id: Any = None,
    ) -> Any:
        if item_id not in (None, ""):
            return item_id
        if media_id not in (None, ""):
            return media_id
        if isinstance(record, Mapping):
            if mode == MediaReadingBackend.SERVER:
                source_id = record.get("source_id")
                if source_id not in (None, ""):
                    return source_id
                raise ValueError("record['source_id'] is required for server reading highlight operations.")
            backing_media_id = record.get("backing_media_id")
            if backing_media_id not in (None, ""):
                return backing_media_id
            raise ValueError("record['backing_media_id'] is required for local reading highlight operations.")
        raise ValueError("A media record, item_id, or media_id is required for reading highlight operations.")

    async def search_media(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        query: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        **filters: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.search_media(query=query, limit=limit, offset=offset, **filters)
        )
        raw_items = list(payload.get("items", [])) if isinstance(payload, Mapping) else list(payload or [])
        items = [self._normalize_media_record(normalized_mode, item) for item in raw_items]
        return {
            "items": items,
            "total": payload.get("total", len(items)) if isinstance(payload, Mapping) else len(items),
            "offset": payload.get("offset", offset) if isinstance(payload, Mapping) else offset,
            "limit": payload.get("limit", limit) if isinstance(payload, Mapping) else limit,
        }

    async def get_media_detail(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        detail = await self._maybe_await(service.get_media_detail(media_id))
        normalized = self._normalize_media_record(normalized_mode, detail)

        backing_media_id = normalized.get("backing_media_id")
        if backing_media_id not in (None, ""):
            progress = await self._maybe_await(service.get_reading_progress(backing_media_id))
            normalized["reading_progress"] = normalize_reading_progress(
                progress,
                backend=normalized_mode.value,
                backing_media_id=backing_media_id,
            )
        return normalized

    async def list_backing_media_keywords(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        query: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_keywords_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.list_media_keywords(query=query, limit=limit))
        return self._as_mapping_payload(payload)

    async def list_backing_media_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.list_backing_media_items(
                page=page,
                results_per_page=results_per_page,
                include_keywords=include_keywords,
            )
        )
        return self._as_mapping_payload(payload)

    async def search_backing_media_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        page: int = 1,
        results_per_page: int = 10,
        **filters: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.search_backing_media_items(
                page=page,
                results_per_page=results_per_page,
                **filters,
            )
        )
        return self._as_mapping_payload(payload)

    async def list_backing_media_trash(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_trash_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.list_media_trash(
                page=page,
                results_per_page=results_per_page,
                include_keywords=include_keywords,
            )
        )
        return self._as_mapping_payload(payload)

    async def empty_backing_media_trash(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_trash_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.empty_media_trash())
        return self._as_mapping_payload(payload)

    async def search_backing_media_metadata(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        **filters: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_metadata_search_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.search_media_metadata(**filters))
        return self._as_mapping_payload(payload)

    async def get_backing_media_by_identifier(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        **identifiers: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_identifier_lookup_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.get_media_by_identifier(**identifiers))
        return self._as_mapping_payload(payload)

    async def get_media_transcription_models(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_processing(normalized_mode)
        self._enforce_policy(self._media_processing_models_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.get_media_transcription_models())
        return self._as_mapping_payload(payload)

    async def reprocess_backing_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_processing(normalized_mode)
        self._enforce_policy(self._media_item_reprocess_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.reprocess_media(media_id, **options))
        return self._as_mapping_payload(payload)

    async def add_media(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.add_media(request_data, file_paths=file_paths))
        return self._as_mapping_payload(payload)

    async def list_unified_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        **filters: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._unified_item_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.list_unified_items(**{key: value for key, value in filters.items() if value is not None})
        )
        return self._as_mapping_payload(payload)

    async def get_unified_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._unified_item_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.get_unified_item(item_id))
        return self._as_mapping_payload(payload)

    async def bulk_update_unified_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_unified_items(normalized_mode)
        action = self._as_mapping_payload(request_data).get("action")
        policy_action = "delete" if action == "delete" else "update"
        self._enforce_policy(self._unified_item_action_id(normalized_mode, policy_action))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.bulk_update_unified_items(request_data))
        return self._as_mapping_payload(payload)

    async def _process_server_media(
        self,
        *,
        mode: MediaReadingBackend | str | None,
        method_name: str,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_processing(normalized_mode)
        self._enforce_policy(self._media_processing_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, method_name)
        payload = await self._maybe_await(method(request_data, file_paths=file_paths))
        return self._as_mapping_payload(payload)

    async def process_media_video(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._process_server_media(
            mode=mode,
            method_name="process_video",
            request_data=request_data,
            file_paths=file_paths,
        )

    async def process_media_audio(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._process_server_media(
            mode=mode,
            method_name="process_audio",
            request_data=request_data,
            file_paths=file_paths,
        )

    async def process_media_pdf(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._process_server_media(
            mode=mode,
            method_name="process_pdf",
            request_data=request_data,
            file_paths=file_paths,
        )

    async def process_media_ebook(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._process_server_media(
            mode=mode,
            method_name="process_ebook",
            request_data=request_data,
            file_paths=file_paths,
        )

    async def process_media_document(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._process_server_media(
            mode=mode,
            method_name="process_document",
            request_data=request_data,
            file_paths=file_paths,
        )

    async def process_media_code(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._process_server_media(
            mode=mode,
            method_name="process_code",
            request_data=request_data,
            file_paths=file_paths,
        )

    async def process_media_email(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._process_server_media(
            mode=mode,
            method_name="process_email",
            request_data=request_data,
            file_paths=file_paths,
        )

    async def process_mediawiki_dump(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        dump_file_path: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_mediawiki_dump_unavailable()
        self._enforce_policy(self._media_processing_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        async for page in service.process_mediawiki_dump(request_data, dump_file_path):
            yield page

    async def ingest_mediawiki_dump(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
        dump_file_path: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_mediawiki_dump_unavailable()
        self._enforce_policy(self._media_processing_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        async for event in service.ingest_mediawiki_dump(request_data, dump_file_path):
            yield event

    async def get_backing_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.get_media_item(
                media_id,
                include_content=include_content,
                include_versions=include_versions,
                include_version_content=include_version_content,
            )
        )
        return self._as_mapping_payload(payload)

    async def update_backing_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        **changes: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.update_media_item(media_id, **changes))
        return self._as_mapping_payload(payload)

    async def trash_backing_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.trash_media_item(media_id))

    async def restore_backing_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "restore"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.restore_media_item(
                media_id,
                include_content=include_content,
                include_versions=include_versions,
                include_version_content=include_version_content,
            )
        )
        return self._as_mapping_payload(payload)

    async def permanently_delete_backing_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "permanent_delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.permanently_delete_media_item(media_id))

    async def update_backing_media_keywords(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        keywords: list[str],
        update_mode: str = "add",
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_keywords_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.update_media_keywords(media_id, keywords=keywords, mode=update_mode)
        )
        return self._as_mapping_payload(payload)

    async def download_backing_media_file(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        file_type: str = "original",
    ) -> bytes:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_media_item_lifecycle(normalized_mode)
        self._enforce_policy(self._media_item_file_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.download_media_file(media_id, file_type=file_type))

    async def get_document_navigation(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        include_generated_fallback: bool = False,
        max_depth: int = 4,
        max_nodes: int = 500,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_navigation_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.get_media_navigation(
                media_id,
                include_generated_fallback=include_generated_fallback,
                max_depth=max_depth,
                max_nodes=max_nodes,
                parent_id=parent_id,
            )
        )
        return self._as_mapping_payload(payload)

    async def get_document_navigation_content(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        node_id: str,
        content_format: str = "auto",
        include_alternates: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_navigation_content_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.get_media_navigation_content(
                media_id,
                node_id,
                content_format=content_format,
                include_alternates=include_alternates,
            )
        )
        return self._as_mapping_payload(payload)

    async def list_read_it_later(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        query: str | None = None,
        limit: int = 20,
        offset: int = 0,
        **filters: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_list_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.search_media(query=query, limit=limit, offset=offset, read_it_later_only=True, **filters)
            if normalized_mode == MediaReadingBackend.LOCAL
            else service.search_media(query=query, limit=limit, offset=offset, status=["saved"], **filters)
        )
        raw_items = list(payload.get("items", []))
        items = [self._normalize_media_record(normalized_mode, item) for item in raw_items]
        return {"items": items, "total": payload.get("total", len(items)), "offset": offset, "limit": limit}

    async def save_to_read_it_later(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_list_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            return await self._maybe_await(service.save_to_read_it_later(media_id))
        return await self._maybe_await(service.update_media_metadata(media_id, status="saved"))

    async def save_reading_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_list_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.save_reading_item(request_data))
        return self._as_mapping_payload(payload)

    async def remove_from_read_it_later(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_list_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            return await self._maybe_await(service.remove_from_read_it_later(media_id))
        return await self._maybe_await(service.update_media_metadata(media_id, status="archived"))

    async def update_media_metadata(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        **metadata: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.update_media_metadata(media_id, **metadata))

    async def delete_media(self, *, mode: MediaReadingBackend | str | None = None, media_id: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_media(media_id))

    async def bulk_update_reading_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_ids: list[int],
        action: str,
        status: str | None = None,
        favorite: bool | None = None,
        tags: list[str] | None = None,
        hard: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_bulk_update_unavailable()
        action_kind = "delete" if action == "delete" else "update"
        self._enforce_policy(self._reading_action_id(normalized_mode, action_kind))
        service = self._service_for_mode(normalized_mode)
        options = {
            key: value
            for key, value in {
                "status": status,
                "favorite": favorite,
                "tags": tags,
            }.items()
            if value is not None
        }
        payload = await self._maybe_await(
            service.bulk_update_reading_items(
                item_ids=item_ids,
                action=action,
                hard=hard,
                **options,
            )
        )
        return normalize_reading_items_bulk_update(payload, backend=normalized_mode.value)

    async def undelete_media(self, *, mode: MediaReadingBackend | str | None = None, media_id: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.undelete_media(media_id))

    async def get_reading_progress(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        record: Optional[Mapping[str, Any]] = None,
        media_id: Any = None,
    ) -> Optional[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_progress_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        backing_media_id = self._resolve_backing_media_id(record=record, media_id=media_id)
        progress = await self._maybe_await(service.get_reading_progress(backing_media_id))
        return normalize_reading_progress(
            progress,
            backend=normalized_mode.value,
            backing_media_id=backing_media_id,
        )

    async def update_reading_progress(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        record: Optional[Mapping[str, Any]] = None,
        media_id: Any = None,
        progress_data: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_progress_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        backing_media_id = self._resolve_backing_media_id(record=record, media_id=media_id)
        progress = await self._maybe_await(service.update_reading_progress(backing_media_id, progress_data))
        return normalize_reading_progress(
            progress,
            backend=normalized_mode.value,
            backing_media_id=backing_media_id,
        )

    async def delete_reading_progress(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        record: Optional[Mapping[str, Any]] = None,
        media_id: Any = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_progress_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        backing_media_id = self._resolve_backing_media_id(record=record, media_id=media_id)
        return await self._maybe_await(service.delete_reading_progress(backing_media_id))

    async def create_reading_highlight(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        record: Optional[Mapping[str, Any]] = None,
        item_id: Any = None,
        media_id: Any = None,
        quote: str,
        start_offset: int | None = None,
        end_offset: int | None = None,
        color: str | None = None,
        note: str | None = None,
        anchor_strategy: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_highlight_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        resolved_item_id = self._resolve_highlight_item_id(
            mode=normalized_mode,
            record=record,
            item_id=item_id,
            media_id=media_id,
        )
        payload = {
            key: value
            for key, value in {
                "quote": quote,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "color": color,
                "note": note,
                "anchor_strategy": anchor_strategy,
            }.items()
            if value is not None
        }
        highlight = await self._maybe_await(service.create_reading_highlight(resolved_item_id, **payload))
        return normalize_reading_highlight(highlight, backend=normalized_mode.value)

    async def list_reading_highlights(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        record: Optional[Mapping[str, Any]] = None,
        item_id: Any = None,
        media_id: Any = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_highlight_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        resolved_item_id = self._resolve_highlight_item_id(
            mode=normalized_mode,
            record=record,
            item_id=item_id,
            media_id=media_id,
        )
        highlights = await self._maybe_await(service.list_reading_highlights(resolved_item_id))
        return [
            normalize_reading_highlight(highlight, backend=normalized_mode.value)
            for highlight in list(highlights or [])
        ]

    async def update_reading_highlight(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        highlight_id: Any,
        **changes: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_highlight_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        highlight = await self._maybe_await(
            service.update_reading_highlight(
                highlight_id,
                **{key: value for key, value in changes.items() if value is not None},
            )
        )
        return normalize_reading_highlight(highlight, backend=normalized_mode.value)

    async def delete_reading_highlight(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        highlight_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_highlight_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_reading_highlight(highlight_id))

    async def create_reading_saved_search(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        name: str,
        query: Mapping[str, Any] | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_saved_search_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        saved_search = await self._maybe_await(
            service.create_reading_saved_search(name=name, query=query, sort=sort)
        )
        return normalize_reading_saved_search(saved_search, backend=normalized_mode.value)

    async def list_reading_saved_searches(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_saved_search_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.list_reading_saved_searches(limit=limit, offset=offset))
        payload_map = self._as_mapping_payload(payload)
        raw_items = list(payload_map.get("items", []))
        return {
            "items": [
                normalize_reading_saved_search(item, backend=normalized_mode.value)
                for item in raw_items
            ],
            "total": payload_map.get("total", len(raw_items)),
            "limit": payload_map.get("limit", limit),
            "offset": payload_map.get("offset", offset),
        }

    async def update_reading_saved_search(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        search_id: Any,
        **changes: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_saved_search_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        saved_search = await self._maybe_await(
            service.update_reading_saved_search(
                search_id,
                **{key: value for key, value in changes.items() if value is not None},
            )
        )
        return normalize_reading_saved_search(saved_search, backend=normalized_mode.value)

    async def delete_reading_saved_search(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        search_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_saved_search_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_reading_saved_search(search_id))

    async def link_reading_item_note(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        note_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_note_link_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        link = await self._maybe_await(service.link_reading_item_note(item_id, note_id=note_id))
        return normalize_reading_note_link(link, backend=normalized_mode.value)

    async def list_reading_item_note_links(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_note_link_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.list_reading_item_note_links(item_id))
        payload_map = self._as_mapping_payload(payload)
        links = [
            normalize_reading_note_link(link, backend=normalized_mode.value)
            for link in list(payload_map.get("links", []))
        ]
        return {"item_id": str(payload_map.get("item_id", item_id)), "links": links}

    async def unlink_reading_item_note(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        note_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_note_link_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.unlink_reading_item_note(item_id, note_id))

    async def import_reading_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        file_path: str,
        source: str = "auto",
        merge_tags: bool = True,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_import_unavailable()
        self._enforce_policy(self._reading_import_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        job = await self._maybe_await(
            service.import_reading_items(file_path, source=source, merge_tags=merge_tags)
        )
        return normalize_reading_import_job(job, backend=normalized_mode.value)

    async def list_reading_import_jobs(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_import_unavailable()
        self._enforce_policy(self._reading_import_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.list_reading_import_jobs(status=status, limit=limit, offset=offset)
        )
        payload_map = self._as_mapping_payload(payload)
        raw_jobs = list(payload_map.get("jobs", []))
        return {
            "jobs": [
                normalize_reading_import_job(job, backend=normalized_mode.value)
                for job in raw_jobs
            ],
            "total": payload_map.get("total", len(raw_jobs)),
            "limit": payload_map.get("limit", limit),
            "offset": payload_map.get("offset", offset),
        }

    async def get_reading_import_job(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        job_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_import_unavailable()
        self._enforce_policy(self._reading_import_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        job = await self._maybe_await(service.get_reading_import_job(job_id))
        return normalize_reading_import_job(job, backend=normalized_mode.value)

    async def create_reading_archive(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        format: str = "html",
        source: str = "auto",
        title: str | None = None,
        retention_days: int | None = None,
        retention_until: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_archives_unavailable()
        self._enforce_policy(self._reading_archive_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        archive = await self._maybe_await(
            service.create_reading_archive(
                item_id,
                format=format,
                source=source,
                title=title,
                retention_days=retention_days,
                retention_until=retention_until,
            )
        )
        return normalize_reading_archive(archive, backend=normalized_mode.value)

    async def export_reading_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        **filters: Any,
    ) -> bytes:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_export_action_id(normalized_mode, "export"))
        service = self._service_for_mode(normalized_mode)
        payload = {key: value for key, value in filters.items() if value is not None}
        return await self._maybe_await(service.export_reading_items(**payload))

    async def summarize_reading_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_summaries_unavailable()
        self._enforce_policy(self._reading_summary_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        payload = {key: value for key, value in options.items() if value is not None}
        summary = await self._maybe_await(service.summarize_reading_item(item_id, **payload))
        return normalize_reading_summary(summary, backend=normalized_mode.value)

    async def tts_reading_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        **options: Any,
    ) -> bytes:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_tts_unavailable()
        self._enforce_policy(self._reading_tts_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        payload = {key: value for key, value in options.items() if value is not None}
        return await self._maybe_await(service.tts_reading_item(item_id, **payload))

    async def create_reading_digest_schedule(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        cron: str,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_digest_schedules_unavailable()
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        payload = {key: value for key, value in options.items() if value is not None}
        return await self._maybe_await(service.create_reading_digest_schedule(cron=cron, **payload))

    async def list_reading_digest_schedules(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_digest_schedules_unavailable()
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        schedules = await self._maybe_await(service.list_reading_digest_schedules(limit=limit, offset=offset))
        return [
            normalize_reading_digest_schedule(schedule, backend=normalized_mode.value)
            for schedule in list(schedules or [])
        ]

    async def get_reading_digest_schedule(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        schedule_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_digest_schedules_unavailable()
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        schedule = await self._maybe_await(service.get_reading_digest_schedule(schedule_id))
        return normalize_reading_digest_schedule(schedule, backend=normalized_mode.value)

    async def update_reading_digest_schedule(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        schedule_id: str,
        **changes: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_digest_schedules_unavailable()
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        payload = {key: value for key, value in changes.items() if value is not None}
        schedule = await self._maybe_await(service.update_reading_digest_schedule(schedule_id, **payload))
        return normalize_reading_digest_schedule(schedule, backend=normalized_mode.value)

    async def delete_reading_digest_schedule(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        schedule_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_digest_schedules_unavailable()
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_reading_digest_schedule(schedule_id))

    async def list_reading_digest_outputs(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        schedule_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_reading_digest_outputs_unavailable()
        self._enforce_policy(self._reading_digest_output_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.list_reading_digest_outputs(schedule_id=schedule_id, limit=limit, offset=offset)
        )
        payload_map = self._as_mapping_payload(payload)
        return {
            "items": [
                normalize_reading_digest_output(output, backend=normalized_mode.value)
                for output in list(payload_map.get("items") or [])
            ],
            "total": payload_map.get("total", 0),
            "limit": payload_map.get("limit", limit),
            "offset": payload_map.get("offset", offset),
        }

    async def list_ingestion_sources(self, *, mode: MediaReadingBackend | str | None = None) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_source_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        sources = await self._maybe_await(service.list_ingestion_sources())
        return [normalize_ingestion_source(source, backend=normalized_mode.value) for source in list(sources or [])]

    async def create_ingestion_source(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_type: str,
        sink_type: str,
        policy: str = "canonical",
        enabled: bool = True,
        schedule_enabled: bool = False,
        schedule: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            normalized_source_type = self._validate_local_create_source_type(source_type)
        else:
            normalized_source_type = self._validate_server_create_source_type(source_type)
        self._enforce_policy(self._ingestion_source_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        source = await self._maybe_await(
            service.create_ingestion_source(
                source_type=normalized_source_type,
                sink_type=sink_type,
                policy=policy,
                enabled=enabled,
                schedule_enabled=schedule_enabled,
                schedule=schedule,
                config=config,
            )
        )
        return normalize_ingestion_source(source, backend=normalized_mode.value)

    async def get_ingestion_source(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_source_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        source = await self._maybe_await(service.get_ingestion_source(source_id))
        return normalize_ingestion_source(source, backend=normalized_mode.value)

    async def patch_ingestion_source(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_id: Any,
        **changes: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_source_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        source = await self._maybe_await(service.patch_ingestion_source(source_id, **changes))
        return normalize_ingestion_source(source, backend=normalized_mode.value)

    async def delete_ingestion_source(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_source_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_ingestion_source(source_id))

    async def list_ingestion_source_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_id: Any,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        items = await self._maybe_await(service.list_ingestion_source_items(source_id))
        return [
            normalize_ingestion_source_item(item, backend=normalized_mode.value)
            for item in list(items or [])
        ]

    async def reattach_ingestion_source_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_id: Any,
        item_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local ingestion sources are not available yet.")
        self._enforce_policy(self._ingestion_source_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        item = await self._maybe_await(service.reattach_ingestion_source_item(source_id, item_id))
        return normalize_ingestion_source_item(item, backend=normalized_mode.value)

    async def trigger_ingestion_source_sync(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.trigger_ingestion_source_sync(source_id))

    async def upload_ingestion_source_archive(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_id: Any,
        archive_path: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.upload_ingestion_source_archive(source_id, archive_path))

    async def submit_media_ingest_jobs(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_ingestion_jobs_unavailable()
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.submit_media_ingest_jobs(
                media_type=media_type,
                urls=urls,
                file_paths=file_paths,
                **options,
            )
        )
        return normalize_media_ingest_job_submission(payload, backend=normalized_mode.value)

    async def get_media_ingest_job(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        job_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_ingestion_jobs_unavailable()
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        job = await self._maybe_await(service.get_media_ingest_job(job_id))
        return normalize_media_ingest_job(job, backend=normalized_mode.value)

    async def list_media_ingest_jobs(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        batch_id: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_ingestion_jobs_unavailable()
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.list_media_ingest_jobs(batch_id=batch_id, limit=limit))
        return normalize_media_ingest_job_list(payload, backend=normalized_mode.value)

    async def stream_media_ingest_job_events(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        batch_id: str | None = None,
        after_id: int = 0,
    ):
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_ingestion_jobs_unavailable()
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        stream = service.stream_media_ingest_job_events(batch_id=batch_id, after_id=after_id)
        async for event in stream:
            yield normalize_media_ingest_job_stream_event(event, backend=normalized_mode.value)

    async def cancel_media_ingest_job(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        job_id: Any,
        reason: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_ingestion_jobs_unavailable()
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(service.cancel_media_ingest_job(job_id, reason=reason))
        return normalize_media_ingest_job_cancel(payload, backend=normalized_mode.value)

    async def cancel_media_ingest_jobs_batch(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_ingestion_jobs_unavailable()
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.cancel_media_ingest_jobs_batch(
                batch_id=batch_id,
                session_id=session_id,
                reason=reason,
            )
        )
        return normalize_media_ingest_batch_cancel(payload)

    async def ingest_web_content(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        urls: list[str],
        **options: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_web_content_ingest_unavailable()
        self._enforce_policy(self._web_content_ingest_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.ingest_web_content(
                urls=urls,
                **{key: value for key, value in options.items() if value is not None},
            )
        )

    async def process_web_scraping(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            self._raise_local_web_content_ingest_unavailable()
        self._enforce_policy(self._web_content_ingest_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.process_web_scraping(request_data))

    async def get_document_outline(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_outline_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.get_document_outline(media_id))

    async def get_document_figures(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        min_size: int = 50,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_figures_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.get_document_figures(media_id, min_size=min_size))

    async def list_document_annotations(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_annotation_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.list_document_annotations(media_id))

    async def create_document_annotation(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        location: str,
        text: str,
        color: str = "yellow",
        note: str | None = None,
        annotation_type: str = "highlight",
        chapter_title: str | None = None,
        percentage: float | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_annotation_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.create_document_annotation(
                media_id,
                location=location,
                text=text,
                color=color,
                note=note,
                annotation_type=annotation_type,
                chapter_title=chapter_title,
                percentage=percentage,
            )
        )

    async def update_document_annotation(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        annotation_id: str,
        **changes: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_annotation_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.update_document_annotation(
                media_id,
                annotation_id,
                **{key: value for key, value in changes.items() if value is not None},
            )
        )

    async def delete_document_annotation(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        annotation_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_annotation_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_document_annotation(media_id, annotation_id))

    async def sync_document_annotations(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        annotations: list[Mapping[str, Any]],
        client_ids: list[str] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_annotation_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.sync_document_annotations(
                media_id,
                annotations=annotations,
                client_ids=client_ids,
            )
        )

    async def generate_document_insights(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        categories: list[str] | None = None,
        model: str | None = None,
        max_content_length: int | None = 5000,
        force: bool | None = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_insights_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.generate_document_insights(
                media_id,
                categories=categories,
                model=model,
                max_content_length=max_content_length,
                force=force,
            )
        )

    async def get_document_references(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        enrich: bool = False,
        reference_index: int | None = None,
        offset: int = 0,
        limit: int = 20,
        parse_cap: int | None = None,
        search: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._require_server_document_workspace(normalized_mode)
        self._enforce_policy(self._document_references_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.get_document_references(
                media_id,
                enrich=enrich,
                reference_index=reference_index,
                offset=offset,
                limit=limit,
                parse_cap=parse_cap,
                search=search,
            )
        )

    async def list_document_versions(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        include_deleted: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.list_document_versions(media_id, include_deleted=include_deleted)
        )

    async def get_analysis_version(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        version_number: Any,
        include_content: bool = True,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "get_analysis_version"):
            raise ValueError("Document version detail is not available for this media backend.")
        return await self._maybe_await(
            service.get_analysis_version(
                media_id,
                version_number=version_number,
                include_content=include_content,
            )
        )

    async def save_analysis_version(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.save_analysis_version(
                media_id,
                content=content,
                analysis_content=analysis_content,
                prompt=prompt,
            )
        )

    async def overwrite_analysis_version(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.overwrite_analysis_version(
                media_id,
                content=content,
                analysis_content=analysis_content,
                prompt=prompt,
            )
        )

    async def delete_analysis_version(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        version_uuid: str,
        media_id: Any | None = None,
        version_number: Any | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.SERVER and (media_id is None or version_number is None):
            raise ValueError("Server document version delete requires media_id and version_number.")
        self._enforce_policy(self._reading_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == MediaReadingBackend.SERVER:
            return await self._maybe_await(
                service.delete_analysis_version(
                    version_uuid,
                    media_id=media_id,
                    version_number=version_number,
                )
            )
        return await self._maybe_await(service.delete_analysis_version(version_uuid))

    async def rollback_analysis_version(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        version_number: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "rollback_analysis_version"):
            raise ValueError("Document version rollback is not available for this media backend.")
        return await self._maybe_await(
            service.rollback_analysis_version(media_id, version_number=version_number)
        )

    async def patch_latest_version_metadata(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        safe_metadata: Mapping[str, Any],
        merge: bool = True,
        new_version: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "patch_latest_version_metadata"):
            raise ValueError("Document version metadata update is not available for this media backend.")
        return await self._maybe_await(
            service.patch_latest_version_metadata(
                media_id,
                safe_metadata=safe_metadata,
                merge=merge,
                new_version=new_version,
            )
        )

    async def update_analysis_version_metadata(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        version_number: Any,
        safe_metadata: Mapping[str, Any],
        merge: bool = True,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "update_analysis_version_metadata"):
            raise ValueError("Document version metadata update is not available for this media backend.")
        return await self._maybe_await(
            service.update_analysis_version_metadata(
                media_id,
                version_number=version_number,
                safe_metadata=safe_metadata,
                merge=merge,
            )
        )

    async def upsert_analysis_version(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        content: str | None = None,
        prompt: str | None = None,
        analysis_content: str | None = None,
        safe_metadata: Mapping[str, Any] | None = None,
        merge: bool = True,
        new_version: bool = True,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "upsert_analysis_version"):
            raise ValueError("Document version advanced upsert is not available for this media backend.")
        return await self._maybe_await(
            service.upsert_analysis_version(
                media_id,
                content=content,
                prompt=prompt,
                analysis_content=analysis_content,
                safe_metadata=safe_metadata,
                merge=merge,
                new_version=new_version,
            )
        )
