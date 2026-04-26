"""Scope-aware seam for local and server media-reading flows."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from .media_reading_normalizers import (
    normalize_file_artifact,
    normalize_ingestion_source,
    normalize_ingestion_source_item,
    normalize_local_media_row,
    normalize_reference_image,
    normalize_reading_progress,
    normalize_server_reading_item,
)

ALLOWED_SERVER_CREATE_SOURCE_TYPES = ("archive_snapshot", "git_repository")

_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "media.web_content_ingest.local",
        "source": "local",
        "supported": False,
        "reason_code": "source_specific_equivalent",
        "user_message": "The direct web-content ingestion endpoint is server-owned; local mode uses local URL ingest jobs instead.",
        "affected_action_ids": [],
    },
    {
        "operation_id": "media.processing.mediawiki.import.local",
        "source": "local",
        "supported": False,
        "reason_code": "source_specific_equivalent",
        "user_message": "MediaWiki dump processing is available locally, but MediaWiki dump import remains server-owned until local persistence semantics are defined.",
        "affected_action_ids": [],
    },
    {
        "operation_id": "media.transcription_models.local",
        "source": "local",
        "supported": False,
        "reason_code": "source_specific_equivalent",
        "user_message": "The server transcription-model discovery endpoint is server-owned; local model discovery remains in local transcription settings.",
        "affected_action_ids": [],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "collections.reading_list.per_media_type.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "Server read-it-later browsing is exposed only as the aggregate All Media saved view; per-media-type saved views remain unavailable.",
        "affected_action_ids": ["collections.reading_list.list.server"],
    },
    {
        "operation_id": "media.ingestion_sources.delete.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server ingestion-source API does not expose deletion.",
        "affected_action_ids": ["media.ingestion_sources.delete.server"],
    },
]


class MediaReadingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


@dataclass(frozen=True)
class ReadItLaterContextCapability:
    """Source-aware read-it-later browse capability for the current UI context."""

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

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    def list_unsupported_capabilities(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    @staticmethod
    def _reading_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.{action}.{mode.value}"

    @staticmethod
    def _saved_search_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.saved_searches.{action}.{mode.value}"

    @staticmethod
    def _note_link_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.note_links.{action}.{mode.value}"

    @staticmethod
    def _reading_progress_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_progress.{action}.{mode.value}"

    @staticmethod
    def _reading_import_job_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_import_jobs.{action}.{mode.value}"

    @staticmethod
    def _reading_digest_schedule_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.digest_schedules.{action}.{mode.value}"

    @staticmethod
    def _reading_digest_output_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.digest_outputs.{action}.{mode.value}"

    @staticmethod
    def _reading_digest_scheduler_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.digest_scheduler.{action}.{mode.value}"

    @staticmethod
    def _web_content_ingest_action_id(action: str) -> str:
        return f"media.web_content_ingest.{action}.server"

    @staticmethod
    def _processing_action_id(kind: str, action: str, mode: MediaReadingBackend | None = None) -> str:
        source = mode.value if mode is not None else MediaReadingBackend.SERVER.value
        return f"media.processing.{kind}.{action}.{source}"

    @staticmethod
    def _transcription_models_action_id(action: str) -> str:
        return f"media.transcription_models.{action}.server"

    @staticmethod
    def _media_item_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.items.{action}.{mode.value}"

    @staticmethod
    def _media_item_subresource_action_id(mode: MediaReadingBackend, subresource: str, action: str) -> str:
        return f"media.items.{subresource}.{action}.{mode.value}"

    @staticmethod
    def _media_add_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.add.{action}.{mode.value}"

    @staticmethod
    def _file_artifact_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.file_artifacts.{action}.{mode.value}"

    @staticmethod
    def _reference_image_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reference_images.{action}.{mode.value}"

    @staticmethod
    def _reading_list_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"collections.reading_list.{action}.{mode.value}"

    @staticmethod
    def _navigation_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.navigation.{action}.{mode.value}"

    @staticmethod
    def _ingestion_source_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.ingestion_sources.{action}.{mode.value}"

    @staticmethod
    def _ingestion_job_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.ingestion_jobs.{action}.{mode.value}"

    @staticmethod
    def _ingestion_source_item_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.ingestion_source_items.{action}.{mode.value}"

    def get_read_it_later_context_capability(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_type_slug: str | None = None,
    ) -> ReadItLaterContextCapability:
        """Return the authoritative read-it-later browse capability for a context."""
        normalized_mode = self._normalize_mode(mode)
        normalized_media_type = str(media_type_slug or "all-media").strip().lower() or "all-media"

        if normalized_mode == MediaReadingBackend.LOCAL:
            return ReadItLaterContextCapability(
                available=True,
                aggregate_only=False,
                reason=None,
            )

        if normalized_media_type == "all-media":
            return ReadItLaterContextCapability(
                available=True,
                aggregate_only=True,
                reason=None,
            )

        return ReadItLaterContextCapability(
            available=False,
            aggregate_only=True,
            reason="Read-it-later is only available in server mode from All Media.",
        )

    def read_it_later_browse_capability(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_type_context: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility adapter for older UI code that expects a dict payload."""
        capability = self.get_read_it_later_context_capability(
            mode=mode,
            media_type_slug=media_type_context,
        )
        return {"available": capability.available, "reason": capability.reason or ""}

    def _service_for_mode(self, mode: MediaReadingBackend) -> Any:
        if mode == MediaReadingBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local media backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server media backend is unavailable.")
        return self.server_service

    def _server_web_content_ingest_service(self, mode: MediaReadingBackend) -> Any:
        if mode == MediaReadingBackend.LOCAL:
            raise ValueError("The direct web-content ingestion is server-only; use local URL ingest jobs in local mode.")
        return self._service_for_mode(mode)

    def _server_processing_service(self, mode: MediaReadingBackend, operation_name: str) -> Any:
        if mode == MediaReadingBackend.LOCAL:
            raise ValueError(f"{operation_name} is server-only; use local/offline ingestion tooling in local mode.")
        return self._service_for_mode(mode)

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

    @staticmethod
    def _to_plain(value: Any) -> Any:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="python")
        return value

    async def create_file_artifact(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        request_data: Any | None = None,
        file_type: str | None = None,
        payload: Mapping[str, Any] | None = None,
        title: str | None = None,
        export: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._file_artifact_action_id(normalized_mode, "create"))
        if request_data is not None:
            response = await self._maybe_await(service.create_file_artifact(request_data=request_data))
        else:
            response = await self._maybe_await(
                service.create_file_artifact(
                    file_type=file_type,
                    payload=dict(payload or {}),
                    title=title,
                    export=dict(export) if export is not None else None,
                    options=dict(options or {"persist": True}),
                )
            )
        return normalize_file_artifact(self._to_plain(response), backend=normalized_mode.value)

    async def list_reference_images(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._reference_image_action_id(normalized_mode, "list"))
        response = self._to_plain(await self._maybe_await(service.list_reference_images()))
        items = response.get("items", []) if isinstance(response, Mapping) else list(response or [])
        normalized_items = [normalize_reference_image(item, backend=normalized_mode.value) for item in items]
        return {
            "items": normalized_items,
            "total": response.get("total", len(normalized_items)) if isinstance(response, Mapping) else len(normalized_items),
        }

    async def get_file_artifact(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        file_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._file_artifact_action_id(normalized_mode, "detail"))
        response = await self._maybe_await(service.get_file_artifact(file_id))
        return normalize_file_artifact(self._to_plain(response), backend=normalized_mode.value)

    async def export_file_artifact(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        file_id: Any,
        format: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._file_artifact_action_id(normalized_mode, "export"))
        return self._to_plain(await self._maybe_await(service.export_file_artifact(file_id, format=format)))

    async def delete_file_artifact(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        file_id: Any,
        hard: bool = False,
        delete_file: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._file_artifact_action_id(normalized_mode, "delete"))
        return self._to_plain(
            await self._maybe_await(service.delete_file_artifact(file_id, hard=hard, delete_file=delete_file))
        )

    async def purge_file_artifacts(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        delete_files: bool = False,
        soft_deleted_grace_days: int = 30,
        include_retention: bool = True,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._file_artifact_action_id(normalized_mode, "purge"))
        return self._to_plain(
            await self._maybe_await(
                service.purge_file_artifacts(
                    delete_files=delete_files,
                    soft_deleted_grace_days=soft_deleted_grace_days,
                    include_retention=include_retention,
                )
            )
        )

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
        media_type_context = filters.pop("media_type_context", None)
        capability = self.read_it_later_browse_capability(
            mode=normalized_mode,
            media_type_context=media_type_context,
        )
        if not capability["available"]:
            raise ValueError(str(capability["reason"]))
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

    async def save_reading_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        url: str,
        title: str | None = None,
        tags: list[str] | None = None,
        status: str | None = "saved",
        archive_mode: str = "use_default",
        favorite: bool = False,
        summary: str | None = None,
        notes: str | None = None,
        content: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        item = await self._maybe_await(
            service.save_reading_item(
                url=url,
                title=title,
                tags=tags,
                status=status,
                archive_mode=archive_mode,
                favorite=favorite,
                summary=summary,
                notes=notes,
                content=content,
            )
        )
        return self._normalize_media_record(normalized_mode, item)

    async def create_saved_search(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        name: str,
        query: Mapping[str, Any] | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._saved_search_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(await self._maybe_await(service.create_saved_search(name=name, query=query, sort=sort)))

    async def list_saved_searches(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._saved_search_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(await self._maybe_await(service.list_saved_searches(limit=limit, offset=offset)))

    async def update_saved_search(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        search_id: Any,
        name: str | None = None,
        query: Mapping[str, Any] | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._saved_search_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(
            await self._maybe_await(
                service.update_saved_search(search_id, name=name, query=query, sort=sort)
            )
        )

    async def delete_saved_search(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        search_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._saved_search_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(await self._maybe_await(service.delete_saved_search(search_id)))

    async def link_note(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        note_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._note_link_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(await self._maybe_await(service.link_note(item_id, note_id)))

    async def list_note_links(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._note_link_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(await self._maybe_await(service.list_note_links(item_id)))

    async def unlink_note(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        note_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._note_link_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(await self._maybe_await(service.unlink_note(item_id, note_id)))

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
        self._enforce_policy(self._reading_action_id(normalized_mode, "bulk_update"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(
            await self._maybe_await(
                service.bulk_update_reading_items(
                    item_ids=item_ids,
                    action=action,
                    status=status,
                    favorite=favorite,
                    tags=tags,
                    hard=hard,
                )
            )
        )

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
        self._enforce_policy(self._reading_action_id(normalized_mode, "archive"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(
            await self._maybe_await(
                service.create_reading_archive(
                    item_id,
                    format=format,
                    source=source,
                    title=title,
                    retention_days=retention_days,
                    retention_until=retention_until,
                )
            )
        )

    async def summarize_reading_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        provider: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        recursive: bool = False,
        chunked: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "summarize"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(
            await self._maybe_await(
                service.summarize_reading_item(
                    item_id,
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    recursive=recursive,
                    chunked=chunked,
                )
            )
        )

    async def tts_reading_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        model: str,
        voice: str = "af_heart",
        response_format: str = "mp3",
        stream: bool = True,
        speed: float | None = None,
        max_chars: int | None = None,
        text_source: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "tts"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(
            await self._maybe_await(
                service.tts_reading_item(
                    item_id,
                    model=model,
                    voice=voice,
                    response_format=response_format,
                    stream=stream,
                    speed=speed,
                    max_chars=max_chars,
                    text_source=text_source,
                )
            )
        )

    async def import_reading_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        import_path: str,
        source: str = "auto",
        merge_tags: bool = True,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "import"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(
            await self._maybe_await(
                service.import_reading_items(
                    import_path,
                    source=source,
                    merge_tags=merge_tags,
                )
            )
        )

    async def export_reading_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
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
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "export"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(
            await self._maybe_await(
                service.export_reading_items(
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
            )
        )

    async def list_reading_import_jobs(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_import_job_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(
            await self._maybe_await(
                service.list_reading_import_jobs(status=status, limit=limit, offset=offset)
            )
        )

    async def get_reading_import_job(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        job_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_import_job_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return self._to_plain(await self._maybe_await(service.get_reading_import_job(job_id)))

    async def create_reading_digest_schedule(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
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
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "create"))
        return await self._maybe_await(
            service.create_reading_digest_schedule(
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
        )

    async def list_reading_digest_schedules(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "list"))
        return await self._maybe_await(service.list_reading_digest_schedules(limit=limit, offset=offset))

    async def get_reading_digest_schedule(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        schedule_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "detail"))
        return await self._maybe_await(service.get_reading_digest_schedule(schedule_id))

    async def update_reading_digest_schedule(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        schedule_id: str,
        **changes: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "update"))
        return await self._maybe_await(service.update_reading_digest_schedule(schedule_id, **changes))

    async def delete_reading_digest_schedule(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        schedule_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._reading_digest_schedule_action_id(normalized_mode, "delete"))
        return await self._maybe_await(service.delete_reading_digest_schedule(schedule_id))

    async def list_reading_digest_outputs(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        schedule_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._reading_digest_output_action_id(normalized_mode, "list"))
        return await self._maybe_await(
            service.list_reading_digest_outputs(schedule_id=schedule_id, limit=limit, offset=offset)
        )

    async def run_due_reading_digest_schedules(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        now: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != MediaReadingBackend.LOCAL:
            raise ValueError("Local reading digest scheduler execution is local-only; server schedules run on the server.")
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._reading_digest_scheduler_action_id(normalized_mode, "trigger"))
        return self._to_plain(await self._maybe_await(service.run_due_reading_digest_schedules(now=now)))

    async def ingest_web_content(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
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
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_web_content_ingest_service(normalized_mode)
        self._enforce_policy(self._web_content_ingest_action_id("launch"))
        return self._to_plain(
            await self._maybe_await(
                service.ingest_web_content(
                    urls=urls,
                    titles=titles,
                    authors=authors,
                    keywords=keywords,
                    scrape_method=scrape_method,
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
            )
        )

    async def process_code(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        perform_chunking: bool = True,
        chunk_method: str | None = "code",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._processing_action_id("code", "process", normalized_mode))
        return self._to_plain(
            await self._maybe_await(
                service.process_code(
                    urls=urls,
                    file_paths=file_paths,
                    perform_chunking=perform_chunking,
                    chunk_method=chunk_method,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        )

    async def _process_existing_no_db_media(
        self,
        *,
        mode: MediaReadingBackend | str | None,
        kind: str,
        method_name: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MediaReadingBackend.LOCAL and kind not in {"video", "audio", "pdf", "ebook", "document", "plaintext"}:
            service = self._server_processing_service(normalized_mode, f"process-{kind}")
        else:
            service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._processing_action_id(kind, "process", normalized_mode))
        service_method = getattr(service, method_name)
        return self._to_plain(
            await self._maybe_await(
                service_method(urls=urls, file_paths=file_paths, **options)
            )
        )

    async def process_video(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        return await self._process_existing_no_db_media(
            mode=mode,
            kind="video",
            method_name="process_video",
            urls=urls,
            file_paths=file_paths,
            **options,
        )

    async def process_audio(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        return await self._process_existing_no_db_media(
            mode=mode,
            kind="audio",
            method_name="process_audio",
            urls=urls,
            file_paths=file_paths,
            **options,
        )

    async def process_pdf(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        return await self._process_existing_no_db_media(
            mode=mode,
            kind="pdf",
            method_name="process_pdf",
            urls=urls,
            file_paths=file_paths,
            **options,
        )

    async def process_ebook(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        return await self._process_existing_no_db_media(
            mode=mode,
            kind="ebook",
            method_name="process_ebook",
            urls=urls,
            file_paths=file_paths,
            **options,
        )

    async def process_document(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        return await self._process_existing_no_db_media(
            mode=mode,
            kind="document",
            method_name="process_document",
            urls=urls,
            file_paths=file_paths,
            **options,
        )

    async def process_plaintext(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        return await self._process_existing_no_db_media(
            mode=mode,
            kind="plaintext",
            method_name="process_plaintext",
            urls=urls,
            file_paths=file_paths,
            **options,
        )

    async def process_emails(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        file_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._processing_action_id("emails", "process", normalized_mode))
        return self._to_plain(
            await self._maybe_await(
                service.process_emails(file_paths=file_paths, **kwargs)
            )
        )

    async def process_web_scraping(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        scrape_method: str,
        url_input: str,
        mode_value: str = "persist",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._processing_action_id("web_scraping", "process", normalized_mode))
        return self._to_plain(
            await self._maybe_await(
                service.process_web_scraping(
                    scrape_method=scrape_method,
                    url_input=url_input,
                    mode=mode_value,
                    **kwargs,
                )
            )
        )

    async def get_transcription_models(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_processing_service(normalized_mode, "transcription model discovery")
        self._enforce_policy(self._transcription_models_action_id("list"))
        return await self._maybe_await(service.get_transcription_models())

    async def list_media_items(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "list"))
        return self._to_plain(
            await self._maybe_await(
                service.list_media_items(
                    page=page,
                    results_per_page=results_per_page,
                    include_keywords=include_keywords,
                )
            )
        )

    async def add_media(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_add_action_id(normalized_mode, "create"))
        return self._to_plain(
            await self._maybe_await(
                service.add_media(
                    media_type=media_type,
                    urls=urls,
                    file_paths=file_paths,
                    **options,
                )
            )
        )

    async def list_media_keywords(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        query: str | None = None,
        limit: int = 100,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "keywords", "list"))
        return self._to_plain(await self._maybe_await(service.list_media_keywords(query=query, limit=limit)))

    async def list_media_trash(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "trash", "list"))
        return self._to_plain(
            await self._maybe_await(
                service.list_media_trash(
                    page=page,
                    results_per_page=results_per_page,
                    include_keywords=include_keywords,
                )
            )
        )

    async def empty_media_trash(self, *, mode: MediaReadingBackend | str | None = None) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "trash", "delete"))
        return self._to_plain(await self._maybe_await(service.empty_media_trash()))

    async def get_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "detail"))
        return self._to_plain(
            await self._maybe_await(
                service.get_media_item(
                    media_id,
                    include_content=include_content,
                    include_versions=include_versions,
                    include_version_content=include_version_content,
                )
            )
        )

    async def update_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        **fields: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "update"))
        return self._to_plain(await self._maybe_await(service.update_media_item(media_id, **fields)))

    async def delete_media_item(self, *, mode: MediaReadingBackend | str | None = None, media_id: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "delete"))
        return self._to_plain(await self._maybe_await(service.delete_media_item(media_id)))

    async def restore_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_action_id(normalized_mode, "restore"))
        return self._to_plain(
            await self._maybe_await(
                service.restore_media_item(
                    media_id,
                    include_content=include_content,
                    include_versions=include_versions,
                    include_version_content=include_version_content,
                )
            )
        )

    async def permanently_delete_media_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "permanent", "delete"))
        return self._to_plain(await self._maybe_await(service.permanently_delete_media_item(media_id)))

    async def update_media_keywords(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        keywords: list[str],
        update_mode: str = "add",
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "keywords", "update"))
        return self._to_plain(
            await self._maybe_await(
                service.update_media_keywords(media_id, keywords=keywords, mode=update_mode)
            )
        )

    async def search_media_metadata(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        **filters: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "metadata_search", "list"))
        return self._to_plain(await self._maybe_await(service.search_media_metadata(**filters)))

    async def get_media_by_identifier(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        **identifiers: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "identifier_lookup", "detail"))
        return self._to_plain(await self._maybe_await(service.get_media_by_identifier(**identifiers)))

    async def process_mediawiki_dump(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        dump_file_path: str,
        **options: Any,
    ):
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._processing_action_id("mediawiki", "process", normalized_mode))
        async for item in service.process_mediawiki_dump(dump_file_path=dump_file_path, **options):
            yield self._to_plain(item)

    async def ingest_mediawiki_dump(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        dump_file_path: str,
        **options: Any,
    ):
        normalized_mode = self._normalize_mode(mode)
        service = self._server_processing_service(normalized_mode, "MediaWiki dump ingest")
        self._enforce_policy(self._processing_action_id("mediawiki", "import", normalized_mode))
        async for item in service.ingest_mediawiki_dump(dump_file_path=dump_file_path, **options):
            yield self._to_plain(item)

    async def download_media_file(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        file_type: str = "original",
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "file", "detail"))
        return self._to_plain(await self._maybe_await(service.download_media_file(media_id, file_type=file_type)))

    async def check_media_file(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        file_type: str = "original",
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._media_item_subresource_action_id(normalized_mode, "file", "detail"))
        return self._to_plain(await self._maybe_await(service.check_media_file(media_id, file_type=file_type)))

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

    async def create_highlight(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
        quote: str,
        start_offset: int | None = None,
        end_offset: int | None = None,
        color: str | None = None,
        note: str | None = None,
        anchor_strategy: str = "fuzzy_quote",
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.create_highlight(
                item_id,
                quote=quote,
                start_offset=start_offset,
                end_offset=end_offset,
                color=color,
                note=note,
                anchor_strategy=anchor_strategy,
            )
        )

    async def list_highlights(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        item_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.list_highlights(item_id))

    async def update_highlight(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        highlight_id: Any,
        color: str | None = None,
        note: str | None = None,
        state: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.update_highlight(
                highlight_id,
                color=color,
                note=note,
                state=state,
            )
        )

    async def delete_highlight(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        highlight_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_highlight(highlight_id))

    async def list_annotations(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.list_annotations(media_id))

    async def create_annotation(
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
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.create_annotation(
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

    async def update_annotation(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        annotation_id: str,
        text: str | None = None,
        color: str | None = None,
        note: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.update_annotation(
                media_id,
                annotation_id,
                text=text,
                color=color,
                note=note,
            )
        )

    async def delete_annotation(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        annotation_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_annotation(media_id, annotation_id))

    async def sync_annotations(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        annotations: list[Mapping[str, Any]],
        client_ids: list[str] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.sync_annotations(
                media_id,
                annotations=annotations,
                client_ids=client_ids,
            )
        )

    async def get_document_outline(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
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
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.get_document_figures(media_id, min_size=min_size))

    async def get_document_references(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        enrich: bool = False,
        reference_index: int | None = None,
        offset: int = 0,
        limit: int = 50,
        parse_cap: int | None = None,
        search: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
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
        self._enforce_policy(self._reading_action_id(normalized_mode, "detail"))
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

    async def get_media_navigation(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        include_generated_fallback: bool = False,
        max_depth: int = 4,
        max_nodes: int = 500,
        parent_id: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._navigation_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.get_media_navigation(
                media_id,
                include_generated_fallback=include_generated_fallback,
                max_depth=max_depth,
                max_nodes=max_nodes,
                parent_id=parent_id,
            )
        )

    async def get_media_navigation_content(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        node_id: str,
        format: str = "auto",
        include_alternates: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._navigation_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.get_media_navigation_content(
                media_id,
                node_id,
                format=format,
                include_alternates=include_alternates,
            )
        )

    async def submit_ingest_jobs(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        keywords: list[str] | None = None,
        **options: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        payload = {
            "media_type": media_type,
            "urls": urls,
            "keywords": keywords,
            **options,
        }
        if file_paths is not None:
            payload["file_paths"] = file_paths
        return await self._maybe_await(
            service.submit_ingest_jobs(**payload)
        )

    async def get_ingest_job(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        job_id: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.get_ingest_job(job_id))

    async def list_ingest_jobs(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        batch_id: str,
        limit: int = 100,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.list_ingest_jobs(batch_id, limit=limit))

    def stream_ingest_job_events(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        batch_id: str | None = None,
        after_id: int = 0,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        return service.stream_ingest_job_events(batch_id=batch_id, after_id=after_id)

    async def cancel_ingest_job(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        job_id: Any,
        reason: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "cancel"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.cancel_ingest_job(job_id, reason=reason))

    async def cancel_ingest_batch(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "cancel"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.cancel_ingest_batch(
                batch_id=batch_id,
                session_id=session_id,
                reason=reason,
            )
        )

    async def reprocess_media(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_id: Any,
        **options: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_job_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.reprocess_media(media_id, **options))

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
        normalized_source_type = (
            source_type
            if normalized_mode == MediaReadingBackend.LOCAL
            else self._validate_server_create_source_type(source_type)
        )
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

    async def reattach_ingestion_source_item(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        source_id: Any,
        item_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._ingestion_source_item_action_id(normalized_mode, "reattach"))
        service = self._service_for_mode(normalized_mode)
        item = await self._maybe_await(service.reattach_ingestion_source_item(source_id, item_id))
        return normalize_ingestion_source_item(item, backend=normalized_mode.value)

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
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._reading_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.delete_analysis_version(version_uuid))
