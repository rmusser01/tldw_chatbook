"""Scope-aware seam for local and server media-reading flows."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping, Optional

from .media_reading_normalizers import (
    normalize_ingestion_source,
    normalize_ingestion_source_item,
    normalize_local_media_row,
    normalize_reading_progress,
    normalize_server_reading_item,
)


class MediaReadingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


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

    @staticmethod
    def _reading_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.{action}.{mode.value}"

    @staticmethod
    def _reading_progress_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading_progress.{action}.{mode.value}"

    @staticmethod
    def _reading_list_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"collections.reading_list.{action}.{mode.value}"

    @staticmethod
    def _ingestion_source_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.ingestion_sources.{action}.{mode.value}"

    @staticmethod
    def _ingestion_job_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.ingestion_jobs.{action}.{mode.value}"

    def _service_for_mode(self, mode: MediaReadingBackend) -> Any:
        if mode == MediaReadingBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local media backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server media backend is unavailable.")
        return self.server_service

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
            if hasattr(service, "save_to_read_it_later"):
                return await self._maybe_await(service.save_to_read_it_later(media_id))
            media_db = getattr(service, "media_db", None)
            if media_db is None or not hasattr(media_db, "save_media_to_read_it_later"):
                raise ValueError("Local read-it-later persistence is not available yet.")
            return await self._maybe_await(media_db.save_media_to_read_it_later(int(media_id)))
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
            if hasattr(service, "remove_from_read_it_later"):
                return await self._maybe_await(service.remove_from_read_it_later(media_id))
            media_db = getattr(service, "media_db", None)
            if media_db is None or not hasattr(media_db, "remove_media_from_read_it_later"):
                raise ValueError("Local read-it-later persistence is not available yet.")
            return await self._maybe_await(media_db.remove_media_from_read_it_later(int(media_id)))
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
        self._enforce_policy(self._ingestion_source_action_id(normalized_mode, "create"))
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local ingestion sources are not available yet.")
        service = self._service_for_mode(normalized_mode)
        source = await self._maybe_await(
            service.create_ingestion_source(
                source_type=source_type,
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
