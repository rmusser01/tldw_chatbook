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

ALLOWED_SERVER_CREATE_SOURCE_TYPES = ("archive_snapshot", "git_repository")


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
    def _saved_search_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.saved_searches.{action}.{mode.value}"

    @staticmethod
    def _note_link_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.reading.note_links.{action}.{mode.value}"

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

    @staticmethod
    def _ingestion_source_item_action_id(mode: MediaReadingBackend, action: str) -> str:
        return f"media.ingestion_source_items.{action}.{mode.value}"

    def read_it_later_browse_capability(
        self,
        *,
        mode: MediaReadingBackend | str | None = None,
        media_type_context: str | None = None,
    ) -> dict[str, Any]:
        """Return whether the read-it-later browse view is valid in a UI context."""
        normalized_mode = self._normalize_mode(mode)
        normalized_media_type = str(media_type_context or "all-media").strip() or "all-media"
        if normalized_mode == MediaReadingBackend.SERVER and normalized_media_type != "all-media":
            reason = "Read-it-later is only available in server mode from All Media."
            return {"available": False, "reason": reason}
        return {"available": True, "reason": ""}

    def _service_for_mode(self, mode: MediaReadingBackend) -> Any:
        if mode == MediaReadingBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local media backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server media backend is unavailable.")
        return self.server_service

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
            return model_dump(mode="json")
        return value

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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading item creation is not available yet. Use local ingest jobs instead.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading saved searches are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading saved searches are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading saved searches are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading saved searches are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading note links are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading note links are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading note links are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading bulk updates are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading archive snapshots are not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local reading summary generation is not available yet.")
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
        if normalized_mode == MediaReadingBackend.LOCAL:
            raise ValueError("Local ingestion source item reattach is not available yet.")
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
