"""Mode-aware routing for the study flashcards compat seam."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping

from .study_normalizers import (
    merge_review_outcome_record,
    normalize_study_deck_record,
    normalize_study_flashcard_record,
    normalize_study_review_candidate,
)


class StudyBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class StudyScopeService:
    """Route study flashcard actions to local or server backends and normalize outputs."""

    _ALLOWED_SCOPE_TYPES = {"global", "workspace"}

    def __init__(self, *, local_service: Any, server_service: Any):
        self.local_service = local_service
        self.server_service = server_service

    def _normalize_mode(self, mode: StudyBackend | str | None) -> StudyBackend:
        if mode is None:
            return StudyBackend.LOCAL
        if isinstance(mode, StudyBackend):
            return mode
        try:
            return StudyBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid study backend: {mode}") from exc

    def _service_for_mode(self, mode: StudyBackend) -> Any:
        if mode == StudyBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local study backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server study backend is unavailable.")
        return self.server_service

    def _server_only_service(self, mode: StudyBackend | str | None, feature_label: str) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == StudyBackend.LOCAL:
            raise ValueError(f"{feature_label} are server-only in Chatbook.")
        return self._service_for_mode(normalized_mode)

    @staticmethod
    def _with_server_source(payload: Any) -> Any:
        if isinstance(payload, Mapping):
            result = dict(payload)
            result.setdefault("source", "server")
            return result
        return payload

    @classmethod
    def _with_server_template_record(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(cls._with_server_source(payload))
        result.setdefault("entity_kind", "flashcard_template")
        template_id = result.get("id")
        if template_id is not None:
            result.setdefault("record_id", f"server:flashcard_template:{template_id}")
        return result

    @classmethod
    def _with_server_template_list(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(cls._with_server_source(payload))
        result.setdefault("entity_kind", "flashcard_template_list")
        result["items"] = [
            cls._with_server_template_record(item)
            for item in list(result.get("items") or [])
            if isinstance(item, Mapping)
        ]
        return result

    @classmethod
    def _with_server_flashcard_list(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(cls._with_server_source(payload))
        result.setdefault("entity_kind", "study_flashcard_list")
        result["items"] = [
            normalize_study_flashcard_record("server", item)
            for item in list(result.get("items") or [])
            if isinstance(item, Mapping)
        ]
        return result

    @classmethod
    def _with_server_bulk_update(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(cls._with_server_source(payload))
        result.setdefault("entity_kind", "study_flashcard_bulk_update")
        normalized_results: list[dict[str, Any]] = []
        for item in list(result.get("results") or []):
            if not isinstance(item, Mapping):
                continue
            normalized = dict(item)
            flashcard = normalized.get("flashcard")
            if isinstance(flashcard, Mapping):
                normalized["flashcard"] = normalize_study_flashcard_record("server", flashcard)
            normalized_results.append(normalized)
        result["results"] = normalized_results
        return result

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _coerce_delete_result(result: Any) -> bool:
        if isinstance(result, Mapping):
            if "deleted" in result:
                return bool(result.get("deleted"))
            return str(result.get("status") or "").strip().lower() == "deleted"
        return bool(result)

    @classmethod
    def _normalize_scope(cls, scope_type: str | None, workspace_id: str | None) -> tuple[str, str | None]:
        normalized_scope = str(scope_type or "global").strip().lower()
        if normalized_scope not in cls._ALLOWED_SCOPE_TYPES:
            raise ValueError(f"Invalid study scope_type: {scope_type}")
        normalized_workspace_id = str(workspace_id or "").strip() or None
        if normalized_scope == "workspace" and normalized_workspace_id is None:
            raise ValueError("workspace_id is required when scope_type='workspace'")
        return normalized_scope, normalized_workspace_id

    @staticmethod
    def _filter_deck_scope(record: Mapping[str, Any], *, scope_type: str, workspace_id: str | None) -> bool:
        record_workspace_id = record.get("workspace_id")
        if scope_type == "global":
            return record_workspace_id is None
        return record_workspace_id == workspace_id

    async def _load_scoped_server_decks(
        self,
        *,
        service: Any,
        limit: int,
        scope_type: str,
        workspace_id: str | None,
    ) -> list[dict[str, Any]]:
        page_size = max(1, int(limit or 100))
        fetched_records: list[dict[str, Any]] = []
        page_offset = 0
        while True:
            page = await self._maybe_await(service.list_decks(limit=page_size, offset=page_offset))
            page_items = list(page or [])
            fetched_records.extend(page_items)
            if len(page_items) < page_size:
                break
            page_offset += page_size

        normalized_records = [
            normalize_study_deck_record("server", record)
            for record in fetched_records
            if self._filter_deck_scope(record, scope_type=scope_type, workspace_id=workspace_id)
        ]
        return normalized_records

    async def list_decks(
        self,
        *,
        mode: StudyBackend | str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        normalized_scope_type, normalized_workspace_id = self._normalize_scope(scope_type, workspace_id)
        service = self._service_for_mode(normalized_mode)

        if normalized_mode == StudyBackend.LOCAL:
            if normalized_scope_type == "workspace":
                raise ValueError("Workspace Study is unavailable in local mode")
            records = await self._maybe_await(service.list_decks(limit=limit, offset=offset))
            return [
                normalize_study_deck_record(normalized_mode.value, record)
                for record in list(records or [])
            ]

        if normalized_scope_type == "workspace":
            records = await self._load_scoped_server_decks(
                service=service,
                limit=limit,
                scope_type=normalized_scope_type,
                workspace_id=normalized_workspace_id,
            )
        else:
            records = await self._load_scoped_server_decks(
                service=service,
                limit=limit,
                scope_type=normalized_scope_type,
                workspace_id=None,
            )

        if offset:
            records = records[offset:]
        if limit >= 0:
            records = records[:limit]
        return records

    async def create_deck(
        self,
        *,
        mode: StudyBackend | str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        name: str,
        description: str | None = None,
        scheduler_type: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        normalized_scope_type, normalized_workspace_id = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_deck(
                name=name,
                description=description,
                workspace_id=normalized_workspace_id if normalized_scope_type == "workspace" else None,
                scheduler_type=scheduler_type,
            )
        )
        return normalize_study_deck_record(normalized_mode.value, record)

    async def update_deck(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int,
        name: str | None = None,
        description: str | None = None,
        workspace_id: str | None = None,
        review_prompt_side: str | None = None,
        scheduler_type: str | None = None,
        scheduler_settings: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard deck update")
        record = await self._maybe_await(
            service.update_deck(
                deck_id,
                name=name,
                description=description,
                workspace_id=workspace_id,
                review_prompt_side=review_prompt_side,
                scheduler_type=scheduler_type,
                scheduler_settings=scheduler_settings,
                expected_version=expected_version,
            )
        )
        return normalize_study_deck_record("server", record)

    async def list_flashcards(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        raw_records = await self._maybe_await(
            service.list_flashcards(deck_id=deck_id, q=q, limit=limit, offset=offset)
        )
        return [
            normalize_study_flashcard_record(normalized_mode.value, record)
            for record in list(raw_records or [])
        ]

    async def create_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None,
        front: str,
        back: str,
        tags: list[str] | None = None,
        notes: str | None = None,
        extra: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_flashcard(
                deck_id=deck_id,
                front=front,
                back=back,
                tags=tags,
                notes=notes,
                extra=extra,
            )
        )
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def get_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        record = await self._maybe_await(self._service_for_mode(normalized_mode).get_flashcard(card_id))
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def create_flashcards_bulk(
        self,
        *,
        mode: StudyBackend | str | None = None,
        cards: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard bulk create")
        payload = await self._maybe_await(service.create_flashcards_bulk(cards))
        return self._with_server_flashcard_list(payload or {})

    async def move_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        target_deck_id: str | int,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        record = await self._maybe_await(
            self._service_for_mode(normalized_mode).move_flashcard(
                card_id,
                target_deck_id=target_deck_id,
                expected_version=expected_version,
            )
        )
        if not record:
            return None
        return normalize_study_flashcard_record(normalized_mode.value, record)

    async def delete_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        expected_version: int | None = None,
        hard_delete: bool = False,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == StudyBackend.SERVER:
            if expected_version is None:
                raise ValueError("expected_version is required for server flashcard deletion.")
            if expected_version < 1:
                raise ValueError("expected_version must be >= 1 for server flashcard deletion.")
        kwargs: dict[str, Any] = {"expected_version": expected_version}
        if normalized_mode == StudyBackend.LOCAL:
            kwargs["hard_delete"] = hard_delete
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).delete_flashcard(
                card_id,
                **kwargs,
            )
        )
        return self._coerce_delete_result(result)

    async def update_flashcards_bulk(
        self,
        *,
        mode: StudyBackend | str | None = None,
        cards: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard bulk update")
        payload = await self._maybe_await(service.update_flashcards_bulk(cards))
        return self._with_server_bulk_update(payload or {})

    async def reset_flashcard_scheduling(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        expected_version: int,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard scheduling reset")
        record = await self._maybe_await(
            service.reset_flashcard_scheduling(card_id, expected_version=expected_version)
        )
        return normalize_study_flashcard_record("server", record)

    async def set_flashcard_tags(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        tags: list[str],
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard tag update")
        record = await self._maybe_await(service.set_flashcard_tags(card_id, tags=tags))
        return normalize_study_flashcard_record("server", record)

    async def get_flashcard_tags(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard tag list")
        return dict(await self._maybe_await(service.get_flashcard_tags(card_id)) or {})

    async def list_flashcard_tag_suggestions(
        self,
        *,
        mode: StudyBackend | str | None = None,
        q: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard tag suggestions")
        payload = await self._maybe_await(service.list_flashcard_tag_suggestions(q=q, limit=limit))
        result = dict(self._with_server_source(payload or {}))
        result.setdefault("entity_kind", "flashcard_tag_suggestions")
        return result

    async def preview_structured_qa_import(
        self,
        *,
        mode: StudyBackend | str | None = None,
        content: str,
        max_lines: int | None = None,
        max_line_length: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard import preview")
        payload = await self._maybe_await(
            service.preview_structured_qa_import(
                content,
                max_lines=max_lines,
                max_line_length=max_line_length,
                max_field_length=max_field_length,
            )
        )
        result = dict(self._with_server_source(payload or {}))
        result.setdefault("entity_kind", "flashcard_import_preview")
        return result

    async def import_flashcards_tsv(
        self,
        *,
        mode: StudyBackend | str | None = None,
        content: str,
        delimiter: str = "\t",
        has_header: bool = False,
        max_lines: int | None = None,
        max_line_length: int | None = None,
        max_field_length: int | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard import")
        payload = await self._maybe_await(
            service.import_flashcards_tsv(
                content,
                delimiter=delimiter,
                has_header=has_header,
                max_lines=max_lines,
                max_line_length=max_line_length,
                max_field_length=max_field_length,
            )
        )
        result = dict(self._with_server_source(payload or {}))
        result.setdefault("entity_kind", "flashcard_import")
        return result

    async def export_flashcards(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool = False,
        tag: str | None = None,
        q: str | None = None,
        export_format: str = "csv",
        include_reverse: bool = False,
        delimiter: str = "\t",
        include_header: bool = False,
        extended_header: bool = False,
    ) -> bytes:
        service = self._server_only_service(mode, "Flashcard export")
        return await self._maybe_await(
            service.export_flashcards(
                deck_id=deck_id,
                workspace_id=workspace_id,
                include_workspace_items=include_workspace_items,
                tag=tag,
                q=q,
                export_format=export_format,
                include_reverse=include_reverse,
                delimiter=delimiter,
                include_header=include_header,
                extended_header=extended_header,
            )
        )

    async def get_flashcard_analytics_summary(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool = False,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard analytics summary")
        payload = await self._maybe_await(
            service.get_flashcard_analytics_summary(
                deck_id=deck_id,
                workspace_id=workspace_id,
                include_workspace_items=include_workspace_items,
            )
        )
        return dict(self._with_server_source(payload or {}))

    async def create_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        name: str,
        model_type: str = "basic",
        front_template: str,
        back_template: str | None = None,
        notes_template: str | None = None,
        extra_template: str | None = None,
        placeholder_definitions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard templates")
        payload = await self._maybe_await(
            service.create_flashcard_template(
                name=name,
                model_type=model_type,
                front_template=front_template,
                back_template=back_template,
                notes_template=notes_template,
                extra_template=extra_template,
                placeholder_definitions=placeholder_definitions,
            )
        )
        return self._with_server_template_record(payload or {})

    async def list_flashcard_templates(
        self,
        *,
        mode: StudyBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard templates")
        payload = await self._maybe_await(service.list_flashcard_templates(limit=limit, offset=offset))
        return self._with_server_template_list(payload or {})

    async def get_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: int,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard templates")
        payload = await self._maybe_await(service.get_flashcard_template(template_id))
        return self._with_server_template_record(payload or {})

    async def update_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: int,
        name: str | None = None,
        model_type: str | None = None,
        front_template: str | None = None,
        back_template: str | None = None,
        notes_template: str | None = None,
        extra_template: str | None = None,
        placeholder_definitions: list[dict[str, Any]] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Flashcard templates")
        payload = await self._maybe_await(
            service.update_flashcard_template(
                template_id,
                name=name,
                model_type=model_type,
                front_template=front_template,
                back_template=back_template,
                notes_template=notes_template,
                extra_template=extra_template,
                placeholder_definitions=placeholder_definitions,
                expected_version=expected_version,
            )
        )
        return self._with_server_template_record(payload or {})

    async def delete_flashcard_template(
        self,
        *,
        mode: StudyBackend | str | None = None,
        template_id: int,
        expected_version: int,
    ) -> bool:
        service = self._server_only_service(mode, "Flashcard templates")
        result = await self._maybe_await(
            service.delete_flashcard_template(template_id, expected_version=expected_version)
        )
        return self._coerce_delete_result(result)

    async def delete_deck(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int,
        expected_version: int | None = None,
        hard_delete: bool = False,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).delete_deck(
                deck_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )
        return self._coerce_delete_result(result)

    async def get_next_review_candidate(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        payload = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_next_review_candidate(deck_id=deck_id)
        )
        payload = dict(payload or {})
        return normalize_study_review_candidate(
            normalized_mode.value,
            card=payload.get("card"),
            selection_reason=payload.get("selection_reason"),
            review_session=payload.get("review_session"),
        )

    async def submit_flashcard_review(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        rating: int,
        current_card: Mapping[str, Any] | None = None,
        answer_time_ms: int | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        service = self._service_for_mode(normalized_mode)
        kwargs = {"rating": rating}
        if normalized_mode == StudyBackend.SERVER and answer_time_ms is not None:
            kwargs["answer_time_ms"] = answer_time_ms
        response = await self._maybe_await(service.submit_flashcard_review(card_id, **kwargs))
        return merge_review_outcome_record(
            normalized_mode.value,
            current_card=current_card,
            review_response=response,
            rating=rating,
        )

    async def end_review_session(
        self,
        *,
        mode: StudyBackend | str | None = None,
        review_session_id: int | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        if normalized_mode == StudyBackend.LOCAL or review_session_id is None:
            return None
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "end_review_session"):
            return None
        return await self._maybe_await(service.end_review_session(review_session_id))

    async def create_study_pack_job(
        self,
        *,
        mode: StudyBackend | str | None = None,
        title: str,
        source_items: list[Mapping[str, Any]],
        workspace_id: str | None = None,
        deck_mode: str = "new",
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study packs")
        payload = await self._maybe_await(
            service.create_study_pack_job(
                title=title,
                source_items=source_items,
                workspace_id=workspace_id,
                deck_mode=deck_mode,
            )
        )
        return self._with_server_source(payload)

    async def get_study_pack_job_status(
        self,
        *,
        mode: StudyBackend | str | None = None,
        job_id: int,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study packs")
        return self._with_server_source(
            await self._maybe_await(service.get_study_pack_job_status(job_id))
        )

    async def get_study_pack(
        self,
        *,
        mode: StudyBackend | str | None = None,
        pack_id: int,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study packs")
        return self._with_server_source(
            await self._maybe_await(service.get_study_pack(pack_id))
        )

    async def regenerate_study_pack(
        self,
        *,
        mode: StudyBackend | str | None = None,
        pack_id: int,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study packs")
        return self._with_server_source(
            await self._maybe_await(service.regenerate_study_pack(pack_id))
        )

    async def get_study_suggestion_status(
        self,
        *,
        mode: StudyBackend | str | None = None,
        anchor_type: str,
        anchor_id: int,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study suggestions")
        return self._with_server_source(
            await self._maybe_await(
                service.get_study_suggestion_status(anchor_type=anchor_type, anchor_id=anchor_id)
            )
        )

    async def get_study_suggestion_snapshot(
        self,
        *,
        mode: StudyBackend | str | None = None,
        snapshot_id: int,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study suggestions")
        return self._with_server_source(
            await self._maybe_await(service.get_study_suggestion_snapshot(snapshot_id))
        )

    async def refresh_study_suggestion_snapshot(
        self,
        *,
        mode: StudyBackend | str | None = None,
        snapshot_id: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study suggestions")
        return self._with_server_source(
            await self._maybe_await(
                service.refresh_study_suggestion_snapshot(snapshot_id, reason=reason)
            )
        )

    async def trigger_study_suggestion_action(
        self,
        *,
        mode: StudyBackend | str | None = None,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        action_kind: str,
        selected_topic_ids: list[str] | None = None,
        selected_topic_edits: list[dict[str, str]] | None = None,
        manual_topic_labels: list[str] | None = None,
        has_explicit_selection: bool = False,
        generator_version: str = "v1",
        force_regenerate: bool = False,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Study suggestions")
        return self._with_server_source(
            await self._maybe_await(
                service.trigger_study_suggestion_action(
                    snapshot_id,
                    target_service=target_service,
                    target_type=target_type,
                    action_kind=action_kind,
                    selected_topic_ids=selected_topic_ids,
                    selected_topic_edits=selected_topic_edits,
                    manual_topic_labels=manual_topic_labels,
                    has_explicit_selection=has_explicit_selection,
                    generator_version=generator_version,
                    force_regenerate=force_regenerate,
                )
            )
        )
