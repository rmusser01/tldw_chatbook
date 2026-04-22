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

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

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

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _deck_action_id(mode: StudyBackend, action: str) -> str:
        return f"study.deck.{action}.{mode.value}"

    @staticmethod
    def _flashcard_action_id(mode: StudyBackend, *, mutation: bool) -> str:
        # The audited registry currently exposes deck-level study actions only, so
        # flashcard and review operations proxy through the owning deck surface.
        action = "update" if mutation else "detail"
        return f"study.deck.{action}.{mode.value}"

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
        self._enforce_policy(self._deck_action_id(normalized_mode, "list"))
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
        self._enforce_policy(self._deck_action_id(normalized_mode, "create"))
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
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=False))
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
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
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

    async def move_flashcard(
        self,
        *,
        mode: StudyBackend | str | None = None,
        card_id: str,
        target_deck_id: str | int,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
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
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
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

    async def delete_deck(
        self,
        *,
        mode: StudyBackend | str | None = None,
        deck_id: str | int,
        expected_version: int | None = None,
        hard_delete: bool = False,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._deck_action_id(normalized_mode, "delete"))
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
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=False))
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
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
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
        self._enforce_policy(self._flashcard_action_id(normalized_mode, mutation=True))
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if normalized_mode == StudyBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        if normalized_mode == StudyBackend.LOCAL or review_session_id is None:
            return None
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "end_review_session"):
            return None
        return await self._maybe_await(service.end_review_session(review_session_id))
