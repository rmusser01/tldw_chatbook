"""Mode-aware routing for the study quiz compat seam."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional

from .quiz_normalizers import (
    normalize_quiz_attempt_record,
    normalize_quiz_question_record,
    normalize_quiz_record,
)


class QuizBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class QuizScopeService:
    """Route quiz actions to local or server backends and normalize outputs."""

    _ALLOWED_SCOPE_TYPES = {"global", "workspace"}

    def __init__(self, *, local_service: Any = None, server_service: Any = None):
        self.local_service = local_service
        self.server_service = server_service

    def _resolve_backend(self, mode: Optional[str]) -> QuizBackend:
        normalized_mode = str(mode or "local").strip().lower()
        try:
            return QuizBackend(normalized_mode)
        except ValueError as exc:
            raise ValueError(f"Invalid quiz backend: {mode}") from exc

    def _service_for(self, backend: QuizBackend) -> Any:
        if backend is QuizBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local quiz backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server quiz backend is unavailable.")
        return self.server_service

    async def _maybe_await(self, result: Any) -> Any:
        if inspect.isawaitable(result):
            return await result
        return result

    @classmethod
    def _normalize_scope(self, scope_type: Optional[str], workspace_id: Optional[str]) -> tuple[str, Optional[str]]:
        normalized_scope = str(scope_type or "global").strip().lower()
        if normalized_scope not in self._ALLOWED_SCOPE_TYPES:
            raise ValueError(f"Invalid quiz scope_type: {scope_type}")
        normalized_workspace_id = str(workspace_id or "").strip() or None
        if normalized_scope == "workspace" and normalized_workspace_id is None:
            raise ValueError("workspace_id is required when scope_type='workspace'")
        return normalized_scope, normalized_workspace_id

    @staticmethod
    def _filter_quiz_scope(record: Mapping[str, Any], *, scope_type: str, workspace_id: Optional[str]) -> bool:
        record_workspace_id = record.get("workspace_id")
        if scope_type == "global":
            return record_workspace_id is None
        return record_workspace_id == workspace_id

    async def _load_scoped_server_quizzes(
        self,
        *,
        service: Any,
        q: Optional[str],
        limit: int,
        scope_type: str,
        workspace_id: Optional[str],
    ) -> list[dict[str, Any]]:
        page_size = max(1, int(limit or 100))
        fetched_records: list[dict[str, Any]] = []
        page_offset = 0
        while True:
            response = await self._maybe_await(service.list_quizzes(q=q, limit=page_size, offset=page_offset))
            page_items = list((response or {}).get("items") or response or [])
            fetched_records.extend(page_items)
            if len(page_items) < page_size:
                break
            page_offset += page_size

        return [
            normalize_quiz_record("server", record)
            for record in fetched_records
            if self._filter_quiz_scope(record, scope_type=scope_type, workspace_id=workspace_id)
        ]

    async def list_quizzes(
        self,
        *,
        mode: Optional[str] = None,
        scope_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        backend = self._resolve_backend(mode)
        service = self._service_for(backend)
        normalized_q = str(q or "").strip() or None
        normalized_scope_type, normalized_workspace_id = self._normalize_scope(scope_type, workspace_id)

        if backend is QuizBackend.LOCAL:
            if normalized_scope_type == "workspace":
                raise ValueError("Workspace Study is unavailable in local mode")
            records = await self._maybe_await(service.list_quizzes(q=normalized_q, limit=limit, offset=offset))
            items = list((records or {}).get("items") or records or [])
            return [normalize_quiz_record(backend.value, record) for record in items]

        records = await self._load_scoped_server_quizzes(
            service=service,
            q=normalized_q,
            limit=limit,
            scope_type=normalized_scope_type,
            workspace_id=normalized_workspace_id,
        )
        if offset:
            records = records[offset:]
        if limit >= 0:
            records = records[:limit]
        return records

    async def create_quiz(
        self,
        *,
        mode: Optional[str] = None,
        scope_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        name: str,
        description: Optional[str] = None,
        time_limit_seconds: Optional[int] = None,
        passing_score: Optional[int] = None,
    ) -> dict[str, Any]:
        backend = self._resolve_backend(mode)
        service = self._service_for(backend)
        normalized_scope_type, normalized_workspace_id = self._normalize_scope(scope_type, workspace_id)
        if backend is QuizBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        result = await self._maybe_await(
            service.create_quiz(
                name=name,
                description=description,
                time_limit_seconds=time_limit_seconds,
                passing_score=passing_score,
                workspace_id=normalized_workspace_id if normalized_scope_type == "workspace" else None,
            )
        )
        return normalize_quiz_record(backend.value, result)

    async def delete_quiz(
        self,
        *,
        mode: Optional[str] = None,
        quiz_id: str | int,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        backend = self._resolve_backend(mode)
        service = self._service_for(backend)
        result = await self._maybe_await(
            service.delete_quiz(
                quiz_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )
        if isinstance(result, Mapping):
            return str(result.get("status") or "").strip().lower() == "deleted"
        return bool(result)

    async def list_questions(
        self,
        *,
        mode: Optional[str] = None,
        quiz_id: str | int,
        q: Optional[str] = None,
        include_answers: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        backend = self._resolve_backend(mode)
        service = self._service_for(backend)
        normalized_q = str(q or "").strip() or None
        records = await self._maybe_await(
            service.list_questions(
                quiz_id,
                q=normalized_q,
                include_answers=include_answers,
                limit=limit,
                offset=offset,
            )
        )
        items = list((records or {}).get("items") or records or [])
        return [normalize_quiz_question_record(backend.value, record) for record in items]

    async def create_question(
        self,
        *,
        mode: Optional[str] = None,
        quiz_id: str | int,
        **payload: Any,
    ) -> dict[str, Any]:
        backend = self._resolve_backend(mode)
        service = self._service_for(backend)
        result = await self._maybe_await(service.create_question(quiz_id, **payload))
        return normalize_quiz_question_record(backend.value, result)

    async def delete_question(
        self,
        *,
        mode: Optional[str] = None,
        question_id: str | int,
        quiz_id: Optional[str | int] = None,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        backend = self._resolve_backend(mode)
        service = self._service_for(backend)
        kwargs: dict[str, Any] = {
            "expected_version": expected_version,
            "hard_delete": hard_delete,
        }
        if quiz_id is not None:
            kwargs["quiz_id"] = quiz_id
        result = await self._maybe_await(service.delete_question(question_id, **kwargs))
        if isinstance(result, Mapping):
            return str(result.get("status") or "").strip().lower() == "deleted"
        return bool(result)

    async def start_attempt(
        self,
        *,
        mode: Optional[str] = None,
        scope_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        quiz_id: str | int,
    ) -> dict[str, Any]:
        backend = self._resolve_backend(mode)
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if backend is QuizBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        service = self._service_for(backend)
        result = await self._maybe_await(service.start_attempt(quiz_id))
        return normalize_quiz_attempt_record(backend.value, result)

    async def submit_attempt(
        self,
        *,
        mode: Optional[str] = None,
        scope_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        attempt_id: str | int,
        answers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        backend = self._resolve_backend(mode)
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if backend is QuizBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        service = self._service_for(backend)
        result = await self._maybe_await(service.submit_attempt(attempt_id, answers=answers))
        return normalize_quiz_attempt_record(backend.value, result)

    async def list_attempts(
        self,
        *,
        mode: Optional[str] = None,
        scope_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        quiz_id: Optional[str | int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        backend = self._resolve_backend(mode)
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if backend is QuizBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        service = self._service_for(backend)
        records = await self._maybe_await(service.list_attempts(quiz_id=quiz_id, limit=limit, offset=offset))
        items = list((records or {}).get("items") or records or [])
        return [normalize_quiz_attempt_record(backend.value, record) for record in items]

    async def get_attempt(
        self,
        *,
        mode: Optional[str] = None,
        scope_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        attempt_id: str | int,
        include_questions: bool = False,
        include_answers: bool = False,
    ) -> dict[str, Any]:
        backend = self._resolve_backend(mode)
        normalized_scope_type, _ = self._normalize_scope(scope_type, workspace_id)
        if backend is QuizBackend.LOCAL and normalized_scope_type == "workspace":
            raise ValueError("Workspace Study is unavailable in local mode")
        service = self._service_for(backend)
        result = await self._maybe_await(
            service.get_attempt(
                attempt_id,
                include_questions=include_questions,
                include_answers=include_answers,
            )
        )
        return normalize_quiz_attempt_record(backend.value, result)
