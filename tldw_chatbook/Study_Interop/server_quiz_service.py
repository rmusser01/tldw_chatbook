"""Thin server-backed quiz service around the shared quizzes API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import (
    QuizAttemptSubmitRequest,
    QuizCreateRequest,
    QuizQuestionCreateRequest,
    TLDWAPIClient,
)


class ServerQuizService:
    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: dict[str, Any]) -> "ServerQuizService":
        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server quiz operations.")
        return self.client

    @staticmethod
    def _coerce_quiz_id(quiz_id: str | int | None) -> Optional[int]:
        if quiz_id in {None, ""}:
            return None
        return int(str(quiz_id))

    @staticmethod
    def _coerce_attempt_id(attempt_id: str | int) -> int:
        return int(str(attempt_id))

    async def list_quizzes(self, *, q: Optional[str] = None, limit: int = 100, offset: int = 0) -> Any:
        return await self._require_client().list_quizzes(q=q, limit=limit, offset=offset)

    async def get_quiz(self, quiz_id: str | int) -> Any:
        return await self._require_client().get_quiz(self._coerce_quiz_id(quiz_id))

    async def create_quiz(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        time_limit_seconds: Optional[int] = None,
        passing_score: Optional[int] = None,
    ) -> Any:
        return await self._require_client().create_quiz(
            QuizCreateRequest(
                name=name,
                description=description,
                workspace_id=workspace_id,
                time_limit_seconds=time_limit_seconds,
                passing_score=passing_score,
            )
        )

    async def delete_quiz(
        self,
        quiz_id: str | int,
        *,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        result = await self._require_client().delete_quiz(
            self._coerce_quiz_id(quiz_id),
            expected_version=expected_version,
            hard=hard_delete,
        )
        if isinstance(result, Mapping):
            return str(result.get("status") or "").strip().lower() == "deleted"
        return bool(result)

    async def list_questions(
        self,
        quiz_id: str | int,
        *,
        q: Optional[str] = None,
        include_answers: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        return await self._require_client().list_quiz_questions(
            self._coerce_quiz_id(quiz_id),
            q=q,
            include_answers=include_answers,
            limit=limit,
            offset=offset,
        )

    async def create_question(self, quiz_id: str | int, **payload: Any) -> Any:
        return await self._require_client().create_quiz_question(
            self._coerce_quiz_id(quiz_id),
            QuizQuestionCreateRequest(**payload),
        )

    async def delete_question(
        self,
        question_id: str | int,
        *,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
        quiz_id: Optional[str | int] = None,
    ) -> bool:
        if quiz_id in {None, ""}:
            raise ValueError("quiz_id is required for server quiz question deletion.")
        result = await self._require_client().delete_quiz_question(
            self._coerce_quiz_id(quiz_id),
            int(str(question_id)),
            expected_version=expected_version,
            hard=hard_delete,
        )
        if isinstance(result, Mapping):
            return str(result.get("status") or "").strip().lower() == "deleted"
        return bool(result)

    async def start_attempt(self, quiz_id: str | int) -> Any:
        return await self._require_client().start_quiz_attempt(self._coerce_quiz_id(quiz_id))

    async def submit_attempt(self, attempt_id: str | int, *, answers: list[dict[str, Any]]) -> Any:
        return await self._require_client().submit_quiz_attempt(
            self._coerce_attempt_id(attempt_id),
            QuizAttemptSubmitRequest(answers=answers),
        )

    async def list_attempts(self, *, quiz_id: Optional[str | int] = None, limit: int = 100, offset: int = 0) -> Any:
        return await self._require_client().list_quiz_attempts(
            quiz_id=self._coerce_quiz_id(quiz_id),
            limit=limit,
            offset=offset,
        )

    async def get_attempt(
        self,
        attempt_id: str | int,
        *,
        include_questions: bool = False,
        include_answers: bool = False,
    ) -> Any:
        return await self._require_client().get_quiz_attempt(
            self._coerce_attempt_id(attempt_id),
            include_questions=include_questions,
            include_answers=include_answers,
        )
