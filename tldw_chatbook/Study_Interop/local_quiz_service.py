"""Thin local study wrapper around ChaChaNotes_DB quiz helpers."""

from __future__ import annotations

from typing import Any, Optional


class LocalQuizService:
    """Thin sync wrapper around local quiz helpers."""

    def __init__(self, db: Any):
        self.db = db

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local quiz backend is unavailable.")
        return self.db

    def list_quizzes(self, *, q: Optional[str] = None, limit: int = 100, offset: int = 0) -> Any:
        normalized_q = str(q or "").strip() or None
        return self._require_db().list_quizzes(q=normalized_q, limit=limit, offset=offset)

    def get_quiz(self, quiz_id: str) -> Any:
        return self._require_db().get_quiz(quiz_id)

    def create_quiz(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        workspace_id: Optional[str] = None,
        time_limit_seconds: Optional[int] = None,
        passing_score: Optional[int] = None,
    ) -> Any:
        quiz_id = self._require_db().create_quiz(
            name=name,
            description=description,
            workspace_id=workspace_id,
            time_limit_seconds=time_limit_seconds,
            passing_score=passing_score,
        )
        return self._require_db().get_quiz(quiz_id)

    def delete_quiz(
        self,
        quiz_id: str,
        *,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        return bool(
            self._require_db().delete_quiz(
                quiz_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )

    def list_questions(
        self,
        quiz_id: str,
        *,
        q: Optional[str] = None,
        include_answers: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        normalized_q = str(q or "").strip() or None
        return self._require_db().list_questions(
            quiz_id,
            q=normalized_q,
            include_answers=include_answers,
            limit=limit,
            offset=offset,
        )

    def create_question(self, quiz_id: str, **payload: Any) -> Any:
        question_id = self._require_db().create_question(quiz_id=quiz_id, **payload)
        return self._require_db().get_question(question_id)

    def delete_question(
        self,
        question_id: str,
        *,
        quiz_id: Optional[str] = None,
        expected_version: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        return bool(
            self._require_db().delete_question(
                question_id,
                expected_version=expected_version,
                hard_delete=hard_delete,
            )
        )

    def start_attempt(self, quiz_id: str) -> Any:
        return self._require_db().start_attempt(quiz_id)

    def submit_attempt(self, attempt_id: str, *, answers: list[dict[str, Any]]) -> Any:
        return self._require_db().submit_attempt(attempt_id, answers)

    def list_attempts(self, *, quiz_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> Any:
        return self._require_db().list_attempts(quiz_id=quiz_id, limit=limit, offset=offset)

    def get_attempt(
        self,
        attempt_id: str,
        *,
        include_questions: bool = False,
        include_answers: bool = False,
    ) -> Any:
        return self._require_db().get_attempt(
            attempt_id,
            include_questions=include_questions,
            include_answers=include_answers,
        )
