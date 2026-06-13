"""Thin local study wrapper around ChaChaNotes_DB quiz helpers."""

from __future__ import annotations

from typing import Any, Optional


class LocalQuizService:
    """Thin sync wrapper around local quiz helpers."""

    def __init__(
        self,
        db: Any,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ):
        self.db = db
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def configure_notification_dispatch(
        self,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ) -> None:
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local quiz backend is unavailable.")
        return self.db

    def _dispatch_local_notification(
        self,
        *,
        title: str,
        message: str,
        source_entity_id: str | None,
        source_entity_kind: str,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        dispatcher = getattr(self, "notification_dispatch_service", None)
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return None
        try:
            return dispatch(
                app=getattr(self, "notification_app", None),
                category="study",
                title=title,
                message=message,
                severity=severity,
                source_backend="local",
                source_entity_id=source_entity_id,
                source_entity_kind=source_entity_kind,
                payload=payload or {},
            )
        except Exception:
            return None

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
        quiz = self._require_db().get_quiz(quiz_id)
        self._dispatch_local_notification(
            title="Local quiz created",
            message=f"Local quiz created: {quiz.get('name') or name}",
            source_entity_id=str(quiz.get("id") or quiz_id),
            source_entity_kind="study_quiz",
            payload={"action": "quiz_created", "quiz_id": str(quiz.get("id") or quiz_id)},
        )
        return quiz

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
        question = self._require_db().get_question(question_id)
        self._dispatch_local_notification(
            title="Local quiz question created",
            message=f"Local quiz question created for quiz {quiz_id}.",
            source_entity_id=str(question.get("id") or question_id),
            source_entity_kind="study_quiz_question",
            payload={
                "action": "quiz_question_created",
                "quiz_id": str(quiz_id),
                "question_id": str(question.get("id") or question_id),
            },
        )
        return question

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
        attempt = self._require_db().submit_attempt(attempt_id, answers)
        self._dispatch_local_notification(
            title="Local quiz attempt completed",
            message=f"Completed local quiz attempt {attempt_id}.",
            source_entity_id=str(attempt.get("id") or attempt_id),
            source_entity_kind="study_quiz_attempt",
            payload={
                "action": "quiz_attempt_completed",
                "attempt_id": str(attempt.get("id") or attempt_id),
                "quiz_id": str(attempt.get("quiz_id")) if attempt.get("quiz_id") is not None else None,
                "score": attempt.get("score"),
                "total_possible": attempt.get("total_possible"),
            },
        )
        return attempt

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
