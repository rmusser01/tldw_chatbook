"""Thin server-backed quiz service around the shared quizzes API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    QuizAttemptSubmitRequest,
    QuizCreateRequest,
    QuizQuestionCreateRequest,
)
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


class ServerQuizService:
    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ):
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: dict[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerQuizService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerQuizService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server quiz operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server quiz action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _quiz_action_id(action: str) -> str:
        return f"quiz.{action}.server"

    @staticmethod
    def _quiz_question_action_id(action: str) -> str:
        return f"quiz.question.{action}.server"

    @staticmethod
    def _quiz_attempt_action_id(action: str) -> str:
        return f"quiz.attempt.{action}.server"

    @staticmethod
    def _coerce_quiz_id(quiz_id: str | int | None) -> Optional[int]:
        if quiz_id in {None, ""}:
            return None
        return int(str(quiz_id))

    @staticmethod
    def _coerce_attempt_id(attempt_id: str | int) -> int:
        return int(str(attempt_id))

    async def list_quizzes(self, *, q: Optional[str] = None, limit: int = 100, offset: int = 0) -> Any:
        self._enforce(self._quiz_action_id("list"))
        return await self._require_client().list_quizzes(q=q, limit=limit, offset=offset)

    async def get_quiz(self, quiz_id: str | int) -> Any:
        self._enforce(self._quiz_action_id("detail"))
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
        self._enforce(self._quiz_action_id("create"))
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
        self._enforce(self._quiz_action_id("delete"))
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
        self._enforce(self._quiz_question_action_id("list"))
        return await self._require_client().list_quiz_questions(
            self._coerce_quiz_id(quiz_id),
            q=q,
            include_answers=include_answers,
            limit=limit,
            offset=offset,
        )

    async def create_question(self, quiz_id: str | int, **payload: Any) -> Any:
        self._enforce(self._quiz_question_action_id("detail"))
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
        self._enforce(self._quiz_question_action_id("detail"))
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
        self._enforce(self._quiz_attempt_action_id("create"))
        return await self._require_client().start_quiz_attempt(self._coerce_quiz_id(quiz_id))

    async def submit_attempt(self, attempt_id: str | int, *, answers: list[dict[str, Any]]) -> Any:
        self._enforce(self._quiz_attempt_action_id("observe"))
        return await self._require_client().submit_quiz_attempt(
            self._coerce_attempt_id(attempt_id),
            QuizAttemptSubmitRequest(answers=answers),
        )

    async def list_attempts(self, *, quiz_id: Optional[str | int] = None, limit: int = 100, offset: int = 0) -> Any:
        self._enforce(self._quiz_attempt_action_id("observe"))
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
        self._enforce(self._quiz_attempt_action_id("observe"))
        return await self._require_client().get_quiz_attempt(
            self._coerce_attempt_id(attempt_id),
            include_questions=include_questions,
            include_answers=include_answers,
        )
