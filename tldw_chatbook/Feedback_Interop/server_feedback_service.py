"""Server-backed explicit feedback service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    ExplicitFeedbackRequest,
    FeedbackUpdateRequest,
    TLDWAPIClient,
)


class ServerFeedbackService:
    """Policy-gated access to server explicit feedback APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerFeedbackService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server feedback operations.")
        return self.client

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
                    user_message=getattr(decision, "user_message", None) or "Server feedback action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def submit_feedback(
        self,
        *,
        conversation_id: str | None = None,
        message_id: str | None = None,
        feedback_type: str,
        helpful: bool | None = None,
        relevance_score: int | None = None,
        document_ids: list[str] | None = None,
        chunk_ids: list[str] | None = None,
        corpus: str | None = None,
        issues: list[str] | None = None,
        user_notes: str | None = None,
        query: str | None = None,
        session_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("feedback.create.server")
        request = ExplicitFeedbackRequest(
            conversation_id=conversation_id,
            message_id=message_id,
            feedback_type=feedback_type,  # type: ignore[arg-type]
            helpful=helpful,
            relevance_score=relevance_score,
            document_ids=document_ids,
            chunk_ids=chunk_ids,
            corpus=corpus,
            issues=issues,
            user_notes=user_notes,
            query=query,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        return self._dump(await self._require_client().submit_explicit_feedback(request))

    async def list_feedback(self, conversation_id: str) -> dict[str, Any]:
        self._enforce("feedback.list.server")
        return self._dump(await self._require_client().list_feedback(conversation_id))

    async def update_feedback(
        self,
        feedback_id: str,
        *,
        issues: list[str] | None = None,
        user_notes: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("feedback.update.server")
        request = FeedbackUpdateRequest(issues=issues, user_notes=user_notes)
        return self._dump(await self._require_client().update_feedback(feedback_id, request))

    async def delete_feedback(self, feedback_id: str) -> dict[str, Any]:
        self._enforce("feedback.delete.server")
        return self._dump(await self._require_client().delete_feedback(feedback_id))
