"""Local Chatbook-owned explicit feedback store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tldw_api.feedback_schemas import ExplicitFeedbackRequest


class LocalFeedbackService:
    """Persist feedback for local/offline Chatbook conversations and RAG queries."""

    def __init__(self, *, store_path: str | Path, policy_enforcer: Any | None = None) -> None:
        self.store_path = Path(store_path).expanduser()
        self.policy_enforcer = policy_enforcer
        self._records: list[dict[str, Any]] = []
        self._next_id = 1
        self._load()

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._records = []
            self._next_id = 1
            return
        records = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return
        self._records = [dict(item) for item in records if isinstance(item, dict)]
        max_id = 0
        for record in self._records:
            raw_id = str(record.get("id") or "")
            if raw_id.startswith("local-fb-"):
                try:
                    max_id = max(max_id, int(raw_id.removeprefix("local-fb-")))
                except ValueError:
                    continue
        self._next_id = max_id + 1

    def _persist(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps({"items": self._records}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.store_path)

    @staticmethod
    def _record_view(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record["id"],
            "conversation_id": record.get("conversation_id"),
            "message_id": record.get("message_id"),
            "feedback_type": record.get("feedback_type"),
            "helpful": record.get("helpful"),
            "relevance_score": record.get("relevance_score"),
            "document_ids": list(record.get("document_ids") or []),
            "chunk_ids": list(record.get("chunk_ids") or []),
            "corpus": record.get("corpus"),
            "issues": list(record.get("issues") or []),
            "user_notes": record.get("user_notes"),
            "query": record.get("query"),
            "session_id": record.get("session_id"),
            "idempotency_key": record.get("idempotency_key"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        }

    def _find(self, feedback_id: str) -> dict[str, Any]:
        for record in self._records:
            if record.get("id") == feedback_id and not record.get("deleted"):
                return record
        raise ValueError(f"local_feedback_not_found:{feedback_id}")

    def _find_by_idempotency_key(self, idempotency_key: str | None) -> dict[str, Any] | None:
        if not idempotency_key:
            return None
        for record in self._records:
            if record.get("idempotency_key") == idempotency_key and not record.get("deleted"):
                return record
        return None

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
        self._enforce("feedback.create.local")
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
        existing = self._find_by_idempotency_key(request.idempotency_key)
        if existing is not None:
            return {"ok": True, "feedback_id": existing["id"]}

        feedback_id = f"local-fb-{self._next_id}"
        self._next_id += 1
        now = self._now()
        record = request.model_dump(mode="json", exclude_none=True)
        record.update(
            {
                "id": feedback_id,
                "created_at": now,
                "updated_at": now,
                "deleted": False,
            }
        )
        record.setdefault("document_ids", [])
        record.setdefault("chunk_ids", [])
        record.setdefault("issues", [])
        self._records.append(record)
        self._persist()
        return {"ok": True, "feedback_id": feedback_id}

    async def list_feedback(self, conversation_id: str) -> dict[str, Any]:
        self._enforce("feedback.list.local")
        records = [
            self._record_view(record)
            for record in self._records
            if not record.get("deleted") and record.get("conversation_id") == conversation_id
        ]
        return {"ok": True, "feedback": records}

    async def get_feedback(self, feedback_id: str) -> dict[str, Any]:
        self._enforce("feedback.detail.local")
        return self._record_view(self._find(feedback_id))

    async def update_feedback(
        self,
        feedback_id: str,
        *,
        issues: list[str] | None = None,
        user_notes: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("feedback.update.local")
        record = self._find(feedback_id)
        if issues is not None:
            record["issues"] = list(issues)
        if user_notes is not None:
            record["user_notes"] = user_notes
        record["updated_at"] = self._now()
        self._persist()
        response = self._record_view(record)
        response.update({"ok": True, "feedback_id": feedback_id})
        return response

    async def delete_feedback(self, feedback_id: str) -> dict[str, Any]:
        self._enforce("feedback.delete.local")
        record = self._find(feedback_id)
        record["deleted"] = True
        record["updated_at"] = self._now()
        self._persist()
        return {"ok": True, "deleted": True, "feedback_id": feedback_id}
