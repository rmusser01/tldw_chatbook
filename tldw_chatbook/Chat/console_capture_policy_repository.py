"""Device-local per-conversation Console capture-detail policy."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class CapturePolicyWriteStatus(str, Enum):
    STORED = "stored"
    DELETED = "deleted"
    UNCHANGED = "unchanged"
    MISSING_CONVERSATION = "missing_conversation"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ConversationCapturePolicy:
    conversation_id: str
    detail: CaptureDetail
    updated_at: str


@dataclass(frozen=True, slots=True)
class CapturePolicyWriteResult:
    status: CapturePolicyWriteStatus
    policy: ConversationCapturePolicy | None


class ConsoleCapturePolicyRepository:
    """Read and replace local capture detail without touching sync state."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def read(self, conversation_id: str) -> ConversationCapturePolicy | None:
        if type(conversation_id) is not str or not conversation_id.strip():
            return None
        try:
            row = self.db.get_connection().execute(
                "SELECT policy.conversation_id, policy.capture_detail, policy.updated_at "
                "FROM console_conversation_capture_policy AS policy "
                "JOIN conversations AS conversation ON conversation.id = policy.conversation_id "
                "WHERE policy.conversation_id = ? AND conversation.deleted = 0",
                (conversation_id,),
            ).fetchone()
            return self._from_row(row)
        except Exception:
            return None

    def replace(
        self, conversation_id: str, detail: CaptureDetail | None
    ) -> CapturePolicyWriteResult:
        if type(conversation_id) is not str or not conversation_id.strip():
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)
        if detail is not None and not isinstance(detail, CaptureDetail):
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)
        try:
            with self.db.transaction(immediate=True) as cursor:
                conversation = cursor.execute(
                    "SELECT deleted FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if conversation is None or conversation["deleted"]:
                    return CapturePolicyWriteResult(
                        CapturePolicyWriteStatus.MISSING_CONVERSATION, None
                    )
                if detail is None:
                    deleted = cursor.execute(
                        "DELETE FROM console_conversation_capture_policy WHERE conversation_id = ?",
                        (conversation_id,),
                    )
                    return CapturePolicyWriteResult(
                        CapturePolicyWriteStatus.DELETED
                        if deleted.rowcount else CapturePolicyWriteStatus.UNCHANGED,
                        None,
                    )
                cursor.execute(
                    "INSERT INTO console_conversation_capture_policy "
                    "(conversation_id, capture_detail, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(conversation_id) DO UPDATE SET "
                    "capture_detail = excluded.capture_detail, updated_at = CURRENT_TIMESTAMP",
                    (conversation_id, detail.value),
                )
                row = cursor.execute(
                    "SELECT conversation_id, capture_detail, updated_at "
                    "FROM console_conversation_capture_policy WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                policy = self._from_row(row)
                if policy is None:
                    raise ValueError("stored capture policy is invalid")
                return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, policy)
        except Exception:
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)

    @staticmethod
    def _from_row(row: object) -> ConversationCapturePolicy | None:
        if row is None:
            return None
        try:
            conversation_id = row["conversation_id"]
            detail = CaptureDetail(row["capture_detail"])
            updated_at = row["updated_at"]
        except (KeyError, TypeError, ValueError):
            return None
        if (
            type(conversation_id) is not str
            or not conversation_id.strip()
            or type(updated_at) is not str
            or not updated_at.strip()
        ):
            return None
        return ConversationCapturePolicy(conversation_id, detail, updated_at)
