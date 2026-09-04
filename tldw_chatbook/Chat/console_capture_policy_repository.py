"""Device-local per-conversation Console trace privacy policy."""

from __future__ import annotations

import sqlite3
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


class CapturePolicyReadStatus(str, Enum):
    """Whether inheritance was conclusively absent or could not be read."""

    ABSENT = "absent"
    FOUND = "found"
    UNAVAILABLE_OR_CORRUPT = "unavailable_or_corrupt"


@dataclass(frozen=True, slots=True)
class ConversationCapturePolicy:
    conversation_id: str
    detail: CaptureDetail | None
    capture_enabled: bool | None
    pii_redaction_enabled: bool | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class CapturePolicyWriteResult:
    status: CapturePolicyWriteStatus
    policy: ConversationCapturePolicy | None


@dataclass(frozen=True, slots=True)
class CapturePolicyReadResult:
    status: CapturePolicyReadStatus
    policy: ConversationCapturePolicy | None


class ConsoleCapturePolicyRepository:
    """Read and replace local capture detail without touching sync state."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def read(self, conversation_id: str) -> CapturePolicyReadResult:
        """Read one local conversation override without inheriting on failure.

        Args:
            conversation_id: Persisted conversation identifier to inspect.

        Returns:
            A typed result distinguishing a found policy, conclusive absence,
            and unavailable or corrupt storage.
        """
        if type(conversation_id) is not str or not conversation_id.strip():
            return CapturePolicyReadResult(
                CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT,
                None,
            )
        try:
            with self.db.transaction() as cursor:
                row = cursor.execute(
                    "SELECT policy.conversation_id, policy.capture_detail, "
                    "policy.capture_enabled, policy.pii_redaction_enabled, "
                    "policy.updated_at "
                    "FROM console_conversation_capture_policy AS policy "
                    "JOIN conversations AS conversation ON conversation.id = policy.conversation_id "
                    "WHERE policy.conversation_id = ? AND conversation.deleted = 0",
                    (conversation_id,),
                ).fetchone()
            if row is None:
                return CapturePolicyReadResult(CapturePolicyReadStatus.ABSENT, None)
            policy = self._from_row(row)
            if policy is None:
                return CapturePolicyReadResult(
                    CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT,
                    None,
                )
            return CapturePolicyReadResult(CapturePolicyReadStatus.FOUND, policy)
        except Exception:
            return CapturePolicyReadResult(
                CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT,
                None,
            )

    def replace(
        self, conversation_id: str, detail: CaptureDetail | None
    ) -> CapturePolicyWriteResult:
        """Replace or inherit one local conversation capture policy.

        Args:
            conversation_id: Persisted conversation identifier to mutate.
            detail: Explicit Safe or Full detail, or ``None`` to inherit.

        Returns:
            A structured status and the stored policy when one remains.
        """
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
                existing = cursor.execute(
                    "SELECT capture_enabled, pii_redaction_enabled "
                    "FROM console_conversation_capture_policy WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                capture_enabled = existing["capture_enabled"] if existing else None
                pii_enabled = existing["pii_redaction_enabled"] if existing else None
                if detail is None and capture_enabled is None and pii_enabled is None:
                    deleted = cursor.execute(
                        "DELETE FROM console_conversation_capture_policy WHERE conversation_id = ?",
                        (conversation_id,),
                    )
                    return CapturePolicyWriteResult(
                        CapturePolicyWriteStatus.DELETED
                        if deleted.rowcount
                        else CapturePolicyWriteStatus.UNCHANGED,
                        None,
                    )
                self._upsert(
                    cursor,
                    conversation_id=conversation_id,
                    detail=detail,
                    capture_enabled=self._optional_bool(capture_enabled),
                    pii_redaction_enabled=self._optional_bool(pii_enabled),
                )
                row = cursor.execute(
                    "SELECT conversation_id, capture_detail, capture_enabled, "
                    "pii_redaction_enabled, updated_at "
                    "FROM console_conversation_capture_policy WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                policy = self._from_row(row)
                if policy is None:
                    raise ValueError("stored capture policy is invalid")
                return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, policy)
        except Exception:
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)

    def replace_privacy(
        self,
        conversation_id: str,
        *,
        capture_enabled: bool | None,
        pii_redaction_enabled: bool | None,
    ) -> CapturePolicyWriteResult:
        """Replace sparse future Capture/PII overrides, preserving provenance.

        Args:
            conversation_id: Durable conversation whose local policy is replaced.
            capture_enabled: Sparse Capture override, or None to inherit.
            pii_redaction_enabled: Sparse PII override, or None to inherit.

        Returns:
            A stored/deleted/unchanged result, a missing-conversation result,
            or an unavailable result when validation or persistence fails.
        """

        if type(conversation_id) is not str or not conversation_id.strip():
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)
        if capture_enabled is not None and type(capture_enabled) is not bool:
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)
        if (
            pii_redaction_enabled is not None
            and type(pii_redaction_enabled) is not bool
        ):
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
                existing = cursor.execute(
                    "SELECT capture_detail FROM console_conversation_capture_policy "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                detail = (
                    CaptureDetail(existing["capture_detail"])
                    if existing and existing["capture_detail"] is not None
                    else None
                )
                if (
                    detail is None
                    and capture_enabled is None
                    and pii_redaction_enabled is None
                ):
                    deleted = cursor.execute(
                        "DELETE FROM console_conversation_capture_policy WHERE conversation_id = ?",
                        (conversation_id,),
                    )
                    return CapturePolicyWriteResult(
                        CapturePolicyWriteStatus.DELETED
                        if deleted.rowcount
                        else CapturePolicyWriteStatus.UNCHANGED,
                        None,
                    )
                self._upsert(
                    cursor,
                    conversation_id=conversation_id,
                    detail=detail,
                    capture_enabled=capture_enabled,
                    pii_redaction_enabled=pii_redaction_enabled,
                )
                row = cursor.execute(
                    "SELECT conversation_id, capture_detail, capture_enabled, "
                    "pii_redaction_enabled, updated_at "
                    "FROM console_conversation_capture_policy WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                policy = self._from_row(row)
                if policy is None:
                    raise ValueError("stored privacy policy is invalid")
                return CapturePolicyWriteResult(CapturePolicyWriteStatus.STORED, policy)
        except Exception:
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)

    @staticmethod
    def _upsert(
        cursor: sqlite3.Cursor,
        *,
        conversation_id: str,
        detail: CaptureDetail | None,
        capture_enabled: bool | None,
        pii_redaction_enabled: bool | None,
    ) -> None:
        cursor.execute(
            "INSERT INTO console_conversation_capture_policy "
            "(conversation_id, capture_detail, capture_enabled, "
            "pii_redaction_enabled, updated_at) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(conversation_id) DO UPDATE SET "
            "capture_detail = excluded.capture_detail, "
            "capture_enabled = excluded.capture_enabled, "
            "pii_redaction_enabled = excluded.pii_redaction_enabled, "
            "updated_at = CURRENT_TIMESTAMP",
            (
                conversation_id,
                detail.value if detail is not None else None,
                int(capture_enabled) if capture_enabled is not None else None,
                int(pii_redaction_enabled)
                if pii_redaction_enabled is not None
                else None,
            ),
        )

    @staticmethod
    def _optional_bool(value: object) -> bool | None:
        if value is None:
            return None
        if value in (0, 1):
            return bool(value)
        raise ValueError("invalid optional boolean")

    @staticmethod
    def _from_row(row: object) -> ConversationCapturePolicy | None:
        if row is None:
            return None
        try:
            conversation_id = row["conversation_id"]
            raw_detail = row["capture_detail"]
            detail = CaptureDetail(raw_detail) if raw_detail is not None else None
            capture_enabled = ConsoleCapturePolicyRepository._optional_bool(
                row["capture_enabled"]
            )
            pii_redaction_enabled = ConsoleCapturePolicyRepository._optional_bool(
                row["pii_redaction_enabled"]
            )
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
        if detail is None and capture_enabled is None and pii_redaction_enabled is None:
            return None
        return ConversationCapturePolicy(
            conversation_id,
            detail,
            capture_enabled,
            pii_redaction_enabled,
            updated_at,
        )
