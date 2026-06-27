"""Local-only conversation organization marks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ConversationLocalMark:
    conversation_id: str
    mark_type: str
    created_at: str
    updated_at: str


class ConversationLocalMarksService:
    """Manage durable local-only marks for conversations."""

    STARRED = "starred"
    _ALLOWED_MARK_TYPES = frozenset({STARRED})

    def __init__(self, db: Any):
        self.db = db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def _mark_type(cls, mark_type: str | None) -> str:
        normalized = cls.STARRED if mark_type is None else str(mark_type).strip().lower()
        if not normalized or normalized not in cls._ALLOWED_MARK_TYPES:
            raise ValueError(f"Unsupported conversation mark_type: {mark_type!r}")
        return normalized

    @staticmethod
    def _conversation_id(conversation_id: str) -> str:
        normalized = str(conversation_id or "").strip()
        if not normalized:
            raise ValueError("conversation_id is required")
        return normalized

    def star_conversation(self, conversation_id: str) -> None:
        self.set_mark(conversation_id, self.STARRED)

    def unstar_conversation(self, conversation_id: str) -> None:
        self.clear_mark(conversation_id, self.STARRED)

    def is_starred(self, conversation_id: str) -> bool:
        return self.has_mark(conversation_id, self.STARRED)

    def set_mark(self, conversation_id: str, mark_type: str | None = None) -> None:
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        now = self._now()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO conversation_local_marks (
                    conversation_id, mark_type, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id, mark_type)
                DO UPDATE SET updated_at = excluded.updated_at
                """,
                (conversation_id, mark_type, now, now),
            )

    def clear_mark(self, conversation_id: str, mark_type: str | None = None) -> None:
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        with self.db.transaction() as conn:
            conn.execute(
                """
                DELETE FROM conversation_local_marks
                 WHERE conversation_id = ? AND mark_type = ?
                """,
                (conversation_id, mark_type),
            )

    def has_mark(self, conversation_id: str, mark_type: str | None = None) -> bool:
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        row = self.db.get_connection().execute(
            """
            SELECT 1
              FROM conversation_local_marks
             WHERE conversation_id = ? AND mark_type = ?
             LIMIT 1
            """,
            (conversation_id, mark_type),
        ).fetchone()
        return row is not None

    def list_marked_conversation_ids(
        self,
        mark_type: str | None = None,
        *,
        limit: int = 100,
    ) -> tuple[str, ...]:
        mark_type = self._mark_type(mark_type)
        safe_limit = int(limit)
        if safe_limit <= 0:
            raise ValueError("limit must be positive")
        rows = self.db.get_connection().execute(
            """
            SELECT conversation_id
              FROM conversation_local_marks
             WHERE mark_type = ?
             ORDER BY updated_at DESC, conversation_id ASC
             LIMIT ?
            """,
            (mark_type, safe_limit),
        ).fetchall()
        return tuple(str(row["conversation_id"]) for row in rows)
