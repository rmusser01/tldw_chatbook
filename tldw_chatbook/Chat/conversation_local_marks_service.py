"""Local-only conversation organization marks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ConversationLocalMark:
    """A durable local-only organization mark for one conversation.

    Attributes:
        conversation_id: Local conversation identifier the mark belongs to.
        mark_type: Type of local mark, such as ``"starred"``.
        created_at: UTC timestamp for first creation.
        updated_at: UTC timestamp for the latest mark update.
    """

    conversation_id: str
    mark_type: str
    created_at: str
    updated_at: str


class ConversationLocalMarksService:
    """Manage durable local-only marks for conversations.

    The service stores organization metadata that should remain local to this
    client and must not be serialized into conversation sync payloads.
    """

    STARRED = "starred"
    _ALLOWED_MARK_TYPES = frozenset({STARRED})

    def __init__(self, db: Any):
        """Initialize the service.

        Args:
            db: Database object that exposes the project ``transaction()``
                context manager.
        """
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
        """Mark a conversation as starred locally.

        Args:
            conversation_id: Conversation identifier to star.

        Raises:
            ValueError: If ``conversation_id`` is blank.
        """
        self.set_mark(conversation_id, self.STARRED)

    def unstar_conversation(self, conversation_id: str) -> None:
        """Remove the local starred mark from a conversation.

        Args:
            conversation_id: Conversation identifier to unstar.

        Raises:
            ValueError: If ``conversation_id`` is blank.
        """
        self.clear_mark(conversation_id, self.STARRED)

    def is_starred(self, conversation_id: str) -> bool:
        """Return whether a conversation is locally starred.

        Args:
            conversation_id: Conversation identifier to check.

        Returns:
            True when the conversation has the local starred mark.

        Raises:
            ValueError: If ``conversation_id`` is blank.
        """
        return self.has_mark(conversation_id, self.STARRED)

    def set_mark(self, conversation_id: str, mark_type: str | None = None) -> None:
        """Create or refresh a local conversation mark.

        Args:
            conversation_id: Conversation identifier to mark.
            mark_type: Supported mark type. Defaults to ``"starred"``.

        Raises:
            ValueError: If ``conversation_id`` is blank or ``mark_type`` is
                unsupported.
        """
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
        """Remove a local conversation mark if present.

        Args:
            conversation_id: Conversation identifier to update.
            mark_type: Supported mark type. Defaults to ``"starred"``.

        Raises:
            ValueError: If ``conversation_id`` is blank or ``mark_type`` is
                unsupported.
        """
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
        """Return whether a local mark exists for a conversation.

        Args:
            conversation_id: Conversation identifier to check.
            mark_type: Supported mark type. Defaults to ``"starred"``.

        Returns:
            True when the requested mark exists.

        Raises:
            ValueError: If ``conversation_id`` is blank or ``mark_type`` is
                unsupported.
        """
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        with self.db.transaction() as conn:
            row = conn.execute(
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
        """List conversation ids carrying a local mark.

        Args:
            mark_type: Supported mark type. Defaults to ``"starred"``.
            limit: Maximum number of conversation ids to return.

        Returns:
            Conversation ids ordered by latest mark update, then id.

        Raises:
            ValueError: If ``mark_type`` is unsupported or ``limit`` is not
                positive.
        """
        mark_type = self._mark_type(mark_type)
        safe_limit = int(limit)
        if safe_limit <= 0:
            raise ValueError("limit must be positive")
        with self.db.transaction() as conn:
            rows = conn.execute(
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
