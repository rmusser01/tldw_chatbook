"""Local-only conversation organization marks."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID


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
    #: PR3a-2 Task 4: a background sub-agent completion the user has not
    #: seen yet. Set by the fleet drain consumer when a SURVIVOR settles
    #: (a child that outlived its spawning turn -- never one that finished
    #: inside it); cleared when the user views that conversation in
    #: Console, or when auto-wake delivers the result (Task 5). Local-only
    #: by design, like every mark here -- never serialized into sync
    #: payloads -- but durable across restarts, which is the whole point:
    #: the completion badge must survive the app that showed the toast.
    FLEET_UNSEEN = "fleet_unseen"
    #: Receipt publication can fail after a survivor settles. This private
    #: companion mark records that the coarse badge is fallback evidence,
    #: rather than a stale derivative safe to reconcile away.
    FLEET_RECEIPT_FALLBACK = "fleet_receipt_fallback"
    _ALLOWED_MARK_TYPES = frozenset(
        {STARRED, FLEET_UNSEEN, FLEET_RECEIPT_FALLBACK}
    )
    CONSOLE_UNSEEN_PREFIX = "console_unseen:"
    CONSOLE_TERMINAL_OUTCOME_PREFIX = "console_terminal_outcome:"

    def __init__(self, db: Any):
        """Initialize the service.

        Args:
            db: Database object that exposes the project ``transaction()``
                context manager.
        """
        self.db = db
        # task-15471: Console's conversation-browser refresh calls
        # `list_marked_conversation_ids` on the event loop from every
        # repaint path, so the answer is cached and only invalidated by
        # this service's own writers (`set_mark`/`clear_mark` -- every star
        # and fleet mark in the process goes through this instance). Guarded
        # by a `threading.Lock`, not just loop discipline: the star toggle
        # now writes from a pool thread via `asyncio.to_thread`.
        #
        # The generation counter closes the populate-after-invalidate race
        # (task-15471 review M1): a cache-missing reader holds its fetched
        # rows across the transaction COMMIT -- a GIL-releasing sqlite call
        # -- before storing them. A writer that commits and invalidates
        # inside that window bumps the generation, so the reader detects
        # its snapshot is outdated and skips the store instead of
        # resurrecting pre-write rows into the cache. Global, not
        # per-mark-type, on purpose: the cost of a false bump is one
        # skipped store, and a single counter is obviously correct.
        self._list_cache: dict[tuple[str, int], tuple[str, ...]] = {}
        self._list_cache_lock = threading.Lock()
        self._list_cache_generation = 0

    def _invalidate_list_cache(self, mark_type: str) -> None:
        """Drop cached id lists for one mark type after a write."""
        with self._list_cache_lock:
            self._list_cache_generation += 1
            for key in [k for k in self._list_cache if k[0] == mark_type]:
                del self._list_cache[key]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def _mark_type(cls, mark_type: str | None) -> str:
        raw = cls.STARRED if mark_type is None else str(mark_type).strip()
        static_mark = raw.lower()
        if static_mark in cls._ALLOWED_MARK_TYPES:
            return static_mark
        if cls.parse_console_unseen_mark_type(raw) is not None:
            return raw
        if cls.parse_console_terminal_outcome_mark_type(raw) is not None:
            return raw
        raise ValueError(f"Unsupported conversation mark_type: {mark_type!r}")

    @staticmethod
    def validate_terminal_receipt_id(receipt_id: str) -> str:
        """Validate one canonical opaque UUID receipt identifier."""
        if type(receipt_id) is not str:
            raise ValueError("terminal receipt id must be a canonical UUID")
        try:
            parsed = UUID(receipt_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("terminal receipt id must be a canonical UUID") from exc
        if str(parsed) != receipt_id:
            raise ValueError("terminal receipt id must be a canonical UUID")
        return receipt_id

    @classmethod
    def console_unseen_mark_type(cls, receipt_id: str) -> str:
        """Build the exact local mark type for one terminal receipt."""
        return cls.CONSOLE_UNSEEN_PREFIX + cls.validate_terminal_receipt_id(receipt_id)

    @classmethod
    def parse_console_unseen_mark_type(cls, mark_type: object) -> str | None:
        """Return the exact receipt ID from a valid namespaced mark."""
        if type(mark_type) is not str or not mark_type.startswith(
            cls.CONSOLE_UNSEEN_PREFIX
        ):
            return None
        receipt_id = mark_type[len(cls.CONSOLE_UNSEEN_PREFIX) :]
        try:
            return cls.validate_terminal_receipt_id(receipt_id)
        except ValueError:
            return None

    @staticmethod
    def validate_terminal_outcome(outcome: str) -> str:
        """Validate the content-free outcome stored beside one receipt."""
        if type(outcome) is not str or outcome not in {"complete", "failed"}:
            raise ValueError("terminal outcome must be complete or failed")
        return outcome

    @classmethod
    def console_terminal_outcome_mark_type(
        cls, receipt_id: str, outcome: str
    ) -> str:
        """Build the local-only companion mark for one exact receipt."""
        receipt_id = cls.validate_terminal_receipt_id(receipt_id)
        outcome = cls.validate_terminal_outcome(outcome)
        return f"{cls.CONSOLE_TERMINAL_OUTCOME_PREFIX}{receipt_id}:{outcome}"

    @classmethod
    def parse_console_terminal_outcome_mark_type(
        cls, mark_type: object
    ) -> tuple[str, str] | None:
        """Parse a valid content-free terminal-outcome companion mark."""
        if type(mark_type) is not str or not mark_type.startswith(
            cls.CONSOLE_TERMINAL_OUTCOME_PREFIX
        ):
            return None
        payload = mark_type[len(cls.CONSOLE_TERMINAL_OUTCOME_PREFIX) :]
        receipt_id, separator, outcome = payload.rpartition(":")
        if not separator:
            return None
        try:
            return (
                cls.validate_terminal_receipt_id(receipt_id),
                cls.validate_terminal_outcome(outcome),
            )
        except ValueError:
            return None

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
        with self.db.transaction() as cursor:
            self.set_mark_with_cursor(
                cursor,
                conversation_id,
                mark_type,
                created_at=now,
                updated_at=now,
            )
        self._invalidate_list_cache(mark_type)

    def set_mark_with_cursor(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        mark_type: str,
        *,
        created_at: str,
        updated_at: str,
    ) -> None:
        """Insert or refresh one mark using an existing DB transaction."""
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        if not created_at or not updated_at:
            raise ValueError("mark timestamps are required")
        cursor.execute(
            """
            INSERT INTO conversation_local_marks (
                conversation_id, mark_type, created_at, updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id, mark_type)
            DO UPDATE SET updated_at = excluded.updated_at
            """,
            (conversation_id, mark_type, created_at, updated_at),
        )

    def set_console_terminal_with_cursor(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        receipt_id: str,
        outcome: str,
        *,
        created_at: str,
        updated_at: str,
    ) -> None:
        """Insert an unseen receipt and its exact outcome atomically."""
        unseen = self.console_unseen_mark_type(receipt_id)
        companion = self.console_terminal_outcome_mark_type(receipt_id, outcome)
        self.set_mark_with_cursor(
            cursor,
            conversation_id,
            unseen,
            created_at=created_at,
            updated_at=updated_at,
        )
        self.set_mark_with_cursor(
            cursor,
            conversation_id,
            companion,
            created_at=created_at,
            updated_at=updated_at,
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
        self._invalidate_list_cache(mark_type)

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

    def get_mark(
        self, conversation_id: str, mark_type: str | None = None
    ) -> ConversationLocalMark | None:
        """Fetch one mark row with its timestamps, or ``None`` if absent.

        PR3a-2 Task 5: the auto-wake mount-claim uses ``created_at`` as
        the since-when boundary for "which terminal sub-agent runs are
        still undelivered" -- ``set_mark`` refreshes only ``updated_at``
        on conflict, so ``created_at`` is stable at "the first undelivered
        completion since the mark was last cleared".

        Args:
            conversation_id: Conversation identifier to look up.
            mark_type: Supported mark type. Defaults to ``"starred"``.

        Returns:
            The mark row, or ``None`` when no such mark exists.

        Raises:
            ValueError: If ``conversation_id`` is blank or ``mark_type``
                is unsupported.
        """
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        with self.db.transaction() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, mark_type, created_at, updated_at
                  FROM conversation_local_marks
                 WHERE conversation_id = ? AND mark_type = ?
                 LIMIT 1
                """,
                (conversation_id, mark_type),
            ).fetchone()
        if row is None:
            return None
        return ConversationLocalMark(
            conversation_id=str(row["conversation_id"]),
            mark_type=str(row["mark_type"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

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
        cache_key = (mark_type, safe_limit)
        with self._list_cache_lock:
            cached = self._list_cache.get(cache_key)
            generation = self._list_cache_generation
        if cached is not None:
            return cached
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
        result = tuple(str(row["conversation_id"]) for row in rows)
        with self._list_cache_lock:
            if self._list_cache_generation == generation:
                # No writer invalidated while this read was in flight, so
                # the snapshot is current and safe to cache. Otherwise the
                # rows may predate a committed write -- return them (they
                # were true when read) but never store them.
                self._list_cache[cache_key] = result
        return result

    def list_console_unseen_marks(
        self, *, limit: int = 100
    ) -> tuple[tuple[str, str], ...]:
        """List exact ``(conversation_id, receipt_id)`` pairs uncached."""
        safe_limit = int(limit)
        if safe_limit <= 0:
            raise ValueError("limit must be positive")
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT conversation_id, mark_type
                  FROM conversation_local_marks
                 WHERE mark_type LIKE ?
                 ORDER BY updated_at DESC, conversation_id ASC, mark_type ASC
                """,
                (self.CONSOLE_UNSEEN_PREFIX + "%",),
            ).fetchall()
        result: list[tuple[str, str]] = []
        for row in rows:
            receipt_id = self.parse_console_unseen_mark_type(row["mark_type"])
            if receipt_id is None:
                continue
            result.append((str(row["conversation_id"]), receipt_id))
            if len(result) >= safe_limit:
                break
        return tuple(result)

    def has_console_unseen_marks(self) -> bool:
        """Return whether any valid terminal receipt mark exists, uncached."""
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                "SELECT mark_type FROM conversation_local_marks WHERE mark_type LIKE ?",
                (self.CONSOLE_UNSEEN_PREFIX + "%",),
            ).fetchall()
        return any(
            self.parse_console_unseen_mark_type(row["mark_type"]) is not None
            for row in rows
        )

    def acknowledge_console_unseen(
        self, conversation_id: str, receipt_id: str
    ) -> bool:
        """Delete only the exact terminal receipt mark being acknowledged."""
        conversation_id = self._conversation_id(conversation_id)
        receipt_id = self.validate_terminal_receipt_id(receipt_id)
        mark_type = self.console_unseen_mark_type(receipt_id)
        companions = tuple(
            self.console_terminal_outcome_mark_type(receipt_id, outcome)
            for outcome in ("complete", "failed")
        )
        with self.db.transaction() as cursor:
            deleted = cursor.execute(
                """
                DELETE FROM conversation_local_marks
                 WHERE conversation_id = ? AND mark_type = ?
                """,
                (conversation_id, mark_type),
            )
            unseen_deleted = deleted.rowcount == 1
            cursor.execute(
                """
                DELETE FROM conversation_local_marks
                 WHERE conversation_id = ? AND mark_type IN (?, ?)
                """,
                (conversation_id, *companions),
            )
        return unseen_deleted

    def console_terminal_outcome(
        self, conversation_id: str, receipt_id: str
    ) -> str | None:
        """Return the exact local outcome companion for one unseen receipt."""
        conversation_id = self._conversation_id(conversation_id)
        receipt_id = self.validate_terminal_receipt_id(receipt_id)
        unseen = self.console_unseen_mark_type(receipt_id)
        prefix = f"{self.CONSOLE_TERMINAL_OUTCOME_PREFIX}{receipt_id}:"
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT mark_type
                  FROM conversation_local_marks
                 WHERE conversation_id = ?
                   AND (mark_type = ? OR mark_type LIKE ?)
                """,
                (conversation_id, unseen, prefix + "%"),
            ).fetchall()
        has_unseen = any(row["mark_type"] == unseen for row in rows)
        outcomes = {
            parsed[1]
            for row in rows
            if (parsed := self.parse_console_terminal_outcome_mark_type(
                row["mark_type"]
            ))
            is not None
            and parsed[0] == receipt_id
        }
        if has_unseen and len(outcomes) == 1:
            return next(iter(outcomes))
        return None
