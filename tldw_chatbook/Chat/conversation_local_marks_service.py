"""Local-only conversation organization marks."""

from __future__ import annotations

import threading
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
        normalized = (
            cls.STARRED if mark_type is None else str(mark_type).strip().lower()
        )
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
        self._invalidate_list_cache(mark_type)

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
