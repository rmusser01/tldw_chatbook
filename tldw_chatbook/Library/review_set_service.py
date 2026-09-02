"""Persistence + navigation for Library review sets (task-28240).

The impure seam over the pure model in ``review_set_state``: it owns the
``review_sets`` / ``review_set_items`` tables (schema v4 of
``Library_Collections_DB``) and enforces the "one active set" invariant
transactionally -- there is deliberately no partial unique index (see the DB
module). Tombstone-aware navigation is delegated to the pure functions; the
caller injects an ``is_live`` predicate (a resolve against the Media DB) so this
layer never imports the Media DB.

See backlog/docs/design-library-review-sets.md.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence
from uuid import uuid4

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.review_set_state import (
    IsLive,
    ReviewSet,
    ReviewSetItem,
    advance_cursor,
    is_complete,
)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp (``...Z``), matching the sibling DBs."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class ReviewSetService:
    """CRUD + cursor persistence for review sets."""

    def __init__(
        self,
        db: LibraryCollectionsDB,
        *,
        id_factory: Callable[[], str] | None = None,
        now: Callable[[], str] | None = None,
    ) -> None:
        """Wire the service to a collections DB.

        Args:
            db: The ``LibraryCollectionsDB`` owning the v4 review-set tables.
            id_factory: Injectable set-id generator (tests pin it); defaults to
                a uuid-based factory.
            now: Injectable clock returning an ISO timestamp (tests pin it).
        """
        self._db = db
        self._id_factory = id_factory or (lambda: f"reviewset-{uuid4().hex}")
        self._now = now or _utc_now

    # -- creation -------------------------------------------------------------

    def create_review_set(
        self,
        name: str,
        origin: str,
        items: Iterable[tuple[int, str]],
    ) -> str:
        """Pin an ordered snapshot and make it the active set.

        Items are ``(backing_media_id, title_snapshot)`` pairs in review order;
        duplicates by backing id are dropped keeping the first occurrence, and
        positions are assigned densely from 0. The previously active set (if
        any) is deactivated in the same transaction, upholding the single-active
        invariant. The new set opens at cursor 0.

        Args:
            name: Human label for the set.
            origin: Provenance tag (``'browse'`` | ``'selection'`` |
                ``'read_later'``).
            items: Ordered ``(backing_media_id, title_snapshot)`` pairs.

        Returns:
            The new set's id.
        """
        pinned: list[tuple[int, str]] = []
        seen: set[int] = set()
        for backing_media_id, title in items:
            if backing_media_id in seen:
                continue
            seen.add(backing_media_id)
            pinned.append((int(backing_media_id), str(title)))

        set_id = self._id_factory()
        timestamp = self._now()
        with self._db.transaction() as conn:
            self._deactivate_all(conn, timestamp)
            conn.execute(
                "INSERT INTO review_sets("
                "set_id, name, origin, cursor, active, completed_at, "
                "created_at, updated_at, deleted_at) "
                "VALUES(?, ?, ?, 0, 1, NULL, ?, ?, NULL)",
                (set_id, name, origin, timestamp, timestamp),
            )
            conn.executemany(
                "INSERT INTO review_set_items("
                "set_id, position, backing_media_id, title_snapshot, done, done_at) "
                "VALUES(?, ?, ?, ?, 0, NULL)",
                [
                    (set_id, position, backing_media_id, title)
                    for position, (backing_media_id, title) in enumerate(pinned)
                ],
            )
        return set_id

    # -- reads ----------------------------------------------------------------

    def get_review_set(self, set_id: str) -> ReviewSet | None:
        """Return the (non-dismissed) set and its items, or ``None``."""
        with self._db.read_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM review_sets "
                "WHERE set_id = ? AND deleted_at IS NULL",
                (set_id,),
            ).fetchone()
            if row is None:
                return None
            item_rows = conn.execute(
                "SELECT * FROM review_set_items WHERE set_id = ? ORDER BY position",
                (set_id,),
            ).fetchall()
        return self._to_review_set(row, item_rows)

    def get_active_review_set(self) -> ReviewSet | None:
        """Return the one active set, or ``None`` when none is active."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT set_id FROM review_sets "
                "WHERE active = 1 AND deleted_at IS NULL",
            ).fetchone()
        return self.get_review_set(row["set_id"]) if row is not None else None

    def list_review_sets(self) -> tuple[ReviewSet, ...]:
        """Return every non-dismissed set, newest activity first."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT set_id FROM review_sets "
                "WHERE deleted_at IS NULL ORDER BY updated_at DESC, set_id",
            ).fetchall()
        loaded = (self.get_review_set(row["set_id"]) for row in rows)
        return tuple(review_set for review_set in loaded if review_set is not None)

    # -- navigation + marks ---------------------------------------------------

    def advance(self, set_id: str, step: int, is_live: IsLive) -> int:
        """Move the cursor one live item forward/back, persist it, and return it.

        Tombstones are skipped (delegated to the pure model). A missing or
        dismissed set leaves nothing to move and returns 0.

        Args:
            set_id: The set to advance.
            step: ``+1`` for Next, ``-1`` for Prev.
            is_live: Media-DB liveness predicate.

        Returns:
            The new absolute cursor position.
        """
        review_set = self.get_review_set(set_id)
        if review_set is None:
            return 0
        new_cursor = advance_cursor(
            review_set.items, review_set.cursor, step, is_live
        )
        self._set_cursor(set_id, new_cursor)
        return new_cursor

    def set_cursor(self, set_id: str, cursor: int) -> None:
        """Persist an absolute cursor position (used by picker jumps)."""
        self._set_cursor(set_id, int(cursor))

    def mark_item_done(
        self, set_id: str, backing_media_id: int, done: bool
    ) -> None:
        """Set or clear an item's done mark by its backing media id."""
        timestamp = self._now()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE review_set_items SET done = ?, done_at = ? "
                "WHERE set_id = ? AND backing_media_id = ?",
                (1 if done else 0, timestamp if done else None,
                 set_id, int(backing_media_id)),
            )
            conn.execute(
                "UPDATE review_sets SET updated_at = ? WHERE set_id = ?",
                (timestamp, set_id),
            )

    def refresh_completion(self, set_id: str, is_live: IsLive) -> bool:
        """Recompute completion over live items and stamp/clear ``completed_at``.

        A set is complete only when it has at least one live item and every live
        item is done; an all-tombstoned set is never complete. Returns the
        current completion state.

        Args:
            set_id: The set to evaluate.
            is_live: Media-DB liveness predicate.

        Returns:
            ``True`` when the set is now complete.
        """
        review_set = self.get_review_set(set_id)
        if review_set is None:
            return False
        complete = is_complete(review_set.items, is_live)
        timestamp = self._now()
        with self._db.transaction() as conn:
            if complete and review_set.completed_at is None:
                conn.execute(
                    "UPDATE review_sets SET completed_at = ?, updated_at = ? "
                    "WHERE set_id = ?",
                    (timestamp, timestamp, set_id),
                )
            elif not complete and review_set.completed_at is not None:
                conn.execute(
                    "UPDATE review_sets SET completed_at = NULL, updated_at = ? "
                    "WHERE set_id = ?",
                    (timestamp, set_id),
                )
        return complete

    # -- lifecycle ------------------------------------------------------------

    def activate(self, set_id: str) -> None:
        """Make ``set_id`` the single active set (deactivating any other)."""
        timestamp = self._now()
        with self._db.transaction() as conn:
            self._deactivate_all(conn, timestamp)
            conn.execute(
                "UPDATE review_sets SET active = 1, updated_at = ? "
                "WHERE set_id = ? AND deleted_at IS NULL",
                (timestamp, set_id),
            )

    def reopen(self, set_id: str) -> None:
        """Clear a set's completion stamp, keeping its done marks."""
        timestamp = self._now()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE review_sets SET completed_at = NULL, updated_at = ? "
                "WHERE set_id = ?",
                (timestamp, set_id),
            )

    def dismiss(self, set_id: str) -> None:
        """Soft-delete a set and deactivate it."""
        timestamp = self._now()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE review_sets SET deleted_at = ?, active = 0, updated_at = ? "
                "WHERE set_id = ?",
                (timestamp, timestamp, set_id),
            )

    # -- internals ------------------------------------------------------------

    def _deactivate_all(self, conn: sqlite3.Connection, timestamp: str) -> None:
        conn.execute(
            "UPDATE review_sets SET active = 0, updated_at = ? "
            "WHERE active = 1 AND deleted_at IS NULL",
            (timestamp,),
        )

    def _set_cursor(self, set_id: str, cursor: int) -> None:
        timestamp = self._now()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE review_sets SET cursor = ?, updated_at = ? "
                "WHERE set_id = ?",
                (int(cursor), timestamp, set_id),
            )

    @staticmethod
    def _to_review_set(
        row: sqlite3.Row, item_rows: Sequence[sqlite3.Row]
    ) -> ReviewSet:
        items = tuple(
            ReviewSetItem(
                position=int(item["position"]),
                backing_media_id=int(item["backing_media_id"]),
                title_snapshot=str(item["title_snapshot"]),
                done=bool(item["done"]),
                done_at=item["done_at"],
            )
            for item in item_rows
        )
        return ReviewSet(
            set_id=str(row["set_id"]),
            name=str(row["name"]),
            origin=str(row["origin"]),
            cursor=int(row["cursor"]),
            active=bool(row["active"]),
            completed_at=row["completed_at"],
            items=items,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
