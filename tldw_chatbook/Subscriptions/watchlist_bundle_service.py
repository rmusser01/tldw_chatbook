"""Watchlist bundle CRUD and source membership.

A watchlist is a named bundle of sources — the unit of organization and
checking, and (in a later slice) of briefing generation. Membership is
many-to-many: a source may belong to any number of watchlists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger

from ..DB.Subscriptions_DB import SubscriptionsDB


logger = logger.bind(module="WatchlistBundleService")


class WatchlistBundleService:
    """Local watchlist bundles backed by ``SubscriptionsDB``."""

    def __init__(self, db: SubscriptionsDB) -> None:
        self._db = db

    # --- Helpers ---

    @staticmethod
    def _split_tags(raw: Any) -> list[str]:
        """Parse the comma-joined tags column used by subscriptions.tags."""
        if not raw:
            return []
        return [part.strip() for part in str(raw).split(",") if part.strip()]

    @staticmethod
    def _join_tags(tags: Sequence[str] | None) -> str | None:
        """Join tags as comma-separated string. Note: tags containing commas will be
        split on round-trip due to this convention (inherited from subscriptions.tags).
        This is a known, deliberate limitation of the shared comma-joined format."""
        if not tags:
            return None
        cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
        return ",".join(cleaned) if cleaned else None

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "tags": WatchlistBundleService._split_tags(row[3]),
            "is_active": bool(row[4]),
            "sort_order": row[5],
        }

    def _unique_name(self, conn: Any, name: str, exclude_id: int | None = None) -> str:
        """Return ``name``, suffixed if it collides case-insensitively.

        Uniqueness lives here rather than in a SQL UNIQUE constraint because a
        constraint would raise mid-migration on case-variant folder values or
        OPML re-imports.
        """
        base = name.strip()
        params: list[Any] = []
        query = "SELECT LOWER(name) FROM watchlists"
        if exclude_id is not None:
            query += " WHERE id != ?"
            params.append(exclude_id)
        taken = {row[0] for row in conn.execute(query, params)}

        if base.lower() not in taken:
            return base
        suffix = 2
        while f"{base.lower()} ({suffix})" in taken:
            suffix += 1
        return f"{base} ({suffix})"

    def _get(self, conn: Any, watchlist_id: int) -> dict[str, Any]:
        row = conn.execute(
            "SELECT id, name, description, tags, is_active, sort_order "
            "FROM watchlists WHERE id = ?",
            (watchlist_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no watchlist with id {watchlist_id}")
        return self._row_to_dict(row)

    # --- CRUD ---

    def create(
        self,
        name: str,
        description: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Create a watchlist, auto-suffixing the name on collision.

        Args:
            name: Display name for the watchlist. Leading/trailing
                whitespace is stripped before storing. If a watchlist with
                the same name already exists (case-insensitively), a
                numeric suffix such as ``" (2)"`` is appended until the
                resolved name is unique.
            description: Optional free-text description.
            tags: Optional tags. Stored as a comma-joined string and split
                back into a list of stripped, non-empty strings on read.

        Returns:
            The newly created watchlist as a dict with keys ``id``,
            ``name``, ``description``, ``tags``, ``is_active``, and
            ``sort_order``.

        Raises:
            ValueError: If ``name`` is empty or whitespace-only.
        """
        if not name.strip():
            raise ValueError("watchlist name cannot be empty or whitespace-only")
        with self._db.transaction() as conn:
            resolved = self._unique_name(conn, name)
            cursor = conn.execute(
                "INSERT INTO watchlists (name, description, tags) VALUES (?, ?, ?)",
                (resolved, description, self._join_tags(tags)),
            )
            return self._get(conn, cursor.lastrowid)

    def rename(self, watchlist_id: int, name: str) -> dict[str, Any]:
        """Rename a watchlist, auto-suffixing on collision with another row.

        Args:
            watchlist_id: id of the watchlist to rename.
            name: New display name. Leading/trailing whitespace is
                stripped before storing. If another watchlist already has
                this name (case-insensitively), a numeric suffix such as
                ``" (2)"`` is appended until the resolved name is unique.
                The watchlist being renamed is excluded from that check,
                so renaming to its own current name (in any case) is a
                no-op, not a collision.

        Returns:
            The updated watchlist as a dict (see :meth:`create` for the
            shape).

        Raises:
            ValueError: If ``name`` is empty or whitespace-only.
            KeyError: If no watchlist with ``watchlist_id`` exists.
        """
        if not name.strip():
            raise ValueError("watchlist name cannot be empty or whitespace-only")
        with self._db.transaction() as conn:
            resolved = self._unique_name(conn, name, exclude_id=watchlist_id)
            conn.execute(
                "UPDATE watchlists SET name = ? WHERE id = ?", (resolved, watchlist_id)
            )
            return self._get(conn, watchlist_id)

    def delete(self, watchlist_id: int) -> None:
        """Delete a watchlist. Membership cascades; sources are untouched.

        Args:
            watchlist_id: id of the watchlist to delete. Deleting an id
                that does not exist is a no-op.

        Returns:
            None.
        """
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))

    def list_watchlists(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """All watchlists in display order.

        Args:
            limit: Maximum number of watchlists to return.
            offset: Number of leading watchlists (in display order) to skip.

        Returns:
            Watchlist dicts ordered by ``sort_order`` then case-insensitive
            name.
        """
        rows = self._db.conn.execute(
            "SELECT id, name, description, tags, is_active, sort_order "
            "FROM watchlists ORDER BY sort_order, LOWER(name) "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # --- Membership ---

    def add_source(self, watchlist_id: int, subscription_id: int) -> None:
        """Add a source to a watchlist. Idempotent.

        Args:
            watchlist_id: id of the watchlist to add the source to.
            subscription_id: id of the subscription to add.

        Returns:
            None.

        Raises:
            sqlite3.IntegrityError: If ``watchlist_id`` or
                ``subscription_id`` does not reference an existing row
                (both columns carry a ``FOREIGN KEY`` constraint).
        """
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist_sources (watchlist_id, subscription_id) "
                "VALUES (?, ?)",
                (watchlist_id, subscription_id),
            )

    def remove_source(self, watchlist_id: int, subscription_id: int) -> None:
        """Remove a source from a watchlist. The source itself survives.

        Args:
            watchlist_id: id of the watchlist to remove the source from.
            subscription_id: id of the subscription to remove. Removing a
                membership that does not exist is a no-op.

        Returns:
            None.
        """
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM watchlist_sources "
                "WHERE watchlist_id = ? AND subscription_id = ?",
                (watchlist_id, subscription_id),
            )

    def list_sources(self, watchlist_id: int) -> list[int]:
        """Subscription ids belonging to a watchlist.

        Args:
            watchlist_id: id of the watchlist to list sources for. An id
                that does not exist, or one with no sources, both yield an
                empty list -- the two cases are not distinguished.

        Returns:
            Subscription ids ordered by when they were added to the
            watchlist, then by id.
        """
        rows = self._db.conn.execute(
            "SELECT subscription_id FROM watchlist_sources "
            "WHERE watchlist_id = ? ORDER BY added_at, subscription_id",
            (watchlist_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def list_source_rows(self, watchlist_id: int) -> list[dict[str, Any]]:
        """Sources in a watchlist, with the fields a tree row needs.

        ``list_sources`` returns bare ids; resolving each to a name would be
        one query per source inside a render. This joins instead, so expanding
        a watchlist costs exactly one query no matter how many sources it has.

        Args:
            watchlist_id: The watchlist whose sources to list.

        Returns:
            One dict per source with ``id``, ``name`` and ``type``, in the
            order the sources were added.
        """
        rows = self._db.conn.execute(
            """
            SELECT s.id, s.name, s.type
            FROM watchlist_sources ws
            JOIN subscriptions s ON s.id = ws.subscription_id
            WHERE ws.watchlist_id = ?
            ORDER BY ws.added_at, s.id
            """,
            (watchlist_id,),
        ).fetchall()
        return [{"id": row[0], "name": row[1], "type": row[2]} for row in rows]

    def list_all_source_rows(self) -> list[dict[str, Any]]:
        """Every source, in the shape the tree and Feeds region render.

        One statement, not a fan-out: the "all sources" scope must cost the
        same one query regardless of how many sources exist, the same
        reasoning `list_source_rows` documents for a single watchlist.

        Returns:
            One dict per source with ``id``, ``name`` and ``type``, ordered
            case-insensitively by name then id.
        """
        rows = self._db.conn.execute(
            "SELECT id, name, type FROM subscriptions ORDER BY LOWER(name), id"
        ).fetchall()
        return [{"id": row[0], "name": row[1], "type": row[2]} for row in rows]

    def list_unassigned_source_rows(self) -> list[dict[str, Any]]:
        """Sources belonging to no watchlist.

        These are otherwise unreachable from a watchlist-only tree, which is
        why the tree carries a permanent Unassigned root and the Feeds
        region needs its own resolver for this scope.

        Returns:
            One dict per unassigned source with ``id``, ``name`` and
            ``type``, ordered case-insensitively by name then id.
        """
        rows = self._db.conn.execute(
            """
            SELECT s.id, s.name, s.type
            FROM subscriptions s
            WHERE NOT EXISTS (
                SELECT 1 FROM watchlist_sources ws WHERE ws.subscription_id = s.id
            )
            ORDER BY LOWER(s.name), s.id
            """
        ).fetchall()
        return [{"id": row[0], "name": row[1], "type": row[2]} for row in rows]

    def get_watchlist_item_counts(self) -> dict[int, dict[str, int]]:
        """Item totals and unread counts for every watchlists tree node.

        Thin delegation to ``SubscriptionsDB.get_watchlist_item_counts()`` (a
        single query returning every bucket) so the tree's loader can reach
        both of its inputs -- this and ``list_watchlists()`` -- through this
        service alone, the same as ``list_source_rows`` above, rather than
        needing a second accessor to ``SubscriptionsDB`` directly.

        Returns:
            Mapping of bucket id to ``{"total": int, "unread": int}``.
        """
        return self._db.get_watchlist_item_counts()
