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
        """Create a watchlist, auto-suffixing the name on collision."""
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
        """Rename a watchlist, auto-suffixing on collision with another row."""
        if not name.strip():
            raise ValueError("watchlist name cannot be empty or whitespace-only")
        with self._db.transaction() as conn:
            resolved = self._unique_name(conn, name, exclude_id=watchlist_id)
            conn.execute(
                "UPDATE watchlists SET name = ? WHERE id = ?", (resolved, watchlist_id)
            )
            return self._get(conn, watchlist_id)

    def delete(self, watchlist_id: int) -> None:
        """Delete a watchlist. Membership cascades; sources are untouched."""
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))

    def list_watchlists(self) -> list[dict[str, Any]]:
        """All watchlists in display order."""
        rows = self._db.conn.execute(
            "SELECT id, name, description, tags, is_active, sort_order "
            "FROM watchlists ORDER BY sort_order, LOWER(name)"
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # --- Membership ---

    def add_source(self, watchlist_id: int, subscription_id: int) -> None:
        """Add a source to a watchlist. Idempotent."""
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist_sources (watchlist_id, subscription_id) "
                "VALUES (?, ?)",
                (watchlist_id, subscription_id),
            )

    def remove_source(self, watchlist_id: int, subscription_id: int) -> None:
        """Remove a source from a watchlist. The source itself survives."""
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM watchlist_sources "
                "WHERE watchlist_id = ? AND subscription_id = ?",
                (watchlist_id, subscription_id),
            )

    def list_sources(self, watchlist_id: int) -> list[int]:
        """Subscription ids belonging to a watchlist."""
        rows = self._db.conn.execute(
            "SELECT subscription_id FROM watchlist_sources "
            "WHERE watchlist_id = ? ORDER BY added_at, subscription_id",
            (watchlist_id,),
        ).fetchall()
        return [row[0] for row in rows]

    # --- Migration ---

    MIGRATION_KEY = "folders_to_watchlists"

    def migrate_folders(self) -> bool:
        """Turn distinct ``subscriptions.folder`` values into watchlists.

        Sources with no folder join a single ``Unsorted`` watchlist. The
        ``folder`` column is left in place and untouched, so this is reversible.

        In practice this migrates almost nothing: no live code path writes
        ``folder``. It exists for hand-seeded databases.

        Returns:
            ``True`` if the migration ran, ``False`` if it had already been
            applied.
        """
        with self._db.transaction() as conn:
            already = conn.execute(
                "SELECT 1 FROM watchlist_migration_state WHERE key = ?",
                (self.MIGRATION_KEY,),
            ).fetchone()
            if already:
                return False

            rows = conn.execute(
                "SELECT id, folder FROM subscriptions ORDER BY id"
            ).fetchall()

            buckets: dict[str, list[int]] = {}
            for subscription_id, folder in rows:
                label = (folder or "").strip() or "Unsorted"
                buckets.setdefault(label, []).append(subscription_id)

            for label, source_ids in buckets.items():
                resolved = self._unique_name(conn, label)
                cursor = conn.execute(
                    "INSERT INTO watchlists (name) VALUES (?)", (resolved,)
                )
                watchlist_id = cursor.lastrowid
                conn.executemany(
                    "INSERT OR IGNORE INTO watchlist_sources "
                    "(watchlist_id, subscription_id) VALUES (?, ?)",
                    [(watchlist_id, source_id) for source_id in source_ids],
                )

            conn.execute(
                "INSERT INTO watchlist_migration_state (key) VALUES (?)",
                (self.MIGRATION_KEY,),
            )
            logger.info(
                "Migrated {} folder group(s) into watchlists.", len(buckets)
            )
            return True
