"""Watchlist bundle CRUD and source membership.

A watchlist is a named bundle of sources — the unit of organization and
checking, and (in a later slice) of briefing generation. Membership is
many-to-many: a source may belong to any number of watchlists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from loguru import logger

from ..DB.Subscriptions_DB import SubscriptionsDB


logger = logger.bind(module="WatchlistBundleService")

CollisionPolicy = Literal["conflict", "return_existing", "auto_suffix"]


class WatchlistBundleService:
    """Local watchlist bundles backed by ``SubscriptionsDB``."""

    def __init__(self, db: SubscriptionsDB) -> None:
        self._db = db

    @property
    def db(self) -> SubscriptionsDB:
        """The store these bundles live in.

        Exposed because the app wires this service, not the database, onto
        the app instance (`app.watchlist_bundle_service`), so a caller that
        legitimately owns its own queries against the same store -- the
        Artifacts pane's briefing reads and the briefing service's writes --
        has no other honest way to reach it. Callers must not use this to
        re-implement anything this class already offers.
        """
        return self._db

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
        query = "SELECT name FROM watchlists"
        if exclude_id is not None:
            query += " WHERE id != ?"
            params.append(exclude_id)
        taken = {str(row[0]).strip().casefold() for row in conn.execute(query, params)}

        folded_base = base.casefold()
        if folded_base not in taken:
            return base
        suffix = 2
        while f"{folded_base} ({suffix})" in taken:
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
        return self.create_with_sources(
            name,
            description=description,
            tags=tags,
            source_ids=(),
            if_exists="auto_suffix",
        )["watchlist"]

    def create_with_sources(
        self,
        name: str,
        *,
        description: str | None,
        tags: Sequence[str] | None,
        source_ids: Sequence[int],
        if_exists: CollisionPolicy,
    ) -> dict[str, Any]:
        """Create or resolve one collection with atomic memberships."""
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("watchlist name cannot be empty or whitespace-only")
        if if_exists not in {"conflict", "return_existing", "auto_suffix"}:
            raise ValueError("invalid collection collision policy")
        ids = list(dict.fromkeys(source_ids))
        if len(ids) > 100:
            raise ValueError("a collection may contain at most 100 sources")

        with self._db.transaction(immediate=True) as conn:
            existing_row = next(
                (
                    row
                    for row in conn.execute(
                        "SELECT id, name, description, tags, is_active, sort_order "
                        "FROM watchlists ORDER BY id"
                    )
                    if str(row[1]).strip().casefold() == cleaned_name.casefold()
                ),
                None,
            )
            if existing_row is not None and if_exists == "conflict":
                raise ValueError("watchlist already exists")
            if existing_row is not None and if_exists == "return_existing":
                return {
                    "outcome": "existing",
                    "watchlist": self._row_to_dict(existing_row),
                    "membership_count": len(
                        conn.execute(
                            "SELECT 1 FROM watchlist_sources WHERE watchlist_id = ?",
                            (existing_row[0],),
                        ).fetchall()
                    ),
                }

            if ids:
                placeholders = ",".join("?" for _ in ids)
                found = {
                    int(row[0])
                    for row in conn.execute(
                        f"SELECT id FROM subscriptions WHERE id IN ({placeholders})",
                        ids,
                    )
                }
                missing = [source_id for source_id in ids if source_id not in found]
                if missing:
                    raise KeyError("source not found")

            resolved = (
                self._unique_name(conn, cleaned_name)
                if if_exists == "auto_suffix"
                else cleaned_name
            )
            cursor = conn.execute(
                "INSERT INTO watchlists (name, description, tags) VALUES (?, ?, ?)",
                (resolved, description, self._join_tags(tags)),
            )
            watchlist_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO watchlist_sources (watchlist_id, subscription_id) "
                "VALUES (?, ?)",
                ((watchlist_id, source_id) for source_id in ids),
            )
            return {
                "outcome": "created",
                "watchlist": self._get(conn, watchlist_id),
                "membership_count": len(ids),
            }

    def update_sources(
        self,
        watchlist_id: int,
        *,
        add_ids: Sequence[int],
        remove_ids: Sequence[int],
    ) -> dict[str, Any]:
        """Apply one validated all-or-nothing collection membership update."""
        add = list(dict.fromkeys(add_ids))
        remove = list(dict.fromkeys(remove_ids))
        if set(add) & set(remove):
            raise ValueError("a source cannot be both added and removed")
        if len(add) + len(remove) > 100:
            raise ValueError("at most 100 membership changes are allowed")

        with self._db.transaction(immediate=True) as conn:
            self._get(conn, watchlist_id)
            referenced = list(dict.fromkeys([*add, *remove]))
            if referenced:
                placeholders = ",".join("?" for _ in referenced)
                found = {
                    int(row[0])
                    for row in conn.execute(
                        f"SELECT id FROM subscriptions WHERE id IN ({placeholders})",
                        referenced,
                    )
                }
                if any(source_id not in found for source_id in referenced):
                    raise KeyError("source not found")
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO watchlist_sources "
                "(watchlist_id, subscription_id) VALUES (?, ?)",
                ((watchlist_id, source_id) for source_id in add),
            )
            added = conn.total_changes - before
            before = conn.total_changes
            conn.executemany(
                "DELETE FROM watchlist_sources "
                "WHERE watchlist_id = ? AND subscription_id = ?",
                ((watchlist_id, source_id) for source_id in remove),
            )
            removed = conn.total_changes - before
            return {
                "watchlist_id": watchlist_id,
                "added": added,
                "removed": removed,
                "membership_count": int(
                    conn.execute(
                        "SELECT COUNT(*) FROM watchlist_sources WHERE watchlist_id = ?",
                        (watchlist_id,),
                    ).fetchone()[0]
                ),
            }

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

    def get_watchlist_by_name_ci(self, name: str) -> dict[str, Any] | None:
        """Find a watchlist by stripped, Unicode-case-insensitive name.

        Args:
            name: Watchlist name to resolve.

        Returns:
            The matching watchlist dict, or ``None`` when no match exists.
        """
        def normalize(value: Any) -> str:
            return str(value or "").strip().lower()

        with self._db.transaction() as conn:
            # SQLite LOWER() is ASCII-only. Register the exact Python
            # normalization this lookup used before it moved into SQL, so
            # ADR-043 reuse still treats names such as ÄI/äi as equal.
            conn.create_function(
                "watchlist_name_key", 1, normalize, deterministic=True
            )
            row = conn.execute(
                "SELECT id, name, description, tags, is_active, sort_order "
                "FROM watchlists "
                "WHERE watchlist_name_key(name) = ? "
                "ORDER BY id LIMIT 1",
                (normalize(name),),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

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

    def list_watchlists_for_source(self, subscription_id: int) -> list[int]:
        """Watchlist ids a source belongs to.

        The mirror of :meth:`list_sources`, added so the source-first assign
        flow can ask "which watchlists is this source NOT in" with ONE query
        instead of calling ``list_sources`` once per watchlist -- the same
        one-query-regardless-of-size reasoning ``list_source_rows`` records
        for the other direction.

        Args:
            subscription_id: id of the subscription to look up. An id that
                does not exist, and one belonging to no watchlist, both
                yield an empty list -- the two cases are not distinguished.

        Returns:
            Watchlist ids ordered by when the source was added to each, then
            by watchlist id.
        """
        rows = self._db.conn.execute(
            "SELECT watchlist_id FROM watchlist_sources "
            "WHERE subscription_id = ? ORDER BY added_at, watchlist_id",
            (subscription_id,),
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
            One dict per source with ``id``, ``name``, ``type`` and ``url``
            (TASK-3604: the OPML export needs the feed address), in the
            order the sources were added.
        """
        with self._db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.name, s.type, s.source
                FROM watchlist_sources ws
                JOIN subscriptions s ON s.id = ws.subscription_id
                WHERE ws.watchlist_id = ?
                ORDER BY ws.added_at, s.id
                """,
                (watchlist_id,),
            ).fetchall()
        return [{"id": row[0], "name": row[1], "type": row[2], "url": row[3]} for row in rows]

    def list_all_source_rows(self) -> list[dict[str, Any]]:
        """Every source, in the shape the tree and the scoped summary render.

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
        why the tree carries a permanent Unassigned root and the scoped
        source readout needs its own resolver for this scope.

        Returns:
            One dict per unassigned source with ``id``, ``name``, ``type``
            and ``url`` (TASK-3604: the OPML export needs the feed
            address), ordered case-insensitively by name then id.
        """
        with self._db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.name, s.type, s.source
                FROM subscriptions s
                WHERE NOT EXISTS (
                    SELECT 1 FROM watchlist_sources ws WHERE ws.subscription_id = s.id
                )
                ORDER BY LOWER(s.name), s.id
                """
            ).fetchall()
        return [{"id": row[0], "name": row[1], "type": row[2], "url": row[3]} for row in rows]

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

    def get_source_item_counts(self) -> dict[int, dict[str, int]]:
        """Per-source {total, unread} for tree source badges.

        Thin delegation, same contract as `get_watchlist_item_counts` above.

        Returns:
            Mapping of source id to ``{"total": int, "unread": int}``;
            sources with no items are absent.
        """
        return self._db.get_source_item_counts()

    def get_flagged_items_count(self) -> int:
        """Global starred-item count, for the Starred root's badge (TASK-3072).

        Thin delegation, same contract as `get_watchlist_item_counts` above:
        the tree's loader reaches every one of its inputs through this
        service rather than holding a second accessor onto
        ``SubscriptionsDB`` directly.

        Returns:
            How many items are starred, across every source and status.
        """
        return self._db.get_flagged_items_count()

    def get_unread_items_count_since(self, since: str) -> int:
        """Unread items at/after `since`, for the Today root's badge (TASK-3791).

        Thin delegation, same contract as `get_flagged_items_count` above.

        Args:
            since: Inclusive ISO floor (the screen passes local midnight,
                UTC-shaped to match the stored dates).

        Returns:
            How many unread items carry an effective date at/after the floor.
        """
        return self._db.get_unread_items_count_since(since)
