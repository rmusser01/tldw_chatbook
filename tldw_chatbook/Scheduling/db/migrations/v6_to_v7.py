"""Migration from schema version 6 to version 7.

schedules-handoff PR-6 task 1: a partial UNIQUE index on
``automation_results(owner_id, server_id)`` (``WHERE server_id IS NOT
NULL``) so a double-pull of the same server-mirrored result can no longer
insert two local rows for it. Partial because locally-authored rows have
``server_id IS NULL`` and must never collide with each other on that
account.

Any pre-existing duplicates (possible pre-v7, since nothing enforced this
before) are deduped first -- keep the newest row per (owner_id, server_id)
by ``updated_at`` (ties broken by ``created_at`` then ``id``, both stable)
-- or the index creation itself would fail.
"""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only

    class _MigrationCapableDB(Protocol):
        def _get_connection(self) -> Any: ...


# Keep exactly one row per (owner_id, server_id): the newest by updated_at
# (NULLs sort last), tiebroken by created_at then id -- same ROW_NUMBER()
# dedupe idiom as `prune_task_runs`. updated_at/created_at are wrapped in
# datetime() rather than compared as raw strings -- the same F7 mixed-offset
# fix `list_automation_results` applies: a server-mirrored row's timestamp
# is copied verbatim from the server payload (unenforced UTC assumption,
# see `_serialize_result_fields`), so a "+05:00"-offset string can be
# lexically greater than an actually-later "+00:00" string. This is a
# one-time DELETE -- picking the wrong "newest" here is not a display bug,
# it permanently discards the real newest row.
_DEDUPE_DUPLICATE_RESULTS = """
    DELETE FROM automation_results
    WHERE server_id IS NOT NULL
    AND id NOT IN (
        SELECT id FROM (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY owner_id, server_id
                ORDER BY datetime(updated_at) DESC, datetime(created_at) DESC, id DESC
            ) AS rn
            FROM automation_results
            WHERE server_id IS NOT NULL
        ) WHERE rn = 1
    )
"""

_CREATE_RESULTS_SERVER_ID_UNIQUE_INDEX = """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_automation_results_owner_server_id
        ON automation_results(owner_id, server_id)
        WHERE server_id IS NOT NULL
"""


def migrate(db: _MigrationCapableDB) -> None:
    """Apply the v6 -> v7 schema migration. Idempotent."""
    with closing(db._get_connection()) as conn:
        existing = conn.execute("PRAGMA table_info(automation_results)").fetchall()
        if not existing:
            return
        removed = conn.execute(_DEDUPE_DUPLICATE_RESULTS).rowcount
        if removed:
            logger.info(
                f"Scheduling v6->v7 migration: removed {removed} duplicate "
                "automation_results row(s) sharing an (owner_id, server_id) "
                "pair, keeping the newest by updated_at"
            )
        conn.execute(_CREATE_RESULTS_SERVER_ID_UNIQUE_INDEX)
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version < 7:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (7,))
        conn.commit()
    logger.debug(
        "Scheduling schema migrated to version 7 (automation_results server_id unique index)"
    )


def rollback(db: _MigrationCapableDB) -> None:
    """Drop the unique index, returning ``db`` to schema version 6."""
    with closing(db._get_connection()) as conn:
        conn.execute("DROP INDEX IF EXISTS idx_automation_results_owner_server_id")
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version >= 6:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (6,))
        conn.commit()
