"""Migration v0 -> v1: add the run-lease columns to ``research_runs``.

The columns themselves pre-date this migration (task-18060 landed them as
idempotent startup ALTERs); this migration rehomes them in the versioned
path Qodo PR-1822 finding 7 asked for, so the change is auditable and
ordered like ``TTS/migrations/`` and the numbered SQL migrations under
``DB/migrations/``.

The ALTERs stay guarded on ``PRAGMA table_info`` inside the migration: a
database created by the interim unversioned code already has the columns
while still reading ``user_version = 0``, and upgrading it must stamp the
version without re-adding anything.
"""

from __future__ import annotations

import sqlite3

TARGET_VERSION = 1

#: Columns this migration adds to research_runs. Single source of truth:
#: the base CREATE TABLE in local_research_service.py ships the PRE-v1
#: shape, so both fresh and pre-existing databases pass through here.
LEASE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("lease_owner", "TEXT"),
    ("lease_id", "TEXT"),
    ("leased_until", "TEXT"),
    ("lease_attempts", "INTEGER NOT NULL DEFAULT 0"),
)


def apply(conn: sqlite3.Connection) -> None:
    """Upgrade the database to schema version 1.

    Args:
        conn: An open connection; the caller owns the transaction.
    """
    existing = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(research_runs)").fetchall()
    }
    for column, declaration in LEASE_COLUMNS:
        if column not in existing:
            conn.execute(
                f"ALTER TABLE research_runs ADD COLUMN {column} {declaration}"
            )
    conn.execute(f"PRAGMA user_version = {TARGET_VERSION}")
