"""Migration from schema version 3 to version 4.

Adds the local automation execution floor for the schedules-handoff
program (spec-2026-08-31-schedules-handoff-parity.md §4): the
``automation_runs`` and ``automation_results`` tables, the eight
reference-parity columns plus ``next_run_at``/``transfer_state`` on
``automation_definitions``, and ``transfer_state`` on ``reminder_tasks``.
"""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only

    class _MigrationCapableDB(Protocol):
        def _get_connection(self) -> Any: ...


_DEFINITION_COLUMNS_V4: tuple[tuple[str, str], ...] = (
    ("disabled_lock_kind", "TEXT"),
    ("disabled_reason", "TEXT"),
    ("resolution_state", "TEXT NOT NULL DEFAULT 'open'"),
    ("resolved_at", "TEXT"),
    ("resolved_by", "TEXT"),
    ("resolved_result_id", "TEXT"),
    ("finding_policy", "TEXT"),
    ("retention_policy", "TEXT"),
    ("next_run_at", "TEXT"),
    ("transfer_state", "TEXT"),
)

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS automation_runs (
    id TEXT PRIMARY KEY,
    server_id TEXT,
    owner_id TEXT NOT NULL,
    definition_id TEXT NOT NULL,
    definition_version INTEGER NOT NULL DEFAULT 1,
    trigger_reason TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'none',
    schedule_slot TEXT,
    scope_snapshot TEXT,
    finding_policy_snapshot TEXT,
    rag_request_snapshot TEXT,
    run_summary TEXT,
    evidence_summary TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    started_at TEXT,
    ended_at TEXT,
    UNIQUE (definition_id, definition_version, schedule_slot)
);
"""

_CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS automation_results (
    id TEXT PRIMARY KEY,
    server_id TEXT,
    owner_id TEXT NOT NULL,
    definition_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    answer TEXT,
    answer_mode TEXT NOT NULL DEFAULT 'none',
    confidence TEXT,
    source_refs TEXT,
    dedupe_key TEXT NOT NULL,
    visibility_destination TEXT,
    review_state TEXT NOT NULL DEFAULT 'unread',
    reviewed_at TEXT,
    reviewed_by TEXT,
    review_note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE (owner_id, dedupe_key)
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_automation_runs_owner_definition_created
    ON automation_runs (owner_id, definition_id, created_at);
CREATE INDEX IF NOT EXISTS idx_automation_runs_owner_status
    ON automation_runs (owner_id, status);
CREATE INDEX IF NOT EXISTS idx_automation_results_owner_review
    ON automation_results (owner_id, review_state, created_at);
CREATE INDEX IF NOT EXISTS idx_automation_definitions_owner_next_run
    ON automation_definitions (owner_id, next_run_at);
"""


def migrate(db: _MigrationCapableDB) -> None:
    """Apply the v3 -> v4 schema migration to ``db``. Idempotent."""
    with closing(db._get_connection()) as conn:
        existing = conn.execute(
            "PRAGMA table_info(automation_definitions)"
        ).fetchall()
        if not existing:
            # No automation_definitions on this connection: nothing to
            # migrate (same memory-correctness rule as v1_to_v2).
            return
        conn.execute(_CREATE_RUNS)
        conn.execute(_CREATE_RESULTS)

        def_cols = {row[1] for row in existing}
        for name, decl in _DEFINITION_COLUMNS_V4:
            if name not in def_cols:
                conn.execute(
                    f"ALTER TABLE automation_definitions ADD COLUMN {name} {decl}"
                )
        rem_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(reminder_tasks)")
        }
        if "transfer_state" not in rem_cols:
            conn.execute(
                "ALTER TABLE reminder_tasks ADD COLUMN transfer_state TEXT"
            )

        # Indexes last: idx_automation_definitions_owner_next_run needs
        # next_run_at, added just above.
        conn.executescript(_INDEXES)

        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version < 4:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (4,))
        conn.commit()
    logger.debug(
        "Scheduling schema migrated to version 4 (automation runs/results)"
    )


def rollback(db: _MigrationCapableDB) -> None:
    """Revert to v3: drop the new tables and the added columns.

    Column removal uses ALTER TABLE DROP COLUMN (SQLite >=3.35; the
    Python >=3.11 floor bundles it). Deviation from v2_to_v3's
    table-recreate rollback is deliberate: ten columns across two tables
    make recreate disproportionate here.
    """
    with closing(db._get_connection()) as conn:
        conn.execute("DROP TABLE IF EXISTS automation_runs")
        conn.execute("DROP TABLE IF EXISTS automation_results")
        def_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(automation_definitions)")
        }
        for name, _decl in _DEFINITION_COLUMNS_V4:
            if name in def_cols:
                conn.execute(
                    f"ALTER TABLE automation_definitions DROP COLUMN {name}"
                )
        rem_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(reminder_tasks)")
        }
        if "transfer_state" in rem_cols:
            conn.execute("ALTER TABLE reminder_tasks DROP COLUMN transfer_state")
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (3,))
        conn.commit()
    logger.debug("Scheduling schema rolled back to version 3")
