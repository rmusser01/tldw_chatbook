"""Contracts for the shared private Notes sync-state schema owner."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from typing import Iterator

import pytest

from tldw_chatbook.Notes import notes_sync_state_schema as schema_module
from tldw_chatbook.Notes import note_import_receipts as receipt_module
from tldw_chatbook.Notes.note_import_execution_models import (
    ImportEffectState,
    ImportItemOutcome,
    ImportSessionState,
    approve_note_import_plan,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportBounds,
    ImportClassification,
    ImportPreviewItem,
    ImportSource,
    ImportSourceKind,
    NoteImportPlan,
    ParsedNotePayload,
    ProposedFolderMembership,
)
from tldw_chatbook.Notes.note_import_receipts import (
    EffectTransition,
    ImportEffectCategory,
    ImportReceiptTransitionError,
    NoteImportReceiptRepository,
)
from tldw_chatbook.Notes.notes_sync_state_schema import (
    NotesSyncStateSchemaError,
    notes_sync_state_transaction,
)


_V1_TABLES = {
    "import_sessions",
    "import_items",
    "import_payload_effects",
    "import_folder_effects",
    "import_membership_effects",
}
_V1_INDEXES = {
    "idx_import_items_outcome",
    "idx_import_payload_state",
    "idx_import_folder_state",
    "idx_import_membership_state",
    "idx_import_payload_target",
    "idx_import_folder_target",
    "idx_import_membership_path",
    "idx_import_folder_parent",
    "idx_import_items_target",
    "idx_import_items_source_session",
}

_V2_TABLES = {
    "sync_migration_runs",
    "sync_roots",
    "sync_bindings",
    "sync_migration_items",
}
_V2_INDEXES = {
    "idx_sync_migration_runs_state",
    "idx_sync_roots_state",
    "idx_sync_roots_legacy_source",
    "idx_sync_bindings_root_state",
    "idx_sync_bindings_live_note",
    "idx_sync_bindings_live_path_key",
    "idx_sync_migration_items_outcome",
}
_MIGRATION_ID = "00000000-0000-4000-8000-000000000097"

# Historical and fresh-oracle SQL is deliberately literal test data. It must not
# import or derive production DDL constants, or production mistakes could confirm
# themselves.
_LEGACY_V1_SQL = """
CREATE TABLE import_sessions (
    session_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL UNIQUE,
    plan_digest TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'running', 'cancelled', 'completed', 'needs_attention')),
    batch_size INTEGER NOT NULL CHECK (batch_size BETWEEN 1 AND 100),
    total_count INTEGER NOT NULL CHECK (total_count >= 0),
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    CHECK (length(session_id) BETWEEN 1 AND 256),
    CHECK (length(approval_id) = 36),
    CHECK (length(plan_digest) = 64 AND plan_digest NOT GLOB '*[^0-9a-f]*')
);
CREATE TABLE import_items (
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    source_locator_digest TEXT NOT NULL,
    selected_action TEXT NOT NULL
        CHECK (selected_action IN ('skip', 'create_new', 'update_existing')),
    outcome_count INTEGER NOT NULL CHECK (outcome_count > 0),
    outcome TEXT NOT NULL DEFAULT 'pending'
        CHECK (outcome IN ('pending', 'imported', 'updated', 'skipped', 'failed')),
    target_note_id TEXT,
    expected_version INTEGER CHECK (expected_version IS NULL OR expected_version >= 0),
    observed_version INTEGER CHECK (observed_version IS NULL OR observed_version >= 0),
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    PRIMARY KEY (session_id, item_id),
    FOREIGN KEY (session_id) REFERENCES import_sessions(session_id) ON DELETE CASCADE,
    CHECK (length(item_id) BETWEEN 1 AND 256),
    CHECK (
        length(source_locator_digest) = 64
        AND source_locator_digest NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),
    CHECK (outcome = 'failed' OR retryable = 0)
);
CREATE TABLE import_payload_effects (
    effect_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    payload_index INTEGER NOT NULL CHECK (payload_index >= 0),
    payload_digest TEXT NOT NULL,
    effect_kind TEXT NOT NULL CHECK (effect_kind IN ('create_note', 'replace_content')),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),
    target_note_id TEXT,
    expected_version INTEGER CHECK (expected_version IS NULL OR expected_version >= 0),
    observed_version INTEGER CHECK (observed_version IS NULL OR observed_version >= 0),
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    FOREIGN KEY (session_id, item_id)
        REFERENCES import_items(session_id, item_id) ON DELETE CASCADE,
    UNIQUE (session_id, item_id, payload_index, effect_kind),
    CHECK (length(effect_id) BETWEEN 1 AND 256),
    CHECK (length(payload_digest) = 64 AND payload_digest NOT GLOB '*[^0-9a-f]*'),
    CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),
    CHECK (state = 'failed' OR retryable = 0)
);
CREATE TABLE import_folder_effects (
    effect_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    folder_ordinal INTEGER NOT NULL CHECK (folder_ordinal >= 0),
    path_digest TEXT NOT NULL,
    parent_effect_id TEXT,
    effect_kind TEXT NOT NULL DEFAULT 'ensure_folder' CHECK (effect_kind = 'ensure_folder'),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),
    target_folder_id TEXT,
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    FOREIGN KEY (session_id) REFERENCES import_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_effect_id)
        REFERENCES import_folder_effects(effect_id) ON DELETE RESTRICT,
    UNIQUE (session_id, path_digest),
    UNIQUE (session_id, folder_ordinal),
    CHECK (length(effect_id) BETWEEN 1 AND 256),
    CHECK (length(path_digest) = 64 AND path_digest NOT GLOB '*[^0-9a-f]*'),
    CHECK (target_folder_id IS NULL OR length(target_folder_id) BETWEEN 1 AND 256),
    CHECK (state = 'failed' OR retryable = 0)
);
CREATE TABLE import_membership_effects (
    effect_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    payload_index INTEGER NOT NULL CHECK (payload_index >= 0),
    membership_ordinal INTEGER NOT NULL CHECK (membership_ordinal >= 0),
    folder_path_digest TEXT NOT NULL,
    effect_kind TEXT NOT NULL DEFAULT 'attach_membership'
        CHECK (effect_kind = 'attach_membership'),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),
    target_note_id TEXT,
    target_folder_id TEXT,
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    FOREIGN KEY (session_id, item_id)
        REFERENCES import_items(session_id, item_id) ON DELETE CASCADE,
    UNIQUE (session_id, item_id, payload_index, membership_ordinal),
    CHECK (length(effect_id) BETWEEN 1 AND 256),
    CHECK (
        length(folder_path_digest) = 64
        AND folder_path_digest NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),
    CHECK (target_folder_id IS NULL OR length(target_folder_id) BETWEEN 1 AND 256),
    CHECK (state = 'failed' OR retryable = 0)
);
CREATE INDEX idx_import_items_outcome ON import_items(session_id, outcome);
CREATE INDEX idx_import_payload_state ON import_payload_effects(session_id, state);
CREATE INDEX idx_import_folder_state ON import_folder_effects(session_id, state);
CREATE INDEX idx_import_membership_state ON import_membership_effects(session_id, state);
CREATE INDEX idx_import_payload_target ON import_payload_effects(session_id, target_note_id);
CREATE INDEX idx_import_folder_target ON import_folder_effects(session_id, target_folder_id);
CREATE INDEX idx_import_membership_path ON import_membership_effects(session_id, folder_path_digest, item_id);
CREATE INDEX idx_import_folder_parent ON import_folder_effects(session_id, parent_effect_id);
CREATE INDEX idx_import_items_target ON import_items(session_id, target_note_id, selected_action);
CREATE INDEX idx_import_items_source_session ON import_items(source_locator_digest, session_id, item_id);
"""

_CANONICAL_V2_SQL = """
CREATE TABLE sync_migration_runs (
    migration_id TEXT NOT NULL PRIMARY KEY
        CHECK (length(migration_id) = 36),
    source_kind TEXT NOT NULL
        CHECK (source_kind = 'legacy_notes_sync_v1'),
    source_revision_before TEXT NOT NULL
        CHECK (
            length(source_revision_before) = 64
            AND source_revision_before NOT GLOB '*[^0-9a-f]*'
        ),
    source_revision_after TEXT
        CHECK (
            source_revision_after IS NULL OR (
                length(source_revision_after) = 64
                AND source_revision_after NOT GLOB '*[^0-9a-f]*'
            )
        ),
    state TEXT NOT NULL DEFAULT 'pending_recheck'
        CHECK (state IN ('pending_recheck', 'matched_recheck', 'drifted')),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    UNIQUE (source_kind, source_revision_before),
    CHECK (
        (state = 'pending_recheck' AND source_revision_after IS NULL)
        OR (state = 'matched_recheck'
            AND source_revision_after IS NOT NULL
            AND source_revision_after = source_revision_before)
        OR (state = 'drifted'
            AND source_revision_after IS NOT NULL
            AND source_revision_after <> source_revision_before)
    )
);
CREATE TABLE sync_roots (
    root_id TEXT NOT NULL PRIMARY KEY
        CHECK (length(root_id) BETWEEN 1 AND 256),
    lexical_root_path TEXT NOT NULL
        CHECK (
            length(lexical_root_path) BETWEEN 1 AND 32768
            AND instr(lexical_root_path, char(0)) = 0
        ),
    display_name TEXT NOT NULL
        CHECK (length(display_name) BETWEEN 1 AND 255),
    direction TEXT NOT NULL
        CHECK (direction IN (
            'unspecified', 'folder_to_notes', 'notes_to_folder', 'bidirectional'
        )),
    state TEXT NOT NULL DEFAULT 'candidate'
        CHECK (state IN ('candidate', 'paused', 'disconnected')),
    row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version > 0),
    needs_rescan INTEGER NOT NULL DEFAULT 1 CHECK (needs_rescan IN (0, 1)),
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    source_kind TEXT CHECK (
        source_kind IS NULL OR source_kind = 'legacy_notes_sync_v1'
    ),
    source_locator_digest TEXT CHECK (
        source_locator_digest IS NULL OR (
            length(source_locator_digest) = 64
            AND source_locator_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    source_migration_id TEXT
        REFERENCES sync_migration_runs(migration_id) ON DELETE RESTRICT,
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    CHECK (
        (source_kind IS NULL
         AND source_locator_digest IS NULL
         AND source_migration_id IS NULL)
        OR (source_kind IS NOT NULL
            AND source_locator_digest IS NOT NULL
            AND source_migration_id IS NOT NULL)
    ),
    CHECK (
        direction <> 'unspecified' OR (
            source_kind IS 'legacy_notes_sync_v1'
            AND needs_rescan = 1
            AND reason_code IS 'legacy_direction_invalid'
        )
    )
);
CREATE TABLE sync_bindings (
    binding_id TEXT NOT NULL PRIMARY KEY
        CHECK (length(binding_id) BETWEEN 1 AND 256),
    root_id TEXT NOT NULL
        REFERENCES sync_roots(root_id) ON DELETE RESTRICT,
    note_id TEXT NOT NULL
        CHECK (length(note_id) BETWEEN 1 AND 256),
    lexical_relative_path TEXT NOT NULL
        CHECK (
            length(lexical_relative_path) BETWEEN 1 AND 32768
            AND instr(lexical_relative_path, char(0)) = 0
        ),
    path_key TEXT CHECK (
        path_key IS NULL OR (
            length(path_key) BETWEEN 1 AND 32768
            AND instr(path_key, char(0)) = 0
        )
    ),
    state TEXT NOT NULL DEFAULT 'candidate'
        CHECK (state IN ('candidate', 'needs_attention', 'disconnected')),
    row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version > 0),
    needs_rescan INTEGER NOT NULL DEFAULT 1 CHECK (needs_rescan IN (0, 1)),
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    source_kind TEXT CHECK (
        source_kind IS NULL OR source_kind = 'legacy_notes_sync_v1'
    ),
    source_locator_digest TEXT CHECK (
        source_locator_digest IS NULL OR (
            length(source_locator_digest) = 64
            AND source_locator_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    source_migration_id TEXT
        REFERENCES sync_migration_runs(migration_id) ON DELETE RESTRICT,
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    CHECK (
        (source_kind IS NULL
         AND source_locator_digest IS NULL
         AND source_migration_id IS NULL)
        OR (source_kind IS NOT NULL
            AND source_locator_digest IS NOT NULL
            AND source_migration_id IS NOT NULL)
    )
);
CREATE TABLE sync_migration_items (
    migration_id TEXT NOT NULL
        REFERENCES sync_migration_runs(migration_id) ON DELETE RESTRICT,
    item_kind TEXT NOT NULL
        CHECK (item_kind IN ('root', 'binding', 'legacy_conflict')),
    source_locator_digest TEXT NOT NULL
        CHECK (
            length(source_locator_digest) = 64
            AND source_locator_digest NOT GLOB '*[^0-9a-f]*'
        ),
    outcome TEXT NOT NULL
        CHECK (outcome IN ('created', 'matched', 'rejected', 'needs_rescan')),
    root_id TEXT REFERENCES sync_roots(root_id) ON DELETE RESTRICT,
    binding_id TEXT REFERENCES sync_bindings(binding_id) ON DELETE RESTRICT,
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    PRIMARY KEY (migration_id, item_kind, source_locator_digest),
    CHECK (
        (
            item_kind = 'root'
            AND outcome IN ('created', 'matched', 'needs_rescan')
            AND root_id IS NOT NULL
            AND binding_id IS NULL
            AND (
                (outcome IN ('created', 'matched') AND reason_code IS NULL)
                OR (outcome = 'needs_rescan' AND reason_code IS NOT NULL)
            )
        ) OR (
            item_kind = 'root'
            AND outcome = 'rejected'
            AND root_id IS NULL
            AND binding_id IS NULL
            AND reason_code IS NOT NULL
        ) OR (
            item_kind = 'binding'
            AND outcome IN ('created', 'matched', 'needs_rescan')
            AND root_id IS NULL
            AND binding_id IS NOT NULL
            AND (
                (outcome IN ('created', 'matched') AND reason_code IS NULL)
                OR (outcome = 'needs_rescan' AND reason_code IS NOT NULL)
            )
        ) OR (
            item_kind = 'binding'
            AND outcome = 'rejected'
            AND root_id IS NULL
            AND binding_id IS NULL
            AND reason_code IS NOT NULL
        ) OR (
            item_kind = 'legacy_conflict'
            AND outcome = 'needs_rescan'
            AND reason_code IS NOT NULL
            AND NOT (root_id IS NOT NULL AND binding_id IS NOT NULL)
        )
    )
);
CREATE INDEX idx_sync_migration_runs_state
    ON sync_migration_runs(state, updated_at);
CREATE INDEX idx_sync_roots_state
    ON sync_roots(state, updated_at);
CREATE UNIQUE INDEX idx_sync_roots_legacy_source
    ON sync_roots(source_kind, source_locator_digest)
    WHERE source_kind IS NOT NULL AND state <> 'disconnected';
CREATE INDEX idx_sync_bindings_root_state
    ON sync_bindings(root_id, state, updated_at);
CREATE UNIQUE INDEX idx_sync_bindings_live_note
    ON sync_bindings(note_id)
    WHERE state <> 'disconnected';
CREATE UNIQUE INDEX idx_sync_bindings_live_path_key
    ON sync_bindings(root_id, path_key)
    WHERE state <> 'disconnected' AND path_key IS NOT NULL;
CREATE INDEX idx_sync_migration_items_outcome
    ON sync_migration_items(migration_id, outcome, item_kind);
"""


@dataclass(frozen=True, slots=True)
class _TableOracle:
    name: str
    sql: str
    columns: tuple[tuple[object, ...], ...]
    foreign_keys: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class _IndexOracle:
    name: str
    sql: str | None
    unique: int
    origin: str
    partial: int
    columns: tuple[tuple[object, ...], ...]


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def _oracle_census(connection: sqlite3.Connection) -> tuple[object, ...]:
    expected_tables = _V1_TABLES | _V2_TABLES
    expected_indexes = _V1_INDEXES | _V2_INDEXES
    table_rows = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    index_rows = connection.execute(
        """SELECT name, sql FROM sqlite_master
        WHERE type = 'index' ORDER BY name"""
    ).fetchall()
    assert {row[0] for row in table_rows} == expected_tables
    observed_indexes = {row[0] for row in index_rows}
    assert expected_indexes <= observed_indexes
    assert all(
        name in expected_indexes or name.startswith("sqlite_autoindex_")
        for name in observed_indexes
    )
    tables = tuple(
        _TableOracle(
            name,
            _normalize_sql(sql),
            tuple(
                connection.execute(
                    """SELECT cid, name, type, \"notnull\", dflt_value, pk, hidden
                    FROM pragma_table_xinfo(?) ORDER BY cid""",
                    (name,),
                )
            ),
            tuple(
                connection.execute(
                    """SELECT id, seq, \"table\", \"from\", \"to\", on_update,
                    on_delete, match FROM pragma_foreign_key_list(?) ORDER BY id, seq""",
                    (name,),
                )
            ),
        )
        for name, sql in table_rows
    )
    indexes = tuple(
        _IndexOracle(
            name,
            None if sql is None else _normalize_sql(sql),
            int(
                connection.execute(
                    'SELECT "unique" FROM pragma_index_list(?) WHERE name = ?',
                    (table_name, name),
                ).fetchone()[0]
            ),
            str(
                connection.execute(
                    "SELECT origin FROM pragma_index_list(?) WHERE name = ?",
                    (table_name, name),
                ).fetchone()[0]
            ),
            int(
                connection.execute(
                    "SELECT partial FROM pragma_index_list(?) WHERE name = ?",
                    (table_name, name),
                ).fetchone()[0]
            ),
            tuple(
                connection.execute(
                    """SELECT seqno, cid, name, desc, coll
                    FROM pragma_index_xinfo(?) WHERE key = 1 ORDER BY seqno""",
                    (name,),
                )
            ),
        )
        for name, sql in index_rows
        for table_name in (
            connection.execute(
                "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
                (name,),
            ).fetchone()[0],
        )
    )
    return (tables, indexes)


def _create_oracle_v2(database: Path) -> tuple[object, ...]:
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_LEGACY_V1_SQL + _CANONICAL_V2_SQL)
        connection.execute("PRAGMA user_version = 2")
        return _oracle_census(connection)


@contextmanager
def _legacy_v1_transaction(
    database_path: str | Path,
    *,
    immediate: bool = False,
) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA user_version").fetchone() == (0,):
            connection.executescript(_LEGACY_V1_SQL)
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _approved_create_plan():
    item = ImportPreviewItem(
        item_id="upgrade-item",
        source=ImportSource(
            kind=ImportSourceKind.SELECTED_FILE,
            display_path="private.md",
            source_path=Path("/private/upgrade/private.md"),
        ),
        payloads=(ParsedNotePayload(title="private", content="private body"),),
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Imported",),
            ),
        ),
        classification=ImportClassification.NEW,
        reason="Ready.",
        default_action=ImportAction.CREATE_NEW,
        selected_action=ImportAction.CREATE_NEW,
        allowed_actions=(ImportAction.SKIP, ImportAction.CREATE_NEW),
        match=None,
        replace_content=False,
        add_membership=True,
    )
    plan = NoteImportPlan(
        bounds=ImportBounds(
            max_files=1,
            max_file_bytes=1024,
            max_total_bytes=1024,
            max_depth=1,
            max_entries=1,
            max_notes_per_file=1,
            max_keywords_per_note=1,
        ),
        items=(item,),
        proposed_folder_paths=(("Imported",),),
    )
    return approve_note_import_plan(
        plan,
        approval_id="00000000-0000-4000-8000-000000000097",
    )


def test_fresh_and_real_v1_upgrade_match_independent_v2_parity_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _create_oracle_v2(tmp_path / "oracle.sqlite3")
    fresh_database = tmp_path / "fresh.sqlite3"
    upgraded_database = tmp_path / "upgraded.sqlite3"
    approved = _approved_create_plan()
    approval_id = approved.approval_id
    repository = NoteImportReceiptRepository(upgraded_database)

    with monkeypatch.context() as patch:
        patch.setattr(
            receipt_module,
            "notes_sync_state_transaction",
            _legacy_v1_transaction,
        )
        repository.begin(approved, batch_size=1)
        repository.transition_session(approval_id, ImportSessionState.RUNNING)
        payload = repository.load_session_snapshot(approval_id).payload_effects[0]
        repository.transition_effects(
            approval_id,
            (
                EffectTransition(
                    category=ImportEffectCategory.PAYLOAD,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.FAILED,
                    reason_code="database_busy",
                    retryable=True,
                ),
            ),
        )
        repository.transition_item(
            approval_id,
            "upgrade-item",
            ImportItemOutcome.FAILED,
            reason_code="database_busy",
            retryable=True,
        )
        repository.transition_session(
            approval_id,
            ImportSessionState.NEEDS_ATTENTION,
        )
        with pytest.raises(ImportReceiptTransitionError):
            repository.transition_session(
                approval_id,
                ImportSessionState.COMPLETED,
            )
        before_snapshot = repository.load_session_snapshot(approval_id)
        before_aggregate = repository.aggregate_receipt(approval_id)

    with sqlite3.connect(upgraded_database) as connection:
        before_rows = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in sorted(_V1_TABLES)
        }
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)

    with notes_sync_state_transaction(fresh_database):
        pass
    with notes_sync_state_transaction(upgraded_database):
        pass

    with sqlite3.connect(fresh_database) as connection:
        assert _oracle_census(connection) == oracle
    with sqlite3.connect(upgraded_database) as connection:
        assert _oracle_census(connection) == oracle
        assert {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in sorted(_V1_TABLES)
        } == before_rows

    assert repository.load_session_snapshot(approval_id) == before_snapshot
    assert repository.aggregate_receipt(approval_id) == before_aggregate
    repository.reset_retryable_item(approval_id, item_id="upgrade-item")
    repository.reset_retryable_effect(
        approval_id,
        category=ImportEffectCategory.PAYLOAD,
        effect_id=before_snapshot.payload_effects[0].effect_id,
    )
    resumed = repository.transition_session(approval_id, ImportSessionState.RUNNING)
    assert resumed.state is ImportSessionState.RUNNING


def test_receipt_and_direct_initialization_orders_have_v2_parity(
    tmp_path: Path,
) -> None:
    receipt_first = tmp_path / "receipt-first.sqlite3"
    direct_first = tmp_path / "direct-first.sqlite3"

    NoteImportReceiptRepository(receipt_first).begin(
        _approved_create_plan(), batch_size=1
    )
    with notes_sync_state_transaction(receipt_first):
        pass
    with notes_sync_state_transaction(direct_first):
        pass
    NoteImportReceiptRepository(direct_first).begin(
        _approved_create_plan(), batch_size=1
    )

    with sqlite3.connect(receipt_first) as first_connection:
        first = _oracle_census(first_connection)
    with sqlite3.connect(direct_first) as second_connection:
        second = _oracle_census(second_connection)
    assert first == second


@pytest.mark.parametrize(
    "malformation",
    (
        "missing_table",
        "changed_table",
        "missing_index",
        "changed_predicate",
        "extra_trigger",
        "extra_view",
    ),
)
def test_claimed_v2_malformed_schema_fails_closed_without_repair(
    tmp_path: Path,
    malformation: str,
) -> None:
    database = tmp_path / f"malformed-{malformation}.sqlite3"
    _create_oracle_v2(database)
    with sqlite3.connect(database) as connection:
        if malformation == "missing_table":
            connection.execute("DROP TABLE sync_migration_items")
        elif malformation == "changed_table":
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("ALTER TABLE sync_roots RENAME TO old_sync_roots")
            connection.execute(
                "CREATE TABLE sync_roots (root_id TEXT NOT NULL PRIMARY KEY)"
            )
            connection.execute("DROP TABLE old_sync_roots")
        elif malformation == "missing_index":
            connection.execute("DROP INDEX idx_sync_roots_legacy_source")
        elif malformation == "changed_predicate":
            connection.execute("DROP INDEX idx_sync_roots_legacy_source")
            connection.execute(
                """CREATE UNIQUE INDEX idx_sync_roots_legacy_source
                ON sync_roots(source_kind, source_locator_digest)
                WHERE source_kind IS NOT NULL"""
            )
        elif malformation == "extra_trigger":
            connection.execute(
                """CREATE TRIGGER private_receipt_sentinel
                BEFORE INSERT ON import_sessions BEGIN
                    SELECT RAISE(ABORT, 'private trigger sentinel');
                END"""
            )
        else:
            connection.execute(
                """CREATE VIEW private_receipt_view AS
                SELECT session_id FROM import_sessions"""
            )
        connection.commit()
        before = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

    with pytest.raises(NotesSyncStateSchemaError, match="incompatible"):
        with notes_sync_state_transaction(database):
            pass

    with sqlite3.connect(database) as connection:
        after = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
    assert after == before


def test_canonical_census_accounts_for_implicit_unique_indexes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "implicit-indexes.sqlite3"
    with notes_sync_state_transaction(database):
        pass

    with sqlite3.connect(database) as connection:
        expected_autoindexes = {
            row[0]
            for table in sorted(_V1_TABLES | _V2_TABLES)
            for row in connection.execute(
                """SELECT name FROM pragma_index_list(?)
                WHERE origin IN ('pk', 'u')""",
                (table,),
            )
        }
        snapshot = schema_module._schema_snapshot(connection)

    assert expected_autoindexes
    assert expected_autoindexes <= {index.name for index in snapshot.indexes}


def test_schema_error_does_not_leak_raw_schema_or_path_text(tmp_path: Path) -> None:
    database = tmp_path / "private-schema-sentinel.sqlite3"
    _create_oracle_v2(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE private_absolute_path_sentinel (raw_text TEXT)"
        )

    with pytest.raises(NotesSyncStateSchemaError) as raised:
        with notes_sync_state_transaction(database):
            pass

    message = str(raised.value)
    assert "private_absolute_path_sentinel" not in message
    assert str(database) not in message
    assert raised.value.__cause__ is None


def test_v2_text_primary_keys_reject_null_primary_key_values(tmp_path: Path) -> None:
    database = tmp_path / "null-primary-key.sqlite3"
    with notes_sync_state_transaction(database) as connection:
        digest = "0" * 64
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO sync_migration_runs (
                    migration_id, source_kind, source_revision_before,
                    created_at, updated_at
                ) VALUES (NULL, 'legacy_notes_sync_v1', ?, 1, 1)""",
                (digest,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO sync_roots (
                    root_id, lexical_root_path, display_name, direction,
                    created_at, updated_at
                ) VALUES (NULL, '/root', 'Root', 'folder_to_notes', 1, 1)"""
            )
        connection.execute(
            """INSERT INTO sync_roots (
                root_id, lexical_root_path, display_name, direction,
                created_at, updated_at
            ) VALUES ('root-1', '/root', 'Root', 'folder_to_notes', 1, 1)"""
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO sync_bindings (
                    binding_id, root_id, note_id, lexical_relative_path,
                    created_at, updated_at
                ) VALUES (NULL, 'root-1', 'note-1', 'note.md', 1, 1)"""
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO sync_migration_items (
                    migration_id, item_kind, source_locator_digest, outcome,
                    reason_code, created_at
                ) VALUES (NULL, 'root', ?, 'rejected', 'invalid_path', 1)""",
                (digest,),
            )


@pytest.mark.parametrize(
    ("source_kind", "source_digest", "migration_id", "reason_code"),
    (
        (None, None, None, "legacy_direction_invalid"),
        ("legacy_notes_sync_v1", "0" * 64, _MIGRATION_ID, None),
        ("legacy_notes_sync_v1", "0" * 64, _MIGRATION_ID, "wrong_reason"),
    ),
)
def test_unspecified_direction_requires_exact_legacy_review_state(
    tmp_path: Path,
    source_kind: str | None,
    source_digest: str | None,
    migration_id: str | None,
    reason_code: str | None,
) -> None:
    database = tmp_path / "unspecified-direction.sqlite3"
    with notes_sync_state_transaction(database) as connection:
        connection.execute(
            """INSERT INTO sync_migration_runs (
                migration_id, source_kind, source_revision_before,
                created_at, updated_at
            ) VALUES (?, 'legacy_notes_sync_v1', ?, 1, 1)""",
            (_MIGRATION_ID, "0" * 64),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO sync_roots (
                    root_id, lexical_root_path, display_name, direction,
                    reason_code, source_kind, source_locator_digest,
                    source_migration_id, created_at, updated_at
                ) VALUES (
                    'root-1', '/root', 'Root', 'unspecified', ?, ?, ?, ?, 1, 1
                )""",
                (reason_code, source_kind, source_digest, migration_id),
            )


def test_two_initializers_reread_version_under_lock_and_converge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "concurrent.sqlite3"
    first_reads = Barrier(2)
    original_connect = schema_module.connect_private_sqlite
    original_connect("notes.sync_state", database).close()

    def synchronized_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        read_count = 0

        def synchronize_initial_version(sql: str) -> None:
            nonlocal read_count
            if sql.strip().upper() == "PRAGMA USER_VERSION":
                read_count += 1
                if read_count == 1:
                    first_reads.wait(timeout=5)

        connection.set_trace_callback(synchronize_initial_version)
        return connection

    monkeypatch.setattr(
        schema_module,
        "connect_private_sqlite",
        synchronized_connect,
    )

    def initialize() -> None:
        with notes_sync_state_transaction(database) as connection:
            assert connection.in_transaction
            assert connection.execute("SELECT count(*) FROM sync_roots").fetchone() == (
                0,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initialize) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)

    with sqlite3.connect(database, timeout=0) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        _oracle_census(connection)
        connection.execute("BEGIN IMMEDIATE")
        connection.rollback()


def test_healthy_v2_validation_completes_while_writer_slot_is_reserved(
    tmp_path: Path,
) -> None:
    database = tmp_path / "healthy-v2.sqlite3"
    with notes_sync_state_transaction(database):
        pass

    with sqlite3.connect(database, timeout=0) as writer:
        writer.execute("BEGIN IMMEDIATE")

        def validate() -> tuple[int]:
            with notes_sync_state_transaction(database) as connection:
                return connection.execute("SELECT count(*) FROM sync_roots").fetchone()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(validate)
            try:
                assert future.result(timeout=1) == (0,)
            finally:
                writer.rollback()


def test_upgrade_writes_version_last_and_rolls_back_partial_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "partial-upgrade.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(_LEGACY_V1_SQL)
        connection.execute("CREATE TABLE sync_roots (root_id TEXT)")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    statements: list[str] = []
    original_connect = schema_module.connect_private_sqlite

    def traced_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(schema_module, "connect_private_sqlite", traced_connect)

    with pytest.raises(NotesSyncStateSchemaError):
        with notes_sync_state_transaction(database):
            pass

    assert not any("PRAGMA user_version = 2" in sql for sql in statements)
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'sync_migration_runs'"
            ).fetchone()
            is None
        )


def test_upgrade_validates_complete_schema_while_version_is_still_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "version-last.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(_LEGACY_V1_SQL)
        connection.execute("PRAGMA user_version = 1")
    observed_versions: list[tuple[int, bool]] = []
    original_validate = schema_module._validate_v2_schema

    def observe_validation(
        connection: sqlite3.Connection,
        *,
        validate_version: bool,
    ) -> None:
        observed_versions.append(
            (
                int(connection.execute("PRAGMA user_version").fetchone()[0]),
                validate_version,
            )
        )
        original_validate(connection, validate_version=validate_version)

    monkeypatch.setattr(schema_module, "_validate_v2_schema", observe_validation)

    with notes_sync_state_transaction(database):
        pass

    assert observed_versions == [(1, False)]


def test_empty_database_initializes_the_canonical_v2_shared_schema(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    with notes_sync_state_transaction(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
            if not row[0].startswith("sqlite_autoindex_")
        }

    assert tables == _V1_TABLES | _V2_TABLES
    assert indexes == _V1_INDEXES | _V2_INDEXES


def test_v2_index_compatibility_failure_rolls_back_every_repair(
    tmp_path: Path,
) -> None:
    database = tmp_path / "malformed-v1.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE import_items (session_id TEXT, outcome TEXT);
            CREATE TABLE import_payload_effects (session_id TEXT);
            PRAGMA user_version = 2;
            """
        )

    with pytest.raises(NotesSyncStateSchemaError, match="incompatible"):
        with notes_sync_state_transaction(database):
            pass

    with sqlite3.connect(database) as connection:
        repaired_index = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("idx_import_items_outcome",),
        ).fetchone()

    assert repaired_index is None


@pytest.mark.parametrize(
    "open_error",
    (
        sqlite3.OperationalError("PRIVATE_OPEN_SENTINEL"),
        OSError("PRIVATE_OPEN_SENTINEL"),
    ),
)
def test_connection_open_failures_are_bounded_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_error: Exception,
) -> None:
    def fail_open(*_args, **_kwargs):
        raise open_error

    monkeypatch.setattr(schema_module, "connect_private_sqlite", fail_open)

    with pytest.raises(NotesSyncStateSchemaError) as raised:
        with notes_sync_state_transaction(tmp_path / "private-sentinel.sqlite3"):
            pass

    assert "PRIVATE_OPEN_SENTINEL" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_unknown_schema_version_is_rejected_without_mutation(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE future_private_state (opaque_value TEXT);
            INSERT INTO future_private_state VALUES ('opaque-marker');
            PRAGMA user_version = 3;
            """
        )
        before = (
            connection.execute("PRAGMA user_version").fetchone(),
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall(),
            connection.execute("SELECT * FROM future_private_state").fetchall(),
        )

    with pytest.raises(NotesSyncStateSchemaError, match="Unsupported"):
        with notes_sync_state_transaction(database):
            pass

    with sqlite3.connect(database) as connection:
        after = (
            connection.execute("PRAGMA user_version").fetchone(),
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall(),
            connection.execute("SELECT * FROM future_private_state").fetchall(),
        )

    assert after == before


def test_schema_phase_commits_before_a_failing_operation(tmp_path: Path) -> None:
    database = tmp_path / "schema-phase.sqlite3"

    with pytest.raises(RuntimeError, match="operation failed"):
        with notes_sync_state_transaction(database) as connection:
            connection.execute("CREATE TABLE operation_only (value TEXT)")
            raise RuntimeError("operation failed")

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert _V1_TABLES <= tables
    assert "operation_only" not in tables


def test_operation_failure_rolls_back_operation_rows(tmp_path: Path) -> None:
    database = tmp_path / "operation-rollback.sqlite3"
    with notes_sync_state_transaction(database):
        pass

    with pytest.raises(RuntimeError, match="operation failed"):
        with notes_sync_state_transaction(database) as connection:
            connection.execute(
                """INSERT INTO sync_migration_runs (
                    migration_id, source_kind, source_revision_before,
                    created_at, updated_at
                ) VALUES (?, 'legacy_notes_sync_v1', ?, 1, 1)""",
                (_MIGRATION_ID, "0" * 64),
            )
            raise RuntimeError("operation failed")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM sync_migration_runs").fetchall() == []


def test_operation_failure_closes_the_private_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[sqlite3.Connection] = []
    original_connect = schema_module.connect_private_sqlite

    def capture_connection(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        captured.append(connection)
        return connection

    monkeypatch.setattr(schema_module, "connect_private_sqlite", capture_connection)

    with pytest.raises(RuntimeError, match="operation failed"):
        with notes_sync_state_transaction(tmp_path / "closed.sqlite3"):
            raise RuntimeError("operation failed")

    assert len(captured) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        captured[0].execute("SELECT 1")


def test_healthy_v2_does_not_reserve_the_writer_slot(tmp_path: Path) -> None:
    database = tmp_path / "healthy-v1.sqlite3"
    with notes_sync_state_transaction(database):
        pass

    with sqlite3.connect(database, timeout=0) as reader:
        reader.execute("BEGIN")
        assert reader.execute("SELECT count(*) FROM import_sessions").fetchone() == (0,)
        with notes_sync_state_transaction(database):
            with sqlite3.connect(database, timeout=0) as second_connection:
                second_connection.execute("BEGIN IMMEDIATE")
                second_connection.rollback()
        reader.rollback()
