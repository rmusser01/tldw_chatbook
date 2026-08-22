"""Shared schema and private connection owner for Notes sync state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


SCHEMA_VERSION = 2

_V1_TABLE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS import_sessions (
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_items (
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_payload_effects (
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_folder_effects (
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS import_membership_effects (
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
    )
    """,
)

_V1_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_import_items_outcome ON import_items(session_id, outcome)",
    "CREATE INDEX IF NOT EXISTS idx_import_payload_state ON import_payload_effects(session_id, state)",
    "CREATE INDEX IF NOT EXISTS idx_import_folder_state ON import_folder_effects(session_id, state)",
    "CREATE INDEX IF NOT EXISTS idx_import_membership_state ON import_membership_effects(session_id, state)",
    "CREATE INDEX IF NOT EXISTS idx_import_payload_target ON import_payload_effects(session_id, target_note_id)",
    "CREATE INDEX IF NOT EXISTS idx_import_folder_target ON import_folder_effects(session_id, target_folder_id)",
    "CREATE INDEX IF NOT EXISTS idx_import_membership_path ON import_membership_effects(session_id, folder_path_digest, item_id)",
    "CREATE INDEX IF NOT EXISTS idx_import_folder_parent ON import_folder_effects(session_id, parent_effect_id)",
    "CREATE INDEX IF NOT EXISTS idx_import_items_target ON import_items(session_id, target_note_id, selected_action)",
    "CREATE INDEX IF NOT EXISTS idx_import_items_source_session ON import_items(source_locator_digest, session_id, item_id)",
)

_V2_TABLE_STATEMENTS = (
    """
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
    )
    """,
    """
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
    )
    """,
    """
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
    )
    """,
    """
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
    )
    """,
)

_V2_INDEX_STATEMENTS = (
    """CREATE INDEX idx_sync_migration_runs_state
    ON sync_migration_runs(state, updated_at)""",
    """CREATE INDEX idx_sync_roots_state
    ON sync_roots(state, updated_at)""",
    """CREATE UNIQUE INDEX idx_sync_roots_legacy_source
    ON sync_roots(source_kind, source_locator_digest)
    WHERE source_kind IS NOT NULL AND state <> 'disconnected'""",
    """CREATE INDEX idx_sync_bindings_root_state
    ON sync_bindings(root_id, state, updated_at)""",
    """CREATE UNIQUE INDEX idx_sync_bindings_live_note
    ON sync_bindings(note_id)
    WHERE state <> 'disconnected'""",
    """CREATE UNIQUE INDEX idx_sync_bindings_live_path_key
    ON sync_bindings(root_id, path_key)
    WHERE state <> 'disconnected' AND path_key IS NOT NULL""",
    """CREATE INDEX idx_sync_migration_items_outcome
    ON sync_migration_items(migration_id, outcome, item_kind)""",
)

_COMPLETE_V1_STATEMENTS = (
    *_V1_TABLE_STATEMENTS,
    *_V1_INDEX_STATEMENTS,
)

_COMPLETE_V2_STATEMENTS = (
    *_COMPLETE_V1_STATEMENTS,
    *_V2_TABLE_STATEMENTS,
    *_V2_INDEX_STATEMENTS,
)


class NotesSyncStateSchemaError(RuntimeError):
    """Report a bounded failure to initialize the private sync-state schema."""


@dataclass(frozen=True, slots=True)
class _ColumnCensus:
    cid: int
    name: str
    declared_type: str
    not_null: int
    default: object
    primary_key_position: int
    hidden: int


@dataclass(frozen=True, slots=True)
class _ForeignKeyCensus:
    identifier: int
    sequence: int
    target_table: str
    source_column: str
    target_column: str | None
    on_update: str
    on_delete: str
    match: str


@dataclass(frozen=True, slots=True)
class TableCensus:
    name: str
    sql: str
    columns: tuple[_ColumnCensus, ...]
    foreign_keys: tuple[_ForeignKeyCensus, ...]


@dataclass(frozen=True, slots=True)
class _IndexColumnCensus:
    sequence: int
    column_id: int
    name: str | None
    descending: int
    collation: str


@dataclass(frozen=True, slots=True)
class IndexCensus:
    name: str
    table_name: str
    sql: str | None
    unique: int
    origin: str
    partial: int
    columns: tuple[_IndexColumnCensus, ...]


@dataclass(frozen=True, slots=True)
class SyncStateSchemaSnapshot:
    user_version: int
    tables: tuple[TableCensus, ...]
    indexes: tuple[IndexCensus, ...]


_V2_TABLE_NAMES = frozenset(
    {
        "import_sessions",
        "import_items",
        "import_payload_effects",
        "import_folder_effects",
        "import_membership_effects",
        "sync_migration_runs",
        "sync_roots",
        "sync_bindings",
        "sync_migration_items",
    }
)
_V2_INDEX_NAMES = frozenset(
    {
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
        "idx_sync_migration_runs_state",
        "idx_sync_roots_state",
        "idx_sync_roots_legacy_source",
        "idx_sync_bindings_root_state",
        "idx_sync_bindings_live_note",
        "idx_sync_bindings_live_path_key",
        "idx_sync_migration_items_outcome",
    }
)


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def _schema_snapshot(connection: sqlite3.Connection) -> SyncStateSchemaSnapshot:
    canonical_objects = {("table", name) for name in _V2_TABLE_NAMES} | {
        ("index", name) for name in _V2_INDEX_NAMES
    }
    observed_objects = {
        (str(row[0]), str(row[1]))
        for row in connection.execute(
            """SELECT type, name FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'"""
        )
    }
    table_names = {
        str(row[0])
        for row in connection.execute(
            """SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE ? ESCAPE '\\'""",
            (r"sqlite\_%",),
        )
    }
    if table_names != _V2_TABLE_NAMES or observed_objects != canonical_objects:
        raise NotesSyncStateSchemaError(
            "The private Notes sync-state schema is incompatible with canonical v2."
        )

    tables: list[TableCensus] = []
    for name in sorted(_V2_TABLE_NAMES):
        sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        if sql_row is None or not isinstance(sql_row[0], str):
            raise NotesSyncStateSchemaError(
                "The private Notes sync-state schema is incompatible with canonical v2."
            )
        table_columns = tuple(
            _ColumnCensus(
                int(row[0]),
                str(row[1]),
                str(row[2]),
                int(row[3]),
                row[4],
                int(row[5]),
                int(row[6]),
            )
            for row in connection.execute(
                """SELECT cid, name, type, \"notnull\", dflt_value, pk, hidden
                FROM pragma_table_xinfo(?) ORDER BY cid""",
                (name,),
            )
        )
        foreign_keys = tuple(
            _ForeignKeyCensus(
                int(row[0]),
                int(row[1]),
                str(row[2]),
                str(row[3]),
                None if row[4] is None else str(row[4]),
                str(row[5]),
                str(row[6]),
                str(row[7]),
            )
            for row in connection.execute(
                """SELECT id, seq, \"table\", \"from\", \"to\", on_update,
                on_delete, match FROM pragma_foreign_key_list(?) ORDER BY id, seq""",
                (name,),
            )
        )
        tables.append(
            TableCensus(
                name=name,
                sql=_normalize_sql(sql_row[0]),
                columns=table_columns,
                foreign_keys=foreign_keys,
            )
        )

    indexes: list[IndexCensus] = []
    for table_name in sorted(_V2_TABLE_NAMES):
        for list_row in connection.execute(
            """SELECT name, \"unique\", origin, partial
            FROM pragma_index_list(?) ORDER BY name""",
            (table_name,),
        ):
            name = str(list_row[0])
            sql_row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                (name,),
            ).fetchone()
            if sql_row is None or (
                sql_row[0] is not None and not isinstance(sql_row[0], str)
            ):
                raise NotesSyncStateSchemaError(
                    "The private Notes sync-state schema is incompatible with canonical v2."
                )
            index_columns = tuple(
                _IndexColumnCensus(
                    int(row[0]),
                    int(row[1]),
                    None if row[2] is None else str(row[2]),
                    int(row[3]),
                    str(row[4]),
                )
                for row in connection.execute(
                    """SELECT seqno, cid, name, desc, coll
                    FROM pragma_index_xinfo(?) WHERE key = 1 ORDER BY seqno""",
                    (name,),
                )
            )
            indexes.append(
                IndexCensus(
                    name=name,
                    table_name=table_name,
                    sql=(
                        None if sql_row[0] is None else _normalize_sql(str(sql_row[0]))
                    ),
                    unique=int(list_row[1]),
                    origin=str(list_row[2]),
                    partial=int(list_row[3]),
                    columns=index_columns,
                )
            )
    return SyncStateSchemaSnapshot(
        user_version=int(connection.execute("PRAGMA user_version").fetchone()[0]),
        tables=tuple(tables),
        indexes=tuple(indexes),
    )


@cache
def _canonical_v2_snapshot() -> SyncStateSchemaSnapshot:
    with sqlite3.connect(":memory:") as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in _COMPLETE_V2_STATEMENTS:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return _schema_snapshot(connection)


def _validate_v2_schema(
    connection: sqlite3.Connection,
    *,
    validate_version: bool,
) -> None:
    observed = _schema_snapshot(connection)
    expected = _canonical_v2_snapshot()
    if observed.tables != expected.tables or observed.indexes != expected.indexes:
        raise NotesSyncStateSchemaError(
            "The private Notes sync-state schema is incompatible with canonical v2."
        )
    if validate_version and observed.user_version != expected.user_version:
        raise NotesSyncStateSchemaError(
            "The private Notes sync-state schema is incompatible with canonical v2."
        )


def _read_schema_version(connection: sqlite3.Connection) -> int:
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    except (sqlite3.Error, TypeError, ValueError):
        raise NotesSyncStateSchemaError(
            "The private Notes sync-state schema could not be inspected."
        ) from None


def _initialize_schema(connection: sqlite3.Connection) -> None:
    current_version = _read_schema_version(connection)
    if current_version not in {0, 1, SCHEMA_VERSION}:
        raise NotesSyncStateSchemaError(
            "Unsupported private Notes sync-state schema version."
        )

    try:
        if current_version == SCHEMA_VERSION:
            connection.execute("BEGIN")
            _validate_v2_schema(connection, validate_version=True)
            connection.commit()
            return

        connection.execute("BEGIN IMMEDIATE")
        current_version = _read_schema_version(connection)
        if current_version == SCHEMA_VERSION:
            _validate_v2_schema(connection, validate_version=True)
        elif current_version in {0, 1}:
            for statement in _COMPLETE_V2_STATEMENTS:
                connection.execute(statement)
            _validate_v2_schema(connection, validate_version=False)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        else:
            raise NotesSyncStateSchemaError(
                "Unsupported private Notes sync-state schema version."
            )
        connection.commit()
    except NotesSyncStateSchemaError:
        connection.rollback()
        raise
    except (sqlite3.Error, TypeError, ValueError):
        connection.rollback()
        raise NotesSyncStateSchemaError(
            "The private Notes sync-state schema is incompatible with canonical v2."
        ) from None


@contextmanager
def notes_sync_state_transaction(
    database_path: str | Path,
    *,
    immediate: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Open the shared schema, commit it, then run one operation transaction.

    Args:
        database_path: Profile-local path to the private sync-state database.
        immediate: Reserve SQLite's writer slot for the operation when true.

    Yields:
        The active private connection inside the operation transaction.

    Raises:
        NotesSyncStateSchemaError: If the database cannot be opened or its
            schema cannot be initialized safely.
        Exception: Re-raises operation failures after rolling back.
    """

    try:
        connection = connect_private_sqlite("notes.sync_state", Path(database_path))
    except (sqlite3.Error, OSError):
        raise NotesSyncStateSchemaError(
            "The private Notes sync-state database could not be opened."
        ) from None
    try:
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            _initialize_schema(connection)
        except sqlite3.Error:
            raise NotesSyncStateSchemaError(
                "The private Notes sync-state schema could not be initialized."
            ) from None
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
