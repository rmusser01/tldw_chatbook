"""Versioned schema for the device-private lasting Notes sync owner."""

from __future__ import annotations

import re
import sqlite3


HISTORICAL_V1_IMPORT_TABLE_STATEMENTS = (
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

HISTORICAL_V1_IMPORT_INDEX_STATEMENTS = (
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

HISTORICAL_V1_IMPORT_LEDGER_DDL = (
    ";\n".join(
        statement.strip()
        for statement in (
            *HISTORICAL_V1_IMPORT_TABLE_STATEMENTS,
            *HISTORICAL_V1_IMPORT_INDEX_STATEMENTS,
        )
    )
    + ";\nPRAGMA user_version = 1;"
)

_LASTING_TABLE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS notes_sync_roots (
        root_id TEXT PRIMARY KEY,
        note_scope_id TEXT NOT NULL,
        logical_folder_id TEXT,
        canonical_path TEXT NOT NULL,
        remote_origin_id TEXT,
        direction TEXT NOT NULL CHECK (direction IN ('bidirectional', 'folder_to_notes', 'notes_to_folder')),
        state TEXT NOT NULL CHECK (state IN ('pending', 'active', 'paused', 'disconnected')),
        cursor TEXT,
        last_status_code TEXT,
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        CHECK (length(root_id) BETWEEN 1 AND 256),
        CHECK (length(note_scope_id) BETWEEN 1 AND 256),
        CHECK (length(canonical_path) BETWEEN 1 AND 4096),
        CHECK (logical_folder_id IS NULL OR length(logical_folder_id) BETWEEN 1 AND 256),
        CHECK (remote_origin_id IS NULL OR length(remote_origin_id) BETWEEN 1 AND 256),
        CHECK (cursor IS NULL OR length(cursor) BETWEEN 1 AND 4096),
        CHECK (state != 'active' OR logical_folder_id IS NOT NULL),
        CHECK (last_status_code IS NULL OR (length(last_status_code) BETWEEN 1 AND 64 AND last_status_code NOT GLOB '*[^a-z0-9_]*'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notes_sync_bindings (
        binding_id TEXT PRIMARY KEY,
        root_id TEXT NOT NULL,
        note_scope_id TEXT NOT NULL,
        note_id TEXT NOT NULL,
        normalized_relative_path TEXT NOT NULL,
        stable_identity_digest TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('candidate', 'active', 'paused', 'needs_attention', 'disconnected')),
        utf8_bom INTEGER NOT NULL CHECK (utf8_bom IN (0, 1)),
        newline TEXT NOT NULL CHECK (newline IN ('lf', 'crlf')),
        final_newline INTEGER NOT NULL CHECK (final_newline IN (0, 1)),
        file_mode INTEGER NOT NULL CHECK (file_mode BETWEEN 0 AND 4095),
        content_digest TEXT NOT NULL,
        note_version INTEGER NOT NULL CHECK (note_version >= 0),
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        FOREIGN KEY (root_id) REFERENCES notes_sync_roots(root_id) ON DELETE RESTRICT,
        CHECK (length(binding_id) BETWEEN 1 AND 256),
        CHECK (length(note_scope_id) BETWEEN 1 AND 256),
        CHECK (length(note_id) BETWEEN 1 AND 256),
        CHECK (length(normalized_relative_path) BETWEEN 1 AND 4096),
        CHECK (length(stable_identity_digest) = 64 AND stable_identity_digest NOT GLOB '*[^0-9a-f]*'),
        CHECK (length(content_digest) = 64 AND content_digest NOT GLOB '*[^0-9a-f]*')
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notes_sync_operations (
        operation_id TEXT PRIMARY KEY,
        root_id TEXT NOT NULL,
        binding_id TEXT,
        kind TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('pending', 'recovery_admitted', 'first_authority_applied', 'second_authority_applied', 'binding_updated', 'verified', 'needs_attention', 'completed')),
        reason_code TEXT,
        observation_token TEXT NOT NULL,
        expected_note_version INTEGER CHECK (expected_note_version IS NULL OR expected_note_version >= 0),
        expected_file_digest TEXT,
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        FOREIGN KEY (root_id) REFERENCES notes_sync_roots(root_id) ON DELETE RESTRICT,
        FOREIGN KEY (binding_id) REFERENCES notes_sync_bindings(binding_id) ON DELETE RESTRICT,
        CHECK (length(operation_id) BETWEEN 1 AND 256),
        CHECK (length(observation_token) BETWEEN 1 AND 256),
        CHECK (length(kind) BETWEEN 1 AND 64 AND kind NOT GLOB '*[^a-z0-9_]*'),
        CHECK (expected_file_digest IS NULL OR (length(expected_file_digest) = 64 AND expected_file_digest NOT GLOB '*[^0-9a-f]*')),
        CHECK (reason_code IS NULL OR (length(reason_code) BETWEEN 1 AND 64 AND reason_code NOT GLOB '*[^a-z0-9_]*'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notes_sync_recovery (
        recovery_id TEXT PRIMARY KEY,
        operation_id TEXT NOT NULL UNIQUE,
        payload BLOB NOT NULL,
        metadata BLOB NOT NULL,
        expires_at INTEGER NOT NULL CHECK (expires_at > 0),
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        FOREIGN KEY (operation_id) REFERENCES notes_sync_operations(operation_id) ON DELETE RESTRICT,
        CHECK (length(recovery_id) BETWEEN 1 AND 256),
        CHECK (typeof(payload) = 'blob' AND typeof(metadata) = 'blob')
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notes_sync_legacy_migrations (
        migration_id TEXT PRIMARY KEY,
        source_fingerprint TEXT NOT NULL UNIQUE,
        state TEXT NOT NULL CHECK (state IN ('pending_review', 'reviewed', 'rejected')),
        reason_code TEXT,
        created_at INTEGER NOT NULL CHECK (created_at > 0),
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        CHECK (length(migration_id) BETWEEN 1 AND 256),
        CHECK (length(source_fingerprint) = 64 AND source_fingerprint NOT GLOB '*[^0-9a-f]*'),
        CHECK (reason_code IS NULL OR (length(reason_code) BETWEEN 1 AND 64 AND reason_code NOT GLOB '*[^a-z0-9_]*'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notes_sync_store_settings (
        setting_key TEXT PRIMARY KEY,
        setting_value TEXT NOT NULL,
        updated_at INTEGER NOT NULL CHECK (updated_at > 0),
        CHECK (length(setting_key) BETWEEN 1 AND 64 AND setting_key NOT GLOB '*[^a-z0-9_]*'),
        CHECK (length(setting_value) BETWEEN 1 AND 256),
        CHECK (
            (
                setting_key = 'recovery_capacity'
                AND length(setting_value) BETWEEN 1 AND 19
                AND substr(setting_value, 1, 1) GLOB '[1-9]'
                AND setting_value NOT GLOB '*[^0-9]*'
                AND printf('%d', CAST(setting_value AS INTEGER)) = setting_value
            ) OR (
                setting_key = 'cutover_marker'
                AND substr(setting_value, 1, 1) GLOB '[A-Za-z0-9]'
                AND setting_value NOT GLOB '*[^A-Za-z0-9_.:-]*'
            )
        )
    )
    """,
)

_LASTING_INDEX_STATEMENTS = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_sync_active_note ON notes_sync_bindings(note_scope_id, note_id) WHERE state = 'active'",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_sync_active_path ON notes_sync_bindings(root_id, normalized_relative_path) WHERE state = 'active'",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_sync_active_identity ON notes_sync_bindings(stable_identity_digest) WHERE state = 'active'",
    "CREATE INDEX IF NOT EXISTS idx_notes_sync_bindings_root ON notes_sync_bindings(root_id, state, binding_id)",
    "CREATE INDEX IF NOT EXISTS idx_notes_sync_operations_incomplete ON notes_sync_operations(root_id, state, operation_id)",
    "CREATE INDEX IF NOT EXISTS idx_notes_sync_recovery_expiry ON notes_sync_recovery(expires_at, recovery_id)",
)

LATEST_NOTES_DEVICE_SCHEMA_VERSION = 2
_V1_TABLES = HISTORICAL_V1_IMPORT_TABLE_STATEMENTS
_V1_INDEXES = HISTORICAL_V1_IMPORT_INDEX_STATEMENTS
_CURRENT_TABLES = (*_V1_TABLES, *_LASTING_TABLE_STATEMENTS)
_CURRENT_INDEXES = (*_V1_INDEXES, *_LASTING_INDEX_STATEMENTS)


class NotesDeviceSchemaError(RuntimeError):
    """A private device-state schema cannot be opened safely."""


def _object_name(statement: str) -> str:
    match = re.search(
        r"CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+IF\s+NOT\s+EXISTS\s+([^\s(]+)",
        statement,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise AssertionError("canonical schema statement has no object name")
    return match.group(1)


def _normalized_sql(value: str) -> str:
    return " ".join(
        value.lower()
        .replace("create table if not exists", "create table")
        .replace("create index if not exists", "create index")
        .replace("create unique index if not exists", "create unique index")
        .split()
    )


def _validate_objects(
    connection: sqlite3.Connection,
    statements: tuple[str, ...],
    *,
    object_type: str,
    allow_missing: bool,
) -> None:
    for statement in statements:
        name = _object_name(statement)
        row = connection.execute(
            "SELECT type, sql FROM sqlite_schema WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            if allow_missing:
                continue
            raise NotesDeviceSchemaError(
                "The private Notes device schema is incompatible with its canonical version."
            )
        if row[0] != object_type or type(row[1]) is not str:
            raise NotesDeviceSchemaError(
                "The private Notes device schema is incompatible with its canonical version."
            )
        if _normalized_sql(row[1]) != _normalized_sql(statement):
            raise NotesDeviceSchemaError(
                "The private Notes device schema is incompatible with its canonical version."
            )


def _validate_user_object_census(
    connection: sqlite3.Connection,
    table_statements: tuple[str, ...],
    index_statements: tuple[str, ...],
    *,
    version_name: str,
) -> None:
    table_names = {_object_name(statement) for statement in table_statements}
    index_names = {_object_name(statement) for statement in index_statements}
    for object_type, name, table_name, sql in connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema"
    ):
        if object_type == "table" and name in table_names:
            continue
        if object_type == "index" and name in index_names:
            continue
        if (
            object_type == "index"
            and str(name).startswith("sqlite_autoindex_")
            and table_name in table_names
            and sql is None
        ):
            continue
        raise NotesDeviceSchemaError(
            f"The private Notes device schema is incompatible with canonical {version_name}."
        )


def _repair_and_validate_indexes(
    connection: sqlite3.Connection,
    statements: tuple[str, ...],
) -> None:
    _validate_objects(
        connection,
        statements,
        object_type="index",
        allow_missing=True,
    )
    for statement in statements:
        connection.execute(statement)
    _validate_objects(
        connection,
        statements,
        object_type="index",
        allow_missing=False,
    )


def _create_historical_v1(connection: sqlite3.Connection) -> None:
    for statement in _V1_TABLES:
        connection.execute(statement)
    for statement in _V1_INDEXES:
        connection.execute(statement)
    connection.execute("PRAGMA user_version = 1")


def migrate_v1_to_current(connection: sqlite3.Connection) -> None:
    """Migrate a validated v1 import ledger to the inert lasting-sync schema."""

    for statement in _LASTING_TABLE_STATEMENTS:
        connection.execute(statement)
    for statement in _LASTING_INDEX_STATEMENTS:
        connection.execute(statement)
    connection.execute(f"PRAGMA user_version = {LATEST_NOTES_DEVICE_SCHEMA_VERSION}")


def initialize_notes_device_schema(connection: sqlite3.Connection) -> None:
    """Initialize or migrate one caller-transactional private database."""

    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, 1, LATEST_NOTES_DEVICE_SCHEMA_VERSION}:
            raise NotesDeviceSchemaError(
                "Unsupported private Notes device schema version."
            )
        if version == 0:
            user_object = connection.execute(
                """
                SELECT 1 FROM sqlite_schema
                WHERE name NOT LIKE 'sqlite_%'
                LIMIT 1
                """
            ).fetchone()
            if user_object is not None:
                raise NotesDeviceSchemaError(
                    "The private Notes device schema is incompatible with canonical v0."
                )
            _create_historical_v1(connection)
            version = 1
        if version == 1:
            _validate_user_object_census(
                connection,
                _V1_TABLES,
                _V1_INDEXES,
                version_name="v1",
            )
            _validate_objects(
                connection,
                _V1_TABLES,
                object_type="table",
                allow_missing=False,
            )
            _repair_and_validate_indexes(connection, _V1_INDEXES)
            migrate_v1_to_current(connection)
        _validate_user_object_census(
            connection,
            _CURRENT_TABLES,
            _CURRENT_INDEXES,
            version_name="current",
        )
        _validate_objects(
            connection,
            _CURRENT_TABLES,
            object_type="table",
            allow_missing=False,
        )
        _repair_and_validate_indexes(connection, _CURRENT_INDEXES)
    except NotesDeviceSchemaError:
        raise
    except Exception:
        raise NotesDeviceSchemaError(
            "The private Notes device schema could not be initialized safely."
        ) from None


__all__ = [
    "HISTORICAL_V1_IMPORT_INDEX_STATEMENTS",
    "HISTORICAL_V1_IMPORT_LEDGER_DDL",
    "HISTORICAL_V1_IMPORT_TABLE_STATEMENTS",
    "LATEST_NOTES_DEVICE_SCHEMA_VERSION",
    "NotesDeviceSchemaError",
    "initialize_notes_device_schema",
    "migrate_v1_to_current",
]
