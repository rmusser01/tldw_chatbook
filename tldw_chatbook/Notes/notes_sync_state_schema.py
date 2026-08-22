"""Shared schema and private connection owner for Notes sync state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


SCHEMA_VERSION = 1

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

_COMPLETE_V1_STATEMENTS = (
    *_V1_TABLE_STATEMENTS,
    *_V1_INDEX_STATEMENTS,
)


class NotesSyncStateSchemaError(RuntimeError):
    """Report a bounded failure to initialize the private sync-state schema."""


def _initialize_schema(connection: sqlite3.Connection) -> None:
    try:
        current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    except (sqlite3.Error, TypeError, ValueError):
        raise NotesSyncStateSchemaError(
            "The private Notes sync-state schema could not be inspected."
        ) from None

    if current_version not in {0, SCHEMA_VERSION}:
        raise NotesSyncStateSchemaError(
            "Unsupported private Notes sync-state schema version."
        )

    try:
        if current_version == SCHEMA_VERSION:
            for statement in _V1_INDEX_STATEMENTS:
                connection.execute(statement)
        else:
            connection.execute("BEGIN IMMEDIATE")
            for statement in _COMPLETE_V1_STATEMENTS:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise NotesSyncStateSchemaError(
            "The private Notes sync-state schema is incompatible with canonical v1."
        ) from None


@contextmanager
def notes_sync_state_transaction(
    database_path: str | Path,
    *,
    immediate: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Open the shared private schema and run one operation transaction."""

    connection = connect_private_sqlite("notes.sync_state", Path(database_path))
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
