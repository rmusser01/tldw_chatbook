"""Installed operational SQLite policies; imports never open runtime stores.

Current physical catalogs were captured from actual installed constructors at
65c77f341. Historical layouts are not qualified by relabeling their version.
Imported history stays inert; activation consumers must allocate fresh live claims.
"""

from contextlib import closing
from dataclasses import dataclass
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from threading import Event

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite

from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration

_FILE_NOTES_SCHEMA = (
    (
        0,
        (
            "CREATE TABLE files (\n                    root TEXT NOT NULL,\n                    relative_path TEXT NOT NULL,\n                    raw_bytes BLOB NOT NULL,\n                    content_hash TEXT NOT NULL,\n                    decoded_text TEXT,\n                    size INTEGER NOT NULL,\n                    mtime_ns INTEGER NOT NULL,\n                    deleted_at TEXT,\n                    UNIQUE(root, relative_path)\n                )",
            "CREATE VIRTUAL TABLE files_fts USING fts5(\n                    root UNINDEXED,\n                    relative_path UNINDEXED,\n                    decoded_text,\n                    tokenize = 'unicode61'\n                )",
            "CREATE TABLE 'files_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID",
            "CREATE TABLE 'files_fts_content'(id INTEGER PRIMARY KEY, c0, c1, c2)",
            "CREATE TABLE 'files_fts_data'(id INTEGER PRIMARY KEY, block BLOB)",
            "CREATE TABLE 'files_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB)",
            "CREATE TABLE 'files_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID",
            "CREATE TABLE protected_paths (\n                    root TEXT NOT NULL,\n                    relative_path TEXT NOT NULL,\n                    is_prefix INTEGER NOT NULL CHECK(is_prefix IN (0, 1)),\n                    UNIQUE(root, relative_path, is_prefix)\n                )",
            "CREATE TABLE revisions (\n                    root TEXT NOT NULL,\n                    relative_path TEXT NOT NULL,\n                    raw_bytes BLOB NOT NULL,\n                    content_hash TEXT NOT NULL,\n                    kind TEXT NOT NULL,\n                    session_key TEXT,\n                    created_at TEXT NOT NULL,\n                    UNIQUE(root, relative_path, kind, session_key)\n                )",
        ),
    ),
)


class _FileNotesAdapter(_SQLiteDeclaration):
    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.operations.file_notes", candidate, read_only=True
                )
            ) as connection:
                return _validate_sqlite(
                    connection, self.versions, self.schemas, version_query="SELECT 0"
                )
        except (OSError, ValueError, sqlite3.Error):
            return ("operational_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.operations.file_notes",
                item.path,
                destination,
                progress_guard=guard,
            )


_RECEIPTS_SCHEMA = (
    (
        1,
        (
            "CREATE INDEX idx_import_folder_parent ON import_folder_effects(session_id, parent_effect_id)",
            "CREATE INDEX idx_import_folder_state ON import_folder_effects(session_id, state)",
            "CREATE INDEX idx_import_folder_target ON import_folder_effects(session_id, target_folder_id)",
            "CREATE INDEX idx_import_items_outcome ON import_items(session_id, outcome)",
            "CREATE INDEX idx_import_items_source_session ON import_items(source_locator_digest, session_id, item_id)",
            "CREATE INDEX idx_import_items_target ON import_items(session_id, target_note_id, selected_action)",
            "CREATE INDEX idx_import_membership_path ON import_membership_effects(session_id, folder_path_digest, item_id)",
            "CREATE INDEX idx_import_membership_state ON import_membership_effects(session_id, state)",
            "CREATE INDEX idx_import_payload_state ON import_payload_effects(session_id, state)",
            "CREATE INDEX idx_import_payload_target ON import_payload_effects(session_id, target_note_id)",
            "CREATE TABLE import_folder_effects (\n        effect_id TEXT PRIMARY KEY,\n        session_id TEXT NOT NULL,\n        folder_ordinal INTEGER NOT NULL CHECK (folder_ordinal >= 0),\n        path_digest TEXT NOT NULL,\n        parent_effect_id TEXT,\n        effect_kind TEXT NOT NULL DEFAULT 'ensure_folder' CHECK (effect_kind = 'ensure_folder'),\n        state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),\n        target_folder_id TEXT,\n        reason_code TEXT CHECK (\n            reason_code IS NULL OR (\n                length(reason_code) BETWEEN 1 AND 64\n                AND reason_code NOT GLOB '*[^a-z0-9_]*'\n                AND substr(reason_code, 1, 1) GLOB '[a-z]'\n            )\n        ),\n        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),\n        created_at INTEGER NOT NULL CHECK (created_at > 0),\n        updated_at INTEGER NOT NULL CHECK (updated_at > 0),\n        FOREIGN KEY (session_id) REFERENCES import_sessions(session_id) ON DELETE CASCADE,\n        FOREIGN KEY (parent_effect_id)\n            REFERENCES import_folder_effects(effect_id) ON DELETE RESTRICT,\n        UNIQUE (session_id, path_digest),\n        UNIQUE (session_id, folder_ordinal),\n        CHECK (length(effect_id) BETWEEN 1 AND 256),\n        CHECK (length(path_digest) = 64 AND path_digest NOT GLOB '*[^0-9a-f]*'),\n        CHECK (target_folder_id IS NULL OR length(target_folder_id) BETWEEN 1 AND 256),\n        CHECK (state = 'failed' OR retryable = 0)\n    )",
            "CREATE TABLE import_items (\n        session_id TEXT NOT NULL,\n        item_id TEXT NOT NULL,\n        source_locator_digest TEXT NOT NULL,\n        selected_action TEXT NOT NULL\n            CHECK (selected_action IN ('skip', 'create_new', 'update_existing')),\n        outcome_count INTEGER NOT NULL CHECK (outcome_count > 0),\n        outcome TEXT NOT NULL DEFAULT 'pending'\n            CHECK (outcome IN ('pending', 'imported', 'updated', 'skipped', 'failed')),\n        target_note_id TEXT,\n        expected_version INTEGER CHECK (expected_version IS NULL OR expected_version >= 0),\n        observed_version INTEGER CHECK (observed_version IS NULL OR observed_version >= 0),\n        reason_code TEXT CHECK (\n            reason_code IS NULL OR (\n                length(reason_code) BETWEEN 1 AND 64\n                AND reason_code NOT GLOB '*[^a-z0-9_]*'\n                AND substr(reason_code, 1, 1) GLOB '[a-z]'\n            )\n        ),\n        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),\n        created_at INTEGER NOT NULL CHECK (created_at > 0),\n        updated_at INTEGER NOT NULL CHECK (updated_at > 0),\n        PRIMARY KEY (session_id, item_id),\n        FOREIGN KEY (session_id) REFERENCES import_sessions(session_id) ON DELETE CASCADE,\n        CHECK (length(item_id) BETWEEN 1 AND 256),\n        CHECK (\n            length(source_locator_digest) = 64\n            AND source_locator_digest NOT GLOB '*[^0-9a-f]*'\n        ),\n        CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),\n        CHECK (outcome = 'failed' OR retryable = 0)\n    )",
            "CREATE TABLE import_membership_effects (\n        effect_id TEXT PRIMARY KEY,\n        session_id TEXT NOT NULL,\n        item_id TEXT NOT NULL,\n        payload_index INTEGER NOT NULL CHECK (payload_index >= 0),\n        membership_ordinal INTEGER NOT NULL CHECK (membership_ordinal >= 0),\n        folder_path_digest TEXT NOT NULL,\n        effect_kind TEXT NOT NULL DEFAULT 'attach_membership'\n            CHECK (effect_kind = 'attach_membership'),\n        state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),\n        target_note_id TEXT,\n        target_folder_id TEXT,\n        reason_code TEXT CHECK (\n            reason_code IS NULL OR (\n                length(reason_code) BETWEEN 1 AND 64\n                AND reason_code NOT GLOB '*[^a-z0-9_]*'\n                AND substr(reason_code, 1, 1) GLOB '[a-z]'\n            )\n        ),\n        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),\n        created_at INTEGER NOT NULL CHECK (created_at > 0),\n        updated_at INTEGER NOT NULL CHECK (updated_at > 0),\n        FOREIGN KEY (session_id, item_id)\n            REFERENCES import_items(session_id, item_id) ON DELETE CASCADE,\n        UNIQUE (session_id, item_id, payload_index, membership_ordinal),\n        CHECK (length(effect_id) BETWEEN 1 AND 256),\n        CHECK (\n            length(folder_path_digest) = 64\n            AND folder_path_digest NOT GLOB '*[^0-9a-f]*'\n        ),\n        CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),\n        CHECK (target_folder_id IS NULL OR length(target_folder_id) BETWEEN 1 AND 256),\n        CHECK (state = 'failed' OR retryable = 0)\n    )",
            "CREATE TABLE import_payload_effects (\n        effect_id TEXT PRIMARY KEY,\n        session_id TEXT NOT NULL,\n        item_id TEXT NOT NULL,\n        payload_index INTEGER NOT NULL CHECK (payload_index >= 0),\n        payload_digest TEXT NOT NULL,\n        effect_kind TEXT NOT NULL CHECK (effect_kind IN ('create_note', 'replace_content')),\n        state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'applied', 'failed')),\n        target_note_id TEXT,\n        expected_version INTEGER CHECK (expected_version IS NULL OR expected_version >= 0),\n        observed_version INTEGER CHECK (observed_version IS NULL OR observed_version >= 0),\n        reason_code TEXT CHECK (\n            reason_code IS NULL OR (\n                length(reason_code) BETWEEN 1 AND 64\n                AND reason_code NOT GLOB '*[^a-z0-9_]*'\n                AND substr(reason_code, 1, 1) GLOB '[a-z]'\n            )\n        ),\n        retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),\n        created_at INTEGER NOT NULL CHECK (created_at > 0),\n        updated_at INTEGER NOT NULL CHECK (updated_at > 0),\n        FOREIGN KEY (session_id, item_id)\n            REFERENCES import_items(session_id, item_id) ON DELETE CASCADE,\n        UNIQUE (session_id, item_id, payload_index, effect_kind),\n        CHECK (length(effect_id) BETWEEN 1 AND 256),\n        CHECK (length(payload_digest) = 64 AND payload_digest NOT GLOB '*[^0-9a-f]*'),\n        CHECK (target_note_id IS NULL OR length(target_note_id) BETWEEN 1 AND 256),\n        CHECK (state = 'failed' OR retryable = 0)\n    )",
            "CREATE TABLE import_sessions (\n        session_id TEXT PRIMARY KEY,\n        approval_id TEXT NOT NULL UNIQUE,\n        plan_digest TEXT NOT NULL,\n        state TEXT NOT NULL DEFAULT 'pending'\n            CHECK (state IN ('pending', 'running', 'cancelled', 'completed', 'needs_attention')),\n        batch_size INTEGER NOT NULL CHECK (batch_size BETWEEN 1 AND 100),\n        total_count INTEGER NOT NULL CHECK (total_count >= 0),\n        reason_code TEXT CHECK (\n            reason_code IS NULL OR (\n                length(reason_code) BETWEEN 1 AND 64\n                AND reason_code NOT GLOB '*[^a-z0-9_]*'\n                AND substr(reason_code, 1, 1) GLOB '[a-z]'\n            )\n        ),\n        created_at INTEGER NOT NULL CHECK (created_at > 0),\n        updated_at INTEGER NOT NULL CHECK (updated_at > 0),\n        CHECK (length(session_id) BETWEEN 1 AND 256),\n        CHECK (length(approval_id) = 36),\n        CHECK (length(plan_digest) = 64 AND plan_digest NOT GLOB '*[^0-9a-f]*')\n    )",
        ),
    ),
)


class _ReceiptsAdapter(_SQLiteDeclaration):
    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.operations.receipts", candidate, read_only=True
                )
            ) as connection:
                return _validate_sqlite(
                    connection,
                    self.versions,
                    self.schemas,
                    version_query="PRAGMA user_version",
                )
        except (OSError, ValueError, sqlite3.Error):
            return ("operational_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.operations.receipts",
                item.path,
                destination,
                progress_guard=guard,
            )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _FileNotesAdapter(
            "notes.file_notes", None, "file_notes.sqlite", (0,), _FILE_NOTES_SCHEMA, ()
        ),
        _ReceiptsAdapter(
            "notes.sync_state",
            None,
            "tldw_chatbook_notes_sync_state.db",
            (1,),
            _RECEIPTS_SCHEMA,
            ("db.chachanotes.primary",),
        ),
        _SyncBindings(),
    )


@dataclass(frozen=True)
class _SyncBindings:
    """Semantic legacy sync/member ownership in the shared core payload."""

    owner_id: str = "notes.sync_bindings"
    activation_required: bool = True

    @staticmethod
    def _core():
        from tldw_chatbook.DB.recovery_core import core_adapters

        return next(
            a for a in core_adapters() if a.owner_id == "db.chachanotes.primary"
        )

    def discover(self, config):
        from dataclasses import replace

        context = discovery_context(config)
        item = self._core().discover(config)[0]
        return (
            replace(
                item,
                owner=self.owner_id,
                logical_id=storage_logical_id(context, self.owner_id),
                dependencies=(storage_logical_id(context, "db.chachanotes.primary"),),
            ),
        )

    def schema_policy(self):
        from dataclasses import replace

        return replace(self._core().schema_policy(), owner=self.owner_id)

    def validate(self, candidate):
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        issues = self._core().validate(candidate)
        if issues:
            return issues
        try:
            with closing(
                connect_private_sqlite(
                    "recovery.operations.note_bindings", candidate, read_only=True
                )
            ) as connection:
                connection.execute("PRAGMA trusted_schema=OFF")
                if connection.execute(
                    "SELECT 1 FROM note_folder_memberships WHERE ownership NOT IN ('manual','managed') OR (ownership='manual' AND (owner_id<>'' OR owner_active<>1)) OR (ownership='managed' AND length(owner_id)=0) LIMIT 1"
                ).fetchone():
                    return ("invalid_managed_membership",)
                # Existing FK integrity checks cover note/folder/session references.
                # Absolute root/file paths are historical evidence, never opened.
                return ()
        except (OSError, ValueError, sqlite3.Error):
            return ("operational_validation_unavailable",)

    def capture(self, item, destination, cancel):
        from dataclasses import replace

        if item.owner != self.owner_id:
            raise ValueError("invalid_capture_item")
        self._core().capture(
            replace(item, owner="db.chachanotes.primary"), destination, cancel
        )
        issues = self.validate(destination)
        if issues:
            raise ValueError(issues[0])

    def relocate(self, candidate, mapping):
        # Never convert managed membership to manual, erase owner IDs or replay
        # old filesystem writes. Task20 owner review supplies new live claims.
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])
