"""Installed operational SQLite policies; imports never open runtime stores.

Current physical catalogs were captured from actual installed constructors at
65c77f341. Historical layouts are not qualified by relabeling their version.
Imported history stays inert; activation consumers must allocate fresh live claims.
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite

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
    def discover(self, config):
        from dataclasses import replace

        from tldw_chatbook.Backup_Recovery.profile_paths import setting
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
        from tldw_chatbook.DB.recovery_core import core_adapters

        item = super().discover(config)[0]
        if item.status != "unused":
            return (item,)
        if setting(config, "file_notes", "root"):
            return (replace(item, status="missing_required"),)
        core = next(
            a for a in core_adapters() if a.owner_id == "db.chachanotes.primary"
        )
        source = core.discover(config)[0]
        if source.status != "included" or core.validate(source.path):
            return (replace(item, status="unavailable"),)
        try:
            with closing(
                connect_private_sqlite(
                    core.backup_owner_id, source.path, read_only=True
                )
            ) as connection:
                if connection.execute(
                    "SELECT 1 FROM notes WHERE file_path_on_disk IS NOT NULL OR sync_root_folder IS NOT NULL LIMIT 1"
                ).fetchone():
                    item = replace(item, status="missing_required")
        except (OSError, ValueError, sqlite3.Error):
            item = replace(item, status="unavailable")
        return (item,)

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


@dataclass(frozen=True)
class _DeviceStateExclusion:
    """Device-local roots, receipts and operation authority are never portable."""

    owner_id: str = "notes.sync_state"
    activation_required: bool = False

    def discover(self, config):
        from tldw_chatbook.Backup_Recovery.config_adapter import _excluded_root
        from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir

        database = user_data_dir(config) / "tldw_chatbook_notes_sync_state.db"
        return tuple(
            _excluded_root(
                config,
                self.owner_id,
                Path(str(database) + suffix),
                kind="file",
                local_id=suffix or "",
            )
            for suffix in ("", "-wal", "-shm", "-journal")
        )

    def schema_policy(self):
        return None

    def validate(self, candidate):
        return ("notes_device_state_excluded",)

    def capture(self, item, destination, cancel):
        raise ValueError("notes_device_state_excluded")

    def relocate(self, candidate, mapping):
        raise ValueError("notes_device_state_excluded")


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _FileNotesAdapter(
            "notes.file_notes",
            None,
            "file_notes.sqlite",
            (0,),
            _FILE_NOTES_SCHEMA,
            (),
            optional_default=True,
        ),
        _DeviceStateExclusion(),
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
                return self._validate_connection(connection)
        except (OSError, ValueError, sqlite3.Error):
            return ("operational_validation_unavailable",)

    def _validate_connection(self, connection):
        """Run existing owned-content checks on an already restricted connection."""
        connection.execute("PRAGMA trusted_schema=OFF")
        if connection.execute(
            "SELECT 1 FROM note_folder_memberships WHERE ownership NOT IN ('manual','managed') OR (ownership='manual' AND (owner_id<>'' OR owner_active<>1)) OR (ownership='managed' AND length(owner_id)=0) LIMIT 1"
        ).fetchone():
            return ("invalid_managed_membership",)
        # Existing FK integrity checks cover note/folder/session references.
        # Absolute root/file paths are historical evidence, never opened.
        return ()

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
