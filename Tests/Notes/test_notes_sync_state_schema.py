"""Contracts for the shared private Notes sync-state schema owner."""

from __future__ import annotations

import sqlite3
from pathlib import Path


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


def test_empty_database_initializes_the_canonical_v1_receipt_schema(
    tmp_path: Path,
) -> None:
    module_path = Path("tldw_chatbook/Notes/notes_sync_state_schema.py")
    assert module_path.exists(), "notes sync-state coordinator must exist"
    from tldw_chatbook.Notes.notes_sync_state_schema import (
        notes_sync_state_transaction,
    )

    database = tmp_path / "notes-sync.sqlite3"
    with notes_sync_state_transaction(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
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

    assert tables == _V1_TABLES
    assert indexes == _V1_INDEXES
