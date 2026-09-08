"""Installed Kanban content/history recovery, without runtime composition."""

from contextlib import closing
from pathlib import Path
import sqlite3
from threading import Event
from tldw_chatbook.Backup_Recovery.models import OwnerAdapter, StorageItem
from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite

_SCHEMA = (
    (
        1,
        (
            "CREATE INDEX idx_kanban_activities_board_created\n            ON kanban_activities(board_id, created_at)",
            "CREATE INDEX idx_kanban_activities_card_created\n            ON kanban_activities(card_id, created_at)",
            "CREATE INDEX idx_kanban_boards_active\n            ON kanban_boards(is_deleted, is_archived, updated_at)",
            "CREATE INDEX idx_kanban_card_labels_label\n            ON kanban_card_labels(label_id)",
            "CREATE INDEX idx_kanban_card_links_card\n            ON kanban_card_links(card_id)",
            "CREATE INDEX idx_kanban_card_links_linked_content\n            ON kanban_card_links(linked_type, linked_id)",
            "CREATE INDEX idx_kanban_cards_board_active\n            ON kanban_cards(board_id, is_deleted, is_archived, position)",
            "CREATE INDEX idx_kanban_cards_list_active\n            ON kanban_cards(list_id, is_deleted, is_archived, position)",
            "CREATE INDEX idx_kanban_checklist_items_checklist\n            ON kanban_checklist_items(checklist_id, position)",
            "CREATE INDEX idx_kanban_checklists_card\n            ON kanban_checklists(card_id, position)",
            "CREATE INDEX idx_kanban_comments_card\n            ON kanban_comments(card_id, is_deleted, created_at)",
            "CREATE INDEX idx_kanban_labels_board\n            ON kanban_labels(board_id, name)",
            "CREATE INDEX idx_kanban_lists_board_active\n            ON kanban_lists(board_id, is_deleted, is_archived, position)",
            "CREATE TABLE kanban_activities (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            board_id INTEGER REFERENCES kanban_boards(id) ON DELETE CASCADE,\n            list_id INTEGER REFERENCES kanban_lists(id) ON DELETE SET NULL,\n            card_id INTEGER REFERENCES kanban_cards(id) ON DELETE SET NULL,\n            entity_type TEXT NOT NULL,\n            entity_id INTEGER,\n            action_type TEXT NOT NULL,\n            details_json TEXT,\n            created_at TEXT NOT NULL\n        )",
            "CREATE TABLE kanban_boards (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            client_id TEXT,\n            name TEXT NOT NULL,\n            description TEXT,\n            color TEXT,\n            user_id TEXT NOT NULL DEFAULT 'local',\n            metadata_json TEXT,\n            is_archived INTEGER NOT NULL DEFAULT 0,\n            is_deleted INTEGER NOT NULL DEFAULT 0,\n            version INTEGER NOT NULL DEFAULT 1,\n            activity_retention_days INTEGER,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL,\n            archived_at TEXT,\n            deleted_at TEXT\n        )",
            "CREATE TABLE kanban_card_labels (\n            card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,\n            label_id INTEGER NOT NULL REFERENCES kanban_labels(id) ON DELETE CASCADE,\n            created_at TEXT NOT NULL,\n            PRIMARY KEY (card_id, label_id)\n        )",
            "CREATE TABLE kanban_card_links (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,\n            linked_type TEXT NOT NULL,\n            linked_id TEXT NOT NULL,\n            linked_title TEXT,\n            metadata_json TEXT,\n            created_at TEXT NOT NULL,\n            UNIQUE(card_id, linked_type, linked_id)\n        )",
            "CREATE TABLE kanban_cards (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,\n            list_id INTEGER NOT NULL REFERENCES kanban_lists(id) ON DELETE CASCADE,\n            client_id TEXT,\n            title TEXT NOT NULL,\n            description TEXT,\n            position REAL NOT NULL DEFAULT 0,\n            due_date TEXT,\n            due_complete INTEGER NOT NULL DEFAULT 0,\n            start_date TEXT,\n            priority TEXT,\n            metadata_json TEXT,\n            is_archived INTEGER NOT NULL DEFAULT 0,\n            is_deleted INTEGER NOT NULL DEFAULT 0,\n            version INTEGER NOT NULL DEFAULT 1,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL,\n            archived_at TEXT,\n            deleted_at TEXT\n        )",
            "CREATE VIRTUAL TABLE kanban_cards_fts USING fts5(\n            title,\n            description,\n            content='kanban_cards',\n            content_rowid='id'\n        )",
            "CREATE TABLE 'kanban_cards_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID",
            "CREATE TABLE 'kanban_cards_fts_data'(id INTEGER PRIMARY KEY, block BLOB)",
            "CREATE TABLE 'kanban_cards_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB)",
            "CREATE TABLE 'kanban_cards_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID",
            "CREATE TABLE kanban_checklist_items (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            checklist_id INTEGER NOT NULL REFERENCES kanban_checklists(id) ON DELETE CASCADE,\n            client_id TEXT,\n            name TEXT NOT NULL,\n            checked INTEGER NOT NULL DEFAULT 0,\n            checked_at TEXT,\n            position REAL NOT NULL DEFAULT 0,\n            version INTEGER NOT NULL DEFAULT 1,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL\n        )",
            "CREATE TABLE kanban_checklists (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,\n            client_id TEXT,\n            name TEXT NOT NULL,\n            position REAL NOT NULL DEFAULT 0,\n            version INTEGER NOT NULL DEFAULT 1,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL\n        )",
            "CREATE TABLE kanban_comments (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,\n            client_id TEXT,\n            user_id TEXT NOT NULL DEFAULT 'local',\n            content TEXT NOT NULL,\n            is_deleted INTEGER NOT NULL DEFAULT 0,\n            version INTEGER NOT NULL DEFAULT 1,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL,\n            deleted_at TEXT\n        )",
            "CREATE TABLE kanban_labels (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,\n            client_id TEXT,\n            name TEXT NOT NULL,\n            color TEXT,\n            version INTEGER NOT NULL DEFAULT 1,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL,\n            UNIQUE(board_id, name)\n        )",
            "CREATE TABLE kanban_lists (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            uuid TEXT NOT NULL UNIQUE,\n            board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,\n            client_id TEXT,\n            name TEXT NOT NULL,\n            position REAL NOT NULL DEFAULT 0,\n            is_archived INTEGER NOT NULL DEFAULT 0,\n            is_deleted INTEGER NOT NULL DEFAULT 0,\n            version INTEGER NOT NULL DEFAULT 1,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL,\n            archived_at TEXT,\n            deleted_at TEXT\n        )",
            "CREATE TABLE local_kanban_schema_meta (\n            key TEXT PRIMARY KEY,\n            value TEXT NOT NULL\n        )",
            "CREATE TABLE sqlite_sequence(name,seq)",
        ),
    ),
)


class _KanbanAdapter(_SQLiteDeclaration):
    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.operations.kanban", candidate, read_only=True
                )
            ) as connection:
                return _validate_sqlite(
                    connection,
                    self.versions,
                    self.schemas,
                    version_query="SELECT CAST(value AS INTEGER) FROM local_kanban_schema_meta WHERE key='schema_version'",
                )
        except (OSError, ValueError, sqlite3.Error, TypeError):
            return ("operational_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.operations.kanban",
                item.path,
                destination,
                progress_guard=guard,
            )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _KanbanAdapter("kanban.local", None, "tldw_chatbook_kanban.db", (1,), _SCHEMA),
    )
