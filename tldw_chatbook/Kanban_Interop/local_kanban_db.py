"""SQLite storage foundation for local Kanban parity."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 1

REQUIRED_KANBAN_TABLES = {
    "local_kanban_schema_meta",
    "kanban_boards",
    "kanban_lists",
    "kanban_cards",
    "kanban_labels",
    "kanban_card_labels",
    "kanban_checklists",
    "kanban_checklist_items",
    "kanban_comments",
    "kanban_activities",
    "kanban_card_links",
}


def open_connection(db_path: str | Path) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS local_kanban_schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kanban_boards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            client_id TEXT,
            name TEXT NOT NULL,
            description TEXT,
            color TEXT,
            user_id TEXT NOT NULL DEFAULT 'local',
            metadata_json TEXT,
            is_archived INTEGER NOT NULL DEFAULT 0,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            activity_retention_days INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT,
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS kanban_lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,
            client_id TEXT,
            name TEXT NOT NULL,
            position REAL NOT NULL DEFAULT 0,
            is_archived INTEGER NOT NULL DEFAULT 0,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT,
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS kanban_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,
            list_id INTEGER NOT NULL REFERENCES kanban_lists(id) ON DELETE CASCADE,
            client_id TEXT,
            title TEXT NOT NULL,
            description TEXT,
            position REAL NOT NULL DEFAULT 0,
            due_date TEXT,
            due_complete INTEGER NOT NULL DEFAULT 0,
            start_date TEXT,
            priority TEXT,
            metadata_json TEXT,
            is_archived INTEGER NOT NULL DEFAULT 0,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT,
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS kanban_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,
            client_id TEXT,
            name TEXT NOT NULL,
            color TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(board_id, name)
        );

        CREATE TABLE IF NOT EXISTS kanban_card_labels (
            card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
            label_id INTEGER NOT NULL REFERENCES kanban_labels(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            PRIMARY KEY (card_id, label_id)
        );

        CREATE TABLE IF NOT EXISTS kanban_checklists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
            client_id TEXT,
            name TEXT NOT NULL,
            position REAL NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kanban_checklist_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            checklist_id INTEGER NOT NULL REFERENCES kanban_checklists(id) ON DELETE CASCADE,
            client_id TEXT,
            name TEXT NOT NULL,
            checked INTEGER NOT NULL DEFAULT 0,
            checked_at TEXT,
            position REAL NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kanban_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
            client_id TEXT,
            user_id TEXT NOT NULL DEFAULT 'local',
            content TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS kanban_activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            board_id INTEGER REFERENCES kanban_boards(id) ON DELETE CASCADE,
            list_id INTEGER REFERENCES kanban_lists(id) ON DELETE SET NULL,
            card_id INTEGER REFERENCES kanban_cards(id) ON DELETE SET NULL,
            entity_type TEXT NOT NULL,
            entity_id INTEGER,
            action_type TEXT NOT NULL,
            details_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kanban_card_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
            linked_type TEXT NOT NULL,
            linked_id TEXT NOT NULL,
            linked_title TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(card_id, linked_type, linked_id)
        );

        CREATE INDEX IF NOT EXISTS idx_kanban_boards_active
            ON kanban_boards(is_deleted, is_archived, updated_at);
        CREATE INDEX IF NOT EXISTS idx_kanban_lists_board_active
            ON kanban_lists(board_id, is_deleted, is_archived, position);
        CREATE INDEX IF NOT EXISTS idx_kanban_cards_board_active
            ON kanban_cards(board_id, is_deleted, is_archived, position);
        CREATE INDEX IF NOT EXISTS idx_kanban_cards_list_active
            ON kanban_cards(list_id, is_deleted, is_archived, position);
        CREATE INDEX IF NOT EXISTS idx_kanban_labels_board
            ON kanban_labels(board_id, name);
        CREATE INDEX IF NOT EXISTS idx_kanban_card_labels_label
            ON kanban_card_labels(label_id);
        CREATE INDEX IF NOT EXISTS idx_kanban_checklists_card
            ON kanban_checklists(card_id, position);
        CREATE INDEX IF NOT EXISTS idx_kanban_checklist_items_checklist
            ON kanban_checklist_items(checklist_id, position);
        CREATE INDEX IF NOT EXISTS idx_kanban_comments_card
            ON kanban_comments(card_id, is_deleted, created_at);
        CREATE INDEX IF NOT EXISTS idx_kanban_activities_board_created
            ON kanban_activities(board_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_kanban_activities_card_created
            ON kanban_activities(card_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_kanban_card_links_card
            ON kanban_card_links(card_id);
        CREATE INDEX IF NOT EXISTS idx_kanban_card_links_linked_content
            ON kanban_card_links(linked_type, linked_id);
        """
    )
    fts_available = _ensure_fts(conn)
    conn.execute(
        """
        INSERT INTO local_kanban_schema_meta(key, value)
        VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(SCHEMA_VERSION),),
    )
    conn.execute(
        """
        INSERT INTO local_kanban_schema_meta(key, value)
        VALUES ('fts_available', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        ("1" if fts_available else "0",),
    )
    conn.commit()


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS kanban_cards_fts USING fts5(
                title,
                description,
                content='kanban_cards',
                content_rowid='id'
            )
            """
        )
    except sqlite3.OperationalError:
        return False
    return True


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
