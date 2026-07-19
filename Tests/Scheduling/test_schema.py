"""Tests for the Scheduling module database schema DDL."""

import sqlite3

from tldw_chatbook.Scheduling.db.schema import CREATE_SCHEMA_SQL


EXPECTED_TABLES = [
    "schema_version",
    "reminder_tasks",
    "automation_definitions",
    "automation_previews",
    "automation_audit_events",
    "sync_state",
    "sync_mapping",
    "sync_tombstones",
    "sync_conflicts",
]


def test_schema_executes() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(CREATE_SCHEMA_SQL)


def test_expected_tables_exist() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(CREATE_SCHEMA_SQL)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )
    tables = {row[0] for row in cursor.fetchall()}
    for table in EXPECTED_TABLES:
        assert table in tables, f"Expected table {table!r} to exist"
