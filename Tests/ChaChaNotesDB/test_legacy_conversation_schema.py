from __future__ import annotations

import sqlite3

import pytest

from Tests.ChaChaNotesDB.legacy_conversation_schema import (
    create_legacy_v12_conversations_db,
    create_legacy_v13_conversations_db,
)


@pytest.mark.parametrize(
    ("schema_version", "factory"),
    (
        (12, create_legacy_v12_conversations_db),
        (13, create_legacy_v13_conversations_db),
    ),
)
def test_legacy_fixture_includes_pre_v21_world_book_dependencies(
    tmp_path,
    schema_version,
    factory,
) -> None:
    db_path = tmp_path / f"legacy-v{schema_version}.sqlite"
    factory(db_path, [])

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        entry_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(world_book_entries)"
            )
        }

    assert {
        "world_books",
        "world_book_entries",
        "conversation_world_books",
    } <= tables
    assert {
        "id",
        "world_book_id",
        "keys",
        "content",
        "insertion_order",
        "last_modified",
    } <= entry_columns
    assert "priority" not in entry_columns
