import sqlite3

import pytest

from tldw_chatbook.Kanban_Interop.local_kanban_db import REQUIRED_KANBAN_TABLES
from tldw_chatbook.Kanban_Interop.local_kanban_service import LocalKanbanService
from tldw_chatbook.Kanban_Interop.server_kanban_service import KANBAN_OPERATION_SPECS


class FakePolicyEnforcer:
    def __init__(self):
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        ).fetchall()
    }


def _index_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
        if row["name"]
    }


def test_local_kanban_service_initializes_schema_and_foreign_keys(tmp_path):
    service = LocalKanbanService(db_path=tmp_path / "kanban.db")

    conn = service.connect()
    try:
        tables = _table_names(conn)
        indexes = _index_names(conn)
        schema_version = conn.execute(
            "SELECT value FROM local_kanban_schema_meta WHERE key = 'schema_version'"
        ).fetchone()["value"]

        assert REQUIRED_KANBAN_TABLES.issubset(tables)
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert schema_version == "1"
        assert {
            "idx_kanban_lists_board_active",
            "idx_kanban_cards_list_active",
            "idx_kanban_card_links_linked_content",
            "idx_kanban_activities_board_created",
        }.issubset(indexes)
    finally:
        conn.close()


def test_local_kanban_transaction_rolls_back_failed_mutations(tmp_path):
    service = LocalKanbanService(db_path=tmp_path / "kanban.db")

    with pytest.raises(RuntimeError, match="rollback"):
        with service.transaction() as conn:
            conn.execute(
                """
                INSERT INTO kanban_boards (uuid, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                ("board-uuid", "Project", service._now(), service._now()),
            )
            raise RuntimeError("rollback")

    conn = service.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM kanban_boards").fetchone()[0] == 0
    finally:
        conn.close()


def test_local_kanban_service_constructs_without_server_client(tmp_path):
    service = LocalKanbanService(db_path=tmp_path / "kanban.db")

    status = service.get_storage_status()
    pagination = service._pagination(limit=10, offset=5, total=12)

    assert service.db_path == tmp_path / "kanban.db"
    assert service.operations is KANBAN_OPERATION_SPECS
    assert status["schema_version"] == 1
    assert status["fts_available"] in {True, False}
    assert service._new_uuid()
    assert pagination == {"limit": 10, "offset": 5, "total": 12, "has_more": False, "next_offset": None}


def test_local_kanban_service_enforces_local_policy_action_ids(tmp_path):
    policy = FakePolicyEnforcer()
    service = LocalKanbanService(db_path=tmp_path / "kanban.db", policy_enforcer=policy)

    service._enforce("kanban.boards.list.local")

    assert policy.calls == ["kanban.boards.list.local"]
