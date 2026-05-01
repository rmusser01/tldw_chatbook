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


@pytest.mark.asyncio
async def test_local_kanban_board_list_card_crud_persists_and_versions(tmp_path):
    db_path = tmp_path / "kanban.db"
    service = LocalKanbanService(db_path=db_path)

    board = await service.create_board({"name": "Project", "client_id": "board-1", "description": "Plan"})
    updated_board = await service.update_board(board["id"], {"name": "Project 2"}, expected_version=1)
    kanban_list = await service.create_list(board["id"], {"name": "Todo", "client_id": "list-1"})
    card = await service.create_card(
        kanban_list["id"],
        {"title": "Task", "client_id": "card-1", "description": "First task", "priority": "high"},
    )
    updated_card = await service.update_card(card["id"], {"title": "Task 2"}, expected_version=1)

    reloaded = LocalKanbanService(db_path=db_path)
    boards = await reloaded.list_boards()
    lists = await reloaded.list_lists(board["id"])
    cards = await reloaded.list_cards(kanban_list["id"])
    loaded_card = await reloaded.get_card(card["id"])

    assert board["version"] == 1
    assert updated_board["name"] == "Project 2"
    assert updated_board["version"] == 2
    assert kanban_list["board_id"] == board["id"]
    assert updated_card["title"] == "Task 2"
    assert updated_card["version"] == 2
    assert boards["pagination"]["total"] == 1
    assert boards["boards"][0]["name"] == "Project 2"
    assert lists["lists"][0]["name"] == "Todo"
    assert cards["cards"][0]["title"] == "Task 2"
    assert loaded_card["priority"] == "high"

    with pytest.raises(ValueError, match=f"local_kanban_version_conflict:card:{card['id']}"):
        await service.update_card(card["id"], {"title": "Stale"}, expected_version=1)


@pytest.mark.asyncio
async def test_local_kanban_reorders_moves_and_copies_cards(tmp_path):
    service = LocalKanbanService(db_path=tmp_path / "kanban.db")
    board = await service.create_board({"name": "Project", "client_id": "board-1"})
    todo = await service.create_list(board["id"], {"name": "Todo", "client_id": "list-1"})
    done = await service.create_list(board["id"], {"name": "Done", "client_id": "list-2"})
    first = await service.create_card(todo["id"], {"title": "First", "client_id": "card-1"})
    second = await service.create_card(todo["id"], {"title": "Second", "client_id": "card-2"})

    await service.reorder_lists(board["id"], {"ids": [done["id"], todo["id"]]})
    await service.reorder_cards(todo["id"], {"ids": [second["id"], first["id"]]})
    moved = await service.move_card(first["id"], {"target_list_id": done["id"], "position": 0})
    copied = await service.copy_card(
        second["id"],
        {"target_list_id": done["id"], "new_client_id": "card-copy", "new_title": "Second Copy"},
    )

    lists = await service.list_lists(board["id"])
    todo_cards = await service.list_cards(todo["id"])
    done_cards = await service.list_cards(done["id"])

    assert [item["id"] for item in lists["lists"]] == [done["id"], todo["id"]]
    assert [item["id"] for item in todo_cards["cards"]] == [second["id"]]
    assert moved["list_id"] == done["id"]
    assert copied["title"] == "Second Copy"
    assert copied["client_id"] == "card-copy"
    assert [item["id"] for item in done_cards["cards"]] == [first["id"], copied["id"]]


@pytest.mark.asyncio
async def test_local_kanban_archives_deletes_restores_and_lists_activities(tmp_path):
    service = LocalKanbanService(db_path=tmp_path / "kanban.db")
    board = await service.create_board({"name": "Project", "client_id": "board-1"})
    kanban_list = await service.create_list(board["id"], {"name": "Todo", "client_id": "list-1"})
    card = await service.create_card(kanban_list["id"], {"title": "Task", "client_id": "card-1"})

    archived = await service.archive_card(card["id"])
    unarchived = await service.unarchive_card(card["id"])
    deleted = await service.delete_card(card["id"])
    restored = await service.restore_card(card["id"])
    await service.archive_list(kanban_list["id"])
    await service.unarchive_list(kanban_list["id"])
    await service.delete_board(board["id"])
    restored_board = await service.restore_board(board["id"])

    board_activities = await service.list_board_activities(board["id"])
    card_activities = await service.list_card_activities(card["id"])

    assert archived["archived"] is True
    assert unarchived["archived"] is False
    assert deleted["deleted"] is True
    assert restored["deleted"] is False
    assert restored_board["deleted"] is False
    assert [activity["action_type"] for activity in board_activities["activities"]] == [
        "create",
        "create",
        "create",
        "archive",
        "restore",
        "delete",
        "restore",
        "archive",
        "restore",
        "delete",
        "restore",
    ]
    assert any(activity["entity_type"] == "card" for activity in card_activities["activities"])
