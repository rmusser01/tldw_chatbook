from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    KanbanBoardCreate,
    KanbanBoardUpdate,
    KanbanCardCopyRequest,
    KanbanCardCreate,
    KanbanCardMoveRequest,
    KanbanCardUpdate,
    KanbanListCreate,
    KanbanListUpdate,
    KanbanReorderRequest,
    TLDWAPIClient,
)


NOW = "2026-04-25T12:00:00Z"


def _board_payload(board_id: int = 11, **overrides) -> dict:
    payload = {
        "id": board_id,
        "uuid": f"board-{board_id}",
        "user_id": "user-1",
        "client_id": f"local-board-{board_id}",
        "name": "Launch Board",
        "description": "Release work",
        "archived": False,
        "archived_at": None,
        "activity_retention_days": 30,
        "created_at": NOW,
        "updated_at": NOW,
        "deleted": False,
        "deleted_at": None,
        "version": 3,
        "metadata": {"project": "launch"},
        "list_count": 2,
        "card_count": 4,
    }
    payload.update(overrides)
    return payload


def _list_payload(list_id: int = 21, **overrides) -> dict:
    payload = {
        "id": list_id,
        "uuid": f"list-{list_id}",
        "board_id": 11,
        "client_id": f"local-list-{list_id}",
        "name": "Doing",
        "position": 1,
        "archived": False,
        "archived_at": None,
        "created_at": NOW,
        "updated_at": NOW,
        "deleted": False,
        "deleted_at": None,
        "version": 2,
        "card_count": 3,
    }
    payload.update(overrides)
    return payload


def _card_payload(card_id: int = 31, **overrides) -> dict:
    payload = {
        "id": card_id,
        "uuid": f"card-{card_id}",
        "board_id": 11,
        "list_id": 21,
        "client_id": f"local-card-{card_id}",
        "title": "Wire API client",
        "description": "Add backend parity",
        "position": 0,
        "due_date": None,
        "due_complete": False,
        "start_date": None,
        "priority": "high",
        "archived": False,
        "archived_at": None,
        "created_at": NOW,
        "updated_at": NOW,
        "deleted": False,
        "deleted_at": None,
        "version": 5,
        "metadata": {"source": "chatbook"},
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_kanban_board_client_routes_crud_archive_restore_and_detail(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"boards": [_board_payload()], "pagination": {"total": 1, "limit": 25, "offset": 5, "has_more": False}},
            _board_payload(lists=[], labels=[], total_cards=0),
            _board_payload(),
            _board_payload(name="Launch v2"),
            _board_payload(archived=True),
            _board_payload(archived=False),
            {"detail": "Board 11 deleted successfully"},
            _board_payload(deleted=False),
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_kanban_boards(include_archived=True, include_deleted=False, limit=25, offset=5)
    fetched = await client.get_kanban_board(11, include_lists=True, include_cards=False, include_archived=True)
    created = await client.create_kanban_board(
        KanbanBoardCreate(
            name="Launch Board",
            description="Release work",
            client_id="local-board-11",
            activity_retention_days=30,
            metadata={"project": "launch"},
        )
    )
    updated = await client.update_kanban_board(11, KanbanBoardUpdate(name="Launch v2"), expected_version=3)
    archived = await client.archive_kanban_board(11)
    unarchived = await client.unarchive_kanban_board(11)
    deleted = await client.delete_kanban_board(11)
    restored = await client.restore_kanban_board(11)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/kanban/boards")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "include_archived": True,
        "include_deleted": False,
        "limit": 25,
        "offset": 5,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/kanban/boards/11")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "include_lists": True,
        "include_cards": False,
        "include_archived": True,
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/kanban/boards")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "name": "Launch Board",
        "description": "Release work",
        "client_id": "local-board-11",
        "activity_retention_days": 30,
        "metadata": {"project": "launch"},
    }
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/kanban/boards/11")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"name": "Launch v2"}
    assert mocked.await_args_list[3].kwargs["headers"] == {"X-Expected-Version": "3"}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/kanban/boards/11/archive")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/kanban/boards/11/unarchive")
    assert mocked.await_args_list[6].args[:2] == ("DELETE", "/api/v1/kanban/boards/11")
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/kanban/boards/11/restore")

    assert listed.pagination.total == 1
    assert fetched.id == 11
    assert created.metadata == {"project": "launch"}
    assert updated.name == "Launch v2"
    assert archived.archived is True
    assert unarchived.archived is False
    assert deleted.detail == "Board 11 deleted successfully"
    assert restored.id == 11


@pytest.mark.asyncio
async def test_kanban_list_client_routes_crud_archive_restore_and_reorder(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"lists": [_list_payload()]},
            _list_payload(),
            {"success": True, "message": "Lists reordered successfully"},
            _list_payload(),
            _list_payload(name="Done"),
            _list_payload(archived=True),
            _list_payload(archived=False),
            {"detail": "List 21 deleted successfully"},
            _list_payload(deleted=False),
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_kanban_lists(11, include_archived=True, include_deleted=True)
    created = await client.create_kanban_list(11, KanbanListCreate(name="Doing", client_id="local-list-21", position=1))
    reordered = await client.reorder_kanban_lists(11, KanbanReorderRequest(ids=[21, 22]))
    fetched = await client.get_kanban_list(21)
    updated = await client.update_kanban_list(21, KanbanListUpdate(name="Done"), expected_version=2)
    archived = await client.archive_kanban_list(21)
    unarchived = await client.unarchive_kanban_list(21)
    deleted = await client.delete_kanban_list(21)
    restored = await client.restore_kanban_list(21)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/kanban/boards/11/lists")
    assert mocked.await_args_list[0].kwargs["params"] == {"include_archived": True, "include_deleted": True}
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/kanban/boards/11/lists")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "name": "Doing",
        "client_id": "local-list-21",
        "position": 1,
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/kanban/boards/11/lists/reorder")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"ids": [21, 22]}
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/kanban/lists/21")
    assert mocked.await_args_list[4].args[:2] == ("PATCH", "/api/v1/kanban/lists/21")
    assert mocked.await_args_list[4].kwargs["headers"] == {"X-Expected-Version": "2"}
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/kanban/lists/21/archive")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/kanban/lists/21/unarchive")
    assert mocked.await_args_list[7].args[:2] == ("DELETE", "/api/v1/kanban/lists/21")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/kanban/lists/21/restore")

    assert listed.lists[0].id == 21
    assert created.position == 1
    assert reordered.success is True
    assert fetched.id == 21
    assert updated.name == "Done"
    assert archived.archived is True
    assert unarchived.archived is False
    assert deleted.detail == "List 21 deleted successfully"
    assert restored.id == 21


@pytest.mark.asyncio
async def test_kanban_card_client_routes_crud_move_copy_archive_restore_and_reorder(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"cards": [_card_payload()]},
            _card_payload(),
            {"success": True, "message": "Cards reordered successfully"},
            {**_card_payload(), "labels": [], "checklists": [], "comment_count": 0},
            _card_payload(title="Wire client wrappers"),
            _card_payload(list_id=22, position=0),
            _card_payload(32, title="Copy of Wire API client"),
            _card_payload(archived=True),
            _card_payload(archived=False),
            {"detail": "Card 31 deleted successfully"},
            _card_payload(deleted=False),
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_kanban_cards(21, include_archived=True, include_deleted=False)
    created = await client.create_kanban_card(
        21,
        KanbanCardCreate(
            title="Wire API client",
            description="Add backend parity",
            client_id="local-card-31",
            position=0,
            priority="high",
            label_ids=[1, 2],
            metadata={"source": "chatbook"},
        ),
    )
    reordered = await client.reorder_kanban_cards(21, KanbanReorderRequest(ids=[31, 32]))
    fetched = await client.get_kanban_card(31)
    updated = await client.update_kanban_card(
        31,
        KanbanCardUpdate(title="Wire client wrappers"),
        expected_version=5,
    )
    moved = await client.move_kanban_card(31, KanbanCardMoveRequest(target_list_id=22, position=0))
    copied = await client.copy_kanban_card(
        31,
        KanbanCardCopyRequest(target_list_id=22, new_client_id="local-card-copy-32", new_title="Copy of Wire API client"),
    )
    archived = await client.archive_kanban_card(31)
    unarchived = await client.unarchive_kanban_card(31)
    deleted = await client.delete_kanban_card(31)
    restored = await client.restore_kanban_card(31)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/kanban/lists/21/cards")
    assert mocked.await_args_list[0].kwargs["params"] == {"include_archived": True, "include_deleted": False}
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/kanban/lists/21/cards")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "title": "Wire API client",
        "description": "Add backend parity",
        "client_id": "local-card-31",
        "position": 0,
        "priority": "high",
        "label_ids": [1, 2],
        "metadata": {"source": "chatbook"},
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/kanban/lists/21/cards/reorder")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"ids": [31, 32]}
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/kanban/cards/31")
    assert mocked.await_args_list[4].args[:2] == ("PATCH", "/api/v1/kanban/cards/31")
    assert mocked.await_args_list[4].kwargs["json_data"] == {"title": "Wire client wrappers"}
    assert mocked.await_args_list[4].kwargs["headers"] == {"X-Expected-Version": "5"}
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/kanban/cards/31/move")
    assert mocked.await_args_list[5].kwargs["json_data"] == {"target_list_id": 22, "position": 0}
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/kanban/cards/31/copy")
    assert mocked.await_args_list[6].kwargs["json_data"] == {
        "target_list_id": 22,
        "new_client_id": "local-card-copy-32",
        "new_title": "Copy of Wire API client",
    }
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/kanban/cards/31/archive")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/kanban/cards/31/unarchive")
    assert mocked.await_args_list[9].args[:2] == ("DELETE", "/api/v1/kanban/cards/31")
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/kanban/cards/31/restore")

    assert listed.cards[0].id == 31
    assert created.priority == "high"
    assert reordered.success is True
    assert fetched.comment_count == 0
    assert updated.title == "Wire client wrappers"
    assert moved.list_id == 22
    assert copied.id == 32
    assert archived.archived is True
    assert unarchived.archived is False
    assert deleted.detail == "Card 31 deleted successfully"
    assert restored.id == 31
