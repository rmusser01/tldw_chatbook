from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    KanbanActivitiesListResponse,
    KanbanBoardExportRequest,
    KanbanBoardExportResponse,
    KanbanBoardImportRequest,
    KanbanBoardImportResponse,
    KanbanBulkArchiveCardsRequest,
    KanbanBulkArchiveCardsResponse,
    KanbanBulkCardLinksAddResponse,
    KanbanBulkCardLinksRemoveResponse,
    KanbanBulkCardLinksRequest,
    KanbanBulkDeleteCardsRequest,
    KanbanBulkDeleteCardsResponse,
    KanbanBulkLabelCardsRequest,
    KanbanBulkLabelCardsResponse,
    KanbanBulkMoveCardsRequest,
    KanbanBulkMoveCardsResponse,
    KanbanBulkUnarchiveCardsResponse,
    KanbanCardCopyWithChecklistsRequest,
    KanbanCardLinkCountsResponse,
    KanbanCardLinkCreate,
    KanbanCardLinkResponse,
    KanbanCardLinksListResponse,
    KanbanCardResponse,
    KanbanCardSearchRequest,
    KanbanCardSearchResponse,
    KanbanChecklistCreate,
    KanbanChecklistItemCreate,
    KanbanChecklistItemReorderRequest,
    KanbanChecklistItemResponse,
    KanbanChecklistItemsListResponse,
    KanbanChecklistItemUpdate,
    KanbanChecklistReorderRequest,
    KanbanChecklistResponse,
    KanbanChecklistsListResponse,
    KanbanChecklistUpdate,
    KanbanChecklistWithItemsResponse,
    KanbanCommentCreate,
    KanbanCommentResponse,
    KanbanCommentsListResponse,
    KanbanCommentUpdate,
    KanbanFilteredCardsResponse,
    KanbanLabelCreate,
    KanbanLabelResponse,
    KanbanLabelsListResponse,
    KanbanLabelUpdate,
    KanbanLinkedCardsListResponse,
    KanbanSearchRequest,
    KanbanSearchResponse,
    KanbanToggleAllChecklistItemsRequest,
    TLDWAPIClient,
)


def _pagination(total: int = 1, limit: int = 50, offset: int = 0) -> dict:
    return {"total": total, "limit": limit, "offset": offset, "has_more": False}


def _board_payload(**overrides) -> dict:
    payload = {
        "id": 1,
        "uuid": "board-uuid",
        "user_id": "user-1",
        "client_id": "board-client",
        "name": "Launch",
        "description": None,
        "archived": False,
        "archived_at": None,
        "activity_retention_days": 90,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
        "deleted": False,
        "deleted_at": None,
        "version": 1,
        "metadata": {},
    }
    payload.update(overrides)
    return payload


def _card_payload(**overrides) -> dict:
    payload = {
        "id": 31,
        "uuid": "card-uuid",
        "board_id": 1,
        "list_id": 21,
        "client_id": "card-client",
        "title": "Ship",
        "description": "Ship the feature",
        "position": 0,
        "due_date": None,
        "due_complete": False,
        "start_date": None,
        "priority": "high",
        "archived": False,
        "archived_at": None,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
        "deleted": False,
        "deleted_at": None,
        "version": 1,
        "metadata": {},
    }
    payload.update(overrides)
    return payload


def _label_payload(**overrides) -> dict:
    payload = {
        "id": 41,
        "uuid": "label-uuid",
        "board_id": 1,
        "name": "Bug",
        "color": "red",
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
    }
    payload.update(overrides)
    return payload


def _checklist_payload(**overrides) -> dict:
    payload = {
        "id": 51,
        "uuid": "checklist-uuid",
        "card_id": 31,
        "name": "QA",
        "position": 0,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
    }
    payload.update(overrides)
    return payload


def _checklist_item_payload(**overrides) -> dict:
    payload = {
        "id": 61,
        "uuid": "item-uuid",
        "checklist_id": 51,
        "name": "Regression test",
        "position": 0,
        "checked": False,
        "checked_at": None,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
    }
    payload.update(overrides)
    return payload


def _comment_payload(**overrides) -> dict:
    payload = {
        "id": 71,
        "uuid": "comment-uuid",
        "card_id": 31,
        "user_id": "user-1",
        "content": "Looks good",
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
        "deleted": False,
    }
    payload.update(overrides)
    return payload


def _activity_payload(**overrides) -> dict:
    payload = {
        "id": 81,
        "uuid": "activity-uuid",
        "board_id": 1,
        "list_id": 21,
        "card_id": 31,
        "user_id": "user-1",
        "action_type": "create",
        "entity_type": "card",
        "entity_id": 31,
        "details": {"title": "Ship"},
        "created_at": "2026-04-25T12:00:00Z",
    }
    payload.update(overrides)
    return payload


def _link_payload(**overrides) -> dict:
    payload = {
        "id": 91,
        "card_id": 31,
        "linked_type": "note",
        "linked_id": "note-1",
        "created_at": "2026-04-25T12:00:00Z",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_kanban_client_routes_labels_comments_checklists_and_items(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _label_payload(),
            {"labels": [_label_payload()]},
            _label_payload(),
            _label_payload(name="Docs"),
            {"detail": "Label assigned to card"},
            {"labels": [_label_payload()]},
            _comment_payload(),
            {"comments": [_comment_payload()], "pagination": _pagination(limit=25)},
            _comment_payload(),
            _comment_payload(content="Updated"),
            _checklist_payload(),
            {"checklists": [_checklist_payload()]},
            {**_checklist_payload(), "items": [_checklist_item_payload()], "total_items": 1, "checked_items": 0, "progress_percent": 0},
            _checklist_payload(name="Launch QA"),
            {"checklists": [_checklist_payload()]},
            _checklist_item_payload(),
            {"items": [_checklist_item_payload()]},
            _checklist_item_payload(),
            _checklist_item_payload(checked=True),
            {"items": [_checklist_item_payload()]},
            _checklist_item_payload(checked=True),
            _checklist_item_payload(checked=False),
            {**_checklist_payload(), "items": [_checklist_item_payload(checked=True)], "total_items": 1, "checked_items": 1, "progress_percent": 100},
            {},
            {},
            {},
            {},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created_label = await client.create_kanban_label(1, KanbanLabelCreate(name="Bug", color="red"))
    labels = await client.list_kanban_labels(1)
    fetched_label = await client.get_kanban_label(41)
    updated_label = await client.update_kanban_label(41, KanbanLabelUpdate(name="Docs"))
    assigned = await client.assign_kanban_label_to_card(31, 41)
    card_labels = await client.list_kanban_card_labels(31)
    comment = await client.create_kanban_comment(31, KanbanCommentCreate(content="Looks good"))
    comments = await client.list_kanban_comments(31, limit=25, include_deleted=True)
    fetched_comment = await client.get_kanban_comment(71, include_deleted=True)
    updated_comment = await client.update_kanban_comment(71, KanbanCommentUpdate(content="Updated"))
    checklist = await client.create_kanban_checklist(31, KanbanChecklistCreate(name="QA", position=0))
    checklists = await client.list_kanban_checklists(31)
    fetched_checklist = await client.get_kanban_checklist(51)
    updated_checklist = await client.update_kanban_checklist(51, KanbanChecklistUpdate(name="Launch QA"))
    reordered_checklists = await client.reorder_kanban_checklists(31, KanbanChecklistReorderRequest(checklist_ids=[51]))
    item = await client.create_kanban_checklist_item(51, KanbanChecklistItemCreate(name="Regression test"))
    items = await client.list_kanban_checklist_items(51)
    fetched_item = await client.get_kanban_checklist_item(61)
    updated_item = await client.update_kanban_checklist_item(61, KanbanChecklistItemUpdate(checked=True))
    reordered_items = await client.reorder_kanban_checklist_items(51, KanbanChecklistItemReorderRequest(item_ids=[61]))
    checked_item = await client.check_kanban_checklist_item(61)
    unchecked_item = await client.uncheck_kanban_checklist_item(61)
    toggled = await client.toggle_all_kanban_checklist_items(51, KanbanToggleAllChecklistItemsRequest(checked=True))
    removed_label = await client.remove_kanban_label_from_card(31, 41)
    deleted_comment = await client.delete_kanban_comment(71, hard_delete=True)
    deleted_item = await client.delete_kanban_checklist_item(61)
    deleted_checklist = await client.delete_kanban_checklist(51)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/kanban/boards/1/labels")
    assert mocked.await_args_list[0].kwargs["json_data"] == {"name": "Bug", "color": "red"}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/kanban/boards/1/labels")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/kanban/cards/31/labels/41")
    assert mocked.await_args_list[7].kwargs["params"] == {"limit": 25, "offset": 0, "include_deleted": True}
    assert mocked.await_args_list[13].kwargs["json_data"] == {"name": "Launch QA"}
    assert mocked.await_args_list[14].args[:2] == ("POST", "/api/v1/kanban/cards/31/checklists/reorder")
    assert mocked.await_args_list[19].args[:2] == ("POST", "/api/v1/kanban/checklists/51/items/reorder")
    assert mocked.await_args_list[22].kwargs["json_data"] == {"checked": True}
    assert mocked.await_args_list[23].args[:2] == ("DELETE", "/api/v1/kanban/cards/31/labels/41")
    assert mocked.await_args_list[24].kwargs["params"] == {"hard_delete": True}
    assert mocked.await_args_list[25].args[:2] == ("DELETE", "/api/v1/kanban/checklist-items/61")
    assert mocked.await_args_list[26].args[:2] == ("DELETE", "/api/v1/kanban/checklists/51")

    assert isinstance(created_label, KanbanLabelResponse)
    assert isinstance(labels, KanbanLabelsListResponse)
    assert isinstance(fetched_label, KanbanLabelResponse)
    assert isinstance(updated_label, KanbanLabelResponse)
    assert assigned.detail == "Label assigned to card"
    assert isinstance(card_labels, KanbanLabelsListResponse)
    assert isinstance(comment, KanbanCommentResponse)
    assert isinstance(comments, KanbanCommentsListResponse)
    assert isinstance(fetched_comment, KanbanCommentResponse)
    assert isinstance(updated_comment, KanbanCommentResponse)
    assert isinstance(checklist, KanbanChecklistResponse)
    assert isinstance(checklists, KanbanChecklistsListResponse)
    assert isinstance(fetched_checklist, KanbanChecklistWithItemsResponse)
    assert isinstance(updated_checklist, KanbanChecklistResponse)
    assert isinstance(reordered_checklists, KanbanChecklistsListResponse)
    assert isinstance(item, KanbanChecklistItemResponse)
    assert isinstance(items, KanbanChecklistItemsListResponse)
    assert isinstance(fetched_item, KanbanChecklistItemResponse)
    assert isinstance(updated_item, KanbanChecklistItemResponse)
    assert isinstance(reordered_items, KanbanChecklistItemsListResponse)
    assert checked_item.checked is True
    assert unchecked_item.checked is False
    assert isinstance(toggled, KanbanChecklistWithItemsResponse)
    assert removed_label is True
    assert deleted_comment is True
    assert deleted_item is True
    assert deleted_checklist is True


@pytest.mark.asyncio
async def test_kanban_client_routes_activity_search_bulk_import_export_and_links(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    search_payload = {
        "query": "ship",
        "search_mode": "fts",
        "results": [
            {
                "id": 31,
                "uuid": "card-uuid",
                "board_id": 1,
                "board_name": "Launch",
                "list_id": 21,
                "list_name": "Doing",
                "title": "Ship",
                "description": "Ship the feature",
                "priority": "high",
                "due_date": None,
                "labels": [_label_payload()],
                "created_at": "2026-04-25T12:00:00Z",
                "updated_at": "2026-04-25T12:00:00Z",
                "relevance_score": 0.9,
            }
        ],
        "pagination": _pagination(limit=10),
    }
    export_payload = {
        "format": "tldw_kanban_v1",
        "exported_at": "2026-04-25T12:00:00Z",
        "board": _board_payload(),
        "labels": [_label_payload()],
        "lists": [],
    }
    linked_cards_payload = {
        "linked_type": "note",
        "linked_id": "note-1",
        "cards": [
            {
                "id": 31,
                "title": "Ship",
                "description": "Ship the feature",
                "board_id": 1,
                "board_name": "Launch",
                "list_id": 21,
                "list_name": "Doing",
                "position": 0,
                "is_archived": False,
                "is_deleted": False,
                "link_id": 91,
                "linked_at": "2026-04-25T12:00:00Z",
            }
        ],
    }
    mocked = AsyncMock(
        side_effect=[
            {"activities": [_activity_payload()], "pagination": _pagination(limit=20)},
            {"activities": [_activity_payload()], "pagination": _pagination(limit=10)},
            export_payload,
            export_payload,
            {"board": _board_payload(), "import_stats": {"board_id": 1, "lists_imported": 1, "cards_imported": 1, "labels_imported": 1, "checklists_imported": 0, "checklist_items_imported": 0, "comments_imported": 0}},
            {"cards": [_card_payload()], "pagination": _pagination(limit=50)},
            {"cards": [_card_payload()], "pagination": _pagination(limit=25)},
            search_payload,
            search_payload,
            {"fts_available": True, "supported_modes": ["fts", "vector", "hybrid"]},
            _card_payload(title="Copy"),
            {"success": True, "moved_count": 1, "cards": [_card_payload(list_id=22)]},
            {"success": True, "archived_count": 1},
            {"success": True, "unarchived_count": 1},
            {"success": True, "deleted_count": 1},
            {"success": True, "updated_count": 1},
            _link_payload(),
            {"links": [_link_payload()]},
            {"media": 0, "note": 1},
            {"detail": "Link removed successfully"},
            {"detail": "Link removed successfully"},
            {"detail": "Link removed successfully"},
            {"added_count": 1, "skipped_count": 0, "links": [_link_payload()]},
            {"removed_count": 1},
            linked_cards_payload,
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    board_activities = await client.list_kanban_board_activities(1, card_id=31, limit=20)
    card_activities = await client.list_kanban_card_activities(31, action_type="update", limit=10)
    export_get = await client.get_kanban_board_export(1, include_archived=True)
    export_post = await client.export_kanban_board(1, KanbanBoardExportRequest(include_deleted=True))
    imported = await client.import_kanban_board(KanbanBoardImportRequest(data={"format": "tldw_kanban_v1"}))
    filtered = await client.filter_kanban_board_cards(1, label_ids=[41, 42], priority="high")
    basic_search = await client.search_kanban_cards_basic(KanbanCardSearchRequest(query="ship", board_id=1, limit=25))
    search_get = await client.search_kanban_cards_get("ship", label_ids=[41], search_mode="fts", limit=10)
    search_post = await client.search_kanban_cards(KanbanSearchRequest(query="ship", label_ids=[41], limit=10))
    search_status = await client.get_kanban_search_status()
    copied = await client.copy_kanban_card_with_checklists(
        31,
        KanbanCardCopyWithChecklistsRequest(target_list_id=22, new_client_id="copy-1", copy_checklists=True, copy_labels=False),
    )
    bulk_moved = await client.bulk_move_kanban_cards(KanbanBulkMoveCardsRequest(card_ids=[31], target_list_id=22))
    bulk_archived = await client.bulk_archive_kanban_cards(KanbanBulkArchiveCardsRequest(card_ids=[31]))
    bulk_unarchived = await client.bulk_unarchive_kanban_cards(KanbanBulkArchiveCardsRequest(card_ids=[31]))
    bulk_deleted = await client.bulk_delete_kanban_cards(KanbanBulkDeleteCardsRequest(card_ids=[31]))
    bulk_labeled = await client.bulk_label_kanban_cards(KanbanBulkLabelCardsRequest(card_ids=[31], add_label_ids=[41]))
    link = await client.add_kanban_card_link(31, KanbanCardLinkCreate(linked_type="note", linked_id="note-1"))
    links = await client.list_kanban_card_links(31, linked_type="note")
    counts = await client.get_kanban_card_link_counts(31)
    removed = await client.remove_kanban_card_link(31, "note", "note-1")
    removed_scoped = await client.remove_kanban_card_link_by_id_for_card(31, 91)
    removed_by_id = await client.remove_kanban_card_link_by_id(91)
    bulk_added_links = await client.bulk_add_kanban_card_links(
        31,
        KanbanBulkCardLinksRequest(links=[KanbanCardLinkCreate(linked_type="note", linked_id="note-1")]),
    )
    bulk_removed_links = await client.bulk_remove_kanban_card_links(
        31,
        KanbanBulkCardLinksRequest(links=[KanbanCardLinkCreate(linked_type="note", linked_id="note-1")]),
    )
    linked_cards = await client.list_kanban_cards_by_linked_content("note", "note-1", include_archived=True)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/kanban/1/activities")
    assert mocked.await_args_list[0].kwargs["params"] == {"card_id": 31, "limit": 20, "offset": 0}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/kanban/1/export")
    assert mocked.await_args_list[2].kwargs["params"] == {"include_archived": True, "include_deleted": False}
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/kanban/1/export")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"include_archived": False, "include_deleted": True}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/kanban/import")
    assert mocked.await_args_list[5].kwargs["params"]["label_ids"] == "41,42"
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/kanban/cards/search")
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/kanban/search")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/kanban/search")
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/kanban/cards/31/copy-with-checklists")
    assert mocked.await_args_list[11].args[:2] == ("POST", "/api/v1/kanban/cards/bulk-move")
    assert mocked.await_args_list[16].args[:2] == ("POST", "/api/v1/kanban/cards/31/links")
    assert mocked.await_args_list[17].kwargs["params"] == {"linked_type": "note"}
    assert mocked.await_args_list[19].args[:2] == ("DELETE", "/api/v1/kanban/cards/31/links/note/note-1")
    assert mocked.await_args_list[24].kwargs["params"] == {"include_archived": True, "include_deleted": False}

    assert isinstance(board_activities, KanbanActivitiesListResponse)
    assert isinstance(card_activities, KanbanActivitiesListResponse)
    assert isinstance(export_get, KanbanBoardExportResponse)
    assert isinstance(export_post, KanbanBoardExportResponse)
    assert isinstance(imported, KanbanBoardImportResponse)
    assert isinstance(filtered, KanbanFilteredCardsResponse)
    assert isinstance(basic_search, KanbanCardSearchResponse)
    assert isinstance(search_get, KanbanSearchResponse)
    assert isinstance(search_post, KanbanSearchResponse)
    assert search_status["fts_available"] is True
    assert isinstance(copied, KanbanCardResponse)
    assert isinstance(bulk_moved, KanbanBulkMoveCardsResponse)
    assert isinstance(bulk_archived, KanbanBulkArchiveCardsResponse)
    assert isinstance(bulk_unarchived, KanbanBulkUnarchiveCardsResponse)
    assert isinstance(bulk_deleted, KanbanBulkDeleteCardsResponse)
    assert isinstance(bulk_labeled, KanbanBulkLabelCardsResponse)
    assert isinstance(link, KanbanCardLinkResponse)
    assert isinstance(links, KanbanCardLinksListResponse)
    assert isinstance(counts, KanbanCardLinkCountsResponse)
    assert removed.detail == "Link removed successfully"
    assert removed_scoped.detail == "Link removed successfully"
    assert removed_by_id.detail == "Link removed successfully"
    assert isinstance(bulk_added_links, KanbanBulkCardLinksAddResponse)
    assert isinstance(bulk_removed_links, KanbanBulkCardLinksRemoveResponse)
    assert isinstance(linked_cards, KanbanLinkedCardsListResponse)


@pytest.mark.asyncio
async def test_kanban_client_routes_label_delete_and_basic_search_get(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {},
            {"cards": [_card_payload()], "pagination": _pagination(limit=5)},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    deleted = await client.delete_kanban_label(41)
    searched = await client.search_kanban_cards_basic_get("ship", board_id=1, limit=5)

    assert mocked.await_args_list[0].args[:2] == ("DELETE", "/api/v1/kanban/labels/41")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/kanban/cards/search")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "q": "ship",
        "board_id": 1,
        "limit": 5,
        "offset": 0,
    }
    assert deleted is True
    assert isinstance(searched, KanbanCardSearchResponse)
