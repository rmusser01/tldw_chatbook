import pytest

from tldw_chatbook.Kanban_Interop.server_kanban_service import ServerKanbanService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeKanbanClient:
    def __init__(self):
        self.calls = []

    async def create_kanban_board(self, request_data):
        self.calls.append(("create_kanban_board", request_data))
        return {"id": 1, "name": request_data.name, "client_id": request_data.client_id}

    async def list_kanban_boards(self, **kwargs):
        self.calls.append(("list_kanban_boards", kwargs))
        return {"boards": [{"id": 1, "name": "Project Board"}], "pagination": {"total": 1}}

    async def get_kanban_board(self, board_id, **kwargs):
        self.calls.append(("get_kanban_board", board_id, kwargs))
        return {"id": board_id, "name": "Project Board", "lists": [{"id": 10, "board_id": board_id, "cards": [{"id": 100, "list_id": 10}]}]}

    async def update_kanban_board(self, board_id, request_data, **kwargs):
        self.calls.append(("update_kanban_board", board_id, request_data, kwargs))
        return {"id": board_id, "name": request_data.name, "version": kwargs.get("expected_version")}

    async def archive_kanban_board(self, board_id):
        self.calls.append(("archive_kanban_board", board_id))
        return {"id": board_id, "archived": True}

    async def delete_kanban_board(self, board_id):
        self.calls.append(("delete_kanban_board", board_id))
        return {"detail": "deleted"}

    async def create_kanban_list(self, board_id, request_data):
        self.calls.append(("create_kanban_list", board_id, request_data))
        return {"id": 10, "board_id": board_id, "name": request_data.name}

    async def list_kanban_lists(self, board_id, **kwargs):
        self.calls.append(("list_kanban_lists", board_id, kwargs))
        return {"lists": [{"id": 10, "board_id": board_id, "name": "Todo"}]}

    async def reorder_kanban_lists(self, board_id, request_data):
        self.calls.append(("reorder_kanban_lists", board_id, request_data))
        return {"success": True}

    async def create_kanban_card(self, list_id, request_data):
        self.calls.append(("create_kanban_card", list_id, request_data))
        return {"id": 100, "list_id": list_id, "title": request_data.title}

    async def get_kanban_card(self, card_id):
        self.calls.append(("get_kanban_card", card_id))
        return {"id": card_id, "list_id": 10, "title": "Task"}

    async def move_kanban_card(self, card_id, request_data):
        self.calls.append(("move_kanban_card", card_id, request_data))
        return {"id": card_id, "list_id": request_data.target_list_id, "title": "Task"}

    async def list_kanban_board_activities(self, board_id, **kwargs):
        self.calls.append(("list_kanban_board_activities", board_id, kwargs))
        return {"activities": [{"id": 900, "board_id": board_id, "action_type": "create"}], "pagination": {"total": 1}}

    async def export_kanban_board(self, board_id, request_data):
        self.calls.append(("export_kanban_board", board_id, request_data))
        return {"format": "json", "exported_at": "2026-04-25T00:00:00Z", "board": {"id": board_id}, "labels": [], "lists": []}

    async def import_kanban_board(self, request_data):
        self.calls.append(("import_kanban_board", request_data))
        return {"board": {"id": 2, "name": request_data.board_name or "Imported"}, "import_stats": {"board_id": 2}}

    async def create_kanban_label(self, board_id, request_data):
        self.calls.append(("create_kanban_label", board_id, request_data))
        return {"id": 7, "board_id": board_id, "name": request_data.name, "color": request_data.color}

    async def assign_kanban_label_to_card(self, card_id, label_id):
        self.calls.append(("assign_kanban_label_to_card", card_id, label_id))
        return {"detail": "assigned"}

    async def create_kanban_checklist(self, card_id, request_data):
        self.calls.append(("create_kanban_checklist", card_id, request_data))
        return {"id": 70, "card_id": card_id, "name": request_data.name}

    async def create_kanban_checklist_item(self, checklist_id, request_data):
        self.calls.append(("create_kanban_checklist_item", checklist_id, request_data))
        return {"id": 700, "checklist_id": checklist_id, "name": request_data.name}

    async def check_kanban_checklist_item(self, item_id):
        self.calls.append(("check_kanban_checklist_item", item_id))
        return {"id": item_id, "checklist_id": 70, "checked": True}

    async def create_kanban_comment(self, card_id, request_data):
        self.calls.append(("create_kanban_comment", card_id, request_data))
        return {"id": 80, "card_id": card_id, "content": request_data.content}

    async def search_kanban_cards(self, request_data):
        self.calls.append(("search_kanban_cards", request_data))
        return {"query": request_data.query, "search_mode": request_data.search_mode, "results": [{"id": 100, "card_id": 100, "title": "Task"}]}

    async def get_kanban_search_status(self):
        self.calls.append(("get_kanban_search_status",))
        return {"index_ready": True}

    async def add_kanban_card_link(self, card_id, request_data):
        self.calls.append(("add_kanban_card_link", card_id, request_data))
        return {"id": 55, "card_id": card_id, "linked_type": request_data.linked_type, "linked_id": request_data.linked_id}

    async def list_kanban_cards_by_linked_content(self, linked_type, linked_id, **kwargs):
        self.calls.append(("list_kanban_cards_by_linked_content", linked_type, linked_id, kwargs))
        return {"linked_type": linked_type, "linked_id": linked_id, "cards": [{"id": 100, "title": "Task", "link_id": 55}]}

    async def bulk_move_kanban_cards(self, request_data):
        self.calls.append(("bulk_move_kanban_cards", request_data))
        return {"success": True, "moved_count": len(request_data.card_ids), "cards": [{"id": card_id} for card_id in request_data.card_ids]}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_server_kanban_service_routes_core_and_subresources_with_policy_and_normalization():
    client = FakeKanbanClient()
    policy = FakePolicyEnforcer()
    service = ServerKanbanService(client, policy_enforcer=policy)

    board = await service.create_board({"name": "Project Board", "client_id": "board-1"})
    boards = await service.list_boards(include_archived=True)
    loaded_board = await service.get_board(1)
    updated_board = await service.update_board(1, {"name": "Renamed"}, expected_version=3)
    archived_board = await service.archive_board(1)
    deleted_board = await service.delete_board(1)
    kanban_list = await service.create_list(1, {"name": "Todo", "client_id": "list-1"})
    lists = await service.list_lists(1)
    reordered_lists = await service.reorder_lists(1, {"ids": [10]})
    card = await service.create_card(10, {"title": "Task", "client_id": "card-1"})
    loaded_card = await service.get_card(100)
    moved_card = await service.move_card(100, {"target_list_id": 11})
    activities = await service.list_board_activities(1)
    exported = await service.export_board(1, {"include_archived": True})
    imported = await service.import_board({"data": {"board": {}}, "board_name": "Imported"})
    label = await service.create_label(1, {"name": "Blocked", "color": "red"})
    assigned = await service.assign_label_to_card(100, 7)
    checklist = await service.create_checklist(100, {"name": "Steps"})
    checklist_item = await service.create_checklist_item(70, {"name": "Step 1"})
    checked_item = await service.check_checklist_item(700)
    comment = await service.create_comment(100, {"content": "Looks good"})
    search = await service.search_cards({"query": "Task"})
    search_status = await service.get_search_status()
    link = await service.add_card_link(100, {"linked_type": "note", "linked_id": "note-1"})
    linked_cards = await service.list_cards_by_linked_content("note", "note-1")
    bulk_move = await service.bulk_move_cards({"card_ids": [100, 101], "target_list_id": 11})

    assert board["record_id"] == "server:kanban_board:1"
    assert boards["boards"][0]["record_id"] == "server:kanban_board:1"
    assert loaded_board["lists"][0]["record_id"] == "server:kanban_list:10"
    assert loaded_board["lists"][0]["cards"][0]["record_id"] == "server:kanban_card:100"
    assert updated_board["version"] == 3
    assert archived_board["record_id"] == "server:kanban_board:1"
    assert deleted_board["record_id"] == "server:kanban_board:1"
    assert kanban_list["record_id"] == "server:kanban_list:10"
    assert lists["lists"][0]["record_id"] == "server:kanban_list:10"
    assert reordered_lists["record_id"] == "server:kanban_reorder:1"
    assert card["record_id"] == "server:kanban_card:100"
    assert loaded_card["record_id"] == "server:kanban_card:100"
    assert moved_card["list_id"] == 11
    assert activities["activities"][0]["record_id"] == "server:kanban_activity:900"
    assert exported["record_id"] == "server:kanban_board_export:1"
    assert imported["record_id"] == "server:kanban_board_import:2"
    assert label["record_id"] == "server:kanban_label:7"
    assert assigned["record_id"] == "server:kanban_card_label:100:7"
    assert checklist["record_id"] == "server:kanban_checklist:70"
    assert checklist_item["record_id"] == "server:kanban_checklist_item:700"
    assert checked_item["checked"] is True
    assert comment["record_id"] == "server:kanban_comment:80"
    assert search["results"][0]["record_id"] == "server:kanban_search_result:100"
    assert search_status["record_id"] == "server:kanban_search_status:active"
    assert link["record_id"] == "server:kanban_card_link:55"
    assert linked_cards["cards"][0]["record_id"] == "server:kanban_card:100"
    assert bulk_move["cards"][0]["record_id"] == "server:kanban_card:100"
    assert policy.calls == [
        "kanban.boards.create.server",
        "kanban.boards.list.server",
        "kanban.boards.detail.server",
        "kanban.boards.update.server",
        "kanban.boards.archive.server",
        "kanban.boards.delete.server",
        "kanban.lists.create.server",
        "kanban.lists.list.server",
        "kanban.lists.reorder.server",
        "kanban.cards.create.server",
        "kanban.cards.detail.server",
        "kanban.cards.launch.server",
        "kanban.activities.list.server",
        "kanban.boards.export.server",
        "kanban.boards.import.server",
        "kanban.labels.create.server",
        "kanban.card_labels.create.server",
        "kanban.checklists.create.server",
        "kanban.checklist_items.create.server",
        "kanban.checklist_items.update.server",
        "kanban.comments.create.server",
        "kanban.search.list.server",
        "kanban.search.detail.server",
        "kanban.card_links.create.server",
        "kanban.card_links.list.server",
        "kanban.cards.update.server",
    ]


@pytest.mark.asyncio
async def test_server_kanban_service_denies_before_dispatch():
    client = FakeKanbanClient()
    service = ServerKanbanService(client, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await service.list_boards()

    assert client.calls == []
