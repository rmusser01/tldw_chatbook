import pytest

from tldw_chatbook.Kanban_Interop.kanban_scope_service import KanbanScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerKanbanService:
    def __init__(self):
        self.calls = []

    async def list_boards(self, **kwargs):
        self.calls.append(("list_boards", kwargs))
        return {"boards": [{"id": 1, "name": "Project Board"}]}

    async def create_card(self, list_id, request_data):
        self.calls.append(("create_card", list_id, request_data))
        return {"id": 100, "list_id": list_id, "title": request_data["title"]}

    async def search_cards(self, request_data):
        self.calls.append(("search_cards", request_data))
        return {"query": request_data["query"], "results": [{"id": 100, "title": "Task"}]}

    async def get_search_status(self):
        self.calls.append(("get_search_status",))
        return {"index_ready": True}


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
async def test_kanban_scope_service_routes_server_and_normalizes_records():
    server = FakeServerKanbanService()
    policy = FakePolicyEnforcer()
    scope = KanbanScopeService(server_service=server, policy_enforcer=policy)

    boards = await scope.list_boards(mode="server", include_archived=True)
    card = await scope.create_card(10, {"title": "Task", "client_id": "card-1"}, mode="server")
    search = await scope.search_cards({"query": "Task"}, mode="server")
    search_status = await scope.get_search_status(mode="server")

    assert boards["boards"][0]["record_id"] == "server:kanban_board:1"
    assert card["record_id"] == "server:kanban_card:100"
    assert search["results"][0]["record_id"] == "server:kanban_search_result:100"
    assert search_status["record_id"] == "server:kanban_search_status:active"
    assert server.calls == [
        ("list_boards", {"include_archived": True}),
        ("create_card", 10, {"title": "Task", "client_id": "card-1"}),
        ("search_cards", {"query": "Task"}),
        ("get_search_status",),
    ]
    assert policy.calls == [
        "kanban.boards.list.server",
        "kanban.cards.create.server",
        "kanban.search.list.server",
        "kanban.search.detail.server",
    ]


@pytest.mark.asyncio
async def test_kanban_scope_service_rejects_local_mode_without_dispatch():
    server = FakeServerKanbanService()
    scope = KanbanScopeService(server_service=server)

    with pytest.raises(ValueError, match="server-only"):
        await scope.list_boards(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_kanban_scope_service_blocks_denied_action_before_dispatch():
    server = FakeServerKanbanService()
    scope = KanbanScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await scope.list_boards(mode="server")

    assert server.calls == []


def test_kanban_scope_service_reports_local_and_server_contract_gaps():
    scope = KanbanScopeService(server_service=None)

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "kanban.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server Kanban boards, lists, cards, labels, comments, checklists, links, search, activity, import/export, and bulk operations are unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
    assert server_report == [
        {
            "operation_id": "kanban.workflow_controls.server",
            "source": "server",
            "supported": False,
            "reason_code": "deferred_workflows_surface",
            "user_message": "Kanban board/list/card REST operations are available; Kanban workflow controls remain deferred with the broader workflows surface.",
            "affected_action_ids": [],
        }
    ]
