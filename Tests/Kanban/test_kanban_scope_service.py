import pytest

from tldw_chatbook.Kanban_Interop.kanban_scope_service import KanbanScopeService
from tldw_chatbook.Kanban_Interop.server_kanban_service import KANBAN_OPERATION_SPECS
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


class FakeLocalKanbanService:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name not in KANBAN_OPERATION_SPECS:
            raise AttributeError(name)

        async def _operation(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"id": 1, "name": name}

        return _operation


class ExplodingServerKanbanService:
    def __getattr__(self, name):
        raise AssertionError(f"server should not dispatch {name}")


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

    with pytest.raises(ValueError, match="Local Kanban backend is unavailable"):
        await scope.list_boards(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_kanban_scope_service_dispatches_every_operation_locally():
    local = FakeLocalKanbanService()
    server = ExplodingServerKanbanService()
    policy = FakePolicyEnforcer()
    scope = KanbanScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    for operation_name, spec in KANBAN_OPERATION_SPECS.items():
        args = _minimal_args_for(spec)
        await scope.invoke(operation_name, *args, mode="local")

    assert [call[0] for call in local.calls] == list(KANBAN_OPERATION_SPECS)
    assert all(action_id.endswith(".local") for action_id in policy.calls)


@pytest.mark.asyncio
async def test_kanban_scope_service_blocks_denied_action_before_dispatch():
    server = FakeServerKanbanService()
    scope = KanbanScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await scope.list_boards(mode="server")

    assert server.calls == []


def test_kanban_scope_service_reports_local_and_server_contract_gaps():
    scope = KanbanScopeService(server_service=None)
    local_scope = KanbanScopeService(local_service=FakeLocalKanbanService(), server_service=None)

    local_report = scope.list_unsupported_capabilities(mode="local")
    ready_local_report = local_scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "kanban.local_backend_unavailable",
            "source": "local",
            "supported": False,
            "reason_code": "local_backend_unavailable",
            "user_message": "Local Kanban backend is unavailable.",
            "affected_action_ids": [],
        }
    ]
    assert ready_local_report == []
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


def _minimal_args_for(spec):
    request_index = spec.request_arg_index if spec.request_arg_index is not None else -1
    highest_index = max(tuple(spec.identifier_arg_indexes) + (request_index,))
    args = [1 for _ in range(highest_index + 1)]
    if spec.request_arg_index is not None:
        args[spec.request_arg_index] = {}
    return tuple(args)
