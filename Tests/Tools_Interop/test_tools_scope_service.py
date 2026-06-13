import pytest

from tldw_chatbook.Tools_Interop.tools_scope_service import ToolsScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeToolsService:
    def __init__(self):
        self.calls = []

    async def list_tools(self):
        self.calls.append(("list_tools",))
        return {"tools": [{"name": "web_search", "canExecute": True}], "total": 1}

    async def execute_tool(self, tool_name, **kwargs):
        self.calls.append(("execute_tool", tool_name, kwargs))
        return {"ok": True, "result": {"validated": True}, "module": "search"}


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
async def test_tools_scope_service_routes_server_operations_and_normalizes_records():
    server = FakeToolsService()
    policy = FakePolicyEnforcer()
    scope = ToolsScopeService(server_service=server, policy_enforcer=policy)

    listed = await scope.list_tools(mode="server")
    executed = await scope.execute_tool(
        "web_search",
        mode="server",
        arguments={"query": "tldw"},
        idempotency_key="tool-run-1",
        dry_run=True,
    )

    assert listed["backend"] == "server"
    assert listed["tools"][0]["record_id"] == "server:tool:web_search"
    assert executed["record_id"] == "server:tool_execution:web_search"
    assert executed["backend"] == "server"
    assert server.calls == [
        ("list_tools",),
        (
            "execute_tool",
            "web_search",
            {"arguments": {"query": "tldw"}, "idempotency_key": "tool-run-1", "dry_run": True},
        ),
    ]
    assert policy.calls == [
        "tools.catalog.list.server",
        "tools.execution.launch.server",
    ]


@pytest.mark.asyncio
async def test_tools_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeToolsService()
    scope = ToolsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Server tools are server-only"):
        await scope.list_tools(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_tools_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeToolsService()
    scope = ToolsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("server_auth_required"))

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.execute_tool("web_search", mode="server")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_tools_scope_service_reports_known_unsupported_capabilities():
    scope = ToolsScopeService(server_service=FakeToolsService())

    assert scope.list_unsupported_capabilities(mode="server") == []
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "tools.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server tools are unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
