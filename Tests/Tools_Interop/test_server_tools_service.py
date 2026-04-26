from unittest.mock import Mock

import pytest

from tldw_chatbook.Tools_Interop import ServerToolsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeToolsClient:
    def __init__(self):
        self.calls = []

    async def list_server_tools(self):
        self.calls.append(("list_server_tools",))
        return {
            "tools": [
                {
                    "name": "web_search",
                    "description": "Search the web",
                    "module": "search",
                    "canExecute": True,
                }
            ]
        }

    async def execute_server_tool(self, request_data):
        self.calls.append(("execute_server_tool", request_data.model_dump(mode="json")))
        return {"ok": True, "result": {"validated": True}, "module": "search"}


@pytest.mark.asyncio
async def test_server_tools_service_routes_tool_surface_with_policy_actions():
    client = FakeToolsClient()
    policy = Mock()
    service = ServerToolsService(client=client, policy_enforcer=policy)

    listed = await service.list_tools()
    executed = await service.execute_tool(
        "web_search",
        arguments={"query": "tldw"},
        idempotency_key="tool-run-1",
        dry_run=True,
    )

    assert listed["tools"][0]["name"] == "web_search"
    assert executed == {"ok": True, "result": {"validated": True}, "module": "search"}
    assert client.calls == [
        ("list_server_tools",),
        (
            "execute_server_tool",
            {
                "tool_name": "web_search",
                "arguments": {"query": "tldw"},
                "idempotency_key": "tool-run-1",
                "dry_run": True,
            },
        ),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "tools.catalog.list.server",
        "tools.execution.launch.server",
    ]


@pytest.mark.asyncio
async def test_server_tools_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_auth_required",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeToolsClient()
    service = ServerToolsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_tools()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []
