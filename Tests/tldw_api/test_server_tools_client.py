from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import ExecuteToolRequest, ExecuteToolResult, TLDWAPIClient, ToolListResponse


@pytest.mark.asyncio
async def test_server_tools_client_routes_list_and_execute(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Search the web",
                        "module": "search",
                        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
                        "canExecute": True,
                    }
                ]
            },
            {"ok": True, "result": {"validated": True}, "module": "search"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_server_tools()
    executed = await client.execute_server_tool(
        ExecuteToolRequest(
            tool_name="web_search",
            arguments={"query": "tldw"},
            idempotency_key="tool-run-1",
            dry_run=True,
        )
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/tools")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/tools/execute")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "tool_name": "web_search",
        "arguments": {"query": "tldw"},
        "idempotency_key": "tool-run-1",
        "dry_run": True,
    }
    assert isinstance(listed, ToolListResponse)
    assert listed.tools[0].name == "web_search"
    assert listed.tools[0].canExecute is True
    assert isinstance(executed, ExecuteToolResult)
    assert executed.result == {"validated": True}
