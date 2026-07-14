from __future__ import annotations

from tldw_chatbook.MCP.hub_tool_catalog import (
    HubTool,
    builtin_tools_from_inventory,
    filter_tools,
    local_tools_from_record,
    server_tools_from_inventory,
)


def _local_record(connected=True, tools=None):
    return {
        "profile_id": "docs", "command": "npx",
        "is_connected": connected,
        "discovery_snapshot": {"tools": tools if tools is not None else [
            {"name": "search", "description": "Search docs.",
             "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
            {"name": "bare", "description": "", "inputSchema": {}},
        ]},
    }


def test_local_tools_carry_schema_and_stale_flag():
    tools = local_tools_from_record(_local_record(connected=False))
    assert [t.name for t in tools] == ["search", "bare"]
    assert tools[0].input_schema["properties"]["q"]["type"] == "string"
    assert tools[1].input_schema is None  # empty schema dict -> None
    assert all(t.stale and t.executable and t.source == "local" for t in tools)
    assert tools[0].server_key == "local:docs"
    assert tools[0].tool_id == "local:docs::search"


def test_local_record_without_snapshot_yields_nothing():
    assert local_tools_from_record({"profile_id": "x", "discovery_snapshot": None}) == []


def test_builtin_tools_have_no_schema_but_execute():
    tools = builtin_tools_from_inventory({"tools": [
        {"name": "chat_with_llm", "description": "Chat."}]})
    assert tools[0].input_schema is None and tools[0].executable
    assert tools[0].server_key == "builtin:tldw_chatbook"


def test_server_tools_read_extras_defensively():
    payload = {"tools": [
        {"name": "web_search", "description": "Search.",
         "risk_class": "High", "capabilities": ["Network", 7, "mutates"],
         "inputSchema": {"type": "object"}},
        {"description": "nameless — skipped"},
        "not-a-dict",
    ]}
    tools = server_tools_from_inventory(payload, target_id="main", target_label="Main")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.server_key == "server:main" and tool.server_label == "Main"
    assert tool.tags == ("high", "network", "mutates")
    assert tool.input_schema == {"type": "object"}
    assert tool.executable is False  # server-source execution is Phase 4


def test_filter_by_server_and_text():
    tools = local_tools_from_record(_local_record()) + builtin_tools_from_inventory(
        {"tools": [{"name": "create_note", "description": "Notes."}]})
    assert [t.name for t in filter_tools(tools, server_key="builtin:tldw_chatbook")] == ["create_note"]
    assert [t.name for t in filter_tools(tools, text="SEARCH")] == ["search"]
