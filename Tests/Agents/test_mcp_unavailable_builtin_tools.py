"""TASK-34100.5 AC#2 (entry-exit-handoff-24).

A fresh profile's first Console chat asked the user to approve
``tldw_chatbook · chat_with_llm`` -- an internal tool the in-process direct
runtime always refuses (``mcp_execution_failed``). The agent's catalog must
never offer a built-in tool the direct runtime marks unavailable, whatever
the caller's own exclusion list says, while a same-named tool on a LOCAL
server stays a different tool.
"""

from __future__ import annotations

import asyncio

from Tests.Agents.test_mcp_tool_provider import (
    FakeMCPService,
    _catalog_record,
    _compose,
    _tool_dict,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.MCP.local_runtime_delegate import LocalMCPRuntimeDelegate


def test_direct_runtime_and_catalog_share_one_unavailable_list() -> None:
    from tldw_chatbook.MCP.hub_tool_catalog import DIRECT_RUNTIME_UNAVAILABLE_TOOLS

    assert "chat_with_llm" in DIRECT_RUNTIME_UNAVAILABLE_TOOLS
    assert set(LocalMCPRuntimeDelegate._UNAVAILABLE_DIRECT_TOOLS) == set(
        DIRECT_RUNTIME_UNAVAILABLE_TOOLS
    )


def test_agent_catalog_never_offers_a_builtin_the_direct_runtime_refuses() -> None:
    service = FakeMCPService(
        inventory={
            "tools": [
                _tool_dict("chat_with_llm", "Chat with an LLM."),
                _tool_dict("create_note", "Create a note."),
            ]
        },
        catalog_records=[_catalog_record("relay", [_tool_dict("chat_with_llm")])],
    )
    for exclusions in (None, frozenset({"create_note"})):
        provider = MCPToolProvider(
            service=service,
            main_loop=asyncio.new_event_loop(),
            builtin_raw_name_exclusions=exclusions,
        )

        _compose(provider)

        names = {entry.name for entry in provider.list_catalog()}
        assert "mcp__tldw_chatbook__chat_with_llm" not in names
        assert provider.pending_gate_for("mcp__tldw_chatbook__chat_with_llm", {}) is None
        # A local server's tool of the same name is a different tool.
        assert "mcp__relay__chat_with_llm" in names
    assert [tool["name"] for tool in service.inventory["tools"]] == [
        "chat_with_llm",
        "create_note",
    ]
