from __future__ import annotations

import pytest

from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.MCP.hub_tool_catalog import (
    builtin_tools_from_inventory,
    filter_tools,
    local_tools_from_record,
    schema_argument_names,
    server_tools_from_inventory,
)


def _local_record(connected=True, tools=None):
    return {
        "profile_id": "docs",
        "command": "npx",
        "is_connected": connected,
        "discovery_snapshot": {
            "tools": tools
            if tools is not None
            else [
                {
                    "name": "search",
                    "description": "Search docs.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
                {"name": "bare", "description": "", "inputSchema": {}},
            ]
        },
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
    assert (
        local_tools_from_record({"profile_id": "x", "discovery_snapshot": None}) == []
    )


@pytest.mark.parametrize(
    ("reserved_id", "tool_name"),
    (("__local__", "fs_write"), ("__virtual_cli__", "cat")),
)
def test_reserved_external_profile_cannot_claim_synthetic_tool_identity(
    tmp_path, reserved_id, tool_name
):
    workspace_tool = LocalToolProvider(workspace_root=tmp_path).hub_tool_for("fs_write")
    reserved_record = _local_record(
        tools=[
            {
                "name": tool_name,
                "description": workspace_tool.description,
                "inputSchema": workspace_tool.input_schema,
            }
        ]
    )
    reserved_record["profile_id"] = reserved_id

    assert workspace_tool.tool_id == "local:__local__::fs_write"
    assert local_tools_from_record(reserved_record) == []
    reserved_record["profile_id"] = f" {reserved_id} "
    assert local_tools_from_record(reserved_record) == []


def test_reserved_external_profile_rule_does_not_drop_case_distinct_profile():
    record = _local_record(tools=[{"name": "search"}])
    record["profile_id"] = "__LOCAL__"

    tools = local_tools_from_record(record)

    assert [tool.tool_id for tool in tools] == ["local:__LOCAL__::search"]


def test_builtin_tools_carry_schema_and_execute():
    """RAG-48 part 2: builtins now carry a real `inputSchema` (synthesized
    from the tool's AST signature, part 1) instead of always `None` --
    deliberate contract change, was `test_builtin_tools_have_no_schema_but_execute`."""
    tools = builtin_tools_from_inventory(
        {
            "tools": [
                {
                    "name": "chat_with_llm",
                    "description": "Chat.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                }
            ]
        }
    )
    assert isinstance(tools[0].input_schema, dict)
    assert tools[0].input_schema["type"] == "object"
    assert tools[0].executable
    assert tools[0].server_key == "builtin:tldw_chatbook"


def test_builtin_tool_without_schema_still_yields_none():
    """A builtin entry with no (or empty) `inputSchema` still normalizes to
    `None`, same as the other two `_normalized_schema()` call sites."""
    tools = builtin_tools_from_inventory(
        {"tools": [{"name": "chat_with_llm", "description": "Chat."}]}
    )
    assert tools[0].input_schema is None and tools[0].executable


def test_builtin_tools_never_carry_risk_tags_even_when_offered_them():
    """TRIPWIRE (PR-T3 Fix Round C, Item 2). `permission_store
    .BY_KEY_HASH_FREE_SERVER_KEYS` exempts `"builtin:tldw_chatbook"` from
    `resolve_effective_state_by_key()`'s "any allow collapses to ask"
    fallback, and that resolver has no `HubTool.tags` to floor a high-risk
    inherited allow with either -- so the exemption is safe ONLY because
    every `HubTool` this function produces carries `tags=()`
    unconditionally, regardless of what the raw inventory entry contains.

    Unlike `server_tools_from_inventory` (see
    `test_server_tools_read_extras_defensively` above), this function does
    not call `_extra_tags()` at all -- it hard-codes `tags=()`. This test
    proves that by handing it a raw tool dict carrying the exact fields
    `_extra_tags()` reads for server tools (`risk_class`, `capabilities`)
    and asserting they are silently ignored. If a future change wires
    `_extra_tags()` (or any tag source) into `builtin_tools_from_inventory`,
    this goes red -- which is the day
    `permission_store.BY_KEY_HASH_FREE_SERVER_KEYS`'s exemption for
    `builtin:tldw_chatbook` stops being safe and needs re-examining, not
    just a comment."""
    tools = builtin_tools_from_inventory(
        {
            "tools": [
                {
                    "name": "delete_everything",
                    "description": "Deletes things.",
                    "risk_class": "high",
                    "capabilities": ["network", "mutates"],
                    "inputSchema": {"type": "object"},
                }
            ]
        }
    )

    assert tools[0].tags == ()


def test_builtin_tools_defensively_copy_input_schema():
    """task-1337 (plan Task 8): an inventory-supplied non-empty `inputSchema`
    is COPIED into the `HubTool`, not aliased -- mutating the source mapping
    after derivation must not rewrite the tool's schema. (No schema is
    synthesized for legacy entries lacking one -- see the None test above.)
    """
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    inventory = {
        "tools": [
            {"name": "chat_with_llm", "description": "Chat.", "inputSchema": schema}
        ]
    }

    tools = builtin_tools_from_inventory(inventory)

    assert tools[0].input_schema == schema
    schema["properties"]["q"]["type"] = "number"
    schema["properties"]["injected"] = {"type": "integer"}
    assert tools[0].input_schema["properties"]["q"]["type"] == "string"
    assert "injected" not in tools[0].input_schema["properties"]


def test_server_tools_read_extras_defensively():
    payload = {
        "tools": [
            {
                "name": "web_search",
                "description": "Search.",
                "risk_class": "High",
                "capabilities": ["Network", 7, "mutates"],
                "inputSchema": {"type": "object"},
            },
            {"description": "nameless — skipped"},
            "not-a-dict",
        ]
    }
    tools = server_tools_from_inventory(payload, target_id="main", target_label="Main")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.server_key == "server:main" and tool.server_label == "Main"
    assert tool.tags == ("high", "network", "mutates")
    assert tool.input_schema == {"type": "object"}
    assert tool.executable is False  # server-source execution is Phase 4


def test_filter_by_server_and_text():
    tools = local_tools_from_record(_local_record()) + builtin_tools_from_inventory(
        {"tools": [{"name": "create_note", "description": "Notes."}]}
    )
    assert [
        t.name for t in filter_tools(tools, server_key="builtin:tldw_chatbook")
    ] == ["create_note"]
    assert [t.name for t in filter_tools(tools, text="SEARCH")] == ["search"]


# -- C1: duplicate tool names within one snapshot/inventory must not yield
# duplicate HubTool.tool_id values (Textual DataTable row keys) --------------


def test_local_tools_dedupe_duplicate_names_keeping_first_occurrence():
    tools = local_tools_from_record(
        _local_record(
            tools=[
                {"name": "search", "description": "first"},
                {"name": "search", "description": "second"},
            ]
        )
    )
    assert [t.name for t in tools] == ["search"]
    assert tools[0].description == "first"
    assert [t.tool_id for t in tools] == ["local:docs::search"]


def test_local_tools_dedupe_whitespace_variant_names():
    tools = local_tools_from_record(
        _local_record(
            tools=[
                {"name": "search", "description": "first"},
                {"name": " search", "description": "second"},
            ]
        )
    )
    assert [t.name for t in tools] == ["search"]
    assert tools[0].description == "first"


def test_builtin_tools_dedupe_duplicate_names():
    tools = builtin_tools_from_inventory(
        {
            "tools": [
                {"name": "chat_with_llm", "description": "first"},
                {"name": "chat_with_llm", "description": "second"},
            ]
        }
    )
    assert [t.name for t in tools] == ["chat_with_llm"]
    assert tools[0].description == "first"


def test_server_tools_dedupe_duplicate_names():
    payload = {
        "tools": [
            {"name": "web_search", "description": "first"},
            {"name": "web_search", "description": "second"},
        ]
    }
    tools = server_tools_from_inventory(payload, target_id="main", target_label="Main")
    assert [t.name for t in tools] == ["web_search"]
    assert tools[0].description == "first"


# -- Task 4 (PR-T3): schema_argument_names() -- the execution log's
# argument-provenance seam reads a tool's registered names from here.


def test_schema_argument_names_reads_top_level_properties():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
    }
    assert schema_argument_names(schema) == {"query", "limit"}


def test_schema_argument_names_none_schema_yields_empty_set():
    assert schema_argument_names(None) == set()


def test_schema_argument_names_malformed_properties_yields_empty_set():
    assert (
        schema_argument_names({"type": "object", "properties": "not-a-dict"}) == set()
    )
    assert schema_argument_names({"type": "object"}) == set()
    assert schema_argument_names("not-a-dict") == set()  # defensive: never raises


def test_schema_argument_names_tolerates_unrenderable_nested_property():
    """Unlike `parse_schema()` (which falls back to raw JSON for a nested
    object property), this only reports NAMES -- an unrenderable nested
    property still counts as a registered top-level argument name."""
    schema = {
        "type": "object",
        "properties": {"filters": {"type": "object", "properties": {"a": {}}}},
    }
    assert schema_argument_names(schema) == {"filters"}
