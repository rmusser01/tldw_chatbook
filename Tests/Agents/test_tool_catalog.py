# Tests/Agents/test_tool_catalog.py
"""Catalog registry + real builtin tools (no network, no DB)."""
from tldw_chatbook.Agents.agent_models import (
    DIRECT_DISCLOSE_THRESHOLD, FIND_TOOLS_NAME, LOAD_TOOLS_NAME,
    RunBudget, SPAWN_TOOL_NAME, ToolCatalogEntry, ToolResult, ToolSchema,
)
from tldw_chatbook.Agents.tool_catalog import (
    FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA, SPAWN_TOOL_SCHEMA,
    BuiltinToolProvider, ToolCatalogRegistry, initial_disclosure,
)


def registry():
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return reg


def test_builtin_catalog_lists_calculator_and_datetime():
    entries = registry().list_catalog()
    names = {e.name for e in entries}
    assert {"calculator", "get_current_datetime"} <= names
    assert all(e.id.startswith("builtin:") for e in entries)
    assert all(e.source == "builtin" for e in entries)


def test_find_matches_name_and_description_case_insensitive():
    reg = registry()
    assert any(e.name == "calculator" for e in reg.find("CALC"))
    assert any(e.name == "get_current_datetime" for e in reg.find("timezone"))
    assert reg.find("no-such-thing-xyz") == []


def test_load_schema_round_trip():
    reg = registry()
    schema = reg.load_schema("builtin:calculator")
    assert isinstance(schema, ToolSchema)
    assert schema.name == "calculator"
    assert schema.parameters.get("type") == "object"


def test_invoke_by_name_executes_real_calculator():
    result = registry().invoke_by_name("calculator", {"expression": "6*7"})
    assert result.ok is True
    assert "42" in result.content


def test_invoke_by_name_unknown_tool_is_error_result():
    result = registry().invoke_by_name("nope", {})
    assert result.ok is False and "nope" in result.error


def test_invoke_captures_tool_exception_as_error_result():
    result = registry().invoke_by_name(
        "get_current_datetime", {"timezone": "Not/AZone"})
    assert result.ok is False
    assert result.error  # message captured, no exception escaped


def test_pseudo_tool_schemas():
    assert SPAWN_TOOL_SCHEMA.name == SPAWN_TOOL_NAME
    assert "task" in SPAWN_TOOL_SCHEMA.parameters["properties"]
    assert FIND_TOOLS_SCHEMA.name == FIND_TOOLS_NAME
    assert LOAD_TOOLS_SCHEMA.name == LOAD_TOOLS_NAME


class FakeBigProvider:
    """A provider with more tools than the threshold."""

    def list_catalog(self):
        return [ToolCatalogEntry(id=f"fake:t{i}", name=f"t{i}",
                                 one_line_description=f"tool {i}",
                                 source="fake")
                for i in range(DIRECT_DISCLOSE_THRESHOLD + 3)]

    def load_schema(self, tool_id):
        return ToolSchema(id=tool_id, name=tool_id.split(":")[1],
                          description="fake", parameters={"type": "object"})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content="fake")


def test_initial_disclosure_small_catalog_direct_discloses():
    schemas, offer_find_load = initial_disclosure(registry(), RunBudget())
    assert offer_find_load is False
    assert {s.name for s in schemas} >= {"calculator", "get_current_datetime"}


def test_initial_disclosure_large_catalog_defers_to_find_load():
    reg = registry()
    reg.register_provider(FakeBigProvider())
    schemas, offer_find_load = initial_disclosure(reg, RunBudget())
    assert offer_find_load is True and schemas == []


def test_initial_disclosure_respects_max_active_tools():
    schemas, _ = initial_disclosure(registry(),
                                    RunBudget(max_active_tools=1))
    assert len(schemas) == 1


class VanishingProvider:
    """Present on the first list_catalog() call, gone by the second.

    Simulates a network-backed provider whose catalog changed between
    resolve_name()'s list_catalog() call and invoke_by_name's own
    _owner_and_id() re-listing (Q8) — the two currently list the catalog
    independently, so a provider could plausibly have deregistered or
    lost the entry in between.
    """

    def __init__(self):
        self.calls = 0

    def list_catalog(self):
        self.calls += 1
        if self.calls == 1:
            return [ToolCatalogEntry(id="vanish:x", name="x",
                                     one_line_description="d",
                                     source="vanish")]
        return []

    def load_schema(self, tool_id):
        raise NotImplementedError

    def invoke(self, tool_id, args):
        raise NotImplementedError


def test_invoke_by_name_returns_error_result_when_owner_vanishes():
    """Q8: a None owner from _owner_and_id must be a ToolResult error,
    never an AttributeError from calling .invoke on None."""
    reg = ToolCatalogRegistry()
    reg.register_provider(VanishingProvider())
    result = reg.invoke_by_name("x", {})
    assert result.ok is False
    assert "x" in result.error
