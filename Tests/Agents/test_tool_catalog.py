# Tests/Agents/test_tool_catalog.py
"""Catalog registry + real builtin tools (no network, no DB)."""

from tldw_chatbook.Agents.agent_models import (
    DIRECT_DISCLOSE_THRESHOLD,
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    RunBudget,
    SPAWN_TOOL_NAME,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.tool_catalog import (
    FIND_TOOLS_SCHEMA,
    LOAD_TOOLS_SCHEMA,
    SPAWN_TOOL_SCHEMA,
    BuiltinToolProvider,
    ToolCatalogRegistry,
    initial_disclosure,
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
        "get_current_datetime", {"timezone": "Not/AZone"}
    )
    assert result.ok is False
    assert result.error  # message captured, no exception escaped


def test_pseudo_tool_schemas():
    assert SPAWN_TOOL_SCHEMA.name == SPAWN_TOOL_NAME
    assert "task" in SPAWN_TOOL_SCHEMA.parameters["properties"]
    assert FIND_TOOLS_SCHEMA.name == FIND_TOOLS_NAME
    assert LOAD_TOOLS_SCHEMA.name == LOAD_TOOLS_NAME


def test_tool_for_returns_the_real_tool_invoke_would_dispatch():
    # Minor 7 (final review): every hook test substitutes a fake provider,
    # so `tool_for` itself -- the only thing standing between "the review
    # hook reviews built-ins" and "the hook silently reviews nothing" --
    # had no direct coverage. Assert it returns the SAME object `invoke()`
    # looks up internally (both read `self._tools`), not merely an
    # equivalent one.
    provider = BuiltinToolProvider()
    tool = provider.tool_for("calculator")
    assert tool is provider._tools["calculator"]
    assert tool.name == "calculator"


def test_tool_for_returns_none_for_unknown_name():
    provider = BuiltinToolProvider()
    assert provider.tool_for("not_a_real_tool") is None


class FakeBigProvider:
    """A provider with more tools than the threshold."""

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"fake:t{i}",
                name=f"t{i}",
                one_line_description=f"tool {i}",
                source="fake",
            )
            for i in range(DIRECT_DISCLOSE_THRESHOLD + 3)
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id.split(":")[1],
            description="fake",
            parameters={"type": "object"},
        )

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
    schemas, _ = initial_disclosure(registry(), RunBudget(max_active_tools=1))
    assert len(schemas) == 1


class VanishingProvider:
    """Present on the first list_catalog() call, gone by the second.

    Simulates a network-backed provider whose remote catalog changed
    between two RUNS (e.g. an MCP server losing a tool). Since
    ``resolve_name()``/``_owner_and_id()`` now share one cache built
    atomically from a single ``list_catalog()`` sweep per provider (the
    tool_catalog fix this test's sibling below regression-locks), the
    "owner vanished between resolve_name() and _owner_and_id()" race this
    test used to simulate WITHIN one lookup is now structurally
    impossible — the two calls always read the same cache generation. The
    only remaining seam that can surface a stale name is
    ``reset_catalog_cache()`` (called once per run), so this now exercises
    that instead: a name resolved in one run, then genuinely gone by the
    next.
    """

    def __init__(self):
        self.calls = 0

    def list_catalog(self):
        self.calls += 1
        if self.calls == 1:
            return [
                ToolCatalogEntry(
                    id="vanish:x", name="x", one_line_description="d", source="vanish"
                )
            ]
        return []

    def load_schema(self, tool_id):
        raise NotImplementedError

    def invoke(self, tool_id, args):
        raise NotImplementedError


def test_invoke_by_name_returns_error_result_when_owner_vanishes():
    """A name that resolved in one run but is gone the next (post-reset)
    must surface as a graceful ToolResult error, never an AttributeError
    or an uncaught exception from calling .invoke on a stale id."""
    reg = ToolCatalogRegistry()
    reg.register_provider(VanishingProvider())
    assert reg.resolve_name("x") == "vanish:x"

    reg.reset_catalog_cache()
    result = reg.invoke_by_name("x", {})
    assert result.ok is False
    assert "x" in result.error


def test_builtin_provider_refuses_when_gate_denies():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    class DenyGate:
        def check(self, tool):
            return "nope"

    out = BuiltinToolProvider(gate=DenyGate()).invoke(
        "builtin:calculator", {"expression": "1+1"}
    )
    assert out.ok is False
    assert "nope" in out.error


def test_builtin_provider_runs_when_gate_permits():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    class AllowGate:
        def check(self, tool):
            return None

    out = BuiltinToolProvider(gate=AllowGate()).invoke(
        "builtin:calculator", {"expression": "6*7"}
    )
    assert out.ok is True


def test_gate_none_is_not_ungated(monkeypatch):
    # Constraint 6: a bare provider must be gated, not open.
    import tldw_chatbook.Agents.tool_catalog as tc

    class DenyGate:
        def check(self, tool):
            return "denied by default gate"

    monkeypatch.setattr(tc, "build_builtin_gate", lambda: DenyGate())
    out = tc.BuiltinToolProvider().invoke("builtin:calculator", {"expression": "1+1"})
    assert out.ok is False
    assert "denied by default gate" in out.error


def test_gate_failure_does_not_raise_into_the_loop():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    class BoomGate:
        def check(self, tool):
            raise RuntimeError("gate exploded")

    out = BuiltinToolProvider(gate=BoomGate()).invoke(
        "builtin:calculator", {"expression": "1+1"}
    )
    assert out.ok is False  # fail closed, never raise


def test_provider_accepts_services_and_still_works_without_them():
    """TASK-656's permissions enumerator builds a bare provider.

    Services must therefore stay optional: metadata is readable with
    services=None, and only execute() needs them.
    """
    from tldw_chatbook.Agents.builtin_services import BuiltinToolServices
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    bare = BuiltinToolProvider()
    assert {e.name for e in bare.list_catalog()} >= {"calculator", "get_current_datetime"}

    services = BuiltinToolServices(notes_library=object())
    injected = BuiltinToolProvider(services=services)
    assert injected.services is services
    assert bare.services is None
