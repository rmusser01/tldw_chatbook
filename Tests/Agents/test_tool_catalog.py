# Tests/Agents/test_tool_catalog.py
"""Catalog registry + real builtin tools (no network, no DB)."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import tldw_chatbook.Agents.tool_catalog as tool_catalog

from tldw_chatbook.Agents.agent_models import (
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
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
    PathAwareToolProvider,
    ToolCatalogRegistry,
    ToolPathTarget,
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


def test_path_target_contract_is_immutable_and_runtime_checkable(tmp_path):
    target = ToolPathTarget(path=Path(tmp_path), kind="exact")

    with pytest.raises(FrozenInstanceError):
        target.kind = "directory"
    assert isinstance(BuiltinToolProvider(), PathAwareToolProvider)


class FakeCatalogProvider:
    """A deterministic provider with configurable cheap metadata."""

    def __init__(self, entries=None, *, count=0):
        self._entries = entries or [
            ToolCatalogEntry(
                id=f"fake:t{i}",
                name=f"t{i}",
                one_line_description=f"tool {i}",
                source="fake",
            )
            for i in range(count)
        ]

    def list_catalog(self):
        return list(self._entries)

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id.split(":")[1],
            description="fake",
            parameters={"type": "object"},
        )

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content="fake")


def test_probe_initial_catalog_uses_schema_cost_not_catalog_count():
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(count=25))
    allowed = frozenset(f"t{i}" for i in range(25))

    schemas = tool_catalog.probe_initial_catalog(
        reg,
        allowed,
        max_schema_tokens=100,
        measure_schema_set=lambda candidate: 99,
    )

    assert schemas is not None
    assert len(schemas) == 25


def test_probe_initial_catalog_defers_fewer_large_schemas_by_cost():
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(count=5))

    schemas = tool_catalog.probe_initial_catalog(
        reg,
        frozenset(f"t{i}" for i in range(5)),
        max_schema_tokens=100,
        measure_schema_set=lambda candidate: 101,
    )

    assert schemas is None


def test_probe_initial_catalog_filters_disallowed_before_measurement():
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(count=4))
    measured_names = []

    def measure(candidate):
        measured_names.append(tuple(schema.name for schema in candidate))
        return len(candidate)

    schemas = tool_catalog.probe_initial_catalog(
        reg,
        frozenset({"t1", "t3"}),
        max_schema_tokens=10,
        measure_schema_set=measure,
    )

    assert tuple(schema.name for schema in schemas or ()) == ("t1", "t3")
    assert measured_names == [("t1",), ("t1", "t3")]


@pytest.mark.parametrize("measured", [0, -1])
def test_probe_initial_catalog_defers_non_positive_measurements(measured):
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(count=1))

    assert (
        tool_catalog.probe_initial_catalog(
            reg,
            frozenset({"t0"}),
            max_schema_tokens=100,
            measure_schema_set=lambda candidate: measured,
        )
        is None
    )


def test_probe_initial_catalog_defers_measurement_failures():
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(count=1))

    def explode(candidate):
        raise RuntimeError("estimator unavailable")

    assert (
        tool_catalog.probe_initial_catalog(
            reg,
            frozenset({"t0"}),
            max_schema_tokens=100,
            measure_schema_set=explode,
        )
        is None
    )


def test_find_ranks_relevance_independent_of_registration_order():
    entries = [
        ToolCatalogEntry("p:timezone", "timezone_lookup", "clock conversion", "p"),
        ToolCatalogEntry("p:wall", "wall_clock", "wall time", "p"),
        ToolCatalogEntry("p:sync", "clock_sync", "synchronize", "p"),
        ToolCatalogEntry("p:clock", "clock", "current time", "p"),
    ]
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(entries))

    found = reg.find("clock", allowed_names=frozenset(entry.name for entry in entries))

    assert [entry.name for entry in found] == [
        "clock",
        "clock_sync",
        "wall_clock",
        "timezone_lookup",
    ]


def test_find_filters_allow_list_before_eight_result_slice():
    entries = [
        ToolCatalogEntry("p:tool", "tool", "exact but denied", "p"),
        *[
            ToolCatalogEntry(f"p:m{i}", f"match_{i}", "tool helper", "p")
            for i in range(10)
        ],
    ]
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(entries))
    allowed = frozenset(f"match_{i}" for i in range(1, 10))

    found = reg.find("tool", allowed_names=allowed)

    assert [entry.name for entry in found] == [f"match_{i}" for i in range(1, 9)]


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
        def check(self, tool, run_id):
            return "nope"

    out = BuiltinToolProvider(gate=DenyGate()).invoke(
        "builtin:calculator", {"expression": "1+1"}
    )
    assert out.ok is False
    assert "nope" in out.error
    assert out.outcome == "blocked"


def test_builtin_provider_runs_when_gate_permits():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    class AllowGate:
        def check(self, tool, run_id):
            return None

    out = BuiltinToolProvider(gate=AllowGate()).invoke(
        "builtin:calculator", {"expression": "6*7"}
    )
    assert out.ok is True


def test_gate_none_is_not_ungated(monkeypatch):
    # Constraint 6: a bare provider must be gated, not open.
    import tldw_chatbook.Agents.tool_catalog as tc

    class DenyGate:
        def check(self, tool, run_id):
            return "denied by default gate"

    monkeypatch.setattr(tc, "build_builtin_gate", lambda: DenyGate())
    out = tc.BuiltinToolProvider().invoke("builtin:calculator", {"expression": "1+1"})
    assert out.ok is False
    assert "denied by default gate" in out.error


def test_gate_failure_does_not_raise_into_the_loop():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    class BoomGate:
        def check(self, tool, run_id):
            raise RuntimeError("gate exploded")

    out = BuiltinToolProvider(gate=BoomGate()).invoke(
        "builtin:calculator", {"expression": "1+1"}
    )
    assert out.ok is False  # fail closed, never raise


class _OpenGate:
    """Approval gate that never refuses -- isolates the ephemeral check
    below from the (separately tested) approval-gate machinery."""

    def check(self, tool, run_id):
        return None


class _RecordingWriteTool:
    """Stub standing in for create_note/update_note/write_file: records
    whether it actually ran, without touching a real DB or the filesystem."""

    def __init__(self, name):
        self.name = name
        self.description = "stub"
        self.parameters = {"type": "object", "properties": {}}
        self.called = False

    async def execute(self, **kwargs):
        self.called = True
        return {"ok": True}


def test_builtin_provider_refuses_write_shaped_tools_in_an_ephemeral_session():
    """F4 (final-review): agent tool calls are a 9th, ungated sink -- an
    ordinary Console reply can compose `create_note`/`update_note`/
    `write_file` (this module's own gateable builtins) exactly like any
    other reply, independently of the Console UI action-id registry in
    `Chat/console_ephemeral.py`. Before this fix `BuiltinToolProvider.
    invoke` never asked whether the running session was temporary.
    """
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    for tool_name in ("create_note", "update_note", "write_file"):
        stub = _RecordingWriteTool(tool_name)
        provider = BuiltinToolProvider(gate=_OpenGate(), ephemeral=True)
        provider._tools[tool_name] = stub

        result = provider.invoke(f"builtin:{tool_name}", {})

        assert result.ok is False, f"{tool_name} must refuse in a temporary chat"
        assert "temporary chat" in result.error
        assert stub.called is False, (
            f"{tool_name} must never execute in a temporary chat"
        )


def test_builtin_provider_write_shaped_tools_run_normally_outside_ephemeral():
    """Control for the test above: the same tools, same gate, same stub --
    only `ephemeral` differs -- must run exactly as before this fix."""
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    for tool_name in ("create_note", "update_note", "write_file"):
        stub = _RecordingWriteTool(tool_name)
        provider = BuiltinToolProvider(gate=_OpenGate(), ephemeral=False)
        provider._tools[tool_name] = stub

        result = provider.invoke(f"builtin:{tool_name}", {})

        assert result.ok is True, result.error
        assert stub.called is True


def test_builtin_provider_ephemeral_does_not_block_read_only_tools():
    """The block is targeted at write-shaped tools, not a blanket ephemeral
    lockout: calculator/get_current_datetime read/compute nothing to disk
    and must keep working in a temporary chat."""
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    out = BuiltinToolProvider(gate=_OpenGate(), ephemeral=True).invoke(
        "builtin:calculator", {"expression": "6*7"}
    )
    assert out.ok is True
    assert "42" in out.content


# --- task-3240 Critical prerequisite: coerced registration read -------------
#
# `BuiltinToolProvider.__init__` reads each `_GATEABLE_BUILTINS` gate via a
# function-local `from ..config import get_cli_setting` -- patching
# `tldw_chatbook.Agents.tool_catalog.get_cli_setting` has nothing to attach
# to (no such module attribute exists); the seam these tests must control is
# `tldw_chatbook.config.get_cli_setting` itself, the module the function-local
# import re-resolves from on every call.


def test_registration_read_coerces_quoted_false_to_not_registered(monkeypatch):
    """A quoted `"false"` in `[tools]` must NOT register the tool.

    Before the fix, `get_cli_setting(...)` returned the raw string `"false"`
    and `not "false"` is `False` (any non-empty string is truthy) -- so a
    mis-typed TOML value silently ENABLED the gate while a coerced UI would
    have shown it OFF. This is the exact class of bug task-3240's design doc
    calls the arc's fifth `bool("false")` site.
    """
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS, BuiltinToolProvider

    target = _GATEABLE_BUILTINS[0]

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "tools" and key == target.gate_key:
            return "false"
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    provider = BuiltinToolProvider()
    assert target.tool_name not in provider._tools


def test_registration_read_coerces_quoted_true_to_registered(monkeypatch):
    """The mirror case: a quoted `"true"` MUST register the tool."""
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS, BuiltinToolProvider

    target = _GATEABLE_BUILTINS[0]

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "tools" and key == target.gate_key:
            return "true"
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    provider = BuiltinToolProvider()
    assert target.tool_name in provider._tools


# --- TASK-26007: fuzzy ranking + deferred-tool listing ----------------------


def _paraphrase_registry():
    entries = [
        ToolCatalogEntry(
            "p:glob",
            "fs_glob",
            "Match files under the workspace with a glob pattern, "
            "newest-mtime first, workspace-relative paths.",
            "p",
        ),
        ToolCatalogEntry("p:read", "fs_read", "Read one workspace file.", "p"),
        ToolCatalogEntry("p:clock", "clock", "current time", "p"),
    ]
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(entries))
    return reg


def test_paraphrased_query_matches_without_a_literal_substring():
    """AC#1/#6: 'find files by name' shares no substring with fs_glob's
    name or description, yet token overlap surfaces it."""
    reg = _paraphrase_registry()

    found = [entry.name for entry in reg.find("find files by name")]

    assert "fs_glob" in found
    assert "clock" not in found, "zero-overlap tools must not ride along"


def test_fuzzy_ranking_is_deterministic_and_below_substring_tiers():
    """AC#4/#5."""
    reg = _paraphrase_registry()

    first = [entry.name for entry in reg.find("read workspace files")]
    second = [entry.name for entry in reg.find("read workspace files")]
    assert first == second, "same query must return the same order"

    # 'read' is a name-substring of fs_read (tier 2); fs_glob only matches
    # by token overlap (tier 4) -- the substring tier must rank first.
    ranked = [entry.name for entry in reg.find("read")]
    assert ranked.index("fs_read") == 0


def test_find_tools_schema_embeds_a_bounded_listing():
    """AC#2/#3."""
    from tldw_chatbook.Agents.tool_catalog import (
        FIND_TOOLS_SCHEMA,
        build_find_tools_schema,
    )

    small = build_find_tools_schema(["fs_read", "fs_glob", "clock"])
    assert "fs_glob" in small.description
    assert "clock" in small.description
    assert small.name == FIND_TOOLS_SCHEMA.name

    many = build_find_tools_schema(
        [f"fs_{'x' * 30}_{i}" for i in range(40)]
        + [f"git_{'y' * 30}_{i}" for i in range(40)]
    )
    assert len(many.description) < len(FIND_TOOLS_SCHEMA.description) + 900
    assert "fs_*" in many.description and "git_*" in many.description, (
        "an oversized list must degrade to group names, not vanish"
    )

    empty = build_find_tools_schema([])
    assert empty is FIND_TOOLS_SCHEMA


def test_deferred_disclosure_plan_lists_available_tools():
    """AC#6: when the catalog is deferred, find_tools' own description
    names what exists so the model cannot conclude absence."""
    from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
    from tldw_chatbook.Agents.agent_service import (
        build_first_request_schema_plan,
    )

    entries = [
        ToolCatalogEntry("p:glob", "fs_glob", "Match files with a glob.", "p"),
        *[
            ToolCatalogEntry(
                f"p:pad{i}", f"pad_tool_{i}", "padding description " * 40, "p"
            )
            for i in range(300)
        ],
    ]
    reg = ToolCatalogRegistry()
    reg.register_provider(FakeCatalogProvider(entries))
    config = AgentConfig(
        model="m",
        system_prompt="s",
        provider="llama_cpp",
        budget=RunBudget(max_steps=5),
    )
    plan = build_first_request_schema_plan(
        reg,
        tuple(entry.name for entry in entries),
        config,
        "llama_cpp",
        [{"role": "user", "content": "hi"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
    )

    assert plan.offer_find_load, "300 padded tools must defer disclosure"
    find_schema = next(
        schema
        for schema in plan.runtime_schemas
        if schema.name == "find_tools"
    )
    # 301 tools exceed the name-listing bound, so the surface degrades to
    # groups (AC#3) -- and the fs group is still named, so the capability
    # is visibly present (AC#2).
    assert "fs_*" in find_schema.description
    assert "pad_* (300)" in find_schema.description
