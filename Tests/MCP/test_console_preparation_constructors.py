"""Constructor substitutions cannot cross the stock preparation boundary."""

import asyncio
import sys

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from Tests.MCP.test_console_tool_preparation import _prepare
from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import bootstrap

catalog_controls = snapshot_controls.catalog_controls
catalog_store = snapshot_controls.catalog_store
local_root = snapshot_controls.local_root
mcp_sources = snapshot_controls.mcp_sources
snapshot_case = snapshot_controls.snapshot_case


async def _issued_consumers(case, phase):
    if phase == "entry":
        return None
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from tldw_chatbook.MCP import console_tool_preparation as preparation_source

    preparation = await _prepare(case)
    assert preparation is not None
    composition = provider_source.capture_standard_controller_composition(
        provider_source.MCPToolProvider, case.service
    )
    assert composition is not None
    provider = provider_source.MCPToolProvider(
        service=case.service,
        main_loop=asyncio.get_running_loop(),
        builtin_raw_name_exclusions=frozenset(),
    )
    # Invoke the captured original directly: looking up an instance method must
    # not itself execute the substituted __getattribute__ before the guard.
    require_current = preparation_source.ConsoleToolPreparation.require_current
    return preparation, require_current, composition, provider


async def _assert_boundary(case, issued, calls):
    probe = snapshot_controls._MaximumProbe(case.source, case.permissions)
    with probe.installed():
        if issued is None:
            declined = await _prepare(case) is None
            decisions = {"entry_declined": declined}
        else:
            preparation, require_current, composition, provider = issued
            current_refused = adoption_refused = False
            try:
                require_current(preparation, case.service)
            except bootstrap.RecoveryRequired:
                current_refused = True
            try:
                provider.adopt_console_preparation(
                    preparation, _controller_composition=composition
                )
            except PermissionError:
                adoption_refused = True
            decisions = {
                "issued_refused": current_refused,
                "adoption_refused": adoption_refused,
                "catalog_empty": not provider.list_catalog(),
            }
    assert {
        **decisions,
        "foreign_calls": calls,
        "policy_reads": len(probe.permission_threads),
        "catalog_reads": len(probe.read_threads),
    } == {
        **dict.fromkeys(decisions, True),
        "foreign_calls": [],
        "policy_reads": 0,
        "catalog_reads": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["entry", "issued"])
@pytest.mark.parametrize(
    "class_name", ["PreparedConsoleTool", "ConsoleToolPreparation"]
)
@private_profile_test
async def test_substituted_allocator_declines_or_refuses(
    request, snapshot_case, class_name, phase
):
    from tldw_chatbook.MCP import console_tool_preparation as preparation_source

    case = snapshot_case
    snapshot_controls._loop_projection(case)
    issued = await _issued_consumers(case, phase)
    target = getattr(preparation_source, class_name)
    calls = []

    def foreign_new(cls, *args, **kwargs):
        calls.append(class_name)
        return object.__new__(cls)

    def cleanup_new(cls, *args, **kwargs):
        return object.__new__(cls)

    target.__new__ = staticmethod(foreign_new)
    try:
        await _assert_boundary(case, issued, calls)
    finally:
        # TASK-34406: deleting a newly introduced __new__ does not restore the
        # CPython allocator fast path. Keep a valid declared allocator through
        # fixture cleanup; only this exact private child's interpreter is changed.
        target.__new__ = staticmethod(cleanup_new)


def _substitute_consumed_lookup(monkeypatch, mutation, calls):
    from tldw_chatbook.MCP import console_tool_preparation as preparation_source
    from tldw_chatbook.MCP import hub_tool_catalog

    if mutation == "tool-schema-descriptor":

        def read_schema(tool):
            calls.append("schema-get")
            return object.__getattribute__(tool, "__dict__")["schema_json"]

        def write_schema(tool, value):
            calls.append("schema-set")
            object.__getattribute__(tool, "__dict__")["schema_json"] = value

        monkeypatch.setattr(
            preparation_source.PreparedConsoleTool,
            "schema_json",
            property(read_schema, write_schema),
            raising=False,
        )
    elif mutation == "result-source-access":

        def read_attribute(preparation, name):
            if name in {"_captured_sources", "tools", "profile_id"}:
                calls.append(name)
            return object.__getattribute__(preparation, name)

        monkeypatch.setattr(
            preparation_source.ConsoleToolPreparation,
            "__getattribute__",
            read_attribute,
        )
    elif mutation == "result-hash":

        def hash_preparation(preparation):
            calls.append("result-hash")
            return object.__hash__(preparation)

        monkeypatch.setattr(
            preparation_source.ConsoleToolPreparation, "__hash__", hash_preparation
        )
    elif mutation == "hub-alias":
        original = preparation_source.HubTool

        def foreign_hub(*args, **kwargs):
            calls.append("hub-alias")
            return original(*args, **kwargs)

        monkeypatch.setattr(preparation_source, "HubTool", foreign_hub)
    else:
        assert mutation == "hub-init"
        _substitute_hub_constructor(monkeypatch, hub_tool_catalog.HubTool, calls)


def _substitute_hub_constructor(monkeypatch, target, calls):
    original = target.__init__

    def foreign_init(self, *args, **kwargs):
        calls.append("hub-init")
        original(self, *args, **kwargs)

    monkeypatch.setattr(target, "__init__", foreign_init)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["entry", "issued"])
@pytest.mark.parametrize(
    "mutation",
    [
        "tool-schema-descriptor",
        "result-source-access",
        "result-hash",
        "hub-alias",
        "hub-init",
    ],
)
async def test_substituted_lookup_declines_or_refuses(
    snapshot_case, monkeypatch, mutation, phase
):
    case = snapshot_case
    snapshot_controls._loop_projection(case)
    issued = await _issued_consumers(case, phase)
    calls = []
    # Load the original module before changing its consumed class/module slots.
    from tldw_chatbook.MCP import console_tool_preparation  # noqa: F401

    with monkeypatch.context() as patch:
        _substitute_consumed_lookup(patch, mutation, calls)
        await _assert_boundary(case, issued, calls)


@pytest.mark.asyncio
@private_profile_test
async def test_defining_hub_constructor_changed_before_preparation_import_declines(
    request, snapshot_case, monkeypatch
):
    from tldw_chatbook.MCP import hub_tool_catalog

    # The exact private child imports the real source fixtures, but no earlier
    # preparation. This tests defining-module provenance without reloading a
    # module underneath issued data or other active source owners.
    assert "tldw_chatbook.MCP.console_tool_preparation" not in sys.modules
    case = snapshot_case
    snapshot_controls._loop_projection(case)
    calls = []
    with monkeypatch.context() as patch:
        _substitute_hub_constructor(patch, hub_tool_catalog.HubTool, calls)
        await _assert_boundary(case, None, calls)


@pytest.mark.asyncio
@private_profile_test
async def test_dependency_rule_changed_before_preparation_import_declines(
    request, snapshot_case, monkeypatch
):
    from tldw_chatbook.MCP import hub_tool_catalog

    assert "tldw_chatbook.MCP.console_tool_preparation" not in sys.modules
    case = snapshot_case
    snapshot_controls._loop_projection(case)
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(
            hub_tool_catalog,
            "maximum_needs_external_catalog",
            lambda maximum: calls.append(maximum) or False,
        )
        await _assert_boundary(case, None, calls)
