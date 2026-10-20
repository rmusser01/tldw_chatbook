"""TASK-34100.13: no setup surface claims the assistant's tools are "all off".

fulltrack-02: the Tools step said "Everything is off by default" while
``all_tool_gates()`` had three default-on groups the step never showed; the
Summary and the Quick-track note repeated it. fulltrack-06: the file-tool
blurbs implied any path. new-full-track-steps-02: with "Create note" off, the
app's own MCP server still gave the agent ``create_note``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static, Switch

from tldw_chatbook.Agents import builtin_tool_gate
from tldw_chatbook.Agents.builtin_tool_gate import all_tool_gates
from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards import first_run_tools_posture as posture_mod
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_tools_step import ToolsStep


class _StepHost(App):
    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _tools_step(app_config):
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=7),
    )
    return wizard, step


def _text(step) -> str:
    return "\n".join(str(static.render()) for static in step.query(Static))


FALSE_CLAIMS = ("Everything is off", "everything is off", "all off", "tools off")


# -- AC#2: the step -----------------------------------------------------------


@pytest.mark.asyncio
async def test_first_run_step_names_the_default_on_groups_and_claims_nothing_false() -> None:
    _wizard, step = _tools_step({})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text = _text(step)
        also = str(step.query_one("#setup-tools-also-available", Static).render())
    for claim in FALSE_CLAIMS:
        assert claim not in text
    assert also.startswith("Also available (asks first)")
    for group in ("web search", "Watchlists", "character cards", "questions for you"):
        assert group in also


@pytest.mark.asyncio
async def test_the_console_master_switch_is_not_offered_in_setup() -> None:
    _wizard, step = _tools_step({})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switches = list(step.query(Switch))
    assert len(switches) == len(_GATEABLE_BUILTINS)
    assert not step.query(f"#setup-tool-{builtin_tool_gate.LOCAL_TOOLS_MASTER_KEY}")


@pytest.mark.asyncio
async def test_rerun_posture_counts_the_switches_that_are_on_from_the_full_toml() -> None:
    """load_settings() has no [tools] table: the gates ride in the raw TOML."""
    raw = {"tools": {"read_file_enabled": True, "write_file_enabled": True}}
    wizard, step = _tools_step({"COMPREHENSIVE_CONFIG_RAW": raw})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        sentence = str(step.query_one("#setup-tools-posture", Static).render())
        assert step.query_one("#setup-tool-read_file", Switch).value is True
        ok, _ = await step.commit()
    assert ok
    assert sentence.startswith(f"2 of {len(_GATEABLE_BUILTINS)} ")
    wizard.commit_config.assert_not_awaited()  # nothing changed


@pytest.mark.asyncio
async def test_master_switch_off_names_no_local_tool_as_available() -> None:
    """Review F3 (TASK-34100.13): with ``[console] local_tools_enabled`` off the
    Console composes no LocalToolProvider at all, so ask_user and the
    character tools are gone with web search and Watchlists. The line used to
    keep promising "character cards and questions for you" (and this test
    pinned that)."""
    raw = {"console": {"local_tools_enabled": False}}
    _wizard, step = _tools_step({"COMPREHENSIVE_CONFIG_RAW": raw})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        also = str(step.query_one("#setup-tools-also-available", Static).render())
    for name in ("web search", "character cards", "questions for you"):
        assert name not in also.replace("Web search", "")
    assert not also.startswith("Also available")
    assert "switched off" in also


def test_master_switch_off_leaves_no_local_group_in_the_posture() -> None:
    gates = all_tool_gates(
        config={"tools": {}, "console": {"local_tools_enabled": False}}
    )
    posture = posture_mod.tools_posture(gates)
    assert posture.also_available == ()
    note = posture_mod.quick_defaults_note(posture, auto_retrieve=False)
    assert "character cards" not in note and "questions for you" not in note


@pytest.mark.asyncio
async def test_the_posture_sentence_follows_the_switches() -> None:
    """Review F4 (TASK-34100.13): the sentence was computed once at compose,
    so after turning a switch on it still said "All 8 are off"."""
    _wizard, step = _tools_step({})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-tool-read_file", Switch).value = True
        await pilot.pause()
        on_one = str(step.query_one("#setup-tools-posture", Static).render())
        step.query_one("#setup-tool-read_file", Switch).value = False
        await pilot.pause()
        off_again = str(step.query_one("#setup-tools-posture", Static).render())
    assert on_one.startswith(f"1 of {len(_GATEABLE_BUILTINS)} ")
    assert f"All {len(_GATEABLE_BUILTINS)} are off" in off_again


def test_posture_is_derived_from_all_tool_gates() -> None:
    gates = all_tool_gates(
        config={"tools": {"ask_user_enabled": False}, "console": {}}
    )
    posture = posture_mod.tools_posture(gates)
    assert posture.switch_total == len(_GATEABLE_BUILTINS)
    assert "questions for you" not in posture.also_available
    assert "web search and Watchlists" in posture.also_available


# -- AC#2: the Summary row and the Quick note ---------------------------------


def _summary_rows(config):
    return {
        row.label: row
        for row in wizard_state.build_summary_rows(config, {}, rag_deps_installed=True)
    }


def test_summary_tools_row_is_derived_not_all_off() -> None:
    row = _summary_rows({})["Tools"]
    assert row.state == wizard_state.ROW_DEFAULT
    for claim in FALSE_CLAIMS:
        assert claim not in row.detail
    assert (
        "also available, asks first, when the context window has room: "
        "web search and Watchlists"
    ) in row.detail
    assert builtin_tool_gate.TOOL_GATES_PANE_PATH in row.detail


def test_summary_tools_row_names_the_switches_turned_on() -> None:
    row = _summary_rows({"tools": {"create_note_enabled": True}})["Tools"]
    assert row.state == wizard_state.ROW_CONFIGURED
    assert row.detail.startswith("1 switched on (Create note)")


def test_quick_note_is_derived_from_the_same_posture() -> None:
    gates = all_tool_gates(config={"tools": {}, "console": {}})
    note = posture_mod.quick_defaults_note(
        posture_mod.tools_posture(gates), auto_retrieve=False
    )
    for claim in ("tools off", "all off", "RAG off"):
        assert claim not in note
    assert "web search and Watchlists" in note


# -- AC#2/#4: blurbs on the shared gate rows -----------------------------------


def _card_session_label() -> str:
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _DECISION_OPTIONS,
    )

    return next(label for label, value in _DECISION_OPTIONS if value == "approve_session")


@pytest.mark.parametrize("entry", _GATEABLE_BUILTINS, ids=lambda e: e.tool_name)
def test_every_blurb_agrees_with_the_approval_card(entry) -> None:
    assert "longer scope" not in entry.blurb
    assert "Asks before each use" in entry.blurb
    assert _card_session_label() in entry.blurb


@pytest.mark.parametrize(
    "entry",
    [e for e in _GATEABLE_BUILTINS if e.tool_name in posture_mod.FILE_TOOL_NAMES],
    ids=lambda e: e.tool_name,
)
def test_file_tool_blurbs_name_workspace_folders_and_the_scratch_area(entry) -> None:
    assert "Workspace folders" in entry.blurb
    assert "chat scratch area" in entry.blurb


def test_mcp_gate_pane_gets_the_same_blurbs() -> None:
    """A parity pin: base already shared the blurbs (review-tests F8), so the
    file-tool copy the pane shows is asserted on its own below."""
    gates = {gate.key: gate for gate in all_tool_gates(config={"tools": {}, "console": {}})}
    for entry in _GATEABLE_BUILTINS:
        assert gates[entry.gate_key].description == entry.blurb


@pytest.mark.parametrize(
    "entry",
    [e for e in _GATEABLE_BUILTINS if e.tool_name in posture_mod.FILE_TOOL_NAMES],
    ids=lambda e: e.tool_name,
)
def test_the_mcp_gate_pane_says_where_file_tools_reach(entry) -> None:
    gates = {gate.key: gate for gate in all_tool_gates(config={"tools": {}, "console": {}})}
    assert "Workspace folders" in gates[entry.gate_key].description
    assert "chat scratch area" in gates[entry.gate_key].description


@pytest.mark.asyncio
async def test_one_static_workspace_hint_sits_under_the_file_rows() -> None:
    _wizard, step = _tools_step({})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        hints = list(step.query("#setup-tools-workspace-hint"))
        assert len(hints) == 1
        text = str(hints[0].render())
        hint_y = hints[0].region.y
        file_rows_y = [
            step.query_one(f"#setup-tool-{name}", Switch).region.y
            for name in posture_mod.FILE_TOOL_NAMES
        ]
    assert "Settings ▸ Workspaces" in text
    assert hint_y > max(file_rows_y)


# -- AC#3: the built-in MCP server's twins of gated-off tools -------------------


def _compose_catalog(monkeypatch, tools_cfg, console_cfg=None):
    from Tests.Agents.test_mcp_tool_provider import FakeMCPService, _compose, _tool_dict
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    monkeypatch.setattr(
        builtin_tool_gate,
        "_gate_config_snapshot",
        lambda: (dict(tools_cfg), dict(console_cfg or {})),
    )
    service = FakeMCPService(
        inventory={
            "tools": [
                _tool_dict(name, f"{name}.")
                for name in (
                    "create_note",
                    "create_character",
                    "update_character",
                    "list_characters",
                )
            ]
        }
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    return {entry.name for entry in provider.list_catalog()}


def test_create_note_is_absent_while_its_native_gate_is_off(monkeypatch) -> None:
    names = _compose_catalog(monkeypatch, {})
    assert "mcp__tldw_chatbook__create_note" not in names
    # Character tools default on, so their twins stay; reads always stay.
    assert "mcp__tldw_chatbook__create_character" in names
    assert "mcp__tldw_chatbook__list_characters" in names


def test_create_note_returns_when_its_native_gate_is_on(monkeypatch) -> None:
    names = _compose_catalog(monkeypatch, {"create_note_enabled": True})
    assert "mcp__tldw_chatbook__create_note" in names


def test_character_writes_are_absent_while_character_tools_are_off(monkeypatch) -> None:
    names = _compose_catalog(monkeypatch, {"character_tools_enabled": False})
    assert "mcp__tldw_chatbook__create_character" not in names
    assert "mcp__tldw_chatbook__update_character" not in names
    assert "mcp__tldw_chatbook__list_characters" in names


def test_character_writes_are_absent_while_the_local_master_is_off(monkeypatch) -> None:
    """Review F3: the master switch drops the native character tools with the
    whole local group, so their MCP twins must go too."""
    names = _compose_catalog(
        monkeypatch, {}, console_cfg={"local_tools_enabled": False}
    )
    assert "mcp__tldw_chatbook__create_character" not in names
    assert "mcp__tldw_chatbook__update_character" not in names
    assert "mcp__tldw_chatbook__create_note" not in names
    assert "mcp__tldw_chatbook__list_characters" in names


def test_unreadable_gates_fall_back_to_their_defaults(monkeypatch) -> None:
    """review-tests F13: the fallback hides create_note (default off) and keeps
    the character writes (default on) -- never everything, never a crash."""

    def unreadable():
        raise OSError("config unreadable")

    names = _compose_catalog(monkeypatch, {})
    monkeypatch.setattr(builtin_tool_gate, "_gate_config_snapshot", unreadable)
    from tldw_chatbook.Agents import mcp_tool_provider

    assert mcp_tool_provider._gated_off_builtin_twins() == frozenset({"create_note"})
    assert "mcp__tldw_chatbook__create_character" in names


# -- AC#3 on dev's stock run start (TASK-34100.13 rebase) -----------------------
# TASK-33620.15.1 / TASK-34561 moved the Console's stock run start off
# ``compose_catalog``: it adopts one shared ``prepare_console_tools`` result,
# and a later run reuses a kept composition while ``composition_key()`` is
# unchanged. The twin rule has to hold on that path, and a gate change has to
# change the key, or a reused catalog keeps a twin the user just turned off.

from Tests.Chat import test_console_async_mcp_snapshot as _snapshot_controls  # noqa: E402

mcp_sources = _snapshot_controls.mcp_sources
local_root = _snapshot_controls.local_root
catalog_store = _snapshot_controls.catalog_store
snapshot_case = _snapshot_controls.snapshot_case


def _twin_inventory_projection(case):
    _snapshot_controls._loop_projection(case)
    case.local.manifest_provider = lambda: {
        "tools": [
            {"name": name, "description": f"{name}.", "inputSchema": {"type": "object"}}
            for name in ("create_note", "list_characters")
        ]
    }


async def _adopted_catalog_names(case, monkeypatch, tools_cfg):
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from tldw_chatbook.Chat.console_chat_controller import (
        CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )
    from tldw_chatbook.MCP.console_tool_preparation import prepare_console_tools

    monkeypatch.setattr(
        builtin_tool_gate, "_gate_config_snapshot", lambda: (dict(tools_cfg), {})
    )
    provider = provider_source.MCPToolProvider(
        service=case.service,
        main_loop=asyncio.get_running_loop(),
        builtin_raw_name_exclusions=CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )
    preparation = await prepare_console_tools(
        case.service,
        include_mcp_catalog=True,
        need_local_switch=True,
        builtin_raw_name_exclusions=CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
        owned_profile_ids=frozenset(),
    )
    assert preparation is not None
    composition = provider_source.capture_standard_controller_composition(
        provider_source.MCPToolProvider, case.service
    )
    provider.adopt_console_preparation(preparation, _controller_composition=composition)
    return provider, {entry.name for entry in provider.list_catalog()}


@pytest.mark.asyncio
async def test_create_note_is_absent_from_the_adopted_console_preparation(
    snapshot_case, monkeypatch
) -> None:
    _twin_inventory_projection(snapshot_case)
    _provider, names = await _adopted_catalog_names(snapshot_case, monkeypatch, {})
    assert "mcp__tldw_chatbook__create_note" not in names
    assert "mcp__tldw_chatbook__list_characters" in names


@pytest.mark.asyncio
async def test_create_note_returns_to_the_adopted_preparation_when_its_gate_is_on(
    snapshot_case, monkeypatch
) -> None:
    _twin_inventory_projection(snapshot_case)
    _provider, names = await _adopted_catalog_names(
        snapshot_case, monkeypatch, {"create_note_enabled": True}
    )
    assert "mcp__tldw_chatbook__create_note" in names


@pytest.mark.asyncio
async def test_a_twin_gate_change_changes_the_run_reuse_key(monkeypatch) -> None:
    from Tests.Agents.test_mcp_tool_provider import FakeMCPService
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    def key_for(tools_cfg):
        monkeypatch.setattr(
            builtin_tool_gate, "_gate_config_snapshot", lambda: (dict(tools_cfg), {})
        )
        provider = MCPToolProvider(
            service=FakeMCPService(inventory={"tools": []}),
            main_loop=asyncio.get_running_loop(),
        )
        return provider.composition_key()

    off = key_for({})
    assert key_for({}) == off
    assert key_for({"create_note_enabled": True}) != off
    assert key_for({"character_tools_enabled": False}) != off
