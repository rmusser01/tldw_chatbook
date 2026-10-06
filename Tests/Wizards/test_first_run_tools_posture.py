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
async def test_master_switch_off_drops_web_and_watchlists_from_the_line() -> None:
    raw = {"console": {"local_tools_enabled": False}}
    _wizard, step = _tools_step({"COMPREHENSIVE_CONFIG_RAW": raw})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        also = str(step.query_one("#setup-tools-also-available", Static).render())
    assert "web search" not in also
    assert "character cards" in also


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
    gates = {gate.key: gate for gate in all_tool_gates(config={"tools": {}, "console": {}})}
    for entry in _GATEABLE_BUILTINS:
        assert gates[entry.gate_key].description == entry.blurb


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


def _compose_catalog(monkeypatch, tools_cfg):
    from Tests.Agents.test_mcp_tool_provider import FakeMCPService, _compose, _tool_dict
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    monkeypatch.setattr(
        builtin_tool_gate, "_gate_config_snapshot", lambda: (dict(tools_cfg), {})
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
