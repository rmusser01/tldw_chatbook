"""TASK-34100.13: no setup surface claims the assistant's tools are "all off".

fulltrack-02: the Tools step said "Everything is off by default" while plain
chat was handed the whole agent surface; the Summary and the Quick-track note
repeated it. fulltrack-06: the file-tool blurbs implied any path.
new-full-track-steps-02: with "Create note" off, the app's own MCP server still
gave the agent ``create_note``.

Rebased onto TASK-34100.5.3 (the light tool surface): the step's posture, the
Summary's Tools row and the Quick note's tools part now come from ONE reading,
``Agents.plain_chat_tool_surface`` -- the same one the send uses. The branch's
own posture derivation (``tools_posture``/``also_available_line``) was retired
in that rebase; these tests pin what this task still adds on top: the step
reads the config as it is now (not ``load_settings()``'s copy), the Quick note
names search and theme truthfully, the file-tool blurbs and Workspace hint,
the untouched-step baseline, and the built-in MCP server's twins.
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
from tldw_chatbook.Agents.plain_chat_tool_surface import (
    summarize_tool_surface,
    tool_surface_detail,
    tools_step_posture,
)
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


def _posture(step) -> str:
    return str(step.query_one("#setup-tools-posture", Static).render())


def _expected_posture(config) -> str:
    return tools_step_posture(summarize_tool_surface(config))


FALSE_CLAIMS = ("Everything is off", "everything is off", "all off", "tools off")


# -- AC#2: the step -----------------------------------------------------------


@pytest.mark.asyncio
async def test_first_run_step_states_the_light_set_and_claims_nothing_false() -> None:
    """The posture is the shared light-surface sentence; the retired
    "Also available (asks first)" line (web, Watchlists, character cards) is
    gone -- plain chat is not offered those until Full agent tools is on."""
    _wizard, step = _tools_step({})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text = _text(step)
        posture = _posture(step)
        also_lines = list(step.query("#setup-tools-also-available"))
    for claim in FALSE_CLAIMS:
        assert claim not in text
    assert posture == _expected_posture({})
    assert posture.startswith("Chat starts with date and time, calculator and ask you a question;")
    assert also_lines == []


@pytest.mark.asyncio
async def test_the_console_master_switch_is_not_offered_in_setup() -> None:
    """A non-regression pin, not RED evidence for AC#2: the base Tools step
    never offered the ``[console]`` master switch either (review-tests F7).
    The switches are Full agent tools (TASK-34100.5.3) plus one per
    ``_GATEABLE_BUILTINS`` row."""
    _wizard, step = _tools_step({})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switches = list(step.query(Switch))
    assert len(switches) == len(_GATEABLE_BUILTINS) + 1
    assert not step.query(f"#setup-tool-{builtin_tool_gate.LOCAL_TOOLS_MASTER_KEY}")


@pytest.mark.asyncio
async def test_rerun_switches_and_posture_come_from_the_full_toml() -> None:
    """load_settings() has no [tools] table: the gates ride in the raw TOML.

    The switches AND the posture read it -- a saved ``ask_user_enabled =
    false`` takes "ask you a question" out of the light set the step names
    (an ``app_config`` read showed the default light set instead)."""
    raw = {
        "tools": {
            "read_file_enabled": True,
            "write_file_enabled": True,
            "ask_user_enabled": False,
        }
    }
    wizard, step = _tools_step({"COMPREHENSIVE_CONFIG_RAW": raw})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        posture = _posture(step)
        assert step.query_one("#setup-tool-read_file", Switch).value is True
        ok, _ = await step.commit()
    assert ok
    assert posture == _expected_posture(raw)
    assert "ask you a question" not in posture
    wizard.commit_config.assert_not_awaited()  # nothing changed


@pytest.mark.asyncio
async def test_a_saved_full_agent_tools_choice_is_part_of_the_baseline() -> None:
    """Rebase onto TASK-34100.5.3: the Full agent tools switch sits on this
    step, so an untouched re-run over a saved ``true`` writes nothing (the
    baseline once held only the built-in rows, so Next wrote it again)."""
    raw = {"tools": {"full_agent_tools_enabled": True}}
    wizard, step = _tools_step({"COMPREHENSIVE_CONFIG_RAW": raw})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        shown = step.query_one("#setup-tool-full_agent_tools", Switch).value
        ok, _ = await step.commit()
    assert shown is True
    assert ok
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_master_switch_off_names_no_local_tool() -> None:
    """Review F3 (TASK-34100.13): with ``[console] local_tools_enabled`` off the
    Console composes no LocalToolProvider at all, so ask_user goes with web
    search and Watchlists; the step names no local tool."""
    raw = {"console": {"local_tools_enabled": False}}
    _wizard, step = _tools_step({"COMPREHENSIVE_CONFIG_RAW": raw})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        posture = _posture(step)
    assert posture == _expected_posture(raw)
    assert "ask you a question" not in posture
    for name in ("web search", "Watchlists", "character"):
        assert name not in posture


def test_master_switch_off_leaves_no_local_tool_in_the_quick_note() -> None:
    note = wizard_state.quick_track_defaults_note(
        {"tools": {}, "console": {"local_tools_enabled": False}}
    )
    assert "ask you a question" not in note
    assert "character cards" not in note and "questions for you" not in note


@pytest.mark.parametrize(("master", "names_local_groups"), [(True, True), (False, False)])
def test_the_summary_names_the_local_groups_only_while_the_local_tools_are_on(
    master, names_local_groups
) -> None:
    """Review V3 (TASK-34100.13), kept through the rebase: what the Summary
    says the agent gets follows the ``[console]`` master -- the to-do, web,
    Watchlists and workspace tools live in the local provider it drops."""
    config = {
        "tools": {"full_agent_tools_enabled": True},
        "console": {"local_tools_enabled": master},
    }
    detail = _summary_rows(config)["Tools"].detail
    assert ("Watchlists" in detail) is names_local_groups
    assert detail.startswith("full agent tools (sub-agents, skills")


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
    assert row.detail == tool_surface_detail(summarize_tool_surface({}))
    assert builtin_tool_gate.TOOL_GATES_PANE_PATH in row.detail


def test_summary_tools_row_names_the_switches_turned_on() -> None:
    row = _summary_rows({"tools": {"create_note_enabled": True}})["Tools"]
    assert row.state == wizard_state.ROW_CONFIGURED
    assert row.detail.startswith(
        "on: date and time, calculator, ask you a question and Create note"
    )


def test_quick_note_is_derived_from_the_same_reading() -> None:
    note = wizard_state.quick_track_defaults_note({"tools": {}, "console": {}})
    for claim in ("tools off", "all off", "RAG off", "notes sync off"):
        assert claim not in note
    assert "light tools only (date and time, calculator and ask you a question)" in note
    assert "Library searched only when you ask" in note


@pytest.mark.parametrize(
    ("general", "claims_default_theme"),
    [({}, True), ({"default_theme": "textual-dark"}, True), ({"default_theme": "nord"}, False)],
)
def test_the_quick_note_claims_only_what_it_can_see(general, claims_default_theme) -> None:
    """Review R6 (TASK-34100.13): the note ended on the literals "default theme;
    no notes folder sync", false on a Quick re-run over a saved theme or with
    notes sync roots added in Library (which config cannot see). The theme is
    named only while it is the default; notes sync is left to its own row."""
    config = {"tools": {}, "console": {}, "general": general}
    note = wizard_state.quick_track_defaults_note(config)
    assert ("default theme" in note) is claims_default_theme
    assert "notes folder sync" not in note
    assert "notes sync" not in note


@pytest.mark.asyncio
async def test_the_summary_quick_note_is_derived_from_the_reread_config() -> None:
    """review-tests F2: the Summary's own note, not just the pure helper.

    A note left as the old literal ("tools off, RAG off …") also contains
    "recommended defaults", so only the derived text pins the wiring: a
    switch turned on and automatic search must both show up in it.
    """
    from tldw_chatbook.UI.Wizards.first_run_rag_state import EmbeddingModelFacts
    from tldw_chatbook.UI.Wizards.first_run_summary_step import SummaryStep

    reread = {
        "tools": {"read_file_enabled": True},
        "console": {},
        "chat_defaults": {"rag_auto_retrieve_on_send": True},
        "general": {"default_theme": "nord"},
    }
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=6),
        load_config=lambda: reread,
        rag_deps_installed=lambda: True,
        speech_installed=lambda: False,
        speech_runtime_installed=lambda: False,
        rag_model=lambda: EmbeddingModelFacts(name="all-MiniLM-L6-v2", cloud=False),
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        for _ in range(30):
            await pilot.pause(0.05)
            note = str(step.query_one("#setup-summary-defaults-note", Static).render())
            if note:
                break
    assert note == wizard_state.quick_track_defaults_note(reread)
    assert note.startswith("Tools as you set them (")
    assert "Read file" in note
    assert "Library searched on every message" in note
    assert "default theme" not in note  # a saved Nord theme (review R6)
    for claim in ("tools off", "RAG off"):
        assert claim not in note


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


# -- review F8: a same-session re-run reads the config as it is now -------------


class _LiveWizard(SimpleNamespace):
    """Stands in for SetupWizardContainer: the app's own wizard reads live config."""

    reads_live_config = True


def test_the_apps_wizard_reads_live_config() -> None:
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer

    assert SetupWizardContainer.reads_live_config is True


@pytest.mark.asyncio
async def test_a_rerun_shows_gates_changed_since_launch(monkeypatch) -> None:
    """Review F8 (TASK-34100.13): the step read COMPREHENSIVE_CONFIG_RAW, the
    TOML as it was at launch. After a gate was turned on in MCP ▸ Tool gates
    the same session's re-run still showed it off and "All 8 are off". The
    posture reads the same live config (here: ask_user switched off)."""
    import tldw_chatbook.config as config_module

    stale = {"COMPREHENSIVE_CONFIG_RAW": {"tools": {}, "console": {}}}
    live = {"tools": {"read_file_enabled": True, "ask_user_enabled": False}, "console": {}}
    monkeypatch.setattr(
        config_module, "load_cli_config_and_ensure_existence", lambda **_: live
    )
    wizard = _LiveWizard(
        app_instance=MagicMock(app_config=stale),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=7),
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switch_on = step.query_one("#setup-tool-read_file", Switch).value
        sentence = str(step.query_one("#setup-tools-posture", Static).render())
        ok, _ = await step.commit()
    assert switch_on is True
    assert sentence == _expected_posture(live)
    assert "ask you a question" not in sentence
    assert ok
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_untouched_step_writes_nothing_when_the_config_moves_after_compose(
    monkeypatch,
) -> None:
    """Review V2 (TASK-34100.13): the switches came from one config read and
    commit() compared them with a second, fresh one. Live: an external edit
    turned list_directory_enabled on while the app ran, the warm cache still
    showed it off when the wizard composed, the wizard's own first save
    reloaded the file, and the untouched Tools step's Next wrote it back off.
    The delta is taken against what the switches were built from."""
    import tldw_chatbook.config as config_module

    live = {"tools": {"read_file_enabled": True}, "console": {}}
    monkeypatch.setattr(
        config_module, "load_cli_config_and_ensure_existence", lambda **_: live
    )
    wizard = _LiveWizard(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=7),
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        shown = step.query_one("#setup-tool-list_directory", Switch).value
        # The file changes under the composed step (another reader reloaded it).
        live["tools"] = {"read_file_enabled": True, "list_directory_enabled": True}
        ok, _ = await step.commit()
        untouched_writes = wizard.commit_config.await_count
        # A real change still writes only itself, and a second Next after it
        # (Back, then Next) compares against what was saved.
        step.query_one("#setup-tool-read_file", Switch).value = False
        await pilot.pause()
        ok_changed, _ = await step.commit()
        first_write = wizard.commit_config.await_args.args[0]
        await step.commit()
        writes = wizard.commit_config.await_count
    assert shown is False
    assert ok and ok_changed
    assert untouched_writes == 0
    assert first_write == wizard_state.build_tools_commit(
        gate_values={"read_file_enabled": False}
    )
    assert writes == 1
