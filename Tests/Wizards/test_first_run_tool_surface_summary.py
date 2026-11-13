"""TASK-34100.5.3 AC#4: setup says exactly which tools chat is offered.

Before: the Summary's Tools row hedged ("gates off; the assistant may still
use chatbook's own tools (sub-agents, web search, notes, Watchlists) when the
model's context window has room"), the Quick-track note said "tools off", and
the Tools step said "Everything is off by default" -- while plain chat was
handed the whole agent surface. Plain chat now offers the light set until the
user turns more on, and every one of those lines is derived from the same
reading of config as the send.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.builtin_tool_gate import TOOL_GATES_PANE_PATH
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    ROW_CONFIGURED,
    ROW_DEFAULT,
    build_summary_rows,
)

pytestmark = pytest.mark.bootstrap_profile


def _tools_row(config):
    rows = {
        row.label: row
        for row in build_summary_rows(config, {}, rag_deps_installed=False)
    }
    return rows["Tools"]


def test_a_fresh_profile_summary_names_the_light_set_without_hedging():
    row = _tools_row({})

    assert row.state == ROW_DEFAULT
    assert row.detail == (
        "on: date and time, calculator and ask you a question, plus tools from "
        "MCP servers you add and Library tools in chats you allow. Sub-agents, "
        "skills, web, Watchlists, workspace tools and characters stay off "
        f"until you turn on Full agent tools under {TOOL_GATES_PANE_PATH}. "
        "A model with a small or not-yet-known context window gets no tools."
    )
    assert "may still" not in row.detail
    assert "all off" not in row.detail


def test_deep_web_research_on_does_not_also_say_web_stays_off():
    """Review round 1 (F4a): with web_deep_search on, the row listed "Deep web
    research" as on and, in the same sentence, said "web ... stay off"."""
    detail = _tools_row({"tools": {"web_deep_search_enabled": True}}).detail

    assert "Deep web research" in detail
    assert "web, Watchlists" not in detail
    assert "web search and fetch, Watchlists" in detail


def test_tools_switched_on_one_by_one_are_named():
    row = _tools_row(
        {"tools": {"read_file_enabled": True, "create_note_enabled": True}}
    )

    assert row.state == ROW_CONFIGURED
    assert row.detail.startswith(
        "on: date and time, calculator, ask you a question, Read file and "
        "Create note, plus tools from MCP servers you add"
    )


def test_full_agent_tools_on_names_the_agent_groups():
    row = _tools_row({"tools": {"full_agent_tools_enabled": True}})

    assert row.state == ROW_CONFIGURED
    assert row.detail.startswith(
        "full agent tools (sub-agents, skills, web, Watchlists, workspace tools, "
        "chat forks, Canvas, character cards and chatbook's own MCP tools), plus "
        "date and time, calculator and ask you a question."
    )


def test_the_light_set_follows_the_ask_user_gate_and_the_local_master():
    """ask_user is composed by the local provider: its own gate off, or the
    [console] master that composes the provider off, removes it."""
    assert (
        "ask you a question"
        not in _tools_row({"tools": {"ask_user_enabled": False}}).detail
    )
    assert (
        "ask you a question"
        not in _tools_row({"console": {"local_tools_enabled": False}}).detail
    )


def test_the_quick_track_note_describes_the_light_surface():
    """TASK-34100.5 owner question 2: the note said "tools off"."""
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        quick_track_defaults_note,
    )

    note = quick_track_defaults_note({})

    assert note.startswith(
        "Left at recommended defaults: light tools only (date and time, "
        "calculator and ask you a question), RAG off"
    )
    assert "tools off" not in note
    assert quick_track_defaults_note(
        {"tools": {"full_agent_tools_enabled": True}}
    ).startswith("Tools as you set them (full agent tools on).")


@pytest.mark.asyncio
async def test_the_tools_step_states_the_light_set_and_offers_full_agent_tools():
    """Setup's Tools step said "Everything is off by default". It now says
    what chat always has, and its Full agent tools switch writes the gate
    the send reads."""
    from unittest.mock import AsyncMock, MagicMock

    from textual.widgets import Static, Switch

    from Tests.Wizards.test_first_run_setup_wizard import _StepHost
    from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ToolsStep

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        posture = str(step.query_one("#setup-tools-posture", Static).render())
        assert "Everything is off by default" not in posture
        assert posture.startswith(
            "Chat starts with date and time, calculator and ask you a question; "
            "a model with a small or not-yet-known context window gets no tools."
        )
        assert "Always available" not in posture
        # One line, like every other row (the full description was clipped).
        desc = str(step.query_one("#setup-tool-desc-full_agent_tools", Static).render())
        assert desc.startswith("Adds sub-agents, skills, web")
        assert len(desc) <= 110
        switch = step.query_one("#setup-tool-full_agent_tools", Switch)
        assert switch.value is False
        switch.value = True
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {"tools": {"full_agent_tools_enabled": True}}


@pytest.mark.parametrize("gate", ["read_file_enabled", "full_agent_tools_enabled"])
def test_review_rerun_intro_respects_prefilled_enabled_switches(gate):
    from tldw_chatbook.Agents.plain_chat_tool_surface import (
        summarize_tool_surface,
        tools_step_posture,
    )
    from tldw_chatbook.UI.Wizards.first_run_setup_state import read_wizard_prefill

    config = {"tools": {gate: True}}
    assert dict(read_wizard_prefill(config).tool_gates)[gate] is True
    posture = tools_step_posture(summarize_tool_surface(config))
    assert "Everything below is off" not in posture
    assert "switches" in posture.lower()


def _tools_step_over(app_config):
    """A Tools step over one real, mutable ``app_config`` dict."""
    from unittest.mock import AsyncMock, MagicMock

    from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ToolsStep

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = ToolsStep(
        wizard=wizard,
        config=WizardStepConfig(id="tools", title="Tools", step_number=5),
    )
    return wizard, step


@pytest.mark.asyncio
async def test_the_tools_step_shows_and_keeps_the_full_agent_tools_upgrade():
    """TASK-34100.5.3.2 review round 1 (F2-3), deterministic, not a race.

    The wizard composes every step when it mounts, and the app pushes it
    before ``_ui_ready``; the one-time Full agent tools upgrade is a
    staggered boot worker that runs after. So on an upgraded profile whose
    setup is incomplete, the Tools step composed its switch OFF, the upgrade
    then mirrored ``full_agent_tools_enabled = true`` into ``app_config``,
    and the step's commit -- comparing the stale OFF switch with the live
    ON value -- wrote ``false``, undoing the upgrade for good.
    """
    from textual.widgets import Switch

    from Tests.Wizards.test_first_run_setup_wizard import _StepHost
    from tldw_chatbook.Agents.full_agent_tools_upgrade import _mirror_into_app_config

    app_config: dict = {}
    wizard, step = _tools_step_over(app_config)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        switch = step.query_one("#setup-tool-full_agent_tools", Switch)
        assert switch.value is False  # composed before the upgrade landed

        _mirror_into_app_config(app_config)  # the upgrade's UI-thread mirror
        step.on_show()  # BaseWizard.show_step, when the user reaches Tools
        await pilot.pause()

        assert switch.value is True, "the Tools step must show the upgrade"
        ok, _ = await step.commit()
        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_upgrade_landing_while_the_tools_step_is_open_is_never_undone():
    from Tests.Wizards.test_first_run_setup_wizard import _StepHost
    from tldw_chatbook.Agents.full_agent_tools_upgrade import _mirror_into_app_config

    app_config: dict = {}
    wizard, step = _tools_step_over(app_config)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        _mirror_into_app_config(app_config)  # lands after the step was shown

        ok, _ = await step.commit()  # the user never touched the switch

        assert ok
        wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_turning_the_upgraded_switch_off_in_setup_writes_false():
    from textual.widgets import Switch

    from Tests.Wizards.test_first_run_setup_wizard import _StepHost
    from tldw_chatbook.Agents.full_agent_tools_upgrade import _mirror_into_app_config

    app_config: dict = {}
    wizard, step = _tools_step_over(app_config)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        _mirror_into_app_config(app_config)
        step.on_show()
        switch = step.query_one("#setup-tool-full_agent_tools", Switch)
        assert switch.value is True
        switch.value = False  # the user's own choice

        ok, _ = await step.commit()

        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {"tools": {"full_agent_tools_enabled": False}}


@pytest.mark.asyncio
async def test_a_switch_the_user_moved_keeps_their_value_when_tools_is_shown_again():
    """Review v5r1 (F3): ``on_show`` brings only switches the user has NOT
    moved up to the saved gates. Here the upgrade had landed before the step
    composed (switch ON); the user turns Full agent tools off, goes Back and
    comes forward again (``show_step`` -> ``on_show``). Their OFF must
    survive -- re-showing the saved ON would silently undo their choice --
    and commit must write it. A switch they never moved still follows a
    gate saved meanwhile."""
    from textual.widgets import Switch

    from Tests.Wizards.test_first_run_setup_wizard import _StepHost

    app_config: dict = {"tools": {"full_agent_tools_enabled": True}}
    wizard, step = _tools_step_over(app_config)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        switch = step.query_one("#setup-tool-full_agent_tools", Switch)
        read_file = step.query_one("#setup-tool-read_file", Switch)
        assert (switch.value, read_file.value) == (True, False)
        switch.value = False  # the user's own choice
        await pilot.pause()

        app_config["tools"]["read_file_enabled"] = True  # saved meanwhile
        step.on_show()  # Back, then forward to Tools again
        await pilot.pause()

        assert switch.value is False, "the user's OFF was overridden"
        assert read_file.value is True

        ok, _ = await step.commit()

        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {"tools": {"full_agent_tools_enabled": False}}
