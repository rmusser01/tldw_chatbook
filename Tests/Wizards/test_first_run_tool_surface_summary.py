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
    quick_track_defaults_note,
)

pytestmark = pytest.mark.bootstrap_profile


def _tools_row(config):
    rows = {row.label: row for row in build_summary_rows(config, {}, rag_deps_installed=False)}
    return rows["Tools"]


def test_a_fresh_profile_summary_names_the_light_set_without_hedging():
    row = _tools_row({})

    assert row.state == ROW_DEFAULT
    assert row.detail == (
        "on: date and time, calculator and ask you a question. Sub-agents, "
        "skills, web, Watchlists, workspace files and characters stay off "
        f"until you turn on Full agent tools under {TOOL_GATES_PANE_PATH}"
    )
    assert "may still" not in row.detail
    assert "all off" not in row.detail


def test_tools_switched_on_one_by_one_are_named():
    row = _tools_row(
        {"tools": {"read_file_enabled": True, "create_note_enabled": True}}
    )

    assert row.state == ROW_CONFIGURED
    assert row.detail.startswith(
        "on: date and time, calculator, ask you a question, Read file and "
        "Create note. Sub-agents, skills"
    )


def test_full_agent_tools_on_names_the_agent_groups():
    row = _tools_row({"tools": {"full_agent_tools_enabled": True}})

    assert row.state == ROW_CONFIGURED
    assert row.detail.startswith(
        "full agent tools (sub-agents, skills, web, Watchlists, workspace files, "
        "chat forks, Canvas, character cards and chatbook's own MCP tools), plus "
        "date and time, calculator and ask you a question."
    )


def test_the_light_set_follows_the_ask_user_gate_and_the_local_master():
    """ask_user is composed by the local provider: its own gate off, or the
    [console] master that composes the provider off, removes it."""
    assert "ask you a question" not in _tools_row(
        {"tools": {"ask_user_enabled": False}}
    ).detail
    assert "ask you a question" not in _tools_row(
        {"console": {"local_tools_enabled": False}}
    ).detail


def test_the_quick_track_note_describes_the_light_surface():
    """TASK-34100.5 owner question 2: the note said "tools off"."""
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
            "Always available in chat: date and time, calculator and ask you "
            "a question."
        )
        switch = step.query_one("#setup-tool-full_agent_tools", Switch)
        assert switch.value is False
        switch.value = True
        ok, _ = await step.commit()
        assert ok
        committed = wizard.commit_config.call_args.args[0]
        assert committed == {"tools": {"full_agent_tools_enabled": True}}
