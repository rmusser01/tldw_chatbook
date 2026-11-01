"""TASK-34100.5.3: the MCP hub's Tool gates pane offers Full agent tools.

Plain Console chat keeps a light, safe tool set until the user turns tools
on. The full agent surface (sub-agents, skills, web, Watchlists, workspace
files, characters) is one switch away: a row in MCP ▸ Servers ▸ built-in row
▸ Tool gates, under a line that names what plain chat already has.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_mcp_servers_mode import CanvasApp
from tldw_chatbook.MCP.readiness import builtin_readiness
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_the_tool_gates_pane_leads_with_full_agent_tools(monkeypatch):
    import tldw_chatbook.Agents.builtin_tool_gate as gate_module

    monkeypatch.setattr(gate_module, "_gate_config_snapshot", lambda: ({}, {}))

    app = CanvasApp()
    async with app.run_test(size=(120, 40)) as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()

        heading = str(app.query_one("#mcp-gate-heading-agent", Static).renderable)
        assert heading.startswith(
            "Plain chat: date and time, calculator and ask you a question, plus "
            "enabled file/note tools and Deep web research."
        )
        # Review round 1 (B-F3): the local master and Character cards read
        # "on" by default, yet add nothing to plain chat -- so the line must
        # not promise "any tool switched on below".
        assert "any tool switched on below" not in heading
        assert "Full agent tools adds the rest: sub-agents" in heading
        button = app.query_one("#mcp-gate-full_agent_tools_enabled", Button)
        # task-35153 (dev): every gate row carries its effect scope.
        assert str(button.label) == "Full agent tools (next chat): off ▸"
        assert "sub-agents" in str(button.tooltip)

        # It leads the pane: above the Agent built-ins group.
        builtin_heading = app.query_one("#mcp-gate-heading-builtin", Static)
        assert button.region.y < builtin_heading.region.y

        button.press()
        await pilot.pause()
        event = app.events[-1]
        assert (event.section, event.key, event.value) == (
            "tools",
            "full_agent_tools_enabled",
            True,
        )


@pytest.mark.asyncio
async def test_the_pane_line_names_only_the_light_tools_that_are_on(monkeypatch):
    """Review round 1 (F4b): the line always named ask-you-a-question, even
    with its own gate off."""
    import tldw_chatbook.Agents.builtin_tool_gate as gate_module

    monkeypatch.setattr(
        gate_module,
        "_gate_config_snapshot",
        lambda: ({"ask_user_enabled": False}, {}),
    )

    app = CanvasApp()
    async with app.run_test(size=(120, 40)) as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()

        heading = str(app.query_one("#mcp-gate-heading-agent", Static).renderable)
        assert heading.startswith("Plain chat: date and time and calculator, plus")
        assert "ask you a question" not in heading


@pytest.mark.asyncio
async def test_a_saved_gate_also_updates_the_apps_config(monkeypatch):
    """Review round 1 (B-F5): switching a gate on in the hub wrote config.toml
    but not the app's in-memory config, so re-running setup in the same
    session showed that gate OFF in its Tools step (which prefills from
    app_config) while the Summary (persisted config) said it was on."""
    import tldw_chatbook.UI.MCP_Modules.mcp_workbench as workbench_module
    from Tests.UI.test_mcp_workbench import WorkbenchApp
    from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
    from tldw_chatbook.UI.Wizards.first_run_setup_state import read_wizard_prefill

    saves: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        workbench_module,
        "save_setting_to_cli_config",
        lambda section, key, value: saves.append((section, key, value)) or True,
    )
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.app_config = {"tools": {"read_file_enabled": False}}
        workbench = app.query_one(MCPWorkbench)

        await workbench._save_tool_gate("tools", "read_file_enabled", True)

        assert saves == [("tools", "read_file_enabled", True)]
        assert app.app_config["tools"]["read_file_enabled"] is True
        assert ("read_file_enabled", True) in read_wizard_prefill(
            app.app_config
        ).tool_gates

        # A write that did not land changes nothing in memory either, and
        # still says so.
        notes: list[str] = []
        monkeypatch.setattr(app, "notify", lambda message, **_k: notes.append(message))
        monkeypatch.setattr(
            workbench_module, "save_setting_to_cli_config", lambda *_a: False
        )
        await workbench._save_tool_gate("tools", "read_file_enabled", False)
        assert app.app_config["tools"]["read_file_enabled"] is True
        assert notes == ["Failed to save read_file_enabled."]
