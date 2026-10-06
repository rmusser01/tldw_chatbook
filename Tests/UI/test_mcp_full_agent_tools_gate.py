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
            "any tool switched on below."
        )
        button = app.query_one("#mcp-gate-full_agent_tools_enabled", Button)
        assert str(button.label) == "Full agent tools: off ▸"
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
