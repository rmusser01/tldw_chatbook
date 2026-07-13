# Tests/UI/test_mcp_workbench.py
from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ContentSwitcher, Static

from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCP_RAIL_ROW_PREFIX, MCPRail
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench


class FakeTarget:
    server_id = "main"
    label = "Main Server"
    base_url = "https://example.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class FakeTargetStore:
    def list_targets(self):
        return [FakeTarget()]


class FakeHubService:
    def __init__(self) -> None:
        self.target_store = FakeTargetStore()
        self.context = UnifiedMCPContext(selected_source="local", selected_section="overview")

    async def load_context(self):
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_server_target(self, server_id):
        self.context = replace(self.context, selected_active_server_id=server_id)
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        if self.context.selected_source == "local":
            return [
                {
                    "profile_id": "docs",
                    "command": "python",
                    "args": [],
                    "env_placeholders": {},
                    "discovery_snapshot": {"tools": [{"name": "a"}], "resources": [], "prompts": []},
                    "is_connected": True,
                }
            ]
        return {"external_servers": [], "source": "server", "section": "external_servers"}

    def available_actions(self):
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class WorkbenchApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = FakeHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_workbench_mounts_rail_canvas_inspector_and_loads_local_servers():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench.active_mode == "servers"
        rail = app.query_one(MCPRail)
        # builtin + docs rows (+ "All servers")
        assert len(list(app.query("Button.mcp-rail-row"))) == 3
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-overview").display


@pytest.mark.asyncio
async def test_mode_switch_shows_placeholder_canvases():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        switcher = app.query_one(ContentSwitcher)
        assert switcher.current == "mcp-mode-canvas-permissions"
        placeholder = str(
            app.query_one("#mcp-mode-canvas-permissions Static", Static).renderable
        )
        assert "later phase" in placeholder.lower()


@pytest.mark.asyncio
async def test_rail_selection_drives_detail_and_view_state():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-detail").display
        state = app.query_one(MCPWorkbench).get_view_state()
        assert state["selected_server_key"] == "local:docs"
        assert state["mode"] == "servers"


@pytest.mark.asyncio
async def test_restore_tolerates_legacy_and_garbage_state():
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # legacy panel shape
        workbench.set_initial_view_state(
            {"selected_source": "local", "selected_section": "inventory"}
        )
        await pilot.pause()
        assert workbench.active_mode == "servers"
        # garbage
        workbench.set_initial_view_state({"mode": "nonsense", "bogus": 1})
        await pilot.pause()
        assert workbench.active_mode == "servers"
