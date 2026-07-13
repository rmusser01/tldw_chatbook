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
        # Mirrors UnifiedMCPControlPlaneService.load_section(): under the
        # local source, every section is a dict EXCEPT "external_servers",
        # which LocalMCPControlService.get_external_servers() returns as a
        # bare list. Only that one section is a shape gap the workbench must
        # normalize (see _AdvancedSectionShim in mcp_workbench.py).
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
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
            return {"source": "local", "section": effective_section}
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


@pytest.mark.asyncio
async def test_scope_change_and_restore_round_trip():
    """Finding 1: scope/scope_ref must be tracked, not hardcoded to personal/None."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        rail = app.query_one(MCPRail)

        # Before any scope change, view state reports the personal default.
        assert workbench.get_view_state()["scope"] == "personal"
        assert workbench.get_view_state()["scope_ref"] is None

        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()
        state = workbench.get_view_state()
        assert state["scope"] == "team"
        assert state["scope_ref"] == "21"

        # A fresh restore (e.g. re-entering the destination) must be able to
        # bring the same scope back.
        workbench.set_initial_view_state(
            {"mode": "servers", "source": "local", "scope": "team", "scope_ref": "21"}
        )
        await pilot.pause()
        restored = workbench.get_view_state()
        assert restored["scope"] == "team"
        assert restored["scope_ref"] == "21"


def test_active_mode_property_rejects_direct_assignment():
    """Finding 4: active_mode is read-only; set_mode() is the only mutator."""
    app = WorkbenchApp()
    workbench = MCPWorkbench(app_instance=app)
    assert workbench.active_mode == "servers"
    with pytest.raises(AttributeError):
        workbench.active_mode = "tools"


@pytest.mark.asyncio
async def test_set_initial_view_state_during_inflight_reload_applies_pending_state_once():
    """Finding 2: a restore requested while a reload is in flight must not race it.

    A fully black-box reproduction of the race (calling set_initial_view_state
    while on_mount's `await self.reload()` is genuinely suspended mid-await)
    isn't reliably forceable through the public API once `app.run_test()` has
    already settled the initial mount. Instead this asserts the `_reloading`
    guard's contract directly: while a reload is marked in flight, a restore
    request is stashed but not applied; the in-flight reload's own
    end-of-method consumption applies it exactly once, and a repeat
    consumption attempt is a no-op.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)

        apply_calls: list[dict] = []
        original_apply = workbench._apply_view_state

        async def tracking_apply(state):
            apply_calls.append(dict(state))
            await original_apply(state)

        workbench._apply_view_state = tracking_apply

        # Simulate set_initial_view_state() arriving while some other reload
        # (e.g. the destination screen refreshing right after mount) is
        # already in flight.
        workbench._reloading = True
        state = {"mode": "servers", "source": "local", "scope": "team", "scope_ref": "21"}
        workbench.set_initial_view_state(state)
        await pilot.pause()

        # No worker should run while a reload is in flight -- the state is
        # only stashed, not applied.
        assert apply_calls == []
        assert workbench._pending_view_state == state

        # The in-flight reload finishes and, per reload()'s own contract,
        # consumes the pending state exactly once.
        workbench._reloading = False
        await workbench.reload()
        await pilot.pause()

        assert len(apply_calls) == 1
        assert workbench._pending_view_state is None
        restored = workbench.get_view_state()
        assert restored["scope"] == "team"
        assert restored["scope_ref"] == "21"

        # A second consumption attempt must not re-apply.
        await workbench._consume_pending_view_state()
        assert len(apply_calls) == 1


from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen


class _StubApp:
    unified_mcp_service = None


def test_screen_hosts_workbench_with_mode_action_and_tolerant_restore():
    screen = MCPScreen(_StubApp())
    # New surface: workbench host + mode action (old screen has mcp_panel, no workbench).
    assert hasattr(screen, "workbench")
    assert not hasattr(screen, "mcp_panel")
    assert callable(getattr(screen, "action_mcp_mode", None))
    # Never crashes on legacy shape, garbage, or empty state pre-mount.
    screen.restore_state({"unified_mcp_view_state": {"selected_source": "server"}})
    screen.restore_state({"mcp_hub_view_state": {"mode": "tools"}})
    screen.restore_state({})
    state = screen.save_state()
    assert isinstance(state, dict)


def test_mcp_hub_modes_registry_is_complete():
    from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCP_HUB_MODES

    assert list(MCP_HUB_MODES) == ["servers", "tools", "permissions", "audit"]
    for spec in MCP_HUB_MODES.values():
        assert spec["label"] and spec["button_id"].startswith("mcp-mode-")


@pytest.mark.asyncio
async def test_workbench_panes_have_nonzero_geometry():
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for selector in ("#mcp-hub-rail", "#mcp-hub-canvas", "#mcp-hub-inspector"):
            widget = app.query_one(selector)
            assert widget.size.width > 0, f"{selector} has zero width"
            assert widget.size.height > 0, f"{selector} has zero height"
        table = app.query_one("#mcp-servers-table")
        assert table.size.height > 0, "servers table clipped to zero height"
