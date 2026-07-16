# Tests/UI/test_mcp_workbench.py
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Checkbox, ContentSwitcher, DataTable, Input, Select, Static, TextArea

import tldw_chatbook.UI.MCP_Modules.mcp_inspector as mcp_inspector_module
import tldw_chatbook.UI.MCP_Modules.mcp_workbench as mcp_workbench_module
from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel, MCPProfileForm
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCP_RAIL_ROW_PREFIX, MCPRail
from tldw_chatbook.UI.MCP_Modules.mcp_server_mutations import MCPServerMutationsPanel
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench


@pytest.fixture(autouse=True)
def _default_advanced_open(monkeypatch):
    """T12: same rationale as test_mcp_inspector.py's fixture of the same
    name -- `MCPWorkbench` mounts a nested `MCPInspector`, whose `compose()`
    reads `mcp.hub_state.advanced_open` via `mcp_inspector.get_cli_setting`
    at mount time. Keep it expanded and never touch the real user config
    file for every workbench test that isn't specifically exercising T12's
    disclosure/persistence behavior itself.
    """
    monkeypatch.setattr(mcp_inspector_module, "get_cli_setting", lambda *a, **k: True)
    monkeypatch.setattr(mcp_inspector_module, "save_setting_to_cli_config", lambda *a, **k: True)


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
        self.disconnect_calls: list[str] = []

    async def disconnect_local_profile(self, profile_id):
        self.disconnect_calls.append(profile_id)
        return True

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
async def test_server_source_add_button_gated_when_mutations_unavailable():
    """T9: `service.available_actions()` not offering `external_server.create`
    (e.g. scope below team/org/system-admin) must disable the overview
    Add-server button in server source, with an explanatory tooltip.
    `FakeHubService.available_actions()` always returns `[]`, so switching
    to server source alone is enough to exercise the gated-off path.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        button = canvas.query_one("#mcp-add-server", Button)
        assert button.disabled is True
        assert button.tooltip == "Requires team, org, or system-admin scope."


@pytest.mark.asyncio
async def test_import_button_gated_off_under_server_source():
    """I3: `MCPWorkbench._apply_import()` always saves to the LOCAL profile
    store (`save_local_profile()`, unconditionally) -- offering Import under
    server source would silently write somewhere invisible in the current
    view. Mirrors `_update_add_server_button()`'s disabled+tooltip gating
    pattern, but on source alone (no scope/target gating applies -- Import
    never touches server-side records at all).
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        button = canvas.query_one("#mcp-import-server", Button)
        assert button.disabled is False
        assert button.tooltip == (
            "Import servers from a Claude-Desktop-style mcpServers JSON file or paste."
        )

        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        assert button.disabled is True
        assert button.tooltip == (
            "Import creates LOCAL server profiles — switch Source to Local."
        )

        rail.post_message(MCPRail.SourceChanged("local"))
        await pilot.pause()
        assert button.disabled is False
        assert button.tooltip == (
            "Import servers from a Claude-Desktop-style mcpServers JSON file or paste."
        )


@pytest.mark.asyncio
async def test_open_add_server_form_local_source_shows_profile_form():
    """T13: `MCPWorkbench.open_add_server_form()` is the `a` keybinding's
    entry point -- it never presses `#mcp-add-server`, so this drives the
    local-source add-form path directly rather than via a Button.Pressed."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench.open_add_server_form()
        await pilot.pause()
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-form").display is True
        form = app.query_one(MCPProfileForm)
        assert form.is_edit is False


@pytest.mark.asyncio
async def test_open_add_server_form_gated_notifies_with_button_tooltip_copy():
    """T13: unlike a click on the already-disabled `#mcp-add-server` button
    (`test_server_source_add_button_gated_when_mutations_unavailable`), the
    `a` keybinding can reach this gate directly with no button to disable --
    it must notify with the SAME copy the button's own tooltip carries
    rather than silently doing nothing.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench.open_add_server_form()
        await pilot.pause()

        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-form").display is False
        assert not app.query(MCPServerMutationsPanel)
        assert notifications
        message, severity = notifications[-1]
        assert message == "Requires team, org, or system-admin scope."
        assert severity == "warning"


class MutationsAvailableTarget:
    server_id = "main"
    label = "Main Server"
    base_url = "https://example.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class MutationsAvailableTargetStore:
    def list_targets(self):
        return [MutationsAvailableTarget()]


class MutationsAvailableHubService:
    """Server-source fake whose `available_actions()` DOES offer the
    external_server.* set (team scope), with a mutable external-records
    list that `external_server.create` appends to -- the "available branch"
    counterpart of `FakeHubService`'s always-gated `[]`.
    """

    def __init__(self) -> None:
        self.target_store = MutationsAvailableTargetStore()
        self.context = UnifiedMCPContext(
            selected_source="server",
            selected_active_server_id="main",
            selected_scope="team",
            selected_section="external_servers",
        )
        self.external_records: list[dict[str, Any]] = []
        self.run_action_calls: list[tuple[str, dict[str, Any]]] = []

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
        return {
            "source": "server",
            "section": "external_servers",
            "external_servers": [dict(r) for r in self.external_records],
        }

    def available_actions(self):
        return [
            {"name": "external_server.create", "label": "Create External Server"},
            {"name": "external_server.update", "label": "Update External Server"},
            {"name": "external_server.slots.list", "label": "List Credential Slots"},
        ]

    async def run_action(self, action_name, payload):
        self.run_action_calls.append((action_name, dict(payload)))
        if action_name == "external_server.create":
            self.external_records.append(
                {
                    "server_id": payload["server_id"],
                    "name": payload["name"],
                    "transport": payload.get("transport", "http"),
                    "enabled": payload.get("enabled", True),
                }
            )
            return {"server_id": payload["server_id"]}
        if action_name == "external_server.slots.list":
            return {"credential_slots": []}
        return {"ok": True}


class MutationsAvailableApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = MutationsAvailableHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_server_source_add_names_implicit_target_and_create_drills_into_new_record():
    """T9 review fix: the "available branch" of the Add-server gate.

    Add-server is only ever reachable from the overview, where nothing is
    selected -- `external_server.create` then attaches to whatever target
    the SERVICE context has active, invisibly. Two behaviors under test:

    1. The enabled button's tooltip names that implicit target ("Adds to
       server: Main Server.") so the attach point is never silent.
    2. After a successful create, the workbench drills into the new record
       (`server:main/<new_id>`): it appears in the collected snapshots and
       the mutation panel re-opens in edit mode for credential setup.
    """
    app = MutationsAvailableApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # Clear the mount-restored target selection: the scenario under test
        # is the overview with NO visible selection while the service still
        # remembers "main" as its active target.
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ServerSelected(None))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        canvas = app.query_one(MCPServersMode)
        button = canvas.query_one("#mcp-add-server", Button)
        assert button.disabled is False
        assert "Main Server" in str(button.tooltip)

        await pilot.click("#mcp-add-server")
        await pilot.pause()
        panel = app.query_one(MCPServerMutationsPanel)
        assert not panel.is_edit
        app.query_one("#mcp-srv-id", Input).value = "docs"
        app.query_one("#mcp-srv-name", Input).value = "Docs"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        svc = app.unified_mcp_service
        assert ("external_server.create", {
            "server_id": "docs", "name": "Docs", "transport": "http",
            "config": {}, "enabled": True,
        }) in svc.run_action_calls
        # Post-create drill: the new record is selected...
        assert workbench.get_view_state()["selected_server_key"] == "server:main/docs"
        # ...its snapshot was actually collected (external-record loading is
        # gated on the ACTIVE target, not the UI selection)...
        assert any(
            snap.server_key == "server:main/docs" for snap in workbench._snapshots
        )
        # ...its credential slots were fetched, and the panel re-opened in
        # edit mode for credential setup.
        assert ("external_server.slots.list", {"server_id": "docs"}) in svc.run_action_calls
        panel = app.query_one(MCPServerMutationsPanel)
        assert panel.is_edit
        assert app.query_one("#mcp-srv-name", Input).value == "Docs"


@pytest.mark.asyncio
async def test_mutations_panel_cancel_clears_selection_and_does_not_reopen_on_resync():
    """I2 regression: `show_server_mutations()` never updates `_detail_snapshot`,
    so Cancel used to restore whatever detail was last shown while
    `_selected_server_key` kept pointing at the external record it was
    hosting -- the very next `_sync_children()` (a background lifecycle
    completion, the `r` keybinding, a runtime-backend refresh) would read
    that stale selection and re-open the SAME mutations panel out of
    nowhere. Cancel must route through the same "clear selection, resync"
    path `ServerRowSelected(None)` uses.
    """
    app = MutationsAvailableApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ServerSelected(None))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Create an external record -- T9's post-create drill leaves the
        # mutations panel open in EDIT mode for it, i.e. exactly the
        # `server:T/R` selection scenario this bug needs.
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-srv-id", Input).value = "docs"
        app.query_one("#mcp-srv-name", Input).value = "Docs"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.get_view_state()["selected_server_key"] == "server:main/docs"
        panel = app.query_one(MCPServerMutationsPanel)
        assert panel.is_edit

        await pilot.click("#mcp-srv-cancel")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench.get_view_state()["selected_server_key"] is None
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-overview").display is True
        assert canvas.query_one("#mcp-servers-detail").display is False
        assert not app.query(MCPServerMutationsPanel)

        # The next resync must not re-open the mutations panel now that the
        # selection that used to point at the external record is gone.
        await workbench._sync_children()
        await pilot.pause()
        assert not app.query(MCPServerMutationsPanel)
        assert canvas.query_one("#mcp-servers-overview").display is True


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
async def test_detail_disconnect_button_routes_through_start_lifecycle():
    """T7: the detail toolbar's Disconnect button (rendered because the
    seeded "docs" profile has `is_connected: True`) must route through the
    same `_start_lifecycle()` dispatch T5 wired for connect/test/refresh --
    not a separate, parallel code path."""
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        await pilot.click("#mcp-detail-disconnect")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.unified_mcp_service.disconnect_calls == ["docs"]


@pytest.mark.asyncio
async def test_builtin_flag_toggle_saves_setting_and_reloads_catalog(monkeypatch):
    """Task 10: toggling the built-in detail's "Enabled" Checkbox must call
    `save_setting_to_cli_config("mcp", "enabled", False)` (monkeypatched
    here, per the task interfaces) and then reload the catalog so the
    checkbox itself -- rebuilt fresh from the post-reload snapshot by
    `MCPServersMode.show_detail()` -- reflects the round trip rather than an
    optimistic local flip. A tiny in-memory `flags` dict stands in for the
    `[mcp]` config section on both the read (`get_cli_setting`) and write
    (`save_setting_to_cli_config`) sides so the reload assertion is a real
    signal instead of coincidentally matching a mocked return value.
    """
    from tldw_chatbook.UI.MCP_Modules import mcp_workbench as workbench_module

    flags: dict[str, Any] = {
        "enabled": True,
        "expose_tools": True,
        "expose_resources": True,
        "expose_prompts": True,
    }
    save_calls: list[tuple[str, str, Any]] = []

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "mcp" and key in flags:
            return flags[key]
        return default

    def fake_save_setting_to_cli_config(section, key, value):
        save_calls.append((section, key, value))
        if section == "mcp":
            flags[key] = value
        return True

    monkeypatch.setattr(workbench_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(
        workbench_module, "save_setting_to_cli_config", fake_save_setting_to_cli_config
    )

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}1")  # builtin row
        await pilot.pause()
        checkbox = app.query_one("#mcp-builtin-enabled", Checkbox)
        assert checkbox.value is True

        await pilot.click("#mcp-builtin-enabled")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert ("mcp", "enabled", False) in save_calls
        # Reload picked up the write: the rebuilt checkbox now shows False,
        # not just an optimistic client-side flip.
        reloaded_checkbox = app.query_one("#mcp-builtin-enabled", Checkbox)
        assert reloaded_checkbox.value is False
        workbench = app.query_one(MCPWorkbench)
        builtin_snap = next(
            s for s in workbench._snapshots if s.server_key == "builtin:tldw_chatbook"
        )
        assert builtin_snap.state.value == "needs_setup"


@pytest.mark.asyncio
async def test_builtin_expose_flag_toggle_saves_matching_key(monkeypatch):
    from tldw_chatbook.UI.MCP_Modules import mcp_workbench as workbench_module

    flags: dict[str, Any] = {
        "enabled": True,
        "expose_tools": True,
        "expose_resources": True,
        "expose_prompts": True,
    }
    save_calls: list[tuple[str, str, Any]] = []

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "mcp" and key in flags:
            return flags[key]
        return default

    def fake_save_setting_to_cli_config(section, key, value):
        save_calls.append((section, key, value))
        if section == "mcp":
            flags[key] = value
        return True

    monkeypatch.setattr(workbench_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(
        workbench_module, "save_setting_to_cli_config", fake_save_setting_to_cli_config
    )

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}1")  # builtin row
        await pilot.pause()
        await pilot.click("#mcp-builtin-expose-resources")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("mcp", "expose_resources", False) in save_calls


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

        # T13: the Add-server form container (`#mcp-servers-form`, styled by
        # the Phase 2 CSS block appended to _agentic_terminal.tcss) must also
        # render with real geometry once shown -- not just the always-visible
        # overview/detail panes checked above.
        workbench = app.query_one(MCPWorkbench)
        await workbench.open_add_server_form()
        await pilot.pause()
        form_container = app.query_one("#mcp-servers-form")
        assert form_container.display is True
        assert form_container.size.width > 0, "add-server form container has zero width"
        assert form_container.size.height > 0, "add-server form container has zero height"


# -- C1: scope-event storm on Server source --------------------------------
#
# Textual 8.2.7 posts a `Select.Changed` for a `Select`'s own constructor
# value as part of mounting it. `MCPRail`'s source select guards this
# mount-echo (`on_select_changed` only forwards when the new value differs
# from `self.source`); the scope and scope-ref selects did not. Because
# `MCPWorkbench._sync_children()` recomposes the rail (new Select instances,
# new mount echoes) and `on_mcp_rail_scope_changed()` used to call
# `_sync_children()` unconditionally, an unguarded echo self-sustained an
# unbounded recompose storm: recompose -> echo -> ScopeChanged ->
# service.select_scope() + `_sync_children()` -> recompose -> echo -> ...


class ScopeTrackingTarget:
    server_id = "main"
    label = "Main Server"
    base_url = "https://example.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class ScopeTrackingTargetStore:
    def list_targets(self):
        return [ScopeTrackingTarget()]


class ScopeTrackingHubService:
    """Like `FakeHubService`, but records every `select_scope()` call."""

    def __init__(self, *, selected_scope: str) -> None:
        self.target_store = ScopeTrackingTargetStore()
        self.context = UnifiedMCPContext(selected_source="server", selected_scope=selected_scope)
        self.select_scope_calls: list[tuple[object, object]] = []

    async def load_context(self):
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_server_target(self, server_id):
        self.context = replace(self.context, selected_active_server_id=server_id)
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        self.select_scope_calls.append((scope, scope_ref))
        self.context = replace(self.context, selected_scope=scope, selected_scope_ref=scope_ref)
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        return {"external_servers": [], "source": "server", "section": section}

    def available_actions(self):
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class ScopeTrackingApp(App):
    def __init__(self, *, selected_scope: str) -> None:
        super().__init__()
        self.unified_mcp_service = ScopeTrackingHubService(selected_scope=selected_scope)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_scope", ["team", "personal"])
async def test_server_source_scope_mount_does_not_storm_select_scope_calls(selected_scope):
    """C1 regression: mounting on Server source must not spam select_scope().

    Covers both halves called out in review: a restored scope outside
    Phase 1's Personal-only rail options ("team", which the rail's display
    clamps to "personal") and the in-options default ("personal", which
    needs no clamp but was never guarded either).
    """
    app = ScopeTrackingApp(selected_scope=selected_scope)
    async with app.run_test() as pilot:
        svc = app.unified_mcp_service
        counts = []
        for _ in range(8):
            await pilot.pause(0.05)
            counts.append(len(svc.select_scope_calls))
        assert all(c == 0 for c in counts), (
            f"select_scope storm at mount (scope={selected_scope!r}): "
            f"{counts} calls={svc.select_scope_calls}"
        )


@pytest.mark.asyncio
async def test_workbench_dedupes_identical_scope_changed_events():
    """C1 fix (b), defense in depth: a repeat ScopeChanged with the same
    (scope, scope_ref) as the workbench's already-tracked state must not
    call service.select_scope() again.
    """
    app = ScopeTrackingApp(selected_scope="personal")
    async with app.run_test() as pilot:
        await pilot.pause()
        rail = app.query_one(MCPRail)
        svc = app.unified_mcp_service
        svc.select_scope_calls.clear()

        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()
        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()

        assert svc.select_scope_calls == [("team", "21")]


# -- T7 carry-over: scope_ref key-absent vs key-present-None ----------------


@pytest.mark.asyncio
async def test_apply_view_state_scope_ref_key_absent_keeps_existing_value():
    """A restore blob with no scope_ref/selected_scope_ref key at all must
    not clobber the currently tracked scope_ref."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()
        assert workbench.get_view_state()["scope_ref"] == "21"

        workbench.set_initial_view_state({"mode": "servers", "source": "local", "scope": "team"})
        await pilot.pause()
        assert workbench.get_view_state()["scope_ref"] == "21"


@pytest.mark.asyncio
async def test_apply_view_state_scope_ref_present_none_clears_existing_value():
    """An explicit `scope_ref: None` key must clear the stale scope_ref
    rather than being treated the same as the key being absent."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.ScopeChanged("team", "21"))
        await pilot.pause()
        assert workbench.get_view_state()["scope_ref"] == "21"

        workbench.set_initial_view_state(
            {"mode": "servers", "source": "local", "scope": "team", "scope_ref": None}
        )
        await pilot.pause()
        assert workbench.get_view_state()["scope_ref"] is None


# -- P1: Advanced > External Servers must not leak unredacted secrets -------
#
# `render_external_servers_section()` (legacy renderer, frozen) keys local
# records by "name", which local profile dicts never have (they use
# "profile_id") -- so it falls back to printing the FULL RAW DICT per entry,
# CLI args and env values included. `_AdvancedSectionShim.load_section()` is
# the seam this surface owns: it must redact each record (via
# `redact_mapping`/`redact_args`) before the payload ever reaches the frozen
# renderer, on both the bare-list local-source normalization path and any
# dict payload that already carries an `external_servers` list.


class SecretLeakHubService:
    """Local-source service whose `external_servers` records carry a
    secret-looking CLI arg (`--api-key sk-qa-test-redact-0001`), mirroring
    what QA saw leak through the Advanced pane."""

    SECRET_VALUE = "sk-qa-test-redact-0001"

    def __init__(self) -> None:
        self.context = UnifiedMCPContext(selected_source="local", selected_section="overview")

    async def load_context(self):
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_server_target(self, server_id):
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "external_servers":
            return [
                {
                    "profile_id": "leaky",
                    "command": "npx",
                    "args": ["--api-key", self.SECRET_VALUE, "--verbose"],
                    "env_placeholders": {},
                    "discovery_snapshot": {"tools": [{"name": "a"}], "resources": [], "prompts": []},
                    "is_connected": True,
                }
            ]
        return {"source": "local", "section": effective_section}

    def available_actions(self):
        return []

    async def run_action(self, action_name, payload):
        return {"ok": True}


class SecretLeakApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = SecretLeakHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_advanced_external_servers_section_redacts_secret_args():
    app = SecretLeakApp()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "external_servers"
        await pilot.pause()
        await pilot.pause()

        rendered = str(app.query_one("#mcp-adv-content", Static).renderable)
        assert SecretLeakHubService.SECRET_VALUE not in rendered, (
            f"secret arg leaked into Advanced > External Servers: {rendered!r}"
        )
        # Non-secret fields must still render -- this isn't just an empty pane.
        assert "npx" in rendered


# -- T5: connect/test/refresh lifecycle wiring, in-flight CHECKING, cancel --


class LifecycleFakeHubService(FakeHubService):
    """Like `FakeHubService`, but wires the typed T2 lifecycle methods and
    records every call so the workbench's dispatch can be asserted on."""

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle_calls: list[tuple[str, str]] = []
        self.connect_gate: asyncio.Event | None = None

    async def load_section(self, section=None):
        # Same shape as FakeHubService.load_section(), except the docs
        # profile fixture is disconnected with no discovery snapshot -- so it
        # derives NEEDS_SETUP (DISCOVERY_NOT_RUN) rather than READY, whose
        # action set is (CONNECT, VALIDATE, VIEW_DETAILS): both lifecycle
        # buttons the tests below click render enabled. (STALE via
        # RUNTIME_UNAVAILABLE would also wire CONNECT, but its action set
        # offers no VALIDATE button at all -- see REASON_TO_ACTIONS in
        # readiness.py.)
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return [
                    {
                        "profile_id": "docs",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": None,
                        "is_connected": False,
                    }
                ]
            return {"source": "local", "section": effective_section}
        return {"external_servers": [], "source": "server", "section": "external_servers"}

    async def local_external_catalog(self):
        return await self.load_section("external_servers")

    async def connect_local_profile(self, profile_id):
        self.lifecycle_calls.append(("connect", profile_id))
        if self.connect_gate is not None:
            await self.connect_gate.wait()
        return {"server_id": profile_id, "tools": [{"name": "a"}], "resources": [], "prompts": []}

    async def test_local_profile(self, profile_id):
        self.lifecycle_calls.append(("test", profile_id))
        return {"ok": True, "profile_id": profile_id, "tools": 1, "resources": 0, "prompts": 0}

    async def refresh_local_profile(self, profile_id):
        self.lifecycle_calls.append(("refresh", profile_id))
        return {"server_id": profile_id, "tools": [], "resources": [], "prompts": []}

    async def disconnect_local_profile(self, profile_id):
        self.lifecycle_calls.append(("disconnect", profile_id))
        return True


class LifecycleApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = LifecycleFakeHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_connect_action_runs_lifecycle_and_notifies():
    app = LifecycleApp()
    # Default 80x24 leaves the inspector pane's action buttons out of the
    # visible region (rail+canvas+inspector min-widths sum to 90 > 80) --
    # see test_workbench_panes_have_nonzero_geometry for the same fix.
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # select docs (local profile, never discovered -> NEEDS_SETUP -> CONNECT wired)
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")
        await pilot.pause()
        await pilot.click("#mcp-inspector-action-connect")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("connect", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight


@pytest.mark.asyncio
async def test_in_flight_shows_checking_and_cancel_then_completes():
    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert selected.state.value == "checking"
        assert list(app.query("#mcp-inspector-cancel"))
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("connect", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight


@pytest.mark.asyncio
async def test_in_flight_checking_message_includes_time_bound(monkeypatch):
    """T7 (P3 UX batch): the CHECKING message ("Working — <action>…") gave
    no indication of how long an in-flight lifecycle op might sit there
    before a user gave up on it. `_display_snapshot()` now appends a time
    bound read from `[mcp] hub_lifecycle_timeout_seconds` (the same setting
    `UnifiedMCPControlPlaneService._lifecycle_timeout()` uses to actually
    enforce the timeout) -- default 45s, formatted as an int.

    Monkeypatches `get_cli_setting` to return each call's own default
    (i.e. "nothing configured") so this assertion can't accidentally pass
    or fail depending on a developer's real `~/.config/tldw_cli/config.toml`.
    """
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )

    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert selected.state.value == "checking"
        assert "(up to" in selected.message
        assert "(up to 45s)" in selected.message
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_in_flight_checking_message_time_bound_honors_config_override(monkeypatch):
    """The time bound is read live from config, not hardcoded -- a
    non-default `hub_lifecycle_timeout_seconds` must show up in the CHECKING
    copy verbatim."""

    def fake_get_cli_setting(section, key=None, default=None):
        if (section, key) == ("mcp", "hub_lifecycle_timeout_seconds"):
            return 12
        return default

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", fake_get_cli_setting)

    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert "(up to 12s)" in selected.message
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_in_flight_checking_message_time_bound_survives_malformed_config(monkeypatch):
    """A non-numeric `hub_lifecycle_timeout_seconds` (e.g. a user fat-fingering
    "soon" into config.toml) must not crash the CHECKING render path --
    `_display_snapshot()` should fall back to the same 45s default that
    `UnifiedMCPControlPlaneService._lifecycle_timeout()` falls back to on the
    same malformed input, rather than letting `float()` raise ValueError
    straight out of a render call."""

    def fake_get_cli_setting(section, key=None, default=None):
        if (section, key) == ("mcp", "hub_lifecycle_timeout_seconds"):
            return "soon"
        return default

    monkeypatch.setattr(mcp_workbench_module, "get_cli_setting", fake_get_cli_setting)

    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert selected.state.value == "checking"
        assert "(up to 45s)" in selected.message
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_cancel_requested_cancels_worker():
    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()  # never set -> hangs
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        workbench.on_mcp_inspector_cancel_requested(
            MCPInspector.CancelRequested("local:docs")
        )
        await pilot.pause()
        assert "local:docs" not in workbench._in_flight


def _capture_notifications(app: App) -> list[tuple[str, str]]:
    """Shadow `app.notify` with a recorder; returns the (message, severity)
    list it appends to. The workbench always notifies via `self.app.notify`,
    so an instance-level shadow intercepts every toast."""
    notifications: list[tuple[str, str]] = []

    def recording_notify(message, *, title="", severity="information", **kwargs):
        notifications.append((str(message), severity))

    app.notify = recording_notify
    return notifications


@pytest.mark.asyncio
async def test_validate_action_runs_test_lifecycle_and_notifies_int_tool_count():
    """VALIDATE dispatch through the real click path, and the
    `_lifecycle_tool_count` int-shape branch: `test_local_profile` returns
    `"tools": 1` (a count, not a list), and the success toast must say
    "1 tool" (singular) from that int."""
    app = LifecycleApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        # select docs (never discovered -> NEEDS_SETUP -> VALIDATE wired)
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")
        await pilot.pause()
        await pilot.click("#mcp-inspector-action-validate")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("test", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight
        successes = [msg for msg, severity in notifications if severity != "error"]
        assert any("docs" in msg and "1 tool" in msg for msg in successes), (
            f"expected an int-derived '1 tool' success toast, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_refresh_lifecycle_dispatches_refresh_method():
    """REFRESH_DISCOVERY's verb mapping ("refresh" -> refresh_local_profile)
    through `_start_lifecycle` -- the third dispatch-table entry the other
    lifecycle tests (all "connect"/"test") leave uncovered."""
    app = LifecycleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "refresh")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("refresh", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight


@pytest.mark.asyncio
async def test_cancel_after_natural_completion_does_not_toast_cancelled():
    """A stale CancelRequested arriving after the operation already finished
    (and popped itself from `_in_flight`) must be a silent no-op -- toasting
    "Cancelled." for something that actually completed would be a lie."""
    app = LifecycleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")  # no gate -> completes
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "local:docs" not in workbench._in_flight
        notifications = _capture_notifications(app)
        workbench.on_mcp_inspector_cancel_requested(
            MCPInspector.CancelRequested("local:docs")
        )
        await pilot.pause()
        assert notifications == [], (
            f"stale cancel must not toast, got: {notifications!r}"
        )


# -- T6: local profile add/edit form hosting + save wiring ------------------


class ProfileFormHubService(FakeHubService):
    """Like `FakeHubService`, but wires `save_local_profile()` -- configurable
    to raise a store-shaped `ValueError` on the next call (mirrors
    `LocalMCPStore`'s "cannot be stored as a literal" copy) or to succeed and
    grow the catalog, so the reload after a successful save has something new
    to show.
    """

    def __init__(self, *, fail_next: bool = False) -> None:
        super().__init__()
        self.save_calls: list[dict] = []
        self._fail_next = fail_next
        # When set, save_local_profile() records its call then blocks on
        # this gate -- lets the double-submit test hold a save in flight.
        self.save_gate: asyncio.Event | None = None
        self.delete_calls: list[str] = []
        # When set, delete_local_profile() records its call then blocks on
        # this gate -- mirrors save_gate, for the double-confirm test.
        self.delete_gate: asyncio.Event | None = None
        self._records: list[dict] = [
            {
                "profile_id": "docs",
                "command": "python",
                "args": [],
                # An unresolved placeholder derives AUTH_MISSING (see
                # local_profile_readiness()), whose action set is
                # (OPEN_CREDENTIALS, EDIT_CONFIG, VIEW_DETAILS) -- unlike a
                # clean READY profile, which offers no EDIT_CONFIG button at
                # all. Needed so the edit-config test below has a button to
                # click.
                "env_placeholders": {"API_KEY": "$MCP_TEST_MISSING_VAR_XYZ"},
                "env_literals": {},
                "discovery_snapshot": {"tools": [{"name": "a"}], "resources": [], "prompts": []},
                "is_connected": True,
            }
        ]

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return list(self._records)
            return {"source": "local", "section": effective_section}
        return {"external_servers": [], "source": "server", "section": "external_servers"}

    async def local_external_catalog(self):
        return list(self._records)

    async def save_local_profile(self, payload):
        self.save_calls.append(dict(payload))
        if self._fail_next:
            self._fail_next = False
            raise ValueError("Secret-bearing env key 'API_KEY' cannot be stored as a literal")
        if self.save_gate is not None:
            await self.save_gate.wait()
        self._records.append(dict(payload))
        return dict(payload)

    async def delete_local_profile(self, profile_id):
        self.delete_calls.append(profile_id)
        if self.delete_gate is not None:
            await self.delete_gate.wait()
        self._records = [r for r in self._records if r.get("profile_id") != profile_id]
        return True


class ProfileFormApp(App):
    def __init__(self, *, fail_next: bool = False) -> None:
        super().__init__()
        self.unified_mcp_service = ProfileFormHubService(fail_next=fail_next)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_add_server_requested_shows_add_mode_form():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert not form.is_edit
        assert app.query_one("#mcp-servers-form").display
        assert not app.query_one("#mcp-servers-overview").display


@pytest.mark.asyncio
async def test_submit_with_service_value_error_renders_store_copy_in_form():
    app = ProfileFormApp(fail_next=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "leaky"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-env", TextArea).text = "API_KEY=raw-literal-not-a-placeholder"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        error_text = str(app.query_one("#mcp-form-error", Static).renderable)
        assert "cannot be stored" in error_text
        # Form stays open on failure -- the user can fix the value and retry.
        assert app.query_one("#mcp-servers-form").display
        assert app.unified_mcp_service.save_calls == [
            {
                "profile_id": "leaky", "command": "npx", "args": [],
                "env_placeholders": {},
                "env_literals": {"API_KEY": "raw-literal-not-a-placeholder"},
            }
        ]


@pytest.mark.asyncio
async def test_submit_success_hides_form_notifies_and_reloads_catalog():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "newprofile"
        app.query_one("#mcp-form-command", Input).value = "npx"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not app.query_one("#mcp-servers-form").display
        assert app.query_one("#mcp-servers-overview").display
        assert app.unified_mcp_service.save_calls[-1]["profile_id"] == "newprofile"
        assert any("newprofile" in msg for msg, _ in notifications)

        # Reload actually picked up the new record -- the overview table now
        # shows both the pre-seeded "docs" profile and the new one.
        workbench = app.query_one(MCPWorkbench)
        keys = {snap.server_key for snap in workbench._snapshots}
        assert "local:newprofile" in keys


@pytest.mark.asyncio
async def test_submit_success_with_secret_shaped_arg_toasts_warning():
    """I4 follow-up (final-review caveat): the in-form
    `#mcp-form-args-warning` Static is unmounted by `hide_form()` sub-second
    after a SUCCESSFUL save, so on exactly the path where the secret got
    persisted the user never saw the warning. The form now carries the
    computed warning on `SubmitRequested`, and the workbench's save-success
    path re-surfaces it as a warning toast alongside the "Saved {id}."
    notify.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "leakyargs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-args", TextArea).text = "-y\nsk-1234567890abcdef"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Save succeeded through the real flow: form gone, record saved.
        assert not app.query_one("#mcp-servers-form").display
        assert app.unified_mcp_service.save_calls[-1]["profile_id"] == "leakyargs"
        assert any("leakyargs" in msg for msg, _ in notifications)
        # ...and the secret-lint warning survived the form's unmount as a toast.
        warnings = [
            msg for msg, severity in notifications
            if severity == "warning" and "visible in process listings" in msg
        ]
        assert warnings, (
            f"expected a secret-lint warning toast on save success, "
            f"got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_submit_success_with_clean_args_toasts_no_warning():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "cleanargs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-args", TextArea).text = (
            "-y\n@modelcontextprotocol/server-filesystem"
        )
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert any("cleanargs" in msg for msg, _ in notifications)
        assert not any(
            severity == "warning" for _, severity in notifications
        ), f"clean args must not produce a warning toast, got: {notifications!r}"


@pytest.mark.asyncio
async def test_cancelled_hides_form_without_saving():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        await pilot.click("#mcp-form-cancel")
        await pilot.pause()
        assert not app.query_one("#mcp-servers-form").display
        assert app.query_one("#mcp-servers-overview").display
        assert app.unified_mcp_service.save_calls == []


@pytest.mark.asyncio
async def test_reload_while_add_form_open_does_not_stack_overview_and_form():
    """I1 regression (review probe): a background resync -- here `reload()`,
    standing in for the `r` keybinding or a runtime-backend refresh --
    must never re-show the overview UNDERNEATH a still-open add/edit form.
    Typed input must survive a resync: the form is only ever hidden/
    remounted by an explicit close (Save/Cancel), never by a passive
    resync.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-command", Input).value = "still-typing"
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-form").display is True
        assert canvas.query_one("#mcp-servers-overview").display is False

        workbench = app.query_one(MCPWorkbench)
        await workbench.reload()
        await pilot.pause()

        form_display = canvas.query_one("#mcp-servers-form").display
        overview_display = canvas.query_one("#mcp-servers-overview").display
        detail_display = canvas.query_one("#mcp-servers-detail").display
        assert form_display is True
        assert not (form_display and overview_display), (
            "overview and form are both visible after reload with form open"
        )
        assert not (form_display and detail_display), (
            "detail and form are both visible after reload with form open"
        )
        # Not just "not stacked" -- the SAME form instance, with the typed
        # value intact, proving the resync never hid/remounted it.
        assert app.query_one("#mcp-form-command", Input).value == "still-typing"


@pytest.mark.asyncio
async def test_rail_selection_while_add_form_open_does_not_stack_detail_and_form():
    """I1 regression (review probe): selecting a different rail row while a
    LOCAL add/edit form is open must keep the form on screen rather than
    stack the detail pane underneath it. Selection interaction decision
    (documented in the final-review-fixes report): the underlying
    `_selected_server_key`/`_detail_snapshot` state DOES still update in the
    background here -- only the container-visibility flip is suppressed --
    so once the form closes (Save/Cancel) the view reflects the latest
    selection rather than snapping back to whatever was selected before the
    form opened. That is a deliberate, minimal-scope consequence of this
    fix, not a separate bug.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench.open_add_server_form()
        await pilot.pause()
        app.query_one("#mcp-form-command", Input).value = "still-typing"
        canvas = app.query_one(MCPServersMode)
        assert canvas.query_one("#mcp-servers-form").display is True

        await workbench._select_server_key("local:docs")
        await pilot.pause()

        form_display = canvas.query_one("#mcp-servers-form").display
        detail_display = canvas.query_one("#mcp-servers-detail").display
        overview_display = canvas.query_one("#mcp-servers-overview").display
        assert form_display is True
        assert not (form_display and detail_display), (
            "detail and form are both visible after rail selection with form open"
        )
        assert not (form_display and overview_display), (
            "overview and form are both visible after rail selection with form open"
        )
        assert app.query_one("#mcp-form-command", Input).value == "still-typing"
        # The background selection DID update (see docstring) -- it just
        # isn't rendered while the form has the floor.
        assert workbench.get_view_state()["selected_server_key"] == "local:docs"


@pytest.mark.asyncio
async def test_edit_config_hub_action_opens_prefilled_form_for_local_profile():
    """EDIT_CONFIG on a local-source snapshot (Task 6 wiring of the
    previously-disabled inspector action) opens the form pre-filled from the
    freshly loaded catalog record for that profile_id -- not just the
    readiness snapshot, which doesn't carry command/args/env.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        assert list(app.query("#mcp-inspector-action-edit_config"))
        edit_button = app.query_one("#mcp-inspector-action-edit_config")
        assert not edit_button.disabled
        await pilot.click("#mcp-inspector-action-edit_config")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert form.is_edit
        assert app.query_one("#mcp-form-id", Input).value == "docs"
        assert app.query_one("#mcp-form-id", Input).disabled
        assert app.query_one("#mcp-form-command", Input).value == "python"


@pytest.mark.asyncio
async def test_detail_edit_button_opens_prefilled_form_for_local_profile():
    """T7: the detail toolbar's Edit button must reuse the exact same
    EDIT_CONFIG path as the inspector's own action button (Task 6) -- not a
    parallel implementation that could drift from it or skip the catalog
    record lookup."""
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        await pilot.click("#mcp-detail-edit")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert form.is_edit
        assert app.query_one("#mcp-form-id", Input).value == "docs"
        assert app.query_one("#mcp-form-command", Input).value == "python"


@pytest.mark.asyncio
async def test_double_submit_dispatches_exactly_one_save():
    """Review fix (Important #1): a second Save while a save is in flight
    must NOT dispatch a second worker. The old handler ran every submit
    through `run_worker(..., exclusive=True)`, so a second click CANCELLED
    the in-flight save mid-write and started a fresh one. Two synchronous
    `Button.press()` calls reproduce it deterministically (pilot.click's
    pump timing masks the race): both `Pressed` messages queue before the
    first handler can disable the button, so two `SubmitRequested` reach
    the workbench -- the in-flight guard must swallow the second with a
    warning toast, leaving exactly one `save_local_profile` call.
    """
    app = ProfileFormApp()
    app.unified_mcp_service.save_gate = asyncio.Event()  # hold save in flight
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-add-server")
        await pilot.pause()
        app.query_one("#mcp-form-id", Input).value = "newprofile"
        app.query_one("#mcp-form-command", Input).value = "npx"
        save_button = app.query_one("#mcp-form-save", Button)
        save_button.press()
        save_button.press()
        await pilot.pause()
        assert any(
            "already running" in msg.lower() and severity == "warning"
            for msg, severity in notifications
        ), f"second submit must toast a warning, got: {notifications!r}"
        app.unified_mcp_service.save_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.unified_mcp_service.save_calls) == 1, (
            f"expected exactly one save call, got: "
            f"{app.unified_mcp_service.save_calls!r}"
        )
        # The single (uncancelled) save completed: form hidden, record saved.
        assert not app.query_one("#mcp-servers-form").display
        workbench = app.query_one(MCPWorkbench)
        assert "local:newprofile" in {s.server_key for s in workbench._snapshots}


@pytest.mark.asyncio
async def test_save_value_error_with_form_gone_notifies_instead_of_vanishing():
    """Review fix (Important #2): if the form is no longer mounted when the
    service raises ValueError (user cancelled while the save worker was in
    flight), the validation failure must surface as an error toast -- never
    disappear silently."""
    app = ProfileFormApp(fail_next=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        # No form mounted -- drive the worker coroutine directly.
        await workbench._save_local_profile(
            {
                "profile_id": "leaky", "command": "npx", "args": [],
                "env_placeholders": {},
                "env_literals": {"API_KEY": "raw-literal"},
            }
        )
        await pilot.pause()
        assert any(
            "cannot be stored" in msg and severity == "error"
            for msg, severity in notifications
        ), f"ValueError with no form must notify, got: {notifications!r}"


# -- T7: DeleteConfirmed wiring -----------------------------------------------


@pytest.mark.asyncio
async def test_delete_confirmed_deletes_profile_clears_selection_and_notifies():
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        assert workbench.get_view_state()["selected_server_key"] == "local:docs"

        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        await pilot.click("#mcp-detail-delete-confirm")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.unified_mcp_service.delete_calls == ["docs"]
        assert workbench.get_view_state()["selected_server_key"] is None
        assert any("Deleted" in msg and "docs" in msg for msg, _ in notifications), (
            f"expected a 'Deleted docs.' toast, got: {notifications!r}"
        )
        assert app.query_one("#mcp-servers-overview").display
        keys = {snap.server_key for snap in workbench._snapshots}
        assert "local:docs" not in keys


@pytest.mark.asyncio
async def test_double_delete_confirm_dispatches_exactly_one_delete():
    """Mirrors test_double_submit_dispatches_exactly_one_save: a second
    `DeleteConfirmed` arriving while a delete worker is already in flight
    must not cancel/duplicate it -- `_profile_delete_in_flight` swallows the
    repeat with a warning toast, leaving exactly one delete call."""
    app = ProfileFormApp()
    app.unified_mcp_service.delete_gate = asyncio.Event()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.on_mcp_servers_mode_delete_confirmed(
            MCPServersMode.DeleteConfirmed("local:docs")
        )
        workbench.on_mcp_servers_mode_delete_confirmed(
            MCPServersMode.DeleteConfirmed("local:docs")
        )
        await pilot.pause()
        assert any(
            "already running" in msg.lower() and severity == "warning"
            for msg, severity in notifications
        ), f"second confirm must toast a warning, got: {notifications!r}"
        app.unified_mcp_service.delete_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.unified_mcp_service.delete_calls == ["docs"]


@pytest.mark.asyncio
async def test_mode_round_trip_disarms_pending_delete_confirmation():
    """Review fix (Important): the arm-then-confirm contract says "any other
    interaction disarms" -- switching modes is such an interaction. Before
    the fix, arming Delete, leaving for Tools, and coming back to Servers
    still rendered the live "Confirm delete" button (the ContentSwitcher
    hides the canvas without unmounting it, so nothing reset the arm state).
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        assert list(app.query("#mcp-detail-delete-confirm"))  # armed

        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        workbench.set_mode("servers")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert list(app.query("#mcp-detail-delete")), (
            "mode round-trip must disarm back to the plain Delete button"
        )
        assert not list(app.query("#mcp-detail-delete-confirm"))
        assert app.unified_mcp_service.delete_calls == []


@pytest.mark.asyncio
async def test_selecting_different_profile_while_armed_disarms():
    """Reviewer's Minor: selecting a DIFFERENT local profile via the rail
    while a delete confirmation is armed must disarm -- otherwise the live
    "Confirm delete" button silently retargets whatever got selected next.
    Already handled by `show_detail()`'s unconditional arm-state reset;
    this locks the behavior in."""
    app = ProfileFormApp()
    app.unified_mcp_service._records.append(
        {
            "profile_id": "web",
            "command": "npx",
            "args": [],
            "env_placeholders": {},
            "env_literals": {},
            "discovery_snapshot": {"tools": [], "resources": [], "prompts": []},
            "is_connected": False,
        }
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")  # local:docs
        await pilot.pause()
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        assert list(app.query("#mcp-detail-delete-confirm"))  # armed

        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}3")  # local:web
        await pilot.pause()

        assert list(app.query("#mcp-detail-delete"))
        assert not list(app.query("#mcp-detail-delete-confirm"))
        assert app.unified_mcp_service.delete_calls == []


@pytest.mark.asyncio
async def test_delete_confirmed_ignores_non_local_server_key():
    """Only local-source server_keys are ever produced by the detail
    toolbar's arm-then-confirm flow (built-in/server-source render no
    toolbar at all -- see test_mcp_servers_mode.py), but the handler must
    not misinterpret a server-source key by calling delete_local_profile
    with a bogus profile id derived from it."""
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.on_mcp_servers_mode_delete_confirmed(
            MCPServersMode.DeleteConfirmed("server:main")
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert app.unified_mcp_service.delete_calls == []


# -- T8: mcpServers import (paste or file) -----------------------------------


class ImportHubService(FakeHubService):
    """Like `FakeHubService`, but wires `save_local_profile()` per-id so a
    test can make one candidate in a batch fail while the rest succeed
    (`fail_ids`), and actually grows its catalog on success -- mirrors
    `ProfileFormHubService` -- so the post-apply reload has something new to
    show. Seeds one "docs" profile, which doubles as the existing-id fixture
    for the overwrite-warning path.
    """

    def __init__(self, *, fail_ids: set[str] | None = None) -> None:
        super().__init__()
        self.save_calls: list[dict] = []
        self._fail_ids = set(fail_ids or ())
        self._records: list[dict] = [
            {
                "profile_id": "docs",
                "command": "python",
                "args": [],
                "env_placeholders": {},
                "env_literals": {},
                "discovery_snapshot": {"tools": [{"name": "a"}], "resources": [], "prompts": []},
                "is_connected": True,
            }
        ]

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return list(self._records)
            return {"source": "local", "section": effective_section}
        return {"external_servers": [], "source": "server", "section": "external_servers"}

    async def local_external_catalog(self):
        return list(self._records)

    async def save_local_profile(self, payload):
        self.save_calls.append(dict(payload))
        if payload.get("profile_id") in self._fail_ids:
            raise ValueError(f"{payload.get('profile_id')}: cannot be saved")
        self._records = [
            r for r in self._records if r.get("profile_id") != payload.get("profile_id")
        ] + [dict(payload)]
        return dict(payload)


class ImportApp(App):
    def __init__(self, *, fail_ids: set[str] | None = None) -> None:
        super().__init__()
        self.unified_mcp_service = ImportHubService(fail_ids=fail_ids)

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_import_paste_preview_apply_calls_save_per_candidate_and_closes_panel():
    app = ImportApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        assert app.query_one("#mcp-servers-form").display
        assert not app.query_one("#mcp-servers-overview").display

        text = json.dumps({"mcpServers": {"web": {"command": "npx", "args": ["-y", "pkg"]}}})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()
        assert not app.query_one("#mcp-import-apply", Button).disabled

        await pilot.click("#mcp-import-apply")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.unified_mcp_service.save_calls == [
            {"profile_id": "web", "command": "npx", "args": ["-y", "pkg"],
             "env_placeholders": {}, "env_literals": {}}
        ]
        assert not app.query_one("#mcp-servers-form").display
        assert app.query_one("#mcp-servers-overview").display
        assert any("web" in msg for msg, _ in notifications)
        workbench = app.query_one(MCPWorkbench)
        keys = {snap.server_key for snap in workbench._snapshots}
        assert "local:web" in keys


@pytest.mark.asyncio
async def test_import_apply_existing_id_warns_and_overwrites():
    """The seeded "docs" profile (from FakeHubService.load_section) is the
    existing-id fixture: previewing an import that reuses "docs" must both
    warn in the panel and still go through save_local_profile (overwrite).
    """
    app = ImportApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        text = json.dumps({"mcpServers": {"docs": {"command": "python3"}}})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()
        body = str(app.query_one("#mcp-import-list Static").renderable)
        assert "overwrite" in body

        await pilot.click("#mcp-import-apply")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.unified_mcp_service.save_calls == [
            {"profile_id": "docs", "command": "python3", "args": [],
             "env_placeholders": {}, "env_literals": {}}
        ]


@pytest.mark.asyncio
async def test_import_apply_failure_produces_summary_notify_without_aborting_rest():
    app = ImportApp(fail_ids={"bad"})
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        text = json.dumps({"mcpServers": {
            "good": {"command": "npx"},
            "bad": {"command": "npx"},
        }})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()

        await pilot.click("#mcp-import-apply")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Both candidates were attempted -- the failure of one did not abort
        # the batch.
        attempted_ids = {call["profile_id"] for call in app.unified_mcp_service.save_calls}
        assert attempted_ids == {"good", "bad"}

        summary = [msg for msg, severity in notifications if "good" in msg or "bad" in msg]
        assert summary, f"expected a combined summary notify, got: {notifications!r}"
        assert any("good" in msg and "bad" in msg for msg in summary), (
            f"expected one summary covering both outcomes, got: {summary!r}"
        )
        # Reload picked up the surviving success.
        workbench = app.query_one(MCPWorkbench)
        keys = {snap.server_key for snap in workbench._snapshots}
        assert "local:good" in keys
        assert "local:bad" not in keys


@pytest.mark.asyncio
async def test_import_double_apply_dispatches_exactly_one_batch():
    app = ImportApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        text = json.dumps({"mcpServers": {"web": {"command": "npx"}}})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()

        workbench = app.query_one(MCPWorkbench)
        panel = app.query_one(MCPImportPanel)
        workbench.on_mcp_import_panel_import_requested(
            MCPImportPanel.ImportRequested(list(panel._candidates))
        )
        workbench.on_mcp_import_panel_import_requested(
            MCPImportPanel.ImportRequested(list(panel._candidates))
        )
        await pilot.pause()
        assert any(
            "already running" in msg.lower() and severity == "warning"
            for msg, severity in notifications
        ), f"second apply must toast a warning, got: {notifications!r}"
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.unified_mcp_service.save_calls) == 1


@pytest.mark.asyncio
async def test_import_cancel_closes_panel_without_saving():
    app = ImportApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        await pilot.click("#mcp-import-cancel")
        await pilot.pause()
        assert not app.query_one("#mcp-servers-form").display
        assert app.query_one("#mcp-servers-overview").display
        assert app.unified_mcp_service.save_calls == []


@pytest.mark.asyncio
async def test_file_requested_pushes_picker_and_loads_selected_file_into_panel(
    tmp_path, monkeypatch
):
    """Workbench's FileRequested handler pushes EnhancedFileOpen filtered to
    JSON and, once a file is picked, writes its text into the panel's
    TextArea (Interfaces: "workbench pushes EnhancedFileOpen(...) and writes
    the file's text into the TextArea").

    F1: `_load_import_file` now validates the picked path with
    `is_safe_path(file_path, home_dir)` -- pytest's `tmp_path` fixture lives
    outside the real home directory, so `expanduser("~")` is patched to
    treat `tmp_path` as home for this test, mirroring the picked file
    legitimately living under the user's home tree in production.
    """
    monkeypatch.setattr(mcp_workbench_module.os.path, "expanduser", lambda _: str(tmp_path))
    app = ImportApp()
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"docs": {"command": "npx"}}}))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#mcp-import-server")
        await pilot.pause()

        pushed: dict[str, Any] = {}

        async def fake_push_screen(screen, callback=None):
            pushed["screen"] = screen
            pushed["title"] = getattr(screen, "title", None)
            if callback is not None:
                callback(config_path)

        app.push_screen = fake_push_screen

        panel = app.query_one(MCPImportPanel)
        panel.post_message(MCPImportPanel.FileRequested())
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pushed, "expected a file picker to be pushed"
        assert app.query_one("#mcp-import-text", TextArea).text.strip() == (
            config_path.read_text().strip()
        )


@pytest.mark.asyncio
async def test_non_utf8_import_file_does_not_crash_app(tmp_path, monkeypatch):
    """C1 regression (review probe): `_load_import_file` previously only
    caught `OSError`. `Path.read_text(encoding="utf-8")` raises
    `UnicodeDecodeError` (a `ValueError` subclass, NOT an `OSError`) for any
    non-UTF-8 file -- e.g. a Claude-Desktop config saved with a UTF-16 BOM.
    Left uncaught, that escapes the worker and, with Textual's default
    `exit_on_error=True`, takes down the whole app. Dispatches through the
    exact `run_worker(..., group="mcp-import-file", exclusive=True)` call
    the real file-picker callback uses (not a bare `await`) -- the crash
    only manifests via that worker boundary.

    F1: `expanduser("~")` is patched to treat `tmp_path` as home so path
    validation doesn't short-circuit before the read is even attempted --
    this test is specifically about the UnicodeDecodeError path, not F1's
    own rejection path (covered separately).
    """
    monkeypatch.setattr(mcp_workbench_module.os.path, "expanduser", lambda _: str(tmp_path))
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe{\"mcpServers\": {}}")
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.run_worker(
            workbench._load_import_file(str(bad)),
            group="mcp-import-file",
            exclusive=True,
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.is_running, "a non-UTF-8 import file must not crash the app"
        assert any(
            "could not read" in msg.lower() and severity == "error"
            for msg, severity in notifications
        ), f"expected an error notify for the unreadable file, got: {notifications!r}"


@pytest.mark.asyncio
async def test_load_import_file_rejects_path_outside_home_directory(tmp_path, monkeypatch):
    """F1 (Qodo compliance finding): the picked import file's path must
    route through `path_validation.is_safe_path()` before it is ever read.
    A path outside the validated root (here, `home`, standing in for
    `expanduser("~")`) is rejected with a plain validation-failure toast --
    no I/O is attempted, and no unread-error message (which would leak the
    fact the path exists) is shown instead.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(mcp_workbench_module.os.path, "expanduser", lambda _: str(home))
    outside = tmp_path / "outside" / "mcp.json"
    outside.parent.mkdir()
    outside.write_text(json.dumps({"mcpServers": {}}))

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.run_worker(
            workbench._load_import_file(str(outside)),
            group="mcp-import-file",
            exclusive=True,
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert notifications == [("Import file path failed validation.", "error")], (
            f"expected exactly one path-validation error toast, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_load_import_file_rejects_oversized_file(tmp_path, monkeypatch):
    """F1 (Qodo compliance finding): a config JSON over the size cap must be
    rejected with a clear size-limit error before its contents are ever read
    into the import panel -- mirrors `attachment_core.MAX_ATTACHMENT_BYTES`'s
    reject-oversized-files precedent (`Tests/Chat/test_attachment_core.py::
    test_process_attachment_path_rejects_oversized_files`), lowering the cap
    via monkeypatch so the test file itself stays small.
    """
    monkeypatch.setattr(mcp_workbench_module.os.path, "expanduser", lambda _: str(tmp_path))
    monkeypatch.setattr(mcp_workbench_module, "MAX_MCP_IMPORT_FILE_BYTES", 16)
    big = tmp_path / "big.json"
    big.write_text("x" * 64)

    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.run_worker(
            workbench._load_import_file(str(big)),
            group="mcp-import-file",
            exclusive=True,
        )
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert any(
            "too large" in msg.lower() and severity == "error"
            for msg, severity in notifications
        ), f"expected an oversized-file error notify, got: {notifications!r}"


@pytest.mark.asyncio
async def test_delete_local_profile_notify_escapes_markup_in_profile_id():
    """F3 (Gemini finding, adapted): the local store keeps a profile id RAW
    on purpose, so a profile id shaped like Rich markup (e.g. embedded via a
    hand-edited config or import) must be escaped by `_toast()` before it
    reaches the "Deleted ..." `app.notify()` toast -- otherwise
    `[red]x[/red]` would be interpreted as styling/control markup instead of
    displayed literally.
    """
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        await workbench._delete_local_profile("local:[red]x[/red]", "[red]x[/red]")
        await pilot.pause()
        assert any("\\[red]x\\[/red]" in msg for msg, _ in notifications), (
            f"expected the escaped literal in the toast, got: {notifications!r}"
        )
        assert not any(msg == "Deleted [red]x[/red]." for msg, _ in notifications), (
            f"profile id markup must not reach notify() unescaped, got: {notifications!r}"
        )


@pytest.mark.asyncio
async def test_notify_survives_markup_bearing_text():
    """Review probe (b): a message that looks like unbalanced Rich markup
    (e.g. embedded in a profile id) must not crash `app.notify()`. Kept as a
    permanent regression guard even though it already passed pre-fix --
    documents that this adjacent surface is NOT the C1 crash and must stay
    that way.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.notify("Saved [/bold]x.")
        await pilot.pause()
        assert app.is_running


# -- Task 12: Advanced disclosure object label + info-callout placeholders --


@pytest.mark.asyncio
async def test_advanced_object_label_updates_on_source_switch():
    """UX-inputs #1: switching source must rebind the inspector's Advanced
    object label (and, per `MCPInspector.set_service_context()`, reset/
    reload its section content) so a previous object's facts never linger.
    `FakeHubService.target_store` labels server_id "main" as "Main Server".
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        label = app.query_one("#mcp-adv-object", Static)
        assert str(label.renderable) == "Showing: Local control plane"

        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        await pilot.pause()
        rail.post_message(MCPRail.ServerSelected("server:main"))
        await pilot.pause()
        await pilot.pause()

        assert str(label.renderable) == "Showing: server Main Server"

        # Switching back to local must rebind the label again, not leave the
        # server-source text stuck on screen.
        rail.post_message(MCPRail.SourceChanged("local"))
        await pilot.pause()
        await pilot.pause()
        assert str(label.renderable) == "Showing: Local control plane"


class AuxTarget:
    server_id = "aux"
    label = "Aux Server"
    base_url = "https://aux.test"
    auth_mode = "api_key"
    last_known_reachability = "reachable"
    last_known_auth_state = "authenticated"


class TwoTargetStore:
    def list_targets(self):
        return [FakeTarget(), AuxTarget()]


@pytest.mark.asyncio
async def test_same_row_reclick_preserves_advanced_section():
    """Review fix (T12): the UX-inputs text says rebind on selection
    CHANGE -- a reclick of the already-selected row is not a change, and
    must not wipe the user's Advanced browsing state (section snapping back
    to Overview). Mirrors the C1 ScopeChanged dedup precedent in this file.
    """
    app = WorkbenchApp()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        await pilot.pause()
        rail.post_message(MCPRail.ServerSelected("server:main"))
        await pilot.pause()
        await pilot.pause()

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "inventory"
        await pilot.pause()
        await pilot.pause()
        assert section_select.value == "inventory"

        # Reclick the SAME row: not a selection change -- browsing state
        # must survive.
        rail.post_message(MCPRail.ServerSelected("server:main"))
        await pilot.pause()
        await pilot.pause()
        assert section_select.value == "inventory"


@pytest.mark.asyncio
async def test_different_target_selection_rebinds_advanced_context():
    """Counterpart guard: a GENUINE selection change (a different server
    target) must still rebind -- the section resets to the first entry and
    the object label follows the new target."""
    app = WorkbenchApp()
    app.unified_mcp_service.target_store = TwoTargetStore()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        rail = app.query_one(MCPRail)
        rail.post_message(MCPRail.SourceChanged("server"))
        await pilot.pause()
        await pilot.pause()
        rail.post_message(MCPRail.ServerSelected("server:main"))
        await pilot.pause()
        await pilot.pause()

        section_select = app.query_one("#mcp-adv-section-select", Select)
        section_select.value = "inventory"
        await pilot.pause()
        await pilot.pause()

        rail.post_message(MCPRail.ServerSelected("server:aux"))
        await pilot.pause()
        await pilot.pause()

        assert section_select.value == "overview"
        label = app.query_one("#mcp-adv-object", Static)
        assert str(label.renderable) == "Showing: server Aux Server"


@pytest.mark.asyncio
async def test_mode_placeholder_canvases_use_info_callout_not_recovery_callout():
    """UX-inputs #4: phase placeholders are informational, not an alarm
    condition -- they must carry `.ds-info-callout`, never the orange-chrome
    `.ds-recovery-callout` used for actionable problems elsewhere in the hub.
    """
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        for mode, spec in MCP_HUB_MODES.items():
            # T5: "tools" now hosts the real `MCPToolsMode` canvas (see
            # test_mcp_tools_mode.py for its own empty-state coverage), not
            # a generic phase placeholder -- only "servers" and "tools" are
            # real canvases at this point.
            if mode in ("servers", "tools"):
                continue
            static = app.query_one(f"#mcp-mode-canvas-{mode} Static", Static)
            assert "ds-info-callout" in static.classes, mode
            assert "ds-recovery-callout" not in static.classes, mode
            assert str(static.renderable) == spec["placeholder"]


# -- T5: Tools mode canvas registration + workbench-fed catalog -------------


@pytest.mark.asyncio
async def test_tools_mode_canvas_replaces_placeholder():
    """`#mcp-mode-canvas-tools` must host the real `MCPToolsMode` widget --
    not the generic placeholder Vertical/Static every other not-yet-built
    mode still renders."""
    app = WorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.query_one("#mcp-mode-canvas-tools"), MCPToolsMode)
        assert not list(app.query("#mcp-mode-canvas-tools > Static"))


@pytest.mark.asyncio
async def test_tools_mode_shows_tools_from_local_catalog_snapshots():
    """T5: `_collect_hub_tools()` derives Tools mode's catalog from the SAME
    local-profile records `_collect_snapshots()` already loaded for the rail/
    overview (`FakeHubService`'s seeded "docs" profile, discovery_snapshot
    tools=[{"name": "a"}]) -- no separate fetch.
    """
    app = WorkbenchApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        table = canvas.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
        row_key, _ = table.coordinate_to_cell_key((0, 0))
        assert row_key.value == "local:docs::a"
        assert canvas.query_one("#mcp-tools-empty").display is False


class NoServersHubService(FakeHubService):
    """Local source with zero local profiles configured (and no builtin
    inventory, since this fake never sets `local_service`) -- the "no
    servers configured" empty-diagnosis bucket."""

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return []
            return {"source": "local", "section": effective_section}
        return {"external_servers": [], "source": "server", "section": "external_servers"}


class NoServersApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = NoServersHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_empty_diagnosis_no_servers_shows_add_server_and_button_opens_form():
    app = NoServersApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")  # the button must be on-screen for pilot.click()
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        empty = canvas.query_one("#mcp-tools-empty")
        assert empty.display is True
        message = str(canvas.query_one("#mcp-tools-empty-message", Static).renderable)
        assert message == "No servers configured — add one to see its tools."

        await pilot.click("#mcp-tools-empty-action")
        await pilot.pause()
        form = app.query_one(MCPProfileForm)
        assert not form.is_edit


@pytest.mark.asyncio
async def test_empty_diagnosis_connect_routes_to_servers_mode_with_notify():
    """LifecycleFakeHubService's seeded "docs" profile is disconnected with
    no discovery snapshot -> NEEDS_SETUP -- the "servers exist but none
    connected/discovered" empty-diagnosis bucket. The empty state's button
    must switch to Servers mode (where the real connect/refresh actions
    live) and notify, not attempt any lifecycle action itself.
    """
    app = LifecycleApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        message = str(canvas.query_one("#mcp-tools-empty-message", Static).renderable)
        assert message == "No tools discovered yet — connect or refresh a server."

        notifications = _capture_notifications(app)
        await pilot.click("#mcp-tools-empty-action")
        await pilot.pause()
        assert workbench.active_mode == "servers"
        assert notifications and notifications[-1] == (
            "Select a server below to connect or refresh its tools.",
            "information",
        )


class ServerToolsHubService(FakeHubService):
    """Server source with one active target ("main") whose sole external
    record embeds its own `tools` list -- mirrors a backend that returns
    per-record tool inventories inline (see `readiness.py`'s own
    `record.get("tools")` tool_count fallback, the same embedded shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.context = UnifiedMCPContext(
            selected_source="server", selected_active_server_id="main"
        )

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if effective_section == "external_servers":
            return {
                "external_servers": [
                    {
                        "server_id": "docs",
                        "name": "Docs",
                        "tools": [{"name": "search", "description": "Search."}],
                    }
                ],
                "source": "server",
                "section": "external_servers",
            }
        return {"external_servers": [], "source": "server", "section": effective_section}


class ServerToolsApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = ServerToolsHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_tools_mode_shows_server_source_tools_from_embedded_inventory():
    app = ServerToolsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.query_one(MCPToolsMode)
        table = canvas.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
        row_key, _ = table.coordinate_to_cell_key((0, 0))
        assert row_key.value == "server:main/docs::search"
        assert canvas.query_one("#mcp-tools-empty").display is False


class DuplicateNameToolsHubService(FakeHubService):
    """A local external profile whose discovery snapshot carries two
    same-named tools -- C1: this used to crash every mount of the Tools mode
    canvas (Textual `DuplicateKey` on the DataTable row key, which is
    `HubTool.tool_id`), and since discovery snapshots persist verbatim to
    disk, it was a permanent crash-loop, not a one-off."""

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_source or "overview"
        if self.context.selected_source == "local" and effective_section == "external_servers":
            return [
                {
                    "profile_id": "docs",
                    "command": "python",
                    "args": [],
                    "env_placeholders": {},
                    "discovery_snapshot": {
                        "tools": [
                            {"name": "search", "description": "a"},
                            {"name": "search", "description": "b"},
                        ],
                        "resources": [],
                        "prompts": [],
                    },
                    "is_connected": True,
                }
            ]
        return {"source": "local", "section": effective_section}

    async def local_external_catalog(self):
        return await self.load_section("external_servers")


class DuplicateNameToolsApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = DuplicateNameToolsHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_duplicate_tool_names_do_not_crash_workbench_mount():
    app = DuplicateNameToolsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 1
        row_key, _ = table.coordinate_to_cell_key((0, 0))
        assert row_key.value == "local:docs::search"


# -- Task 6: inspector tool detail + Test Tool runner wiring -----------------
#
# `ToolTestHubService` seeds two local profiles: "docs" (tools "fetch" --
# no inputSchema, raw-mode -- and "search" -- a schema with a required
# "query" string, form-mode) and "notes" (tool "list_notes"). Sorted by
# (server_label, name) per MCPToolsMode._apply_filter(), the table rows are:
# 0=docs::fetch, 1=docs::search, 2=notes::list_notes.


class ToolTestHubService(FakeHubService):
    def __init__(self) -> None:
        super().__init__()
        self.test_calls: list[tuple[str, str, dict]] = []
        self.test_result: Any = {"ok": True}
        self.raise_error: Exception | None = None
        # When set, test_hub_tool() records its call then blocks on this
        # gate -- mirrors LifecycleFakeHubService.connect_gate, for the
        # double-run test.
        self.test_gate: asyncio.Event | None = None

    async def load_section(self, section=None):
        effective_section = section or self.context.selected_section or "overview"
        if self.context.selected_source == "local":
            if effective_section == "external_servers":
                return [
                    {
                        "profile_id": "docs",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": {
                            "tools": [
                                {"name": "fetch", "description": "Fetch a doc."},
                                {
                                    "name": "search",
                                    "description": "Search the docs.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "Search text",
                                            }
                                        },
                                        "required": ["query"],
                                    },
                                },
                            ],
                            "resources": [],
                            "prompts": [],
                        },
                        "is_connected": True,
                    },
                    {
                        "profile_id": "notes",
                        "command": "python",
                        "args": [],
                        "env_placeholders": {},
                        "discovery_snapshot": {
                            "tools": [{"name": "list_notes", "description": "List notes."}],
                            "resources": [],
                            "prompts": [],
                        },
                        "is_connected": True,
                    },
                ]
            return {"source": "local", "section": effective_section}
        return {"external_servers": [], "source": "server", "section": "external_servers"}

    async def local_external_catalog(self):
        return await self.load_section("external_servers")

    async def test_hub_tool(self, server_key, tool_name, arguments=None):
        self.test_calls.append((server_key, tool_name, dict(arguments or {})))
        if self.test_gate is not None:
            await self.test_gate.wait()
        if self.raise_error is not None:
            raise self.raise_error
        return self.test_result


class ToolTestApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = ToolTestHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


async def _select_tools_mode_row(app: App, pilot, row: int) -> None:
    table = app.query_one("#mcp-tools-table", DataTable)
    table.focus()
    table.move_cursor(row=row)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_tool_row_selection_shows_tool_detail_with_test_button():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch
        name_text = str(app.query_one("#mcp-inspector-tool-name", Static).renderable)
        assert "docs" in name_text
        test_button = app.query_one("#mcp-inspector-test-tool", Button)
        assert test_button.tooltip == "Run this tool with test arguments."


@pytest.mark.asyncio
async def test_switching_mode_clears_tool_detail():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        assert list(app.query("#mcp-inspector-tool-name"))

        workbench.set_mode("servers")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-tool-name"))


@pytest.mark.asyncio
async def test_switching_selected_server_clears_tool_detail():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)
        assert list(app.query("#mcp-inspector-tool-name"))

        await workbench._select_server_key("local:notes")
        await pilot.pause()
        assert not list(app.query("#mcp-inspector-tool-name"))


@pytest.mark.asyncio
async def test_second_tool_selection_back_to_back_does_not_duplicate_ids():
    """Mandatory regression (mirrors test_mcp_inspector.py's
    test_second_show_tool_back_to_back_does_not_duplicate_ids): selecting a
    second tool before the first selection's inspector refresh has settled
    must not raise DuplicateIds."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        tools = workbench._last_hub_tools
        tool_a = next(t for t in tools if t.name == "fetch")
        tool_b = next(t for t in tools if t.name == "search")
        await workbench.on_mcp_tools_mode_tool_selected(MCPToolsMode.ToolSelected(tool_a.tool_id))
        # No pause here on purpose.
        await workbench.on_mcp_tools_mode_tool_selected(MCPToolsMode.ToolSelected(tool_b.tool_id))
        await pilot.pause()
        names = list(app.query("#mcp-inspector-tool-name"))
        assert len(names) == 1
        assert "search" in str(names[0].renderable)


def test_mcp_workbench_source_never_parses_packed_tool_id():
    """task-233 grep-gate: the execute path now carries (server_key,
    tool_name) as separate fields end to end -- nothing in mcp_workbench.py
    may reconstruct them by parsing a packed "server_key::tool_name" string
    anymore. Row keys elsewhere (mcp_tools_mode.py) are unaffected -- this
    only asserts mcp_workbench.py's own source never parses one."""
    source = Path(mcp_workbench_module.__file__).read_text()
    assert 'partition("::")' not in source
    assert "partition('::')" not in source
    assert 'split("::")' not in source
    assert "split('::')" not in source


@pytest.mark.asyncio
async def test_tool_for_resolves_by_server_key_and_tool_name():
    """`_tool_for(server_key, tool_name)` compares fields, not a packed
    string -- distinct from `_tool_for_row_key()`, which still resolves the
    Tools-mode DataTable's packed row key."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        tools = workbench._last_hub_tools
        fetch_tool = next(t for t in tools if t.name == "fetch")

        resolved = workbench._tool_for(fetch_tool.server_key, fetch_tool.name)
        assert resolved is fetch_tool

        # Same tool_name, different server -- must not match.
        assert workbench._tool_for("local:notes", "fetch") is None
        # Same server, unknown tool_name -- must not match.
        assert workbench._tool_for(fetch_tool.server_key, "does-not-exist") is None


@pytest.mark.asyncio
async def test_test_tool_run_success_calls_service_and_renders_ok():
    app = ToolTestApp()
    app.unified_mcp_service.test_result = {"ok": True}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)  # docs::search (form schema)
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        app.query_one("#mcp-schema-field-0", Input).value = "hello"
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.unified_mcp_service.test_calls == [
            ("local:docs", "search", {"query": "hello"})
        ]
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        first_line = result.split("\n", 1)[0]
        assert first_line.startswith("OK · ")
        assert first_line.endswith("ms")
        assert app.query_one("#mcp-inspector-test-run", Button).disabled is False


@pytest.mark.asyncio
async def test_test_tool_run_error_renders_failed_with_message():
    app = ToolTestApp()
    app.unified_mcp_service.raise_error = RuntimeError("boom")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        first_line = result.split("\n", 1)[0]
        assert first_line.startswith("Failed · ")
        assert first_line.endswith("ms")
        assert "boom" in result


@pytest.mark.asyncio
async def test_test_tool_run_redacts_secret_shaped_result():
    app = ToolTestApp()
    app.unified_mcp_service.test_result = {"ok": True, "api_key": "sk-live-secret"}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "sk-live-secret" not in result
        assert "***" in result


@pytest.mark.asyncio
async def test_test_tool_run_error_with_dict_shaped_args_is_redacted():
    """I1 (ledger #5): some errors carry a raw dict payload in `exc.args`
    (e.g. an echoed request/arguments dict) -- `str(exc)` would otherwise
    dump that dict's raw repr, including any secret-shaped values in it,
    straight into the result panel."""
    app = ToolTestApp()
    app.unified_mcp_service.raise_error = RuntimeError(
        {"api_key": "sk-live-secret", "detail": "bad request"}
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "sk-live-secret" not in result
        assert "***" in result
        assert "bad request" in result


@pytest.mark.asyncio
async def test_test_tool_double_run_dispatches_exactly_one_service_call():
    """Mirrors test_double_submit_dispatches_exactly_one_save: the workbench
    in-flight guard, not just the Run button's own disabled state, is the
    authoritative dedupe for a second ToolTestRequested reaching the
    workbench before the first has completed (two Pressed messages queued
    before the first handler can disable anything)."""
    app = ToolTestApp()
    app.unified_mcp_service.test_gate = asyncio.Event()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        tools = workbench._last_hub_tools
        tool = next(t for t in tools if t.name == "search")
        event = MCPInspector.ToolTestRequested(tool.server_key, tool.name, {"query": "hello"})
        workbench.on_mcp_inspector_tool_test_requested(event)
        workbench.on_mcp_inspector_tool_test_requested(event)
        await pilot.pause()
        assert any(
            "already running" in msg.lower() and severity == "warning"
            for msg, severity in notifications
        ), f"expected a warning toast, got: {notifications!r}"
        app.unified_mcp_service.test_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.unified_mcp_service.test_calls) == 1


@pytest.mark.asyncio
async def test_slow_tool_result_does_not_render_under_a_different_selected_tool():
    """I1: tool A's ("docs::fetch") slow test run must not land in tool B's
    ("notes::list_notes") panel when the user switches selection before A
    resolves -- and must not re-enable B's Run button on A's behalf.
    Mirrors the whole-branch review's end-to-end probe."""
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        service = app.unified_mcp_service
        workbench.set_mode("tools")
        await pilot.pause()

        # Select docs::fetch (row 0, raw mode), open Test panel, Run (gated).
        await _select_tools_mode_row(app, pilot, 0)
        app.query_one("#mcp-inspector-test-tool", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        service.test_gate = asyncio.Event()
        service.test_result = {"result": "FETCH-DOC-PAYLOAD"}
        app.query_one("#mcp-inspector-test-run", Button).press()
        await pilot.pause()
        assert service.test_calls and service.test_calls[0][1] == "fetch"

        # While fetch is in flight, select notes::list_notes and open ITS panel.
        await _select_tools_mode_row(app, pilot, 2)
        await pilot.pause()
        inspector = app.query_one(MCPInspector)
        assert inspector.current_tool.name == "list_notes"
        app.query_one("#mcp-inspector-test-tool", Button).press()
        await pilot.pause()
        await pilot.pause()
        result_widget = app.query_one("#mcp-inspector-test-result", Static)
        run_button = app.query_one("#mcp-inspector-test-run", Button)
        assert str(result_widget.renderable) == ""

        # Release the gate: fetch's late result must be dropped, not shown
        # under notes::list_notes, and must not touch list_notes's own Run.
        service.test_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = str(result_widget.renderable)
        assert "FETCH-DOC-PAYLOAD" not in rendered
        assert rendered == ""
        assert inspector.current_tool.name == "list_notes"
        assert run_button.disabled is False  # never pressed for list_notes


@pytest.mark.asyncio
async def test_test_tool_run_non_str_dict_key_result_does_not_crash():
    """Critical regression: `_run_tool_test()`'s success-path result
    formatting (`json.dumps(redact_mapping(result), default=str)`) used to
    sit OUTSIDE the inner try/except. A result dict with a non-str key (a
    tuple, here) makes `json.dumps` raise `TypeError` -- `default=str` only
    covers values, not keys -- and that exception used to escape the worker
    body entirely. Textual's `run_worker()` defaults to `exit_on_error=True`,
    so an uncaught exception there panics the whole app rather than just
    failing this one tool test. After the fix, formatting errors must be
    caught and rendered as a failed result like any other test failure."""
    app = ToolTestApp()
    app.unified_mcp_service.test_result = {("tuple", "key"): 1}
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw, default "{}")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        first_line = result.split("\n", 1)[0]
        assert first_line.startswith("Failed · ")
        assert app.query_one("#mcp-inspector-test-run", Button).disabled is False
        assert workbench._tool_test_in_flight == set()


@pytest.mark.asyncio
async def test_collect_arguments_value_error_does_not_call_service():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)  # docs::search (required "query")
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        # required "query" field left empty
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        assert app.unified_mcp_service.test_calls == []
        result = str(app.query_one("#mcp-inspector-test-result", Static).renderable)
        assert "required" in result


@pytest.mark.asyncio
async def test_raw_mode_tool_run_posts_parsed_json_to_service():
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        table = app.query_one("#mcp-tools-table", DataTable)
        assert table.row_count == 3
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch (raw mode)
        await pilot.click("#mcp-inspector-test-tool")
        await pilot.pause()
        raw_area = app.query_one("#mcp-schema-raw", TextArea)
        raw_area.text = '{"id": 42}'
        await pilot.click("#mcp-inspector-test-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("local:docs", "fetch", {"id": 42}) in app.unified_mcp_service.test_calls


# -- Task 8: `t` keybinding entry point (open_test_for_selected_tool) --------


@pytest.mark.asyncio
async def test_open_test_for_selected_tool_with_no_selection_notifies():
    """T8: the `t` keybinding's workbench entry point -- with nothing
    selected in the inspector's tool-detail view, notifies instead of
    silently no-opping, mirroring `open_add_server_form()`'s T13 rationale
    for a keybinding that can reach a state no disabled button gates. Also
    switches to Tools mode even though nothing is selected there yet --
    same "the keybinding always lands you in the right mode" contract as
    `action_mcp_add_server`.
    """
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        assert workbench.active_mode == "servers"

        await workbench.open_test_for_selected_tool()
        await pilot.pause()

        assert workbench.active_mode == "tools"
        assert not list(app.query("#mcp-inspector-test-panel"))
        assert notifications
        message, severity = notifications[-1]
        assert message == "Select a tool first."
        assert severity == "warning"


@pytest.mark.asyncio
async def test_open_test_for_selected_tool_with_selection_opens_panel():
    """T8: with a tool already selected in the inspector (Tools mode,
    row already clicked), `open_test_for_selected_tool()` opens the SAME
    Test Tool panel the button's own press handler mounts
    (`MCPInspector._mount_test_tool_panel()`, reused via
    `MCPInspector.open_test_panel()`) -- not a second, duplicate mount path.
    """
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 1)  # docs::search (form schema)

        await workbench.open_test_for_selected_tool()
        await pilot.pause()

        assert workbench.active_mode == "tools"
        panel = app.query_one("#mcp-inspector-test-panel", Vertical)
        assert panel.display is not False
        assert app.query_one("#mcp-inspector-test-tool", Button).disabled is True
        # The mounted panel carries the selected tool's schema-driven form,
        # not a blank/duplicate one.
        assert app.query_one("#mcp-inspector-test-form")


@pytest.mark.asyncio
async def test_open_test_for_selected_tool_does_not_duplicate_already_open_panel():
    """T8: pressing `t` a second time while the panel is already open (same
    tool still selected) must not raise `DuplicateIds` -- relies on
    `_mount_test_tool_panel()`'s own existence-check guard, exercised here
    through the keybinding's entry point rather than the button.
    """
    app = ToolTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # docs::fetch

        await workbench.open_test_for_selected_tool()
        await pilot.pause()
        await workbench.open_test_for_selected_tool()
        await pilot.pause()

        assert len(list(app.query("#mcp-inspector-test-panel"))) == 1


@pytest.mark.asyncio
async def test_open_test_for_selected_tool_with_non_executable_selection_notifies_phase_note():
    """T8 regression: `MCPInspector.open_test_panel()` used to return the
    same `False` for BOTH "nothing selected" and "a tool IS selected but
    isn't executable yet" (server-source, Phase 4), so
    `open_test_for_selected_tool()` notified "Select a tool first." for
    both -- misleading when a tool is in fact selected. With a
    server-source tool selected (never executable -- see
    `server_tools_from_inventory`), the `t` keybinding must notify with
    the SAME copy the inline detail view already shows for that tool
    (`mcp_inspector.py`'s "Testing server-source tools arrives in Phase
    4." `Static`), not the generic no-selection message, and must not
    mount a test panel (there is no schema-driven form to open for a
    tool that can't be invoked).
    """
    app = ServerToolsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        notifications = _capture_notifications(app)
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("tools")
        await pilot.pause()
        await _select_tools_mode_row(app, pilot, 0)  # server:main/docs::search
        assert app.query_one("#mcp-inspector-tool-phase-note", Static)

        await workbench.open_test_for_selected_tool()
        await pilot.pause()

        assert workbench.active_mode == "tools"
        assert not list(app.query("#mcp-inspector-test-panel"))
        assert notifications
        message, severity = notifications[-1]
        assert message == "Testing server-source tools arrives in Phase 4."
        assert severity == "information"
