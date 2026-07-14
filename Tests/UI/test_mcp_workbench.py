# Tests/UI/test_mcp_workbench.py
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, ContentSwitcher, Input, Select, Static, TextArea

from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel, MCPProfileForm
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
async def test_file_requested_pushes_picker_and_loads_selected_file_into_panel(tmp_path):
    """Workbench's FileRequested handler pushes EnhancedFileOpen filtered to
    JSON and, once a file is picked, writes its text into the panel's
    TextArea (Interfaces: "workbench pushes EnhancedFileOpen(...) and writes
    the file's text into the TextArea")."""
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
