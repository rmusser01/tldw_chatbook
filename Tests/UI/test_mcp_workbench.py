# Tests/UI/test_mcp_workbench.py
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ContentSwitcher, Select, Static

from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
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
        # profile fixture is disconnected -- so it derives STALE
        # (RUNTIME_UNAVAILABLE) rather than READY, which is what makes
        # CONNECT a wired, enabled action to click in the first test below.
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
        # select docs (local profile, disconnected -> STALE -> CONNECT wired)
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
