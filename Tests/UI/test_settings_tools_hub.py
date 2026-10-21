"""Settings hosts the MCP workbench as the Tools & MCP category."""

import time

import pytest

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_workbench import PermissionsHubService
from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
)
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_tools_hub import ToolsHubPanel


class _PanelHarnessApp(ConsolidatedCSSApp):
    """Mounts one ToolsHubPanel the way DestinationHarness mounts screens."""

    def __init__(self, initial_view_state=None):
        super().__init__()
        self.app_instance = type("AppInstanceStub", (), {"unified_mcp_service": None})()
        self._initial_view_state = initial_view_state

    def compose(self):
        yield ToolsHubPanel(
            self.app_instance,
            initial_view_state=self._initial_view_state,
            id="settings-tools-hub",
        )


def test_tools_category_id_exists_and_mcp_defaults_retired():
    assert SettingsCategoryId.TOOLS.value == "tools"
    with pytest.raises(AttributeError):
        _ = SettingsCategoryId.MCP_DEFAULTS


async def test_panel_mounts_workbench_and_mode_chips():
    host = _PanelHarnessApp()
    async with host.run_test() as pilot:
        panel = host.query_one("#settings-tools-hub", ToolsHubPanel)
        workbench = panel.query_one(MCPWorkbench)
        assert workbench.active_mode == "servers"
        chips = panel.query("#mcp-mode-strip Button")
        assert [c.id for c in chips] == [
            "mcp-mode-servers",
            "mcp-mode-tools",
            "mcp-mode-permissions",
            "mcp-mode-audit",
        ]
        await pilot.click("#mcp-mode-permissions")
        assert workbench.active_mode == "permissions"
        assert "is-active" in panel.query_one("#mcp-mode-permissions").classes


async def test_panel_view_state_round_trip():
    host = _PanelHarnessApp(initial_view_state={"mode": "audit"})
    async with host.run_test() as pilot:
        panel = host.query_one("#settings-tools-hub", ToolsHubPanel)
        workbench = panel.query_one(MCPWorkbench)

        # Wait on the chip, not just on `active_mode`. The stashed initial
        # view state is applied at the END of the workbench's initial reload
        # worker (mcp_workbench.py, TASK-1320 moved that load off mount), so
        # asserting the mode right after run_test() entry races the worker.
        # The chip highlight is synced by this panel's ModeChanged handler,
        # one pump cycle after the mode itself -- polling it and then
        # asserting everything follows the poll-then-assert precedent in
        # test_destination_shells.py
        # (`test_mcp_destination_mode_chip_syncs_to_restored_mode`). The end
        # state asserted is unchanged: this still fails
        # if the restore never applies or the chip never syncs.
        audit_chip = panel.query_one("#mcp-mode-audit")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not audit_chip.has_class("is-active"):
            await pilot.pause(0.01)

        assert workbench.active_mode == "audit"
        assert audit_chip.has_class("is-active")
        state = panel.get_view_state()
        assert isinstance(state, dict)
        assert state["mode"] == "audit"


@pytest.mark.asyncio
async def test_settings_registers_tools_category_and_mounts_pane(request):
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        summaries = {s.category: s.title for s in settings._category_summaries()}
        assert summaries.get(SettingsCategoryId.TOOLS) == "Tools & MCP"
        assert SettingsCategoryId.MCP_DEFAULTS not in summaries
        await _open_settings_category(pilot, "#settings-category-tools")
        await pilot.pause()

        assert settings.active_category == SettingsCategoryId.TOOLS.value
        assert settings.query_one("#settings-tools-hub", ToolsHubPanel)
        assert settings.query_one(MCPWorkbench)


@pytest.mark.asyncio
async def test_mode_keys_and_verb_keys_dispatch_only_in_tools_category():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        # Outside TOOLS: the a/r/t actions keep their existing no-op/toast
        # behavior and never touch a pane (none mounted).
        settings.action_settings_test_category()  # must not raise
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        called: list[str] = []
        pane.activate_mode = lambda mode: called.append(mode)  # type: ignore[assignment]
        pane.test_selected_tool = lambda: called.append("test")  # type: ignore[assignment]
        # The 1-4 keys are pane-held BINDINGS only (ADR-240 ratchet offset;
        # rail-focus digits stay the sanctioned edge). a/r/t are pane-held
        # first, with screen-level TOOLS branches behind them so the
        # advertised keys also work with focus OUTSIDE the pane (pinned in
        # the verb-dispatch proof test below).
        pane.action_tools_mode("audit")
        assert called == ["audit"]
        pane.action_tools_test_tool()
        assert called == ["audit", "test"]


@pytest.mark.asyncio
@private_profile_test
async def test_screen_verb_actions_reach_pane_with_focus_outside(request, tmp_path):
    """PR #3140 pre-merge review, finding 1: with the TOOLS category active
    but focus OUTSIDE the pane (e.g. on the category rail), the screen-level
    s/r/t actions must still reach the hub pane -- `t` must never fall
    through to the "No test action is available" stub toast. This is the
    proof the footer-hints drift guard
    (test_settings_footer_hints.py::test_advertised_capabilities_match_real_
    action_branches) asserts for TOOLS; that file is locally env-blocked
    (RecoveryRequired), so the proof is pinned here on the private-profile
    runner instead."""
    app = _build_test_app()
    # A real (fake-seam) hub service: the workbench's initial load worker
    # reads app_instance.unified_mcp_service, and the real service raises
    # RecoveryRequired in this sandbox (pre-existing env block, see
    # baseline-failures.txt) -- the handoff test stubs the same seam.
    app.unified_mcp_service = PermissionsHubService(
        tmp_path / "mcp_permissions.json"
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        toasts: list[str] = []
        host.notify = lambda message, **kwargs: toasts.append(message)  # type: ignore[assignment]
        called: list[str] = []
        pane.test_selected_tool = lambda: called.append("test")  # type: ignore[assignment]
        pane.refresh_now = lambda: called.append("refresh")  # type: ignore[assignment]
        pane.add_server = lambda: called.append("add")  # type: ignore[assignment]
        # Direct (stub-free) calls stand in for keypresses arriving while
        # focus sits outside the pane -- the pane-held bindings do not fire
        # from there, so the screen-level TOOLS branches must.
        settings.action_settings_test_category()
        settings.action_settings_revert_category()
        settings.action_settings_rag_set_active()
        assert called == ["test", "refresh", "add"], called
        assert not any("No test action is available" in t for t in toasts), toasts


@pytest.mark.asyncio
async def test_footer_shortcuts_follow_active_mode():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        base = settings._category_footer_shortcuts(SettingsCategoryId.TOOLS)
        assert ("1-4", "mode") in base
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        pane.activate_mode("tools")
        # The tools canvas mounts deferred (task-2901): a mode request that
        # lands in the pre-mount window is stashed and replayed by the
        # initial load worker, so poll for the mode instead of assuming one
        # pause settled it (same precedent as
        # test_panel_view_state_round_trip above).
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and pane.active_mode != "tools":
            await pilot.pause(0.01)
        in_tools_mode = settings._category_footer_shortcuts(SettingsCategoryId.TOOLS)
        assert ("t", "test tool") in in_tools_mode


def _valid_profile_context() -> dict[str, object]:
    return {
        "category": "tools",
        "mode": "permissions",
        "tool_policy_profile_id": "research",
        "profile_revision": 3,
        "profile_policy_digest": "a" * 64,
    }


def test_parse_tools_navigation_context_accepts_mode_and_profile_links():
    from tldw_chatbook.UI.Screens.settings_tools_hub import (
        parse_tools_navigation_context,
    )

    assert parse_tools_navigation_context({}) == {}
    assert parse_tools_navigation_context({"category": "tools"}) == {}
    assert parse_tools_navigation_context({"category": "tools", "mode": "audit"}) == {
        "mode": "audit"
    }
    assert parse_tools_navigation_context(_valid_profile_context()) == {
        "mode": "permissions",
        "tool_policy_profile_id": "research",
        "profile_revision": 3,
        "profile_policy_digest": "a" * 64,
    }


def test_parse_tools_navigation_context_rejects_unsafe_or_malformed_payloads():
    from tldw_chatbook.UI.Screens.settings_tools_hub import (
        parse_tools_navigation_context,
    )

    # Retired MCP-destination bounds, re-pinned on the extracted helper.
    assert parse_tools_navigation_context({"extra": True}) is None
    assert parse_tools_navigation_context({"mode": "nope"}) is None
    bad_digest = _valid_profile_context()
    bad_digest["profile_policy_digest"] = "zz"
    assert parse_tools_navigation_context(bad_digest) is None
    missing_digest = _valid_profile_context()
    missing_digest["profile_policy_digest"] = None
    assert parse_tools_navigation_context(missing_digest) is None
    long_id = _valid_profile_context()
    long_id["tool_policy_profile_id"] = "x" * 129
    assert parse_tools_navigation_context(long_id) is None
    bad_revision = _valid_profile_context()
    bad_revision["profile_revision"] = 0
    assert parse_tools_navigation_context(bad_revision) is None


@pytest.mark.asyncio
@private_profile_test
async def test_apply_navigation_context_accepts_bounded_tools_context(request, tmp_path):
    app = _build_test_app()
    # A real (fake-seam) hub service: the workbench's initial load worker
    # reads app_instance.unified_mcp_service, and the real service raises
    # RecoveryRequired in this sandbox (pre-existing env block, see
    # baseline-failures.txt) -- the handoff test stubs the same seam.
    app.unified_mcp_service = PermissionsHubService(
        tmp_path / "mcp_permissions.json"
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        settings.apply_navigation_context(_valid_profile_context())
        await pilot.pause()
        assert settings.active_category == SettingsCategoryId.TOOLS.value
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        # The stashed initial view state is applied at the END of the
        # workbench's initial reload worker (see
        # test_panel_view_state_round_trip above) -- poll for the mode
        # instead of assuming one pause settled it.
        deadline = time.monotonic() + 2.0
        while (
            time.monotonic() < deadline
            and pane.workbench.active_mode != "permissions"
        ):
            await pilot.pause(0.01)
        assert pane.workbench.active_mode == "permissions"


@pytest.mark.asyncio
@private_profile_test
async def test_apply_navigation_context_rejects_malformed_tools_context(request):
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        settings._select_category(SettingsCategoryId.APPEARANCE.value)
        await pilot.pause()
        # Retired MCP-destination bounds, re-pinned on the Settings TOOLS branch:
        # a bad digest, an unexpected extra key, and a profile id without
        # its digest triple are all rejected (category stays where it was).
        bad_digest = _valid_profile_context()
        bad_digest["profile_policy_digest"] = "zz"  # not 64 hex chars
        settings.apply_navigation_context(bad_digest)
        await pilot.pause()
        assert settings.active_category != SettingsCategoryId.TOOLS.value

        extra_key = _valid_profile_context()
        extra_key["extra"] = True
        settings.apply_navigation_context(extra_key)
        await pilot.pause()
        assert settings.active_category != SettingsCategoryId.TOOLS.value

        missing_digest = _valid_profile_context()
        missing_digest["tool_policy_profile_id"] = "digest-missing"
        missing_digest["profile_revision"] = None
        missing_digest["profile_policy_digest"] = None
        settings.apply_navigation_context(missing_digest)
        await pilot.pause()
        assert settings.active_category != SettingsCategoryId.TOOLS.value


@pytest.mark.asyncio
@private_profile_test
async def test_tools_triad_fits_settings_detail_pane_at_120_columns(request, tmp_path):
    """task-35151 AC: Tools & MCP is usable at 120 terminal columns.

    The detail pane gives the workbench ~56 columns there -- below the
    compact triad's minimum rail+canvas+inspector width (66), where
    Textual's fr resolution degenerates and every pane takes fraction x
    pane width (measured: rail 112, canvas 392, inspector 112 -- the
    canvas fully offscreen). ADR-148's stacked triad (inspector band
    under a rail+canvas row, keyed on the workbench's OWN width, not the
    terminal's) must take over and keep every pane inside the pane."""
    app = _build_test_app()
    # A real (fake-seam) hub service: the workbench's initial load worker
    # reads app_instance.unified_mcp_service, and the real service raises
    # RecoveryRequired in this sandbox (pre-existing env block, see
    # baseline-failures.txt) -- same seam stub as the tests above.
    app.unified_mcp_service = PermissionsHubService(tmp_path / "mcp_perms.json")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 30)) as pilot:
        settings = _active_destination_screen(host)
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()

        grid = settings.query_one("#mcp-hub-grid")
        assert grid.has_class("mcp-compact"), "56-col pane must be compact"
        assert grid.has_class("mcp-stacked"), (
            "below the compact triad minimums (66 cols) the ADR-148 stacked "
            "layout must take over"
        )
        compositor = host.screen._compositor
        for pane_id in ("#mcp-hub-rail", "#mcp-hub-canvas", "#mcp-hub-inspector"):
            pane = settings.query_one(pane_id)
            assert pane.outer_size.width > 0, f"{pane_id} must render"
            region, clip = compositor.visible_widgets[pane]
            assert region.intersection(clip) == region, (
                f"{pane_id} escapes the Settings detail pane at 120 columns "
                f"(region={region}, clip={clip})"
            )
        # The stacked contract: rail+canvas share one row; the inspector is
        # the bounded band BELOW it (ADR-148 decision 2).
        rail = settings.query_one("#mcp-hub-rail")
        canvas = settings.query_one("#mcp-hub-canvas")
        inspector = settings.query_one("#mcp-hub-inspector")
        assert rail.region.y == canvas.region.y
        assert inspector.region.y > rail.region.y
        assert inspector.region.width >= rail.region.width


@pytest.mark.asyncio
@private_profile_test
async def test_suspending_settings_invalidates_in_flight_recovery_review(request):
    """ADR-240 port check: leaving Settings mid-review must invalidate an
    accepted recovery review's UI receipt AND clear the TOOLS footer hints --
    both halves of the retired MCP shell's on_screen_suspend, driven through
    the REAL SettingsScreen handler (not an inlined copy)."""
    from textual.screen import Screen as TextualScreen

    from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)

        # Busy guard: an idle workbench keeps its (synthetic) receipt -- the
        # retired screen only invalidated an ACCEPTED review.
        idle_token = object()
        pane.workbench._mcp_recovery_token = idle_token
        await host.push_screen(TextualScreen())
        await pilot.pause()
        assert pane.workbench._mcp_recovery_token is idle_token
        # First half of the port: the footer hints cleared under the overlay.
        footer = settings.query_one(AppFooterStatus)
        assert footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT
        await host.pop_screen()
        await pilot.pause()

        # Second half: an accepted (busy) review is invalidated by leaving.
        busy_token = object()
        pane.workbench._mcp_recovery_busy = True
        pane.workbench._mcp_recovery_token = busy_token
        await host.push_screen(TextualScreen())
        await pilot.pause()
        assert pane.workbench._mcp_recovery_token is None
        pane.workbench._mcp_recovery_busy = False


@pytest.mark.asyncio
@private_profile_test
async def test_permission_toggle_keeps_tools_category_undrafted(request, tmp_path):
    """Spec amendment A3: the workbench writes permission changes immediately,
    so the Settings draft machinery must stay empty for TOOLS -- no phantom
    "unsaved changes" marker and no save/revert keys that would lie about
    staged edits that do not exist."""
    from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode

    app = _build_test_app()
    app.unified_mcp_service = PermissionsHubService(
        tmp_path / "mcp_permissions.json"
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        pane.activate_mode("permissions")
        deadline = time.monotonic() + 2.0
        while (
            time.monotonic() < deadline
            and pane.workbench.active_mode != "permissions"
        ):
            await pilot.pause(0.01)
        assert pane.workbench.active_mode == "permissions"
        # The Space-cycle handler needs the captured tool-policy context the
        # permissions sync builds (same step the recovery journeys drive).
        await pane.workbench._sync_children()
        await pilot.pause()
        assert pane.workbench._tool_policy_profile_context is not None

        await pane.workbench.on_mcp_permissions_mode_state_cycle_requested(
            MCPPermissionsMode.StateCycleRequested(
                row_kind="global",
                server_key="",
                tool_name=None,
                # Store vocabulary is allow/ask/deny (the UI's "off" rung).
                new_state="deny",
                # The real canvas sends the workbench's captured context with
                # every Space-cycle; the handler validates the EVENT's copy.
                profile_context=pane.workbench._tool_policy_profile_context,
            )
        )
        await host.workers.wait_for_complete()
        await pilot.pause()

        # The toggle took effect on the store (immediate write)...
        assert (
            app.unified_mcp_service.permission_store.get_global_default() == "deny"
        )
        # ...and the category still reports NO staged Settings draft.
        assert not settings._category_has_unsaved_changes(SettingsCategoryId.TOOLS)
        assert SettingsCategoryId.TOOLS not in settings._settings_drafts


@pytest.mark.asyncio
@private_profile_test
async def test_runtime_backend_change_forces_pane_reload(request, tmp_path):
    app = _build_test_app()
    # A real (fake-seam) hub service: the workbench's initial load worker
    # reads app_instance.unified_mcp_service, and the real service raises
    # RecoveryRequired in this sandbox (pre-existing env block, see
    # baseline-failures.txt) -- the handoff test stubs the same seam.
    app.unified_mcp_service = PermissionsHubService(
        tmp_path / "mcp_permissions.json"
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        # Pre-mount (no pane in another category): the forward must no-op,
        # not raise.
        await settings.handle_runtime_backend_changed("local")
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        reloaded: list[bool] = []
        pane.handle_runtime_backend_change = lambda backend: reloaded.append(True)  # type: ignore[assignment]
        await settings.handle_runtime_backend_changed("local")
        assert reloaded == [True]


@pytest.mark.asyncio
@private_profile_test
async def test_category_switch_away_and_back_restores_hub_view_state(
    request, tmp_path
):
    """Final-review fix (task-35150): an intra-Settings category switch swaps
    panes and DESTROYS the hub pane -- the leave path must snapshot its live
    view state so switching back remounts the workbench in the mode the user
    left, instead of resetting to Servers."""
    app = _build_test_app()
    # A real (fake-seam) hub service: the restore is applied at the END of
    # the workbench's initial reload worker, which reads
    # app_instance.unified_mcp_service -- the real service raises
    # RecoveryRequired in this sandbox (pre-existing env block, see
    # baseline-failures.txt), same seam stub as the tests above.
    app.unified_mcp_service = PermissionsHubService(
        tmp_path / "mcp_permissions.json"
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        settings = _active_destination_screen(host)
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        pane.activate_mode("audit")
        # A mode request landing before the deferred audit canvas mounts is
        # stashed and replayed by the initial load worker -- poll for the
        # mode instead of assuming one pause settled it (same precedent as
        # test_footer_shortcuts_follow_active_mode above).
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and pane.workbench.active_mode != "audit":
            await pilot.pause(0.01)
        assert pane.workbench.active_mode == "audit"

        # Leave TOOLS: the pane swap destroys the pane, so the leave path
        # must have captured its live view state first.
        settings._select_category(SettingsCategoryId.APPEARANCE.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        captured = settings._tools_hub_initial_view_state
        assert isinstance(captured, dict)
        assert captured.get("mode") == "audit"

        # Come back: the fresh pane consumes the captured state at the end
        # of its initial reload worker (see test_panel_view_state_round_trip
        # above) -- poll for the synced chip, then assert the full state.
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        remounted = settings.query_one("#settings-tools-hub", ToolsHubPanel)
        audit_chip = remounted.query_one("#mcp-mode-audit")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not audit_chip.has_class("is-active"):
            await pilot.pause(0.01)
        assert remounted.workbench.active_mode == "audit"
        assert audit_chip.has_class("is-active")
        assert remounted.get_view_state()["mode"] == "audit"


@pytest.mark.asyncio
@private_profile_test
async def test_tools_category_gate_rows_carry_the_shared_effect_scope_suffix(
    request, tmp_path
):
    """task-35153: the Settings ▸ Tools & MCP category's Servers ▸ built-in
    detail renders the per-gate effect-scope suffixes from the shared
    `ToolGate.effect_scope` field -- the same field the hub pane and the
    first-run wizard render -- with `web_deep_search` the one restart gate.
    """
    from tldw_chatbook.Agents.local_tool_provider import WEB_DEEP_SEARCH_GATE_KEY

    app = _build_test_app()
    # A real (fake-seam) hub service: the workbench's initial load worker
    # reads app_instance.unified_mcp_service, and the real service raises
    # RecoveryRequired in this sandbox (pre-existing env block, see
    # baseline-failures.txt) -- the handoff test stubs the same seam.
    app.unified_mcp_service = PermissionsHubService(
        tmp_path / "mcp_permissions.json"
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(150, 45)) as pilot:
        settings = _active_destination_screen(host)
        settings._select_category(SettingsCategoryId.TOOLS.value)
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = settings.query_one("#settings-tools-hub", ToolsHubPanel)

        # Wait for the initial snapshot collection, then open the built-in
        # server's detail (the only source whose pane carries gate rows).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not pane.workbench._snapshots:
            await pilot.pause(0.01)
        await pane.workbench._select_server_key("builtin:tldw_chatbook")
        await pilot.pause(0.2)

        labels = [
            str(button.label)
            for button in pane.query("#mcp-detail-tool-gates Button")
        ]
        assert labels, "the built-in detail must render one Button per gate"
        assert any("(next chat)" in label for label in labels), labels
        deep = [
            label
            for label in labels
            if label.startswith("Deep web research (web_deep_search)")
        ]
        assert deep and "(restart to apply)" in deep[0], labels
