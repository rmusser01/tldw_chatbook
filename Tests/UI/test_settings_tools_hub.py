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
        settings.action_settings_tools_mode("audit")
        assert called == ["audit"]
        settings.action_settings_test_category()
        assert called == ["audit", "test"]


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
async def test_suspending_settings_invalidates_in_flight_recovery_review(request):
    """ADR-233 port check: leaving Settings mid-review must invalidate an
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
