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
        bad = _valid_profile_context()
        bad["profile_policy_digest"] = "zz"  # not 64 hex chars
        settings.apply_navigation_context(bad)
        await pilot.pause()
        assert settings.active_category != SettingsCategoryId.TOOLS.value
