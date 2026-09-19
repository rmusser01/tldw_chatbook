"""Real Settings routing keeps the permission matrix and its controls visible."""

import pytest
from textual.widgets import DataTable, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_workbench import (
    PermissionsHubService,
    _imported_tool_policy_profile,
)
from Tests.UI.test_settings_tool_profiles import (
    ToolProfileListing,
    _profile,
    _WorkflowService,
)
from Tests.UI.test_tool_profile_review_lifetime import _activate, _wait
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen


def _visible(app, widget):
    region, clip = app.screen._compositor.visible_widgets[widget]
    assert region.intersection(clip) == region


def _cursor_visible(app, table):
    row = table._get_row_region(table.cursor_row)
    y = table.content_region.y + row.y - int(table.scroll_y)
    clip = app.screen._compositor.visible_widgets[table][1]
    assert clip.y <= y and y + row.height <= clip.bottom


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_edit_routes_to_visible_current_profile_and_keyboard_rows(
    request, tmp_path, theme, size
):
    store_path = tmp_path / "permissions.json"
    store = MCPPermissionStore(store_path)
    payload = store.load()
    payload["profiles"]["research"] = _imported_tool_policy_profile()
    store.save(payload)
    lifecycle = store.load()["profiles"]["research"]["tool_pack_lifecycle"]
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        await _wait(pilot, lambda: getattr(app, "_initial_screen_pushed", False))
        app.unified_mcp_service = PermissionsHubService(store_path)
        app.tool_pack_service = _WorkflowService(
            ToolProfileListing(
                profiles=(
                    _profile(
                        "research",
                        origin="imported",
                        binding_state="unbound",
                        revision=lifecycle["revision"],
                        policy_digest=lifecycle["policy_digest"],
                    ),
                )
            )
        )
        before = store_path.read_bytes()
        for warm in (False, True):
            if warm:
                workbench = app.screen.query_one(MCPWorkbench)
                await workbench._apply_view_state(
                    {"mode": "audit", "source": "server", "selected_server_key": None}
                )
                await pilot.pause()
            await app.handle_screen_navigation(NavigateToScreen("settings"))
            await _wait(
                pilot, lambda: getattr(app.screen, "screen_name", None) == "settings"
            )
            app.screen._select_category("tool-profiles")
            await _wait(
                pilot,
                lambda: (
                    bool(app.screen.query("#tool-profile-edit-0"))
                    and not app.screen._category_pane_swap_pending
                ),
            )
            await _activate(pilot, "#tool-profile-edit-0")
            await _wait(pilot, lambda: isinstance(app.screen, MCPScreen))
            await _wait(
                pilot,
                lambda: (
                    app.screen.workbench is not None
                    and not app.screen.workbench.is_loading
                    and not app.screen.workbench._reloading
                    and app.screen.workbench._pending_view_state is None
                ),
            )
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            workbench = app.screen.query_one(MCPWorkbench)
            table = workbench.query_one("#mcp-perm-table", DataTable)
            selector = workbench.query_one("#mcp-perm-tool-profile", Select)
            assert workbench.active_mode == "permissions"
            assert selector.value == "research"
            assert (
                workbench._tool_policy_profile_context.revision == lifecycle["revision"]
            )
            assert app.focused is table
            _visible(app, table)
            for key, expected in (("ctrl+end", table.row_count - 1), ("ctrl+home", 0)):
                await pilot.press(key)
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
                assert table.cursor_row == expected
                _cursor_visible(app, table)
            field = workbench.query_one("#mcp-perm-filter-text", Input)
            field.focus()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            _visible(app, field)
            tool_name = next(
                row.tool_name
                for row in workbench.query_one(MCPPermissionsMode)._all_rows
                if row.kind == "tool"
            )
            await pilot.press(*tool_name)
            await pilot.pause()
            table.focus()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            _visible(app, table)
            _cursor_visible(app, table)
            selector.focus()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            _visible(app, selector)
            canvas = workbench.query_one(MCPPermissionsMode)
            canvas.focus()
            await pilot.pause()
            await pilot.press("end", "space")
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            assert app.focused is canvas
            # Reading the scrollable canvas must not cycle an offscreen row.
            assert store_path.read_bytes() == before
            await pilot.press("home")
            await pilot.wait_for_scheduled_animations()
            preview = canvas.query_one("#mcp-perm-preview")
            for _ in range(120):
                pair = app.screen._compositor.visible_widgets.get(preview)
                if pair and pair[0].intersection(pair[1]) == pair[0]:
                    break
                await pilot.press("down")
                await pilot.wait_for_scheduled_animations()
            _visible(app, preview)
            assert store_path.read_bytes() == before
