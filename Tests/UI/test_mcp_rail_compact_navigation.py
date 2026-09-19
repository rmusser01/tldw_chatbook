"""MCP rail navigation fits actual compact panes and retains target identity."""

from dataclasses import replace

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_mcp_compact_readability import _compact, _open, _paint, _settle
from Tests.UI.test_mcp_permissions_handoff import _visible
from Tests.UI.test_mcp_rail import RailApp, _snap
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_all_servers_is_fully_painted_in_the_real_compact_rail(
    request, tmp_path, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        for size in [(80, 24), (100, 30), (120, 40), (170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            await _settle(pilot)
            row = workbench.query_one("#mcp-rail-row-0", Button)
            row.focus()
            await _settle(pilot)
            _visible(app, row)
            assert "Allservers" in _compact(_paint(app.screen, row.content_region))
            await pilot.press("enter")
            await _settle(pilot)
            assert workbench._selected_server_key is None


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_long_rail_rows_fit_and_keep_exact_targets_through_resize_and_refresh(
    request, tmp_path, theme
):
    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        workbench = await _open(pilot, tmp_path / "permissions.json")
        snapshots = [
            replace(_snap(f"local:server-{i}", name), tool_count=count)
            for i, (name, count) in enumerate(
                [
                    ("漢字資料庫と文書", 1234),
                    ("[bold]literal[/bold]", 7),
                    ("e\u0301quipe", 81),
                    ("Documents", 999),
                    *[(f"Research server {i}", i) for i in range(12)],
                ]
            )
        ]
        workbench._snapshots = snapshots
        await workbench._sync_children()
        await _settle(pilot)
        rail = workbench.query_one(MCPRail)
        for size in [(80, 24), (100, 30), (170, 48), (80, 24)]:
            prior = app.focused
            await pilot.resize_terminal(*size)
            await _settle(pilot)
            if prior in rail._row_targets:
                assert app.focused is prior
                _visible(app, prior)
            assert rail._row_budget == rail._label_budget()
            for index in (1, 2, 3, 4, len(snapshots)):
                row = rail.query_one(f"#mcp-rail-row-{index}", Button)
                row.focus()
                await _settle(pilot)
                _visible(app, row)
                paint = _compact(_paint(app.screen, row.content_region))
                label = _compact(str(row.label))
                assert label in paint, (label, paint, row.region, row.content_region)
                assert str(snapshots[index - 1].tool_count) in paint
                await pilot.press("enter")
                await _settle(pilot)
                assert workbench._selected_server_key == snapshots[index - 1].server_key
                assert app.focused is row
                _visible(app, row)
            selected = workbench._selected_server_key
            await workbench._sync_children()
            await _settle(pilot)
            assert workbench._selected_server_key == selected
            active = rail.query_one("Button.mcp-rail-row.is-active", Button)
            assert app.focused is active
            _visible(app, active)
            assert await pilot.click(active)
            await _settle(pilot)
            assert workbench._selected_server_key == selected
        # Same outer size, different scrollbars: refit to the final viewport.
        await pilot.resize_terminal(170, 48)
        await _settle(pilot)
        outside = app.screen.query_one("#mcp-mode-tools", Button)
        outside.focus()
        await _settle(pilot)
        widths = []
        many = [replace(snapshots[0], server_key=f"local:large-{i}") for i in range(64)]
        for catalog, scrolling in [
            (snapshots[:1], False),
            (many, True),
            (snapshots[:1], False),
        ]:
            workbench._snapshots = catalog
            await workbench._sync_children()
            await _settle(pilot)
            assert app.focused is outside
            assert rail.show_vertical_scrollbar is scrolling
            assert rail._row_budget == rail._label_budget()
            widths.append(rail.scrollable_content_region.width)
        assert widths[0] == widths[2] > widths[1]


@private_profile_test
async def test_catalog_refresh_keeps_the_focused_server_control(request):
    app = RailApp()
    async with app.run_test() as pilot:
        rail = app.query_one(MCPRail)
        row = rail.query_one("#mcp-rail-row-2", Button)
        row.focus()
        await _settle(pilot)
        rail.sync_state(
            source=rail.source,
            snapshots=[replace(snap, tool_count=42) for snap in rail.snapshots],
            selected_server_key="local:docs",
            scope_options=rail.scope_options,
            scope_value=rail.scope_value,
            scope_ref_options=rail.scope_ref_options,
            scope_ref_value=rail.scope_ref_value,
        )
        await _settle(pilot)
        assert app.focused is row
        assert row.has_class("is-active")
        assert "42" in str(row.label)
        await pilot.press("enter")
        await _settle(pilot)
        assert [
            e.server_key for e in app.events if isinstance(e, MCPRail.ServerSelected)
        ] == ["local:docs"]


@private_profile_test
async def test_a_delayed_rail_press_cannot_select_a_replacement_row(request):
    app = RailApp()
    async with app.run_test() as pilot:
        rail = app.query_one(MCPRail)
        old_row = rail.query_one("#mcp-rail-row-2", Button)
        rail.sync_state(
            source="local",
            snapshots=[_snap("local:new-a", "A"), _snap("local:new-b", "B")],
            selected_server_key=None,
            scope_options=[("Personal", "personal")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
        )
        await _settle(pilot)
        app.events.clear()
        rail.post_message(Button.Pressed(old_row))
        await _settle(pilot)
        assert [e for e in app.events if isinstance(e, MCPRail.ServerSelected)] == []
        await pilot.click("#mcp-rail-row-2")
        await _settle(pilot)
        selected = [
            e.server_key for e in app.events if isinstance(e, MCPRail.ServerSelected)
        ]
        assert selected == ["local:new-b"]
