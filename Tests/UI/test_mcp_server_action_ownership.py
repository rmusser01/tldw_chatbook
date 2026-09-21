"""Server toolbar intent must not follow a later selection or redraw."""

import asyncio

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_servers_mode import CanvasApp, _snap
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "button_id",
    [
        "edit",
        "connect",
        "refresh",
        "disconnect",
        "delete",
        "delete-confirm",
        "delete-cancel",
    ],
)
@private_profile_test
async def test_queued_server_action_is_rejected_after_detail_replacement(
    request, monkeypatch, button_id
):
    """Deliver a real press after its originating toolbar has been replaced."""
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(
            _snap("local:alpha", "Alpha", is_connected=button_id != "connect")
        )
        if button_id.startswith("delete-"):
            canvas.query_one("#mcp-detail-delete", Button).press()
            await pilot.pause()
        button = canvas.query_one(f"#mcp-detail-{button_id}", Button)
        pending = []
        post = canvas.post_message

        def hold(message):
            if isinstance(message, Button.Pressed) and message.button is button:
                pending.append(message)
                return True
            return post(message)

        monkeypatch.setattr(canvas, "post_message", hold)
        button.press()
        await pilot.pause()
        assert len(pending) == 1
        await canvas.show_detail(_snap("local:beta", "Beta", is_connected=True))
        if button_id == "delete-cancel":
            canvas.query_one("#mcp-detail-delete", Button).press()
            await pilot.pause()
            assert canvas._delete_armed
        toolbar_before = list(canvas.query("#mcp-detail-toolbar Button"))
        post(pending[0])
        await pilot.pause()
        assert app.events == [], "retired action dispatched against the new server"
        assert list(canvas.query("#mcp-detail-toolbar Button")) == toolbar_before
        assert canvas._delete_armed is (button_id == "delete-cancel")


@pytest.mark.asyncio
@private_profile_test
async def test_old_server_action_is_retired_before_first_detail_refresh_await(
    request,
    monkeypatch,
):
    """The old control can remain mounted while a new detail is rendering."""
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(_snap("local:alpha", "Alpha"))
        button = canvas.query_one("#mcp-detail-connect", Button)
        entered, release = asyncio.Event(), asyncio.Event()
        rebuild = canvas._rebuild_toggle_groups

        async def held_rebuild():
            entered.set()
            await release.wait()
            await rebuild()

        monkeypatch.setattr(canvas, "_rebuild_toggle_groups", held_rebuild)
        task = asyncio.create_task(canvas.show_detail(_snap("local:beta", "Beta")))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert button.is_attached
            button.press()
            await pilot.pause()
            assert app.events == [], "mounted old action used the new detail target"
        finally:
            release.set()
            await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
@private_profile_test
async def test_accepted_delete_keeps_its_target_across_toolbar_rebuild(
    request, monkeypatch
):
    """An accepted confirmation owns Alpha even while Beta becomes selected."""
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(_snap("local:alpha", "Alpha"))
        canvas.query_one("#mcp-detail-delete", Button).press()
        await pilot.pause()
        entered, release = asyncio.Event(), asyncio.Event()
        rebuild = canvas._rebuild_detail_toolbar
        calls = 0

        async def held_rebuild():
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            await rebuild()

        monkeypatch.setattr(canvas, "_rebuild_detail_toolbar", held_rebuild)
        canvas.query_one("#mcp-detail-delete-confirm", Button).press()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await canvas.show_detail(_snap("local:beta", "Beta"))
        finally:
            release.set()
        await pilot.pause()
        confirmed = [e for e in app.events if isinstance(e, canvas.DeleteConfirmed)]
        assert [e.server_key for e in confirmed] == ["local:alpha"]
        assert canvas._detail_snapshot.server_key == "local:beta"


@pytest.mark.asyncio
@private_profile_test
async def test_hidden_server_toolbar_cannot_dispatch_while_edit_form_is_open(request):
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(_snap("local:alpha", "Alpha"))
        button = canvas.query_one("#mcp-detail-connect", Button)
        await canvas.show_form(None)
        button.press()
        await pilot.pause()
        assert app.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id", ["delete", "delete-confirm", "disconnect"])
@private_profile_test
async def test_queued_action_cannot_run_in_hidden_mode_while_disarm_waits(
    request, monkeypatch, button_id
):
    from Tests.UI.test_mcp_workbench import ProfileFormApp
    from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench

    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench._select_server_key("local:docs")
        canvas = app.query_one(MCPServersMode)
        if button_id == "delete-confirm":
            canvas.query_one("#mcp-detail-delete", Button).press()
            await pilot.pause()
        button = canvas.query_one(f"#mcp-detail-{button_id}", Button)
        pending = []
        post = canvas.post_message

        def hold(message):
            if isinstance(message, Button.Pressed) and message.button is button:
                pending.append(message)
                return True
            return post(message)

        monkeypatch.setattr(canvas, "post_message", hold)
        button.press()
        await pilot.pause()
        assert len(pending) == 1
        async with workbench._sync_children_lock:
            workbench.set_mode("tools")
            await pilot.pause()
            assert not canvas.display
            armed = canvas._delete_armed
            post(pending[0])
            await pilot.pause()
            observed = (
                list(app.unified_mcp_service.delete_calls),
                list(app.unified_mcp_service.disconnect_calls),
                canvas._delete_armed,
            )
        await app.workers.wait_for_complete()
        assert observed == ([], [], armed)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_server_toolbar_and_confirmation_are_fully_visible(request, theme, size):
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_mcp_compact_readability import _paint, _settle
    from Tests.UI.test_mcp_workbench import ProfileFormHubService
    from Tests.UI.test_tool_profile_review_lifetime import _wait
    from tldw_chatbook.config import save_setting_to_cli_config
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app("home")
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        await _wait(pilot, lambda: getattr(app, "_initial_screen_pushed", False))
        app.unified_mcp_service = ProfileFormHubService()
        await app.handle_screen_navigation(NavigateToScreen("mcp"))
        await _wait(pilot, lambda: getattr(app.screen, "screen_name", None) == "mcp")
        workbench = app.screen.workbench
        await _wait(
            pilot, lambda: not workbench.is_loading and not workbench._reloading
        )
        await workbench._select_server_key("local:docs")
        canvas = app.query_one(MCPServersMode)
        buttons = list(canvas.query("#mcp-detail-toolbar Button"))
        assert len(buttons) == 4
        buttons[0].focus()
        await _settle(pilot)
        for index, button in enumerate(buttons):
            assert app.focused is button
            region, clip = app.screen._compositor.visible_widgets[button]
            assert region.intersection(clip) == region, button.id
            assert button.label.plain in _paint(app.screen, region)
            if index + 1 < len(buttons):
                await pilot.press("tab")
                await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert app.focused is canvas.query_one("#mcp-detail-delete-cancel", Button)
        for button in canvas.query("#mcp-detail-toolbar Button"):
            region, clip = app.screen._compositor.visible_widgets[button]
            assert region.intersection(clip) == region, button.id
            assert button.label.plain in _paint(app.screen, region)
        await pilot.press("escape")
        await _settle(pilot)
        assert not canvas._delete_armed
        assert app.unified_mcp_service.delete_calls == []
