"""Queued server controls must lose authority when their view becomes unavailable."""

import asyncio

import pytest
from textual.screen import Screen
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_servers_mode import CanvasApp, _snap
from Tests.UI.test_mcp_workbench import ProfileFormApp
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id", ["delete", "delete-confirm", "disconnect"])
@private_profile_test
async def test_queued_server_action_stays_retired_after_mode_round_trip(
    request, monkeypatch, button_id
):
    """Returning to Servers cannot revive a press queued before departure."""
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
            workbench.set_mode("servers")
            await pilot.pause()
            assert canvas.display and button.is_attached
            post(pending[0])
            await pilot.pause()
            calls = (
                list(app.unified_mcp_service.delete_calls),
                list(app.unified_mcp_service.disconnect_calls),
            )
            if button_id == "delete":
                assert not canvas._delete_armed, "retired Delete armed a confirmation"
        await app.workers.wait_for_complete()
        assert calls == ([], []), (
            "a retired press dispatched after returning to Servers"
        )
        assert not canvas._delete_armed
        assert not canvas.query("#mcp-detail-delete-confirm")
        # Departure must also leave fresh controls usable once presentation settles.
        current = canvas.query_one("#mcp-detail-delete", Button)
        assert current is not button
        current.press()
        await pilot.pause()
        assert canvas._delete_armed
        canvas.query_one("#mcp-detail-delete-cancel", Button).press()
        await pilot.pause()
        assert not canvas._delete_armed
        assert app.unified_mcp_service.delete_calls == []


@pytest.mark.asyncio
@private_profile_test
async def test_accepted_delete_survives_an_overlapping_departure_toolbar_refresh(
    request, monkeypatch
):
    """Departure can replace a pruned toolbar without cancelling accepted deletion."""
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench._select_server_key("local:docs")
        canvas = app.query_one(MCPServersMode)
        canvas.query_one("#mcp-detail-delete", Button).press()
        await pilot.pause()
        toolbar = canvas.query_one("#mcp-detail-toolbar")
        remove = toolbar.remove_children
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def held_remove(*args, **kwargs):
            nonlocal calls
            calls += 1
            first = calls == 1
            await remove(*args, **kwargs)
            if first:
                entered.set()
                await release.wait()

        monkeypatch.setattr(toolbar, "remove_children", held_remove)
        canvas.query_one("#mcp-detail-delete-confirm", Button).press()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            workbench.set_mode("tools")
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            assert calls == 2
            assert len(toolbar.query("#mcp-detail-delete")) == 1
        finally:
            release.set()
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert app.unified_mcp_service.delete_calls == ["docs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id", ["connect", "delete-confirm"])
@pytest.mark.parametrize("state", ["disabled", "ancestor_disabled", "covered"])
@private_profile_test
async def test_queued_server_action_cannot_use_an_unavailable_view(
    request, monkeypatch, button_id, state
):
    """A real press queued while enabled cannot bypass later view admission."""
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(_snap("local:alpha", "Alpha"))
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
        if state == "disabled":
            button.disabled = True
        elif state == "ancestor_disabled":
            canvas.disabled = True
        else:
            await app.push_screen(Screen())
        post(pending[0])
        await pilot.pause()
        assert app.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["before_release", "after_release"])
@private_profile_test
async def test_departure_refresh_preserves_actions_from_a_newer_sync(
    request, monkeypatch, delivery
):
    """A delayed departure refresh must not retire a newly published Delete press."""
    app = ProfileFormApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        await workbench._select_server_key("local:docs")
        canvas = app.query_one(MCPServersMode)
        entered, publish = asyncio.Event(), asyncio.Event()
        published, release = asyncio.Event(), asyncio.Event()
        show = workbench._show_selected_detail

        async def held_detail(*args, **kwargs):
            entered.set()
            await publish.wait()
            await show(*args, **kwargs)
            published.set()
            await release.wait()

        monkeypatch.setattr(workbench, "_show_selected_detail", held_detail)
        sync = asyncio.create_task(workbench._sync_children())
        pending = []
        post = canvas.post_message
        try:
            await asyncio.wait_for(entered.wait(), 3)
            workbench.set_mode("tools")
            workbench.set_mode("servers")
            publish.set()
            await asyncio.wait_for(published.wait(), 3)
            current = canvas.query_one("#mcp-detail-delete", Button)

            def hold(message):
                if isinstance(message, Button.Pressed) and message.button is current:
                    pending.append(message)
                    return True
                return post(message)

            if delivery == "after_release":
                monkeypatch.setattr(canvas, "post_message", hold)
            current.press()
            await pilot.pause()
            if delivery == "before_release":
                assert canvas._delete_armed
            else:
                assert len(pending) == 1
        finally:
            publish.set()
            release.set()
            await asyncio.wait_for(sync, 5)
        await app.workers.wait_for_complete()
        if delivery == "after_release":
            post(pending[0])
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
        assert canvas._delete_armed, "departure refresh discarded newer delete consent"
        assert app.focused is canvas.query_one("#mcp-detail-delete-cancel", Button)
        assert app.unified_mcp_service.delete_calls == []
