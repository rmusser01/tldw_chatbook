"""Permission jumps belong to the tool/profile displayed by their controls."""

from __future__ import annotations

import asyncio

import pytest
from textual.screen import Screen
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_inspector import InspectorApp, _test_preview, _tool
from Tests.UI.test_mcp_session_revoke_ownership import CONTEXT, held_press
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import PermissionProfileContext

BUTTONS = ["mcp-inspector-goto-permission", "mcp-inspector-goto-permission-test"]


async def show(inspector, tool=None, context=CONTEXT):
    tool = tool or _tool(server_key="local:docs", name="search")
    await inspector.show_tool(
        tool,
        effective=EffectiveToolState(state="ask", origin="server_default"),
        profile_context=context,
    )
    await inspector._mount_test_tool_panel()
    inspector.show_test_preview(_test_preview(tool, gate="ask"))


def jumps(app):
    return [
        event
        for event in app.events
        if isinstance(event, MCPInspector.ChangeInPermissionsRequested)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id", BUTTONS)
@pytest.mark.parametrize(
    "transition", ["replace", "profile", "clear", "same", "roundtrip"]
)
@private_profile_test
async def test_retired_permission_jump_cannot_target_a_new_view(
    request, monkeypatch, button_id, transition
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        event = held_press(app.query_one(f"#{button_id}", Button), monkeypatch)
        if transition == "clear":
            await inspector.show_tool(None)
        elif transition == "profile":
            await show(
                inspector, context=PermissionProfileContext("other", 8, "c" * 64, 4)
            )
        elif transition in {"replace", "roundtrip"}:
            await show(inspector, _tool(server_key="local:files", name="read"))
            if transition == "roundtrip":
                await show(inspector)
        else:
            await show(inspector)
        inspector.post_message(event)
        await pilot.pause()
        assert jumps(app) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id", BUTTONS)
@private_profile_test
async def test_permission_jump_is_invalid_before_tool_pruning_yields(
    request, monkeypatch, button_id
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        event = held_press(app.query_one(f"#{button_id}", Button), monkeypatch)
        container = app.query_one("#mcp-inspector-tool")
        original = container.remove_children
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold_removal():
            entered.set()
            await release.wait()
            await original()

        monkeypatch.setattr(container, "remove_children", hold_removal)
        refresh = asyncio.create_task(show(inspector, _tool(name="replacement")))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            inspector.post_message(event)
            await pilot.pause()
            assert jumps(app) == []
        finally:
            release.set()
            await refresh


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id", BUTTONS)
@private_profile_test
async def test_current_permission_jump_preserves_identity_and_is_retryable(
    request, button_id
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        tool = _tool(server_key="local:docs", name="search")
        await show(inspector, tool)
        await pilot.pause()
        button = app.query_one(f"#{button_id}", Button)
        button.press()
        await pilot.pause()
        button.press()
        await pilot.pause()
        assert len(jumps(app)) == 2
        assert all(
            (event.server_key, event.tool_name, event.profile_context)
            == ("local:docs", "search", CONTEXT)
            for event in jumps(app)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["hidden", "disabled", "closed"])
@private_profile_test
async def test_unavailable_test_jump_is_ignored_but_tool_details_jump_still_works(
    request, monkeypatch, state
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        button = app.query_one(f"#{BUTTONS[1]}", Button)
        event = held_press(button, monkeypatch)
        if state == "closed":
            await inspector._close_test_tool_panel()
        elif state == "disabled":
            button.disabled = True
        else:
            inspector.show_tool_result(
                server_key="local:docs",
                tool_name="search",
                ok=True,
                text="{}",
                duration_ms=0,
            )
            assert not button.display
        inspector.post_message(event)
        await pilot.pause()
        assert jumps(app) == []
        app.query_one(f"#{BUTTONS[0]}", Button).press()
        await pilot.pause()
        assert len(jumps(app)) == 1
        assert jumps(app)[0].tool_name == "search"


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id", BUTTONS)
@pytest.mark.parametrize("state", ["ancestor_hidden", "ancestor_disabled", "covered"])
@private_profile_test
async def test_permission_jump_ignores_an_unavailable_owning_view(
    request, monkeypatch, button_id, state
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        event = held_press(app.query_one(f"#{button_id}", Button), monkeypatch)
        if state == "ancestor_hidden":
            inspector.display = False
        elif state == "ancestor_disabled":
            inspector.disabled = True
        else:
            await app.push_screen(Screen())
        inspector.post_message(event)
        await pilot.pause()
        assert jumps(app) == []
