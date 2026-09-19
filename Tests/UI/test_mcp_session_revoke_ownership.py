"""Revoke gestures must never be retargeted by a refreshed grant listing."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_inspector import InspectorApp, _tool
from Tests.UI.test_mcp_workbench import PermissionsApp
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import PermissionProfileContext
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench

CONTEXT = PermissionProfileContext("research", 7, "b" * 64, 3)
GRANTS = [("agent:builtin", "calculator"), ("local:docs", "search")]


async def show(inspector, grants=GRANTS, context=CONTEXT):
    await inspector.show_permission(
        _tool(server_key="local:docs", name="search"),
        EffectiveToolState(state="ask", origin="server_default"),
        profile_context=context,
        session_approvals=grants,
    )


def held_press(button, monkeypatch):
    """Hold only the actual public press message, leaving widget events real."""
    messages = []
    original = button.post_message

    def hold(message):
        if isinstance(message, Button.Pressed):
            messages.append(message)
            return True
        return original(message)

    monkeypatch.setattr(button, "post_message", hold)
    button.press()
    assert len(messages) == 1
    return messages[0]


def revocations(app):
    return [
        e
        for e in app.events
        if isinstance(e, MCPInspector.RevokeSessionApprovalRequested)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition", ["reorder", "replace", "profile", "clear", "roundtrip", "same"]
)
@private_profile_test
async def test_queued_revoke_cannot_target_a_replacement_view(
    request, monkeypatch, transition
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        button = app.query_one("#mcp-inspector-session-approval-revoke-0", Button)
        event = held_press(button, monkeypatch)

        if transition == "clear":
            await inspector.show_tool(None)
        elif transition == "profile":
            await show(
                inspector, context=PermissionProfileContext("other", 8, "c" * 64, 4)
            )
        elif transition == "reorder":
            await show(inspector, list(reversed(GRANTS)))
        elif transition == "replace":
            await show(inspector, [("local:files", "read")])
        elif transition == "roundtrip":
            await show(inspector, [("local:files", "read")])
            await show(inspector)
        else:
            await show(inspector)
        await pilot.pause()
        inspector.post_message(event)
        await pilot.pause()
        assert revocations(app) == []


@pytest.mark.asyncio
@private_profile_test
async def test_revoke_is_consumed_once_and_fresh_controls_target_their_own_grant(
    request,
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        button = app.query_one("#mcp-inspector-session-approval-revoke-0", Button)
        button.press()
        button.press()
        await pilot.pause()
        assert len(revocations(app)) == 1
        first = revocations(app)[0]
        assert (first.server_key, first.tool_name, first.profile_context) == (
            "agent:builtin",
            "calculator",
            CONTEXT,
        )
        await show(inspector, [("local:docs", "search")])
        await pilot.pause()
        app.query_one("#mcp-inspector-session-approval-revoke-0", Button).press()
        await pilot.pause()
        assert len(revocations(app)) == 2
        second = revocations(app)[1]
        assert (second.server_key, second.tool_name, second.profile_context) == (
            "local:docs",
            "search",
            CONTEXT,
        )


@pytest.mark.asyncio
@private_profile_test
async def test_revoke_is_invalid_as_soon_as_permission_replacement_starts(
    request, monkeypatch
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        button = app.query_one("#mcp-inspector-session-approval-revoke-0", Button)
        event = held_press(button, monkeypatch)
        container = app.query_one("#mcp-inspector-permission")
        original = container.remove_children
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold_removal():
            entered.set()
            await release.wait()
            await original()

        monkeypatch.setattr(container, "remove_children", hold_removal)
        refresh = asyncio.create_task(show(inspector, [("local:files", "read")]))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            inspector.post_message(event)
            await pilot.pause()
            assert revocations(app) == []
        finally:
            release.set()
            await refresh


async def open_permissions(app, pilot):
    workbench = app.query_one(MCPWorkbench)
    async with asyncio.timeout(5):
        while workbench.is_loading or workbench._reloading:
            await pilot.pause()
    workbench.set_mode("permissions")
    await pilot.pause()
    inspector = app.query_one(MCPInspector)
    await show(inspector, context=workbench._tool_policy_profile_context)
    await pilot.pause()
    return workbench, inspector


@pytest.mark.asyncio
@private_profile_test
async def test_failed_revoke_can_be_retried_without_reselecting_the_tool(
    request, tmp_path, monkeypatch
):
    app = PermissionsApp(tmp_path / "permissions.json")
    service = app.unified_mcp_service
    service.session_approvals.update(("default", *grant) for grant in GRANTS)
    original = service.revoke_session_approval
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("synthetic revoke failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "revoke_session_approval", fail_once)
    async with app.run_test(size=(100, 60)) as pilot:
        await open_permissions(app, pilot)
        app.query_one("#mcp-inspector-session-approval-revoke-0", Button).press()
        await pilot.pause()
        assert failed
        assert service.is_session_approved("agent:builtin", "calculator")
        button = app.query_one("#mcp-inspector-session-approval-revoke-0", Button)
        assert not button.disabled
        button.press()
        await pilot.pause()
        assert not service.is_session_approved("agent:builtin", "calculator")
        assert service.is_session_approved("local:docs", "search")


@pytest.mark.asyncio
@private_profile_test
async def test_older_revoke_refresh_preserves_a_newer_profiles_listing(
    request, tmp_path, monkeypatch
):
    app = PermissionsApp(tmp_path / "permissions.json")
    service = app.unified_mcp_service
    service.session_approvals.update(("default", *grant) for grant in GRANTS)
    async with app.run_test(size=(100, 60)) as pilot:
        workbench, inspector = await open_permissions(app, pilot)
        context = workbench._tool_policy_profile_context
        entered, release = asyncio.Event(), asyncio.Event()
        original = workbench._sync_permissions_mode

        async def held_sync():
            entered.set()
            await release.wait()
            await original()

        monkeypatch.setattr(workbench, "_sync_permissions_mode", held_sync)
        revoke = asyncio.create_task(
            workbench.on_mcp_inspector_revoke_session_approval_requested(
                MCPInspector.RevokeSessionApprovalRequested(
                    "agent:builtin", "calculator", context
                )
            )
        )
        other = PermissionProfileContext("other", 8, "c" * 64, 4)
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await show(inspector, [("local:files", "read")], other)
        finally:
            release.set()
            await revoke
        await pilot.pause()
        assert inspector._current_permission_session_approvals == [
            ("local:files", "read")
        ]
        assert inspector._current_permission_profile_context == other
        app.query_one("#mcp-inspector-session-approval-revoke-0", Button).press()
        await pilot.pause()
        assert service.is_session_approved("local:docs", "search")
