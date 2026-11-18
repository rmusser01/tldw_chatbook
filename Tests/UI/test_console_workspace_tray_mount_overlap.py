"""Actual tray mounts must not race a newer workspace publication."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App
from textual.widget import Widget
from textual.widgets import Input

from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


def _state(title: str) -> ConsoleWorkspaceContextState:
    browser = build_console_conversation_browser_state(
        rows=(ConsoleConversationBrowserInputRow(
            row_key="saved", conversation_id="saved", native_session_id=None,
            title=title, scope_type="global", workspace_id=None,
            workspace_label="Chats", selected=True,
        ),),
        active_workspace_id="", group_collapse_preferences={"section:chats": False},
        query="", marks_available=True,
    )
    return ConsoleWorkspaceContextState(
        heading="Chats", workspace_label="Workspace: Default",
        authority_label="Authority: local registry ready", sync_label="Sync: not configured",
        runtime_label="Local file tools: Private scratch", conversation_rows=(),
        conversation_empty_copy="No conversations.", conversation_browser=browser,
        change_workspace_enabled=True, change_workspace_recovery="",
        new_conversation_enabled=True, new_conversation_recovery="", recovery_copy="",
    )


def _assert_current_dom(tray: ConsoleWorkspaceContextTray, state) -> None:
    ids = [node.id for node in tray.walk_children() if node.id is not None]
    assert len(ids) == len(set(ids))
    assert tray._composed_publication_state is state
    assert tray._mounted_signatures(tray._composed_fixed_signature) == (
        tray._composed_row_signature, tray._composed_fixed_signature,
    )
    assert len(tray.query("#console-workspace-conversation-search")) == 1


@pytest.mark.asyncio
async def test_first_mount_and_newer_recompose_share_one_tray(monkeypatch) -> None:
    """Hold the real initial mount; a real recompose must wait for its children."""
    tray = ConsoleWorkspaceContextTray(_state("Initial"), content="conversations", id="tray")
    entered, release, requested = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = Widget.mount_composed_widgets

    async def hold_initial(widget, widgets):
        if widget is tray:
            entered.set()
            await release.wait()
        await original(widget, widgets)

    monkeypatch.setattr(Widget, "mount_composed_widgets", hold_initial)
    app = App()
    async with app.run_test(size=(120, 45)) as pilot:
        async def mount():
            await app.screen.mount(tray)

        async def recompose():
            requested.set()
            await tray.recompose()

        mounting = asyncio.create_task(mount())
        rebuilding = None
        try:
            async with asyncio.timeout(10):
                await entered.wait()
                assert tray.is_attached and not tray.is_mounted
                latest = _state("Latest")
                tray.state = latest
                rebuilding = asyncio.create_task(recompose())
                await requested.wait()
                # The requested task has entered the original rebuild and
                # reached its first await. Do not rely on wall-clock sleeps.
                await asyncio.sleep(0)
                release.set()
                await asyncio.gather(mounting, rebuilding)
                await pilot.pause()
                _assert_current_dom(tray, latest)
        finally:
            release.set()
            for task in (mounting, rebuilding):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (mounting, rebuilding) if task is not None),
                return_exceptions=True,
            )


@pytest.mark.asyncio
async def test_overlapping_retained_filter_rebuilds_keep_latest_state(monkeypatch) -> None:
    """The original search input survives two overlapping actual rebuilds."""
    app = App()
    tray = ConsoleWorkspaceContextTray(_state("Initial"), content="conversations", id="tray")
    async with app.run_test(size=(120, 45)) as pilot:
        await app.screen.mount(tray)
        await pilot.pause()
        search = tray.query_one("#console-workspace-conversation-search", Input)
        search.value = "Typed query"
        search.cursor_position = 3
        search.focus()
        await pilot.pause()
        entered, release, requested = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = tray.mount_all
        held = False

        async def hold_first(widgets, *, before=None, after=None):
            nonlocal held
            if not held:
                held = True
                entered.set()
                await release.wait()
            await original(widgets, before=before, after=after)

        monkeypatch.setattr(tray, "mount_all", hold_first)
        tray.state = _state("Intermediate")
        first = asyncio.create_task(tray.recompose())
        second = None
        try:
            async with asyncio.timeout(10):
                await entered.wait()
                latest = _state("Latest")
                tray.state = latest

                async def rebuild_latest():
                    requested.set()
                    await tray.recompose()

                second = asyncio.create_task(rebuild_latest())
                await requested.wait()
                await asyncio.sleep(0)
                release.set()
                await asyncio.gather(first, second)
                await pilot.pause()
                _assert_current_dom(tray, latest)
                assert tray.query_one("#console-workspace-conversation-search", Input) is search
                assert search.value == "Typed query" and search.cursor_position == 3
                assert app.focused is search
                assert not tray.recompose_in_flight
        finally:
            release.set()
            for task in (first, second):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (first, second) if task is not None),
                return_exceptions=True,
            )
