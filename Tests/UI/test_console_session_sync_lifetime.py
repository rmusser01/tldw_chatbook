"""Real tab-strip removal fences asynchronous session publication."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Console.console_session_surface import ConsoleSessionSurface

pytestmark = pytest.mark.bootstrap_profile


class SessionSurfaceHost(App[None]):
    def compose(self):
        yield ConsoleSessionSurface(SimpleNamespace(notify=MagicMock()))


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["lock", "remove", "first_mount", "later_mount"])
async def test_removed_surface_does_not_publish_after_session_sync_await(
    monkeypatch, boundary
):
    app = SessionSurfaceHost()
    async with app.run_test(size=(80, 24)):
        surface = app.query_one(ConsoleSessionSurface)
        strip = surface.query_one("#console-native-tab-strip")
        reached = asyncio.Event()
        release = asyncio.Event()
        sessions = [ConsoleChatSession(title="Original", id="s1")]
        parent_disposed = False
        disposed_mounts = []
        original_mount = strip.mount
        mount_calls = 0
        pause_mount = {"first_mount": 1, "later_mount": 2}.get(boundary)

        async def mount_then_wait(*widgets, **kwargs):
            nonlocal mount_calls
            if parent_disposed:
                disposed_mounts.append(tuple(widget.id for widget in widgets))
            await original_mount(*widgets, **kwargs)
            mount_calls += 1
            if mount_calls == pause_mount:
                reached.set()
                await release.wait()

        monkeypatch.setattr(strip, "mount", mount_then_wait)
        if boundary == "lock":
            await surface._session_sync_lock.acquire()
        elif boundary == "remove":
            child = strip.children[0]
            original_remove = child.remove

            async def remove_then_wait():
                await original_remove()
                reached.set()
                await release.wait()

            monkeypatch.setattr(child, "remove", remove_then_wait)

        pending = asyncio.create_task(
            surface.sync_sessions(sessions=sessions, active_session_id="s1")
        )
        try:
            if boundary == "lock":
                await asyncio.sleep(0)
                assert surface._session_sync_lock.locked() and not pending.done()
            else:
                await asyncio.wait_for(reached.wait(), 5)
            # Actual Textual removal disposes this exact parent and strip while
            # the original session-sync coroutine is suspended at its await.
            await surface.remove()
            assert not surface.is_attached and not strip.is_attached
            # Textual retains is_mounted after removal. Attachment and actual
            # post-disposal stock mount calls are the current lifetime oracle.
            parent_disposed = True
            if boundary == "lock":
                surface._session_sync_lock.release()
            release.set()
            await asyncio.wait_for(pending, 5)
            assert (
                not disposed_mounts
            ), "disposed session sync attempted new tab publication"
            assert not strip.children

            # A new mounted surface still publishes the original session tabs.
            replacement = ConsoleSessionSurface(SimpleNamespace(notify=MagicMock()))
            await app.screen.mount(replacement)
            await replacement.sync_sessions(sessions=sessions, active_session_id="s1")
            assert replacement.query_one("#console-session-tab-s1", Button)
            replacement_strip = replacement.query_one("#console-native-tab-strip")
            assert [
                child.id for child in replacement_strip.children
            ] == replacement._desired_tab_child_ids(
                sessions=sessions, active_session_id="s1"
            )
        finally:
            release.set()
            if boundary == "lock" and surface._session_sync_lock.locked():
                surface._session_sync_lock.release()
            await asyncio.gather(pending, return_exceptions=True)
