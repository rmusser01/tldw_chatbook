"""Evidence-only real Textual pruning and exact strip replacement controls.

The coordinator installs this additive draft before native RED. Its author does
not import, launch or modify the managed application. No private lifecycle flag
is assigned: stock App._prune initiates each retirement.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Console.console_session_surface import (
    ConsoleSessionSurface,
    ConsoleSessionTabStrip,
)

pytestmark = pytest.mark.bootstrap_profile


class SessionSurfaceHost(App[None]):
    def __init__(self):
        super().__init__()
        self.churn = []
        self.owner = SimpleNamespace(notify=MagicMock(), ui_responsiveness_monitor=self)

    def record_mounts(self, kind, *, mounted, removed):
        self.churn.append((kind, mounted, removed))

    def compose(self):
        yield ConsoleSessionSurface(self.owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["surface_pruning", "strip_replacement"])
async def test_session_sync_refuses_retired_exact_strip_before_publication(
    monkeypatch, retirement
):
    app = SessionSurfaceHost()
    async with app.run_test(size=(80, 24)):
        surface = app.query_one(ConsoleSessionSurface)
        strip = surface.query_one("#console-native-tab-strip")
        strip_parent = strip.parent
        sync_reached = asyncio.Event()
        sync_release = asyncio.Event()
        prune_reached = asyncio.Event()
        prune_release = asyncio.Event()
        original_mount = strip.mount
        original_exit = strip._message_loop_exit
        mount_calls = 0
        retirement_started = False
        retired_dispatches = []
        removal_task = None
        sessions = [ConsoleChatSession(title="Original", id="s1")]

        async def original_mount_then_wait(*widgets, **kwargs):
            nonlocal mount_calls
            if retirement_started:
                retired_dispatches.append(tuple(widget.id for widget in widgets))
            await original_mount(*widgets, **kwargs)
            mount_calls += 1
            if mount_calls == 1:
                sync_reached.set()
                await sync_release.wait()

        async def held_original_exit():
            # The actual strip pump has received stock Prune and is closing.
            # Delay only physical detach so attachment cannot mask pruning.
            prune_reached.set()
            await prune_release.wait()
            await original_exit()

        monkeypatch.setattr(strip, "mount", original_mount_then_wait)
        if retirement == "surface_pruning":
            monkeypatch.setattr(strip, "_message_loop_exit", held_original_exit)
        pending = asyncio.create_task(
            surface.sync_sessions(sessions=sessions, active_session_id="s1")
        )
        try:
            await asyncio.wait_for(sync_reached.wait(), 5)
            assert mount_calls == 1 and strip.query_one(
                "#console-session-tab-s1", Button
            )
            if retirement == "surface_pruning":
                original_remove = surface.remove()
                removal_task = asyncio.ensure_future(original_remove)
                await asyncio.wait_for(prune_reached.wait(), 5)
                assert surface.is_attached and strip.is_attached
                assert surface._pruning and strip._pruning
                assert strip._closing
            else:
                original_remove = strip.remove()
                removal_task = asyncio.ensure_future(original_remove)
                await asyncio.wait_for(asyncio.shield(removal_task), 5)
                assert surface.is_attached and not strip.is_attached
                replacement_strip = ConsoleSessionTabStrip(
                    id="console-native-tab-strip"
                )
                await strip_parent.mount(replacement_strip)
                assert (
                    surface.query_one("#console-native-tab-strip") is replacement_strip
                )
                assert replacement_strip.is_attached
            retirement_started = True
            sync_release.set()
            await asyncio.wait_for(pending, 5)
            captured_sync_churn = list(app.churn)

            # Retire the actual original pump before any failure assertion.
            prune_release.set()
            await asyncio.wait_for(asyncio.shield(removal_task), 5)
            assert not strip.is_attached and not strip.children
            if retirement == "surface_pruning":
                assert not surface.is_attached
                target = ConsoleSessionSurface(app.owner)
                await app.screen.mount(target)
            else:
                target = surface
                assert target.is_attached
            await target.sync_sessions(sessions=sessions, active_session_id="s1")
            target_strip = target.query_one("#console-native-tab-strip")
            assert target.query_one("#console-session-tab-s1", Button)
            assert [child.id for child in target_strip.children] == (
                target._desired_tab_child_ids(sessions=sessions, active_session_id="s1")
            )
            assert app.churn[-1] == (
                "console-tabs",
                4,
                0 if retirement == "strip_replacement" else 1,
            )
            assert not retired_dispatches, (
                "retired/pruning exact strip received stock mount dispatch "
                f"after its original sync await: {retired_dispatches}"
            )
            assert (
                not captured_sync_churn
            ), "retired original sync recorded completed mount churn"
        finally:
            sync_release.set()
            prune_release.set()
            await asyncio.gather(pending, return_exceptions=True)
            if removal_task is not None:
                await asyncio.gather(removal_task, return_exceptions=True)
