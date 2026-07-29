"""Screen mount work must not run on the App's message pump (TASK-1320).

Textual awaits a screen's mount inside `switch_screen`, and a widget's
`on_mount` is awaited as part of that mount. `handle_screen_navigation` is an
`@on` handler on the App, so anything awaited during mount is awaited ON the
App's message pump: while it runs the app handles no clicks, no bindings and no
further navigation.

Measured on the pre-fix code: during a 3s mount the App handled 0 of 5 posted
messages. With a backing service that is merely unreachable rather than slow,
that window was minutes.

These tests pin the contract: mounting returns promptly and the pump keeps
draining, with the data arriving afterwards.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.message import Message

from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench

# How long the fake backing service takes to answer. Long enough that a mount
# which waits for it is unmistakable, short enough to keep the suite quick.
SERVICE_DELAY = 1.0


class Ping(Message):
    """Stands in for any app-level input while a destination is mounting."""


class SlowHubService:
    """A hub service whose every read is slow, like an unreachable server."""

    def __init__(self) -> None:
        self.target_store = None
        self.context = UnifiedMCPContext(
            selected_source="local", selected_section="overview"
        )

    async def load_context(self):
        await asyncio.sleep(SERVICE_DELAY)
        return self.context

    async def select_source(self, source):
        self.context = replace(self.context, selected_source=source)
        return self.context

    async def select_scope(self, scope, scope_ref=None):
        return self.context

    async def select_section(self, section):
        return self.context

    async def load_section(self, section=None):
        await asyncio.sleep(SERVICE_DELAY)
        return []

    async def local_external_catalog(self):
        await asyncio.sleep(SERVICE_DELAY)
        return []

    def available_actions(self):
        return []


class SlowWorkbenchApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = SlowHubService()
        self.pings_handled = 0

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")

    async def _on_ping(self, _message: Ping) -> None:  # pragma: no cover - see on_ping
        self.pings_handled += 1

    async def on_ping(self, _message: Ping) -> None:
        self.pings_handled += 1


@pytest.mark.asyncio
async def test_workbench_mount_does_not_wait_for_its_backing_service():
    """Mounting must return promptly instead of awaiting the service."""
    app = SlowWorkbenchApp()
    started = time.perf_counter()
    async with app.run_test() as pilot:
        await pilot.pause()
        mounted_after = time.perf_counter() - started

        assert mounted_after < SERVICE_DELAY, (
            f"mount took {mounted_after:.2f}s with a {SERVICE_DELAY}s service: "
            "the mount is waiting for I/O, so the App pump is blocked for that "
            "whole window and the app is frozen"
        )


class Navigate(Message):
    """Stands in for NavigateToScreen."""


class NavigatingApp(App):
    """Mirrors the real freeze path: an App-level handler that AWAITS the mount.

    This is the shape that matters. `handle_screen_navigation` is an `@on`
    handler on the App and it awaits `switch_screen`, which awaits the incoming
    screen's mount. Mounting from a plain task instead would not reproduce the
    bug at all, because then nothing on the App's pump is waiting.
    """

    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = SlowHubService()
        self.pings_handled = 0
        self.mount_finished = False

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_navigate(self, _message: Navigate) -> None:
        await self.mount(MCPWorkbench(app_instance=self, id="mcp-workbench"))
        self.mount_finished = True

    async def on_ping(self, _message: Ping) -> None:
        self.pings_handled += 1


@pytest.mark.asyncio
async def test_app_pump_keeps_draining_while_a_destination_loads():
    """The app must keep handling messages while mount work is in flight."""
    app = NavigatingApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        app.post_message(Navigate())
        await asyncio.sleep(0.05)

        # Everything below lands while the destination is still loading.
        for _ in range(5):
            app.post_message(Ping())
        await asyncio.sleep(0.3)

        assert app.pings_handled == 5, (
            f"app handled {app.pings_handled}/5 messages while the destination "
            "was loading: the message pump is blocked, which is a frozen app"
        )


@pytest.mark.asyncio
async def test_workbench_data_still_arrives_after_the_deferred_load():
    """Deferring the load must not lose it -- the data still lands."""
    app = SlowWorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)

        # Let the deferred load finish on its own.
        deadline = time.perf_counter() + SERVICE_DELAY * 6
        while time.perf_counter() < deadline:
            await pilot.pause()
            await asyncio.sleep(0.1)
            if not workbench.is_loading:
                break

        assert not workbench.is_loading, "the deferred load never completed"


# ---------------------------------------------------------------------------
# Chatbooks: the mount body is SYNCHRONOUS disk work (glob/stat/zipfile) inside
# an `async def`, so it blocks the event loop itself, not merely the pump.
# Deferring it to a coroutine worker would not be enough -- it has to leave the
# loop entirely.
# ---------------------------------------------------------------------------


async def _max_event_loop_stall(during, *, tick: float = 0.02) -> float:
    """Run `during()` and return the longest the event loop went unserviced.

    A ping counter cannot measure synchronous blocking: while the loop is
    blocked the *test* cannot post pings either, so everything serializes and
    the count comes out clean. A heartbeat measures it directly -- each tick
    records how late it actually woke up, and blocking work shows up as one
    long gap.
    """
    stalls: list[float] = []
    stop = False

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(tick)
            now = time.perf_counter()
            stalls.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    try:
        await during()
    finally:
        stop = True
        await asyncio.sleep(tick * 2)
        beat.cancel()
    return max(stalls) if stalls else 0.0


@pytest.mark.asyncio
async def test_chatbooks_scan_does_not_block_the_event_loop(monkeypatch):
    """The chatbook directory scan must run off the event loop."""
    import tldw_chatbook.UI.Chatbooks_Window_Improved as chatbooks_module
    from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved

    def blocking_secure(path):
        # Stands in for the real glob/stat/zipfile work: blocking, not awaitable.
        # If this runs on the event loop, nothing else in the app can run.
        time.sleep(SERVICE_DELAY)
        return path

    monkeypatch.setattr(chatbooks_module, "secure_chatbook_directory", blocking_secure)

    class ChatbooksHostApp(App):
        def __init__(self) -> None:
            super().__init__()
            self.pings_handled = 0

        def compose(self) -> ComposeResult:
            yield from ()

        async def on_navigate(self, _message: Navigate) -> None:
            await self.mount(ChatbooksWindowImproved(self))

        async def on_ping(self, _message: Ping) -> None:
            self.pings_handled += 1

    app = ChatbooksHostApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        async def mount_it() -> None:
            app.post_message(Navigate())
            await asyncio.sleep(SERVICE_DELAY + 0.5)

        stall = await _max_event_loop_stall(mount_it)

        assert stall < SERVICE_DELAY / 2, (
            f"event loop stalled {stall:.2f}s during the chatbook scan "
            f"(blocking work takes {SERVICE_DELAY}s): the scan is running on "
            "the loop, so the whole app is frozen for its duration"
        )
