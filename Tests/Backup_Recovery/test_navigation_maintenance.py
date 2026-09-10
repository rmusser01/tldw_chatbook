"""Exact navigation workers and shielded flushes settle before screen census."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, time
from types import SimpleNamespace
from textual.app import App
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

class Harness(App):
    _screen_navigation_lock = TldwCli._screen_navigation_lock
    _dispatch_screen_navigation = TldwCli._dispatch_screen_navigation
    handle_screen_navigation = TldwCli.handle_screen_navigation
    _retain_unfinished_flush = TldwCli._retain_unfinished_flush
    NAVIGATION_FLUSH_TIMEOUT_SECONDS = 10

async def main():
    app = Harness()
    # Bind the actual new hooks; no whole product startup or owner construction.
    for name in ("_screen_navigation_close_admission", "_screen_navigation_drain",
                 "_screen_navigation_resume", "_run_admitted_screen_navigation"):
        setattr(app, name, getattr(TldwCli, name).__get__(app, Harness))
    if sys.argv[1] == "inert":
        app._screen_navigation_close_admission()
        assert await app._screen_navigation_drain(time.monotonic())
        release = asyncio.Event()
        app._initial_screen_setup_task = asyncio.create_task(release.wait())
        assert not await app._screen_navigation_drain(time.monotonic())
        release.set()
        await app._initial_screen_setup_task
        assert await app._screen_navigation_drain(time.monotonic())
        print("retired and reopened")
        return
    app._initial_screen_pushed = True
    app._ui_ready = True
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def navigate(message):
        calls.append(message.screen_name)
        entered.set()
        await release.wait()
    app._handle_screen_navigation_locked = navigate
    app._notify_navigation_failure = lambda route: None
    async with app.run_test() as pilot:
        message = NavigateToScreen("chat")
        if sys.argv[1] == "worker":
            app._dispatch_screen_navigation(message)
            app._screen_navigation_close_admission()
            assert not await app._screen_navigation_drain(time.monotonic())
            await asyncio.wait_for(entered.wait(), 1)
            app._dispatch_screen_navigation(NavigateToScreen("settings"))
            await app.handle_screen_navigation(NavigateToScreen("settings"))
            assert calls == ["chat"]
            release.set()
            assert await app._screen_navigation_drain(time.monotonic()+1)
            app._screen_navigation_resume()
            await app.handle_screen_navigation(NavigateToScreen("settings"))
            assert calls == ["chat", "settings"]
        elif sys.argv[1] == "flush":
            async def flush():
                entered.set()
                await release.wait()
                calls.append("saved")
                return False
            app._handle_screen_navigation_locked = TldwCli._handle_screen_navigation_locked.__get__(app, Harness)
            app._resolve_screen_navigation_target = lambda route: (route, route, None)
            app._navigation_outgoing_screen = lambda: SimpleNamespace(flush_pending_work=flush)
            task = asyncio.create_task(app.handle_screen_navigation(message))
            await asyncio.wait_for(entered.wait(), 1)
            app._screen_navigation_close_admission()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert not await app._screen_navigation_drain(time.monotonic())
            release.set()
            assert await app._screen_navigation_drain(time.monotonic()+1)
            assert calls == ["saved"]
        else:
            app._screen_navigation_close_admission()
            app._initial_screen_pushed = False
            assert not await app._screen_navigation_drain(time.monotonic())
            app._initial_screen_pushed = True
            app._ui_ready = False
            assert not await app._screen_navigation_drain(time.monotonic())
            app._ui_ready = True
            assert await app._screen_navigation_drain(time.monotonic()+1)
    print("retired and reopened")
asyncio.run(main())
"""


@pytest.mark.parametrize("case", ["worker", "flush", "initial", "inert"])
def test_navigation_maintenance(tmp_path, case):
    _run(tmp_path, case, "success", script=_SCRIPT)
