"""Queued Console rebuilds must not register controls after app shutdown."""

import asyncio

import pytest
from textual.app import App

from Tests.UI.test_console_workspace_details_tray import TrayApp, _state
from tldw_chatbook.app_lifecycle import LifecycleMixin
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)

pytestmark = pytest.mark.bootstrap_profile


class ShutdownTrayApp(LifecycleMixin, TrayApp):
    """Use production exit sequencing with only the real tray and view workers."""

    _handle_exception = App._handle_exception

    def __init__(self):
        super().__init__(_state())
        self._console_runtime_shutdown_task = None

    async def _shutdown_app_owned_lifecycles(self):
        await self._shutdown_console_runtime()

    def on_unmount(self, event):
        event.prevent_default()

    def on_worker_state_changed(self, event):
        event.prevent_default()


@pytest.mark.asyncio
async def test_queued_automatic_rebuilds_crossing_shutdown_do_not_remount(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.app_lifecycle.arm_exit_watchdog", lambda **kw: None
    )
    app = ShutdownTrayApp()
    removal_entered = asyncio.Event()
    release = asyncio.Event()
    rebuilt_twice = asyncio.Event()
    mounts_after_stop = []
    original_close_all = app._close_all

    async def finish_queued_rebuilds_before_close():
        assert not app.is_running
        release.set()
        try:
            await asyncio.wait_for(rebuilt_twice.wait(), timeout=10)
        finally:
            await original_close_all()

    monkeypatch.setattr(app, "_close_all", finish_queued_rebuilds_before_close)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tray = app.query_one("#tray", ConsoleWorkspaceDetailsTray)
        child = tray.query_one("#console-workspace-server-features-collapsed")
        original_child_exit = child._message_loop_exit

        async def held_child_removal():
            removal_entered.set()
            await release.wait()
            await original_child_exit()

        monkeypatch.setattr(child, "_message_loop_exit", held_child_removal)
        original_mount_all = tray.mount_all
        original_check = tray._check_recompose
        checks = 0

        def observed_mount_all(widgets, *, before=None, after=None):
            try:
                return original_mount_all(widgets, before=before, after=after)
            finally:
                if not app.is_running:
                    mounts_after_stop.append(tuple(tray.children))

        async def observe_check():
            nonlocal checks
            checks += 1
            try:
                await original_check()
            finally:
                if checks >= 2:
                    rebuilt_twice.set()

        monkeypatch.setattr(tray, "mount_all", observed_mount_all)
        monkeypatch.setattr(tray, "_check_recompose", observe_check)

        async def sync_once():
            tray.sync_state(_state(runtime_label="Local file tools: First"))

        await app.run_worker(sync_once(), group="console-sync").wait()
        await asyncio.wait_for(removal_entered.wait(), timeout=10)
        tray.sync_state(_state(runtime_label="Local file tools: Second"))
    assert checks == 2
    assert mounts_after_stop == []
    assert app._exception is None
