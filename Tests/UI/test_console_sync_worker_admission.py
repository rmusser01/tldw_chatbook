"""Native refresh requests do not allocate work before Textual admission."""

import asyncio
import gc
import sys
import warnings

import pytest
from textual._context import NoActiveAppError
from textual.worker import WorkerState

from Tests.UI.test_console_hook_review_send_freeze import _until as until
from Tests.UI.test_console_tab_switch_binds_composer import _host, _running
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_native_refresh_survives_worker_replacement_and_pre_entry_cancel():
    host = _host()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        refused = ChatScreen(host.app_instance)
        refused._console_sync_maintenance_paused = True
        refused._console_sync_requested = True
        with pytest.raises(NoActiveAppError):
            refused._console_sync_maintenance_resume()
        del refused
        gc.collect()
        async with _running(host) as pilot:
            screen = host.screen
            assert await until(lambda: not screen._console_sync_in_progress, 8)
            screen._console_sync_maintenance_close_admission()
            assert await screen._console_sync_maintenance_drain(
                asyncio.get_running_loop().time() + 5
            )
            screen._console_sync_requested = False
            screen._console_sync_maintenance_resume()
            await pilot.pause()
            assert await until(lambda: not screen._console_sync_in_progress, 8)

            def request():
                screen._console_sync_maintenance_close_admission()
                screen._console_sync_requested = True
                screen._console_sync_maintenance_resume()

            async def exercise(cancel):
                entered = []
                previous = sys.getprofile()

                def observe(frame, event, _arg):
                    if (
                        event == "call"
                        and frame.f_code
                        is ChatScreen._sync_native_console_chat_ui.__code__
                    ):
                        entered.append(asyncio.current_task())

                request()
                first = next(
                    w
                    for w in host.workers
                    if w.node is screen and w.group == "console-sync"
                )
                first_task = first._task
                sys.setprofile(observe)
                try:
                    if cancel:
                        first.cancel()
                        await asyncio.gather(first_task, return_exceptions=True)
                        assert first.state != WorkerState.SUCCESS
                        assert first_task not in entered
                    else:
                        request()
                        second = next(
                            w
                            for w in host.workers
                            if w.node is screen
                            and w.group == "console-sync"
                            and w is not first
                        )
                        await asyncio.gather(
                            first_task, second._task, return_exceptions=True
                        )
                        assert first_task not in entered
                        assert second._task in entered
                        assert second.state == WorkerState.SUCCESS
                        assert not screen._console_sync_in_progress
                finally:
                    sys.setprofile(previous)

            await exercise(True)
            store = screen._ensure_console_chat_store()
            session = store.create_session(title="Admitted refresh", activate=False)
            await exercise(False)
            assert await until(
                lambda: bool(screen.query("#console-session-tab-" + session.id)), 8
            )
            assert await until(
                lambda: (
                    not screen._console_sync_in_progress
                    and not screen._console_control_bar_replay_whole_sync
                ),
                8,
            )
            original_tabs = screen._sync_console_native_session_tabs

            async def fail_live_tabs():
                raise RuntimeError("live native sync failure control")

            screen._sync_console_native_session_tabs = fail_live_tabs
            try:
                with pytest.raises(
                    RuntimeError, match="live native sync failure control"
                ):
                    await screen._sync_native_console_chat_ui()
            finally:
                screen._sync_console_native_session_tabs = original_tabs
            await asyncio.sleep(0)
            gc.collect()
        gc.collect()
    leaked = [
        str(item.message)
        for item in caught
        if issubclass(item.category, RuntimeWarning)
    ]
    assert not leaked, leaked
