"""Native refresh requests do not allocate work before Textual admission.

Entry is observed through recording seams on the screen instance, which the
production schedulers read when they schedule: a coroutine created before
admission would be the recording wrapper's, so a refused or cancelled worker
would leave its "never awaited" RuntimeWarning.
"""

import asyncio
import gc
import warnings

import pytest
from textual._context import NoActiveAppError
from textual.worker import WorkerState

from Tests.UI.test_console_hook_review_send_freeze import _until as until
from Tests.UI.test_console_tab_switch_binds_composer import _host, _running
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def record_native_sync_entry(screen, entered):
    """Record each task that enters the screen's native sync.

    The wrapper is a plain function, so a run poll's display path would treat
    it as a custom callback and drop ``_preparing_poll``; use it only while no
    run poll is active.
    """
    native_sync = screen._sync_native_console_chat_ui

    async def recording_sync(*args, **kwargs):
        entered.append(asyncio.current_task())
        return await native_sync(*args, **kwargs)

    screen._sync_native_console_chat_ui = recording_sync


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

            entered = []
            record_native_sync_entry(screen, entered)

            def quiet():
                return not screen._console_sync_in_progress and not getattr(
                    screen, "_console_control_bar_replay_whole_sync", False
                )

            async def exercise(cancel):
                # A pending whole-sync replay takes the resume's request
                # instead of a console-sync worker.
                assert await until(quiet, 8)
                before = set(host.workers)
                request()
                first = next(
                    w
                    for w in host.workers
                    if w not in before
                    and w.node is screen
                    and w.group == "console-sync"
                )
                first_task = first._task
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
                        if w not in before
                        and w.node is screen
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

            await exercise(True)
            store = screen._ensure_console_chat_store()
            session = store.create_session(title="Admitted refresh", activate=False)
            await exercise(False)
            assert await until(
                lambda: bool(screen.query("#console-session-tab-" + session.id)), 8
            )
            assert await until(quiet, 8)
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


@pytest.mark.parametrize("cancel_before_entry", [True, False])
async def test_context_snapshot_native_publication_owns_deferred_refresh(
    cancel_before_entry,
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleContextReadSnapshot,
    )

    host = _host()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        async with _running(host):
            screen = host.screen
            controller = screen._ensure_console_chat_controller()
            store = controller.store
            session_id = store.active_session_id
            snapshot = ConsoleContextReadSnapshot.for_screen(screen, max_age=1)
            assert snapshot.schedule == screen.run_worker
            assert snapshot.refresh == screen._sync_native_console_chat_ui
            assert await snapshot.warm(controller, session_id)
            assert await until(lambda: not screen._console_sync_in_progress, 8)
            tab = store.create_session(title="Context publication", activate=False)
            store.append_message(
                session_id,
                role=ConsoleMessageRole.USER,
                content="Changed context requires native publication.",
                persist=False,
            )
            key = snapshot._cache_key(controller, session_id)
            assert key is not None and key != snapshot.key
            # The publication reads both seams from the screen when it runs.
            publications, entered = [], []
            record_native_sync_entry(screen, entered)
            schedule = screen.run_worker

            def recording_schedule(*args, **kwargs):
                worker = schedule(*args, **kwargs)
                if kwargs.get("group") == "console-context-publication":
                    publications.append(worker)
                    if cancel_before_entry:
                        worker.cancel()
                return worker

            screen.run_worker = recording_schedule
            await snapshot._refresh(controller, session_id, key)
            assert len(publications) == 1
            worker = publications.pop()
            assert worker.group == "console-context-publication"
            await asyncio.gather(worker._task, return_exceptions=True)
            if cancel_before_entry:
                assert worker._task not in entered
                assert worker.is_cancelled
            else:
                assert worker._task in entered
                assert worker.state == WorkerState.SUCCESS
                assert await until(
                    lambda: bool(screen.query("#console-session-tab-" + tab.id)), 8
                )
            del worker
            gc.collect()
        gc.collect()
    leaked = [
        str(item.message)
        for item in caught
        if issubclass(item.category, RuntimeWarning)
    ]
    assert not leaked, leaked
