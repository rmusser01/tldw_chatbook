"""Console synchronization settles without tearing down its live view."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, time
from types import SimpleNamespace, MethodType
from unittest.mock import Mock, AsyncMock
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

scenario = sys.argv[1]
async def main():
    screen = SimpleNamespace(
        is_running=True, _closing=False, _closed=False,
        _console_sync_in_progress=False, _console_sync_requested=False,
        _console_chat_store=None, _console_session_tabs_sync_calls=0,
    )
    for name in (
        '_record_ui_worker_started', '_record_ui_worker_finished',
        '_sync_console_rail_and_controls', '_sync_console_settings_recovery_surfaces',
        '_sync_console_live_work_readiness_rows',
        '_sync_console_chat_core_state', '_sync_console_changed_files_if_scope_changed',
        '_current_console_rail_state', '_sync_console_control_bar',
        '_sync_console_settings_summary', '_sync_console_mode_bar',
        '_dispatch_active_console_roleplay_refresh', '_sync_console_workspace_context',
        '_sync_console_rail_visibility_if_changed', '_dispatch_console_rail_preference_prune',
        '_sync_console_temporary_chip',
    ):
        setattr(screen, name, Mock())
    screen._message = SimpleNamespace(reconcile_console_speech_context=Mock())
    screen._session = SimpleNamespace(_sync_console_session_draft=Mock(), schedule_manual_read_acknowledgement=Mock())
    screen._character_context = SimpleNamespace(refresh_if_scope_changed=AsyncMock())
    from contextlib import nullcontext
    screen._workspace = SimpleNamespace(tick_workspace_build_scope=nullcontext)
    screen._retrieval = SimpleNamespace(
        _warm_console_effective_scope_cache_if_stale=AsyncMock(),
        _refresh_active_dictionaries_summary_if_scope_changed=AsyncMock(),
        _refresh_active_world_books_summary_if_scope_changed=AsyncMock(),
    )
    screen._character = SimpleNamespace(_refresh_active_character_avatar_if_scope_changed=AsyncMock())
    # Every FULL sync reads its attach visit (63cbf3177a) and the temporary
    # chip (TASK-37840); this double models an attached, visible Console.
    from tldw_chatbook.UI.Console_Modules.attach_visit import ConsoleAttachVisit
    runtime = SimpleNamespace(view=screen, _attached_generation=1)
    screen._current_console_attach_visit = lambda: ConsoleAttachVisit(runtime, screen, 1, 1, True)
    screen._sync_console_native_session_tabs = AsyncMock()
    screen._sync_native_console_transcript = AsyncMock()
    import tldw_chatbook.UI.Screens.chat_screen as module
    module.project_instruction_ui.sync_project_instruction_status_for_screen = Mock()
    screen._sync_native_console_chat_ui = MethodType(ChatScreen._sync_native_console_chat_ui, screen)
    scheduled, replays = [], []
    def schedule_worker(worker, *, exclusive, group):
        assert callable(worker) and exclusive and group == "console-sync"
        scheduled.append(worker)
    screen.run_worker = schedule_worker
    screen._request_console_control_bar_sync = lambda **kwargs: replays.append(kwargs)
    if scenario == 'closed_entry':
        screen._console_sync_maintenance_paused = True
        await screen._sync_native_console_chat_ui()
        assert screen._record_ui_worker_started.call_count == 0
        assert screen._console_sync_requested
    elif scenario in ('active', 'timeout', 'error'):
        entered, release = asyncio.Event(), asyncio.Event()
        async def block():
            entered.set()
            await release.wait()
            if scenario == 'error':
                raise ValueError('sync failure')
        screen._retrieval._warm_console_effective_scope_cache_if_stale = block
        active = asyncio.create_task(screen._sync_native_console_chat_ui())
        waiter = asyncio.ensure_future(entered.wait())
        await asyncio.wait((waiter, active), return_when=asyncio.FIRST_COMPLETED)
        if not entered.is_set():
            # A sync that dies before the held step (a double missing an
            # attribute) re-raises its own error now, instead of waiting
            # out the subprocess timeout (TASK-38973).
            waiter.cancel()
            await active
            raise AssertionError('sync finished before reaching the held step')
        ChatScreen._console_sync_maintenance_close_admission(screen)
        await screen._sync_native_console_chat_ui()
        assert screen._console_sync_requested
        assert not await ChatScreen._console_sync_maintenance_drain(screen, time.monotonic())
        if scenario != 'timeout':
            drain = asyncio.create_task(ChatScreen._console_sync_maintenance_drain(screen, time.monotonic()+2))
            await asyncio.sleep(0)
            assert not drain.done()
        release.set()
        result = (await asyncio.gather(active, return_exceptions=True))[0]
        assert isinstance(result, ValueError) if scenario == 'error' else result is False
        if scenario != 'timeout':
            assert await drain
        assert not scheduled
        assert screen._console_sync_requested
        ChatScreen._console_sync_maintenance_resume(screen)
        if scenario == 'error':
            assert len(scheduled) == 1 and not replays
        else:
            # Since 63cbf3177a a pass that meets the maintenance pause hands
            # its whole sync to the replay timer, so resume starts no worker;
            # since TASK-38696 (1f7e722564) resume also cuts that timer short.
            assert not scheduled and screen._console_control_bar_replay_whole_sync
            assert len(replays) == 2 and replays[-1] == {}
        assert not screen._console_sync_maintenance_paused
    else:
        ChatScreen._console_sync_maintenance_close_admission(screen)
        await screen._sync_native_console_chat_ui()
        if scenario == 'torn_down':
            screen._closing = True
        ChatScreen._console_sync_maintenance_resume(screen)
        ChatScreen._console_sync_maintenance_resume(screen)
        assert len(scheduled) == (0 if scenario == 'torn_down' else 1)
asyncio.run(main())
print("retired and reopened")
"""


@pytest.mark.parametrize(
    "scenario", ["closed_entry", "active", "timeout", "error", "resume", "torn_down"]
)
def test_console_sync_maintenance_preserves_pending_pass(tmp_path, scenario):
    _run(tmp_path, scenario, "success", script=_SCRIPT)
