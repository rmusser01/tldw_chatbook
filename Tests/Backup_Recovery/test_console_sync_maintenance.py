"""Console synchronization settles without tearing down its live view."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, time
from types import SimpleNamespace, MethodType
from contextlib import nullcontext
from unittest.mock import Mock, AsyncMock
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

scenario = sys.argv[1]
async def main():
    screen = SimpleNamespace(
        is_running=True, _closing=False, _closed=False,
        _console_sync_in_progress=False, _console_sync_requested=False,
        _console_session_tabs_sync_lock=asyncio.Lock(), _console_session_tabs_sync_calls=0,
        _console_chat_store=None,
        _console_chat_controller=None,
        _console_context_read_snapshot=None,
        _current_console_attach_visit=lambda: None,
        _request_console_control_bar_sync=Mock(),
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
        '_sync_console_rail_and_controls', '_sync_console_settings_recovery_surfaces',
        '_sync_console_live_work_readiness_rows',
    ):
        setattr(screen, name, Mock())
    screen._message = SimpleNamespace(reconcile_console_speech_context=Mock())
    screen._session = SimpleNamespace(_sync_console_session_draft=Mock(), schedule_manual_read_acknowledgement=Mock())
    screen._retrieval = SimpleNamespace(
        _warm_console_effective_scope_cache_if_stale=AsyncMock(),
        _refresh_active_dictionaries_summary_if_scope_changed=AsyncMock(),
        _refresh_active_world_books_summary_if_scope_changed=AsyncMock(),
    )
    screen._character = SimpleNamespace(_refresh_active_character_avatar_if_scope_changed=AsyncMock())
    screen._character_context = SimpleNamespace(refresh_presentation_if_scope_changed=AsyncMock())
    screen._workspace = SimpleNamespace(tick_workspace_build_scope=nullcontext)
    screen._sync_console_native_session_tabs = AsyncMock()
    screen._sync_native_console_transcript = AsyncMock()
    import tldw_chatbook.UI.Screens.chat_screen as module
    module.project_instruction_ui.sync_project_instruction_status_for_screen = Mock()
    screen._sync_native_console_chat_ui = MethodType(ChatScreen._sync_native_console_chat_ui, screen)
    scheduled = []
    screen.run_worker = lambda coro, **kwargs: scheduled.append(coro)
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
        witness = asyncio.create_task(entered.wait())
        try:
            done, _ = await asyncio.wait(
                (witness, active), return_when=asyncio.FIRST_COMPLETED
            )
            if active in done:
                await active  # Surface setup/guard errors instead of a blind event wait.
                raise AssertionError('FULL finished before the held retrieval boundary')
            assert entered.is_set()
        finally:
            witness.cancel()
            await asyncio.gather(witness, return_exceptions=True)
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
        if scenario == 'error':
            assert isinstance(result, ValueError)
            screen._request_console_control_bar_sync.assert_not_called()
            assert not getattr(screen, '_console_control_bar_replay_whole_sync', False)
        else:
            # The resumed pass reaches the real roleplay config gate while paused.
            assert result is False
            screen._request_console_control_bar_sync.assert_called_once_with(delayed=True)
            assert screen._console_control_bar_replay_whole_sync
        if scenario != 'timeout':
            assert await drain
        assert not scheduled
        assert screen._console_sync_requested
        ChatScreen._console_sync_maintenance_resume(screen)
        assert not screen._console_sync_maintenance_paused
        if scenario == 'error':
            assert len(scheduled) == 1
            assert not screen._console_sync_requested
        else:
            # The existing coalesced control-bar owner retains the pending FULL.
            assert not scheduled
            assert screen._console_sync_requested
            assert screen._console_control_bar_replay_whole_sync
            screen._request_console_control_bar_sync.assert_called_once_with(delayed=True)
    else:
        ChatScreen._console_sync_maintenance_close_admission(screen)
        await screen._sync_native_console_chat_ui()
        if scenario == 'torn_down':
            screen._closing = True
        ChatScreen._console_sync_maintenance_resume(screen)
        ChatScreen._console_sync_maintenance_resume(screen)
        assert len(scheduled) == (0 if scenario == 'torn_down' else 1)
    for coro in scheduled:
        coro.close()
asyncio.run(main())
print("retired and reopened")
"""


@pytest.mark.parametrize(
    "scenario", ["closed_entry", "active", "timeout", "error", "resume", "torn_down"]
)
def test_console_sync_maintenance_preserves_pending_pass(tmp_path, scenario):
    _run(tmp_path, scenario, "success", script=_SCRIPT)
