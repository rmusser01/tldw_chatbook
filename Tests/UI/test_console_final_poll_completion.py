"""A deferred FULL refresh retains the real mounted transcript timer."""

import asyncio
import sys
import time

import pytest

from Tests.UI.response_rules_fixtures import mounted_rules_console
from Tests.UI.test_console_hook_review_send_freeze import _until
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


async def test_idle_final_poll_waits_for_original_full_refresh(tmp_path):
    async with mounted_rules_console(tmp_path) as case:
        console = case.console
        assert await _until(
            lambda: (
                not console._console_sync_in_progress
                and console._console_transcript_sync_timer is None
            ),
            5,
        )
        assert not console._console_transcript_poll_needed()
        console._console_sync_maintenance_close_admission()
        results = []
        timer = None
        code = ChatScreen._sync_native_console_chat_ui.__code__
        tool_id = next(
            i
            for i in range(5, 0, -1)
            if i != sys.monitoring.DEBUGGER_ID and sys.monitoring.get_tool(i) is None
        )

        def returned(_code, _offset, result):
            if timer is not None and asyncio.current_task() is timer._task:
                results.append(result)

        sys.monitoring.use_tool_id(tool_id, "final-poll-completion")
        sys.monitoring.register_callback(
            tool_id, sys.monitoring.events.PY_RETURN, returned
        )
        sys.monitoring.set_local_events(tool_id, code, sys.monitoring.events.PY_RETURN)
        try:
            assert await console._console_sync_maintenance_drain(time.monotonic() + 5)
            console._start_console_transcript_sync_timer()
            timer = console._console_transcript_sync_timer
            assert timer is not None
            assert await _until(lambda: bool(results), 5)
            assert results[0] is False
            assert console._console_sync_requested
            assert console._console_transcript_sync_timer is timer
            console._console_sync_maintenance_resume()
            assert await _until(
                lambda: (
                    True in results and console._console_transcript_sync_timer is None
                ),
                10,
            )
            assert not console._console_sync_requested
            assert not console._console_transcript_poll_needed()
        finally:
            console._console_sync_maintenance_resume()
            sys.monitoring.set_local_events(tool_id, code, 0)
            sys.monitoring.register_callback(
                tool_id, sys.monitoring.events.PY_RETURN, None
            )
            sys.monitoring.free_tool_id(tool_id)
