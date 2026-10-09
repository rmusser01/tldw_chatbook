"""A naturally deferred final poll must retain its original refresh owner."""

import asyncio
import sys
import time

import pytest

from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_poll_reconciliation import (
    _original_transition_events,
    _transition_parent,
)
from Tests.UI.test_console_received_intent_feedback import _received_console_case
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_idle_final_poll_waits_for_original_full_refresh(
    monkeypatch, record_property
):
    async with _received_console_case(
        monkeypatch, "deferred-final-poll", durable=True
    ) as case:
        console = case.console
        assert await _until(
            lambda: not console._console_sync_in_progress
            and console._console_transcript_sync_timer is None,
            5,
        )
        assert not console._console_transcript_poll_needed()
        console._console_sync_maintenance_close_admission()
        results = []
        timer = poll = None

        def observe(kind, label, frame, value):
            if (
                kind == sys.monitoring.events.PY_RETURN
                and label == "full"
                and frame.f_locals.get("self") is console
                and timer is not None
                and asyncio.current_task() is timer._task
                and _transition_parent(frame, poll.__code__, console) is not None
            ):
                results.append(value)

        try:
            assert await console._console_sync_maintenance_drain(time.monotonic() + 5)
            with _original_transition_events(
                [("full", ChatScreen, "_sync_native_console_chat_ui")], observe
            ):
                console._start_console_transcript_sync_timer()
                timer = console._console_transcript_sync_timer
                assert timer is not None
                poll = timer._callback
                assert (
                    poll.__code__
                    in ChatScreen._start_console_transcript_sync_timer.__code__.co_consts
                )
                assert await _until(lambda: bool(results), 5)
                record_property("original_deferred_refresh_results", list(results))
                assert results[0] is False
                assert console._console_sync_requested
                assert (
                    console._console_transcript_sync_timer is timer
                ), "A deferred full refresh prematurely stopped the final poll"
                console._console_sync_maintenance_resume()
                assert await _until(
                    lambda: True in results
                    and console._console_transcript_sync_timer is None,
                    10,
                )
                assert not console._console_sync_requested
                assert not console._console_transcript_poll_needed()
        finally:
            console._console_sync_maintenance_resume()
