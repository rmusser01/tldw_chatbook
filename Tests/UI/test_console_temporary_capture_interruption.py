"""A completed card send survives interruption of its final UI refresh."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.test_console_received_intent_feedback import _received_console_case, _until
from Tests.UI.test_console_temporary_capture_flow import (
    _paused_for_capture,
    _replied,
    _replace_with_authored_draft,
    _send_draft,
    _temporary_tab,
    _turn_capture_on,
    _unsent_turns,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("newer_draft", [False, True])
async def test_answered_capture_turn_never_returns_after_ui_cleanup_is_cancelled(
    monkeypatch, newer_draft
):
    async with _received_console_case(
        monkeypatch, f"capture-interrupted-cleanup-{newer_draft}", durable=True
    ) as case:
        session_id = await _temporary_tab(case)
        await _turn_capture_on(case, session_id)
        sent = "Again"
        await _send_draft(case, sent)
        assert await _until(lambda: _paused_for_capture(case, session_id), 15)
        runtime = case.console._console_runtime()
        assert await _until(lambda: bool(_unsent_turns(runtime, session_id)), 10)
        reached = asyncio.Event()
        original_finished = case.console._session._read_trace_recovery_finished()

        async def interrupted_finished():
            if reached.is_set():
                # The independent refresh worker survives the cancelled action.
                await original_finished()
                return
            # Only presentation is interrupted: the real controller already
            # returned a successful send and the provider's reply is present.
            assert _replied(case, session_id)
            if newer_draft and not reached.is_set():
                await _replace_with_authored_draft(case, sent)
            reached.set()
            raise asyncio.CancelledError

        monkeypatch.setattr(
            case.console._session, "_read_trace_recovery_finished",
            lambda: interrupted_finished,
        )
        send = case.console.query_one("#console-trace-send-without", Button)
        assert await _until(lambda: send.display and not send.disabled, 10)
        send.press()
        assert await _until(reached.is_set, 45)
        assert len(case.provider_calls) == 1
        assert await _until(lambda: not _unsent_turns(runtime, session_id), 10), (
            runtime.recoveries_for_session(session_id)
        )
        assert runtime.recoveries_for_session(session_id) == ()
        expected = sent if newer_draft else ""
        assert case.store.session_draft(session_id) == expected
        assert await _until(
            lambda: case.console._console_composer_or_none().draft_text() == expected,
            10,
        )
