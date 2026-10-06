"""TASK-34100.3 review G3V2-03: a refused Retry capture stays a trace pause.

With trace capture on, the third send of a saved Kimi chat stops before the
provider with "Trace capture blocked" (TASK-34100.3.1 owns that refusal). The
card's first action, Retry capture, met the same refusal and then showed the
false "Response delivery status is unknown ... Retry anyway may send a
duplicate request" card, with "Send blocked" in the composer, although the
card had just said the provider was not contacted.

The pause records that no call boundary was ever built (the planner refused
before reservation), so nothing of this send can have reached the provider.
A retry of that shape cannot be verified and is refused again; it must leave
the send paused at the same card, with Send without capture still working.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_blocked_send_recovery import _frame_rows, _until
from Tests.UI.test_console_moonshot_reasoning_reply import (
    REPLY,
    TRACE_BLOCKED,
    _mounted_console,
    _send,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

pytestmark = pytest.mark.bootstrap_profile

DELIVERY_UNKNOWN = ("delivery status is unknown", "Send blocked", "Retry anyway")


def _screen(h) -> str:
    return "\n".join(_frame_rows(h.host))


def _checkpoints(h) -> list[tuple]:
    return [
        tuple(row)
        for row in h.database.get_connection().execute(
            "SELECT state, checkpoint_revision, assistant_message_version "
            "FROM console_dispatch_checkpoints ORDER BY created_at"
        )
    ]


def _replies(h, session_id: str) -> list[str]:
    return [
        message.content
        for message in h.store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.ASSISTANT and message.content
    ]


@pytest.mark.asyncio
async def test_refused_retry_capture_keeps_the_trace_pause(
    tmp_path, monkeypatch
) -> None:
    async with _mounted_console(
        tmp_path, monkeypatch, provider="moonshot", model="kimi-k3"
    ) as h:
        saved = h.store.ensure_session()
        for turn, text in enumerate(("Say hello", "And again"), 1):
            await _send(h, text)
            await _until(
                h.pilot,
                lambda turn=turn: _replies(h, saved.id) == [REPLY] * turn,
                timeout=30.0,
                what=f"turn {turn} did not reply",
            )
        await _send(h, "Once more")
        await _until(
            h.pilot,
            lambda: TRACE_BLOCKED in _screen(h),
            timeout=30.0,
            what="turn 3 did not stop at the trace pause (TASK-34100.3.1 landed?)",
        )
        assert len(h.calls) == 2
        paused_checkpoints = _checkpoints(h)

        await _wait_for_selector(h.console, h.pilot, "#console-trace-retry")
        await h.pilot.click("#console-trace-retry")
        await _until(
            h.pilot,
            lambda: (
                "refused again" in _screen(h)
                or any(copy in _screen(h) for copy in DELIVERY_UNKNOWN)
            ),
            timeout=30.0,
            what="Retry capture did not settle",
        )

        screen = _screen(h)
        assert [copy for copy in DELIVERY_UNKNOWN if copy in screen] == [], screen
        assert TRACE_BLOCKED in screen
        assert "The provider was not contacted" in screen
        recovery = h.store.dispatch_recovery_for_session(saved.id)
        assert recovery is None or not recovery.recovery_needed
        assert h.controller.trace_call_recovery_preparation() is not None
        assert len(h.calls) == 2, "a refused retry must not reach the provider"
        # The refused retry wrote no dispatch state of its own.
        assert _checkpoints(h) == paused_checkpoints

        # The pause is still the ordinary one: Send without capture answers.
        await h.pilot.click("#console-trace-send-without")
        await _until(
            h.pilot,
            lambda: _replies(h, saved.id) == [REPLY] * 3,
            timeout=30.0,
            what="Send without capture did not reply after the refused retry",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)
        assert len(h.calls) == 3
        assert h.store.dispatch_recovery_for_session(saved.id) is None
        assert [copy for copy in DELIVERY_UNKNOWN if copy in _screen(h)] == []
        assert (
            h.database.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )
