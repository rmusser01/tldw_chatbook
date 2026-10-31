"""TASK-33621.23 AC2: no BLOCKED exit before provider entry stays silent.

``resume_durable_postcommit`` guards provider entry: with Capture On and no
frozen trace request, the send must not reach the provider. That guard
returned a BLOCKED result and published nothing else. The turn stayed
ACCEPTED, with no pause, no transcript card and no blocked run state, which
is the reported "Blocked with no visible reason" shape. It now raises into
the trace-call handler, so the send pauses with the card's Retry / Send
without capture / Cancel.

The durable commit always freezes a request when capture is on, so the test
breaks that invariant on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.Chat.test_console_trace_first_send_atomicity import (
    _force_capture_on,
    _record_provider_entries,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    trace_call_recovery_state,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_a_missing_trace_request_at_dispatch_pauses_with_a_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    monkeypatch.setattr(
        controller, "_build_durable_trace_request", lambda **_kwargs: None
    )

    result = await controller.submit_draft("hello", session_id="session-1")

    assert result.terminal_status is ConsoleRunStatus.BLOCKED
    assert entries == [], "the provider must not be entered without a trace"
    paused = store.preparation_for_session("session-1")
    assert paused is not None
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_CALL
    assert trace_call_recovery_state(paused) is not None
    state = controller.run_state_for("session-1")
    assert state.status is ConsoleRunStatus.BLOCKED
    assert "Send without capture" in state.visible_copy

    # The card's way out works: Send without capture reaches the provider.
    # (The recorded stream stands in for the provider, so the run state it
    # would settle is not asserted here.)
    resumed = await controller.send_without_capture(paused.preparation_id)

    assert resumed.accepted is True and resumed.provider_started is True
    assert len(entries) == 1
    assert entries[0]["capture_mode_override"] is ConsoleTraceCaptureMode.CAPTURE_OFF
