"""TASK-33621.23 AC2: no BLOCKED exit before provider entry stays silent.

``resume_durable_postcommit`` guards provider entry: with Capture On and no
frozen trace request, the send must not reach the provider. That guard
returned a BLOCKED result and published nothing else. The turn stayed
ACCEPTED, with no pause, no transcript card and no blocked run state, which
is the reported "Blocked with no visible reason" shape. It now raises into
the trace-call handler, so the send pauses with the card's Retry / Send
without capture / Cancel, and each of the three works: Retry rebuilds the
missing request, and Cancel is not refused at the reservation check, because
nothing was reserved.

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


async def _paused_on_a_missing_request(tmp_path: Path, monkeypatch, build):
    """Send once with Capture On while the durable trace build yields nothing."""
    _db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    monkeypatch.setattr(controller, "_build_durable_trace_request", build(controller))
    await controller.submit_draft("hello", session_id="session-1")
    paused = store.preparation_for_session("session-1")
    assert paused is not None and paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_CALL
    return store, controller, entries, paused


def _never_builds(_controller):
    return lambda **_kwargs: None


def _builds_after_the_first_try(controller):
    real = controller._build_durable_trace_request
    calls: list[None] = []

    def build(**kwargs):
        calls.append(None)
        return None if len(calls) == 1 else real(**kwargs)

    return build


@pytest.mark.asyncio
async def test_cancel_on_the_missing_request_card_ends_the_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing was reserved, so Cancel must not stop at the reservation check."""
    store, controller, entries, paused = await _paused_on_a_missing_request(
        tmp_path, monkeypatch, _never_builds
    )

    cancelled = controller.cancel_library_preparation(paused.preparation_id)

    assert cancelled.visible_copy == "Trace-captured send canceled."
    after = store.preparation_for_session("session-1")
    assert after is None or after.state is not ConsoleTurnPreparationState.PAUSED
    state = controller.run_state_for("session-1")
    assert state.status is ConsoleRunStatus.STOPPED
    assert entries == []


@pytest.mark.asyncio
async def test_retry_on_the_missing_request_card_rebuilds_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry must rebuild the lost request, not hit the same check again."""
    _store, controller, entries, paused = await _paused_on_a_missing_request(
        tmp_path, monkeypatch, _builds_after_the_first_try
    )

    retried = await controller.retry_library_preparation(paused.preparation_id)

    assert retried.accepted is True and retried.provider_started is True
    assert len(entries) == 1
    assert entries[0]["capture_mode_override"] is ConsoleTraceCaptureMode.CAPTURE_ON
    assert entries[0]["trace_request"] is not None


@pytest.mark.asyncio
async def test_a_retry_that_still_cannot_build_keeps_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed rebuild leaves the card's other ways out in place."""
    store, controller, entries, paused = await _paused_on_a_missing_request(
        tmp_path, monkeypatch, _never_builds
    )

    retried = await controller.retry_library_preparation(paused.preparation_id)

    assert retried.accepted is False
    assert "Trace provenance could not be saved" in retried.visible_copy
    assert entries == []
    still = store.preparation_for_session("session-1")
    assert still is not None and still.state is ConsoleTurnPreparationState.PAUSED
    assert still.pause_kind is ConsolePreparationPauseKind.TRACE_CALL
    cancelled = controller.cancel_library_preparation(paused.preparation_id)
    assert cancelled.visible_copy == "Trace-captured send canceled."
