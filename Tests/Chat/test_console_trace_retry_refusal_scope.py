"""TASK-34100.3 review G3V2-03: the refused-retry re-pause acts on one shape only.

``repause_refused_trace_retry`` returns a send to its trace pause only when the
pause recorded that no call boundary was ever built, for a manual send that
the Retry action has just accepted. Every other recorded shape -- above all a
real boundary, whose dispatch outcome may be unknown -- must fall through to
the controller's unknown-delivery handling. The end-to-end behaviour is pinned
by ``Tests/UI/test_console_trace_retry_refusal.py``; these tests pin the guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_retry_refusal import (
    repause_refused_trace_retry,
)
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState


class _Store:
    def __init__(self) -> None:
        self.transitions: list[object] = []

    def compare_and_set_preparation(self, session_id, transition):
        self.transitions.append(transition)
        return SimpleNamespace(state=transition.new_state)


def _controller(recorded, state=ConsoleTurnPreparationState.ACCEPTED):
    preparation = SimpleNamespace(
        preparation_id="prep-1",
        session_id="session-1",
        state=state,
        origin="manual",
    )
    return SimpleNamespace(
        _trace_call_boundaries_by_preparation={"prep-1": recorded},
        _preparation_by_id=lambda preparation_id: preparation,
        store=_Store(),
        _set_run_state=lambda *args, **kwargs: None,
    )


def _continuation(origin=ConsoleSubmissionOrigin.MANUAL):
    return SimpleNamespace(
        origin=origin,
        session_id="session-1",
        commit=SimpleNamespace(user_message_id="u-1", assistant_message_id="a-1"),
        queue_entry_id=None,
        committed_context_epoch=None,
    )


def test_an_unbuilt_boundary_returns_the_send_to_its_pause(monkeypatch) -> None:
    from tldw_chatbook.Chat import console_trace_retry_refusal
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsolePreparationPauseKind,
    )

    # The real transition needs a full preparation; its legality from
    # ACCEPTED is the controller's own original-pause path. Here only the
    # guard is under test.
    monkeypatch.setattr(
        console_trace_retry_refusal,
        "pause_for_trace_call_failure",
        lambda current, failure: SimpleNamespace(
            state=ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.TRACE_CALL,
        ),
    )
    controller = _controller(TraceCallPersistenceError(reservation_status="unknown"))

    result = repause_refused_trace_retry(controller, "prep-1", _continuation())

    assert result is not None
    assert "refused again" in result.visible_copy
    assert "not contacted" in result.visible_copy
    assert [
        (transition.new_state, transition.pause_kind.value)
        for transition in controller.store.transitions
    ] == [(ConsoleTurnPreparationState.PAUSED, "trace_call")]


@pytest.mark.parametrize(
    ("recorded", "origin", "state"),
    [
        pytest.param(
            SimpleNamespace(dispatch_outcome="unknown"),
            ConsoleSubmissionOrigin.MANUAL,
            ConsoleTurnPreparationState.ACCEPTED,
            id="a-real-boundary",
        ),
        pytest.param(
            TraceCallPersistenceError(boundary=SimpleNamespace()),
            ConsoleSubmissionOrigin.MANUAL,
            ConsoleTurnPreparationState.ACCEPTED,
            id="an-error-that-carries-a-boundary",
        ),
        pytest.param(
            None,
            ConsoleSubmissionOrigin.MANUAL,
            ConsoleTurnPreparationState.ACCEPTED,
            id="no-recorded-pause",
        ),
        pytest.param(
            TraceCallPersistenceError(reservation_status="unknown"),
            ConsoleSubmissionOrigin.QUEUED,
            ConsoleTurnPreparationState.ACCEPTED,
            id="a-queued-send",
        ),
        pytest.param(
            TraceCallPersistenceError(reservation_status="unknown"),
            ConsoleSubmissionOrigin.MANUAL,
            ConsoleTurnPreparationState.PAUSED,
            id="not-accepted-by-the-retry",
        ),
    ],
)
def test_every_other_shape_keeps_unknown_delivery_handling(
    recorded, origin, state
) -> None:
    controller = _controller(recorded, state)

    assert (
        repause_refused_trace_retry(controller, "prep-1", _continuation(origin)) is None
    )
    assert controller.store.transitions == []
