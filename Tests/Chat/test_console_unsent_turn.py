"""An unsent turn leaving the shelf releases the send it paused (TASK-33621.20).

Restore and Discard on the unsent-turn shelf must release the exact paused
preparation behind the turn, or the store refuses every later send in the
conversation ("Last send is blocked; resolve it first."). The pre-merge review
found the release covered only Library (RETRIEVAL) pauses: a Send once without
Library or a Retry that met a provider that was not ready re-paused the send
as DESTINATION_CHANGED, and nothing could release it.

Real controller and store; the Library service is never called.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    _controller_for_preparation,
    _preparation,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.console_unsent_turn import release_unsent_turn_preparation

SESSION = "session-1"
PREPARATION = "preparation-1"


class _UnusedLibrary:
    async def search(self, *_args, **_kwargs):  # pragma: no cover - never called
        raise AssertionError("releasing a paused send must not search")


def _paused(kind: ConsolePreparationPauseKind):
    return _controller_for_preparation(
        _preparation(state=ConsoleTurnPreparationState.PAUSED, pause_kind=kind),
        _UnusedLibrary(),
    )


def _released(store) -> bool:
    current = store.preparation_for_session(SESSION)
    return current is None or current.state is ConsoleTurnPreparationState.CANCELLED


@pytest.mark.parametrize(
    "kind",
    [
        ConsolePreparationPauseKind.RETRIEVAL,
        ConsolePreparationPauseKind.DESTINATION_CHANGED,
        ConsolePreparationPauseKind.PERSISTENCE,
    ],
)
def test_release_cancels_every_pause_the_shelf_can_end(kind):
    controller, store = _paused(kind)

    assert release_unsent_turn_preparation(controller, SESSION, PREPARATION) is True
    assert _released(store)


@pytest.mark.parametrize("draft", ["", "my next question"])
def test_a_discarded_turn_does_not_come_back_as_the_draft(draft):
    """Cancelling writes the sent text into the draft; the release undoes it."""
    controller, store = _paused(ConsolePreparationPauseKind.RETRIEVAL)
    store.set_session_draft(SESSION, draft)

    assert release_unsent_turn_preparation(controller, SESSION, PREPARATION) is True
    assert store.session_draft(SESSION) == draft


@pytest.mark.parametrize(
    "kind",
    [
        ConsolePreparationPauseKind.TRACE_CALL,
        ConsolePreparationPauseKind.TRACE_PROVENANCE,
        ConsolePreparationPauseKind.TEMPORARY_CAPTURE,
        ConsolePreparationPauseKind.CONTEXT_COMPACTION,
    ],
)
def test_pauses_with_their_own_recovery_are_left_to_it(kind):
    controller, store = _paused(kind)

    assert release_unsent_turn_preparation(controller, SESSION, PREPARATION) is False
    assert store.preparation_for_session(SESSION).pause_kind is kind


def test_another_preparation_is_never_released():
    controller, store = _paused(ConsolePreparationPauseKind.DESTINATION_CHANGED)

    assert release_unsent_turn_preparation(controller, SESSION, "other") is False
    assert store.preparation_for_session(SESSION).state is (
        ConsoleTurnPreparationState.PAUSED
    )
