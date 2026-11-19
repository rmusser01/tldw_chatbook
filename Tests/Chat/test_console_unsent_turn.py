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
from tldw_chatbook.Chat.console_unsent_turn import (
    card_action_finished,
    card_holds_refusal,
    clear_answered_draft,
    release_unsent_turn_preparation,
)

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


def test_a_release_that_fails_still_keeps_the_composers_draft(monkeypatch):
    """Dev's TASK-35992 guarantee, kept through the rebase onto 7700c63a68.

    Dev's shelf release put the composer's draft back in a ``finally``: a
    cancel that wrote the sent text into the draft and then failed in storage
    must not leave that text behind. The failure itself still propagates, so
    the shelf keeps the recovery (dev's
    ``test_recovery_storage_failure_is_nonfatal_and_retains_inputs_in_worker``).
    """
    controller, store = _paused(ConsolePreparationPauseKind.RETRIEVAL)
    store.set_session_draft(SESSION, "my next question")

    def failing_cancel(preparation_id):
        store.set_session_draft(SESSION, "the sent text")
        raise RuntimeError("storage failed")

    monkeypatch.setattr(controller, "cancel_library_preparation", failing_cancel)

    with pytest.raises(RuntimeError, match="storage failed"):
        release_unsent_turn_preparation(controller, SESSION, PREPARATION)
    assert store.session_draft(SESSION) == "my next question"


@pytest.mark.parametrize(
    ("kind", "held"),
    [
        (ConsolePreparationPauseKind.TEMPORARY_CAPTURE, True),
        (ConsolePreparationPauseKind.TRACE_CALL, True),
        (ConsolePreparationPauseKind.TRACE_PROVENANCE, True),
        # TASK-34350's hold records no unsent turn at all; its card gives the
        # text back itself.
        (ConsolePreparationPauseKind.CONTEXT_COMPACTION, False),
        # The shelf's own pauses stay offered there (TASK-33621.20).
        (ConsolePreparationPauseKind.RETRIEVAL, False),
        (ConsolePreparationPauseKind.DESTINATION_CHANGED, False),
    ],
)
def test_only_a_pause_its_card_ends_keeps_its_unsent_turn_off_the_shelf(kind, held):
    """TASK-33621.18: the turn is recorded either way; held only for the card."""
    controller, _store = _paused(kind)

    assert card_holds_refusal(controller, PREPARATION) is held
    assert card_holds_refusal(controller, "other") is False
    assert card_holds_refusal(controller, None) is False


class _Shelf:
    """The runtime's unsent-turn calls a settled card action makes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def forget_turn_recoveries_for_preparation(self, _preparation_id):
        self.calls.append("forget")

    def return_turn_recovery_to_draft(self, _preparation_id, **_kwargs):
        self.calls.append("return")

    def unhold_turn_recoveries_for_preparation(self, _preparation_id):
        self.calls.append("unhold")


@pytest.mark.parametrize("interrupted", [False, True])
def test_a_capture_pause_stays_on_its_card_after_an_interrupted_action(interrupted):
    """The card is drawn from the pause, so the shelf never takes the turn."""
    controller, _store = _paused(ConsolePreparationPauseKind.TEMPORARY_CAPTURE)
    shelf = _Shelf()

    card_action_finished(
        shelf,
        controller,
        PREPARATION,
        accepted=False,
        return_to_composer=False,
        interrupted=interrupted,
    )

    assert shelf.calls == []


def test_a_library_pause_goes_back_to_the_shelf_after_an_interrupted_action():
    """TASK-33621.20 unchanged: its card may be gone; the shelf offers it."""
    controller, _store = _paused(ConsolePreparationPauseKind.RETRIEVAL)
    shelf = _Shelf()

    card_action_finished(
        shelf,
        controller,
        PREPARATION,
        accepted=False,
        return_to_composer=False,
        interrupted=True,
    )

    assert shelf.calls == ["unhold"]


@pytest.mark.parametrize(
    ("shown", "stored", "sent", "after"),
    [
        # AC#6: the answered prompt goes from the composer and the draft.
        ("Answer me once", "Answer me once", "Answer me once", ("", "")),
        ("Answer me once  ", "Answer me once  ", "Answer me once", ("", "")),
        # A follow-up typed while the card acted is the user's.
        ("And a follow-up", "And a follow-up", "Answer me once", None),
        # The composer shows something other than what was sent (an expanded
        # or prefilled send): never erase text that is not the sent prompt.
        ("/summarize", "/summarize", "Summarize this chat.", None),
        # An attachment-only send answers no prompt.
        ("", "", "", None),
        # Only the composer still held it: the stored draft is left alone.
        ("Answer me once", "something else", "Answer me once", None),
    ],
)
def test_a_card_sent_turn_clears_only_its_own_answered_prompt(
    shown, stored, sent, after
):
    _controller, store = _paused(ConsolePreparationPauseKind.TEMPORARY_CAPTURE)
    store.set_session_draft(SESSION, shown)
    inputs = store.session_input_snapshot(SESSION)
    store.set_session_draft(SESSION, stored)
    cleared = clear_answered_draft(store, SESSION, sent, inputs=inputs)
    expected = bool(after) and shown == stored
    assert cleared is expected
    assert store.session_draft(SESSION) == ("" if expected else stored)
