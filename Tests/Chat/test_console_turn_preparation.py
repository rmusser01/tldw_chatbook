"""Pure Console turn-preparation state and action contracts (Task 12)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_turn_preparation import (
    PAUSE_ACTIONS,
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
    apply_preparation_transition,
    initial_preparation_state,
    preparation_actions,
)


def _preparation(
    state: ConsoleTurnPreparationState = ConsoleTurnPreparationState.PREPARING,
    *,
    pause_kind: ConsolePreparationPauseKind | None = None,
    attempt_id: str = "attempt-1",
    bypass: bool = False,
) -> ConsoleTurnPreparation:
    return ConsoleTurnPreparation(
        preparation_id="preparation-1",
        attempt_id=attempt_id,
        session_id="session-1",
        origin="manual",
        queue_entry_id=None,
        executed_draft="exact draft",
        execution_context=object(),  # type: ignore[arg-type]
        transient_user_message_id="transient-user",
        attachment_ids=("attachment-1",),
        evidence_ids=("evidence-1",),
        prefill_id="prefill-1",
        queue_generation=None,
        pre_send_title="New conversation",
        pre_send_conversation_id=None,
        state=state,
        pause_kind=pause_kind,
        one_shot_bypass=bypass,
        ephemeral=False,
    )


def _transition(
    expected: ConsoleTurnPreparationState,
    new: ConsoleTurnPreparationState,
    *,
    pause_kind: ConsolePreparationPauseKind | None = None,
    new_attempt_id: str | None = None,
    preparation_id: str = "preparation-1",
) -> ConsolePreparationTransition:
    return ConsolePreparationTransition(
        preparation_id=preparation_id,
        expected_state=expected,
        new_state=new,
        pause_kind=pause_kind,
        new_attempt_id=new_attempt_id,
    )


def test_preparation_contract_is_immutable() -> None:
    preparation = _preparation()

    with pytest.raises(FrozenInstanceError):
        preparation.state = ConsoleTurnPreparationState.READY  # type: ignore[misc]


def test_pause_actions_are_the_exact_frozen_data_matrix() -> None:
    assert PAUSE_ACTIONS == {
        ConsolePreparationPauseKind.RETRIEVAL: ("retry", "bypass", "cancel"),
        ConsolePreparationPauseKind.PERSISTENCE: ("retry", "cancel"),
        ConsolePreparationPauseKind.DESTINATION_CHANGED: ("retry", "cancel"),
    }


@pytest.mark.parametrize(
    ("state", "pause_kind", "expected"),
    [
        (ConsoleTurnPreparationState.PREPARING, None, ("cancel",)),
        (ConsoleTurnPreparationState.READY, None, ("cancel",)),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            ("retry", "bypass", "cancel"),
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.PERSISTENCE,
            ("retry", "cancel"),
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.DESTINATION_CHANGED,
            ("retry", "cancel"),
        ),
        (ConsoleTurnPreparationState.COMMITTING, None, ()),
        (ConsoleTurnPreparationState.ACCEPTED, None, ()),
        (ConsoleTurnPreparationState.DISPATCH_STARTED, None, ()),
        (ConsoleTurnPreparationState.DISPATCHED, None, ()),
        (ConsoleTurnPreparationState.CANCELLED, None, ()),
        (ConsoleTurnPreparationState.SETTLED, None, ()),
    ],
)
def test_preparation_actions_are_derived_only_from_state_and_pause_kind(
    state: ConsoleTurnPreparationState,
    pause_kind: ConsolePreparationPauseKind | None,
    expected: tuple[str, ...],
) -> None:
    assert preparation_actions(_preparation(state, pause_kind=pause_kind)) == expected


def test_never_enters_ready_without_a_library_preparation_state() -> None:
    assert initial_preparation_state(ConsoleAutoRetrieve.NEVER) is (
        ConsoleTurnPreparationState.READY
    )
    assert initial_preparation_state(ConsoleAutoRetrieve.AUTOMATIC) is (
        ConsoleTurnPreparationState.PREPARING
    )


@pytest.mark.parametrize(
    ("current", "transition", "want_state", "want_pause", "want_attempt", "want_bypass"),
    [
        (
            _preparation(),
            _transition(
                ConsoleTurnPreparationState.PREPARING,
                ConsoleTurnPreparationState.READY,
            ),
            ConsoleTurnPreparationState.READY,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(),
            _transition(
                ConsoleTurnPreparationState.PREPARING,
                ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
            ),
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            "attempt-1",
            False,
        ),
        (
            _preparation(
                ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
            ),
            _transition(
                ConsoleTurnPreparationState.PAUSED,
                ConsoleTurnPreparationState.PREPARING,
                new_attempt_id="attempt-2",
            ),
            ConsoleTurnPreparationState.PREPARING,
            None,
            "attempt-2",
            False,
        ),
        (
            _preparation(
                ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
            ),
            _transition(
                ConsoleTurnPreparationState.PAUSED,
                ConsoleTurnPreparationState.READY,
            ),
            ConsoleTurnPreparationState.READY,
            None,
            "attempt-1",
            True,
        ),
        (
            _preparation(ConsoleTurnPreparationState.READY),
            _transition(
                ConsoleTurnPreparationState.READY,
                ConsoleTurnPreparationState.COMMITTING,
            ),
            ConsoleTurnPreparationState.COMMITTING,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.COMMITTING),
            _transition(
                ConsoleTurnPreparationState.COMMITTING,
                ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.PERSISTENCE,
            ),
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.PERSISTENCE,
            "attempt-1",
            False,
        ),
        (
            _preparation(
                ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.PERSISTENCE,
            ),
            _transition(
                ConsoleTurnPreparationState.PAUSED,
                ConsoleTurnPreparationState.COMMITTING,
            ),
            ConsoleTurnPreparationState.COMMITTING,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.COMMITTING),
            _transition(
                ConsoleTurnPreparationState.COMMITTING,
                ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.DESTINATION_CHANGED,
            ),
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.DESTINATION_CHANGED,
            "attempt-1",
            False,
        ),
        (
            _preparation(
                ConsoleTurnPreparationState.PAUSED,
                pause_kind=ConsolePreparationPauseKind.DESTINATION_CHANGED,
            ),
            _transition(
                ConsoleTurnPreparationState.PAUSED,
                ConsoleTurnPreparationState.COMMITTING,
            ),
            ConsoleTurnPreparationState.COMMITTING,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.COMMITTING),
            _transition(
                ConsoleTurnPreparationState.COMMITTING,
                ConsoleTurnPreparationState.ACCEPTED,
            ),
            ConsoleTurnPreparationState.ACCEPTED,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.ACCEPTED),
            _transition(
                ConsoleTurnPreparationState.ACCEPTED,
                ConsoleTurnPreparationState.DISPATCH_STARTED,
            ),
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.DISPATCH_STARTED),
            _transition(
                ConsoleTurnPreparationState.DISPATCH_STARTED,
                ConsoleTurnPreparationState.DISPATCH_STARTED,
                new_attempt_id="attempt-2",
            ),
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            None,
            "attempt-2",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.DISPATCH_STARTED),
            _transition(
                ConsoleTurnPreparationState.DISPATCH_STARTED,
                ConsoleTurnPreparationState.DISPATCHED,
            ),
            ConsoleTurnPreparationState.DISPATCHED,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.DISPATCHED),
            _transition(
                ConsoleTurnPreparationState.DISPATCHED,
                ConsoleTurnPreparationState.SETTLED,
            ),
            ConsoleTurnPreparationState.SETTLED,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.ACCEPTED),
            _transition(
                ConsoleTurnPreparationState.ACCEPTED,
                ConsoleTurnPreparationState.SETTLED,
            ),
            ConsoleTurnPreparationState.SETTLED,
            None,
            "attempt-1",
            False,
        ),
        (
            _preparation(ConsoleTurnPreparationState.DISPATCH_STARTED),
            _transition(
                ConsoleTurnPreparationState.DISPATCH_STARTED,
                ConsoleTurnPreparationState.SETTLED,
            ),
            ConsoleTurnPreparationState.SETTLED,
            None,
            "attempt-1",
            False,
        ),
    ],
)
def test_every_non_cancel_spec_transition_is_applied(
    current: ConsoleTurnPreparation,
    transition: ConsolePreparationTransition,
    want_state: ConsoleTurnPreparationState,
    want_pause: ConsolePreparationPauseKind | None,
    want_attempt: str,
    want_bypass: bool,
) -> None:
    result = apply_preparation_transition(current, transition)

    assert result.state is want_state
    assert result.pause_kind is want_pause
    assert result.attempt_id == want_attempt
    assert result.one_shot_bypass is want_bypass


@pytest.mark.parametrize(
    ("state", "pause_kind"),
    [
        (ConsoleTurnPreparationState.PREPARING, None),
        (ConsoleTurnPreparationState.READY, None),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.PERSISTENCE,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.DESTINATION_CHANGED,
        ),
    ],
)
def test_cancel_is_legal_only_from_spec_owned_precommit_states(
    state: ConsoleTurnPreparationState,
    pause_kind: ConsolePreparationPauseKind | None,
) -> None:
    current = _preparation(state, pause_kind=pause_kind)
    result = apply_preparation_transition(
        current,
        _transition(state, ConsoleTurnPreparationState.CANCELLED),
    )

    assert result.state is ConsoleTurnPreparationState.CANCELLED


@pytest.mark.parametrize(
    "state",
    [
        ConsoleTurnPreparationState.COMMITTING,
        ConsoleTurnPreparationState.ACCEPTED,
        ConsoleTurnPreparationState.DISPATCH_STARTED,
        ConsoleTurnPreparationState.DISPATCHED,
        ConsoleTurnPreparationState.CANCELLED,
        ConsoleTurnPreparationState.SETTLED,
    ],
)
def test_cancel_is_ignored_after_commit_starts(state: ConsoleTurnPreparationState) -> None:
    current = _preparation(state)

    assert apply_preparation_transition(
        current,
        _transition(state, ConsoleTurnPreparationState.CANCELLED),
    ) is current


@pytest.mark.parametrize(
    "transition",
    [
        _transition(
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.ACCEPTED,
        ),
        _transition(
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.PERSISTENCE,
        ),
        _transition(
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.DESTINATION_CHANGED,
        ),
        _transition(
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
            new_attempt_id="unexpected-attempt",
        ),
    ],
)
def test_illegal_transition_shapes_fail_closed(
    transition: ConsolePreparationTransition,
) -> None:
    current = _preparation()

    assert apply_preparation_transition(current, transition) is current


def test_wrong_preparation_id_and_wrong_expected_state_fail_closed() -> None:
    current = _preparation()

    assert apply_preparation_transition(
        current,
        _transition(
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
            preparation_id="other-preparation",
        ),
    ) is current
    assert apply_preparation_transition(
        current,
        _transition(
            ConsoleTurnPreparationState.READY,
            ConsoleTurnPreparationState.COMMITTING,
        ),
    ) is current


def test_repeated_and_racing_transitions_are_idempotently_ignored() -> None:
    original = _preparation()
    to_ready = _transition(
        ConsoleTurnPreparationState.PREPARING,
        ConsoleTurnPreparationState.READY,
    )
    ready = apply_preparation_transition(original, to_ready)

    assert apply_preparation_transition(ready, to_ready) is ready
    racing_pause = _transition(
        ConsoleTurnPreparationState.PREPARING,
        ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )
    assert apply_preparation_transition(ready, racing_pause) is ready


def test_bypass_is_rejected_for_persistence_and_destination_pauses() -> None:
    for pause_kind in (
        ConsolePreparationPauseKind.PERSISTENCE,
        ConsolePreparationPauseKind.DESTINATION_CHANGED,
    ):
        current = _preparation(
            ConsoleTurnPreparationState.PAUSED,
            pause_kind=pause_kind,
        )
        bypass = _transition(
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.READY,
        )
        assert apply_preparation_transition(current, bypass) is current


def test_destination_retry_cannot_start_a_new_retrieval_attempt() -> None:
    current = _preparation(
        ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.DESTINATION_CHANGED,
    )

    assert apply_preparation_transition(
        current,
        _transition(
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.PREPARING,
            new_attempt_id="attempt-2",
        ),
    ) is current


@pytest.mark.parametrize(
    "new_attempt_id",
    ["", " ", "attempt-1", "attempt 2", "a" * 201],
)
def test_retrieval_retry_requires_a_distinct_bounded_opaque_attempt_id(
    new_attempt_id: str,
) -> None:
    current = _preparation(
        ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )

    assert apply_preparation_transition(
        current,
        _transition(
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.PREPARING,
            new_attempt_id=new_attempt_id,
        ),
    ) is current


def test_paused_state_without_a_reason_exposes_no_actions_or_transition() -> None:
    corrupt = replace(_preparation(), state=ConsoleTurnPreparationState.PAUSED)

    assert preparation_actions(corrupt) == ()
    assert apply_preparation_transition(
        corrupt,
        _transition(
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.CANCELLED,
        ),
    ) is corrupt


def test_nonpaused_state_with_a_pause_reason_fails_closed() -> None:
    corrupt = replace(
        _preparation(ConsoleTurnPreparationState.READY),
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )

    assert preparation_actions(corrupt) == ()
    assert apply_preparation_transition(
        corrupt,
        _transition(
            ConsoleTurnPreparationState.READY,
            ConsoleTurnPreparationState.COMMITTING,
        ),
    ) is corrupt
