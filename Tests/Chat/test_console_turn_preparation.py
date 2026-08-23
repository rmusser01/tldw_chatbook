"""Pure Console turn-preparation state and action contracts (Task 12)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    CONSOLE_PREPARATION_DRAFT_MAX_BYTES,
    CONSOLE_PREPARATION_ID_COLLECTION_MAX_ITEMS,
    CONSOLE_PREPARATION_TITLE_MAX_BYTES,
    PAUSE_ACTIONS,
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationValidationError,
    ConsoleTurnPreparationState,
    apply_preparation_transition,
    initial_preparation_state,
    preparation_actions,
)


def _execution_context(
    *,
    session_id: str = "session-1",
    attempt_id: str = "attempt-1",
    auto_retrieve: ConsoleAutoRetrieve = ConsoleAutoRetrieve.AUTOMATIC,
    capabilities: dict[str, object] | None = None,
) -> ConsoleTurnExecutionContext:
    authority = ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=auto_retrieve,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=1,
            source="durable",
        ),
        direct_library_tools=True,
        source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent("openai", "model-1", None),
        attempt_id=attempt_id,
    )
    return ConsoleTurnExecutionContext(
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=ConsoleProviderSelection(
                provider="openai",
                explicit_model="model-1",
            ),
            capabilities=capabilities,
        ),
        library_authority=authority,
        resolved_destination=ConsoleResolvedDestination(
            provider="openai",
            model="model-1",
            endpoint_identity="https://api.openai.com",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )


def _preparation_values(
    state: ConsoleTurnPreparationState = ConsoleTurnPreparationState.PREPARING,
    *,
    pause_kind: ConsolePreparationPauseKind | None = None,
    attempt_id: str = "attempt-1",
    bypass: bool = False,
    session_id: str = "session-1",
    auto_retrieve: ConsoleAutoRetrieve = ConsoleAutoRetrieve.AUTOMATIC,
    execution_context: ConsoleTurnExecutionContext | None = None,
) -> dict[str, object]:
    return {
        "preparation_id": "preparation-1",
        "attempt_id": attempt_id,
        "session_id": session_id,
        "origin": "manual",
        "queue_entry_id": None,
        "executed_draft": "exact draft",
        "execution_context": execution_context
        or _execution_context(
            session_id=session_id,
            attempt_id=attempt_id,
            auto_retrieve=auto_retrieve,
        ),
        "transient_user_message_id": "transient-user",
        "attachment_ids": ("attachment-1",),
        "evidence_ids": ("evidence-1",),
        "prefill_id": "prefill-1",
        "queue_generation": None,
        "pre_send_title": "New conversation",
        "pre_send_conversation_id": None,
        "state": state,
        "pause_kind": pause_kind,
        "one_shot_bypass": bypass,
        "ephemeral": False,
    }


def _preparation(
    state: ConsoleTurnPreparationState = ConsoleTurnPreparationState.PREPARING,
    *,
    pause_kind: ConsolePreparationPauseKind | None = None,
    attempt_id: str = "attempt-1",
    bypass: bool = False,
) -> ConsoleTurnPreparation:
    return ConsoleTurnPreparation(  # type: ignore[arg-type]
        **_preparation_values(
            state,
            pause_kind=pause_kind,
            attempt_id=attempt_id,
            bypass=bypass,
        )
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
    (
        "current",
        "transition",
        "want_state",
        "want_pause",
        "want_attempt",
        "want_bypass",
    ),
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
def test_cancel_is_ignored_after_commit_starts(
    state: ConsoleTurnPreparationState,
) -> None:
    current = _preparation(state)

    assert (
        apply_preparation_transition(
            current,
            _transition(state, ConsoleTurnPreparationState.CANCELLED),
        )
        is current
    )


def test_wrong_preparation_id_and_wrong_expected_state_fail_closed() -> None:
    current = _preparation()

    assert (
        apply_preparation_transition(
            current,
            _transition(
                ConsoleTurnPreparationState.PREPARING,
                ConsoleTurnPreparationState.READY,
                preparation_id="other-preparation",
            ),
        )
        is current
    )
    assert (
        apply_preparation_transition(
            current,
            _transition(
                ConsoleTurnPreparationState.READY,
                ConsoleTurnPreparationState.COMMITTING,
            ),
        )
        is current
    )


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

    assert (
        apply_preparation_transition(
            current,
            _transition(
                ConsoleTurnPreparationState.PAUSED,
                ConsoleTurnPreparationState.PREPARING,
                new_attempt_id="attempt-2",
            ),
        )
        is current
    )


@pytest.mark.parametrize(
    "new_attempt_id",
    ["", " ", "attempt 2", "a" * 201],
)
def test_transition_constructor_rejects_malformed_attempt_id(
    new_attempt_id: str,
) -> None:
    with pytest.raises(ConsoleTurnPreparationValidationError):
        _transition(
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.PREPARING,
            new_attempt_id=new_attempt_id,
        )


def test_retrieval_retry_requires_a_distinct_attempt_id() -> None:
    current = _preparation(
        ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )

    assert (
        apply_preparation_transition(
            current,
            _transition(
                ConsoleTurnPreparationState.PAUSED,
                ConsoleTurnPreparationState.PREPARING,
                new_attempt_id="attempt-1",
            ),
        )
        is current
    )


def test_paused_state_without_a_reason_exposes_no_actions_or_transition() -> None:
    corrupt = _preparation()
    object.__setattr__(corrupt, "state", ConsoleTurnPreparationState.PAUSED)

    assert preparation_actions(corrupt) == ()
    assert (
        apply_preparation_transition(
            corrupt,
            _transition(
                ConsoleTurnPreparationState.PAUSED,
                ConsoleTurnPreparationState.CANCELLED,
            ),
        )
        is corrupt
    )


def test_nonpaused_state_with_a_pause_reason_fails_closed() -> None:
    corrupt = _preparation(ConsoleTurnPreparationState.READY)
    object.__setattr__(
        corrupt,
        "pause_kind",
        ConsolePreparationPauseKind.RETRIEVAL,
    )

    assert preparation_actions(corrupt) == ()
    assert (
        apply_preparation_transition(
            corrupt,
            _transition(
                ConsoleTurnPreparationState.READY,
                ConsoleTurnPreparationState.COMMITTING,
            ),
        )
        is corrupt
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("preparation_id", "bad id"),
        ("attempt_id", ""),
        ("session_id", "s" * 201),
        ("transient_user_message_id", ""),
        ("attachment_ids", ["attachment-1"]),
        ("attachment_ids", (["mutable-element"],)),
        ("attachment_ids", ("bad attachment",)),
        ("attachment_ids", ("attachment-1", "attachment-1")),
        (
            "attachment_ids",
            tuple(
                f"attachment-{index}"
                for index in range(CONSOLE_PREPARATION_ID_COLLECTION_MAX_ITEMS + 1)
            ),
        ),
        ("evidence_ids", ["evidence-1"]),
        ("evidence_ids", ("bad evidence",)),
        ("prefill_id", "bad prefill"),
        ("pre_send_conversation_id", "bad conversation"),
        ("executed_draft", 7),
        ("executed_draft", ""),
        ("executed_draft", "   "),
        ("executed_draft", "x" * (CONSOLE_PREPARATION_DRAFT_MAX_BYTES + 1)),
        ("executed_draft", "\ud800"),
        ("pre_send_title", 7),
        ("pre_send_title", ""),
        ("pre_send_title", "x" * (CONSOLE_PREPARATION_TITLE_MAX_BYTES + 1)),
        ("pre_send_title", "\ud800"),
    ],
    ids=(
        "preparation-id-grammar",
        "attempt-id-empty",
        "session-id-too-long",
        "transient-id-empty",
        "attachment-container-list",
        "attachment-mutable-element",
        "attachment-id-grammar",
        "attachment-duplicate",
        "attachment-count-too-large",
        "evidence-container-list",
        "evidence-id-grammar",
        "prefill-id-grammar",
        "conversation-id-grammar",
        "draft-non-string",
        "draft-empty",
        "draft-whitespace",
        "draft-too-large",
        "draft-surrogate",
        "title-non-string",
        "title-empty",
        "title-too-large",
        "title-surrogate",
    ),
)
def test_constructor_rejects_malformed_or_unbounded_fields(
    field: str,
    value: object,
) -> None:
    values = _preparation_values()
    values[field] = value

    with pytest.raises(
        ConsoleTurnPreparationValidationError,
        match="Invalid Console turn preparation",
    ):
        ConsoleTurnPreparation(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"origin": "MANUAL"},
        {"origin": 1},
        {"queue_entry_id": "queue-1"},
        {"queue_generation": 0},
        {"origin": "queued", "queue_entry_id": None, "queue_generation": 0},
        {"origin": "queued", "queue_entry_id": "queue-1", "queue_generation": None},
        {"origin": "queued", "queue_entry_id": "queue-1", "queue_generation": True},
        {"origin": "queued", "queue_entry_id": "queue-1", "queue_generation": -1},
        {"origin": "queued", "queue_entry_id": "bad queue", "queue_generation": 0},
    ],
)
def test_constructor_rejects_inconsistent_origin_and_queue_authority(
    overrides: dict[str, object],
) -> None:
    values = _preparation_values()
    values.update(overrides)

    with pytest.raises(ConsoleTurnPreparationValidationError):
        ConsoleTurnPreparation(**values)  # type: ignore[arg-type]


def test_constructor_accepts_exact_queued_authority() -> None:
    values = _preparation_values()
    values.update(
        origin="queued",
        queue_entry_id="queue-1",
        queue_generation=0,
    )

    preparation = ConsoleTurnPreparation(**values)  # type: ignore[arg-type]

    assert preparation.origin == "queued"
    assert preparation.queue_entry_id == "queue-1"
    assert preparation.queue_generation == 0


def test_constructor_accepts_exact_text_and_collection_bounds() -> None:
    values = _preparation_values()
    values.update(
        executed_draft="d" * CONSOLE_PREPARATION_DRAFT_MAX_BYTES,
        pre_send_title="t" * CONSOLE_PREPARATION_TITLE_MAX_BYTES,
        attachment_ids=tuple(
            f"attachment-{index}"
            for index in range(CONSOLE_PREPARATION_ID_COLLECTION_MAX_ITEMS)
        ),
    )

    preparation = ConsoleTurnPreparation(**values)  # type: ignore[arg-type]

    assert len(preparation.executed_draft) == CONSOLE_PREPARATION_DRAFT_MAX_BYTES
    assert len(preparation.pre_send_title) == CONSOLE_PREPARATION_TITLE_MAX_BYTES
    assert (
        len(preparation.attachment_ids) == CONSOLE_PREPARATION_ID_COLLECTION_MAX_ITEMS
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"state": "ready"},
        {"pause_kind": "retrieval"},
        {"state": ConsoleTurnPreparationState.PAUSED, "pause_kind": None},
        {
            "state": ConsoleTurnPreparationState.READY,
            "pause_kind": ConsolePreparationPauseKind.RETRIEVAL,
        },
        {"one_shot_bypass": 1},
        {"one_shot_bypass": "false"},
        {"ephemeral": 0},
        {"ephemeral": None},
    ],
)
def test_constructor_rejects_malformed_state_pause_or_boolean_shapes(
    overrides: dict[str, object],
) -> None:
    values = _preparation_values()
    values.update(overrides)

    with pytest.raises(ConsoleTurnPreparationValidationError):
        ConsoleTurnPreparation(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("auto_retrieve", "state", "pause_kind", "bypass"),
    [
        (ConsoleAutoRetrieve.NEVER, ConsoleTurnPreparationState.PREPARING, None, False),
        (
            ConsoleAutoRetrieve.NEVER,
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            False,
        ),
        (ConsoleAutoRetrieve.NEVER, ConsoleTurnPreparationState.READY, None, True),
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleTurnPreparationState.PREPARING,
            None,
            True,
        ),
        (
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            True,
        ),
    ],
)
def test_constructor_rejects_impossible_policy_state_and_bypass_combinations(
    auto_retrieve: ConsoleAutoRetrieve,
    state: ConsoleTurnPreparationState,
    pause_kind: ConsolePreparationPauseKind | None,
    bypass: bool,
) -> None:
    values = _preparation_values(
        state,
        pause_kind=pause_kind,
        bypass=bypass,
        auto_retrieve=auto_retrieve,
    )

    with pytest.raises(ConsoleTurnPreparationValidationError):
        ConsoleTurnPreparation(**values)  # type: ignore[arg-type]


def test_constructor_accepts_automatic_bypass_after_retrieval_pause() -> None:
    preparation = ConsoleTurnPreparation(  # type: ignore[arg-type]
        **_preparation_values(
            ConsoleTurnPreparationState.READY,
            bypass=True,
        )
    )

    assert preparation.one_shot_bypass is True


def test_constructor_requires_complete_matching_execution_authority() -> None:
    base = _execution_context()
    malformed_destination = ConsoleTurnExecutionContext(
        configuration=base.configuration,
        library_authority=base.library_authority,
        resolved_destination=ConsoleResolvedDestination(
            provider="openai",
            model="model-1",
            endpoint_identity="https://api.openai.com?token=private-canary",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )
    for execution_context in (
        object(),
        _execution_context(session_id="other-session"),
        _execution_context(attempt_id="other-attempt"),
        malformed_destination,
    ):
        values = _preparation_values()
        values["execution_context"] = execution_context
        with pytest.raises(ConsoleTurnPreparationValidationError) as caught:
            ConsoleTurnPreparation(**values)  # type: ignore[arg-type]
        assert "private-canary" not in str(caught.value)


def test_validation_error_is_bounded_and_does_not_echo_draft() -> None:
    values = _preparation_values()
    values["executed_draft"] = "private-canary" * (
        CONSOLE_PREPARATION_DRAFT_MAX_BYTES // len("private-canary") + 1
    )

    with pytest.raises(ConsoleTurnPreparationValidationError) as caught:
        ConsoleTurnPreparation(**values)  # type: ignore[arg-type]

    assert "private-canary" not in str(caught.value)
    assert len(str(caught.value)) <= 80


def test_preparation_retains_task8_context_deep_freeze() -> None:
    mutable_formats = ["image/png"]
    execution_context = _execution_context(
        capabilities={"formats": mutable_formats},
    )
    preparation = ConsoleTurnPreparation(  # type: ignore[arg-type]
        **_preparation_values(execution_context=execution_context)
    )

    mutable_formats.append("image/jpeg")

    assert preparation.execution_context.capabilities["formats"] == ("image/png",)
    with pytest.raises(TypeError):
        preparation.execution_context.capabilities["formats"] = ()


def test_retry_updates_preparation_and_frozen_authority_attempt_together() -> None:
    paused = _preparation(
        ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
    )

    retried = apply_preparation_transition(
        paused,
        _transition(
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.PREPARING,
            new_attempt_id="attempt-2",
        ),
    )

    assert retried.attempt_id == "attempt-2"
    assert retried.execution_context.library_authority.attempt_id == "attempt-2"


def test_postconstruction_corruption_exposes_no_action_or_transition() -> None:
    corrupt = _preparation()
    object.__setattr__(corrupt, "execution_context", object())

    assert preparation_actions(corrupt) == ()
    assert (
        apply_preparation_transition(
            corrupt,
            _transition(
                ConsoleTurnPreparationState.PREPARING,
                ConsoleTurnPreparationState.READY,
            ),
        )
        is corrupt
    )


_SOURCE_SHAPES = (
    *(
        (state, None)
        for state in ConsoleTurnPreparationState
        if state is not ConsoleTurnPreparationState.PAUSED
    ),
    *(
        (ConsoleTurnPreparationState.PAUSED, pause)
        for pause in ConsolePreparationPauseKind
    ),
)
_TRANSITION_PAUSE_SHAPES = (None, *tuple(ConsolePreparationPauseKind))
_NEW_ATTEMPT_SHAPES = (None, "attempt-2")
_LEGAL_TRANSITION_SHAPES = frozenset(
    {
        (
            ConsoleTurnPreparationState.PREPARING,
            None,
            ConsoleTurnPreparationState.READY,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PREPARING,
            None,
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            False,
        ),
        (
            ConsoleTurnPreparationState.PREPARING,
            None,
            ConsoleTurnPreparationState.CANCELLED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.READY,
            None,
            ConsoleTurnPreparationState.COMMITTING,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.READY,
            None,
            ConsoleTurnPreparationState.CANCELLED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.COMMITTING,
            None,
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.PERSISTENCE,
            False,
        ),
        (
            ConsoleTurnPreparationState.COMMITTING,
            None,
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.DESTINATION_CHANGED,
            False,
        ),
        (
            ConsoleTurnPreparationState.COMMITTING,
            None,
            ConsoleTurnPreparationState.ACCEPTED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.ACCEPTED,
            None,
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.ACCEPTED,
            None,
            ConsoleTurnPreparationState.SETTLED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            None,
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            None,
            True,
        ),
        (
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            None,
            ConsoleTurnPreparationState.DISPATCHED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            None,
            ConsoleTurnPreparationState.SETTLED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.DISPATCHED,
            None,
            ConsoleTurnPreparationState.SETTLED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            ConsoleTurnPreparationState.PREPARING,
            None,
            True,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            ConsoleTurnPreparationState.READY,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            ConsoleTurnPreparationState.CANCELLED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.PERSISTENCE,
            ConsoleTurnPreparationState.COMMITTING,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.PERSISTENCE,
            ConsoleTurnPreparationState.CANCELLED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.DESTINATION_CHANGED,
            ConsoleTurnPreparationState.COMMITTING,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.DESTINATION_CHANGED,
            ConsoleTurnPreparationState.CANCELLED,
            None,
            False,
        ),
    }
)
_TRANSITION_MATRIX = tuple(
    product(
        _SOURCE_SHAPES,
        tuple(ConsoleTurnPreparationState),
        _TRANSITION_PAUSE_SHAPES,
        _NEW_ATTEMPT_SHAPES,
    )
)


def test_transition_cartesian_matrix_matches_only_the_frozen_legal_edges() -> None:
    for source_shape, new_state, new_pause, new_attempt_id in _TRANSITION_MATRIX:
        source_state, source_pause = source_shape
        preparation = _preparation(source_state, pause_kind=source_pause)
        transition = _transition(
            source_state,
            new_state,
            pause_kind=new_pause,
            new_attempt_id=new_attempt_id,
        )
        shape = (
            source_state,
            source_pause,
            new_state,
            new_pause,
            new_attempt_id is not None,
        )

        result = apply_preparation_transition(preparation, transition)

        assert (result is not preparation) is (shape in _LEGAL_TRANSITION_SHAPES), shape
        if shape in _LEGAL_TRANSITION_SHAPES:
            assert result.state is new_state, shape
            assert result.pause_kind is (
                new_pause if new_state is ConsoleTurnPreparationState.PAUSED else None
            ), shape
            assert result.execution_context.library_authority.attempt_id == (
                new_attempt_id or "attempt-1"
            ), shape
