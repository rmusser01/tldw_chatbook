"""Pure Console turn-preparation state and action contracts (Task 12)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from itertools import product
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat import console_turn_preparation as turn_preparation
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
    admit_one_shot_capture_off,
    apply_preparation_transition,
    build_console_request_for_preparation,
    initial_preparation_state,
    preparation_actions,
    pause_for_trace_call_failure,
    pause_for_trace_provenance_failure,
)
from tldw_chatbook.Chat.console_trace_service import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    RequestRouteTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceAlignmentError,
    TraceProvenancePersistenceError,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id


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


class _AdmissionDatabase:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.fail_commit = fail_commit
        self.committed: list[str] = []
        self.rolled_back = False

    @contextmanager
    def transaction(self, *, immediate: bool = False):
        assert immediate is True
        staged: list[str] = []
        try:
            yield staged
            if self.fail_commit:
                raise RuntimeError("PRIVATE-COMMIT-FAILURE-CANARY")
        except Exception:
            self.rolled_back = True
            raise
        self.committed.extend(staged)


class _AdmissionCoordinator:
    def __init__(self, *, fail_body: bool = False) -> None:
        self.fail_body = fail_body

    def ensure_current_revision(self, cursor, *, message_id, creation_reason):
        assert creation_reason == "request_capture"
        cursor.append(message_id)
        if self.fail_body:
            raise RuntimeError("PRIVATE-BODY-FAILURE-CANARY")
        return SimpleNamespace(revision_id=new_opaque_id())


def _admit_preparation_trace_provenance(
    preparation: ConsoleTurnPreparation,
    *,
    database: _AdmissionDatabase,
    coordinator: _AdmissionCoordinator,
):
    admit = getattr(
        turn_preparation,
        "admit_preparation_trace_provenance",
        None,
    )
    assert callable(admit), "transaction-owning preparation admission is required"
    return admit(
        preparation,
        database=database,
        coordinator=coordinator,
        message_ids=("message-1",),
    )


def test_trace_revision_admission_wrapper_retries_after_rollback() -> None:
    first_database = _AdmissionDatabase()
    paused, first_descriptors = _admit_preparation_trace_provenance(
        _preparation(ConsoleTurnPreparationState.COMMITTING),
        database=first_database,
        coordinator=_AdmissionCoordinator(fail_body=True),
    )

    retrying = apply_preparation_transition(
        paused,
        _transition(
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.COMMITTING,
        ),
    )
    second_database = _AdmissionDatabase()
    admitted, descriptors = _admit_preparation_trace_provenance(
        retrying,
        database=second_database,
        coordinator=_AdmissionCoordinator(),
    )

    assert first_descriptors == ()
    assert first_database.rolled_back is True
    assert first_database.committed == []
    assert admitted is retrying
    assert len(descriptors) == 1
    assert isinstance(descriptors[0], SavedRevisionTraceProvenance)
    assert second_database.committed == ["message-1"]


def test_trace_revision_commit_failure_rolls_back_and_returns_sanitized_pause() -> None:
    database = _AdmissionDatabase(fail_commit=True)

    paused, descriptors = _admit_preparation_trace_provenance(
        _preparation(ConsoleTurnPreparationState.COMMITTING),
        database=database,
        coordinator=_AdmissionCoordinator(),
    )

    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_PROVENANCE
    assert descriptors == ()
    assert database.rolled_back is True
    assert database.committed == []
    assert "CANARY" not in repr(paused)


def test_trace_revision_admission_failure_fails_autonomous_turn_content_free() -> None:
    values = _preparation_values(ConsoleTurnPreparationState.COMMITTING)
    values.update(
        origin="queued",
        queue_entry_id="queue-1",
        queue_generation=1,
    )
    preparation = ConsoleTurnPreparation(**values)  # type: ignore[arg-type]

    with pytest.raises(TraceProvenancePersistenceError) as raised:
        _admit_preparation_trace_provenance(
            preparation,
            database=_AdmissionDatabase(),
            coordinator=_AdmissionCoordinator(fail_body=True),
        )

    assert str(raised.value) == "trace_provenance_persistence_failed"
    assert "CANARY" not in repr(raised.value)
    assert raised.value.__context__ is None


def test_trace_revision_admission_failure_pauses_committing_without_error_body() -> (
    None
):
    preparation = _preparation(ConsoleTurnPreparationState.COMMITTING)
    failure = TraceProvenancePersistenceError()

    paused = pause_for_trace_provenance_failure(preparation, failure)

    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_PROVENANCE
    assert preparation_actions(paused) == (
        "retry",
        "send_without_capture",
        "cancel",
    )
    assert "content" not in repr(paused).lower()


def test_trace_capture_off_action_creates_a_fresh_capture_off_preparation() -> None:
    paused = pause_for_trace_provenance_failure(
        _preparation(ConsoleTurnPreparationState.COMMITTING),
        TraceProvenancePersistenceError(),
    )

    capture_off = admit_one_shot_capture_off(
        paused,
        new_preparation_id="preparation-2",
        new_attempt_id="attempt-2",
    )

    assert capture_off is not paused
    assert capture_off.preparation_id == "preparation-2"
    assert capture_off.attempt_id == "attempt-2"
    assert capture_off.execution_context.library_authority.attempt_id == "attempt-2"
    assert capture_off.state is ConsoleTurnPreparationState.READY
    assert capture_off.pause_kind is None
    assert capture_off.one_shot_capture_off is True
    capture_mode = getattr(turn_preparation, "ConsoleTraceCaptureMode", None)
    assert capture_mode is not None
    assert capture_off.capture_mode is capture_mode.CAPTURE_OFF
    assert paused.one_shot_capture_off is False


def test_preparation_bridge_rejects_stale_capture_on_state_for_capture_off() -> None:
    paused = pause_for_trace_provenance_failure(
        _preparation(ConsoleTurnPreparationState.COMMITTING),
        TraceProvenancePersistenceError(),
    )
    capture_off = admit_one_shot_capture_off(
        paused,
        new_preparation_id="preparation-2",
        new_attempt_id="attempt-2",
    )
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )

    with pytest.raises(TraceProvenanceAlignmentError, match="Capture Off"):
        build_console_request_for_preparation(
            capture_off,
            [{"role": "user", "content": "fresh attempt"}],
            route=ConsoleRequestRoute.FRESH,
            message_provenance=(SavedRevisionTraceProvenance(new_opaque_id()),),
            memory_provenance=(),
            mandatory_provenance=(),
            tool_provenance=(),
            capture_policy=policy,
        )


def test_preparation_bridge_requires_explicit_capture_on_policy_and_binds_route() -> (
    None
):
    preparation = _preparation(ConsoleTurnPreparationState.READY)

    with pytest.raises(TraceProvenanceAlignmentError, match="capture-on"):
        build_console_request_for_preparation(
            preparation,
            [{"role": "user", "content": "captured"}],
            route=ConsoleRequestRoute.FRESH,
        )

    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )
    request = build_console_request_for_preparation(
        preparation,
        [{"role": "user", "content": "captured"}],
        route=ConsoleRequestRoute.FRESH,
        message_provenance=(SavedRevisionTraceProvenance(new_opaque_id()),),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    assert request.provenance is not None
    route_descriptor = next(
        item
        for item in request.provenance.metadata
        if isinstance(item, RequestRouteTraceProvenance)
    )
    assert route_descriptor.route is ConsoleRequestRoute.FRESH
    assert route_descriptor.predicate == "fresh_submit"


def test_capture_off_attempt_can_enter_ordinary_persistence_pause() -> None:
    paused = pause_for_trace_provenance_failure(
        _preparation(ConsoleTurnPreparationState.COMMITTING),
        TraceProvenancePersistenceError(),
    )
    capture_off = admit_one_shot_capture_off(
        paused,
        new_preparation_id="preparation-2",
        new_attempt_id="attempt-2",
    )
    committing = apply_preparation_transition(
        capture_off,
        ConsolePreparationTransition(
            preparation_id=capture_off.preparation_id,
            expected_state=ConsoleTurnPreparationState.READY,
            new_state=ConsoleTurnPreparationState.COMMITTING,
            pause_kind=None,
            new_attempt_id=None,
        ),
    )
    persistence_paused = apply_preparation_transition(
        committing,
        ConsolePreparationTransition(
            preparation_id=committing.preparation_id,
            expected_state=ConsoleTurnPreparationState.COMMITTING,
            new_state=ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.PERSISTENCE,
            new_attempt_id=None,
        ),
    )

    assert persistence_paused.state is ConsoleTurnPreparationState.PAUSED
    assert persistence_paused.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    assert persistence_paused.one_shot_capture_off is True


def test_capture_off_admission_cannot_create_saved_provenance() -> None:
    paused = pause_for_trace_provenance_failure(
        _preparation(ConsoleTurnPreparationState.COMMITTING),
        TraceProvenancePersistenceError(),
    )
    capture_off = admit_one_shot_capture_off(
        paused,
        new_preparation_id="preparation-2",
        new_attempt_id="attempt-2",
    )
    committing = apply_preparation_transition(
        capture_off,
        ConsolePreparationTransition(
            preparation_id=capture_off.preparation_id,
            expected_state=ConsoleTurnPreparationState.READY,
            new_state=ConsoleTurnPreparationState.COMMITTING,
            pause_kind=None,
            new_attempt_id=None,
        ),
    )

    admitted, descriptors = _admit_preparation_trace_provenance(
        committing,
        database=_AdmissionDatabase(),
        coordinator=_AdmissionCoordinator(fail_body=True),
    )

    assert admitted is committing
    assert descriptors == ()


def test_capture_off_is_consumed_before_a_fresh_dispatch_retry_attempt() -> None:
    paused = pause_for_trace_provenance_failure(
        _preparation(ConsoleTurnPreparationState.COMMITTING),
        TraceProvenancePersistenceError(),
    )
    preparation = admit_one_shot_capture_off(
        paused,
        new_preparation_id="preparation-2",
        new_attempt_id="attempt-2",
    )
    for expected, new in (
        (ConsoleTurnPreparationState.READY, ConsoleTurnPreparationState.COMMITTING),
        (
            ConsoleTurnPreparationState.COMMITTING,
            ConsoleTurnPreparationState.ACCEPTED,
        ),
        (
            ConsoleTurnPreparationState.ACCEPTED,
            ConsoleTurnPreparationState.DISPATCH_STARTED,
        ),
    ):
        preparation = apply_preparation_transition(
            preparation,
            ConsolePreparationTransition(
                preparation_id=preparation.preparation_id,
                expected_state=expected,
                new_state=new,
                pause_kind=None,
                new_attempt_id=None,
            ),
        )

    retried = apply_preparation_transition(
        preparation,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=ConsoleTurnPreparationState.DISPATCH_STARTED,
            new_state=ConsoleTurnPreparationState.DISPATCH_STARTED,
            pause_kind=None,
            new_attempt_id="attempt-3",
        ),
    )

    assert retried.attempt_id == "attempt-3"
    assert retried.capture_mode is turn_preparation.ConsoleTraceCaptureMode.CAPTURE_ON
    assert retried.one_shot_capture_off is False


def test_trace_capture_off_admission_rejects_reused_identity() -> None:
    paused = pause_for_trace_provenance_failure(
        _preparation(ConsoleTurnPreparationState.COMMITTING),
        TraceProvenancePersistenceError(),
    )

    for preparation_id, attempt_id in (
        (paused.preparation_id, "attempt-2"),
        ("preparation-2", paused.attempt_id),
    ):
        assert (
            admit_one_shot_capture_off(
                paused,
                new_preparation_id=preparation_id,
                new_attempt_id=attempt_id,
            )
            is paused
        )


def test_trace_revision_admission_failure_fails_queued_turn_without_pause() -> None:
    values = _preparation_values(ConsoleTurnPreparationState.COMMITTING)
    values.update(
        origin="queued",
        queue_entry_id="queue-1",
        queue_generation=1,
    )
    preparation = ConsoleTurnPreparation(**values)  # type: ignore[arg-type]
    failure = TraceProvenancePersistenceError()

    with pytest.raises(TraceProvenancePersistenceError) as raised:
        pause_for_trace_provenance_failure(preparation, failure)

    assert raised.value is failure
    assert preparation_actions(preparation) == ()


def test_trace_call_failure_pauses_initial_interactive_dispatch_admission() -> None:
    preparation = _preparation(ConsoleTurnPreparationState.ACCEPTED)
    failure = TraceCallPersistenceError()

    paused = pause_for_trace_call_failure(preparation, failure)

    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_CALL
    assert preparation_actions(paused) == (
        "retry",
        "send_without_capture",
        "cancel",
    )
    assert "content" not in repr(paused).lower()


def test_trace_call_send_without_capture_admits_new_run_without_mutating_policy() -> (
    None
):
    paused = pause_for_trace_call_failure(
        _preparation(ConsoleTurnPreparationState.ACCEPTED),
        TraceCallPersistenceError(),
    )

    capture_off = admit_one_shot_capture_off(
        paused,
        new_preparation_id="preparation-2",
        new_attempt_id="attempt-2",
    )

    assert capture_off.preparation_id == "preparation-2"
    assert capture_off.attempt_id == "attempt-2"
    assert (
        capture_off.capture_mode is turn_preparation.ConsoleTraceCaptureMode.CAPTURE_OFF
    )
    assert capture_off.state is ConsoleTurnPreparationState.READY
    assert paused.capture_mode is turn_preparation.ConsoleTraceCaptureMode.CAPTURE_ON
    assert paused.one_shot_capture_off is False


def test_trace_call_failure_fails_queued_run_without_pause() -> None:
    values = _preparation_values(ConsoleTurnPreparationState.ACCEPTED)
    values.update(origin="queued", queue_entry_id="queue-1", queue_generation=1)
    preparation = ConsoleTurnPreparation(**values)  # type: ignore[arg-type]
    failure = TraceCallPersistenceError()

    with pytest.raises(TraceCallPersistenceError) as raised:
        pause_for_trace_call_failure(preparation, failure)

    assert raised.value is failure
    assert preparation.state is ConsoleTurnPreparationState.ACCEPTED


def test_preparation_contract_is_immutable() -> None:
    preparation = _preparation()

    with pytest.raises(FrozenInstanceError):
        preparation.state = ConsoleTurnPreparationState.READY  # type: ignore[misc]


def test_pause_actions_are_the_exact_frozen_data_matrix() -> None:
    assert PAUSE_ACTIONS == {
        ConsolePreparationPauseKind.RETRIEVAL: ("retry", "bypass", "cancel"),
        ConsolePreparationPauseKind.PERSISTENCE: ("retry", "cancel"),
        ConsolePreparationPauseKind.DESTINATION_CHANGED: ("retry", "cancel"),
        ConsolePreparationPauseKind.TRACE_PROVENANCE: (
            "retry",
            "send_without_capture",
            "cancel",
        ),
        ConsolePreparationPauseKind.TRACE_CALL: (
            "retry",
            "send_without_capture",
            "cancel",
        ),
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
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_PROVENANCE,
            ("retry", "send_without_capture", "cancel"),
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
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_PROVENANCE,
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
        ConsolePreparationPauseKind.TRACE_PROVENANCE,
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
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_PROVENANCE,
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
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_CALL,
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
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_CALL,
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
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_PROVENANCE,
            ConsoleTurnPreparationState.COMMITTING,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_PROVENANCE,
            ConsoleTurnPreparationState.CANCELLED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_CALL,
            ConsoleTurnPreparationState.ACCEPTED,
            None,
            False,
        ),
        (
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.TRACE_CALL,
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


def test_dispatch_started_can_pause_before_a_failed_next_trace_call() -> None:
    preparation = _preparation(ConsoleTurnPreparationState.DISPATCH_STARTED)

    paused = apply_preparation_transition(
        preparation,
        _transition(
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.TRACE_CALL,
        ),
    )

    assert paused is not preparation
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_CALL
    assert paused.attempt_id == preparation.attempt_id
