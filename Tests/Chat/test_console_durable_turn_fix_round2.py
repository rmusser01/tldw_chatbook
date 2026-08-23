"""Task 14 review fix round 2: retry owners, fingerprints, and queue binding."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    _RagService,
    _row,
)
from Tests.Chat.test_console_durable_turn_acceptance import (
    _database_snapshot,
    _install_failure,
    _ready_store,
    _resume_persistence_retry,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchReconstructability,
    ConsoleEgressClass,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.Chat.console_prompt_queue import (
    ConsolePromptQueueRegistry,
    PromptQueueMode,
    PromptQueuePauseReason,
    QueueMutationStatus,
)
from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain
from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
    QueueGenerationAuthorization,
    _AUTHORIZATION_KEY,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
    preparation_actions,
)
from tldw_chatbook.Chat.library_preparation import (
    LibraryPreparationContribution,
    LibraryPreparationEvent,
)


def _set_auto_policy(store, auto_retrieve: ConsoleAutoRetrieve) -> None:
    store.stage_session_library_policy(
        "session-1",
        ConsoleLibraryPolicyCandidate(
            auto_retrieve=auto_retrieve,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        ),
    )


async def _submit_queued(controller, text: str):
    coordinator = controller.prompt_queue_coordinator
    registry = coordinator.registry
    begun = registry.begin_chain("session-1", context_epoch=0, expected_revision=0)
    admitted = registry.admit(
        "session-1", text=text, expected_revision=begun.snapshot.revision
    )
    assert admitted.entry_id is not None
    claimed = registry.claim_next(
        "session-1", expected_revision=admitted.snapshot.revision
    )
    assert claimed.claim is not None
    coordinator._chains["session-1"] = _PromptChain(current_entry_id=admitted.entry_id)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id="session-1"
    )
    authorization = QueueGenerationAuthorization(
        coordinator, "session-1", _key=_AUTHORIZATION_KEY
    )
    result = await controller.submit_draft(
        text,
        session_id="session-1",
        origin=ConsoleSubmissionOrigin.QUEUED,
        queue_entry_id=admitted.entry_id,
        queue_authorization=authorization,
    )
    if not result.accepted:
        coordinator._return_claim(
            "session-1",
            admitted.entry_id,
            PromptQueuePauseReason.DISPATCH_REFUSED,
        )
    return admitted.entry_id, result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_point", "auto_retrieve", "origin"),
    (
        ("conversation", ConsoleAutoRetrieve.NEVER, "manual"),
        ("policy", ConsoleAutoRetrieve.AUTOMATIC, "manual"),
        ("checkpoint", ConsoleAutoRetrieve.NEVER, "queued"),
        ("commit", ConsoleAutoRetrieve.AUTOMATIC, "queued"),
    ),
)
async def test_real_persistence_retry_reuses_exact_staged_message_owners(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    auto_retrieve: ConsoleAutoRetrieve,
    origin: str,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    _set_auto_policy(store, auto_retrieve)
    if auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC:
        controller.app = SimpleNamespace(
            library_rag_search_service=_RagService(
                result={"runtime_backend": "local", "results": [_row()]}
            )
        )
    persistence = store.persistence
    assert persistence is not None
    original_commit = persistence.commit_durable_turn
    attempted_owners: list[tuple[str, str, str]] = []

    def capture_commit(**kwargs: Any):
        acceptance = kwargs["acceptance"]
        attempted_owners.append(
            (
                acceptance.conversation_id,
                acceptance.user_message_id,
                acceptance.assistant_message_id,
            )
        )
        return original_commit(**kwargs)

    monkeypatch.setattr(persistence, "commit_durable_turn", capture_commit)
    cleanup = _install_failure(db, persistence, failure_point, monkeypatch)

    if origin == "queued":
        _entry_id, first = await _submit_queued(controller, "retry exact owners")
    else:
        first = await controller.submit_draft(
            "retry exact owners", session_id="session-1"
        )

    assert first.accepted is False
    paused = store.preparation_for_session("session-1")
    assert paused is not None, (
        first,
        store.preparation_by_id(first.preparation_id or ""),
        controller.prompt_queue_registry.snapshot("session-1"),
    )
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    assert preparation_actions(paused) == ("retry", "cancel")
    assert gateway.calls == 0
    cleanup()

    retried = await controller.retry_library_preparation(paused.preparation_id)

    assert retried.accepted is True
    assert gateway.calls == 1
    assert len(attempted_owners) == 2
    assert attempted_owners[0] == attempted_owners[1]
    conversation_id, user_message_id, assistant_message_id = attempted_owners[0]
    rows = db.get_messages_for_conversation(conversation_id, limit=20)
    assert [row["id"] for row in rows] == [user_message_id, assistant_message_id]
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints "
            "WHERE preparation_id = ?",
            (paused.preparation_id,),
        )
        .fetchone()[0]
        == 1
    )
    assert store.preparation_for_session("session-1") is None


def test_fingerprint_failure_returns_committing_owner_to_persistence_pause(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db, service, store, _preparation, acceptance = _ready_store(tmp_path)

    def fail_commit(**_kwargs: Any):
        raise RuntimeError("first persistence failure")

    monkeypatch.setattr(service, "commit_durable_turn", fail_commit)
    with pytest.raises(RuntimeError, match="first persistence"):
        store.commit_durable_turn(acceptance)
    paused = store.preparation_by_id(acceptance.preparation_id)
    assert (
        paused is not None
        and paused.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    )

    _resume_persistence_retry(store, paused)
    with pytest.raises(RuntimeError, match="fingerprint"):
        store.commit_durable_turn(replace(acceptance, user_content="forged body"))

    current = store.preparation_by_id(acceptance.preparation_id)
    assert current is not None
    assert current.state is ConsoleTurnPreparationState.PAUSED
    assert current.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    assert preparation_actions(current) == ("retry", "cancel")


def _mutated_acceptance(acceptance, field: str):
    if field == "content":
        return replace(acceptance, user_content="forged private body")
    if field == "parent":
        return replace(acceptance, parent_message_id="forged-parent")
    if field == "attachment":
        attachment = dict(acceptance.attachments[0])
        attachment["data"] = b"forged attachment bytes"
        return replace(
            acceptance, attachments=(attachment, *acceptance.attachments[1:])
        )
    if field == "authority":
        authority = replace(acceptance.frozen_authority, direct_library_tools=False)
        return replace(acceptance, frozen_authority=authority)
    if field == "destination":
        destination = replace(
            acceptance.resolved_destination,
            endpoint_identity="https://changed.invalid",
            egress_class=ConsoleEgressClass.UNKNOWN,
        )
        return replace(acceptance, resolved_destination=destination)
    if field == "reconstructability":
        reconstruction = ConsoleDispatchReconstructability(
            attachments_reconstructable=False,
            evidence_reconstructable=False,
            prefill_reconstructable=False,
            opaque_reference="opaque:changed",
        )
        return replace(acceptance, reconstructability=reconstruction)
    if field == "contribution":
        contribution = LibraryPreparationContribution(
            LibraryPreparationEvent(
                version=1,
                outcome="zero_matches",
                attempt_id="attempt-1",
                result_count=0,
                source_types=("notes",),
            )
        )
        return replace(acceptance, contributions=(contribution,))
    raise AssertionError(field)


@pytest.mark.parametrize(
    "field",
    (
        "content",
        "parent",
        "attachment",
        "authority",
        "destination",
        "reconstructability",
        "contribution",
    ),
)
def test_cached_commit_rejects_each_material_acceptance_mutation(
    tmp_path, field: str
) -> None:
    db, _service, store, _preparation, acceptance = _ready_store(tmp_path)
    committed = store.commit_durable_turn(acceptance)
    before = _database_snapshot(db)

    with pytest.raises(RuntimeError, match="fingerprint"):
        store.commit_durable_turn(_mutated_acceptance(acceptance, field))

    assert _database_snapshot(db) == before
    fingerprint = store.durable_acceptance_fingerprint_for(acceptance.preparation_id)
    assert fingerprint is not None
    assert (
        store.durable_turn_commit_for(
            acceptance.preparation_id, fingerprint=fingerprint
        )
        == committed
    )


@pytest.mark.parametrize("mutation", ("policy", "conversation_kwargs"))
def test_cached_commit_rejects_policy_or_conversation_plan_mutation(
    tmp_path,
    mutation: str,
) -> None:
    db, _service, store, _preparation, acceptance = _ready_store(tmp_path)
    committed = store.commit_durable_turn(acceptance)
    before = _database_snapshot(db)
    session = store.sessions()[0]
    if mutation == "policy":
        _set_auto_policy(store, ConsoleAutoRetrieve.NEVER)
    else:
        session.runtime_backend = "forged-runtime"

    with pytest.raises(RuntimeError, match="fingerprint"):
        store.commit_durable_turn(acceptance)

    assert _database_snapshot(db) == before
    assert committed.assistant_message_id == acceptance.assistant_message_id


def test_exact_acceptance_returns_cached_commit_and_fingerprint_retains_no_body(
    tmp_path,
) -> None:
    _db, _service, store, _preparation, acceptance = _ready_store(tmp_path)

    first = store.commit_durable_turn(acceptance)
    second = store.commit_durable_turn(acceptance)
    fingerprint = store.durable_acceptance_fingerprint_for(acceptance.preparation_id)

    assert second == first
    assert fingerprint is not None
    retained = repr(fingerprint)
    assert "exact captured draft" not in retained
    assert "first-image" not in retained
    assert "second-image" not in retained
    assert "forged" not in retained


def test_two_thread_in_flight_cache_refusal_cannot_steal_commit_ownership(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, service, store, _preparation, acceptance = _ready_store(tmp_path)
    original_commit = service.commit_durable_turn
    entered = Event()
    release = Event()

    def blocked_commit(**kwargs: Any):
        entered.set()
        assert release.wait(timeout=5)
        return original_commit(**kwargs)

    monkeypatch.setattr(service, "commit_durable_turn", blocked_commit)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.commit_durable_turn, acceptance)
        assert entered.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="already in flight"):
                store.commit_durable_turn(acceptance)
            current = store.preparation_by_id(acceptance.preparation_id)
            assert current is not None
            assert current.state is ConsoleTurnPreparationState.COMMITTING
        finally:
            release.set()
        committed = future.result(timeout=5)

    assert store.commit_durable_turn(acceptance) == committed
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints "
            "WHERE preparation_id = ?",
            (acceptance.preparation_id,),
        )
        .fetchone()[0]
        == 1
    )


def test_unsupported_mutable_contribution_fingerprint_fails_before_sqlite(
    tmp_path,
) -> None:
    class MutableContribution:
        def write(self, **_kwargs: Any) -> None:
            return None

    db, _service, store, _preparation, acceptance = _ready_store(
        tmp_path, contribution=MutableContribution()
    )
    before = _database_snapshot(db)

    with pytest.raises((TypeError, ValueError), match="fingerprint|canonical"):
        store.commit_durable_turn(acceptance)

    assert _database_snapshot(db) == before
    paused = store.preparation_by_id(acceptance.preparation_id)
    assert (
        paused is not None
        and paused.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    )


def test_every_postcommit_cache_api_requires_exact_non_none_fingerprint(
    tmp_path,
) -> None:
    _db, _service, store, _preparation, acceptance = _ready_store(tmp_path)
    store.commit_durable_turn(acceptance)
    fingerprint = store.durable_acceptance_fingerprint_for(acceptance.preparation_id)
    assert fingerprint is not None
    wrong = replace(fingerprint, assistant_message_id="wrong-assistant")

    calls = (
        lambda: store.begin_durable_postcommit_effects(
            preparation_id=acceptance.preparation_id,
            session_id="session-1",
            assistant_message_id=acceptance.assistant_message_id,
        ),
        lambda: store.durable_postcommit_effects_for(acceptance.preparation_id),
        lambda: store.durable_turn_commit_for(acceptance.preparation_id),
        lambda: store.claim_durable_postcommit_effect(
            acceptance.preparation_id, "probe"
        ),
        lambda: store.complete_durable_postcommit_effect(
            acceptance.preparation_id, "probe"
        ),
        lambda: store.abandon_durable_postcommit_effect(
            acceptance.preparation_id, "probe"
        ),
    )
    for call in calls:
        with pytest.raises(TypeError):
            call()

    for supplied in (None, wrong):
        with pytest.raises((TypeError, RuntimeError), match="fingerprint"):
            store.durable_postcommit_effects_for(
                acceptance.preparation_id, fingerprint=supplied
            )
    assert (
        store.durable_postcommit_effects_for(
            acceptance.preparation_id, fingerprint=fingerprint
        )
        is not None
    )


def _claimed_registry():
    registry = ConsolePromptQueueRegistry(
        id_factory=iter(("entry-1", "entry-2")).__next__
    )
    begun = registry.begin_chain("session-1", context_epoch=0, expected_revision=0)
    first = registry.admit(
        "session-1", text="first", expected_revision=begun.snapshot.revision
    )
    second = registry.admit(
        "session-1", text="second", expected_revision=first.snapshot.revision
    )
    claim = registry.claim_next("session-1", expected_revision=second.snapshot.revision)
    assert claim.claim is not None
    return registry


def test_claim_binding_rejects_forged_ack_then_correct_owner_settles() -> None:
    registry = _claimed_registry()
    bound = registry.bind_claimed_preparation(
        "session-1", entry_id="entry-1", preparation_id="preparation-1"
    )
    assert bound.status is QueueMutationStatus.APPLIED

    forged = registry.settle_durable_acceptance(
        "session-1", entry_id="entry-1", preparation_id="forged-preparation"
    )
    assert forged.status is QueueMutationStatus.LOCKED
    unchanged = registry.snapshot("session-1")
    assert unchanged.claimed_count == 1
    assert unchanged.waiting_count == 1

    settled = registry.settle_durable_acceptance(
        "session-1", entry_id="entry-1", preparation_id="preparation-1"
    )
    assert settled.status is QueueMutationStatus.APPLIED
    snapshot = registry.snapshot("session-1")
    assert snapshot.claimed_count == 0
    assert snapshot.waiting_count == 1
    assert snapshot.mode is PromptQueueMode.PAUSED
    repeated = registry.settle_durable_acceptance(
        "session-1", entry_id="entry-1", preparation_id="preparation-1"
    )
    assert repeated.status is QueueMutationStatus.UNCHANGED


def test_return_then_reclaim_clears_and_rebinds_exact_preparation() -> None:
    registry = _claimed_registry()
    registry.bind_claimed_preparation(
        "session-1", entry_id="entry-1", preparation_id="preparation-1"
    )
    before_return = registry.snapshot("session-1")
    returned = registry.return_claim_to_head(
        "session-1",
        entry_id="entry-1",
        reason=PromptQueuePauseReason.DISPATCH_REFUSED,
        expected_revision=before_return.revision,
    )
    assert returned.status is QueueMutationStatus.APPLIED
    reserved = registry.reserve(
        "session-1", expected_revision=returned.snapshot.revision
    )
    resumed = registry.resume("session-1", expected_revision=reserved.snapshot.revision)
    reclaimed = registry.claim_next(
        "session-1", expected_revision=resumed.snapshot.revision
    )
    assert reclaimed.entry_id == "entry-1"
    assert (
        registry.bind_claimed_preparation(
            "session-1", entry_id="entry-1", preparation_id="preparation-2"
        ).status
        is QueueMutationStatus.APPLIED
    )
    assert (
        registry.settle_durable_acceptance(
            "session-1", entry_id="entry-1", preparation_id="preparation-1"
        ).status
        is QueueMutationStatus.LOCKED
    )
    assert (
        registry.settle_durable_acceptance(
            "session-1", entry_id="entry-1", preparation_id="preparation-2"
        ).status
        is QueueMutationStatus.APPLIED
    )


def test_two_claimed_entries_cannot_cross_settle_preparation_bindings() -> None:
    registry = ConsolePromptQueueRegistry(
        id_factory=iter(("entry-a", "entry-b")).__next__
    )
    for session_id, text in (("session-a", "a"), ("session-b", "b")):
        begun = registry.begin_chain(session_id, context_epoch=0, expected_revision=0)
        admitted = registry.admit(
            session_id, text=text, expected_revision=begun.snapshot.revision
        )
        claimed = registry.claim_next(
            session_id, expected_revision=admitted.snapshot.revision
        )
        assert claimed.claim is not None
    registry.bind_claimed_preparation(
        "session-a", entry_id="entry-a", preparation_id="preparation-a"
    )
    registry.bind_claimed_preparation(
        "session-b", entry_id="entry-b", preparation_id="preparation-b"
    )

    assert (
        registry.settle_durable_acceptance(
            "session-a", entry_id="entry-a", preparation_id="preparation-b"
        ).status
        is QueueMutationStatus.LOCKED
    )
    assert (
        registry.settle_durable_acceptance(
            "session-b", entry_id="entry-b", preparation_id="preparation-a"
        ).status
        is QueueMutationStatus.LOCKED
    )
    assert registry.snapshot("session-a").claimed_count == 1
    assert registry.snapshot("session-b").claimed_count == 1
