from __future__ import annotations

from pathlib import Path

import pytest

import tldw_chatbook.Chat.console_chat_models as recovery_models
from Tests.Chat.test_console_automatic_library_preparation import _StreamingFence
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
    ConsoleDispatchReconstructability,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_prompt_queue import (
    ConsolePromptQueueRegistry,
    PromptQueueEntryPhase,
    PromptQueueMode,
    PromptQueuePauseReason,
    QueueMutationStatus,
)
from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
    ConsolePromptQueueCoordinator,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


PROMOTION_BLOCK_COPY = "Finish or discard the pending turn before saving."


def _symbols():
    required = (
        "ConsoleDispatchRecoveryActionId",
        "ConsoleDispatchRecoveryKind",
        "ConsoleDispatchRecoveryState",
    )
    missing = [name for name in required if not hasattr(recovery_models, name)]
    assert not missing, f"dispatch recovery model is missing: {', '.join(missing)}"
    return tuple(getattr(recovery_models, name) for name in required)


def _authority(*, attempt_id: str = "attempt-1") -> ConsoleTurnLibraryAuthority:
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=None,
            source="temporary",
        ),
        direct_library_tools=False,
        source_types=("notes", "media", "conversations"),
        scope_snapshot=ConsoleLibraryItemScopeSnapshot(
            note_ids=(), media_ids=(), conversations_allowed=True
        ),
        provider_intent=ConsoleProviderIntent(
            provider="llama_cpp",
            model="test-model",
            endpoint="http://127.0.0.1:9099",
        ),
        attempt_id=attempt_id,
    )


def _destination() -> ConsoleResolvedDestination:
    return ConsoleResolvedDestination(
        provider="llama_cpp",
        model="test-model",
        endpoint_identity="http://127.0.0.1:9099",
        egress_class=ConsoleEgressClass.ON_DEVICE,
    )


def _truth() -> ConsoleDispatchReconstructability:
    return ConsoleDispatchReconstructability(
        attachments_reconstructable=True,
        evidence_reconstructable=True,
        prefill_reconstructable=True,
        opaque_reference="opaque:ephemeral-1",
    )


def _accepted_queue() -> tuple[ConsolePromptQueueRegistry, str, str, str, str]:
    registry = ConsolePromptQueueRegistry(
        id_factory=iter(("accepted-entry", "later-entry")).__next__,
        monotonic=iter((1.0, 2.0, 3.0, 4.0)).__next__,
    )
    session_id = "session-1"
    begun = registry.begin_chain(session_id, context_epoch=0, expected_revision=0)
    first = registry.admit(
        session_id,
        text="accepted prompt",
        expected_revision=begun.snapshot.revision,
    )
    second = registry.admit(
        session_id,
        text="later prompt",
        expected_revision=first.snapshot.revision,
    )
    assert first.entry_id == "accepted-entry"
    assert second.entry_id == "later-entry"
    claim = registry.claim_next(session_id, expected_revision=second.snapshot.revision)
    assert claim.entry_id == first.entry_id
    bound = registry.bind_claimed_preparation(
        session_id,
        entry_id=first.entry_id,
        preparation_id="preparation-1",
    )
    assert bound.applied
    settled = registry.settle_durable_acceptance(
        session_id,
        entry_id=first.entry_id,
        preparation_id="preparation-1",
    )
    assert settled.applied
    return registry, session_id, first.entry_id, second.entry_id, "preparation-1"


def _coordinator(registry, submitted):
    async def submit(text, *, session_id, entry_id, authorization):
        assert authorization.session_id == session_id
        submitted.append((entry_id, text))
        coordinator.turn_accepted(
            session_id,
            origin=recovery_models.ConsoleSubmissionOrigin.QUEUED,
            context_epoch=0,
            entry_id=entry_id,
        )
        return type(
            "Result",
            (),
            {
                "accepted": True,
                "terminal_status": ConsoleRunStatus.COMPLETED,
            },
        )()

    coordinator = ConsolePromptQueueCoordinator(
        registry=registry,
        context_epoch=lambda _session_id: 0,
        run_status=lambda _session_id: ConsoleRunStatus.COMPLETED,
        submit_queued=submit,
    )
    return coordinator


def test_precommit_cancel_releases_only_exact_claim_to_pending() -> None:
    registry = ConsolePromptQueueRegistry(
        id_factory=iter(("first", "second")).__next__,
        monotonic=iter((1.0, 2.0, 3.0)).__next__,
    )
    begun = registry.begin_chain("session-1", context_epoch=0, expected_revision=0)
    first = registry.admit(
        "session-1", text="first", expected_revision=begun.snapshot.revision
    )
    second = registry.admit(
        "session-1", text="second", expected_revision=first.snapshot.revision
    )
    claimed = registry.claim_next(
        "session-1", expected_revision=second.snapshot.revision
    )
    returned = registry.return_claim_to_head(
        "session-1",
        entry_id="first",
        reason=PromptQueuePauseReason.DISPATCH_REFUSED,
        expected_revision=claimed.snapshot.revision,
    )

    assert returned.applied
    assert [entry.entry_id for entry in returned.snapshot.entries] == [
        "first",
        "second",
    ]
    assert all(
        entry.phase is PromptQueueEntryPhase.WAITING
        for entry in returned.snapshot.entries
    )


def test_postcommit_hydration_keeps_accepted_entry_retired_and_later_work_paused() -> (
    None
):
    _symbols()
    registry, session_id, accepted_id, later_id, preparation_id = _accepted_queue()
    coordinator = _coordinator(registry, [])

    hydrated = coordinator.hydrate_dispatch_recovery(
        session_id,
        queue_entry_id=accepted_id,
        preparation_id=preparation_id,
        checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
    )

    snapshot = registry.snapshot(session_id)
    assert hydrated is True
    assert [entry.entry_id for entry in snapshot.entries] == [later_id]
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert coordinator.dispatch_recovery_blocks_queue(session_id) is True
    assert (
        registry.settle_durable_acceptance(
            session_id,
            entry_id=accepted_id,
            preparation_id=preparation_id,
        ).status
        is QueueMutationStatus.UNCHANGED
    )


@pytest.mark.asyncio
async def test_recovery_hydrates_before_wake_and_refuses_automatic_resume() -> None:
    _symbols()
    registry, session_id, accepted_id, _later_id, preparation_id = _accepted_queue()
    submitted: list[tuple[str, str]] = []
    coordinator = _coordinator(registry, submitted)

    coordinator.hydrate_dispatch_recovery(
        session_id,
        queue_entry_id=accepted_id,
        preparation_id=preparation_id,
        checkpoint_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
    )
    resumed = await coordinator.resume_and_drain(session_id)

    assert resumed.status is QueueMutationStatus.INVALID
    assert submitted == []
    assert registry.snapshot(session_id).mode is PromptQueueMode.PAUSED


@pytest.mark.asyncio
async def test_retry_or_discard_settlement_advances_later_work_exactly_once() -> None:
    _symbols()
    registry, session_id, accepted_id, later_id, preparation_id = _accepted_queue()
    submitted: list[tuple[str, str]] = []
    coordinator = _coordinator(registry, submitted)
    coordinator.hydrate_dispatch_recovery(
        session_id,
        queue_entry_id=accepted_id,
        preparation_id=preparation_id,
        checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
    )

    first = await coordinator.settle_dispatch_recovery_and_drain(
        session_id,
        queue_entry_id=accepted_id,
        preparation_id=preparation_id,
        terminal_status=ConsoleRunStatus.COMPLETED,
    )
    second = await coordinator.settle_dispatch_recovery_and_drain(
        session_id,
        queue_entry_id=accepted_id,
        preparation_id=preparation_id,
        terminal_status=ConsoleRunStatus.COMPLETED,
    )

    assert first.status in {QueueMutationStatus.APPLIED, QueueMutationStatus.UNCHANGED}
    assert second.status is QueueMutationStatus.UNCHANGED
    assert submitted == [(later_id, "later prompt")]
    assert registry.snapshot(session_id).total_count == 0


def test_wrong_recovered_queue_identity_cannot_release_or_advance() -> None:
    _symbols()
    registry, session_id, accepted_id, later_id, preparation_id = _accepted_queue()
    coordinator = _coordinator(registry, [])
    coordinator.hydrate_dispatch_recovery(
        session_id,
        queue_entry_id=accepted_id,
        preparation_id=preparation_id,
        checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
    )

    assert (
        coordinator.clear_dispatch_recovery(
            session_id,
            queue_entry_id="wrong-entry",
            preparation_id=preparation_id,
        )
        is False
    )
    assert coordinator.dispatch_recovery_blocks_queue(session_id) is True
    assert [entry.entry_id for entry in registry.snapshot(session_id).entries] == [
        later_id
    ]


def _ephemeral_store(
    tmp_path: Path,
) -> tuple[CharactersRAGDB, ConsoleChatStore, str]:
    db = CharactersRAGDB(tmp_path / "ephemeral.sqlite", client_id="ephemeral-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(title="Temporary", ephemeral=True)
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        persist=False,
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )
    register = getattr(store, "register_ephemeral_dispatch_recovery", None)
    assert callable(register), "store has no runtime-owned ephemeral recovery"
    register(
        session.id,
        user_message_id=user.id,
        assistant_message_id=assistant.id,
        preparation_id="ephemeral-preparation",
        attempt_id="attempt-1",
        checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
        origin="manual",
        queue_entry_id=None,
        frozen_authority=_authority(),
        resolved_destination=_destination(),
        reconstructability=_truth(),
    )
    return db, store, session.id


def test_ephemeral_recovery_is_store_owned_and_writes_no_checkpoint(
    tmp_path: Path,
) -> None:
    _action_id, kind, _state = _symbols()
    db, store, session_id = _ephemeral_store(tmp_path)

    first = store.dispatch_recovery_for_session(session_id)
    replacement_view = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        agent_runtime_enabled=False,
    )
    second = replacement_view.store.dispatch_recovery_for_session(session_id)

    assert first == second
    assert first.kind is kind.EPHEMERAL_ACCEPTED
    assert [action.label for action in first.actions] == [
        "Retry response",
        "Discard",
    ]
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_ephemeral_send_crosses_started_before_provider_then_settles_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _action_id, kind, _state = _symbols()
    db = CharactersRAGDB(tmp_path / "ephemeral-send.sqlite", "ephemeral-send")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(title="Temporary", ephemeral=True)
    gateway = _StreamingFence()
    original_stream = gateway.stream_chat
    states_seen: list[tuple[object, bool, str]] = []

    async def observe_recovery(*args, **kwargs):
        recovery = store.dispatch_recovery_for_session(session.id)
        assert recovery is not None
        assert recovery.checkpoint is not None
        states_seen.append(
            (recovery.kind, recovery.in_flight, recovery.checkpoint.state.value)
        )
        async for chunk in original_stream(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(gateway, "stream_chat", observe_recovery)
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    result = await controller.submit_draft("hello", session_id=session.id)

    assert result.accepted is True
    assert states_seen == [(kind.EPHEMERAL_DISPATCH_STARTED, True, "dispatch_started")]
    assert store.dispatch_recovery_for_session(session.id) is None
    assert store.messages_for_session(session.id)[-1].status == "complete"
    for table in ("conversations", "messages", "console_dispatch_checkpoints"):
        assert (
            db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            == 0
        )


def test_unresolved_ephemeral_recovery_blocks_promotion_before_any_write(
    tmp_path: Path,
) -> None:
    _symbols()
    db, store, session_id = _ephemeral_store(tmp_path)
    before = {
        table: db.get_connection()
        .execute(f"SELECT COUNT(*) FROM {table}")
        .fetchone()[0]
        for table in ("conversations", "messages", "console_dispatch_checkpoints")
    }

    with pytest.raises(RuntimeError, match=f"^{PROMOTION_BLOCK_COPY}$"):
        store.promote_ephemeral_session(session_id)

    after = {
        table: db.get_connection()
        .execute(f"SELECT COUNT(*) FROM {table}")
        .fetchone()[0]
        for table in before
    }
    assert after == before
    assert store.dispatch_recovery_for_session(session_id) is not None


@pytest.mark.asyncio
async def test_ephemeral_discard_settles_in_memory_without_checkpoint_write(
    tmp_path: Path,
) -> None:
    _symbols()
    db, store, session_id = _ephemeral_store(tmp_path)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        agent_runtime_enabled=False,
    )

    result = await controller.discard_dispatch_recovery(session_id)

    assert result.accepted is True
    assert store.dispatch_recovery_for_session(session_id) is None
    assistant = store.messages_for_session(session_id)[-1]
    assert (assistant.status, assistant.content) == ("discarded", "Response discarded.")
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )


def test_ephemeral_recovery_is_lost_only_with_app_runtime_store(tmp_path: Path) -> None:
    _symbols()
    db, store, session_id = _ephemeral_store(tmp_path)
    assert store.dispatch_recovery_for_session(session_id) is not None

    replacement_runtime = ConsoleChatStore(persistence=ChatPersistenceService(db))

    assert replacement_runtime.sessions() == []
    assert replacement_runtime.dispatch_recovery_for_session(session_id) is None
