"""Task 14 review fix round 1: ownership, queue, recovery, and retention ratchets."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from typing import Any

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    _PolicyCoordinator,
    _StreamingFence,
    _capture_staged_evidence,
    _real_retrieval_controller_for_launch,
    _staged_evidence_launch,
)
from Tests.Chat.test_console_durable_turn_acceptance import (
    _authority,
    _context,
    _ready_store,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleDispatchSettlementError,
)
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_prompt_queue import (
    ConsolePromptQueueRegistry,
    PromptQueueMode,
    PromptQueuePauseReason,
)
from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
    ConsolePromptQueueCoordinator,
    _PromptChain,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import retrieval as retrieval_module


class _DbNoneWrapper:
    """Delegate a real adapter while reporting no raw DB handle."""

    db = None

    def __init__(self, delegate: ChatPersistenceService) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        if name == "commit_durable_turn":
            raise AttributeError(name)
        return getattr(self._delegate, name)


class _DbNoneAtomicWrapper(_DbNoneWrapper):
    """A db-less wrapper which advertises the required atomic capability."""

    def commit_durable_turn(self, **kwargs: Any):
        return self._delegate.commit_durable_turn(**kwargs)


@pytest.mark.asyncio
async def test_db_none_adapter_without_atomic_capability_fails_closed(
    tmp_path,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    assert isinstance(store.persistence, ChatPersistenceService)
    store.persistence = _DbNoneWrapper(store.persistence)

    result = await controller.submit_draft("must stay unsent", session_id="session-1")

    assert result.accepted is False
    assert gateway.calls == 0
    assert store.messages_for_session("session-1") == []
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    )
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_db_none_adapter_with_atomic_capability_uses_durable_path(
    tmp_path,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    assert isinstance(store.persistence, ChatPersistenceService)
    store.persistence = _DbNoneAtomicWrapper(store.persistence)

    result = await controller.submit_draft("atomic control", session_id="session-1")

    assert result.accepted is True
    assert gateway.calls == 1
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )


def _preparation(session_id: str, preparation_id: str) -> ConsoleTurnPreparation:
    return ConsoleTurnPreparation(
        preparation_id=preparation_id,
        attempt_id="attempt-1",
        session_id=session_id,
        origin="manual",
        queue_entry_id=None,
        executed_draft=f"draft-{session_id}",
        execution_context=_context(session_id, _authority()),
        transient_user_message_id=None,
        attachment_ids=(),
        evidence_ids=(),
        prefill_id=None,
        queue_generation=None,
        pre_send_title="Chat",
        pre_send_conversation_id=None,
        state=ConsoleTurnPreparationState.COMMITTING,
        pause_kind=None,
        one_shot_bypass=False,
        ephemeral=False,
    )


def test_preparation_id_is_globally_unique_under_two_session_thread_race() -> None:
    store = ConsoleChatStore()
    store.create_session(session_id="session-a")
    store.create_session(session_id="session-b")
    first = _preparation("session-a", "shared-preparation")
    second = _preparation("session-b", "shared-preparation")
    barrier = Barrier(2)

    def begin(preparation: ConsoleTurnPreparation):
        barrier.wait()
        return store.begin_preparation(preparation)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(begin, (first, second)))

    winners = tuple(result for result in results if result is not None)
    assert len(winners) == 1
    assert store.preparation_by_id("shared-preparation") is winners[0]
    loser_session = "session-b" if winners[0].session_id == "session-a" else "session-a"
    assert store.preparation_for_session(loser_session) is None

    store.create_session(session_id="session-c")
    separate = (
        _preparation(loser_session, "separate-id-a"),
        _preparation("session-c", "separate-id-b"),
    )
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        separate_results = tuple(executor.map(begin, separate))
    assert all(result is not None for result in separate_results)


def test_cached_commit_and_identity_reject_forged_preparation_reuse(tmp_path) -> None:
    _db, _service, store, _preparation_row, acceptance = _ready_store(tmp_path)
    committed = store.commit_durable_turn(acceptance)
    other = store.create_session(session_id="session-2", title="Chat 2")

    with pytest.raises(RuntimeError, match="owner|fingerprint|changed"):
        store.stage_durable_turn_identity(
            other.id,
            acceptance.preparation_id,
            title="forged title",
        )
    with pytest.raises(RuntimeError, match="owner|fingerprint|changed"):
        store.commit_durable_turn(
            replace(
                acceptance,
                assistant_message_id="forged-assistant",
                attempt_id="forged-attempt",
            )
        )
    fingerprint = store.durable_acceptance_fingerprint_for(acceptance.preparation_id)
    assert fingerprint is not None
    assert (
        store.durable_turn_commit_for(
            acceptance.preparation_id, fingerprint=fingerprint
        )
        == committed
    )


def _queue_coordinator():
    registry = ConsolePromptQueueRegistry(
        id_factory=iter(("accepted-entry", "later-entry")).__next__
    )
    coordinator = ConsolePromptQueueCoordinator(
        registry=registry,
        context_epoch=lambda _session_id: 0,
        run_status=lambda _session_id: ConsoleRunStatus.COMPLETED,
        submit_queued=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
    )
    begun = registry.begin_chain("session-1", context_epoch=0, expected_revision=0)
    first = registry.admit(
        "session-1", text="accepted body", expected_revision=begun.snapshot.revision
    )
    second = registry.admit(
        "session-1", text="later body", expected_revision=first.snapshot.revision
    )
    claim = registry.claim_next("session-1", expected_revision=second.snapshot.revision)
    assert claim.claim is not None
    return registry, coordinator


def test_detached_durable_queue_ack_settles_exact_claim_and_pauses_later() -> None:
    registry, coordinator = _queue_coordinator()
    assert registry.bind_claimed_preparation(
        "session-1",
        entry_id="accepted-entry",
        preparation_id="preparation-1",
    ).applied

    assert coordinator.acknowledge_durable_acceptance(
        "session-1",
        entry_id="accepted-entry",
        preparation_id="preparation-1",
        context_epoch=0,
    )
    snapshot = registry.snapshot("session-1")
    assert snapshot.claimed_count == 0
    assert [entry.entry_id for entry in snapshot.entries] == ["later-entry"]
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.FAILED
    assert coordinator.acknowledge_durable_acceptance(
        "session-1",
        entry_id="accepted-entry",
        preparation_id="preparation-1",
        context_epoch=0,
    )


def test_durable_queue_ack_cannot_settle_a_different_claim() -> None:
    registry, coordinator = _queue_coordinator()
    assert registry.bind_claimed_preparation(
        "session-1",
        entry_id="accepted-entry",
        preparation_id="preparation-1",
    ).applied

    assert not coordinator.acknowledge_durable_acceptance(
        "session-1",
        entry_id="later-entry",
        preparation_id="preparation-1",
        context_epoch=0,
    )
    snapshot = registry.snapshot("session-1")
    assert snapshot.claimed_count == 1
    assert snapshot.entries[0].entry_id == "accepted-entry"


@pytest.mark.asyncio
async def test_explicit_frozen_evidence_makes_checkpoint_unreconstructable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {
        "launch": _staged_evidence_launch("private staged title"),
        "released": [],
    }
    retrieval = _real_retrieval_controller_for_launch(state)
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        _capture_staged_evidence,
    )
    db = CharactersRAGDB(tmp_path / "explicit.sqlite", "task14-fix")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.NEVER)
    session = store.create_session(session_id="session-1")
    gateway = _StreamingFence()
    captured_reconstructability: list[str] = []
    original_stream = gateway.stream_chat

    async def observe_checkpoint(*args: Any, **kwargs: Any):
        row = (
            db.get_connection()
            .execute("SELECT reconstructability_json FROM console_dispatch_checkpoints")
            .fetchone()
        )
        assert row is not None
        captured_reconstructability.append(row["reconstructability_json"])
        async for chunk in original_stream(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(gateway, "stream_chat", observe_checkpoint)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=retrieval._capture_console_staged_rag,
        staged_evidence_provider=lambda _session_id: state["launch"] is not None,
    )

    result = await controller.submit_draft("explicit evidence", session_id=session.id)

    assert result.accepted is True
    assert len(captured_reconstructability) == 1
    payload = json.loads(captured_reconstructability[0])
    assert payload["evidence_reconstructable"] is False
    assert "private staged title" not in captured_reconstructability[0]
    assert "query" not in captured_reconstructability[0]
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )
    assert state["released"]


@pytest.mark.asyncio
async def test_success_cleanup_drops_content_and_bounds_minimal_tombstones(
    tmp_path,
) -> None:
    _db, store, controller, _gateway = _controller(tmp_path)

    for index in range(1000):
        result = await controller.submit_draft(
            f"private prompt {index}", session_id="session-1"
        )
        assert result.accepted is True

    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert store.durable_tombstone_count() <= store.DURABLE_TOMBSTONE_CAP
    retained = store.durable_retention_debug_snapshot()
    assert "private prompt" not in repr(retained)

    assert store.durable_tombstone_count() == store.DURABLE_TOMBSTONE_CAP


@pytest.mark.asyncio
async def test_postcommit_failure_retains_content_until_resume_then_cleans(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db, store, controller, _gateway = _controller(tmp_path)
    original = store.publish_durable_turn_identity
    calls = 0

    def fail_once(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected identity publication")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "publish_durable_turn_identity", fail_once)
    first = await controller.submit_draft(
        "retained private draft", session_id="session-1"
    )
    assert first.accepted is True
    assert store.durable_content_retention_count() > 0
    assert first.preparation_id in controller._durable_postcommit_continuations

    second = await controller.resume_durable_postcommit(first.preparation_id)
    assert second.accepted is True
    assert calls == 2
    assert store.durable_content_retention_count() == 0
    assert controller._durable_postcommit_continuations == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("issue_newer_token", [False, True])
async def test_postcommit_settlement_rollback_uses_issued_generation_token(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    issue_newer_token: bool,
) -> None:
    """Durable resume rollback fences only the provider entry it issued."""

    _db, store, controller, _gateway = _controller(tmp_path)
    original = store.publish_durable_turn_identity
    identity_calls = 0

    def fail_identity_once(*args: Any, **kwargs: Any):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 1:
            raise RuntimeError("injected identity publication")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "publish_durable_turn_identity", fail_identity_once)
    first = await controller.submit_draft(
        "postcommit generation rollback",
        session_id="session-1",
    )
    assert first.preparation_id is not None
    issued: dict[str, int | str | None] = {}

    async def fail_after_token(*_args: Any, **kwargs: Any):
        assistant_id = str(kwargs["assistant_message_id"])
        issued["assistant_id"] = assistant_id
        issued["replacement"] = store.begin_generation_attempt(assistant_id)
        issued["newer"] = (
            store.begin_generation_attempt(assistant_id) if issue_newer_token else None
        )
        raise ConsoleDispatchSettlementError("injected provider settlement")

    monkeypatch.setattr(controller, "_stream_assistant_response", fail_after_token)

    resumed = await controller.resume_durable_postcommit(first.preparation_id)

    assert resumed.accepted is True
    assert resumed.terminal_status is ConsoleRunStatus.BLOCKED
    assistant_id = str(issued["assistant_id"])
    replacement = int(issued["replacement"])
    newer = issued["newer"]
    if newer is None:
        assert not store._generation_attempt_is_current(assistant_id, replacement)
    else:
        assert store._generation_attempt_is_current(assistant_id, int(newer))


@pytest.mark.asyncio
async def test_session_close_drops_unresolved_live_postcommit_content(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)

    def fail_identity(*_args: Any, **_kwargs: Any):
        raise RuntimeError("injected identity publication")

    monkeypatch.setattr(store, "publish_durable_turn_identity", fail_identity)
    result = await controller.submit_draft(
        "close private draft", session_id="session-1"
    )
    assert result.accepted is True
    assert result.preparation_id in controller._durable_postcommit_continuations

    controller.close_session("session-1")

    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 1
    )


_REAL_POSTCOMMIT_SEAMS = (
    "identity_publication",
    "durable_owner_publication",
    "staged_input_clearing",
    "workspace_projection",
    "queue_acknowledgement",
    "accepted_hook",
    "prompt_history",
    "preparation_publication",
    "checkpoint_transition",
    "provider_entry",
)


def _install_real_effect_failure(
    controller: ConsoleChatController,
    store: ConsoleChatStore,
    effect_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    counts = {"attempts": 0, "successes": 0}

    def should_fail() -> bool:
        counts["attempts"] += 1
        if counts["attempts"] == 1:
            return True
        return False

    def wrap_sync(owner: object, attribute: str) -> None:
        original = getattr(owner, attribute)

        def wrapper(*args: Any, **kwargs: Any):
            if should_fail():
                raise RuntimeError(f"injected {effect_name}")
            result = original(*args, **kwargs)
            counts["successes"] += 1
            return result

        monkeypatch.setattr(owner, attribute, wrapper)

    if effect_name == "identity_publication":
        wrap_sync(store, "publish_durable_turn_identity")
    elif effect_name == "durable_owner_publication":
        wrap_sync(store, "publish_durable_turn_owners")
    elif effect_name == "staged_input_clearing":
        wrap_sync(controller, "_release_prepared_evidence")
    elif effect_name == "workspace_projection":
        wrap_sync(store, "_project_workspace_membership_after_commit")
    elif effect_name == "queue_acknowledgement":
        wrap_sync(controller.prompt_queue_coordinator, "turn_accepted")
    elif effect_name == "accepted_hook":

        def accepted_hook() -> None:
            if should_fail():
                raise RuntimeError("injected accepted_hook")
            counts["successes"] += 1

        controller.on_submission_accepted = accepted_hook
    elif effect_name == "prompt_history":
        assert controller.prompt_history is not None
        original_append = controller.prompt_history.append

        async def append(*args: Any, **kwargs: Any):
            if should_fail():
                raise RuntimeError("injected prompt_history")
            result = await original_append(*args, **kwargs)
            counts["successes"] += 1
            return result

        monkeypatch.setattr(controller.prompt_history, "append", append)
    elif effect_name == "preparation_publication":
        original_transition = controller._transition_preparation

        def transition(*args: Any, **kwargs: Any):
            expected = args[1] if len(args) > 1 else kwargs.get("expected_state")
            new = args[2] if len(args) > 2 else kwargs.get("new_state")
            targeted = (
                expected is ConsoleTurnPreparationState.COMMITTING
                and new is ConsoleTurnPreparationState.ACCEPTED
            )
            if targeted and should_fail():
                raise RuntimeError("injected preparation_publication")
            result = original_transition(*args, **kwargs)
            if targeted:
                counts["successes"] += 1
            return result

        monkeypatch.setattr(controller, "_transition_preparation", transition)
    elif effect_name == "checkpoint_transition":
        persistence = store.persistence
        assert persistence is not None
        repository = persistence.console_dispatch_repository
        wrap_sync(repository, "cas_state")
    elif effect_name == "provider_entry":
        original_stream = controller._stream_assistant_response

        async def stream(*args: Any, **kwargs: Any):
            if should_fail():
                raise RuntimeError("injected provider_entry")
            result = await original_stream(*args, **kwargs)
            counts["successes"] += 1
            return result

        monkeypatch.setattr(controller, "_stream_assistant_response", stream)
    else:  # pragma: no cover - closed parameter set
        raise AssertionError(effect_name)
    return counts


@pytest.mark.asyncio
@pytest.mark.parametrize("effect_name", _REAL_POSTCOMMIT_SEAMS)
async def test_real_postcommit_seam_failure_resumes_same_owner_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    effect_name: str,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    counts = _install_real_effect_failure(controller, store, effect_name, monkeypatch)

    first = await controller.submit_draft("exact private body", session_id="session-1")

    assert first.accepted is True
    assert first.preparation_id is not None
    fingerprint = store.durable_acceptance_fingerprint_for(first.preparation_id)
    assert fingerprint is not None
    effects = store.durable_postcommit_effects_for(
        first.preparation_id, fingerprint=fingerprint
    )
    assert effects is not None
    assert effect_name not in effects.completed
    counts_before = tuple(
        db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("conversations", "messages", "console_dispatch_checkpoints")
    )

    second = await controller.resume_durable_postcommit(first.preparation_id)

    assert second.accepted is True
    assert second.user_message_id == first.user_message_id
    assert second.assistant_message_id == first.assistant_message_id
    if effect_name == "provider_entry":
        assert "Retry anyway or Discard" in second.visible_copy
        assert counts == {"attempts": 1, "successes": 0}
        assert gateway.calls == 0
        checkpoint = (
            db.get_connection()
            .execute("SELECT state FROM console_dispatch_checkpoints")
            .fetchone()
        )
        assert checkpoint is not None
        assert checkpoint["state"] == "dispatch_started"

        retried = await controller.retry_dispatch_recovery("session-1")

        assert retried.accepted is True
        assert counts == {"attempts": 2, "successes": 1}
        assert gateway.calls == 1
    else:
        assert counts == {"attempts": 2, "successes": 1}
        assert gateway.calls == 1
    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )
    assert (
        tuple(
            db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("conversations", "messages")
        )
        == counts_before[:2]
    )


@pytest.mark.asyncio
async def test_real_queued_ack_reentry_settles_detached_claim_and_pauses_later(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    coordinator = controller.prompt_queue_coordinator
    registry = coordinator.registry
    begun = registry.begin_chain("session-1", context_epoch=0, expected_revision=0)
    first = registry.admit(
        "session-1",
        text="accepted queued body",
        expected_revision=begun.snapshot.revision,
    )
    second = registry.admit(
        "session-1", text="later queued body", expected_revision=first.snapshot.revision
    )
    assert first.entry_id is not None
    assert second.entry_id is not None
    coordinator._chains["session-1"] = _PromptChain()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id="session-1"
    )
    submitted_results = []
    original_inner = controller._submit_draft_inner

    async def inspect_inner(*args: Any, **kwargs: Any):
        assert kwargs["origin"] is ConsoleSubmissionOrigin.QUEUED
        return await original_inner(*args, **kwargs)

    monkeypatch.setattr(controller, "_submit_draft_inner", inspect_inner)

    async def capture_submit(text: str, **kwargs: Any):
        result = await controller.submit_draft(
            text,
            session_id=kwargs["session_id"],
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id=kwargs["entry_id"],
            queue_authorization=kwargs["authorization"],
        )
        submitted_results.append(result)
        return result

    coordinator._submit_queued = capture_submit
    original_ack = coordinator.acknowledge_durable_acceptance
    attempts = 0

    def fail_once(*args: Any, **kwargs: Any) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected exact queue acknowledgement")
        return original_ack(*args, **kwargs)

    monkeypatch.setattr(coordinator, "acknowledge_durable_acceptance", fail_once)

    await coordinator._drain_waiting("session-1", ConsoleRunStatus.COMPLETED)

    assert submitted_results, "queue drain did not submit its claimed entry"
    assert submitted_results[0].accepted is True, submitted_results[0].visible_copy
    checkpoint = (
        db.get_connection()
        .execute(
            "SELECT preparation_id, user_message_id, assistant_message_id "
            "FROM console_dispatch_checkpoints"
        )
        .fetchone()
    )
    assert checkpoint is not None
    preparation_id = checkpoint["preparation_id"]
    fingerprint = store.durable_acceptance_fingerprint_for(preparation_id)
    assert fingerprint is not None
    effects = store.durable_postcommit_effects_for(
        preparation_id, fingerprint=fingerprint
    )
    assert effects is not None
    assert "queue_acknowledgement" not in effects.completed
    assert "session-1" not in coordinator._chains
    before = registry.snapshot("session-1")
    assert before.claimed_count == 1
    assert before.waiting_count == 1

    resumed = await controller.resume_durable_postcommit(preparation_id)

    assert resumed.accepted is True
    assert resumed.user_message_id == checkpoint["user_message_id"]
    assert resumed.assistant_message_id == checkpoint["assistant_message_id"]
    assert attempts == 2
    assert gateway.calls == 1
    after = registry.snapshot("session-1")
    assert after.claimed_count == 0
    assert [entry.entry_id for entry in after.entries] == [second.entry_id]
    assert after.mode is PromptQueueMode.PAUSED
    assert after.pause_reason is PromptQueuePauseReason.FAILED
    assert coordinator.acknowledge_durable_acceptance(
        "session-1",
        entry_id=first.entry_id,
        preparation_id=preparation_id,
        context_epoch=0,
    )
    assert registry.snapshot("session-1") == after
