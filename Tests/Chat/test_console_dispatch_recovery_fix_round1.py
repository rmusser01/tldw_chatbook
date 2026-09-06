"""Task 15 review fixes: recovery ownership and terminal barriers."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
)

from Tests.Chat.test_console_automatic_library_preparation import _RagService, _row
from Tests.Chat.test_console_dispatch_queue_recovery import _ephemeral_store
from Tests.Chat.test_console_dispatch_recovery import (
    _NoReplayGateway,
    _acceptance,
    _database,
    _insert,
    _restored_store,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
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
from tldw_chatbook.Chat.console_prompt_queue import PromptQueueMode
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain


class _OriginalSettlementGateway:
    """Capture the exact durable owner at provider entry, then end as requested."""

    def __init__(self, db, *, outcome: str) -> None:
        self.db = db
        self.outcome = outcome
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.assistant_before_terminal: tuple[object, ...] | None = None
        self.checkpoint_before_terminal: tuple[object, ...] | None = None

    async def resolve_for_send(self, _selection: object) -> object:
        return SimpleNamespace(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            visible_copy="",
            resolved_destination=ConsoleResolvedDestination(
                provider="llama_cpp",
                model="test-model",
                endpoint_identity="http://127.0.0.1:9099",
                egress_class=ConsoleEgressClass.ON_DEVICE,
            ),
        )

    async def stream_chat(
        self, _resolution: object, _messages: list[dict[str, Any]], **_kwargs: Any
    ):
        self.calls += 1
        connection = self.db.get_connection()
        assistant = connection.execute(
            "SELECT * FROM messages WHERE role = 'assistant' ORDER BY timestamp LIMIT 1"
        ).fetchone()
        assert assistant is not None
        self.assistant_before_terminal = tuple(assistant)
        checkpoint = connection.execute(
            "SELECT * FROM console_dispatch_checkpoints ORDER BY created_at LIMIT 1"
        ).fetchone()
        assert checkpoint is not None
        self.checkpoint_before_terminal = tuple(checkpoint)
        self.started.set()
        if self.outcome == "failure":
            raise RuntimeError("provider failed")
        yield "partial" if self.outcome == "cancel" else "complete"
        if self.outcome == "cancel":
            await self.release.wait()


def _assert_original_owner_rolled_back(db, store, controller) -> None:
    connection = db.get_connection()
    assistant = connection.execute(
        "SELECT content, assistant_generation_state, version, deleted "
        "FROM messages WHERE role = 'assistant'"
    ).fetchone()
    checkpoint = connection.execute(
        "SELECT state, checkpoint_revision FROM console_dispatch_checkpoints"
    ).fetchone()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert tuple(assistant) == ("", "dispatch_started", 2, 0)
    assert tuple(checkpoint) == ("dispatch_started", 2)
    session_id = store.active_session_id
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert (
        recovery.assistant_message_id == store.messages_for_session(session_id)[-1].id
    )
    assert recovery.in_flight is False
    assert recovery.runtime_active is False
    assert recovery.recovery_needed is True
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.BLOCKED


@pytest.mark.asyncio
async def test_restored_source_owner_refuses_fresh_submit_before_echo_or_acceptance(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "fresh-owner.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _NoReplayGateway(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )

    result = await controller.submit_draft("second prompt", session_id=session_id)

    assert result.accepted is False
    assert "pending response" in result.visible_copy.lower()
    assert gateway.resolve_calls == 0
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    )
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 1
    )


@pytest.mark.parametrize("replay_kind", ["exact", "display-name", "contribution"])
def test_repository_rejects_every_existing_conversation_owner(
    tmp_path: Path,
    replay_kind: str,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "repository-owner.sqlite")
    first = replace(
        _acceptance(conversation_id),
        attachments=(
            {
                "position": 0,
                "data": b"attachment",
                "mime_type": "image/png",
                "display_name": "first.png",
            },
        ),
    )
    _insert(db, repository, first)
    second = replace(
        first,
        user_message_id="user-2",
        assistant_message_id="assistant-2",
        preparation_id="preparation-2",
        attempt_id="attempt-2",
        frozen_authority=replace(first.frozen_authority, attempt_id="attempt-2"),
    )

    with pytest.raises(RuntimeError, match="active dispatch checkpoint"):
        _insert(db, repository, second)
    replay = first
    if replay_kind == "display-name":
        replay = replace(
            first,
            attachments=(
                {
                    "position": 0,
                    "data": b"attachment",
                    "mime_type": "image/png",
                    "display_name": "changed.png",
                },
            ),
        )
    elif replay_kind == "contribution":
        replay = replace(first, contributions=(object(),))
    with pytest.raises(RuntimeError, match="active dispatch checkpoint"):
        _insert(db, repository, replay)
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    )
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 1
    )


@pytest.mark.asyncio
async def test_healthy_durable_owner_is_not_recovery_before_checkpoint_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-transition owner is live truth, never a user-visible recovery.

    TASK-22000 (owner decision, 2026-08-24): this test originally also pinned
    ``dispatch_recovery_blocks_submission(...) is True`` for this exact
    healthy window. That half is now wrong -- it disabled Send for the whole
    duration of every live turn, which contradicts ADR-098 / TASK-14808 /
    TASK-15121 (an accepted live turn re-labels Send to "Queue" and admits the
    draft as a FIFO follow-up). What this test is *named* for is unchanged and
    still pinned below: the owner published before the checkpoint transition
    is runtime truth (``runtime_active=True, recovery_needed=False``) and must
    never surface as a recovery card. The genuine blocking contract this file
    protects lives in
    ``test_restored_source_owner_refuses_fresh_submit_before_echo_or_acceptance``
    and ``_assert_original_owner_rolled_back`` -- both unhealthy owners, both
    still refused -- plus
    ``test_unhealthy_recovery_owner_still_blocks_submission_and_a_queued_turn``
    in ``Tests/Chat/test_console_send_gate_queue_race.py``.
    """

    _db, store, controller, _gateway = _controller(tmp_path)
    observed: list[tuple[object, bool, bool, bool]] = []
    publish = store.publish_durable_turn_owners

    def capture_owner_window(*args: object, **kwargs: object):
        result = publish(*args, **kwargs)
        recovery = store.dispatch_recovery_for_session("session-1")
        assert recovery is not None
        observed.append(
            (
                store.dispatch_recovery_for_presentation("session-1"),
                store.dispatch_recovery_blocks_submission("session-1"),
                recovery.runtime_active,
                recovery.recovery_needed,
            )
        )
        return result

    monkeypatch.setattr(store, "publish_durable_turn_owners", capture_owner_window)

    result = await controller.submit_draft("healthy durable", session_id="session-1")

    assert result.accepted is True
    assert observed == [(None, False, True, False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure"])
async def test_original_send_terminal_settlement_failure_restores_blocked_owner(
    tmp_path: Path,
    outcome: str,
) -> None:
    db, store, controller, _old_gateway = _controller(tmp_path)
    gateway = _OriginalSettlementGateway(db, outcome=outcome)
    controller.provider_gateway = gateway
    db.get_connection().execute(
        "CREATE TRIGGER task15_fix_fail_terminal_delete "
        "BEFORE DELETE ON console_dispatch_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'task15 settlement failure'); END"
    )
    db.get_connection().commit()

    result = await controller.submit_draft(
        "first durable prompt", session_id="session-1"
    )

    assert result.accepted is True
    _assert_original_owner_rolled_back(db, store, controller)


@pytest.mark.asyncio
async def test_original_send_cancel_settlement_failure_restores_blocked_owner(
    tmp_path: Path,
) -> None:
    db, store, controller, _old_gateway = _controller(tmp_path)
    gateway = _OriginalSettlementGateway(db, outcome="cancel")
    controller.provider_gateway = gateway
    db.get_connection().execute(
        "CREATE TRIGGER task15_fix_fail_cancel_delete "
        "BEFORE DELETE ON console_dispatch_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'task15 settlement failure'); END"
    )
    db.get_connection().commit()

    task = asyncio.create_task(
        controller.submit_draft("first durable prompt", session_id="session-1")
    )
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    assert controller.stop_active_run(record_user_stop=False) is True
    result = await asyncio.wait_for(task, timeout=1)

    assert result.accepted is True
    _assert_original_owner_rolled_back(db, store, controller)


@pytest.mark.asyncio
async def test_queued_settlement_failure_hydrates_exact_fence_before_return(
    tmp_path: Path,
) -> None:
    db, store, controller, _old_gateway = _controller(tmp_path)
    gateway = _OriginalSettlementGateway(db, outcome="success")
    controller.provider_gateway = gateway
    coordinator = controller.prompt_queue_coordinator
    registry = coordinator.registry
    begun = registry.begin_chain("session-1", context_epoch=0, expected_revision=0)
    first = registry.admit(
        "session-1",
        text="accepted queued prompt",
        expected_revision=begun.snapshot.revision,
    )
    second = registry.admit(
        "session-1",
        text="later queued prompt",
        expected_revision=first.snapshot.revision,
    )
    assert first.entry_id is not None
    assert second.entry_id is not None
    coordinator._chains["session-1"] = _PromptChain()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED),
        session_id="session-1",
    )
    submitted: list[str] = []
    fenced_at_submit_return: list[bool] = []

    async def submit_queued(text: str, **kwargs: Any):
        submitted.append(kwargs["entry_id"])
        result = await controller.submit_draft(
            text,
            session_id=kwargs["session_id"],
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id=kwargs["entry_id"],
            queue_authorization=kwargs["authorization"],
        )
        fenced_at_submit_return.append(
            coordinator.dispatch_recovery_blocks_queue("session-1")
        )
        return result

    coordinator._submit_queued = submit_queued
    db.get_connection().execute(
        "CREATE TRIGGER task15_fix_fail_queued_delete "
        "BEFORE DELETE ON console_dispatch_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'task15 settlement failure'); END"
    )
    db.get_connection().commit()

    await coordinator._drain_waiting("session-1", ConsoleRunStatus.COMPLETED)

    assert submitted == [first.entry_id]
    assert fenced_at_submit_return == [True]
    recovery = store.dispatch_recovery_for_session("session-1")
    assert recovery is not None
    assert recovery.queue_entry_id == first.entry_id
    assert recovery.in_flight is False
    assert recovery.runtime_active is False
    assert recovery.recovery_needed is True
    assert coordinator.dispatch_recovery_blocks_queue("session-1")
    snapshot = registry.snapshot("session-1")
    assert [entry.entry_id for entry in snapshot.entries] == [second.entry_id]
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert controller.run_state_for("session-1").status is ConsoleRunStatus.BLOCKED


@pytest.mark.asyncio
async def test_automatic_accepted_restart_retry_requeries_durable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    controller.base_url = "http://127.0.0.1:9099"
    session = store.sessions()[0]
    session.library_policy_holder.snapshot = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=None,
        source="new_session",
    )
    session.library_policy_holder.explicitly_staged = True
    service = _RagService({"results": [_row()]})
    controller.app = SimpleNamespace(library_rag_search_service=service)
    persistence = store.persistence
    assert persistence is not None

    def fail_before_provider(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("crash before provider")

    monkeypatch.setattr(
        persistence.console_dispatch_repository,
        "cas_state",
        fail_before_provider,
    )
    first = await controller.submit_draft("automatic prompt", session_id=session.id)
    assert first.accepted is True
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None

    restarted, restarted_session_id = _restored_store(db, conversation_id)
    retry_gateway = _OriginalSettlementGateway(db, outcome="success")
    retry = ConsoleChatController(
        store=restarted,
        provider_gateway=retry_gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    retry.app = SimpleNamespace(library_rag_search_service=service)
    recovery = restarted.dispatch_recovery_for_session(restarted_session_id)
    assert recovery is not None
    assert recovery.actions[0].enabled is True

    compared: list[tuple[object, object]] = []
    authority_matches = ConsoleChatController._dispatch_authority_matches

    def capture_authority(current: object, frozen: object) -> bool:
        compared.append((current, frozen))
        return authority_matches(current, frozen)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ConsoleChatController,
        "_dispatch_authority_matches",
        staticmethod(capture_authority),
    )

    result = await retry.retry_dispatch_recovery(restarted_session_id)

    assert result.accepted is True, compared
    assert retry_gateway.calls == 1
    assert len(service.calls) == 2
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    )
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )


def _new_session_authority() -> ConsoleTurnLibraryAuthority:
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=None,
            source="new_session",
        ),
        direct_library_tools=False,
        source_types=("notes", "media", "conversations"),
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent(
            provider="llama_cpp",
            model="test-model",
            endpoint="http://127.0.0.1:9099",
        ),
        attempt_id="attempt-1",
    )


def test_authority_none_allows_only_exact_first_save_transition() -> None:
    frozen = _new_session_authority()
    first_save = replace(
        frozen,
        policy=replace(frozen.policy, source="durable", policy_revision=1),
        attempt_id="attempt-retry",
    )
    later_save = replace(
        first_save,
        policy=replace(first_save.policy, policy_revision=2),
    )
    temporary = replace(
        frozen,
        policy=replace(frozen.policy, source="temporary"),
    )

    assert ConsoleChatController._dispatch_authority_matches(first_save, frozen)
    assert not ConsoleChatController._dispatch_authority_matches(later_save, frozen)
    assert not ConsoleChatController._dispatch_authority_matches(first_save, temporary)


@pytest.mark.asyncio
async def test_ephemeral_owner_survives_close_restore_and_dies_only_at_app_teardown(
    tmp_path: Path,
) -> None:
    _db, store, session_id = _ephemeral_store(tmp_path)
    session = store.sessions()[0]
    messages = tuple(store.messages_for_session(session_id))
    recovery = store.dispatch_recovery_for_session(session_id)

    replacement_controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        agent_runtime_enabled=False,
    )
    assert (
        replacement_controller.store.dispatch_recovery_for_session(session_id)
        == recovery
    )

    with pytest.raises(RuntimeError, match="pending turn"):
        replacement_controller.close_session(session_id)
    assert not replacement_controller.prompt_queue_registry.snapshot(session_id).closing
    with pytest.raises(RuntimeError, match="pending turn"):
        store.close_session(session_id)
    assert store.dispatch_recovery_for_session(session_id) == recovery
    store.restore_state(
        sessions=(session,),
        messages_by_session={session_id: messages},
        active_session_id=session_id,
    )
    assert store.dispatch_recovery_for_session(session_id) == recovery

    runtime = ConsoleRuntime(SimpleNamespace(persona_buddy_controller=None))
    runtime._chat_store = store
    await runtime.dispose()

    assert store.dispatch_recovery_for_session(session_id) is None
