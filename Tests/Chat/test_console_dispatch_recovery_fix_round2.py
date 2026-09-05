"""Task 15 fix round 2: postcommit interruption and retry-settlement ratchets."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _SettlementFaultGateway,
    _acceptance,
    _assert_terminal_fault_retained,
    _database,
    _insert,
    _patch_exact_retry_context,
    _restored_store,
)
from Tests.Chat.test_console_durable_turn_fix_round1 import (
    _install_real_effect_failure,
)
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsoleSubmitResult,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_prompt_queue import PromptQueueMode
from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain


_PRE_PROVIDER_POSTCOMMIT_EFFECTS = (
    "identity_publication",
    "durable_owner_publication",
    "staged_input_clearing",
    "workspace_projection",
    "queue_acknowledgement",
    "accepted_hook",
    "prompt_history",
    "preparation_publication",
    "checkpoint_transition",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_phase", ("immediate_checkpoint", "deferred_checkpoint", "provider")
)
async def test_dispatch_callback_failure_is_not_a_provider_terminal_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    try:
        terminal_writes = []
        original_mark_failed = store.mark_message_failed

        def mark_failed(message_id, *args, **kwargs):
            terminal_writes.append(message_id)
            return original_mark_failed(message_id, *args, **kwargs)

        monkeypatch.setattr(store, "mark_message_failed", mark_failed)
        if failure_phase != "provider":
            counts = _install_real_effect_failure(
                controller, store, "checkpoint_transition", monkeypatch
            )
        if failure_phase != "immediate_checkpoint":
            original_stream = gateway.stream_chat

            async def deferred_stream(resolution, messages, **kwargs):
                callback = kwargs.pop("before_provider_dispatch", None)
                if callback is not None:
                    try:
                        await callback()
                    except Exception:
                        # The real worker replaces the callback exception
                        # with a sanitized provider error on the event loop.
                        raise ChatProviderError("sanitized callback failure") from None
                if failure_phase == "provider":
                    gateway.calls += 1
                    raise ChatProviderError("ordinary provider failure")
                async for chunk in original_stream(resolution, messages, **kwargs):
                    yield chunk

            monkeypatch.setattr(
                gateway, "deferred_dispatch_boundary", True, raising=False
            )
            monkeypatch.setattr(gateway, "stream_chat", deferred_stream)

        first = await controller.submit_draft("retained body", session_id="session-1")

        assert first.accepted
        if failure_phase == "provider":
            assert first.provider_started
            assert gateway.calls == 1
            assert terminal_writes == [first.assistant_message_id]
            assistant = store.get_message(first.assistant_message_id)
            assert assistant.status == "failed"
            assert assistant.content == ""
            assert (
                controller.run_state_for("session-1").status is ConsoleRunStatus.FAILED
            )
            assert (
                db.get_connection()
                .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
                .fetchone()[0]
                == 0
            )
            return

        assert terminal_writes == []
        assert first.provider_started is False
        assert gateway.calls == 0
        assert counts == {"attempts": 1, "successes": 0}
        checkpoint = (
            db.get_connection()
            .execute("SELECT state FROM console_dispatch_checkpoints")
            .fetchone()
        )
        assert checkpoint["state"] == "accepted"
        _assert_exact_postcommit_recovery(
            controller, assistant_message_id=first.assistant_message_id
        )

        retried = await controller.retry_dispatch_recovery("session-1")

        assert retried.accepted
        assert retried.user_message_id == first.user_message_id
        assert retried.assistant_message_id == first.assistant_message_id
        assert counts == {"attempts": 2, "successes": 1}
        assert gateway.calls == 1
        assert terminal_writes == []
        assert store.get_message(first.assistant_message_id).status == "complete"
        assert store.dispatch_recovery_for_session("session-1") is None
    finally:
        await controller.shutdown()
        with db.quiesce_connections(timeout_seconds=2):
            pass
        assert db.registered_connection_count() == 0


def _assert_exact_postcommit_recovery(
    controller: ConsoleChatController,
    *,
    assistant_message_id: str,
) -> None:
    recovery = controller.store.dispatch_recovery_for_session("session-1")
    assert recovery is not None
    assert recovery.assistant_message_id == assistant_message_id
    assert recovery.runtime_active is False
    assert recovery.recovery_needed is True
    assert recovery.in_flight is False
    assert all(action.enabled for action in recovery.actions)
    assert controller.run_state_for("session-1").status is ConsoleRunStatus.BLOCKED


@pytest.mark.asyncio
@pytest.mark.parametrize("effect_name", _PRE_PROVIDER_POSTCOMMIT_EFFECTS)
async def test_explicit_retry_resumes_every_unfinished_postcommit_effect_before_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    effect_name: str,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    counts = _install_real_effect_failure(controller, store, effect_name, monkeypatch)

    first = await controller.submit_draft("retained exact body", session_id="session-1")

    assert first.accepted is True
    assert first.provider_started is False
    assert first.preparation_id is not None
    assert gateway.calls == 0
    assert counts == {"attempts": 1, "successes": 0}
    fingerprint = store.durable_acceptance_fingerprint_for(first.preparation_id)
    assert fingerprint is not None
    effects = store.durable_postcommit_effects_for(
        first.preparation_id,
        fingerprint=fingerprint,
    )
    assert effects is not None
    assert effect_name not in effects.completed
    _assert_exact_postcommit_recovery(
        controller,
        assistant_message_id=first.assistant_message_id or "",
    )
    before_rows = tuple(
        db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("conversations", "messages", "console_dispatch_checkpoints")
    )

    # This is the production action.  The regression must not call the old
    # internal resume_durable_postcommit escape hatch directly.
    retried = await controller.retry_dispatch_recovery("session-1")

    assert retried.accepted is True
    assert retried.user_message_id == first.user_message_id
    assert retried.assistant_message_id == first.assistant_message_id
    assert counts == {"attempts": 2, "successes": 1}
    assert gateway.calls == 1
    assert controller._durable_postcommit_continuations == {}
    assert store.durable_content_retention_count() == 0
    assert store.dispatch_recovery_for_session("session-1") is None
    assert tuple(
        db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("conversations", "messages", "console_dispatch_checkpoints")
    ) == before_rows[:2] + (0,)


@pytest.mark.asyncio
async def test_queued_postcommit_interruption_hydrates_before_return_and_drains_once(
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
        "session-1",
        text="later queued body",
        expected_revision=first.snapshot.revision,
    )
    assert first.entry_id is not None
    assert second.entry_id is not None
    coordinator._chains["session-1"] = _PromptChain()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id="session-1"
    )
    first_results: list[ConsoleSubmitResult] = []

    async def submit_first(text: str, **kwargs: Any) -> ConsoleSubmitResult:
        result = await controller.submit_draft(
            text,
            session_id=kwargs["session_id"],
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id=kwargs["entry_id"],
            queue_authorization=kwargs["authorization"],
        )
        first_results.append(result)
        return result

    coordinator._submit_queued = submit_first
    original_ack = coordinator.acknowledge_durable_acceptance
    attempts = 0

    def fail_ack_once(*args: Any, **kwargs: Any) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected exact queue acknowledgement")
        return original_ack(*args, **kwargs)

    monkeypatch.setattr(coordinator, "acknowledge_durable_acceptance", fail_ack_once)

    await coordinator._drain_waiting("session-1", ConsoleRunStatus.COMPLETED)

    assert len(first_results) == 1
    accepted = first_results[0]
    assert accepted.accepted is True
    assert accepted.assistant_message_id is not None
    assert gateway.calls == 0
    _assert_exact_postcommit_recovery(
        controller,
        assistant_message_id=accepted.assistant_message_id,
    )
    assert coordinator.dispatch_recovery_blocks_queue("session-1") is True
    blocked = registry.snapshot("session-1")
    assert blocked.claimed_count == 1
    assert blocked.waiting_count == 1
    assert blocked.mode is PromptQueueMode.DRAINING

    drained: list[str] = []

    async def submit_later(_text: str, **kwargs: Any) -> ConsoleSubmitResult:
        entry_id = kwargs["entry_id"]
        drained.append(entry_id)
        coordinator.turn_accepted(
            "session-1",
            origin=ConsoleSubmissionOrigin.QUEUED,
            context_epoch=0,
            entry_id=entry_id,
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id="session-1"
        )
        return ConsoleSubmitResult(
            True,
            True,
            terminal_status=ConsoleRunStatus.COMPLETED,
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id=entry_id,
        )

    coordinator._submit_queued = submit_later

    retried = await controller.retry_dispatch_recovery("session-1")

    assert retried.accepted is True
    assert attempts == 2
    assert gateway.calls == 1
    assert drained == [second.entry_id]
    assert coordinator.dispatch_recovery_blocks_queue("session-1") is False
    assert registry.snapshot("session-1").total_count == 0
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    )


@pytest.mark.asyncio
async def test_replacement_controller_cannot_bypass_live_postcommit_continuation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    counts = _install_real_effect_failure(
        controller, store, "accepted_hook", monkeypatch
    )
    first = await controller.submit_draft("retained exact body", session_id="session-1")
    assert first.accepted is True
    _assert_exact_postcommit_recovery(
        controller,
        assistant_message_id=first.assistant_message_id or "",
    )

    replacement = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_runtime_enabled=False,
    )
    refused = await replacement.retry_dispatch_recovery("session-1")

    assert refused.accepted is False
    assert "continuation is unavailable" in refused.visible_copy.lower()
    assert gateway.calls == 0
    assert counts == {"attempts": 1, "successes": 0}
    _assert_exact_postcommit_recovery(
        controller,
        assistant_message_id=first.assistant_message_id or "",
    )
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    )

    recovered = await controller.retry_dispatch_recovery("session-1")
    assert recovered.accepted is True
    assert gateway.calls == 1
    assert counts == {"attempts": 2, "successes": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["manual", "queued"])
@pytest.mark.parametrize("provider_outcome", ["success", "failure"])
async def test_post_cas_retry_settlement_fault_restores_runtime_truth_and_queue_fence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
    provider_outcome: str,
) -> None:
    queue_entry_id = "queue-1" if origin == "queued" else None
    db, conversation_id, repository = _database(
        tmp_path / f"retry-{origin}-{provider_outcome}.sqlite"
    )
    _insert(
        db,
        repository,
        _acceptance(
            conversation_id,
            origin=origin,
            queue_entry_id=queue_entry_id,
        ),
    )
    store, session_id = _restored_store(db, conversation_id)
    gateway = _SettlementFaultGateway(db, outcome=provider_outcome)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    await _patch_exact_retry_context(monkeypatch, controller, gateway)
    db.get_connection().execute(
        "CREATE TRIGGER fail_round2_terminal_delete BEFORE DELETE ON "
        "console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT, 'fail'); END"
    )
    db.get_connection().commit()

    result = await controller.retry_dispatch_recovery(session_id)

    assert result.accepted is False
    assert result.visible_copy == "Response recovery failed. Try again or discard."
    _assert_terminal_fault_retained(db, store, session_id, gateway)
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert recovery.runtime_active is False
    assert recovery.recovery_needed is True
    assert recovery.in_flight is False
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.BLOCKED
    assert controller.prompt_queue_coordinator.dispatch_recovery_blocks_queue(
        session_id
    ) is (origin == "queued")


@pytest.mark.asyncio
async def test_post_cas_retry_cancel_settlement_fault_restores_runtime_truth(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "retry-cancel.sqlite")
    _insert(
        db,
        repository,
        _acceptance(
            conversation_id,
            origin="queued",
            queue_entry_id="queue-1",
        ),
    )
    store, session_id = _restored_store(db, conversation_id)
    gateway = _SettlementFaultGateway(db, outcome="cancel")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    await _patch_exact_retry_context(monkeypatch, controller, gateway)
    db.get_connection().execute(
        "CREATE TRIGGER fail_round2_cancel_delete BEFORE DELETE ON "
        "console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT, 'fail'); END"
    )
    db.get_connection().commit()

    task = asyncio.create_task(controller.retry_dispatch_recovery(session_id))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert controller.stop_active_run(record_user_stop=False) is True
    result = await asyncio.wait_for(task, timeout=1)

    assert result.accepted is False
    _assert_terminal_fault_retained(db, store, session_id, gateway)
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert recovery.runtime_active is False
    assert recovery.recovery_needed is True
    assert recovery.in_flight is False
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.BLOCKED
    assert (
        controller.prompt_queue_coordinator.dispatch_recovery_blocks_queue(session_id)
        is True
    )


@pytest.mark.asyncio
async def test_pre_cas_retry_refusal_only_releases_action_claim(
    tmp_path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "retry-pre-cas.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _SettlementFaultGateway(db, outcome="success")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="different-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    before = store.dispatch_recovery_for_session(session_id)
    assert before is not None

    result = await controller.retry_dispatch_recovery(session_id)

    assert result.accepted is False
    assert gateway.provider_states == []
    after = store.dispatch_recovery_for_session(session_id)
    assert after is not None
    assert after.assistant_message_id == before.assistant_message_id
    assert after.checkpoint == before.checkpoint
    assert after.runtime_active is False
    assert after.recovery_needed is True
    assert after.in_flight is False
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    )


@pytest.mark.asyncio
async def test_retry_exception_after_checkpoint_cas_restores_runtime_truth(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "retry-post-cas.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _SettlementFaultGateway(db, outcome="success")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    await _patch_exact_retry_context(monkeypatch, controller, gateway)
    original_transition = store.transition_dispatch_recovery_for_retry

    def transition_then_raise(*args: Any, **kwargs: Any):
        transitioned = original_transition(*args, **kwargs)
        assert transitioned is not None
        raise RuntimeError("injected local exception after checkpoint CAS")

    monkeypatch.setattr(
        store,
        "transition_dispatch_recovery_for_retry",
        transition_then_raise,
    )

    result = await controller.retry_dispatch_recovery(session_id)

    assert result.accepted is False
    assert gateway.provider_states == []
    checkpoint_row = (
        db.get_connection()
        .execute("SELECT state, checkpoint_revision FROM console_dispatch_checkpoints")
        .fetchone()
    )
    assert checkpoint_row is not None
    assert checkpoint_row["state"] == "dispatch_started"
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert recovery.checkpoint is not None
    assert recovery.checkpoint.state.value == checkpoint_row["state"]
    assert (
        recovery.checkpoint.checkpoint_revision == checkpoint_row["checkpoint_revision"]
    )
    assert recovery.runtime_active is False
    assert recovery.recovery_needed is True
    assert recovery.in_flight is False
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.BLOCKED


@pytest.mark.asyncio
async def test_retry_does_not_restore_a_checkpoint_mutated_after_cas(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "retry-mutated-cas.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    gateway = _SettlementFaultGateway(db, outcome="success")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    await _patch_exact_retry_context(monkeypatch, controller, gateway)
    original_transition = store.transition_dispatch_recovery_for_retry
    original_restore = controller._restore_dispatch_recovery_after_settlement_failure
    restore_calls: list[tuple[str, str]] = []

    def record_restore(target_session_id: str, assistant_message_id: str) -> None:
        restore_calls.append((target_session_id, assistant_message_id))
        original_restore(target_session_id, assistant_message_id)

    def transition_mutate_then_raise(*args: Any, **kwargs: Any):
        transitioned = original_transition(*args, **kwargs)
        assert transitioned is not None
        assert transitioned.checkpoint is not None
        mutated_checkpoint = replace(
            transitioned.checkpoint,
            frozen_authority=replace(
                transitioned.checkpoint.frozen_authority,
                direct_library_tools=(
                    not transitioned.checkpoint.frozen_authority.direct_library_tools
                ),
            ),
        )
        store._dispatch_recoveries_by_session[session_id] = replace(
            transitioned,
            checkpoint=mutated_checkpoint,
        )
        raise RuntimeError("injected owner mutation after checkpoint CAS")

    monkeypatch.setattr(
        controller,
        "_restore_dispatch_recovery_after_settlement_failure",
        record_restore,
    )
    monkeypatch.setattr(
        store,
        "transition_dispatch_recovery_for_retry",
        transition_mutate_then_raise,
    )

    result = await controller.retry_dispatch_recovery(session_id)

    assert result.accepted is False
    assert gateway.provider_states == []
    assert restore_calls == []
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert recovery.runtime_active is True
    assert recovery.in_flight is False
