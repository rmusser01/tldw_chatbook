"""Native accepted setup errors reconcile actual writes, never retry them."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import replace

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller, _until
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


@pytest.mark.parametrize("error_order", ["after-write", "before-write"])
async def test_accepted_setup_native_error_preserves_actual_outcome(
    tmp_path, monkeypatch, owned_console_databases, error_order
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    configuration = await controller.capture_turn_configuration_snapshot("session-1")
    entered, release = asyncio.Event(), asyncio.Event()
    original_hook = controller.hook_admission_reason

    async def held_hook():
        if (
            not entered.is_set()
            and store.dispatch_recovery_for_session("session-1") is not None
        ):
            entered.set()
            await release.wait()
        return await original_hook()

    monkeypatch.setattr(controller, "hook_admission_reason", held_hook)
    original_settle = store.settle_accepted_durable_turn
    settlements = []

    def failed_cleanup(*args, **kwargs):
        if error_order == "before-write":
            settlements.append(False)
            raise RuntimeError("terminal callback failed before write")
        result = original_settle(*args, **kwargs)
        settlements.append(result)
        assert result is True
        raise RuntimeError("terminal callback failed after write")

    monkeypatch.setattr(store, "settle_accepted_durable_turn", failed_cleanup)
    task = asyncio.create_task(
        controller.submit_draft(
            "one accepted setup write",
            session_id="session-1",
            configuration=configuration,
        )
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        preparation = store.preparation_for_session("session-1")
        assert preparation is not None
        assert controller._ordinary_native_commit_tasks("session-1") == ()
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assistant_id, version = fresh.execute(
                "SELECT id, version FROM messages WHERE role='assistant'"
            ).fetchone()
        controller._deferred_user_stop_markers.add("another-assistant")
        assert controller.stop_active_run()
        assert assistant_id in controller._deferred_user_stop_markers
        controller.stop_active_run()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted and not result.provider_started
        assert gateway.calls == 0
        assert "session-1" not in controller._active_cancel_events
        assert "session-1" not in controller._active_stop_reasons
        assert assistant_id not in controller._deferred_user_stop_markers
        assert "another-assistant" in controller._deferred_user_stop_markers
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            state, final_version = fresh.execute(
                "SELECT assistant_generation_state, version FROM messages WHERE id=?",
                (assistant_id,),
            ).fetchone()
            checkpoint_count = fresh.execute(
                "SELECT COUNT(*) FROM console_dispatch_checkpoints WHERE assistant_message_id=?",
                (assistant_id,),
            ).fetchone()[0]
        if error_order == "before-write":
            assert settlements == [False]
            assert result.terminal_status is ConsoleRunStatus.BLOCKED
            assert (
                state == "accepted"
                and final_version == version
                and checkpoint_count == 1
            )
            assert store.preparation_for_session("session-1") is preparation
            recovery = store.dispatch_recovery_for_session("session-1")
            assert recovery is not None and recovery.recovery_needed
            return
        assert settlements == [True]
        assert result.terminal_status is ConsoleRunStatus.STOPPED
        assert (
            state == "stopped"
            and final_version == version + 1
            and checkpoint_count == 0
        )
        assert store.dispatch_recovery_for_session("session-1") is None
        assert store.preparation_for_session("session-1") is None
        assert (
            preparation.preparation_id
            not in controller._durable_postcommit_continuations
        )
        assert (
            sum(
                row.content == "Response stopped by user."
                for row in store.all_messages_for_session("session-1")
            )
            == 1
        )
        configuration = await controller.capture_turn_configuration_snapshot(
            "session-1"
        )
        next_result = await asyncio.wait_for(
            controller.submit_draft(
                "next accepted send",
                session_id="session-1",
                configuration=configuration,
            ),
            5,
        )
        assert (
            next_result.accepted
            and next_result.terminal_status is ConsoleRunStatus.COMPLETED
        )
        assert gateway.calls == 1
        assert any(
            row.content == "done" for row in store.all_messages_for_session("session-1")
        )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.dispose()


@pytest.mark.parametrize("cleanup_error", [False, True], ids=["clean", "cleanup-error"])
@pytest.mark.parametrize("replacement", ["store", "persistence", "database", "session"])
async def test_accepted_setup_settlement_does_not_publish_into_successor(
    tmp_path, monkeypatch, owned_console_databases, cleanup_error, replacement
):
    db, original_store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    entered, release = asyncio.Event(), asyncio.Event()
    original_hook = controller.hook_admission_reason

    async def held_hook():
        if (
            not entered.is_set()
            and original_store.dispatch_recovery_for_session("session-1") is not None
        ):
            entered.set()
            await release.wait()
        return await original_hook()

    monkeypatch.setattr(controller, "hook_admission_reason", held_hook)
    settled, native_release = threading.Event(), threading.Event()
    original_settle = original_store.settle_accepted_durable_turn
    writes = []

    def held_terminal(*args, **kwargs):
        result = original_settle(*args, **kwargs)
        writes.append(result)
        assert result is True
        settled.set()
        assert native_release.wait(10)
        if cleanup_error:
            raise RuntimeError("original terminal cleanup failed")
        return result

    monkeypatch.setattr(original_store, "settle_accepted_durable_turn", held_terminal)
    successor = ConsoleChatStore()
    next_session = successor.create_session(session_id="session-1", title="Successor")
    next_session.draft = "successor draft"
    next_state = ConsoleRunState(ConsoleRunStatus.IDLE, "Successor ready")
    original_persistence = original_store.persistence
    original_session = next(
        row for row in original_store.sessions() if row.id == "session-1"
    )
    task = asyncio.create_task(
        controller.submit_draft("original setup turn", session_id="session-1")
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assistant_id, version = fresh.execute(
                "SELECT id, version FROM messages WHERE role='assistant'"
            ).fetchone()
        assert controller.stop_active_run()
        assert await _until(settled.is_set, timeout=5)
        assert not task.done()
        if replacement == "store":
            controller.store = successor
        elif replacement == "persistence":
            original_store.persistence = successor.persistence
        elif replacement == "database":
            original_persistence.db = None
        else:
            original_store._sessions["session-1"] = next_session
        original_messages = tuple(original_store.all_messages_for_session("session-1"))
        controller._set_run_state(next_state, session_id="session-1")
        native_release.set()
        (result,) = await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), 5
        )
        assert successor.messages_for_session("session-1") == []
        assert (
            tuple(original_store.all_messages_for_session("session-1"))
            == original_messages
        )
        assert not isinstance(result, BaseException), result
        assert (
            result.accepted
            and not result.provider_started
            and not result.should_clear_draft
        )
        assert writes == [True] and gateway.calls == 0
        assert next_session.draft == "successor draft"
        assert next_session.persisted_conversation_id is None
        assert controller.run_state_for("session-1") == next_state
        assert original_store.dispatch_recovery_for_session("session-1") is None
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assert fresh.execute(
                "SELECT assistant_generation_state, version FROM messages WHERE id=?",
                (assistant_id,),
            ).fetchone() == ("stopped", version + 1)
            assert (
                fresh.execute(
                    "SELECT COUNT(*) FROM console_dispatch_checkpoints"
                ).fetchone()[0]
                == 0
            )
    finally:
        release.set()
        native_release.set()
        await asyncio.gather(task, return_exceptions=True)
        controller.store = original_store
        original_store.persistence = original_persistence
        original_persistence.db = db
        original_store._sessions["session-1"] = original_session


@pytest.mark.parametrize(
    "window",
    [
        "chain",
        "direct-ready",
        "direct-prefill",
        "direct-prefill-prewrite",
        "direct-prefill-extra",
        "direct-prefill-newer",
    ],
)
async def test_stop_during_real_chain_creation_retires_accepted_owner(
    tmp_path, monkeypatch, owned_console_databases, window
):
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runs = AgentRunsDB(tmp_path / "chain-runs.sqlite", client_id="accepted-chain-test")
    controller._agent_bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    controller._agent_runtime_enabled = (
        False  # real ledger creation also precedes direct replies
    )
    entered, release = threading.Event(), threading.Event()
    create_chain = runs.automatic_work.create_chain

    def held_chain(*args, **kwargs):
        if window == "chain" and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return create_chain(*args, **kwargs)

    original_ready = controller._wait_for_trace_maintenance_dispatch

    async def held_ready():
        if window.startswith("direct-") and not entered.is_set():
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        return await original_ready()

    monkeypatch.setattr(controller, "_wait_for_trace_maintenance_dispatch", held_ready)
    monkeypatch.setattr(runs.automatic_work, "create_chain", held_chain)
    failures = []
    original_failure = store.mark_dispatch_recovery_needed

    def record_failure(*args, **kwargs):
        failures.append("recovery_needed")
        return original_failure(*args, **kwargs)

    original_state = controller._set_run_state

    def record_state(state, **kwargs):
        if state.status in {ConsoleRunStatus.BLOCKED, ConsoleRunStatus.FAILED}:
            failures.append(state.status.value)
        return original_state(state, **kwargs)

    monkeypatch.setattr(store, "mark_dispatch_recovery_needed", record_failure)
    monkeypatch.setattr(controller, "_set_run_state", record_state)
    prefill = "Exact prefix:" if "prefill" in window else ""
    if prefill:
        store.set_session_one_shot_prefill("session-1", prefill)
    if window == "direct-prefill-prewrite":

        def refuse_terminal(*args, **kwargs):
            raise RuntimeError("accepted prefill write failed before persistence")

        monkeypatch.setattr(store, "settle_accepted_durable_turn", refuse_terminal)
    task = asyncio.create_task(
        controller.submit_draft("stop before the stream owner", session_id="session-1")
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        recovery = store.dispatch_recovery_for_session("session-1")
        assert recovery is not None and not recovery.in_flight
        assert controller._ordinary_native_commit_tasks("session-1") == ()
        assert controller._active_stream_tasks["session-1"] is task
        assert ("session-1" in controller._active_cancel_events) == (
            window.startswith("direct-")
        )
        # Durable acceptance has already consumed the exact original one-shot.
        # Stop must preserve this boundary and never consume a newly armed slot.
        assert store.session_one_shot_prefill("session-1") is None
        # Preserve immediate prefix display, including a mounted materialization.
        assert store.get_message(recovery.assistant_message_id).content == prefill
        if window == "direct-prefill-extra":
            # Leave this suffix buffered: the materialized prefix alone is not proof.
            store.append_stream_chunk(recovery.assistant_message_id, "unexpected")
        if window in {
            "direct-prefill-newer",
            "direct-prefill-prewrite",
            "direct-prefill-extra",
        }:
            store.set_session_one_shot_prefill("session-1", "newer prefix")
        slot_before_stop = store.session_one_shot_prefill_snapshot("session-1")
        assert controller.stop_active_run()
        controller.stop_active_run()
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        if window in {"direct-prefill-prewrite", "direct-prefill-extra"}:
            assert (
                result.accepted and result.terminal_status is ConsoleRunStatus.BLOCKED
            )
            assert gateway.calls == 0
            assert not result.provider_started
            assert (
                store.session_one_shot_prefill_snapshot("session-1") == slot_before_stop
            )
            assert store.get_message(recovery.assistant_message_id).content == (
                prefill + ("unexpected" if window.endswith("extra") else "")
            )
            retained = store.dispatch_recovery_for_session("session-1")
            assert retained is not None and retained.recovery_needed
            with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
                assert fresh.execute(
                    "SELECT assistant_generation_state FROM messages WHERE id=?",
                    (recovery.assistant_message_id,),
                ).fetchone() == ("accepted",)
            return
        assert failures == []
        assert result.accepted and result.terminal_status is ConsoleRunStatus.STOPPED
        assert not result.provider_started and gateway.calls == 0
        assert store.get_message(recovery.assistant_message_id).content == prefill
        assert store.session_one_shot_prefill("session-1") == (
            "newer prefix" if window.endswith("newer") else None
        )
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assert fresh.execute(
                "SELECT assistant_generation_state, content FROM messages WHERE id=?",
                (recovery.assistant_message_id,),
            ).fetchone() == ("stopped", prefill)
        assert store.get_message(recovery.assistant_message_id).status == "stopped"
        assert store.dispatch_recovery_for_session("session-1") is None
        assert store.preparation_for_session("session-1") is None
        assert (
            sum(
                row.content == "Response stopped by user."
                for row in store.all_messages_for_session("session-1")
            )
            == 1
        )
        next_result = await asyncio.wait_for(
            controller.submit_draft("next send", session_id="session-1"), 5
        )
        assert (
            next_result.accepted
            and next_result.terminal_status is ConsoleRunStatus.COMPLETED
        )
        assert gateway.calls == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await controller.shutdown()
        runs.close()
        assert not runs._maintenance_participant.connections


@pytest.mark.parametrize("outcome", ["complete", "failed", "empty", "stopped"])
async def test_direct_prefill_preserves_payload_and_terminal_contract(
    tmp_path, monkeypatch, owned_console_databases, outcome
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    prefix = "User-authored prefix:"
    store.set_session_one_shot_prefill("session-1", prefix)
    original_stream = gateway.stream_chat
    observed = []

    async def stream(resolution, messages, **kwargs):
        observed.append(messages)
        assert messages[-1] == {"role": "assistant", "content": prefix}
        assistant = next(
            row
            for row in reversed(store.messages_for_session("session-1"))
            if row.role.value == "assistant"
        )
        assert assistant.content == prefix
        assert store.session_one_shot_prefill("session-1") is None
        if outcome in {"failed", "empty"}:
            store.set_session_one_shot_prefill("session-1", "newly armed prefix")
        async for chunk in original_stream(resolution, messages, **kwargs):
            if outcome == "failed":
                raise RuntimeError("provider failed before its first chunk")
            if outcome == "empty":
                continue
            if outcome == "stopped":
                yield "partial"
                controller._signal_stop(session_id="session-1")
                yield "never-shown"
            else:
                yield chunk

    monkeypatch.setattr(gateway, "stream_chat", stream)
    result = await controller.submit_draft("preserve prefix", session_id="session-1")
    assert result.accepted
    assert len(observed) == gateway.calls == 1
    assistant = store.get_message(result.assistant_message_id)
    expected = prefix + (
        "done" if outcome == "complete" else "partial" if outcome == "stopped" else ""
    )
    assert assistant.content == expected
    assert assistant.status == (
        "complete"
        if outcome == "complete"
        else "stopped"
        if outcome == "stopped"
        else "failed"
    )
    assert store.session_one_shot_prefill("session-1") == (
        "newly armed prefix" if outcome in {"failed", "empty"} else None
    )
    with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
        assert fresh.execute(
            "SELECT content FROM messages WHERE id=?", (assistant.id,)
        ).fetchone() == (expected,)


@pytest.mark.parametrize(
    "evidence",
    [
        "accepted",
        "after-dispatch",
        "missing-recovery",
        "wrong-preparation",
        "stale-session",
    ],
)
async def test_direct_stop_after_store_replacement_reports_original_dispatch(
    tmp_path, monkeypatch, owned_console_databases, evidence
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    entered, release = asyncio.Event(), asyncio.Event()
    original_ready = controller._wait_for_trace_maintenance_dispatch
    original_stream = gateway.stream_chat

    async def held_ready():
        if evidence != "after-dispatch":
            entered.set()
            await release.wait()
        return await original_ready()

    async def held_stream(*args, **kwargs):
        async for chunk in original_stream(*args, **kwargs):
            entered.set()
            await release.wait()
            yield chunk

    monkeypatch.setattr(controller, "_wait_for_trace_maintenance_dispatch", held_ready)
    monkeypatch.setattr(gateway, "stream_chat", held_stream)
    successor = ConsoleChatStore()
    next_session = successor.create_session(session_id="session-1", title="Successor")
    next_session.draft = "successor draft"
    next_state = ConsoleRunState(ConsoleRunStatus.IDLE, "Successor ready")
    original_session = store.sessions()[0]
    task = asyncio.create_task(
        controller.submit_draft("cancel original direct reply", session_id="session-1")
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        preparation = store.preparation_for_session("session-1")
        recovery = store.dispatch_recovery_for_session("session-1")
        assert preparation is not None and recovery is not None
        assert gateway.calls == (1 if evidence == "after-dispatch" else 0)
        assert gateway.states_seen == (
            ["dispatch_started"] if evidence == "after-dispatch" else []
        )
        assert recovery.checkpoint.state is (
            ConsoleDispatchCheckpointState.DISPATCH_STARTED
            if evidence == "after-dispatch"
            else ConsoleDispatchCheckpointState.ACCEPTED
        )
        if evidence == "missing-recovery":
            store._dispatch_recoveries_by_session.pop("session-1")
        elif evidence == "wrong-preparation":
            store._dispatch_recoveries_by_session["session-1"] = replace(
                recovery, preparation_id="different-preparation"
            )
        elif evidence == "stale-session":
            store._sessions["session-1"] = next_session
        retained_recovery = store.dispatch_recovery_for_session("session-1")
        original_messages = tuple(store.all_messages_for_session("session-1"))
        controller._active_cancel_events["session-1"].set()
        controller.store = successor
        controller._set_run_state(next_state, session_id="session-1")
        task.cancel()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted and result.terminal_status is ConsoleRunStatus.BLOCKED
        assert result.provider_started is (evidence != "accepted")
        assert not result.should_clear_draft
        assert successor.messages_for_session("session-1") == []
        assert next_session.draft == "successor draft"
        assert next_session.persisted_conversation_id is None
        assert controller.run_state_for("session-1") == next_state
        assert tuple(store.all_messages_for_session("session-1")) == original_messages
        assert store.dispatch_recovery_for_session("session-1") is retained_recovery
        assert store.preparation_for_session("session-1") is preparation
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        controller.store = store
        store._sessions["session-1"] = original_session


@pytest.mark.parametrize(
    "changed_identity", ["preparation", "checkpoint", "fingerprint", "commit"]
)
async def test_direct_settlement_error_requires_exact_original_acceptance(
    tmp_path, monkeypatch, owned_console_databases, changed_identity
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    entered, release = asyncio.Event(), asyncio.Event()
    original_ready = controller._wait_for_trace_maintenance_dispatch
    failures = []
    original_continuation = None

    async def held_ready():
        entered.set()
        await release.wait()
        return await original_ready()

    def refuse_terminal(preparation_id, **kwargs):
        # Lose one exact identity only after the real accepted owner enters its
        # terminal callback. No write or provider dispatch occurs in this control.
        with store.durable_preparation_lock:
            recovery = store.dispatch_recovery_for_session("session-1")
            if changed_identity == "preparation":
                store._dispatch_recoveries_by_session["session-1"] = replace(
                    recovery, preparation_id="different-preparation"
                )
            elif changed_identity == "checkpoint":
                store._dispatch_recoveries_by_session["session-1"] = replace(
                    recovery,
                    checkpoint=replace(
                        recovery.checkpoint,
                        checkpoint_revision=recovery.checkpoint.checkpoint_revision + 1,
                    ),
                )
            elif changed_identity == "fingerprint":
                fingerprint = store._durable_fingerprint_by_preparation[preparation_id]
                store._durable_fingerprint_by_preparation[preparation_id] = replace(
                    fingerprint, digest="different-fingerprint"
                )
            else:
                commit = store._durable_commit_by_preparation[preparation_id]
                store._durable_commit_by_preparation[preparation_id] = replace(commit)
            failures.append(changed_identity)
        raise RuntimeError("terminal callback failed before write")

    monkeypatch.setattr(controller, "_wait_for_trace_maintenance_dispatch", held_ready)
    monkeypatch.setattr(store, "settle_accepted_durable_turn", refuse_terminal)
    task = asyncio.create_task(
        controller.submit_draft(
            "retain uncertain dispatch evidence", session_id="session-1"
        )
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        recovery = store.dispatch_recovery_for_session("session-1")
        assert recovery is not None and not recovery.in_flight
        assert recovery.checkpoint.state is ConsoleDispatchCheckpointState.ACCEPTED
        original_continuation = controller._durable_postcommit_continuations[
            recovery.preparation_id
        ]
        assert controller.stop_active_run()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert failures == [changed_identity]
        assert result.accepted and result.terminal_status is ConsoleRunStatus.BLOCKED
        assert gateway.calls == 0
        assert result.provider_started
        retained = store.dispatch_recovery_for_session("session-1")
        assert retained is not None and retained.recovery_needed
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assert fresh.execute(
                "SELECT assistant_generation_state FROM messages WHERE id=?",
                (recovery.assistant_message_id,),
            ).fetchone() == ("accepted",)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if original_continuation is not None:
            # Restore only injected identity changes before the real owner drains.
            with store.durable_preparation_lock:
                store._durable_fingerprint_by_preparation[
                    original_continuation.preparation_id
                ] = original_continuation.fingerprint
                store._durable_commit_by_preparation[
                    original_continuation.preparation_id
                ] = original_continuation.commit


@pytest.mark.parametrize("replacement", ["store", "persistence", "database", "session"])
async def test_temporary_chain_stop_refuses_successor_publication(
    tmp_path, monkeypatch, owned_console_databases, replacement
):
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import ConsoleDispatchRecoveryKind
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    session = store.create_session(session_id="temporary", title="Temporary", ephemeral=True)
    gateway.deferred_dispatch_boundary = True
    runs = AgentRunsDB(tmp_path / "temporary-runs.sqlite", client_id="temporary-stop-test")
    controller._agent_bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    controller._agent_runtime_enabled = False
    entered, release = threading.Event(), threading.Event()
    original_chain = runs.automatic_work.create_chain

    def chain(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_chain(*args, **kwargs)

    monkeypatch.setattr(runs.automatic_work, "create_chain", chain)
    successor = ConsoleChatStore()
    next_session = successor.create_session(session_id=session.id, ephemeral=True)
    next_session.draft = "successor draft"
    next_state = ConsoleRunState(ConsoleRunStatus.IDLE, "Successor ready")
    persistence = store.persistence
    task = asyncio.create_task(controller.submit_draft("temporary stop", session_id=session.id))
    try:
        assert await _until(entered.is_set, timeout=5)
        recovery = store.dispatch_recovery_for_session(session.id)
        assert recovery.kind is ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED
        preparation = store.preparation_for_session(session.id)
        assistant_id = recovery.assistant_message_id
        assert controller._active_stream_tasks[session.id] is task
        assert session.id not in controller._active_cancel_events
        controller._deferred_user_stop_markers.add("other-assistant")
        assert controller.stop_active_run()
        if replacement == "store":
            controller.store = successor
        elif replacement == "persistence":
            store.persistence = successor.persistence
        elif replacement == "database":
            persistence.db = None
        else:
            store._sessions[session.id] = next_session
        messages = tuple(store.all_messages_for_session(session.id))
        controller._set_run_state(next_state, session_id=session.id)
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted and not result.should_clear_draft and not result.provider_started
        assert result.terminal_status is ConsoleRunStatus.BLOCKED
        assert gateway.calls == 0
        assert controller.run_state_for(session.id) == next_state
        assert successor.messages_for_session(session.id) == []
        assert tuple(store.all_messages_for_session(session.id)) == messages
        assert next_session.draft == "successor draft"
        assert store.dispatch_recovery_for_session(session.id) is recovery
        assert store.preparation_for_session(session.id) is preparation
        assert assistant_id not in controller._deferred_user_stop_markers
        assert "other-assistant" in controller._deferred_user_stop_markers
        assert session.id not in controller._active_cancel_events
        assert session.id not in controller._active_stop_reasons
        assert assistant_id not in controller._replies_in_pre_provider_setup
        assert await _until(lambda: not controller._maintenance_background, timeout=5)
        assert db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints"
        ).fetchone()[0] == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        controller.store = store
        store.persistence = persistence
        persistence.db = db
        store._sessions[session.id] = session
