"""Ordinary save cancellation before dispatch and runtime retirement controls."""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from Tests.Chat.test_console_first_send_atomicity import (
    _WriteLockHolder,
    _controller,
    _until,
)
from Tests.Chat.test_console_runtime_shutdown import _recovery_record
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import (
    CONSOLE_SESSION_CLOSE_GRACE_SECONDS,
    CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS,
    ConsoleRuntime,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


def _hold_original_commit(monkeypatch, target, *, after_commit):
    """Hold the original callback, never substitute a successful commit."""
    original = target.commit_durable_turn
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def held(*args, **kwargs):
        calls.append((args, kwargs))
        result = original(*args, **kwargs) if after_commit else None
        entered.set()
        try:
            assert release.wait(10)
            return result if after_commit else original(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(target, "commit_durable_turn", held)
    return entered, release, exited, calls


def _saved_state(db_path):
    with sqlite3.connect(db_path) as fresh:
        return (
            fresh.execute("SELECT state FROM console_dispatch_checkpoints").fetchall(),
            fresh.execute(
                "SELECT role, content, assistant_generation_state, metadata_json FROM messages ORDER BY role"
            ).fetchall(),
        )


@pytest.mark.parametrize(
    ("cancellation", "new_draft"),
    [("stop", "original draft"), ("cancel", "a newer draft")],
)
async def test_cancelled_actual_accepted_save_settles_without_dispatch_or_clearing_new_input(
    tmp_path, monkeypatch, owned_console_databases, cancellation, new_draft
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    session = store.sessions()[0]
    session.draft = "original draft"
    cleared = []
    controller.on_submission_accepted = lambda: cleared.append(True)
    entered, release, exited, calls = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    baseline_connections = db.registered_connection_count()
    task = asyncio.create_task(
        controller.submit_draft("original draft", session_id=session.id)
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert checkpoints == [("accepted",)]
        assert [row[2] for row in rows if row[0] == "assistant"] == ["accepted"]
        assert gateway.calls == 0
        session.draft = ""
        session.draft = new_draft
        attachment = PendingAttachment(
            "/new.png",
            "new.png",
            "image",
            "attachment",
            data=b"new bytes",
            mime_type="image/png",
        )
        assert store.add_pending_attachment(session.id, attachment)
        store.set_session_one_shot_prefill(session.id, "new prefill")
        prefill = store.session_one_shot_prefill_snapshot(session.id)
        if cancellation == "stop":
            assert controller.stop_active_run(record_user_stop=False)
            controller.stop_active_run(record_user_stop=False)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done()
            assert not exited.is_set()
            assert gateway.calls == 0
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted is True
        assert result.provider_started is False
        assert result.should_clear_draft is False
        assert len(calls) == 1
        assert exited.is_set()
        assert db.registered_connection_count() == baseline_connections
        assert session.draft == new_draft
        assert cleared == []
        assert store.pending_attachments(session.id) == [attachment]
        assert store.session_one_shot_prefill_snapshot(session.id) == prefill
        assert controller.prompt_history.size == 1
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert checkpoints == []
        assert len(rows) == 2
        assistant = next(row for row in rows if row[0] == "assistant")
        assert assistant[2] == ("stopped" if cancellation == "stop" else "failed")
        metadata = MessageMetadata.from_json(assistant[3])
        assert bool(metadata and metadata.terminal_receipt_id) is (
            cancellation == "cancel"
        )
        assert gateway.calls == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_exception_after_actual_commit_never_retries_or_enters_provider(
    tmp_path, monkeypatch, owned_console_databases
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    original = store.commit_durable_turn
    calls = []

    def committed_then_failed(acceptance):
        result = original(acceptance)
        calls.append(result)
        raise RuntimeError("callback failed after durable success")

    monkeypatch.setattr(store, "commit_durable_turn", committed_then_failed)
    store.sessions()[0].draft = "keep this draft"
    result = await controller.submit_draft("keep this draft", session_id="session-1")
    assert len(calls) == 1
    assert result.accepted is True
    assert result.provider_started is False
    assert result.should_clear_draft is False
    assert gateway.calls == 0
    assert store.sessions()[0].draft == "keep this draft"
    checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
    assert len(rows) == 2
    assert sum(row[0] == "user" for row in rows) == 1
    if checkpoints:
        assert checkpoints == [("accepted",)]
        assert store.dispatch_recovery_for_session("session-1") is not None
    else:
        assert next(row[2] for row in rows if row[0] == "assistant") == "failed"


async def test_cancelled_accepted_save_retains_recovery_if_terminal_write_fails(
    tmp_path, monkeypatch, owned_console_databases
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    db.get_connection().execute(
        "CREATE TRIGGER fail_owned_cancel BEFORE UPDATE ON messages "
        "WHEN NEW.assistant_generation_state='failed' "
        "BEGIN SELECT RAISE(ABORT, 'cannot settle'); END"
    )
    entered, release, _exited, calls = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    task = asyncio.create_task(
        controller.submit_draft("saved once", session_id="session-1")
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        assert _saved_state(tmp_path / "controller.sqlite")[0] == [("accepted",)]
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted is True and result.provider_started is False
        assert gateway.calls == 0 and len(calls) == 1
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert checkpoints == [("accepted",)]
        assert len(rows) == 2
        recovery = store.dispatch_recovery_for_session("session-1")
        assert recovery is not None
        assert recovery.recovery_needed is True
        assert not recovery.in_flight
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_controller_store_replacement_cannot_publish_or_dispatch_original_save_into_successor(
    tmp_path, monkeypatch, owned_console_databases
):
    db, original_store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    entered, release, _exited, calls = _hold_original_commit(
        monkeypatch, original_store, after_commit=True
    )
    successor = ConsoleChatStore()
    next_session = successor.create_session(session_id="session-1", title="Successor")
    next_session.draft = "successor draft"
    task = asyncio.create_task(
        controller.submit_draft("original saved turn", session_id="session-1")
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        controller.store = successor
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted is True
        assert result.provider_started is False
        assert result.should_clear_draft is False
        assert len(calls) == 1 and gateway.calls == 0
        assert next_session.draft == "successor draft"
        assert next_session.persisted_conversation_id is None
        assert successor.messages_for_session(next_session.id) == []
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert len(rows) == 2
        if checkpoints:
            assert checkpoints == [("accepted",)]
            assert original_store.dispatch_recovery_for_session("session-1") is not None
        else:
            assert next(row[2] for row in rows if row[0] == "assistant") == "failed"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        controller.store = original_store


@pytest.mark.parametrize("action", ["close", "dispose", "dispose_early"])
async def test_lifecycle_outlives_original_grace_and_repeated_cancellation_until_native_retirement(
    tmp_path, monkeypatch, owned_console_databases, action
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    grace_entered = asyncio.Event()
    if action == "dispose_early":
        original_wait = runtime._bounded_wait

        async def observe_original_grace(*args, **kwargs):
            grace_entered.set()
            return await original_wait(*args, **kwargs)

        monkeypatch.setattr(runtime, "_bounded_wait", observe_original_grace)
    session = store.sessions()[0]
    record, _attachment_ref = _recovery_record(
        runtime, session.id, "held-ordinary-save"
    )
    entered, exited = threading.Event(), threading.Event()
    calls = []
    original_commit = store.commit_durable_turn

    def observe_commit(acceptance):
        calls.append(acceptance)
        entered.set()
        try:
            return original_commit(acceptance)
        finally:
            exited.set()

    monkeypatch.setattr(store, "commit_durable_turn", observe_commit)
    baseline_connections = db.registered_connection_count()
    runtime_ended = []
    original_end = store.end_app_runtime

    def observed_end():
        assert exited.is_set(), "runtime ended before native callback retirement"
        assert not store._durable_commit_in_flight
        runtime_ended.append(True)
        return original_end()

    monkeypatch.setattr(store, "end_app_runtime", observed_end)
    lock = _WriteLockHolder(tmp_path / "controller.sqlite")
    lock.__enter__()
    submit = asyncio.create_task(
        controller.submit_draft("held at close", session_id=session.id)
    )
    record.task = submit
    closing = None
    try:
        assert await _until(
            lambda: entered.is_set()
            and bool(store._durable_commit_in_flight)
            and db.registered_connection_count() == baseline_connections + 1,
            timeout=5,
        )
        assert db.registered_connection_count() == baseline_connections + 1
        preparation = store.preparation_for_session(session.id)
        assert preparation is not None and store._durable_commit_in_flight
        assert CONSOLE_SESSION_CLOSE_GRACE_SECONDS == 2.0
        assert CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS == 3.0
        if action == "close":
            revision = controller.lifecycle_impact(session_id=session.id).revision
            closing = asyncio.create_task(
                runtime.close_session(session.id, expected_revision=revision)
            )
            grace = CONSOLE_SESSION_CLOSE_GRACE_SECONDS
        else:
            closing = asyncio.create_task(runtime.dispose())
            grace = CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS
        if action == "dispose_early":
            # Cancel while the real original 3s wait still owns the held save.
            await asyncio.wait_for(grace_entered.wait(), 5)
        else:
            await asyncio.sleep(grace + 0.05)
        for _ in range(2):
            closing.cancel()
            await asyncio.sleep(0.05)
            assert not closing.done()
            assert not submit.done()
            assert not exited.is_set()
            assert db.registered_connection_count() == baseline_connections + 1
            assert store._durable_commit_in_flight
            assert store.preparation_by_id(preparation.preparation_id) is not None
            assert [item.id for item in store.sessions()] == [session.id]
            assert runtime.has_custodied_turns(session.id)
            assert record.request is not None
            assert runtime_ended == []
        lock.release()
        close_results = await asyncio.wait_for(
            asyncio.gather(closing, return_exceptions=True), 5
        )
        assert close_results[0] is None or isinstance(
            close_results[0], asyncio.CancelledError
        )
        result = await asyncio.wait_for(asyncio.shield(submit), 5)
        assert result.accepted is True and result.provider_started is False
        assert exited.is_set() and len(calls) == 1
        assert not store._durable_commit_in_flight
        assert db.registered_connection_count() == baseline_connections
        assert not runtime.has_custodied_turns(session.id)
        assert gateway.calls == 0
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert checkpoints == []
        assert next(row[2] for row in rows if row[0] == "assistant") == "stopped"
        if action == "close":
            assert store.sessions() == []
        else:
            assert runtime_ended == [True]
    finally:
        lock.release()
        await asyncio.gather(
            submit, *([closing] if closing is not None else []), return_exceptions=True
        )
        if action == "dispose_early":
            await runtime.dispose()


async def test_close_retains_issued_save_while_executor_work_is_still_queued(
    tmp_path, monkeypatch, owned_console_databases
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    assert (
        await controller.submit_draft("previous saved turn", session_id="session-1")
    ).accepted
    provider_calls_before = gateway.calls
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    session = store.sessions()[0]
    loop = asyncio.get_running_loop()
    original_executor = loop.run_in_executor
    queued = []
    pending = loop.create_future()

    def defer_owned(executor, callback, *args):
        if not queued and controller._ordinary_native_commit_tasks(session.id):
            queued.append((executor, callback, args))
            return pending
        return original_executor(executor, callback, *args)

    closing = None
    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_in_executor", defer_owned)
        submit = asyncio.create_task(
            controller.submit_draft("queued native save", session_id=session.id)
        )
        try:
            assert await _until(lambda: bool(queued), timeout=5)
            preparation = store.preparation_for_session(session.id)
            assert preparation is not None
            assert not store._durable_commit_in_flight
            identity = store._durable_identity_by_preparation[
                preparation.preparation_id
            ]
            owners = store._durable_owner_ids_by_preparation[preparation.preparation_id]
            assert (
                store._release_native_commit_owner(preparation.preparation_id, object())
                is False
            )
            with pytest.raises(RuntimeError):
                store.discard_uncommitted_durable_preparation(
                    preparation.preparation_id
                )
            assert (
                store._durable_identity_by_preparation[preparation.preparation_id]
                is identity
            )
            assert (
                store._durable_owner_ids_by_preparation[preparation.preparation_id]
                is owners
            )
            assert store.preparation_by_id(preparation.preparation_id) is preparation
            assert controller._ordinary_native_commit_tasks(session.id) == (submit,)
            revision = controller.lifecycle_impact(session_id=session.id).revision
            closing = asyncio.create_task(
                runtime.close_session(session.id, expected_revision=revision)
            )
            await asyncio.sleep(CONSOLE_SESSION_CLOSE_GRACE_SECONDS + 0.05)
            assert not closing.done() and not submit.done()
            assert store.preparation_by_id(preparation.preparation_id) is not None
            assert [item.id for item in store.sessions()] == [session.id]
            executor, callback, args = queued[0]
            try:
                value = await original_executor(executor, callback, *args)
            except BaseException as error:
                pending.set_exception(error)
            else:
                pending.set_result(value)
            await asyncio.wait_for(asyncio.shield(closing), 5)
            result = await asyncio.wait_for(asyncio.shield(submit), 5)
            assert result.accepted is True and result.provider_started is False
            assert gateway.calls == provider_calls_before
            assert store.sessions() == []
            assert _saved_state(tmp_path / "controller.sqlite")[0] == []
        finally:
            if not pending.done():
                if queued:
                    executor, callback, args = queued[0]
                    try:
                        pending.set_result(
                            await original_executor(executor, callback, *args)
                        )
                    except BaseException as error:
                        pending.set_exception(error)
                else:
                    pending.cancel()
            await asyncio.gather(
                submit,
                *([closing] if closing is not None else []),
                return_exceptions=True,
            )


async def test_normal_save_releases_native_owner_before_long_provider_response(
    tmp_path, monkeypatch, owned_console_databases
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    entered, release = asyncio.Event(), asyncio.Event()
    original = gateway.stream_chat

    async def held_provider(*args, **kwargs):
        entered.set()
        await release.wait()
        async for chunk in original(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(gateway, "stream_chat", held_provider)
    task = asyncio.create_task(
        controller.submit_draft("ordinary send", session_id="session-1")
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not task.done()
        assert controller._ordinary_native_commit_tasks() == ()
        assert not store._durable_commit_in_flight
        release.set()
        assert (await asyncio.wait_for(asyncio.shield(task), 5)).accepted is True
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_runtime_native_drain_ends_before_unrelated_submit_tail(
    tmp_path, monkeypatch, owned_console_databases
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    commit_entered, commit_release, _exited, _calls = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    provider_entered, provider_release = asyncio.Event(), asyncio.Event()
    original_provider = gateway.stream_chat

    async def held_provider(*args, **kwargs):
        provider_entered.set()
        await provider_release.wait()
        async for chunk in original_provider(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(gateway, "stream_chat", held_provider)
    submit = asyncio.create_task(
        controller.submit_draft("finite save, longer response", session_id="session-1")
    )
    drain = None
    try:
        assert await _until(commit_entered.is_set, timeout=5)
        assert controller._ordinary_native_commit_tasks("session-1") == (submit,)
        drain = asyncio.create_task(
            runtime._drain_ordinary_native_commits(controller, "session-1")
        )
        await asyncio.sleep(0)
        assert not drain.done()
        commit_release.set()
        await asyncio.wait_for(provider_entered.wait(), 5)
        assert controller._ordinary_native_commit_tasks("session-1") == ()
        assert not submit.done()
        assert await _until(
            drain.done, timeout=0.5
        ), "native drain retained the outer submit after its exact save owner retired"
        assert drain.result() is False
        assert not submit.done()
    finally:
        commit_release.set()
        provider_release.set()
        await asyncio.gather(
            submit, *([drain] if drain is not None else []), return_exceptions=True
        )
        await runtime.dispose()


async def test_cancelled_accepted_settlement_keeps_loop_and_native_owner_live_under_sqlite_lock(
    tmp_path, monkeypatch, owned_console_databases
):
    import time

    from Tests.Chat.test_console_durable_commit_offload import _hold_write_lock

    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    # Use the supported prepared-configuration route: this control measures
    # native settlement ownership, not cold configuration/import latency.
    configuration = await controller.capture_turn_configuration_snapshot("session-1")
    baseline_connections = db.registered_connection_count()
    commit_entered, commit_release, _exited, _calls = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    settlement_entered, settlement_exited = threading.Event(), threading.Event()
    original_settle = store.settle_accepted_durable_turn
    settlement_threads = []
    acquired = threading.Event()
    blocker = None

    def observe_settlement(*args, **kwargs):
        nonlocal blocker
        # Compete only after the original commit worker has closed its handle;
        # otherwise its TRUNCATE checkpoint consumes the lock interval first.
        blocker = threading.Thread(
            target=_hold_write_lock,
            args=(str(tmp_path / "controller.sqlite"), 2.0, acquired),
            daemon=True,
        )
        blocker.start()
        assert acquired.wait(timeout=5)
        settlement_threads.append(threading.get_ident())
        settlement_entered.set()
        try:
            return original_settle(*args, **kwargs)
        finally:
            settlement_exited.set()

    monkeypatch.setattr(store, "settle_accepted_durable_turn", observe_settlement)
    stop = asyncio.Event()
    stalls = []

    async def heartbeat():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            stalls.append(now - last)
            last = now

    monitor = None
    submit = asyncio.create_task(
        controller.submit_draft(
            "accepted before settlement contention",
            session_id="session-1",
            configuration=configuration,
        )
    )
    try:
        assert await _until(commit_entered.is_set, timeout=5)
        assert _saved_state(tmp_path / "controller.sqlite")[0] == [("accepted",)]
        # Measure only settlement after the actual saved checkpoint is visible.
        monitor = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.05)
        assert stalls, "heartbeat must run before terminal settlement starts"
        stalls.clear()
        submit.cancel()
        await asyncio.sleep(0.05)
        commit_release.set()
        assert await _until(settlement_entered.is_set, timeout=5)
        await asyncio.sleep(0.02)
        assert (
            max(stalls) < 0.5
        ), f"accepted settlement blocked the event loop for {max(stalls):.3f}s"
        assert blocker.is_alive(), "settlement did not reach the competing SQLite lock"
        assert len(settlement_threads) == 1
        assert settlement_threads[0] != threading.get_ident()
        for _ in range(2):
            submit.cancel()
            await asyncio.sleep(0.05)
            assert not submit.done()
            assert not settlement_exited.is_set()
            assert controller._ordinary_native_commit_tasks("session-1") == (submit,)
            assert db.registered_connection_count() == baseline_connections + 1
            assert gateway.calls == 0
        assert await _until(lambda: not blocker.is_alive(), timeout=5)
        result = await asyncio.wait_for(asyncio.shield(submit), 5)
        assert result.accepted is True and result.provider_started is False
        assert settlement_exited.is_set()
        assert controller._ordinary_native_commit_tasks("session-1") == ()
        assert db.registered_connection_count() == baseline_connections
        assert gateway.calls == 0
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert checkpoints == []
        assert next(row[2] for row in rows if row[0] == "assistant") == "failed"
        assert max(stalls) < 0.5
    finally:
        commit_release.set()
        stop.set()
        if monitor is not None:
            await monitor
        if blocker is not None:
            blocker.join(timeout=5)
        await asyncio.gather(submit, return_exceptions=True)
        await runtime.dispose()


async def test_error_after_actual_terminal_settlement_does_not_leave_preparation_wedged(
    tmp_path, monkeypatch, owned_console_databases
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus

    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    configuration = await controller.capture_turn_configuration_snapshot("session-1")
    entered, release, _exited, commit_calls = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    original_settle = store.settle_accepted_durable_turn
    settlement_results = []

    def committed_then_failed(*args, **kwargs):
        settled = original_settle(*args, **kwargs)
        settlement_results.append(settled)
        assert settled is True
        raise RuntimeError("custom settlement wrapper failed after durable success")

    monkeypatch.setattr(store, "settle_accepted_durable_turn", committed_then_failed)
    task = asyncio.create_task(
        controller.submit_draft(
            "one accepted terminal write",
            session_id="session-1",
            configuration=configuration,
        )
    )
    try:
        assert await _until(entered.is_set, timeout=5)
        preparation = store.preparation_for_session("session-1")
        assert preparation is not None
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assert fresh.execute(
                "SELECT state FROM console_dispatch_checkpoints"
            ).fetchall() == [("accepted",)]
            assistant_id, accepted_version = fresh.execute(
                "SELECT id, version FROM messages WHERE role='assistant'"
            ).fetchone()
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted is True and result.provider_started is False
        assert len(commit_calls) == 1 and settlement_results == [True]
        assert gateway.calls == 0
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assert (
                fresh.execute(
                    "SELECT COUNT(*) FROM console_dispatch_checkpoints"
                ).fetchone()[0]
                == 0
            )
            assert fresh.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
            terminal_state, terminal_version, metadata_json = fresh.execute(
                "SELECT assistant_generation_state, version, metadata_json FROM messages WHERE id=?",
                (assistant_id,),
            ).fetchone()
        assert terminal_state == "failed"
        assert (
            terminal_version == accepted_version + 1
        ), "terminal settlement was written twice"
        metadata = MessageMetadata.from_json(metadata_json)
        assert metadata is not None and metadata.terminal_receipt_id
        assert (
            store.preparation_by_id(preparation.preparation_id) is None
        ), "durable terminal success left an accepted preparation with no recovery action"
        assert (
            preparation.preparation_id
            not in controller._durable_postcommit_continuations
        )
        assert store.dispatch_recovery_for_session("session-1") is None
        assert controller._ordinary_native_commit_tasks("session-1") == ()
        assert result.terminal_status is ConsoleRunStatus.FAILED
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.dispose()


@pytest.mark.parametrize("route", ["queued", "cancelled-manual", "commit-error-manual"])
async def test_trace_failure_after_actual_commit_releases_owners_and_preserves_cancelled_draft(
    tmp_path, monkeypatch, owned_console_databases, route
):
    from Tests.Chat.test_console_trace_first_send_atomicity import (
        _fail_trace_request_once,
        _force_capture_on,
    )
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
        ConsoleSubmissionOrigin,
    )
    from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsolePreparationPauseKind,
        ConsoleTurnPreparationState,
        preparation_actions,
    )

    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    configuration = await controller.capture_turn_configuration_snapshot("session-1")
    _force_capture_on(monkeypatch)
    restore_failure = _fail_trace_request_once(
        "missing_repository", store=store, monkeypatch=monkeypatch
    )
    session = store.sessions()[0]
    session.draft = "keep the manual draft"
    cleared = []
    controller.on_submission_accepted = lambda: cleared.append(True)
    release = None
    task = None
    try:
        if route == "queued":
            coordinator = controller.prompt_queue_coordinator
            registry = coordinator.registry
            begun = registry.begin_chain(
                "session-1", context_epoch=0, expected_revision=0
            )
            admitted = registry.admit(
                "session-1",
                text="autonomous queued body",
                expected_revision=begun.snapshot.revision,
            )
            assert admitted.entry_id is not None
            coordinator._chains["session-1"] = _PromptChain()
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id="session-1"
            )
            submitted = []

            async def submit_queued(text, **kwargs):
                result = await controller.submit_draft(
                    text,
                    session_id=kwargs["session_id"],
                    origin=ConsoleSubmissionOrigin.QUEUED,
                    queue_entry_id=kwargs["entry_id"],
                    queue_authorization=kwargs["authorization"],
                    configuration=configuration,
                )
                submitted.append(result)
                return result

            coordinator._submit_queued = submit_queued
            await coordinator._drain_waiting("session-1", ConsoleRunStatus.COMPLETED)
            assert len(submitted) == 1
            result = submitted[0]
            assert result.accepted is True and result.provider_started is False
            assert result.terminal_status is ConsoleRunStatus.FAILED
            assert store.preparation_for_session("session-1") is None
            assert controller.trace_call_recovery_preparation() is None
            assert controller._durable_postcommit_continuations == {}
            assert "session-1" not in coordinator._chains
            assert registry.snapshot("session-1").total_count == 0
        else:
            entered, release, exited, calls = _hold_original_commit(
                monkeypatch, store, after_commit=True
            )
            if route == "commit-error-manual":
                held_commit = store.commit_durable_turn

                def committed_then_failed(*args, **kwargs):
                    held_commit(*args, **kwargs)
                    raise RuntimeError("callback failed after durable success")

                monkeypatch.setattr(store, "commit_durable_turn", committed_then_failed)
            task = asyncio.create_task(
                controller.submit_draft(
                    session.draft,
                    session_id="session-1",
                    configuration=configuration,
                )
            )
            assert await _until(entered.is_set, timeout=5)
            assert _saved_state(tmp_path / "controller.sqlite")[0] == [("accepted",)]
            # Re-entering identical text is still a fresh composer value.
            session.draft = ""
            session.draft = "keep the manual draft"
            if route == "cancelled-manual":
                task.cancel()
                await asyncio.sleep(0.05)
            assert not task.done()
            release.set()
            result = await asyncio.wait_for(asyncio.shield(task), 5)
            assert result.accepted is True and result.provider_started is False
            assert result.should_clear_draft is False
            assert session.draft == "keep the manual draft"
            assert cleared == []
            assert len(calls) == 1 and exited.is_set()
            checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
            assert len(rows) == 2
            if checkpoints:
                assert checkpoints == [("accepted",)]
                paused = store.preparation_for_session("session-1")
                assert paused is not None
                assert paused.state is ConsoleTurnPreparationState.PAUSED
                assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_PROVENANCE
                assert preparation_actions(paused) == (
                    "retry",
                    "send_without_capture",
                    "cancel",
                )
                assert (
                    paused.preparation_id
                    in controller._durable_postcommit_continuations
                )
            else:
                assert next(row[2] for row in rows if row[0] == "assistant") == "failed"
                assert store.preparation_for_session("session-1") is None
                assert controller._durable_postcommit_continuations == {}
        assert gateway.calls == 0
        assert controller._ordinary_native_commit_tasks("session-1") == ()
    finally:
        if release is not None:
            release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        restore_failure()
        await runtime.dispose()


async def test_revoked_continuation_rolls_back_real_acceptance_and_releases_native_owner(
    tmp_path, monkeypatch, owned_console_databases
):
    from uuid import uuid4

    from Tests.Agents.test_hooks_v2_execution import command
    from Tests.Chat.test_console_fleet_wake import _controller_rig
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationAdmissionRefused
    from tldw_chatbook.Agents.hooks_v2.engine import HookEventOutcome
    from tldw_chatbook.Agents.hooks_v2.validation import parse_result
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

    db, app, runs_db, store, session, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    owned_console_databases(db, controller)
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    authority = [True]
    handler = command(name="Stop", effects=["continuation"])
    engine = runtime.ensure_hooks_v2(session.id, (handler,), lambda *_: authority[0])
    original_fire = engine.fire_async
    produced = []

    async def checked_stop_proposal(event, *args, **kwargs):
        if event.event != "Stop":
            return await original_fire(event, *args, **kwargs)
        # Exercise the host's checked-proposal boundary without launching a
        # platform-dependent command. The scheduler and live authority remain real.
        result = parse_result(
            {
                "version": 2,
                "decision": "pass",
                "continuation": {"message": "revoked machine"},
            },
            handler,
        )
        produced.append(event.event_id)
        return HookEventOutcome(accepted=((handler.id, result),))

    monkeypatch.setattr(engine, "fire_async", checked_stop_proposal)
    original_current = engine.effects_current
    observed_authority = []

    def observe_current(*args, **kwargs):
        current = original_current(*args, **kwargs)
        observed_authority.append(current)
        return current

    monkeypatch.setattr(engine, "effects_current", observe_current)
    entered, release = threading.Event(), threading.Event()
    original_commit = store.commit_durable_turn
    machine_attempts, refusals = [], []

    def held_commit(acceptance):
        if acceptance.continuation_receipt is not None:
            machine_attempts.append(acceptance.preparation_id)
            # This handle belongs to the real finite worker scope. Keep it live
            # while authority is revoked, then enter the original transaction.
            db.get_connection()
            entered.set()
            assert release.wait(30)
        try:
            return original_commit(acceptance)
        except ContinuationAdmissionRefused as error:
            refusals.append(type(error))
            raise

    monkeypatch.setattr(store, "commit_durable_turn", held_commit)
    request = ConsoleTurnCustodyRequest(
        turn_id=str(uuid4()),
        session_id=session.id,
        draft="parent turn",
        configuration=controller.resolve_turn_configuration_snapshot(session.id),
    )
    baseline_connections = db.registered_connection_count()
    pending = None
    try:
        turn_id = runtime.accept_turn(request)
        pending = asyncio.create_task(runtime.wait_for_turn(turn_id))
        assert await asyncio.to_thread(entered.wait, 30)
        assert len(gateway.payloads) == 1
        assert len(machine_attempts) == 1 and len(produced) == 1
        assert True in observed_authority
        assert len(controller._ordinary_native_commit_tasks(session.id)) == 1
        assert db.registered_connection_count() == baseline_connections + 1
        assert not pending.done()
        authority[0] = False
        release.set()
        await pending
        assert refusals == [ContinuationAdmissionRefused]
        assert False in observed_authority
        assert len(gateway.payloads) == 1
        with sqlite3.connect(tmp_path / "chacha.sqlite") as fresh:
            assert (
                fresh.execute(
                    "SELECT COUNT(*) FROM messages WHERE role='user'"
                ).fetchone()[0]
                == 1
            )
            assert (
                fresh.execute(
                    "SELECT COUNT(*) FROM console_hook_continuation_receipts"
                ).fetchone()[0]
                == 0
            )
            assert (
                fresh.execute(
                    "SELECT COUNT(*) FROM console_dispatch_checkpoints"
                ).fetchone()[0]
                == 0
            )
        assert store.preparation_for_session(session.id) is None
        assert store.dispatch_recovery_for_session(session.id) is None
        assert controller._durable_postcommit_continuations == {}
        assert controller.prompt_queue_registry.snapshot(session.id).total_count == 0
        assert not controller.prompt_queue_coordinator._machine_entries
        assert controller._ordinary_native_commit_tasks(session.id) == ()
        assert db.registered_connection_count() == baseline_connections
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await runtime.close_hooks_v2()
        await runtime.dispose()
        runs_db.close()
