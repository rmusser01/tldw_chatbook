"""Finite ordinary-save ownership and exact ACCEPTED settlement controls."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import replace
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_durable_turn_acceptance import _ready_store
from Tests.Chat.test_console_first_send_atomicity import _until
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationTransition,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


async def test_owned_native_commit_drains_original_callback_and_connection_after_global_cancel(
    tmp_path, monkeypatch, owned_console_databases
):
    from tldw_chatbook.Chat.console_native_commit import commit_durable_turn_owned

    db, _service, store, _preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    baseline_connections = db.registered_connection_count()
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    original = store.commit_durable_turn
    observed = []
    request_context = ContextVar("ordinary-native-test-context", default="missing")
    token = request_context.set("captured")

    def held(exact_acceptance):
        db.get_connection()
        observed.append(
            (exact_acceptance, request_context.get(), threading.get_ident())
        )
        entered.set()
        try:
            assert release.wait(10)
            return original(exact_acceptance)
        finally:
            exited.set()

    monkeypatch.setattr(store, "commit_durable_turn", held)
    existing_tasks = asyncio.all_tasks()
    task = asyncio.create_task(commit_durable_turn_owned(store, acceptance))
    try:
        assert await _until(entered.is_set, timeout=5)
        assert db.registered_connection_count() == baseline_connections + 1
        # Include any helper-created Task: a cancelled to_thread wrapper must
        # never masquerade as native completion during global shutdown.
        for _ in range(2):
            for owned_task in asyncio.all_tasks() - existing_tasks:
                owned_task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done()
            assert not exited.is_set()
            assert db.registered_connection_count() == baseline_connections + 1
        release.set()
        completion = await asyncio.wait_for(asyncio.shield(task), 5)
        assert completion.caller_cancelled is True
        assert completion.error is None
        assert completion.commit.assistant_message_id == acceptance.assistant_message_id
        assert exited.is_set()
        assert db.registered_connection_count() == baseline_connections
        assert len(observed) == 1
        assert observed[0][:2] == (acceptance, "captured")
        assert observed[0][2] != threading.get_ident()
        with sqlite3.connect(tmp_path / "acceptance.sqlite") as fresh:
            assert fresh.execute(
                "SELECT state FROM console_dispatch_checkpoints"
            ).fetchall() == [("accepted",)]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        request_context.reset(token)


@pytest.mark.parametrize("raises", [False, True])
async def test_memory_commit_preserves_inline_affinity_and_exact_custom_outcome(
    monkeypatch, owned_console_databases, raises
):
    from tldw_chatbook.Chat.console_native_commit import commit_durable_turn_owned

    db = CharactersRAGDB(":memory:", client_id="ordinary-memory-owner")
    owned_console_databases(db)
    acceptance, committed = object(), object()
    failure = RuntimeError("custom save failure")
    observed = []

    def custom(exact_acceptance):
        observed.append((exact_acceptance, threading.get_ident(), db.get_connection()))
        if raises:
            raise failure
        return committed

    store = SimpleNamespace(
        persistence=SimpleNamespace(db=db), commit_durable_turn=custom
    )
    connection = db.get_connection()
    loop = asyncio.get_running_loop()

    def no_executor(*_args, **_kwargs):
        pytest.fail("memory commit must retain its original connection/thread")

    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_in_executor", no_executor)
        completion = await commit_durable_turn_owned(store, acceptance)
    assert observed == [(acceptance, threading.get_ident(), connection)]
    assert completion.caller_cancelled is False
    assert completion.commit is (None if raises else committed)
    assert completion.error is (failure if raises else None)


@pytest.mark.parametrize("replacement", ["callback", "persistence", "database"])
async def test_queued_native_commit_captures_source_and_refuses_rebound_storage(
    tmp_path, monkeypatch, owned_console_databases, replacement
):
    from tldw_chatbook.Chat.console_native_commit import commit_durable_turn_owned

    db, service, store, _preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    original_callback = store.commit_durable_turn
    calls, successor_calls, queued = [], [], []
    loop = asyncio.get_running_loop()
    original_executor = loop.run_in_executor
    pending = loop.create_future()

    def custom(exact_acceptance):
        calls.append(exact_acceptance)
        return original_callback(exact_acceptance)

    def defer(executor, callback, *args):
        queued.append((executor, callback, args))
        return pending

    monkeypatch.setattr(store, "commit_durable_turn", custom)
    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_in_executor", defer)
        task = asyncio.create_task(commit_durable_turn_owned(store, acceptance))
        assert await _until(lambda: bool(queued), timeout=5)
    original_persistence = store.persistence
    try:
        if replacement == "callback":
            monkeypatch.setattr(
                store,
                "commit_durable_turn",
                lambda value: successor_calls.append(value),
            )
        elif replacement == "persistence":
            monkeypatch.setattr(store, "persistence", SimpleNamespace(db=db))
        else:
            monkeypatch.setattr(service, "db", SimpleNamespace(is_memory_db=False))
        executor, callback, args = queued[0]
        try:
            value = await original_executor(executor, callback, *args)
        except BaseException as error:
            pending.set_exception(error)
        else:
            pending.set_result(value)
        completion = await asyncio.wait_for(asyncio.shield(task), 5)
        assert successor_calls == []
        if replacement == "callback":
            assert calls == [acceptance]
            assert completion.error is None
            assert (
                completion.commit.assistant_message_id
                == acceptance.assistant_message_id
            )
        else:
            assert calls == []
            assert completion.commit is None
            assert completion.error is not None
            with sqlite3.connect(tmp_path / "acceptance.sqlite") as fresh:
                assert fresh.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    finally:
        store.persistence = original_persistence
        service.db = db
        if not pending.done():
            pending.set_exception(RuntimeError("test cleanup before issuance"))
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("queues_first", [False, True])
async def test_executor_submission_failure_never_allows_late_native_entry(
    monkeypatch, queues_first
):
    from tldw_chatbook.Chat.console_native_commit import commit_durable_turn_owned

    calls, queued = [], []
    acceptance = object()
    store = SimpleNamespace(
        persistence=SimpleNamespace(db=None),
        commit_durable_turn=lambda value: calls.append(value),
    )
    error = RuntimeError("executor cannot start worker")
    loop = asyncio.get_running_loop()

    def failed_submit(_executor, callback, *args):
        if queues_first:
            queued.append((callback, args))
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(loop, "run_in_executor", failed_submit)
        completion = await commit_durable_turn_owned(store, acceptance)
    assert completion.commit is None
    assert completion.error is error
    assert completion.caller_cancelled is False
    # Model ThreadPoolExecutor.submit enqueuing work before worker startup
    # raises: the abandoned work item must not reach the user's callback.
    for callback, args in queued:
        try:
            callback(*args)
        except BaseException:
            pass
    assert calls == []


def _publish_accepted(store, preparation, acceptance):
    commit = store.commit_durable_turn(acceptance)
    store.publish_durable_turn_identity(preparation.session_id, commit)
    store.publish_durable_turn_owners(preparation.session_id, commit)
    accepted = store.compare_and_set_preparation(
        preparation.session_id,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=ConsoleTurnPreparationState.COMMITTING,
            new_state=ConsoleTurnPreparationState.ACCEPTED,
            pause_kind=None,
            new_attempt_id=None,
        ),
    )
    assert accepted is not None
    fingerprint = store.durable_acceptance_fingerprint_for(preparation.preparation_id)
    assert fingerprint is not None
    return commit, fingerprint


@pytest.mark.parametrize("terminal_state", ["stopped", "failed"])
async def test_accepted_settlement_atomically_finishes_without_dispatch_generation(
    tmp_path, owned_console_databases, terminal_state
):
    db, _service, store, preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    commit, fingerprint = _publish_accepted(store, preparation, acceptance)
    with sqlite3.connect(tmp_path / "acceptance.sqlite") as fresh:
        assert fresh.execute(
            "SELECT state FROM console_dispatch_checkpoints"
        ).fetchall() == [("accepted",)]
        assert fresh.execute(
            "SELECT assistant_generation_state FROM messages WHERE role='assistant'"
        ).fetchall() == [("accepted",)]
    tokens_before = dict(store._dispatch_recovery_generation_tokens)
    assert (
        store.settle_accepted_durable_turn(
            preparation.preparation_id,
            fingerprint=fingerprint,
            terminal_state=terminal_state,
            content="Cancelled before dispatch.",
        )
        is True
    )
    assert store._dispatch_recovery_generation_tokens == tokens_before == {}
    with sqlite3.connect(tmp_path / "acceptance.sqlite") as fresh:
        assert (
            fresh.execute(
                "SELECT COUNT(*) FROM console_dispatch_checkpoints"
            ).fetchone()[0]
            == 0
        )
        row = fresh.execute(
            "SELECT assistant_generation_state, content, metadata_json FROM messages WHERE id=?",
            (commit.assistant_message_id,),
        ).fetchone()
        assert row[:2] == (terminal_state, "Cancelled before dispatch.")
        metadata = MessageMetadata.from_json(row[2])
        assert bool(metadata and metadata.terminal_receipt_id) is (
            terminal_state == "failed"
        )
        assert (
            fresh.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[
                0
            ]
            == 1
        )
    assert store.dispatch_recovery_for_session(preparation.session_id) is None


@pytest.mark.parametrize("changed", ["fingerprint", "checkpoint", "assistant"])
async def test_accepted_settlement_refuses_changed_exact_owner_without_deleting_checkpoint(
    tmp_path, owned_console_databases, changed
):
    db, _service, store, preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    commit, fingerprint = _publish_accepted(store, preparation, acceptance)
    if changed == "fingerprint":
        fingerprint = replace(fingerprint, digest="changed")
    elif changed == "checkpoint":
        db.get_connection().execute(
            "UPDATE console_dispatch_checkpoints SET checkpoint_revision=checkpoint_revision+1"
        )
    else:
        assistant = db.get_message_by_id(commit.assistant_message_id)
        assert assistant is not None
        assert db.update_message(
            commit.assistant_message_id,
            {"content": "newer body"},
            expected_version=assistant["version"],
        )
        with sqlite3.connect(tmp_path / "acceptance.sqlite") as fresh:
            assert fresh.execute(
                "SELECT content, version FROM messages WHERE id=?",
                (commit.assistant_message_id,),
            ).fetchone() == ("newer body", assistant["version"] + 1)
    assert (
        store.settle_accepted_durable_turn(
            preparation.preparation_id,
            fingerprint=fingerprint,
            terminal_state="stopped",
            content="must not replace newer state",
        )
        is False
    )
    with sqlite3.connect(tmp_path / "acceptance.sqlite") as fresh:
        assert fresh.execute(
            "SELECT state FROM console_dispatch_checkpoints"
        ).fetchall() == [("accepted",)]
        assert (
            fresh.execute(
                "SELECT assistant_generation_state FROM messages WHERE id=?",
                (commit.assistant_message_id,),
            ).fetchone()[0]
            == "accepted"
        )
    assert store.dispatch_recovery_for_session(preparation.session_id) is not None


async def test_accepted_settlement_write_failure_keeps_original_recovery_and_checkpoint(
    tmp_path, owned_console_databases
):
    db, _service, store, preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    commit, fingerprint = _publish_accepted(store, preparation, acceptance)
    db.get_connection().execute(
        "CREATE TRIGGER fail_native_accepted_settlement BEFORE UPDATE ON messages "
        "WHEN NEW.assistant_generation_state='failed' "
        "BEGIN SELECT RAISE(ABORT, 'settlement unavailable'); END"
    )
    assert (
        store.settle_accepted_durable_turn(
            preparation.preparation_id,
            fingerprint=fingerprint,
            terminal_state="failed",
            content="Interrupted before dispatch.",
        )
        is False
    )
    with sqlite3.connect(tmp_path / "acceptance.sqlite") as fresh:
        assert fresh.execute(
            "SELECT state FROM console_dispatch_checkpoints"
        ).fetchall() == [("accepted",)]
        assert fresh.execute(
            "SELECT assistant_generation_state, content FROM messages WHERE id=?",
            (commit.assistant_message_id,),
        ).fetchone() == ("accepted", "")
    recovery = store.dispatch_recovery_for_session(preparation.session_id)
    assert (
        recovery is not None
        and recovery.assistant_message_id == commit.assistant_message_id
    )


async def test_storage_replacement_during_original_admission_cannot_enter_successor(
    tmp_path, monkeypatch, owned_console_databases
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import participants
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_native_commit import commit_durable_turn_owned

    db, original_service, store, _preparation, acceptance = _ready_store(tmp_path)
    successor_db = CharactersRAGDB(
        tmp_path / "successor.sqlite", client_id="native-source-replacement"
    )
    successor_service = ChatPersistenceService(successor_db)
    owned_console_databases(db)
    owned_console_databases(successor_db)
    baseline_connections = db.registered_connection_count()
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    original_admission = participants._core_operation
    original_callback = store.commit_durable_turn
    callback_calls = []
    paused = False

    @contextmanager
    def held_original_admission(database):
        nonlocal paused
        this_entry = False
        try:
            with original_admission(database):
                if database is db and not paused:
                    paused = this_entry = True
                    db.get_connection()
                    entered.set()
                    assert release.wait(10)
                yield
        finally:
            if this_entry:
                retired.set()

    def observed_callback(exact_acceptance):
        callback_calls.append(exact_acceptance)
        return original_callback(exact_acceptance)

    monkeypatch.setattr(participants, "_core_operation", held_original_admission)
    monkeypatch.setattr(store, "commit_durable_turn", observed_callback)
    task = asyncio.create_task(commit_durable_turn_owned(store, acceptance))
    try:
        assert await _until(entered.is_set, timeout=5)
        assert db.registered_connection_count() == baseline_connections + 1
        assert callback_calls == []
        store.persistence = successor_service
        release.set()
        completion = await asyncio.wait_for(asyncio.shield(task), 5)
        with sqlite3.connect(tmp_path / "successor.sqlite") as fresh:
            successor_messages = fresh.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
        with sqlite3.connect(tmp_path / "acceptance.sqlite") as fresh:
            original_messages = fresh.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
        assert callback_calls == [] and successor_messages == 0, (
            "post-admission source replacement entered the captured callback "
            f"{len(callback_calls)} time(s) and wrote {successor_messages} successor messages"
        )
        assert original_messages == 0
        assert completion.commit is None and completion.error is not None
        assert retired.is_set()
        assert db.registered_connection_count() == baseline_connections
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        store.persistence = original_service
