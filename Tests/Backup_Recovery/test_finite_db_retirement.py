"""Finite offloads retire real native handles on their executing threads."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.db_offload import run_db_off_loop
from tldw_chatbook.Workspaces import change_retention


def worker_leases(db):
    """Observe native ownership without borrowing the connection."""
    with storage._lock:
        return [
            lease for lease in storage._live_leases
            if lease.resource_path == db.db_path
            and lease.resource_thread is not threading.current_thread()
        ]


@pytest.fixture(params=["subscriptions", "conversations", "owned_notes", "owned_runs", "owned_workspaces"])
def offload(request, tmp_path, local_root, monkeypatch):
    if request.param.startswith("owned_"):
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.DB.base_db import run_owned_db_call
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

        if request.param == "owned_notes":
            db = CharactersRAGDB(tmp_path / "notes.db", "test")
            read = db.list_all_active_conversations
        elif request.param == "owned_runs":
            db = AgentRunsDB(tmp_path / "runs.db")
            read = db.pending_wake_conversation_ids
        else:
            db = WorkspaceDB(tmp_path / "workspaces.db")
            def read():
                with db.connection() as connection:
                    return connection.execute("SELECT 1").fetchone()[0]

        async def call(body):
            return await run_owned_db_call(db, body)
    elif request.param == "subscriptions":
        db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
        row_id = db.add_subscription("retained", "rss", "https://example.invalid/feed", auto_pause_threshold=3)

        def read():
            return db.get_subscription(row_id)["name"]

        async def call(body):
            return await run_db_off_loop(db, body)
    else:
        db = CharactersRAGDB(tmp_path / "conversations.db", "test")
        service = ChatConversationService(db)
        service.create_conversation(title="retained")
        scope = ChatConversationScopeService(local_service=service, server_service=None)
        original = service.list_conversations

        def read():
            return original()["items"][0]["title"]

        async def call(body):
            monkeypatch.setattr(service, "list_conversations", body)
            return await scope.list_conversations()
    try:
        yield db, call, read
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_finite_read_retires_native_handle_and_owner_can_be_reused(offload, fail):
    db, call, read = offload
    expected = read()
    error = ValueError("callback failed")

    def body():
        result = read()
        if fail:
            raise error
        return result

    if fail:
        with pytest.raises(ValueError) as caught:
            await call(body)
        assert caught.value is error
    else:
        assert await call(body) == expected
    assert not worker_leases(db)
    assert await call(read) == expected
    assert not worker_leases(db)
    assert read() == expected


@pytest.mark.asyncio
async def test_cancelled_awaiter_does_not_retire_running_native_work(offload):
    db, call, read = offload
    entered, finish, exited = threading.Event(), threading.Event(), threading.Event()

    def body():
        try:
            result = read()
            entered.set()
            assert finish.wait(3)
            return result
        finally:
            exited.set()

    task = asyncio.create_task(call(body))
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker_leases(db), "a cancelled awaiter cannot retire a running callback"
        finish.set()
        for _ in range(200):
            if not worker_leases(db):
                break
            await asyncio.sleep(0.005)
        assert not worker_leases(db)
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        # Wait for the actual finite callback even when an assertion fails.
        assert await asyncio.to_thread(exited.wait, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_app_retention_retires_its_constructed_database(tmp_path, local_root, monkeypatch, fail):
    observed = []
    run_ids = []
    report = change_retention.PruneReport(rows_pruned=7)

    def prune(db, service):
        observed.append(db)
        run_ids.append(db.create_run(conversation_id="kept", agent_kind="primary"))
        if fail:
            raise ValueError("prune failed")
        return report

    monkeypatch.setattr(change_retention, "prune_change_history", prune)
    # The pass prunes an existing runs DB only (TASK-34100.16.1).
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    AgentRunsDB(tmp_path / "agent_runs.db").close()
    result = await asyncio.to_thread(
        change_retention.run_retention_for_app,
        tmp_path / "chachanotes.db",
        service=SimpleNamespace(available=True),
    )
    assert result is (None if fail else report)
    assert not worker_leases(observed[0])
    # Retirement leaves committed rows and the owner usable on a later call.
    try:
        assert observed[0].get_run(run_ids[0])["conversation_id"] == "kept"
    finally:
        observed[0].close()


def test_app_retention_never_creates_an_absent_runs_db(tmp_path, local_root):
    """No runs DB means nothing to prune; the pass must not create one.

    TASK-34100.16.1: the startup pass ran 65 s after launch and created
    ``agent_runs.db`` on a profile that had none. Backup Review lists that
    store as unused, so a Review taken in the first minute was voided at
    Create with ``scope_changed``.
    """
    runs = tmp_path / "agent_runs.db"

    result = change_retention.run_retention_for_app(
        tmp_path / "chachanotes.db", service=SimpleNamespace(available=True)
    )

    assert result is None
    assert not runs.exists(), "the retention pass created the runs store"
    assert not list(tmp_path.glob("agent_runs.db*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_sync_log_retention_retires_its_worker_connections(
    tmp_path, local_root, monkeypatch, fail
):
    """The startup sync-log pass leaves no pool-thread handle holding storage.

    TASK-34100.16.1: each store's cached connection on an executor thread kept
    a storage lease that a backup's local pause can never drain, so every
    Create backup after the 65 s pass ended in ``admission_timeout``.
    """
    from loguru import logger

    from tldw_chatbook import app_lifecycle
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    chat = CharactersRAGDB(tmp_path / "chat.db", "test")
    media = MediaDatabase(tmp_path / "media.db", client_id="test")
    ran = []
    for db in (chat, media):
        original = db.delete_sync_log_entries_older_than

        def prune(cutoff, original=original, db=db):
            ran.append((db, threading.current_thread()))
            deleted = original(cutoff)
            if fail:
                raise ValueError("prune failed after its delete")
            return deleted

        monkeypatch.setattr(db, "delete_sync_log_entries_older_than", prune)
    monkeypatch.setattr(app_lifecycle, "get_cli_setting", lambda *args, **kwargs: 90)
    host = SimpleNamespace(chachanotes_db=chat, media_db=media, loguru_logger=logger)
    try:
        await app_lifecycle.LifecycleMixin._perform_sync_log_retention(host)

        assert [db for db, _ in ran] == ([chat] if fail else [chat, media])
        assert all(thread is not threading.current_thread() for _, thread in ran)
        assert not worker_leases(chat), "ChaChaNotes worker kept its connection"
        assert not worker_leases(media), "Media worker kept its connection"
    finally:
        chat.close_connection()
        media.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["custom", "subclass", "memory"])
async def test_subscription_unqualified_owner_keeps_previous_route(shape):
    class CustomDB:
        is_memory_db = False

        def close(self):
            raise AssertionError("unqualified close")

    class DerivedDB(SubscriptionsDB):
        def close(self):
            raise AssertionError("subclass close")

    db = object.__new__(DerivedDB) if shape == "subclass" else CustomDB()
    db.is_memory_db = shape == "memory"
    origin = threading.current_thread()
    result = await run_db_off_loop(db, threading.current_thread)
    assert (result is origin) is (shape == "memory")


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["custom", "service_subclass", "db_subclass", "memory", "coroutine", "server"])
async def test_conversation_unqualified_route_never_closes(shape):
    class DerivedService(ChatConversationService):
        pass

    class DerivedDB(CharactersRAGDB):
        pass

    def forbidden_close():
        raise AssertionError("unqualified close")

    db_type = DerivedDB if shape == "db_subclass" else CharactersRAGDB
    db = object.__new__(db_type)
    db.is_memory_db = shape == "memory"
    db.close_connection = forbidden_close
    service_type = DerivedService if shape == "service_subclass" else ChatConversationService
    service = object.__new__(service_type)
    service.db = db
    if shape == "custom":
        service = SimpleNamespace(db=db)
    origin = threading.current_thread()
    if shape == "coroutine":
        async def listing():
            return threading.current_thread()
    else:
        def listing():
            return threading.current_thread()
    service.list_conversations = listing
    scope = ChatConversationScopeService(local_service=service, server_service=service)
    result = await scope.list_conversations(mode="server" if shape == "server" else "local")
    assert (result is origin) is (shape in {"memory", "coroutine", "server"})


@pytest.mark.asyncio
async def test_conversation_reassignment_does_not_redirect_worker_cleanup(tmp_path, local_root):
    db = CharactersRAGDB(tmp_path / "original.db", "test")
    service = ChatConversationService(db)
    scope = ChatConversationScopeService(local_service=service, server_service=None)
    service.create_conversation(title="retained")
    listing = service.list_conversations
    entered, release = threading.Event(), threading.Event()
    replacement_closes = []

    def read_then_wait():
        result = listing()
        entered.set()
        assert release.wait(3)
        return result

    service.list_conversations = read_then_wait
    task = asyncio.create_task(scope.list_conversations())
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        service.db = SimpleNamespace(close_connection=lambda: replacement_closes.append(True))
        release.set()
        result = await task
        assert result["items"][0]["title"] == "retained"
        assert not replacement_closes
        assert not worker_leases(db)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("offload", ["owned_notes", "owned_runs", "owned_workspaces"], indirect=True)
async def test_owned_callback_remains_admitted_after_last_repository_read(offload):
    db, call, read = offload
    observed = []

    def body():
        result = read()
        with storage._lock:
            observed.append(any(
                operation.thread is threading.current_thread()
                and operation.participant.repository() is db
                for operation in storage._operations
            ))
        return result

    await call(body)
    assert observed == [True], "the complete callback must own admission"
    assert not worker_leases(db)
