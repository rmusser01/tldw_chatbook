"""Real chat SQLite authority and atomic temporary progress promotion."""

import json
import re

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Agents.fleet_messages import (
    MessageError,
    MessageIdentity,
    MessageStore,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def source(index=0, chain="chain"):
    return MessageIdentity(f"handle-{index}", f"run-{index}", "parent", chain, "reader")


def durable(db, conversation_id, owner="owner"):
    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository

    repository = FleetProgressRepository(db)
    store = MessageStore()
    inbox = store.open_inbox(
        owner,
        repository=repository,
        saved_conversation_id=conversation_id,
        messages=repository.load(conversation_id),
    )
    return store, inbox


@private_profile_test
def test_saved_reopen_preserves_fifo_and_revokes_old_capabilities(tmp_path, request):
    path = tmp_path / "progress.sqlite"
    db = CharactersRAGDB(path, "progress-test")
    conversation_id = ChatPersistenceService(db).create_conversation(
        conversation_title="Saved"
    )
    store, inbox = durable(db, conversation_id)
    ids = []
    for n, chain in enumerate([None, "foreign", "chain", "foreign", "chain"]):
        ids.append(inbox.sender(source(n, chain)).send(f"body-{n}"))
    assert all(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", item.created_at)
        for item in inbox.snapshot()
    )
    sender = inbox.sender(source(10))
    reader = inbox.reader("primary", chain_id="chain", automatic=True)
    store.close()
    db.close()
    with pytest.raises(MessageError, match="unavailable"):
        sender.send("stale")
    with pytest.raises(MessageError, match="unavailable"):
        reader.collect()
    db = CharactersRAGDB(path, "progress-test")
    try:
        connection = db.get_connection()
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'"
            ).fetchone()
            is None
        )
        statements = []
        connection.set_trace_callback(statements.append)
        try:
            reopened, restored = durable(db, conversation_id, "new-owner")
        finally:
            connection.set_trace_callback(None)
        query = next(
            sql
            for sql in statements
            if "FROM fleet_progress_messages WHERE conversation_id =" in sql
        )
        plan = [row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + query)]
        assert any(
            "SEARCH" in detail and "idx_fleet_progress_conversation_sequence" in detail
            for detail in plan
        ), plan
        assert not any("TEMP B-TREE" in detail for detail in plan), plan
        control = query.replace(
            "FROM fleet_progress_messages WHERE",
            "FROM fleet_progress_messages NOT INDEXED WHERE",
        )
        assert any(
            "SCAN fleet_progress_messages" in row[3]
            for row in connection.execute("EXPLAIN QUERY PLAN " + control)
        )
        assert restored.pending_metadata() == tuple(
            zip(
                ids,
                [
                    source(n, c)
                    for n, c in enumerate(
                        [None, "foreign", "chain", "foreign", "chain"]
                    )
                ],
            )
        )
        automatic = restored.reader("fresh", chain_id="chain", automatic=True)
        assert [
            r["body"] for r in json.loads(automatic.collect().content)["messages"]
        ] == ["body-2", "body-4"]
        automatic.close()
        assert [
            r["body"]
            for r in json.loads(
                restored.reader("manual", chain_id=None, automatic=False)
                .collect()
                .content
            )["messages"]
        ] == ["body-0", "body-1", "body-3"]
        reopened.close()
        assert durable(db, conversation_id)[1].snapshot() == ()
    finally:
        db.close()


@private_profile_test
def test_collection_checkpoint_ambiguity_stops_receipts_and_replay(
    tmp_path, monkeypatch, request
):
    db = CharactersRAGDB(tmp_path / "progress.sqlite", "progress-test")
    try:
        conversation_id = ChatPersistenceService(db).create_conversation(
            conversation_title="Saved"
        )
        _, inbox = durable(db, conversation_id)
        message_id = inbox.sender(source()).send("private report")
        reader = inbox.reader("primary", chain_id=None, automatic=False)
        original = inbox._repository.remove

        def committed_then_failed(*args):
            original(*args)
            raise RuntimeError("checkpoint uncertain")

        monkeypatch.setattr(inbox._repository, "remove", committed_then_failed)
        with pytest.raises(MessageError, match="durable_unavailable"):
            reader.collect()
        with pytest.raises(MessageError, match="durable_unavailable"):
            reader.collect()
        assert inbox.pending_metadata() == ((message_id, source()),)
        assert durable(db, conversation_id)[1].snapshot() == ()
    finally:
        db.close()


@private_profile_test
def test_explicit_save_commits_chat_and_pending_reports_atomically(
    tmp_path, request, monkeypatch
):
    db = CharactersRAGDB(tmp_path / "progress.sqlite", "progress-test")
    try:
        native = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = native.create_session(
            ephemeral=True,
            settings=ConsoleSessionSettings(provider="openai", model="test"),
        )
        store = MessageStore()
        native.register_progress_message_store(store)
        inbox = store.open_inbox(native.progress_owner_id(session.id))
        message_id = inbox.sender(source()).send("body capture remains off")
        created_at = inbox.snapshot()[0].created_at
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM fleet_progress_messages")
            .fetchone()[0]
            == 0
        )

        from tldw_chatbook.DB.fleet_progress_repository import (
            FleetProgressPromotionContribution,
        )

        original_write = FleetProgressPromotionContribution.write

        def rollback_after_reports(contribution, **kwargs):
            original_write(contribution, **kwargs)
            assert (
                db.get_connection()
                .execute("SELECT count(*) FROM fleet_progress_messages")
                .fetchone()[0]
                == 1
            )
            assert not native._progress_identity_lock._is_owned()
            with pytest.raises(MessageError, match="saving"):
                inbox.sender(source(9)).send("must not join a frozen save")
            raise RuntimeError("cancel save")

        with monkeypatch.context() as patch:
            patch.setattr(
                FleetProgressPromotionContribution, "write", rollback_after_reports
            )
            with pytest.raises(RuntimeError, match="cancel save"):
                native.promote_ephemeral_session(session.id)
        assert session.ephemeral
        assert inbox.pending_metadata() == ((message_id, source()),)
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM fleet_progress_messages")
            .fetchone()[0]
            == 0
        )
        conversation_id = native.promote_ephemeral_session(session.id)
        assert not session.ephemeral
        assert (
            db.get_connection()
            .execute(
                "SELECT created_at FROM fleet_progress_messages WHERE message_id = ?",
                (message_id,),
            )
            .fetchone()[0]
            == created_at
        )
        assert native.promote_ephemeral_session(session.id) is None
        assert durable(db, conversation_id)[1].snapshot() == inbox.snapshot()
        inbox.sender(source(1)).send("after save")
        store.close()
        assert len(durable(db, conversation_id)[1].snapshot()) == 2
    finally:
        db.close()


@private_profile_test
def test_chat_v74_installed_backup_and_shared_subscription_schema(tmp_path, request):
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.recovery_operations import recovery_adapters
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    path = tmp_path / "progress.sqlite"
    db = CharactersRAGDB(path, "progress-test")
    try:
        actual = tuple(
            row[0]
            for row in db.get_connection().execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        core = next(
            a for a in core_adapters() if a.owner_id == "db.chachanotes.primary"
        )
        assert core.schema_policy().schema_sql[0] == (74, actual)
        assert core.validate(path) == ()
        subscriptions = SubscriptionsDB(path)
        try:
            shared_actual = tuple(
                row[0]
                for row in db.get_connection().execute(
                    "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                )
            )
            adapter = next(
                a for a in recovery_adapters() if a.owner_id == "db.subscriptions"
            )
            assert shared_actual in tuple(
                schema for _, schema in adapter.schema_policy().schema_sql
            )
            assert adapter.validate(path) == ()
        finally:
            subscriptions.close()
    finally:
        db.close()


@private_profile_test
def test_durable_limits_discard_and_committed_metadata_observation(tmp_path, request):
    db = CharactersRAGDB(tmp_path / "progress.sqlite", "progress-test")
    try:
        conversation_id = ChatPersistenceService(db).create_conversation(
            conversation_title="Saved"
        )
        store, inbox = durable(db, conversation_id)
        observed = []

        def committed(owner, message_id, identity):
            assert store._lock.acquire(blocking=False)
            store._lock.release()
            assert (
                db.get_connection()
                .execute(
                    "SELECT message_id FROM fleet_progress_messages WHERE message_id = ?",
                    (message_id,),
                )
                .fetchone()[0]
                == message_id
            )
            observed.append((owner, message_id, identity))

        store.on_enqueue = committed
        sender = inbox.sender(source())
        ids = [sender.send("private report") for _ in range(8)]
        with pytest.raises(MessageError, match="queue_full"):
            sender.send("refused")
        assert len(observed) == 8
        assert [item[1] for item in observed] == ids
        assert inbox.discard([ids[0]]) == 1
        assert inbox._senders[source().handle_id].accepted_count == 8
        assert [
            m.message_id for m in durable(db, conversation_id)[1].snapshot()
        ] == ids[1:]
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM fleet_progress_messages")
            .fetchone()[0]
            == 7
        )
    finally:
        db.close()


@private_profile_test
def test_saved_native_close_replacement_and_prepare_owner_fence(
    tmp_path, monkeypatch, request
):
    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository

    db = CharactersRAGDB(tmp_path / "progress.sqlite", "progress-test")
    try:
        persistence = ChatPersistenceService(db)
        conversation_id = persistence.create_conversation(conversation_title="Saved")
        _, original = durable(db, conversation_id)
        message_id = original.sender(source()).send("saved report")
        native = ConsoleChatStore(persistence=persistence)
        session = native.restore_persisted_session(
            title="Saved",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
        )
        store = MessageStore()
        native.register_progress_message_store(store)
        old_owner = native.progress_owner_id(session.id)
        old_inbox = store.get_inbox(old_owner)
        reader = old_inbox.reader("old", chain_id=None, automatic=False)
        replacement_store = MessageStore()
        hints = []

        def loaded_hint(owner, report_id, identity):
            assert not native._progress_identity_lock._is_owned()
            assert replacement_store._lock.acquire(blocking=False)
            replacement_store._lock.release()
            hints.append((owner, report_id, identity))

        replacement_store.on_enqueue = loaded_hint
        native.register_progress_message_store(replacement_store)
        assert hints == [(old_owner, message_id, source())]
        with pytest.raises(MessageError, match="unavailable"):
            reader.collect()
        assert replacement_store.pending_counts() == {old_owner: 1}
        native.close_session(session.id)
        assert replacement_store.pending_counts() == {}
        assert durable(db, conversation_id)[1].pending_metadata() == (
            (message_id, source()),
        )

        session = native.restore_persisted_session(
            title="Saved",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
        )
        replacement_store.close_inbox(native.progress_owner_id(session.id))
        original_load = FleetProgressRepository.load

        def replace_during_load(repository, target):
            assert not native._progress_identity_lock._is_owned()
            reports = original_load(repository, target)
            native.close_session(session.id)
            native.create_session(session_id=session.id, ephemeral=True)
            return reports

        monkeypatch.setattr(FleetProgressRepository, "load", replace_during_load)
        assert native.prepare_progress_inbox(session.id) is None
        assert replacement_store.pending_counts() == {}
    finally:
        db.close()


@private_profile_test
def test_existing_v73_migrates_and_progress_metadata_never_exposes_bodies(
    tmp_path, request
):
    from loguru import logger

    class V73ChatDB(CharactersRAGDB):
        _CURRENT_SCHEMA_VERSION = 73

    path = tmp_path / "existing.sqlite"
    old = V73ChatDB(path, "progress-test")
    conversation_id = ChatPersistenceService(old).create_conversation(
        conversation_title="Saved"
    )
    old.close()
    db = CharactersRAGDB(path, "progress-test")
    try:
        assert (
            db.get_connection()
            .execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (db._SCHEMA_NAME,),
            )
            .fetchone()[0]
            == 74
        )
        store, inbox = durable(db, conversation_id)
        logs = []
        token = logger.add(lambda event: logs.append(str(event)))
        try:
            secret = "PRIVATE-INBOX-CAPTURE-OFF-42917"
            sender = inbox.sender(source())
            message_id = sender.send(secret)
            assert secret not in repr(store.pending_counts())
            assert secret not in repr(inbox.pending_metadata())
            assert inbox.snapshot()[0].body == secret
            assert (
                db.get_connection()
                .execute(
                    "SELECT body FROM fleet_progress_messages WHERE message_id = ?",
                    (message_id,),
                )
                .fetchone()[0]
                == secret
            )
            columns = [
                row[1]
                for row in db.get_connection().execute(
                    "PRAGMA table_info(fleet_progress_messages)"
                )
            ]
            assert set(columns) == {
                "sequence",
                "message_id",
                "conversation_id",
                "handle_id",
                "run_id",
                "parent_run_id",
                "chain_id",
                "agent",
                "body",
                "created_at",
            }
            assert secret not in "".join(logs)
        finally:
            logger.remove(token)
    finally:
        db.close()


@private_profile_test
def test_runtime_capacity_defers_saved_bodies_without_losing_navigation_counts(
    tmp_path, request
):
    db = CharactersRAGDB(tmp_path / "progress.sqlite", "progress-test")
    try:
        persistence = ChatPersistenceService(db)
        native = ConsoleChatStore(persistence=persistence)
        sessions = []
        for queue in range(9):
            conversation_id = persistence.create_conversation(
                conversation_title=f"Saved {queue}"
            )
            history, inbox = durable(db, conversation_id)
            for child in range(4):
                sender = inbox.sender(source(queue * 10 + child))
                for report in range(8):
                    sender.send(f"private saved report {queue}/{child}/{report}")
            history.close()
            sessions.append(
                native.restore_persisted_session(
                    title=f"Saved {queue}",
                    workspace_id=None,
                    persisted_conversation_id=conversation_id,
                    all_nodes=[],
                )
            )
        runtime = MessageStore()
        native.register_progress_message_store(runtime)
        owners = [native.progress_owner_id(session.id) for session in sessions]
        assert runtime.pending_counts() == dict.fromkeys(owners, 32)
        assert runtime._pending_count == 256
        assert runtime.get_inbox(owners[-1]) is None
        with pytest.raises(MessageError, match="queue_full"):
            native.prepare_progress_inbox(sessions[-1].id)
        assert "private saved report" not in repr(runtime.pending_counts())
        native.close_session(sessions[0].id)
        assert native.prepare_progress_inbox(sessions[-1].id) == owners[-1]
        assert len(runtime.get_inbox(owners[-1]).snapshot()) == 32
        assert runtime._pending_count == 256
        assert owners[0] not in runtime.pending_counts()
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM fleet_progress_messages")
            .fetchone()[0]
            == 288
        )

        deleted = persistence.create_conversation(conversation_title="Deleted")
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE conversations SET deleted = 1 WHERE id = ?", (deleted,)
            )
        unavailable = []
        for conversation_id in (deleted, "missing-conversation"):
            session = native.create_session()
            session.persisted_conversation_id = conversation_id
            unavailable.append(session)
        successor = MessageStore()
        native.register_progress_message_store(successor)
        assert successor._pending_count == 256
        for session in unavailable:
            assert successor.get_inbox(native.progress_owner_id(session.id)) is None
            with pytest.raises(MessageError, match="unavailable"):
                native.prepare_progress_inbox(session.id)
    finally:
        db.close()


@private_profile_test
def test_loaded_hints_leave_bridge_initialization_lock_before_reentrant_metadata(
    tmp_path, request
):
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = CharactersRAGDB(tmp_path / "chat.sqlite", "progress-test")
    runs = AgentRunsDB(tmp_path / "runs.sqlite", client_id="progress-test")
    try:
        conversation_id = ChatPersistenceService(db).create_conversation(
            conversation_title="Saved"
        )
        history, inbox = durable(db, conversation_id)
        message_id = inbox.sender(source()).send("private loaded report")
        history.close()
        native = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = native.restore_persisted_session(
            title="Saved",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs, store=native, provider_gateway=object()
        )
        observed = []

        def consume(owner, report_id, identity):
            observed.append((bridge._message_store_lock._is_owned(), report_id))
            assert bridge.message_store.get_inbox(owner) is not None
            assert bridge.progress_pending_metadata(session.id) == (
                (message_id, source()),
            )

        bridge.on_progress_enqueued("reentrant-metadata", consume)
        active_store = bridge.message_store
        assert active_store.get_inbox(native.progress_owner_id(session.id)) is not None
        assert observed == [(False, message_id)]
    finally:
        runs.close()
        db.close()


@pytest.mark.parametrize("borrowed", [False, True])
@private_profile_test
async def test_threaded_saved_child_report_retires_only_new_chat_db_cache(
    tmp_path, request, monkeypatch, borrowed
):
    import threading
    import time

    from Tests.Agents.conftest import join_fleet_children, pin_agent_settings
    from Tests.Agents.test_agent_service import FleetChat, fence
    from Tests.Agents.test_fleet_runtime import FLEET_CFG
    from tldw_chatbook.Agents import agent_service, run_log
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository

    pin_agent_settings(monkeypatch, run_log_enabled=False, subagents_outlive_turn=False)
    monkeypatch.setattr(run_log, "_setting", agent_service._setting)
    db = CharactersRAGDB(tmp_path / "child-chat.sqlite", "progress-test")
    runs = AgentRunsDB(tmp_path / "child-runs.sqlite", client_id="progress-test")
    saved_id = ChatPersistenceService(db).create_conversation(
        conversation_title="Saved"
    )
    messages = MessageStore()
    inbox = messages.open_inbox(
        "exact-owner",
        repository=FleetProgressRepository(db),
        saved_conversation_id=saved_id,
    )
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic, message_inbox=inbox)
    main_connection = db.get_connection()
    prior = []
    observed = []

    def report():
        if borrowed:
            prior.append(db.get_connection())
        return fence("report_to_supervisor", {"message": "PRIVATE-SAVED-CHILD-REPORT"})

    def child_done():
        try:
            observed.append(
                (threading.current_thread(), getattr(db._local, "conn", None))
            )
            return "A done"
        finally:
            # The prior caller owns borrowed retirement; also clean the RED leak.
            db.close_connection()

    chat = FleetChat(
        [
            fence("spawn_subagent", {"task": "A"}),
            fence("wait_agents", {}),
            "done",
        ],
        {"A": [report, child_done]},
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db=runs, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    try:
        _, outcome = service.run_turn(
            conversation_id=saved_id,
            messages=[],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        join_fleet_children(service)
        assert outcome.status == "done", outcome.steps
        assert len(observed) == 1
        thread, cached = observed[0]
        assert thread.name.startswith("fleet-") and not thread.is_alive()
        assert cached is (prior[0] if borrowed else None), (
            "threaded saved report retained its newly opened Chat DB cache"
        )
        receipt = chat.child_calls["A"][1]["messages_payload"][-1]["content"]
        assert "queued" in receipt and "PRIVATE-SAVED-CHILD-REPORT" not in receipt
        assert [message.body for message in inbox.snapshot()] == [
            "PRIVATE-SAVED-CHILD-REPORT"
        ]
        assert main_connection.execute("SELECT 1").fetchone()[0] == 1
        messages.close()
        db.close()
        assert not db._maintenance_participant.connections
        participant = db._maintenance_participant
        participant.close_admission()
        try:
            assert participant.drain(time.monotonic() + 0.05)
        finally:
            participant.resume()
    finally:
        join_fleet_children(service)
        messages.close()
        runs.close()
        db.close()
