"""An unresolved durable send must survive stale navigation cursor writes."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _restored_store,
    _start,
    _NoReplayGateway,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleDispatchRecoveryKind
from tldw_chatbook.DB.ChaChaNotes_DB import InputError


@pytest.fixture
def database():
    result = _database(":memory:")
    try:
        yield result
    finally:
        result[0].close_connection()


def _raw_semantic_corruption(db, sql, params):
    connection = db.get_connection()
    authorization = db._semantic_mutation_authorization_for_coordinator(connection)
    connection.create_function("console_semantic_mutation_authorized", 2, lambda *_: 1)
    try:
        with db.transaction(immediate=True) as cursor:
            cursor.execute(sql, params)
    finally:
        connection.create_function(
            "console_semantic_mutation_authorized", 2, authorization._sqlite_authorized
        )


def _stranded(database):
    db, conversation_id, repository = database
    greeting = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "greeting",
        }
    )
    checkpoint = _insert(
        db,
        repository,
        replace(
            _acceptance(conversation_id),
            parent_message_id=greeting,
        ),
    )
    return db, conversation_id, repository, greeting, checkpoint


@pytest.mark.parametrize("clear", [False, True])
def test_cursor_writer_cannot_strand_unresolved_dispatch(database, clear):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    with pytest.raises(InputError, match="dispatch"):
        db.set_conversation_active_leaf(conversation_id, None if clear else greeting)
    assert (
        db.get_conversation_active_leaf(conversation_id)
        == checkpoint.assistant_message_id
    )
    assert repository.reconcile_for_session(conversation_id).checkpoint == checkpoint


def test_restore_repairs_old_cursor_and_publishes_same_pending_turn(database):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
            (greeting, conversation_id),
        )
    store, session_id = _restored_store(db, conversation_id)
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.ACCEPTED
    assert recovery.checkpoint == checkpoint
    assert (
        db.get_conversation_active_leaf(conversation_id)
        == checkpoint.assistant_message_id
    )
    assert [m.persisted_message_id for m in store.messages_for_session(session_id)] == [
        greeting,
        checkpoint.user_message_id,
        checkpoint.assistant_message_id,
    ]
    assert repository.reconcile_for_session(conversation_id).checkpoint == checkpoint
    assert db.get_message_by_id(checkpoint.assistant_message_id)["content"] == ""


@pytest.mark.parametrize("corruption", ["version", "parent", "payload"])
def test_invalid_stranded_owner_is_not_repaired(database, corruption):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
            (greeting, conversation_id),
        )
        if corruption == "payload":
            cursor.execute(
                "UPDATE console_dispatch_checkpoints SET frozen_authority_json='{}'"
            )
    if corruption != "payload":
        statement = (
            "UPDATE messages SET version=2 WHERE id=?"
            if corruption == "version"
            else "UPDATE messages SET parent_message_id=NULL WHERE id=?"
        )
        _raw_semantic_corruption(
            db,
            statement,
            (checkpoint.assistant_message_id,),
        )
    recovery = repository.reconcile_for_session(conversation_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert recovery.actions == ()
    assert db.get_conversation_active_leaf(conversation_id) == greeting


def test_store_rejected_navigation_keeps_runtime_and_database_cursor(database):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    store, session_id = _restored_store(db, conversation_id)
    with pytest.raises(RuntimeError, match="cursor"):
        store.set_active_leaf(session_id, greeting)
    assert (
        store.messages_for_session(session_id)[-1].persisted_message_id
        == checkpoint.assistant_message_id
    )
    assert repository.reconcile_for_session(conversation_id).checkpoint == checkpoint


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.asyncio
async def test_repaired_owner_can_be_discarded_without_provider_replay(
    database, started
):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    if started:
        checkpoint = _start(repository, checkpoint)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
            (greeting, conversation_id),
        )
    store, session_id = _restored_store(db, conversation_id)
    assert store.dispatch_recovery_for_session(session_id).checkpoint == checkpoint
    gateway = _NoReplayGateway(db)
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=False
    )
    result = await controller.discard_dispatch_recovery(session_id)
    assert result.accepted
    assert gateway.resolve_calls == 0
    assert db.get_message_by_id(checkpoint.user_message_id)["deleted"] == 0
    assert (
        db.get_message_by_id(checkpoint.assistant_message_id)[
            "assistant_generation_state"
        ]
        == "discarded"
    )
    assert repository.reconcile_for_session(conversation_id) is None
    store.set_active_leaf(session_id, greeting)
    assert db.get_conversation_active_leaf(conversation_id) == greeting


def test_repair_rereads_owner_after_read_pass(database, monkeypatch):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
            (greeting, conversation_id),
        )
    original = repository._reconcile_pass

    def change_between_passes(conversation_id, *, allow_writes):
        result = original(conversation_id, allow_writes=allow_writes)
        if not allow_writes:
            _raw_semantic_corruption(
                db,
                "UPDATE messages SET version=2 WHERE id=?",
                (checkpoint.assistant_message_id,),
            )
        return result

    monkeypatch.setattr(repository, "_reconcile_pass", change_between_passes)
    recovery = repository.reconcile_for_session(conversation_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert db.get_conversation_active_leaf(conversation_id) == greeting


def test_repair_write_failure_retains_checkpoint_and_cursor(database):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
            (greeting, conversation_id),
        )
        cursor.execute(
            "CREATE TRIGGER fail_cursor BEFORE UPDATE OF active_leaf_message_id ON conversations "
            "BEGIN SELECT RAISE(ABORT, 'fail'); END"
        )
    recovery = repository.reconcile_for_session(conversation_id)
    assert recovery.error_code == "checkpoint_reconcile_error"
    assert db.get_conversation_active_leaf(conversation_id) == greeting
    assert (
        db.get_connection()
        .execute("SELECT checkpoint_revision FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == checkpoint.checkpoint_revision
    )


def test_before_first_rewind_cannot_strand_pending_turn(database):
    db, conversation_id, repository = database
    checkpoint = _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    assert store.set_active_path_before(session_id, checkpoint.user_message_id) is False
    assert store.active_leaf(session_id) == checkpoint.assistant_message_id
    assert (
        db.get_conversation_active_leaf(conversation_id)
        == checkpoint.assistant_message_id
    )


@pytest.mark.parametrize("problem", ["cycle", "competing_owner"])
def test_repair_rejects_ambiguous_ancestry(database, problem):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
            (greeting, conversation_id),
        )
    if problem == "cycle":
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET parent_message_id=? WHERE id=?",
            (checkpoint.assistant_message_id, checkpoint.user_message_id),
        )
    else:
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET assistant_generation_state='accepted' WHERE id=?",
            (greeting,),
        )
    recovery = repository.reconcile_for_session(conversation_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert db.get_conversation_active_leaf(conversation_id) == greeting


def test_cursor_can_stay_on_pending_owner_path(database):
    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    db.set_conversation_active_leaf(conversation_id, checkpoint.assistant_message_id)
    assert repository.reconcile_for_session(conversation_id).checkpoint == checkpoint


@pytest.mark.parametrize("action", ["sibling", "delete", "edit"])
def test_pending_dispatch_blocks_branch_mutation_without_side_effects(database, action):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    store, session_id = _restored_store(db, conversation_id)
    before = store.messages_for_session(session_id)
    with pytest.raises((RuntimeError, ValueError), match="dispatch"):
        if action == "sibling":
            store.create_sibling(
                greeting, role=ConsoleMessageRole.USER, content="fork", persist=True
            )
        elif action == "delete":
            store.delete_message(greeting)
        else:
            store.update_message_content(greeting, "changed")
    assert store.messages_for_session(session_id) == before
    assert db.get_message_by_id(greeting)["content"] == "greeting"
    assert db.get_message_by_id(greeting)["deleted"] == 0
    assert (
        db.get_conversation_active_leaf(conversation_id)
        == checkpoint.assistant_message_id
    )
    assert repository.reconcile_for_session(conversation_id).checkpoint == checkpoint
    with db.transaction() as cursor:
        assert cursor.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3


def test_deleted_conversation_cursor_returns_false_with_retained_checkpoint(database):
    db, conversation_id, _, _, _ = _stranded(database)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET deleted=1 WHERE id=?", (conversation_id,)
        )
    assert (
        db.set_conversation_active_cursor(
            conversation_id, active_leaf_message_id=None, before_message_id=None
        )
        is False
    )


def test_voice_recovery_selection_preserves_cursor_when_dispatch_refuses(database):
    from tldw_chatbook.Chat.console_voice_promotion import (
        ConsoleVoicePromotionLease,
        ResolvedVoicePromotionDestination,
    )

    db, conversation_id, repository, greeting, checkpoint = _stranded(database)
    store, session_id = _restored_store(db, conversation_id)
    incarnation = store._settings_session_incarnations[session_id]
    lease = ConsoleVoicePromotionLease(
        "lease",
        session_id,
        incarnation,
        1,
        "promotion",
        checkpoint.assistant_message_id,
        ResolvedVoicePromotionDestination(
            session_id,
            incarnation,
            conversation_id,
            checkpoint.assistant_message_id,
            True,
        ),
    )
    store._voice_promotion_leases[session_id] = lease
    store._voice_promotion_contexts[session_id] = object()
    assert store.select_voice_promotion_recovery_leaf(lease, greeting) is False
    assert store.active_leaf(session_id) == checkpoint.assistant_message_id
    assert repository.reconcile_for_session(conversation_id).checkpoint == checkpoint
    assert session_id not in store._voice_promotion_selection_decisions
