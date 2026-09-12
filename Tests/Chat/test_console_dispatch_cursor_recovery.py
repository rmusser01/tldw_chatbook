"""An unresolved durable send must survive stale navigation cursor writes."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _raw_semantic_corruption,
    _restored_store,
    _start,
    _NoReplayGateway,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleDispatchRecoveryKind
from tldw_chatbook.DB.ChaChaNotes_DB import InputError


def _stranded(tmp_path):
    db, conversation_id, repository = _database(tmp_path / "stranded.sqlite")
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
def test_cursor_writer_cannot_strand_unresolved_dispatch(tmp_path, clear):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
    with pytest.raises(InputError, match="dispatch"):
        db.set_conversation_active_leaf(conversation_id, None if clear else greeting)
    assert (
        db.get_conversation_active_leaf(conversation_id)
        == checkpoint.assistant_message_id
    )
    assert repository.reconcile_for_session(conversation_id).checkpoint == checkpoint


def test_restore_repairs_old_cursor_and_publishes_same_pending_turn(tmp_path):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
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
def test_invalid_stranded_owner_is_not_repaired(tmp_path, corruption):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
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
        assignment = (
            "version=2" if corruption == "version" else "parent_message_id=NULL"
        )
        _raw_semantic_corruption(
            db,
            f"UPDATE messages SET {assignment} WHERE id=?",
            (checkpoint.assistant_message_id,),
        )
        db.get_connection().commit()
    recovery = repository.reconcile_for_session(conversation_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert recovery.actions == ()
    assert db.get_conversation_active_leaf(conversation_id) == greeting


def test_store_rejected_navigation_keeps_runtime_and_database_cursor(tmp_path):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
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
    tmp_path, started
):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
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


def test_repair_rereads_owner_after_read_pass(tmp_path, monkeypatch):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
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
            db.get_connection().commit()
        return result

    monkeypatch.setattr(repository, "_reconcile_pass", change_between_passes)
    recovery = repository.reconcile_for_session(conversation_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert db.get_conversation_active_leaf(conversation_id) == greeting


def test_repair_write_failure_retains_checkpoint_and_cursor(tmp_path):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
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


def test_before_first_rewind_cannot_strand_pending_turn(tmp_path):
    db, conversation_id, repository = _database(tmp_path / "before.sqlite")
    checkpoint = _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    assert store.set_active_path_before(session_id, checkpoint.user_message_id) is False
    assert store.active_leaf(session_id) == checkpoint.assistant_message_id
    assert (
        db.get_conversation_active_leaf(conversation_id)
        == checkpoint.assistant_message_id
    )


@pytest.mark.parametrize("problem", ["cycle", "competing_owner"])
def test_repair_rejects_ambiguous_ancestry(tmp_path, problem):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
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
    db.get_connection().commit()
    recovery = repository.reconcile_for_session(conversation_id)
    assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
    assert db.get_conversation_active_leaf(conversation_id) == greeting


def test_cursor_can_stay_on_pending_owner_path(tmp_path):
    db, conversation_id, repository, greeting, checkpoint = _stranded(tmp_path)
    db.set_conversation_active_leaf(conversation_id, checkpoint.assistant_message_id)
    assert repository.reconcile_for_session(conversation_id).checkpoint == checkpoint
