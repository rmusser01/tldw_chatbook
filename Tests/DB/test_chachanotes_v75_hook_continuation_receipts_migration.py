"""Real historical upgrade and atomic continuation acceptance/reopen proofs."""

import sqlite3
from dataclasses import replace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
    _acceptance,
    _db_and_conversation,
    _insert,
)
from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt
from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleAssistantSettlement
from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("source_version", [73, 74])
def test_genuine_upgrade_creates_receipts(tmp_path, source_version):
    path = tmp_path / f"v{source_version}.db"
    with chachanotes_db_at_version(path, source_version, client_id="historical") as db:
        assert (
            not db.get_connection()
            .execute(
                "SELECT 1 FROM sqlite_master WHERE name='console_hook_continuation_receipts'"
            )
            .fetchone()
        )
        if source_version == 74:
            conversation = db.add_conversation({"title": "Before hook receipts"})
            with db.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO console_auxiliary_attempts "
                    "(operation_id, conversation_id, purpose, provider, model, "
                    "requested_output_cap, estimated_input_tokens, status, started_at, failure_reason) "
                    "VALUES ('existing-failure', ?, 'conversation_compaction', 'openai', "
                    "'test-model', 100, 1000, 'failed', '2026-10-01T00:00:00Z', 'invalid_summary_output')",
                    (conversation,),
                )
    db = CharactersRAGDB(path, client_id="upgraded")
    try:
        assert (
            db.get_connection()
            .execute(
                "SELECT 1 FROM sqlite_master WHERE name='console_hook_continuation_receipts'"
            )
            .fetchone()
        )
        assert db._get_db_version(db.get_connection()) == 76
        if source_version == 74:
            assert (
                db.get_connection()
                .execute(
                    "SELECT failure_reason FROM console_auxiliary_attempts WHERE operation_id='existing-failure'"
                )
                .fetchone()[0]
                == "invalid_summary_output"
            )
        assert "failure_reason" in {
            row[1]
            for row in db.get_connection().execute(
                "PRAGMA table_info(console_auxiliary_attempts)"
            )
        }
    finally:
        db.close()


def _settle(repository, owner):
    return repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id=owner.assistant_message_id,
            expected_checkpoint_state=owner.state,
            expected_checkpoint_revision=owner.checkpoint_revision,
            expected_user_message_version=owner.user_message_version,
            expected_assistant_message_version=owner.assistant_message_version,
            terminal_state="discarded",
            content="done",
            metadata_json=None,
        )
    )


def test_receipt_survives_terminal_restart_and_duplicate_rolls_back(tmp_path):
    path = tmp_path / "receipts.db"
    db, conversation = _db_and_conversation(path)
    repository = ConsoleDispatchRepository(db)
    parent = _insert(db, repository, _acceptance(conversation))
    _settle(repository, parent)
    receipt = ContinuationReceipt(
        "parent-turn", "stop-event", parent.assistant_message_id, "chain", 1
    )
    acceptance = replace(
        _acceptance(conversation, suffix="2"),
        origin="queued",
        queue_entry_id="machine-entry",
        parent_message_id=parent.assistant_message_id,
        continuation_receipt=receipt,
    )
    child = _insert(db, repository, acceptance)
    _settle(repository, child)
    db.close()
    db = CharactersRAGDB(path, client_id="reopened")
    try:
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
            .fetchone()[0]
            == 1
        )
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )
        with pytest.raises(sqlite3.IntegrityError):
            _insert(
                db,
                ConsoleDispatchRepository(db),
                replace(
                    acceptance,
                    user_message_id="competing-user",
                    assistant_message_id="competing-assistant",
                    preparation_id="competing-preparation",
                    queue_entry_id="competing-entry",
                ),
            )
        assert (
            db.get_connection()
            .execute("SELECT 1 FROM messages WHERE id='competing-user'")
            .fetchone()
            is None
        )
    finally:
        db.close()


def test_v75_failure_rolls_back_ddl_and_reopens_cleanly(tmp_path, monkeypatch):
    from tldw_chatbook.DB.ChaChaNotes_DB import SchemaError

    path = tmp_path / "interrupted.db"
    with chachanotes_db_at_version(path, 74, client_id="historical"):
        pass
    execute = CharactersRAGDB._execute_migration_statements

    def fail(self, cursor, script, label):
        execute(self, cursor, script, label)
        if label == "V74→V75":
            raise sqlite3.OperationalError("controlled post-DDL failure")

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_execute_migration_statements", fail)
        with pytest.raises(SchemaError):
            CharactersRAGDB(path, client_id="failed")
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT version FROM db_schema_version").fetchone()[0]
            == 74
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='console_hook_continuation_receipts'"
            ).fetchone()
            is None
        )
    reopened = CharactersRAGDB(path, client_id="retry")
    assert reopened._get_db_version(reopened.get_connection()) == 76
    reopened.close()


def test_uncertain_dispatch_restart_retains_machine_identity_without_replay(tmp_path):
    import json

    from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
        _start_dispatch,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleDispatchRecoveryKind

    path = tmp_path / "uncertain.db"
    db, conversation = _db_and_conversation(path)
    repository = ConsoleDispatchRepository(db)
    parent = _insert(db, repository, _acceptance(conversation))
    _settle(repository, parent)
    receipt = ContinuationReceipt(
        "parent", "stop", parent.assistant_message_id, "chain", 1
    )
    child = _insert(
        db,
        repository,
        replace(
            _acceptance(conversation, suffix="child"),
            origin="queued",
            queue_entry_id="machine",
            parent_message_id=parent.assistant_message_id,
            continuation_receipt=receipt,
        ),
    )
    _start_dispatch(repository, child)
    db.close()
    db = CharactersRAGDB(path, client_id="restart")
    try:
        repository = ConsoleDispatchRepository(db)
        recovery = repository.reconcile_for_session(conversation)
        assert recovery.kind is ConsoleDispatchRecoveryKind.DISPATCH_STARTED
        assert recovery.checkpoint.origin == "queued"
        for _ in range(2):
            assert (
                repository.reconcile_for_session(conversation).kind
                is ConsoleDispatchRecoveryKind.DISPATCH_STARTED
            )
        metadata = (
            db.get_connection()
            .execute(
                "SELECT metadata_json FROM messages WHERE id=?",
                (child.user_message_id,),
            )
            .fetchone()[0]
        )
        assert json.loads(metadata)["initiator"] == "hook_continuation"
        assert (
            db.get_connection()
            .execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id=?", (conversation,)
            )
            .fetchone()[0]
            == 4
        )
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
            .fetchone()[0]
            == 1
        )
    finally:
        db.close()


def test_conversation_cascade_uses_receipt_index_without_statistics(tmp_path):
    """Deleting a conversation searches its receipt index without ANALYZE."""
    db, conversation = _db_and_conversation(tmp_path / "cascade-plan.db")
    try:
        repository = ConsoleDispatchRepository(db)
        parent = _insert(db, repository, _acceptance(conversation))
        _settle(repository, parent)
        child = _insert(
            db,
            repository,
            replace(
                _acceptance(conversation, suffix="plan-child"),
                origin="queued",
                queue_entry_id="plan-machine",
                parent_message_id=parent.assistant_message_id,
                continuation_receipt=ContinuationReceipt(
                    "plan-parent",
                    "plan-stop",
                    parent.assistant_message_id,
                    "plan-chain",
                    1,
                ),
            ),
        )
        _settle(repository, child)
        connection = db.get_connection()
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'"
            ).fetchone()
            is None
        )
        plan = "\n".join(
            row["detail"]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN DELETE FROM conversations WHERE id=?",
                (conversation,),
            )
        )
        assert (
            "USING COVERING INDEX idx_hook_continuation_receipts_conversation" in plan
        )
        assert "SCAN console_hook_continuation_receipts" not in plan
    finally:
        db.close()
