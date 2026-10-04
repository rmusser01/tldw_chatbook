"""Native machine-origin dispatch receipts preserve historical checkpoint owners."""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
    _acceptance,
    _db_and_conversation,
    _insert,
)
from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
from tldw_chatbook.Chat import message_metadata
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("standalone", [False, True])
def test_v77_keeps_every_predecessor_checkpoint_column_and_indexes(
    tmp_path, standalone
):
    path = tmp_path / "migration.sqlite"
    with chachanotes_db_at_version(path, 76 if standalone else 75) as db:
        conversation = db.add_conversation({"title": "saved"})
        user = db.add_message(
            {"conversation_id": conversation, "sender": "user", "content": "old"}
        )
        assistant = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "content": "pending",
            }
        )
        db.get_connection().execute(
            "INSERT INTO console_dispatch_checkpoints "
            "(assistant_message_id,user_message_id,conversation_id,preparation_id,attempt_id,state,"
            "checkpoint_revision,user_message_version,assistant_message_version,origin,queue_entry_id,"
            "frozen_authority_json,resolved_destination_json,reconstructability_json) "
            "VALUES (?,?,?,?,?,'accepted',7,1,1,'queued','queue-old','{}','{}','{}')",
            (assistant, user, conversation, "prepare-old", "attempt-old"),
        )
        db.get_connection().commit()
        before = dict(
            db.get_connection()
            .execute("SELECT * FROM console_dispatch_checkpoints")
            .fetchone()
        )
    if standalone:
        migration = (
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook/DB/migrations/chachanotes_v76_to_v77_agent_chat_starts.sql"
        )
        with sqlite3.connect(path) as connection:
            connection.executescript(migration.read_text())
            connection.execute(
                "UPDATE db_schema_version SET version=77 WHERE schema_name=?",
                (CharactersRAGDB._SCHEMA_NAME,),
            )
    reopened = CharactersRAGDB(path, client_id="reopen")
    try:
        row = dict(
            reopened.get_connection()
            .execute("SELECT * FROM console_dispatch_checkpoints")
            .fetchone()
        )
        assert reopened._get_db_version(reopened.get_connection()) == 77
        assert row.pop("agent_chat_start_attempt_id") is None
        assert row == before
        indexes = {
            row[1]
            for row in reopened.get_connection().execute(
                "PRAGMA index_list(console_dispatch_checkpoints)"
            )
        }
        assert "idx_console_dispatch_checkpoint_conversation" in indexes
        assert "idx_console_dispatch_checkpoints_user_message" in indexes
        assert (
            not reopened.get_connection().execute("PRAGMA foreign_key_check").fetchall()
        )
    finally:
        reopened.close_connection()


def test_machine_request_and_checkpoint_share_exact_attempt(tmp_path):
    db, conversation = _db_and_conversation(tmp_path / "receipt.sqlite")
    try:
        provenance = message_metadata.AgentChatStartMetadata(
            attempt_id="chat-start-1",
            source_run_id="source-run",
            source_conversation_id="source-chat",
        )
        acceptance = replace(
            _acceptance(conversation),
            origin="agent_chat_start",
            agent_chat_start_attempt_id="chat-start-1",
            agent_chat_start=provenance,
            handoff_draft_revision=1,
        )
        checkpoint = _insert(db, ConsoleDispatchRepository(db), acceptance)
        assert checkpoint.agent_chat_start_attempt_id == "chat-start-1"
        row = db.get_message_by_id(acceptance.user_message_id)
        restored = message_metadata.MessageMetadata.from_json(row["metadata_json"])
        assert restored.origin == "agent_chat_start"
        assert restored.agent_chat_start == provenance
        duplicate = _insert(db, ConsoleDispatchRepository(db), acceptance)
        assert duplicate == checkpoint
        assert len(db.get_messages_for_conversation(conversation)) == 2
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {
            "attempt_id": "a",
            "source_run_id": "r",
            "source_conversation_id": "c",
            "body": "private",
        },
    ],
)
def test_malformed_machine_provenance_never_hydrates_as_human(payload):
    metadata = message_metadata.MessageMetadata.from_json(
        json.dumps(
            {
                "origin": "agent_chat_start",
                "agent_chat_start": payload,
            }
        )
    )
    assert metadata is not None
    assert metadata.origin
    assert metadata.origin != "agent_chat_start"


def test_duplicate_machine_receipt_refuses_changed_request(tmp_path):
    db, conversation = _db_and_conversation(tmp_path / "duplicate.sqlite")
    try:
        acceptance = replace(
            _acceptance(conversation),
            origin="agent_chat_start",
            agent_chat_start_attempt_id="start",
            agent_chat_start=message_metadata.AgentChatStartMetadata(
                "start", "run", "source"
            ),
            handoff_draft_revision=1,
        )
        repository = ConsoleDispatchRepository(db)
        _insert(db, repository, acceptance)
        with pytest.raises((ValueError, RuntimeError)):
            _insert(db, repository, replace(acceptance, user_content="rewritten"))
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "origin,root_fork,hook",
    [
        ("agent_chat_start", True, False),
        ("agent_chat_start", False, True),
        ("manual", True, True),
    ],
)
def test_incompatible_machine_and_root_fork_provenance_rolls_back(
    tmp_path, origin, root_fork, hook
):
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt

    db, conversation = _db_and_conversation(tmp_path / "mixed.sqlite")
    try:
        acceptance = replace(
            _acceptance(conversation),
            origin=origin,
            user_root_fork=root_fork,
            continuation_receipt=ContinuationReceipt(
                "parent", "stop", "assistant", "scheduler", 1
            )
            if hook
            else None,
            agent_chat_start=message_metadata.AgentChatStartMetadata(
                "start", "run", "source"
            )
            if origin == "agent_chat_start"
            else None,
            agent_chat_start_attempt_id="start"
            if origin == "agent_chat_start"
            else None,
            handoff_draft_revision=1 if origin == "agent_chat_start" else None,
        )
        with pytest.raises(ValueError):
            _insert(db, ConsoleDispatchRepository(db), acceptance)
        assert not db.get_messages_for_conversation(conversation)
    finally:
        db.close()


def test_mixed_hook_replay_keeps_native_receipt_and_messages_exact(tmp_path):
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt

    db, conversation = _db_and_conversation(tmp_path / "mixed-replay.sqlite")
    try:
        acceptance = replace(
            _acceptance(conversation),
            origin="agent_chat_start",
            agent_chat_start_attempt_id="start",
            agent_chat_start=message_metadata.AgentChatStartMetadata(
                "start", "run", "source"
            ),
            handoff_draft_revision=1,
        )
        repository = ConsoleDispatchRepository(db)
        checkpoint = _insert(db, repository, acceptance)
        assert _insert(db, repository, acceptance) == checkpoint
        before = dict(
            db.get_connection()
            .execute("SELECT * FROM console_dispatch_checkpoints")
            .fetchone()
        )
        messages = db.get_messages_for_conversation(conversation)
        with pytest.raises(ValueError):
            _insert(
                db,
                repository,
                replace(
                    acceptance,
                    continuation_receipt=ContinuationReceipt(
                        "parent", "stop", "assistant", "scheduler", 1
                    ),
                ),
            )
        assert (
            dict(
                db.get_connection()
                .execute("SELECT * FROM console_dispatch_checkpoints")
                .fetchone()
            )
            == before
        )
        assert db.get_messages_for_conversation(conversation) == messages
        assert len(messages) == 2
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
            .fetchone()[0]
            == 0
        )
    finally:
        db.close()


def _legacy_native76(path, *, dictionary=False, subscriptions=False):
    """Replay the previously shipped feature SQL over its actual v75 chain."""
    if subscriptions:
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

        subscriptions_db = SubscriptionsDB(path)
        subscriptions_db.close()
    with chachanotes_db_at_version(path, 75) as db:
        migration = (
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook/DB/migrations/chachanotes_v76_to_v77_agent_chat_starts.sql"
        )
        db.get_connection().executescript(migration.read_text())
        db.get_connection().execute(
            "UPDATE db_schema_version SET version=76 WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        )
        if dictionary:
            from tldw_chatbook.DB.recovery_core_schema import (
                _CHAT_DICTIONARIES_UPDATED_TRIGGER,
            )

            db.get_connection().execute("DROP TRIGGER chat_dictionaries_au")
            db.get_connection().execute(_CHAT_DICTIONARIES_UPDATED_TRIGGER)
        db.get_connection().commit()
        conversation = db.add_conversation({"title": "historical native receipt"})
        acceptance = replace(
            _acceptance(conversation),
            origin="agent_chat_start",
            agent_chat_start_attempt_id="legacy-exact-attempt",
            agent_chat_start=message_metadata.AgentChatStartMetadata(
                "legacy-exact-attempt", "legacy-run", "legacy-source"
            ),
            handoff_draft_revision=1,
        )
        _insert(db, ConsoleDispatchRepository(db), acceptance)
        rows = tuple(
            tuple(row)
            for row in db.get_connection().execute(
                "SELECT * FROM console_dispatch_checkpoints"
            )
        )
        messages = tuple(
            tuple(row)
            for row in db.get_connection().execute("SELECT * FROM messages ORDER BY id")
        )
    return rows, messages


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("subscriptions", [False, True])
def test_legacy_native76_upgrade_preserves_exact_receipts(
    tmp_path, dictionary, subscriptions
):
    path = tmp_path / "legacy.sqlite"
    before = _legacy_native76(path, dictionary=dictionary, subscriptions=subscriptions)
    for _ in range(2):
        db = CharactersRAGDB(path, client_id="reopen-native")
        try:
            connection = db.get_connection()
            assert db._get_db_version(connection) == 77
            assert (
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM console_dispatch_checkpoints"
                    )
                )
                == before[0]
            )
            assert (
                tuple(
                    tuple(row)
                    for row in connection.execute("SELECT * FROM messages ORDER BY id")
                )
                == before[1]
            )
            assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            db.close_connection()


@pytest.mark.parametrize(
    "alteration", ["extra_table", "unguarded_notes", "changed_index"]
)
def test_unknown_native76_catalog_refuses_without_rewriting(tmp_path, alteration):
    from tldw_chatbook.DB.ChaChaNotes_DB import SchemaError

    path = tmp_path / "hybrid.sqlite"
    _legacy_native76(path)
    with sqlite3.connect(path) as connection:
        if alteration == "extra_table":
            connection.execute("CREATE TABLE unqualified(value)")
        elif alteration == "unguarded_notes":
            connection.execute("DROP TRIGGER notes_au")
            connection.execute(
                "CREATE TRIGGER notes_au AFTER UPDATE ON notes BEGIN SELECT 1; END"
            )
        else:
            connection.execute(
                "DROP INDEX idx_console_dispatch_checkpoints_user_message"
            )
            connection.execute(
                "CREATE INDEX idx_console_dispatch_checkpoints_user_message ON console_dispatch_checkpoints(user_message_id, state)"
            )
    with sqlite3.connect(path) as connection:
        before = tuple(connection.iterdump())
    with pytest.raises(SchemaError):
        CharactersRAGDB(path, client_id="refuse-hybrid")
    with sqlite3.connect(path) as connection:
        assert tuple(connection.iterdump()) == before


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("subscriptions", [False, True])
def test_v77_constructor_catalog_matches_exact_native76_capture(
    tmp_path, dictionary, subscriptions
):
    import hashlib
    from tldw_chatbook.DB.recovery_core_schema import (
        CHACHANOTES_V76_SCHEMA,
        _CHAT_DICTIONARIES_INITIAL_TRIGGER,
        _CHAT_DICTIONARIES_UPDATED_TRIGGER,
    )
    from tldw_chatbook.DB.recovery_operations import _SUBSCRIPTIONS_V76_SCHEMA

    path = tmp_path / "capture.sqlite"
    if subscriptions:
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

        other = SubscriptionsDB(path)
        other.close()
    db = CharactersRAGDB(path, client_id="constructor-capture")
    try:
        connection = db.get_connection()
        if dictionary:
            connection.execute("DROP TRIGGER chat_dictionaries_au")
            connection.execute(_CHAT_DICTIONARIES_UPDATED_TRIGGER)
        actual = tuple(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        expected = (
            _SUBSCRIPTIONS_V76_SCHEMA if subscriptions else CHACHANOTES_V76_SCHEMA
        )
        if dictionary:
            expected = tuple(
                _CHAT_DICTIONARIES_UPDATED_TRIGGER
                if sql == _CHAT_DICTIONARIES_INITIAL_TRIGGER
                else sql
                for sql in expected
            )
        assert actual == expected
        assert db._get_db_version(connection) == 77
        print(
            json.dumps(
                {
                    "subscriptions": subscriptions,
                    "dictionary": dictionary,
                    "entries": len(actual),
                    "catalog_sha256": hashlib.sha256(
                        json.dumps(actual).encode()
                    ).hexdigest(),
                }
            )
        )
    finally:
        db.close_connection()
