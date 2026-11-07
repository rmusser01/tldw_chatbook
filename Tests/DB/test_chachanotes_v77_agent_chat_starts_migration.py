"""Native machine-origin dispatch receipts preserve historical checkpoint owners."""

from contextlib import closing
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
        assert reopened._get_db_version(reopened.get_connection()) == 80
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


# Exact historical strong native76 artifact; independent of installed repaired77.
_FROZEN_NATIVE76_MIGRATION = """-- Native agent-chat-start receipts. Local-only operational ownership.
CREATE TABLE console_dispatch_checkpoints_v77 (
    assistant_message_id TEXT PRIMARY KEY
        REFERENCES messages(id) ON DELETE CASCADE,
    user_message_id TEXT NOT NULL
        REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE CASCADE,
    schema_version INTEGER NOT NULL DEFAULT 1
        CHECK(schema_version > 0),
    preparation_id TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL,
    state TEXT NOT NULL
        CHECK(state IN ('accepted', 'dispatch_started')),
    checkpoint_revision INTEGER NOT NULL DEFAULT 1
        CHECK(checkpoint_revision > 0),
    user_message_version INTEGER NOT NULL
        CHECK(user_message_version > 0),
    assistant_message_version INTEGER NOT NULL
        CHECK(assistant_message_version > 0),
    origin TEXT NOT NULL CHECK(origin IN ('manual', 'queued', 'agent_chat_start')),
    queue_entry_id TEXT,
    agent_chat_start_attempt_id TEXT UNIQUE,
    frozen_authority_json TEXT NOT NULL,
    resolved_destination_json TEXT NOT NULL,
    reconstructability_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((origin = 'queued' AND queue_entry_id IS NOT NULL)
        OR (origin IN ('manual', 'agent_chat_start') AND queue_entry_id IS NULL)),
    CHECK ((origin = 'agent_chat_start' AND agent_chat_start_attempt_id IS NOT NULL
            AND length(agent_chat_start_attempt_id) BETWEEN 1 AND 200)
        OR (origin IN ('manual', 'queued') AND agent_chat_start_attempt_id IS NULL))
);

INSERT INTO console_dispatch_checkpoints_v77 (assistant_message_id, user_message_id, conversation_id, schema_version, preparation_id, attempt_id, state, checkpoint_revision, user_message_version, assistant_message_version, origin, queue_entry_id, frozen_authority_json, resolved_destination_json, reconstructability_json, created_at, updated_at)
SELECT assistant_message_id, user_message_id, conversation_id, schema_version, preparation_id, attempt_id, state, checkpoint_revision, user_message_version, assistant_message_version, origin, queue_entry_id, frozen_authority_json, resolved_destination_json, reconstructability_json, created_at, updated_at FROM console_dispatch_checkpoints;
DROP TABLE console_dispatch_checkpoints;
ALTER TABLE console_dispatch_checkpoints_v77 RENAME TO console_dispatch_checkpoints;
CREATE INDEX idx_console_dispatch_checkpoint_conversation
    ON console_dispatch_checkpoints(conversation_id);
CREATE INDEX idx_console_dispatch_checkpoints_user_message
    ON console_dispatch_checkpoints(user_message_id);
"""


def _legacy_native76(path, *, dictionary=False, subscriptions=False):
    """Replay the previously shipped feature SQL over its actual v75 chain."""
    if subscriptions:
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

        subscriptions_db = SubscriptionsDB(path)
        subscriptions_db.close()
    with chachanotes_db_at_version(path, 75) as db:
        db.get_connection().executescript(_FROZEN_NATIVE76_MIGRATION)
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
        from tldw_chatbook.DB.recovery_core_schema import CHACHANOTES_V76_NATIVE_SCHEMAS
        from tldw_chatbook.DB.recovery_operations import (
            _SUBSCRIPTIONS_NATIVE_RECEIPT_SCHEMAS,
        )

        catalog = tuple(
            row[0]
            for row in db.get_connection().execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert catalog in (
            _SUBSCRIPTIONS_NATIVE_RECEIPT_SCHEMAS
            if subscriptions
            else CHACHANOTES_V76_NATIVE_SCHEMAS
        )
        assert db._get_db_version(db.get_connection()) == 76
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
    with closing(sqlite3.connect(path)) as connection:
        catalog = tuple(
            connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
    from tldw_chatbook.DB.recovery_core_schema import (
        _browse_order_catalog,
        _fleet_progress_catalog,
        _sargable_catalog,
    )

    catalog = tuple(
        (sql,)
        for sql in _browse_order_catalog(
            _sargable_catalog(_fleet_progress_catalog(tuple(row[0] for row in catalog)))
        )
    )
    for _ in range(2):
        db = CharactersRAGDB(path, client_id="reopen-native")
        try:
            connection = db.get_connection()
            assert db._get_db_version(connection) == 80
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
            assert (
                tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                    )
                )
                == catalog
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
        CHACHANOTES_V80_SCHEMAS,
        _CHAT_DICTIONARIES_INITIAL_TRIGGER,
        _CHAT_DICTIONARIES_UPDATED_TRIGGER,
    )
    from tldw_chatbook.DB.recovery_operations import _SUBSCRIPTIONS_BROWSE_ORDER_SCHEMAS

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
            _SUBSCRIPTIONS_BROWSE_ORDER_SCHEMAS[0]
            if subscriptions
            else CHACHANOTES_V80_SCHEMAS[0]
        )
        if dictionary:
            expected = tuple(
                _CHAT_DICTIONARIES_UPDATED_TRIGGER
                if sql == _CHAT_DICTIONARIES_INITIAL_TRIGGER
                else sql
                for sql in expected
            )
        assert actual == expected
        assert db._get_db_version(connection) == 80
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


@pytest.mark.parametrize("dictionary", [False, True], ids=["initial", "updated"])
@pytest.mark.parametrize(
    "origin,queue_entry_id",
    [("queued", None), ("manual", "legacy-queue")],
    ids=["queued-null", "manual-queue"],
)
def test_legacy_queue_constructor_preserves_predecessor(
    tmp_path, dictionary, origin, queue_entry_id
):
    path = tmp_path / "legacy-queue.sqlite"
    with chachanotes_db_at_version(path, 76) as db:
        conversation = db.add_conversation({"title": "legacy queue owner"})
        user = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "user",
                "content": "legacy request",
            }
        )
        assistant = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "content": "legacy pending",
            }
        )
        connection = db.get_connection()
        connection.execute(
            "INSERT INTO console_dispatch_checkpoints "
            "(assistant_message_id,user_message_id,conversation_id,schema_version,preparation_id,attempt_id,state,"
            "checkpoint_revision,user_message_version,assistant_message_version,origin,queue_entry_id,"
            "frozen_authority_json,resolved_destination_json,reconstructability_json,created_at,updated_at) "
            "VALUES (?,?,?,3,'legacy-prepare','legacy-attempt','dispatch_started',7,5,6,?,?,?,? ,?,'2026-09-01T01:02:03','2026-09-02T04:05:06')",
            (
                assistant,
                user,
                conversation,
                origin,
                queue_entry_id,
                '{ "saved" : "authority" }',
                '{ "saved" : "destination" }',
                '{ "saved" : "reconstructability" }',
            ),
        )
        if dictionary:
            from tldw_chatbook.DB.recovery_core_schema import (
                _CHAT_DICTIONARIES_UPDATED_TRIGGER,
            )

            connection.execute("DROP TRIGGER chat_dictionaries_au")
            connection.execute(_CHAT_DICTIONARIES_UPDATED_TRIGGER)
        connection.commit()
        before = dict(
            connection.execute("SELECT * FROM console_dispatch_checkpoints").fetchone()
        )
        assert len(before) == 17
        assert db._get_db_version(connection) == 76
        assert "agent_chat_start_attempt_id" not in before
    db = CharactersRAGDB(path, client_id="legacy-queue-reopen")
    try:
        row = dict(
            db.get_connection()
            .execute("SELECT * FROM console_dispatch_checkpoints")
            .fetchone()
        )
        assert row.pop("agent_chat_start_attempt_id") is None
        assert row == before
        assert db._get_db_version(db.get_connection()) == 80
    finally:
        db.close_connection()


def _seed_legacy_queue_owner(db, origin, queue_entry_id):
    """Store an otherwise valid historical owner without new-write admission."""
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        dump_console_turn_library_authority_json,
        dump_console_resolved_destination_json,
        dump_console_dispatch_reconstructability_json,
    )

    conversation = db.add_conversation({"title": "legacy-" + origin})
    acceptance = _acceptance(conversation, suffix="legacy-" + origin)
    db.add_message(
        {
            "id": acceptance.user_message_id,
            "conversation_id": conversation,
            "sender": "user",
            "content": acceptance.user_content,
        }
    )
    db.add_message(
        {
            "id": acceptance.assistant_message_id,
            "conversation_id": conversation,
            "parent_message_id": acceptance.user_message_id,
            "sender": "assistant",
            "content": "",
            "assistant_generation_state": "accepted",
        }
    )
    db.set_conversation_active_leaf(conversation, acceptance.assistant_message_id)
    connection = db.get_connection()
    connection.execute(
        "UPDATE messages SET version=5 WHERE id=?", (acceptance.user_message_id,)
    )
    connection.execute(
        "UPDATE messages SET version=6 WHERE id=?", (acceptance.assistant_message_id,)
    )
    connection.execute(
        "INSERT INTO console_dispatch_checkpoints "
        "(assistant_message_id,user_message_id,conversation_id,schema_version,preparation_id,attempt_id,state,"
        "checkpoint_revision,user_message_version,assistant_message_version,origin,queue_entry_id,"
        "frozen_authority_json,resolved_destination_json,reconstructability_json,created_at,updated_at) "
        "VALUES (?,?,?,1,?,?,'accepted',9,5,6,?,?,?,?,?,'2026-09-03T01:02:03','2026-09-04T04:05:06')",
        (
            acceptance.assistant_message_id,
            acceptance.user_message_id,
            conversation,
            acceptance.preparation_id,
            acceptance.attempt_id,
            origin,
            queue_entry_id,
            dump_console_turn_library_authority_json(acceptance.frozen_authority),
            dump_console_resolved_destination_json(acceptance.resolved_destination),
            dump_console_dispatch_reconstructability_json(
                acceptance.reconstructability
            ),
        ),
    )
    connection.commit()
    return conversation


def _assert_legacy_queue_quarantine(db):
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleDispatchResultStatus,
    )
    from tldw_chatbook.Chat.console_dispatch_repository import _OWNER_SELECT

    connection = db.get_connection()
    before = tuple(connection.iterdump())
    rows = connection.execute(
        _OWNER_SELECT + " WHERE checkpoint.preparation_id LIKE 'preparation-legacy-%'"
    ).fetchall()
    assert len(rows) == 2
    for stored in rows:
        row = dict(stored)
        # The same owner's payload/state/versions validate with only its queue corrected in memory.
        row["queue_entry_id"] = "valid-queue" if row["origin"] == "queued" else None
        checkpoint, error = ConsoleDispatchRepository._checkpoint_from_row(row)
        assert checkpoint is not None and error is None
        result = ConsoleDispatchRepository(db).read_for_session(
            stored["conversation_id"]
        )
        assert result.status is ConsoleDispatchResultStatus.QUARANTINED
        assert result.error_code == "invalid_checkpoint_owner"
        assert result.checkpoint is None
    assert tuple(connection.iterdump()) == before


@pytest.mark.parametrize("dictionary", [False, True], ids=["initial", "updated"])
@pytest.mark.parametrize("route", ["constructor76", "standalone76", "constructor75"])
def test_populated_legacy_queue_paths_are_lossless(tmp_path, dictionary, route):
    from Tests.Backup_Recovery.test_chachanotes_native76_compatibility import (
        _checkpoint_state,
        _shipped76,
    )
    from tldw_chatbook.DB.recovery_core_schema import (
        CHACHANOTES_V80_SCHEMAS,
        CHACHANOTES_DICTIONARY_UPDATE_SCHEMA,
    )

    path = tmp_path / "populated.sqlite"
    before, messages, indexes = _shipped76(
        path, dictionary=dictionary, version=75 if route == "constructor75" else 76
    )
    with closing(sqlite3.connect(path)) as connection:
        notes = tuple(connection.execute("SELECT * FROM notes ORDER BY id"))
        index_sql = tuple(
            connection.execute(
                "SELECT name,sql FROM sqlite_schema WHERE type='index' AND name LIKE 'idx_console_dispatch_checkpoint%' ORDER BY name"
            )
        )
    if route == "standalone76":
        migration = (
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook/DB/migrations/chachanotes_v76_to_v77_agent_chat_starts.sql"
        )
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            statement = ""
            for line in migration.read_text().splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    connection.execute(statement)
                    statement = ""
            assert not statement.strip()
            assert (
                connection.execute(
                    "UPDATE db_schema_version SET version=77 WHERE schema_name=? AND version=76",
                    (CharactersRAGDB._SCHEMA_NAME,),
                ).rowcount
                == 1
            )
            connection.commit()
    for _ in range(2):
        db = CharactersRAGDB(path, client_id="lossless-reopen")
        try:
            connection = db.get_connection()
            after, restored_messages, restored_indexes = _checkpoint_state(path)
            for row in after:
                assert row.pop("agent_chat_start_attempt_id") is None
                assert len(row) == 17
            assert after == before
            assert restored_messages == messages
            assert all(index in restored_indexes for index in indexes)
            actual_indexes = tuple(
                tuple(row)
                for row in connection.execute(
                    "SELECT name,sql FROM sqlite_schema WHERE type='index' AND name LIKE 'idx_console_dispatch_checkpoint%' ORDER BY name"
                )
            )
            assert actual_indexes == (
                (
                    "idx_console_dispatch_checkpoint_conversation",
                    "CREATE INDEX idx_console_dispatch_checkpoint_conversation\n    ON console_dispatch_checkpoints(conversation_id)",
                ),
                (
                    "idx_console_dispatch_checkpoints_user_message",
                    "CREATE INDEX idx_console_dispatch_checkpoints_user_message\n    ON console_dispatch_checkpoints(user_message_id)",
                ),
            )
            assert index_sql == (
                (
                    "idx_console_dispatch_checkpoint_conversation",
                    "CREATE INDEX idx_console_dispatch_checkpoint_conversation\n    ON console_dispatch_checkpoints(conversation_id)",
                ),
                (
                    "idx_console_dispatch_checkpoints_user_message",
                    "CREATE INDEX idx_console_dispatch_checkpoints_user_message\n  ON console_dispatch_checkpoints(user_message_id)",
                ),
            )
            with closing(sqlite3.connect(path)) as notes_connection:
                assert (
                    tuple(notes_connection.execute("SELECT * FROM notes ORDER BY id"))
                    == notes
                )
            catalog = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                )
            )
            assert catalog == (
                CHACHANOTES_DICTIONARY_UPDATE_SCHEMA
                if dictionary
                else CHACHANOTES_V80_SCHEMAS[0]
            )
            assert db._get_db_version(connection) == 80
            assert not connection.execute("PRAGMA foreign_key_check").fetchall()
            assert tuple(connection.execute("PRAGMA quick_check").fetchone()) == ("ok",)
            assert connection.execute(
                "SELECT rowid FROM notes_fts WHERE notes_fts MATCH 'lossless'"
            ).fetchall()
            _assert_legacy_queue_quarantine(db)
        finally:
            db.close_connection()


@pytest.mark.parametrize(
    "origin,queue",
    [
        ("queued", None),
        ("queued", ""),
        ("manual", "queue"),
        ("agent_chat_start", "queue"),
    ],
    ids=["queued-null", "queued-empty", "manual-queue", "machine-queue"],
)
def test_new_queue_writes_refuse_atomically(tmp_path, origin, queue):
    db, conversation = _db_and_conversation(tmp_path / "refusal.sqlite")
    try:
        repository = ConsoleDispatchRepository(db)
        _insert(db, repository, _acceptance(conversation))
        other = db.add_conversation({"title": "refused new owner"})
        acceptance = replace(
            _acceptance(other, suffix="new"), origin=origin, queue_entry_id=queue
        )
        if origin == "agent_chat_start":
            acceptance = replace(
                acceptance,
                agent_chat_start_attempt_id="start",
                agent_chat_start=message_metadata.AgentChatStartMetadata(
                    "start", "run", "source"
                ),
                handoff_draft_revision=1,
            )
        before = tuple(db.get_connection().iterdump())
        with pytest.raises(ValueError):
            _insert(db, repository, acceptance)
        assert tuple(db.get_connection().iterdump()) == before
        assert not db.get_messages_for_conversation(other)
    finally:
        db.close_connection()


@pytest.mark.parametrize("origin", ["manual", "queued", "agent_chat_start"])
def test_valid_queue_writes_remain_readable(tmp_path, origin):
    db, conversation = _db_and_conversation(tmp_path / "valid.sqlite")
    try:
        acceptance = replace(
            _acceptance(conversation),
            origin=origin,
            queue_entry_id="queue" if origin == "queued" else None,
        )
        if origin == "agent_chat_start":
            acceptance = replace(
                acceptance,
                agent_chat_start_attempt_id="start",
                agent_chat_start=message_metadata.AgentChatStartMetadata(
                    "start", "run", "source"
                ),
                handoff_draft_revision=1,
            )
        repository = ConsoleDispatchRepository(db)
        checkpoint = _insert(db, repository, acceptance)
        assert repository.read_for_session(conversation).checkpoint == checkpoint
        assert len(db.get_messages_for_conversation(conversation)) == 2
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "origin,queue,attempt",
    [
        ("agent_chat_start", "queue", "start"),
        ("agent_chat_start", None, None),
        ("agent_chat_start", None, ""),
        ("agent_chat_start", None, "x" * 201),
        ("manual", None, "start"),
        ("queued", "queue", "start"),
        ("agent_chat_start", None, "duplicate"),
    ],
    ids=[
        "machine-queue",
        "missing-attempt",
        "empty-attempt",
        "overlong-attempt",
        "manual-attempt",
        "queued-attempt",
        "duplicate-attempt",
    ],
)
def test_raw_machine_constraints_remain_strict(tmp_path, origin, queue, attempt):
    db, conversation = _db_and_conversation(tmp_path / "raw.sqlite")
    try:
        repository = ConsoleDispatchRepository(db)
        acceptance = replace(
            _acceptance(conversation),
            origin="agent_chat_start",
            agent_chat_start_attempt_id="duplicate",
            agent_chat_start=message_metadata.AgentChatStartMetadata(
                "duplicate", "run", "source"
            ),
            handoff_draft_revision=1,
        )
        _insert(db, repository, acceptance)
        other = db.add_conversation({"title": "raw attempt"})
        _insert(db, repository, _acceptance(other, suffix="other"))
        before = tuple(db.get_connection().iterdump())
        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE console_dispatch_checkpoints SET origin=?,queue_entry_id=?,agent_chat_start_attempt_id=? WHERE conversation_id=?",
                    (origin, queue, attempt, other),
                )
        assert tuple(db.get_connection().iterdump()) == before
    finally:
        db.close_connection()


@pytest.mark.parametrize("dictionary", [False, True], ids=["initial", "updated"])
@pytest.mark.parametrize("subscriptions", [False, True], ids=["primary", "shared"])
def test_repaired77_false_native76_constructor_refuses(
    tmp_path, dictionary, subscriptions
):
    from tldw_chatbook.DB.ChaChaNotes_DB import SchemaError
    from tldw_chatbook.DB.recovery_core_schema import _CHAT_DICTIONARIES_UPDATED_TRIGGER

    if subscriptions:
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

        db = SubscriptionsDB(tmp_path / "false76.sqlite")
        db.close()
    path = tmp_path / "false76.sqlite"
    db = CharactersRAGDB(path, client_id="fresh77")
    try:
        connection = db.get_connection()
        if dictionary:
            connection.execute("DROP TRIGGER chat_dictionaries_au")
            connection.execute(_CHAT_DICTIONARIES_UPDATED_TRIGGER)
        connection.execute(
            "UPDATE db_schema_version SET version=76 WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        )
        connection.commit()
    finally:
        db.close_connection()
    with closing(sqlite3.connect(path)) as connection:
        before = tuple(connection.iterdump())
    with pytest.raises(SchemaError, match="unsupported native receipt catalog"):
        CharactersRAGDB(path, client_id="false76-refuse")
    with closing(sqlite3.connect(path)) as connection:
        assert tuple(connection.iterdump()) == before
