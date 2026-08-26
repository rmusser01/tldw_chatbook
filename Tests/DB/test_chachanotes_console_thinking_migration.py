"""ChaChaNotes v51 -> v52 Console thinking persistence contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
    SchemaError,
)


SCHEMA_NAME = "rag_char_chat_schema"


def _thinking(text: str = "thinkingonlycanary") -> str:
    raw = dump_thinking_blocks_json(
        ThinkingEnvelope(
            blocks=(
                DisplayableThinkingBlock(
                    block_id="round-0",
                    round_ordinal=0,
                    provider="llama_cpp",
                    model="qwen3",
                    protocol="openai_chat",
                    source_format="start_anchored_think",
                    status="complete",
                    text=text,
                ),
            )
        )
    )
    assert raw is not None
    return raw


def _schema_version(db: CharactersRAGDB) -> int:
    row = (
        db.get_connection()
        .execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        )
        .fetchone()
    )
    return int(row[0])


def _payloads(db: CharactersRAGDB, entity: str) -> list[dict[str, object]]:
    rows = db.get_connection().execute(
        "SELECT payload FROM sync_log WHERE entity = ? ORDER BY change_id", (entity,)
    )
    return [json.loads(row[0]) for row in rows]


def _seed_v49(path: Path) -> tuple[str, str, str]:
    conversation_id = "conversation-1"
    assistant_id = "assistant-1"
    user_id = "user-1"
    with chachanotes_db_at_version(path, 49, client_id="v49-seed") as historical:
        connection = historical.get_connection()
        connection.execute(
            """
            INSERT INTO conversations(id, root_id, title, client_id)
            VALUES (?, ?, 'historical', 'v49-seed')
            """,
            (conversation_id, conversation_id),
        )
        connection.executemany(
            """
            INSERT INTO messages(
                id, conversation_id, sender, role, content, client_id
            ) VALUES (?, ?, ?, ?, ?, 'v49-seed')
            """,
            (
                (assistant_id, conversation_id, "assistant", "assistant", "visible"),
                (user_id, conversation_id, "user", "user", "user text"),
            ),
        )
        connection.commit()
        message_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(messages)")
        }
        conversation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(conversations)")
        }
        assert "thinking_blocks_json" not in message_columns
        assert "thinking_history_policy" not in conversation_columns
        assert _schema_version(historical) == 49
    return conversation_id, assistant_id, user_id


def test_console_thinking_migration_is_additive_without_evidence_backfill(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "historical-v49.sqlite"
    conversation_id, assistant_id, user_id = _seed_v49(db_path)

    db = CharactersRAGDB(db_path, client_id="v52-test")
    try:
        assert _schema_version(db) == CharactersRAGDB._CURRENT_SCHEMA_VERSION == 52
        assert db.get_message_by_id(assistant_id)["thinking_blocks_json"] is None
        assert db.get_message_by_id(user_id)["thinking_blocks_json"] is None
        assert (
            db.get_conversation_by_id(conversation_id)["thinking_history_policy"]
            is None
        )
    finally:
        db.close_connection()


def test_migration_rebuilds_sync_triggers_without_widening_fts(tmp_path: Path) -> None:
    db_path = tmp_path / "historical-v49.sqlite"
    conversation_id, assistant_id, _ = _seed_v49(db_path)
    db = CharactersRAGDB(db_path, client_id="v52-test")
    try:
        triggers = {
            row[0]: row[1]
            for row in db.get_connection().execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
            )
        }
        for name in (
            "messages_sync_create",
            "messages_sync_update",
            "messages_sync_undelete",
        ):
            assert "thinking_blocks_json" in triggers[name]
        assert "thinking_blocks_json" not in triggers["messages_sync_delete"]
        for name in (
            "conversations_sync_create",
            "conversations_sync_update",
            "conversations_sync_undelete",
        ):
            assert "thinking_history_policy" in triggers[name]
        assert "thinking_history_policy" not in triggers["conversations_sync_delete"]

        before_fts = triggers["messages_au"].lower()
        assert "after update of content, deleted on messages" in before_fts
        db.get_connection().execute(
            "UPDATE messages SET thinking_blocks_json = ? WHERE id = ?",
            (_thinking(), assistant_id),
        )
        db.get_connection().commit()
        matches = (
            db.get_connection()
            .execute(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
                ("thinkingonlycanary",),
            )
            .fetchall()
        )
        assert matches == []
        assert db.get_conversation_by_id(conversation_id) is not None
    finally:
        db.close_connection()


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    instance = CharactersRAGDB(tmp_path / "current.sqlite", client_id="thinking-test")
    yield instance
    instance.close_connection()


def test_message_and_conversation_boundaries_canonicalize_and_sync(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation(
        {"title": "thinking", "thinking_history_policy": "include"}
    )
    assistant_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "answer",
            "thinking_blocks_json": "  " + _thinking("boundary-canary") + "  ",
        }
    )
    assert assistant_id is not None

    message = db.get_message_by_id(assistant_id)
    assert message["thinking_blocks_json"] == _thinking("boundary-canary")
    assert (
        db.get_conversation_by_id(conversation_id)["thinking_history_policy"]
        == "include"
    )
    assert _payloads(db, "messages")[-1]["thinking_blocks_json"] == _thinking(
        "boundary-canary"
    )
    assert _payloads(db, "conversations")[-1]["thinking_history_policy"] == "include"

    assert db.update_message(
        assistant_id,
        {"thinking_blocks_json": "\n" + _thinking("updated-canary") + "\n"},
        expected_version=1,
    )
    assert db.get_message_by_id(assistant_id)["thinking_blocks_json"] == _thinking(
        "updated-canary"
    )
    assert _payloads(db, "messages")[-1]["thinking_blocks_json"] == _thinking(
        "updated-canary"
    )

    assert db.update_conversation(
        conversation_id,
        {"thinking_history_policy": "exclude"},
        expected_version=1,
    )
    assert (
        db.get_conversation_by_id(conversation_id)["thinking_history_policy"]
        == "exclude"
    )
    assert _payloads(db, "conversations")[-1]["thinking_history_policy"] == "exclude"


def test_invalid_thinking_and_policy_are_rejected_before_mutation(
    db: CharactersRAGDB,
) -> None:
    with pytest.raises(InputError, match="thinking history policy"):
        db.add_conversation({"title": "bad", "thinking_history_policy": "required"})

    conversation_id = db.add_conversation({"title": "valid"})
    with pytest.raises(InputError, match="thinking history policy"):
        db.update_conversation(
            conversation_id,
            {"thinking_history_policy": "required"},
            expected_version=1,
        )
    assert db.get_conversation_by_id(conversation_id)["version"] == 1

    with pytest.raises(InputError, match="thinking"):
        db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "answer",
                "thinking_blocks_json": '{"version":2,"blocks":[]}',
            }
        )
    with pytest.raises(InputError, match="assistant message"):
        db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "question",
                "thinking_blocks_json": _thinking(),
            }
        )


def test_selected_assistant_projection_is_atomic_and_optimistically_guarded(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "projection"})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "old answer",
            "thinking_blocks_json": _thinking("old-thought"),
        }
    )
    assert message_id is not None
    sync_count_before = (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM sync_log WHERE entity = 'messages'")
        .fetchone()[0]
    )

    version = db.replace_assistant_generation_projection(
        message_id=message_id,
        content="new answer",
        thinking_blocks_json="\n" + _thinking("new-thought") + "\n",
        provider_continuation_json=None,
        assistant_generation_state="complete",
        usage_json='{"output_tokens":7}',
        expected_version=1,
    )
    assert version == 2
    projected = db.get_message_by_id(message_id)
    assert projected["content"] == "new answer"
    assert projected["thinking_blocks_json"] == _thinking("new-thought")
    assert projected["assistant_generation_state"] == "complete"
    assert projected["usage_json"] == '{"output_tokens":7}'
    message_payloads = _payloads(db, "messages")
    assert len(message_payloads) == sync_count_before + 1
    assert message_payloads[-1]["content"] == "new answer"
    assert message_payloads[-1]["thinking_blocks_json"] == _thinking("new-thought")

    with pytest.raises(ConflictError, match="version conflict"):
        db.replace_assistant_generation_projection(
            message_id=message_id,
            content="stale answer",
            thinking_blocks_json=None,
            provider_continuation_json=None,
            assistant_generation_state="failed",
            usage_json=None,
            expected_version=1,
        )
    assert db.get_message_by_id(message_id) == projected

    with pytest.raises(InputError, match="thinking"):
        db.replace_assistant_generation_projection(
            message_id=message_id,
            content="invalid answer",
            thinking_blocks_json="not-json",
            provider_continuation_json=None,
            assistant_generation_state="failed",
            usage_json=None,
            expected_version=2,
        )
    assert db.get_message_by_id(message_id) == projected

    with pytest.raises(InputError, match="continuation"):
        db.replace_assistant_generation_projection(
            message_id=message_id,
            content="invalid answer",
            thinking_blocks_json=None,
            provider_continuation_json="{}",
            assistant_generation_state="failed",
            usage_json=None,
            expected_version=2,
        )
    assert db.get_message_by_id(message_id) == projected


def test_local_only_writes_stay_out_of_sync_and_tombstones_hide_thinking(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "privacy"})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "answer",
            "thinking_blocks_json": _thinking(),
        }
    )
    assert message_id is not None
    db.get_connection().execute("DELETE FROM sync_log")
    db.get_connection().commit()

    assert db.update_message_usage_local(message_id, '{"output_tokens":2}')
    assert db.update_message_metadata_local(message_id, '{"local":true}')
    assert _payloads(db, "messages") == []

    assert db.soft_delete_message(message_id, expected_version=1)
    assert db.get_message_by_id(message_id) is None


def test_v49_to_v50_requires_exact_entry_version(db: CharactersRAGDB) -> None:
    with pytest.raises(SchemaError, match="requires schema version 49"):
        db._migrate_from_v49_to_v50(db.get_connection())
