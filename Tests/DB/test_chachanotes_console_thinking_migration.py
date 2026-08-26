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
MESSAGE_PAYLOAD_KEYS = {
    "id",
    "conversation_id",
    "parent_message_id",
    "sender",
    "content",
    "image_mime_type",
    "provider_continuation_json",
    "thinking_blocks_json",
    "assistant_generation_state",
    "timestamp",
    "ranking",
    "last_modified",
    "deleted",
    "client_id",
    "version",
}
CONVERSATION_PAYLOAD_KEYS = {
    "id",
    "root_id",
    "forked_from_message_id",
    "parent_conversation_id",
    "character_id",
    "assistant_kind",
    "assistant_id",
    "persona_memory_mode",
    "scope_type",
    "workspace_id",
    "state",
    "topic_label",
    "topic_label_source",
    "topic_last_tagged_at",
    "topic_last_tagged_message_id",
    "cluster_id",
    "source",
    "external_ref",
    "runtime_backend",
    "discovery_owner",
    "discovery_entity_id",
    "system_prompt",
    "metadata",
    "thinking_history_policy",
    "title",
    "rating",
    "created_at",
    "last_modified",
    "deleted",
    "client_id",
    "version",
}
MESSAGE_DELETE_PAYLOAD_KEYS = {
    "id",
    "deleted",
    "last_modified",
    "assistant_generation_state",
    "version",
    "client_id",
}
UNSUPPORTED_THINKING = '{"version":2,"opaque":"must-not-escape"}'


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


def _active_continuation() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": ["private continuation"],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                            "state": "pending",
                        }
                    ],
                }
            ],
        }
    )


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


def _raw_message(db: CharactersRAGDB, message_id: str) -> dict[str, object]:
    row = (
        db.get_connection()
        .execute(
            """
        SELECT content, thinking_blocks_json, provider_continuation_json,
               assistant_generation_state, usage_json, version, deleted
          FROM messages
         WHERE id = ?
        """,
            (message_id,),
        )
        .fetchone()
    )
    assert row is not None
    return dict(row)


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
    message_create = _payloads(db, "messages")[-1]
    conversation_create = _payloads(db, "conversations")[-1]
    assert set(message_create) == MESSAGE_PAYLOAD_KEYS
    assert set(conversation_create) == CONVERSATION_PAYLOAD_KEYS
    assert message_create["thinking_blocks_json"] == _thinking("boundary-canary")
    assert conversation_create["thinking_history_policy"] == "include"

    assert db.update_message(
        assistant_id,
        {"thinking_blocks_json": "\n" + _thinking("updated-canary") + "\n"},
        expected_version=1,
    )
    assert db.get_message_by_id(assistant_id)["thinking_blocks_json"] == _thinking(
        "updated-canary"
    )
    message_update = _payloads(db, "messages")[-1]
    assert set(message_update) == MESSAGE_PAYLOAD_KEYS
    assert message_update["thinking_blocks_json"] == _thinking("updated-canary")

    assert db.update_conversation(
        conversation_id,
        {"thinking_history_policy": "exclude"},
        expected_version=1,
    )
    assert (
        db.get_conversation_by_id(conversation_id)["thinking_history_policy"]
        == "exclude"
    )
    conversation_update = _payloads(db, "conversations")[-1]
    assert set(conversation_update) == CONVERSATION_PAYLOAD_KEYS
    assert conversation_update["thinking_history_policy"] == "exclude"


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


def test_projection_rejects_unsupported_current_thinking_without_mutation(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "opaque projection"})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "old answer",
            "thinking_blocks_json": _thinking("old thought"),
        }
    )
    assert message_id is not None
    db.get_connection().execute(
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = ?",
        (UNSUPPORTED_THINKING, message_id),
    )
    db.get_connection().commit()
    before = _raw_message(db, message_id)

    with pytest.raises(InputError) as error:
        db.replace_assistant_generation_projection(
            message_id=message_id,
            content="replacement",
            thinking_blocks_json=_thinking("replacement thought"),
            provider_continuation_json=None,
            assistant_generation_state="complete",
            usage_json='{"output_tokens":9}',
            expected_version=1,
        )

    assert "must-not-escape" not in str(error.value)
    assert _raw_message(db, message_id) == before


@pytest.mark.parametrize("requested_thinking", [None, _thinking("replacement")])
def test_update_message_rejects_requested_thinking_over_unsupported_current_value(
    db: CharactersRAGDB,
    requested_thinking: str | None,
) -> None:
    conversation_id = db.add_conversation({"title": "opaque update"})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "old answer",
            "thinking_blocks_json": _thinking("old thought"),
        }
    )
    assert message_id is not None
    db.get_connection().execute(
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = ?",
        (UNSUPPORTED_THINKING, message_id),
    )
    db.get_connection().commit()
    before = _raw_message(db, message_id)

    with pytest.raises(InputError) as error:
        db.update_message(
            message_id,
            {"content": "replacement", "thinking_blocks_json": requested_thinking},
            expected_version=1,
        )

    assert "must-not-escape" not in str(error.value)
    assert _raw_message(db, message_id) == before

    assert db.update_message(
        message_id,
        {"content": "unrelated edit"},
        expected_version=1,
    )
    after_omission = _raw_message(db, message_id)
    assert after_omission["thinking_blocks_json"] == UNSUPPORTED_THINKING
    assert after_omission["content"] == "unrelated edit"
    assert after_omission["version"] == 2


def test_projection_owner_guards_and_optional_version(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "projection owners"})
    user_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "question",
        }
    )
    deleted_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "deleted answer",
            "thinking_blocks_json": _thinking("deleted thought"),
        }
    )
    active_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "active answer",
            "thinking_blocks_json": _thinking("active thought"),
        }
    )
    assert user_id is not None and deleted_id is not None and active_id is not None
    assert db.soft_delete_message(deleted_id, expected_version=1)
    user_before = _raw_message(db, user_id)
    deleted_before = _raw_message(db, deleted_id)

    with pytest.raises(InputError, match="assistant message"):
        db.replace_assistant_generation_projection(
            message_id=user_id,
            content="bad owner",
            thinking_blocks_json=None,
            provider_continuation_json=None,
            assistant_generation_state="complete",
            usage_json=None,
        )
    with pytest.raises(ConflictError, match="version conflict"):
        db.replace_assistant_generation_projection(
            message_id=deleted_id,
            content="deleted owner",
            thinking_blocks_json=None,
            provider_continuation_json=None,
            assistant_generation_state="complete",
            usage_json=None,
        )
    assert _raw_message(db, user_id) == user_before
    assert _raw_message(db, deleted_id) == deleted_before

    version = db.replace_assistant_generation_projection(
        message_id=active_id,
        content="unguarded replacement",
        thinking_blocks_json=_thinking("unguarded thought"),
        provider_continuation_json=None,
        assistant_generation_state="complete",
        usage_json=None,
        expected_version=None,
    )
    assert version == 2
    assert _raw_message(db, active_id)["content"] == "unguarded replacement"


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
    db.get_connection().execute(
        "UPDATE messages SET provider_continuation_json = ? WHERE id = ?",
        ("single-continuation", message_id),
    )
    db.get_connection().commit()
    db.get_connection().execute("DELETE FROM sync_log")
    db.get_connection().commit()

    assert db.update_message_usage_local(message_id, '{"output_tokens":2}')
    assert db.update_message_metadata_local(message_id, '{"local":true}')
    assert _payloads(db, "messages") == []

    assert db.soft_delete_message(message_id, expected_version=1)
    assert db.get_message_by_id(message_id) is None
    tombstone = _raw_message(db, message_id)
    assert tombstone["thinking_blocks_json"] is None
    assert tombstone["provider_continuation_json"] == "single-continuation"
    delete_payload = _payloads(db, "messages")[-1]
    assert set(delete_payload) == MESSAGE_DELETE_PAYLOAD_KEYS


def test_subtree_tombstones_clear_thinking_but_retain_continuation(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "subtree"})
    root_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "root",
            "thinking_blocks_json": _thinking("root thought"),
        }
    )
    child_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": root_id,
            "sender": "assistant",
            "content": "child",
            "thinking_blocks_json": _thinking("child thought"),
        }
    )
    assert root_id is not None and child_id is not None
    db.get_connection().executemany(
        "UPDATE messages SET provider_continuation_json = ? WHERE id = ?",
        (("root-continuation", root_id), ("child-continuation", child_id)),
    )
    db.get_connection().commit()

    tombstones = db.soft_delete_message_subtree(root_id, expected_version=1)

    assert {row["message_id"] for row in tombstones} == {root_id, child_id}
    root = _raw_message(db, root_id)
    child = _raw_message(db, child_id)
    assert root["deleted"] == child["deleted"] == 1
    assert root["thinking_blocks_json"] is None
    assert child["thinking_blocks_json"] is None
    assert root["provider_continuation_json"] == "root-continuation"
    assert child["provider_continuation_json"] == "child-continuation"


def test_content_edit_tombstones_descendant_thinking_only(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "edit subtree"})
    root_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "original question",
        }
    )
    child_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": root_id,
            "sender": "assistant",
            "content": "old answer",
            "thinking_blocks_json": _thinking("old descendant thought"),
        }
    )
    assert root_id is not None and child_id is not None
    db.get_connection().execute(
        "UPDATE messages SET provider_continuation_json = ? WHERE id = ?",
        ("descendant-continuation", child_id),
    )
    db.get_connection().commit()

    assert db.update_message(
        root_id,
        {"content": "edited question"},
        expected_version=1,
    )

    child = _raw_message(db, child_id)
    assert child["deleted"] == 1
    assert child["thinking_blocks_json"] is None
    assert child["provider_continuation_json"] == "descendant-continuation"


def test_continuation_discard_tombstone_clears_thinking_in_raw_storage(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "continuation discard"})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "",
            "thinking_blocks_json": _thinking("discarded thought"),
            "provider_continuation_json": _active_continuation(),
        }
    )
    assert message_id is not None

    assert db.update_provider_continuation(
        message_id=message_id,
        expected_message_version=1,
        provider_continuation_json=None,
    )

    assert db.get_message_by_id(message_id) is None
    tombstone = _raw_message(db, message_id)
    assert tombstone["deleted"] == 1
    assert tombstone["version"] == 2
    assert tombstone["provider_continuation_json"] is None
    assert tombstone["thinking_blocks_json"] is None


def test_v49_to_v50_requires_exact_entry_version(db: CharactersRAGDB) -> None:
    with pytest.raises(SchemaError, match="requires schema version 49"):
        db._migrate_from_v49_to_v50(db.get_connection())
