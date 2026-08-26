"""Selected Console generations retain bounded thinking as one owner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Tests.Chat.test_console_dispatch_recovery import _database, _insert, _acceptance
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleThinkingCompatibilityError,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
    parse_thinking_blocks_json,
)


def _thinking(text: str, *, status: str = "complete") -> ThinkingEnvelope:
    return ThinkingEnvelope(
        (
            DisplayableThinkingBlock(
                block_id="reasoning-1",
                round_ordinal=0,
                provider="llama_cpp",
                model="test-model",
                protocol="chat_completions",
                source_format="think_tag",
                status=status,
                text=text,
            ),
        )
    )


def _restored_store(db, conversation_id: str) -> tuple[ConsoleChatStore, str]:
    rows = db.get_messages_for_conversation(conversation_id, limit=100)
    nodes = [
        ConsoleChatMessage(
            id=str(row["id"]),
            role=ConsoleMessageRole(str(row["role"])),
            content=str(row.get("content") or ""),
            persisted_message_id=str(row["id"]),
            parent_message_id=row.get("parent_message_id"),
        )
        for row in rows
    ]
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="thinking",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=nodes,
        active_leaf_persisted_id="assistant-1",
    )
    return store, session.id


def test_restore_hydrates_supported_thinking(tmp_path: Path) -> None:
    db, conversation_id, repository = _database(tmp_path / "supported.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    canonical = dump_thinking_blocks_json(_thinking("restored reasoning"))
    db.get_connection().execute(
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = 'assistant-1'",
        (canonical,),
    )

    store, _session_id = _restored_store(db, conversation_id)
    restored = store.get_message("assistant-1")

    assert restored.thinking == _thinking("restored reasoning")
    assert restored.opaque_thinking_json is None
    assert restored.thinking_actions_enabled is True


def test_restore_preserves_unknown_opaque_and_blocks_generation_mutations(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "unknown.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    raw = '{ "version" : 99, "future" : {"secret":"value"} }'
    db.get_connection().execute(
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = 'assistant-1'",
        (raw,),
    )
    store, _session_id = _restored_store(db, conversation_id)
    restored = store.get_message("assistant-1")

    assert restored.opaque_thinking_json == raw
    assert restored.thinking_warning == "Thinking data version is unsupported."
    assert restored.thinking_actions_enabled is False
    assert "secret" not in repr(restored)
    with pytest.raises(
        ConsoleThinkingCompatibilityError, match="newer thinking format"
    ):
        store.begin_variant_stream("assistant-1")
    with pytest.raises(
        ConsoleThinkingCompatibilityError, match="upgrade before editing"
    ):
        store.update_message_content("assistant-1", "must not commit")
    assert store.get_message("assistant-1").content == ""

    store.set_message_feedback("assistant-1", "up")
    durable = (
        db.get_connection()
        .execute("SELECT thinking_blocks_json FROM messages WHERE id = 'assistant-1'")
        .fetchone()
    )
    assert durable["thinking_blocks_json"] == raw


def test_persist_selected_generation_replaces_projection_and_refreshes_version(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "projection.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    db.get_connection().execute(
        "UPDATE messages SET content = 'old', assistant_generation_state = 'complete' "
        "WHERE id = 'assistant-1'"
    )
    store, _session_id = _restored_store(db, conversation_id)
    message = store._message_or_raise("assistant-1")
    message.content = "new answer"
    message.thinking = _thinking("new reasoning")
    message.assistant_generation_state = "complete"

    committed = store.persist_selected_generation("assistant-1")
    row = db.get_message_by_id("assistant-1")

    assert committed is True
    assert row["content"] == "new answer"
    assert row["thinking_blocks_json"] == dump_thinking_blocks_json(message.thinking)
    assert message.provider_continuation_message_version == row["version"]


def test_malformed_known_thinking_is_content_free_and_blocks_generation(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "malformed.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    raw = json.dumps({"version": 1, "blocks": [{"text": "do-not-leak"}]})
    db.get_connection().execute(
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = 'assistant-1'",
        (raw,),
    )

    store, _session_id = _restored_store(db, conversation_id)
    restored = store.get_message("assistant-1")

    assert restored.thinking is None
    assert restored.opaque_thinking_json is None
    assert restored.thinking_warning is not None
    assert "do-not-leak" not in restored.thinking_warning
    assert restored.thinking_actions_enabled is False


def test_explicit_assistant_edit_clears_generation_provenance(tmp_path: Path) -> None:
    db, conversation_id, repository = _database(tmp_path / "edit.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    canonical = dump_thinking_blocks_json(_thinking("old reasoning"))
    db.get_connection().execute(
        "UPDATE messages SET content = 'old answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (canonical,),
    )
    store, _session_id = _restored_store(db, conversation_id)

    edited = store.update_message_content("assistant-1", "human correction")
    row = db.get_message_by_id("assistant-1")

    assert edited.thinking is None
    assert edited.assistant_generation_state == "complete"
    assert row["content"] == "human correction"
    assert row["thinking_blocks_json"] is None
    assert row["provider_continuation_json"] is None
    assert row["assistant_generation_state"] == "complete"


@pytest.mark.parametrize(
    ("terminal", "expected_status"),
    [
        ("mark_message_complete", "complete"),
        ("mark_message_stopped", "stopped"),
        ("mark_message_failed", "failed"),
    ],
)
def test_normal_terminal_projects_paired_thinking_status(
    tmp_path: Path, terminal: str, expected_status: str
) -> None:
    db, conversation_id, repository = _database(
        tmp_path / f"terminal-{expected_status}.sqlite"
    )
    _insert(db, repository, _acceptance(conversation_id))
    db.get_connection().execute(
        "DELETE FROM console_dispatch_checkpoints WHERE assistant_message_id = "
        "'assistant-1'"
    )
    store, session_id = _restored_store(db, conversation_id)
    store._dispatch_recoveries_by_session.pop(session_id, None)
    live = store._message_or_raise("assistant-1")
    live.status = "streaming"
    live.content = "terminal answer"
    live.thinking = _thinking("terminal reasoning")
    live.assistant_generation_state = "streaming"

    getattr(store, terminal)("assistant-1")
    row = db.get_message_by_id("assistant-1")
    durable = parse_thinking_blocks_json(row["thinking_blocks_json"])

    assert row["assistant_generation_state"] == expected_status
    assert {block.status for block in durable.blocks} == {expected_status}
