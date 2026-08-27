"""Chatbook V2 round-trip tests for canonical model-thinking envelopes."""

from __future__ import annotations

import copy
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import tldw_chatbook.Chatbooks.chatbook_importer as importer_module
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.Chat.thinking_blocks import (
    MAX_THINKING_BLOCKS,
    MAX_THINKING_TEXT_BYTES,
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
)
from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


THINKING_EXPORT_WARNING = (
    "This conversation export contains model thinking or private provider "
    "continuation. Treat it as sensitive conversation data."
)
UNKNOWN_POLICY_WARNING = "Unknown thinking history policy was reset to Auto."


def _thinking(*, proprietary: bool = False, text: str = "CHATBOOK-THINKING-CANARY"):
    block = (
        ProprietaryThinkingBlock(
            block_id="proprietary-1",
            round_ordinal=0,
            provider="moonshot",
            model="kimi-k2.6",
            protocol="chat_completions",
            source_format="reasoning_content",
            status="complete",
        )
        if proprietary
        else DisplayableThinkingBlock(
            block_id="displayable-1",
            round_ordinal=0,
            provider="llama.cpp",
            model="local-reasoner",
            protocol="chat_completions",
            source_format="reasoning_content",
            status="complete",
            text=text,
        )
    )
    return ThinkingEnvelope(blocks=(block,))


def _too_many_exchange_blocks() -> dict[str, object]:
    block = json.loads(dump_thinking_blocks_json(_thinking()) or "null")[
        "blocks"
    ][0]
    blocks = []
    for ordinal in range(MAX_THINKING_BLOCKS + 1):
        item = dict(block)
        item.update(block_id=f"block-{ordinal}", round_ordinal=ordinal)
        blocks.append(item)
    return {"version": 1, "blocks": blocks}


def _oversized_exchange_text() -> dict[str, object]:
    value = json.loads(dump_thinking_blocks_json(_thinking()) or "null")
    value["blocks"][0]["text"] = "x" * (MAX_THINKING_TEXT_BYTES + 1)
    return value


def _private_checkpoint(content: str) -> str:
    payload = {
        "schema_version": 1,
        "checkpoint_revision": 1,
        "provider": "moonshot",
        "protocol": "chat_completions",
        "model": "kimi-k2.6",
        "api_base_url": "https://api.moonshot.ai/v1",
        "state": "complete",
        "rounds": [
            {
                "assistant_content": content,
                "reasoning_blocks": ["PRIVATE-CONTINUATION-CANARY"],
                "calls": [],
            }
        ],
    }
    checkpoint = parse_provider_continuation_json(payload)
    return dump_provider_continuation_json(checkpoint) or ""


def _source_graph(
    tmp_path: Path, chachanotes_template_db: Path
) -> tuple[dict[str, str], str, dict[str, str]]:
    database_path = tmp_path / "thinking-source.db"
    shutil.copyfile(chachanotes_template_db, database_path)
    database = CharactersRAGDB(database_path, "thinking-source")
    conversation_id = database.add_conversation(
        {
            "id": "thinking-conversation",
            "root_id": "thinking-conversation",
            "title": "Thinking graph",
            "thinking_history_policy": "include",
        }
    )
    assert conversation_id == "thinking-conversation"
    ids = {"user": "user-1", "base": "assistant-1", "selected": "assistant-2"}
    assert database.add_message(
        {
            "id": ids["user"],
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Question",
            "timestamp": "2026-08-26T00:00:00+00:00",
        }
    )
    assert database.add_message(
        {
            "id": ids["base"],
            "conversation_id": conversation_id,
            "parent_message_id": ids["user"],
            "sender": "assistant",
            "content": "Base answer",
            "thinking_blocks_json": dump_thinking_blocks_json(_thinking()),
            "timestamp": "2026-08-26T00:00:01+00:00",
        }
    )
    assert database.add_message(
        {
            "id": ids["selected"],
            "conversation_id": conversation_id,
            "parent_message_id": ids["user"],
            "sender": "assistant",
            "content": "Selected answer",
            "thinking_blocks_json": dump_thinking_blocks_json(
                _thinking(proprietary=True)
            ),
            "provider_continuation_json": _private_checkpoint("Selected answer"),
            "timestamp": "2026-08-26T00:00:02+00:00",
        }
    )
    with database.transaction() as connection:
        connection.execute(
            "UPDATE messages SET variant_number = 1, is_selected_variant = 0, "
            "total_variants = 2 WHERE id = ?",
            (ids["base"],),
        )
        connection.execute(
            "UPDATE messages SET variant_of = ?, variant_number = 2, "
            "is_selected_variant = 1, total_variants = 2 WHERE id = ?",
            (ids["base"], ids["selected"]),
        )
    database.set_conversation_active_leaf(conversation_id, ids["selected"])
    database.close_connection()
    return {"ChaChaNotes": str(database_path)}, conversation_id, ids


def _create_export(
    tmp_path: Path, paths: dict[str, str], conversation_id: str
) -> tuple[Path, tuple[bool, str]]:
    archive_path = tmp_path / "thinking.chatbook.zip"
    creator = ChatbookCreator(paths)
    creator.temp_dir = tmp_path
    success, message, _ = creator.create_chatbook(
        name="Thinking",
        description="Thinking graph",
        content_selections={ContentType.CONVERSATION: [conversation_id]},
        output_path=archive_path,
    )
    return archive_path, (success, message)


def _rewrite_export(
    source: Path,
    destination: Path,
    mutate_conversation,
) -> Path:
    with zipfile.ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    conversation_name = next(
        name
        for name in files
        if name.startswith("content/conversations/conversation_")
        and name.endswith(".json")
    )
    conversation = json.loads(files[conversation_name])
    mutate_conversation(conversation)
    files[conversation_name] = json.dumps(conversation, ensure_ascii=False).encode()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return destination


def _import(
    archive_path: Path, destination_path: Path, tmp_path: Path
) -> tuple[bool, str, ImportStatus]:
    status = ImportStatus()
    importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
    importer.temp_dir = tmp_path / f"imports-{destination_path.stem}"
    importer.temp_dir.mkdir()
    success, message = importer.import_chatbook(archive_path, import_status=status)
    return success, message, status


def test_chatbook_v2_round_trip_preserves_every_graph_owner_policy_and_warning(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, ids = _source_graph(
        tmp_path, chachanotes_template_db
    )
    archive_path, result = _create_export(tmp_path, source_paths, conversation_id)
    assert result[0], result[1]

    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        readme = archive.read("README.md").decode("utf-8")
        conversation = json.loads(
            archive.read(
                "content/conversations/conversation_thinking-conversation.json"
            )
        )

    assert conversation["thinking_history_policy"] == "include"
    assert conversation["sensitive_data_warning"] == THINKING_EXPORT_WARNING
    metadata = manifest["content_items"][0]["metadata"]
    assert metadata["contains_model_thinking"] is True
    assert metadata["sensitive_data_warning"] == THINKING_EXPORT_WARNING
    assert THINKING_EXPORT_WARNING in readme
    by_id = {message["id"]: message for message in conversation["messages"]}
    assert by_id[ids["base"]]["_thinking"] == json.loads(
        dump_thinking_blocks_json(_thinking()) or "null"
    )
    assert by_id[ids["selected"]]["_thinking"] == json.loads(
        dump_thinking_blocks_json(_thinking(proprietary=True)) or "null"
    )
    assert "_thinking" not in by_id[ids["user"]]
    assert "Proprietary thinking obfuscated - not available" not in json.dumps(
        conversation
    )

    destination_path = tmp_path / "thinking-destination.db"
    shutil.copyfile(chachanotes_template_db, destination_path)
    success, message, status = _import(archive_path, destination_path, tmp_path)
    assert success, message
    assert status.failed_items == 0
    destination = CharactersRAGDB(destination_path, "thinking-assert")
    try:
        imported = destination.get_conversation_by_name("Thinking graph")[0]
        assert imported["thinking_history_policy"] == "include"
        rows = destination.execute_query(
            "SELECT role, content, thinking_blocks_json FROM messages "
            "WHERE conversation_id = ? ORDER BY timestamp, rowid",
            (imported["id"],),
        ).fetchall()
        assistant_thinking = {
            row["content"]: row["thinking_blocks_json"]
            for row in rows
            if row["role"] == "assistant"
        }
        assert assistant_thinking == {
            "Base answer": dump_thinking_blocks_json(_thinking()),
            "Selected answer": dump_thinking_blocks_json(
                _thinking(proprietary=True)
            ),
        }
    finally:
        destination.close_connection()


def test_chatbook_export_blocks_opaque_future_thinking_with_upgrade_copy(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, ids = _source_graph(
        tmp_path, chachanotes_template_db
    )
    database = CharactersRAGDB(source_paths["ChaChaNotes"], "future-thinking")
    try:
        with database.transaction() as connection:
            connection.execute(
                "UPDATE messages SET thinking_blocks_json = ? WHERE id = ?",
                (
                    json.dumps(
                        {
                            "version": 2,
                            "blocks": [],
                            "secret": "FUTURE-THINKING-CANARY",
                        }
                    ),
                    ids["base"],
                ),
            )
    finally:
        database.close_connection()

    archive_path, (success, message) = _create_export(
        tmp_path, source_paths, conversation_id
    )

    assert success is False
    assert not archive_path.exists()
    assert "Upgrade Chatbook" in message
    assert "FUTURE-THINKING-CANARY" not in message


@pytest.mark.parametrize(
    "role,thinking",
    [
        ("user", json.loads(dump_thinking_blocks_json(_thinking()) or "null")),
        ("assistant", {"version": 2, "blocks": []}),
        (
            "assistant",
            {
                "version": 1,
                "blocks": [
                    {
                        "block_id": "bad",
                        "round_ordinal": 0,
                        "provider": "provider",
                        "model": "model",
                        "protocol": "protocol",
                        "source_format": "source",
                        "status": "complete",
                        "visibility": "proprietary",
                        "text": "PROPRIETARY-TEXT-CANARY",
                    }
                ],
            },
        ),
        ("assistant", _too_many_exchange_blocks()),
        ("assistant", _oversized_exchange_text()),
    ],
)
def test_chatbook_v2_graph_preflight_rejects_invalid_thinking(role, thinking) -> None:
    graph = _linear_graph(role=role)
    graph["messages"][0]["_thinking"] = thinking

    with pytest.raises(ValueError, match="Invalid V2 conversation graph"):
        ChatbookImporter._validate_v2_conversation_graph(graph)


def _linear_graph(*, role: str = "assistant") -> dict[str, object]:
    return {
        "thinking_history_policy": "auto",
        "messages": [
            {
                "id": "message-1",
                "parent_id": None,
                "variant_of": None,
                "order": 0,
                "role": role,
                "content": "Visible answer",
                "deleted": False,
                "variant_number": 1,
                "is_selected_variant": True,
                "total_variants": 1,
            }
        ],
        "active_leaf_message_id": "message-1",
        "selected_path_message_ids": ["message-1"],
    }


@pytest.mark.parametrize("role", ["assistant", "user"])
def test_chatbook_v2_graph_rejects_present_null_thinking(role: str) -> None:
    graph = _linear_graph(role=role)
    graph["messages"][0]["_thinking"] = None

    with pytest.raises(ValueError, match="Invalid V2 conversation graph"):
        ChatbookImporter._validate_v2_conversation_graph(graph)


def test_chatbook_v2_graph_rejects_thinking_on_deleted_message() -> None:
    graph = _linear_graph(role="user")
    graph["messages"].append(
        {
            "id": "message-2",
            "parent_id": "message-1",
            "variant_of": None,
            "order": 1,
            "role": "assistant",
            "content": "Deleted answer",
            "deleted": True,
            "variant_number": 1,
            "is_selected_variant": True,
            "total_variants": 1,
            "_thinking": json.loads(
                dump_thinking_blocks_json(_thinking()) or "null"
            ),
        }
    )

    with pytest.raises(ValueError, match="Invalid V2 conversation graph"):
        ChatbookImporter._validate_v2_conversation_graph(graph)


def test_chatbook_v2_graph_rejects_aggregate_thinking_utf8_bytes(monkeypatch) -> None:
    graph = _linear_graph()
    graph["messages"][0]["_thinking"] = json.loads(
        dump_thinking_blocks_json(_thinking(text="😀")) or "null"
    )
    monkeypatch.setattr(
        importer_module, "_MAX_V2_TOTAL_THINKING_BYTES", 1, raising=False
    )

    with pytest.raises(ValueError, match="Invalid V2 conversation graph"):
        ChatbookImporter._validate_v2_conversation_graph(graph)


@pytest.mark.parametrize("policy", [42, "x" * 10_000])
def test_chatbook_v2_rejects_invalid_policy_before_conversation_mutation(
    tmp_path: Path, chachanotes_template_db: Path, policy
) -> None:
    source_paths, conversation_id, _ = _source_graph(
        tmp_path, chachanotes_template_db
    )
    archive_path, result = _create_export(tmp_path, source_paths, conversation_id)
    assert result[0], result[1]
    broken = _rewrite_export(
        archive_path,
        tmp_path / f"invalid-policy-{type(policy).__name__}.chatbook.zip",
        lambda conversation: conversation.update(thinking_history_policy=policy),
    )
    destination_path = tmp_path / f"invalid-policy-{type(policy).__name__}.db"
    shutil.copyfile(chachanotes_template_db, destination_path)

    success, _, status = _import(broken, destination_path, tmp_path)

    assert success is False
    assert status.successful_items == 0
    destination = CharactersRAGDB(destination_path, "invalid-policy-assert")
    try:
        assert destination.get_conversation_by_name("Thinking graph") == []
    finally:
        destination.close_connection()


@pytest.mark.parametrize("message_id", ["user-1", "assistant-1"])
def test_chatbook_v2_rejects_present_null_thinking_before_conversation_mutation(
    tmp_path: Path,
    chachanotes_template_db: Path,
    message_id: str,
) -> None:
    source_paths, conversation_id, _ = _source_graph(
        tmp_path, chachanotes_template_db
    )
    archive_path, result = _create_export(tmp_path, source_paths, conversation_id)
    assert result[0], result[1]

    def insert_null_thinking(conversation: dict[str, object]) -> None:
        messages = {message["id"]: message for message in conversation["messages"]}
        messages[message_id]["_thinking"] = None

    broken = _rewrite_export(
        archive_path,
        tmp_path / f"null-thinking-{message_id}.chatbook.zip",
        insert_null_thinking,
    )
    destination_path = tmp_path / f"null-thinking-{message_id}.db"
    shutil.copyfile(chachanotes_template_db, destination_path)

    success, _, status = _import(broken, destination_path, tmp_path)

    assert success is False
    assert status.successful_items == 0
    destination = CharactersRAGDB(destination_path, "null-thinking-assert")
    try:
        assert destination.get_conversation_by_name("Thinking graph") == []
    finally:
        destination.close_connection()


def test_chatbook_v2_rejects_deleted_thinking_before_conversation_mutation(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, ids = _source_graph(
        tmp_path, chachanotes_template_db
    )
    archive_path, result = _create_export(tmp_path, source_paths, conversation_id)
    assert result[0], result[1]

    def tombstone_thinking_owner(conversation: dict[str, object]) -> None:
        messages = {message["id"]: message for message in conversation["messages"]}
        assert "_thinking" in messages[ids["base"]]
        messages[ids["base"]]["deleted"] = True

    broken = _rewrite_export(
        archive_path,
        tmp_path / "deleted-thinking.chatbook.zip",
        tombstone_thinking_owner,
    )
    destination_path = tmp_path / "deleted-thinking.db"
    shutil.copyfile(chachanotes_template_db, destination_path)

    success, _, status = _import(broken, destination_path, tmp_path)

    assert success is False
    assert status.successful_items == 0
    destination = CharactersRAGDB(destination_path, "deleted-thinking-assert")
    try:
        assert destination.get_conversation_by_name("Thinking graph") == []
    finally:
        destination.close_connection()


def test_durable_soft_delete_clears_thinking_from_tombstone() -> None:
    database = CharactersRAGDB(":memory:", "deleted-thinking-control")
    try:
        conversation_id = database.add_conversation({"title": "Tombstone control"})
        message_id = database.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "Visible answer",
                "thinking_blocks_json": dump_thinking_blocks_json(_thinking()),
            }
        )
        row = database.execute_query(
            "SELECT version FROM messages WHERE id = ?", (message_id,)
        ).fetchone()

        assert database.soft_delete_message(message_id, row["version"])

        tombstone = database.execute_query(
            "SELECT deleted, thinking_blocks_json FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        assert (tombstone["deleted"], tombstone["thinking_blocks_json"]) == (
            1,
            None,
        )
    finally:
        database.close_connection()


def test_chatbook_v2_unknown_policy_falls_back_to_auto_with_content_free_warning(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, _ = _source_graph(
        tmp_path, chachanotes_template_db
    )
    archive_path, result = _create_export(tmp_path, source_paths, conversation_id)
    assert result[0], result[1]
    rewritten = _rewrite_export(
        archive_path,
        tmp_path / "unknown-policy.chatbook.zip",
        lambda conversation: conversation.update(
            thinking_history_policy="future-policy-canary"
        ),
    )
    destination_path = tmp_path / "unknown-policy.db"
    shutil.copyfile(chachanotes_template_db, destination_path)

    success, message, status = _import(rewritten, destination_path, tmp_path)

    assert success, message
    assert status.warnings == [UNKNOWN_POLICY_WARNING]
    assert "future-policy-canary" not in json.dumps(status.to_dict())
    destination = CharactersRAGDB(destination_path, "unknown-policy-assert")
    try:
        imported = destination.get_conversation_by_name("Thinking graph")[0]
        assert imported["thinking_history_policy"] == "auto"
    finally:
        destination.close_connection()


def test_chatbook_v2_empty_policy_falls_back_with_unknown_policy_warning(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, _ = _source_graph(
        tmp_path, chachanotes_template_db
    )
    archive_path, result = _create_export(tmp_path, source_paths, conversation_id)
    assert result[0], result[1]
    rewritten = _rewrite_export(
        archive_path,
        tmp_path / "empty-policy.chatbook.zip",
        lambda conversation: conversation.update(thinking_history_policy=""),
    )
    destination_path = tmp_path / "empty-policy.db"
    shutil.copyfile(chachanotes_template_db, destination_path)

    success, message, status = _import(rewritten, destination_path, tmp_path)

    assert success, message
    assert status.warnings == [UNKNOWN_POLICY_WARNING]
    destination = CharactersRAGDB(destination_path, "empty-policy-assert")
    try:
        imported = destination.get_conversation_by_name("Thinking graph")[0]
        assert imported["thinking_history_policy"] == "auto"
    finally:
        destination.close_connection()


def test_chatbook_v2_invalid_conversation_does_not_block_valid_neighbor(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    source_paths, conversation_id, _ = _source_graph(
        tmp_path, chachanotes_template_db
    )
    archive_path, result = _create_export(tmp_path, source_paths, conversation_id)
    assert result[0], result[1]
    with zipfile.ZipFile(archive_path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files["manifest.json"])
    original_path = (
        "content/conversations/conversation_thinking-conversation.json"
    )
    invalid = json.loads(files[original_path])
    valid = copy.deepcopy(invalid)
    invalid["selected_path_message_ids"] = ["missing-message-owner"]
    valid["id"] = "valid-neighbor"
    valid["name"] = "Valid neighbor"
    valid_path = "content/conversations/conversation_valid-neighbor.json"
    item = copy.deepcopy(manifest["content_items"][0])
    item.update(id="valid-neighbor", title="Valid neighbor", file_path=valid_path)
    manifest["content_items"].append(item)
    manifest["statistics"]["total_conversations"] = 2
    files["manifest.json"] = json.dumps(manifest).encode()
    files[original_path] = json.dumps(invalid).encode()
    files[valid_path] = json.dumps(valid).encode()
    mixed = tmp_path / "mixed.chatbook.zip"
    with zipfile.ZipFile(mixed, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    destination_path = tmp_path / "mixed.db"
    shutil.copyfile(chachanotes_template_db, destination_path)

    success, message, status = _import(mixed, destination_path, tmp_path)

    assert success, message
    assert status.successful_items == 1
    assert status.failed_items == 1
    destination = CharactersRAGDB(destination_path, "mixed-assert")
    try:
        assert destination.get_conversation_by_name("Thinking graph") == []
        assert len(destination.get_conversation_by_name("Valid neighbor")) == 1
    finally:
        destination.close_connection()
