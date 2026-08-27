"""Explicit JSON inclusion and ordinary-surface continuation privacy."""

from __future__ import annotations

import json
import logging
import io
import shutil
from pathlib import Path

import pytest

from tldw_chatbook.Chat.Chat_Functions import generate_chat_history_content
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Chat.document_generator import DocumentGenerator
from tldw_chatbook.Chat.provider_continuation import parse_provider_continuation_json
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    load_chat_history_from_file_and_save_to_db,
)
import tldw_chatbook.Character_Chat.Character_Chat_Lib as character_chat_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _checkpoint() -> dict:
    return {
        "schema_version": 1,
        "checkpoint_revision": 1,
        "provider": "deepseek",
        "protocol": "responses",
        "model": "deepseek-v4-flash",
        "api_base_url": "https://api.deepseek.com",
        "state": "active",
        "rounds": [
            {
                "assistant_content": "visible answer",
                "reasoning_blocks": ["PRIVATE-JSON-REASONING-CANARY"],
                "calls": [
                    {
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": '{"value":"PRIVATE-JSON-ARGUMENT-CANARY"}',
                        "state": "pending",
                    }
                ],
            }
        ],
    }


def test_conversation_json_uses_explicit_private_projection() -> None:
    checkpoint = _checkpoint()
    history = [
        {
            "id": "assistant-1",
            "parent_message_id": "user-1",
            "role": "assistant",
            "content": "visible answer",
            "timestamp": "2026-08-12T00:00:00+00:00",
            "variant_of": None,
            "variant_number": 1,
            "is_selected_variant": True,
            "total_variants": 1,
            "provider_continuation_json": json.dumps(checkpoint),
            "metadata_json": "PRIVATE-METADATA-CANARY",
            "api_key": "PRIVATE-CREDENTIAL-CANARY",
            "streaming": True,
            "widget": "PRIVATE-UI-CANARY",
        }
    ]

    payload_json, _ = generate_chat_history_content(history, "conversation-1", None)

    payload = json.loads(payload_json)
    assert payload["format"] == "tldw_chat_history"
    assert payload["format_version"] == 1
    assert payload["private_data_warning"] == (
        "This JSON contains private provider continuation data."
    )
    assert payload["sensitive_data_warning"] == (
        "This conversation export contains model thinking or private provider "
        "continuation. Treat it as sensitive conversation data."
    )
    assert payload["history"] == [
        {
            "id": "assistant-1",
            "parent_id": "user-1",
            "role": "assistant",
            "content": "visible answer",
            "timestamp": "2026-08-12T00:00:00+00:00",
            "variant_of": None,
            "variant_number": 1,
            "is_selected_variant": True,
            "total_variants": 1,
            "_private": {"provider_continuation": checkpoint},
        }
    ]
    serialized = json.dumps(payload)
    for canary in (
        "PRIVATE-METADATA-CANARY",
        "PRIVATE-CREDENTIAL-CANARY",
        "PRIVATE-UI-CANARY",
    ):
        assert canary not in serialized


def test_exported_json_import_restores_exact_assistant_owner_without_running() -> None:
    database = CharactersRAGDB(":memory:", "json-continuation-import")
    try:
        history = [
            {"role": "user", "content": "Use a tool"},
            {
                "role": "assistant",
                "content": "visible answer",
                "provider_continuation_json": json.dumps(_checkpoint()),
            },
        ]
        payload_json, _ = generate_chat_history_content(history, None, None)

        conversation_id, character_id = load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(payload_json.encode())
        )

        assert conversation_id is not None
        assert character_id is None
        rows = database.get_messages_for_conversation(conversation_id)
        assert [(row["role"], row["content"]) for row in rows] == [
            ("user", "Use a tool"),
            ("assistant", "visible answer"),
        ]
        assert rows[0]["provider_continuation_json"] is None
        assert (
            parse_provider_continuation_json(
                rows[1]["provider_continuation_json"]
            ).state
            == "active"
        )

        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.restore_persisted_session(
            title="Imported JSON",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
            active_leaf_persisted_id=str(rows[1]["id"]),
        )
        interrupted = store.interrupted_provider_continuation_message(session.id)
        assert interrupted is not None
        assert interrupted.content == "visible answer"
        assert interrupted.provider_continuation is not None
        assert interrupted.provider_continuation.rounds[-1].calls[-1].state == "pending"
    finally:
        database.close_connection()


def _kimi_family_checkpoint(
    content: str, *, model: str = "kimi-k2.6", post_tool_only: bool = False
) -> dict:
    rounds: list[dict]
    if post_tool_only:
        rounds = [
            {
                "assistant_content": "",
                "reasoning_blocks": ["PRIVATE-FAMILY-REASONING"],
                "calls": [
                    {
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                        "state": "completed",
                        "result": "4",
                    }
                ],
            }
        ]
    else:
        rounds = [
            {
                "assistant_content": content,
                "reasoning_blocks": ["PRIVATE-FAMILY-REASONING"],
                "calls": [],
            }
        ]
    return {
        "schema_version": 1,
        "checkpoint_revision": 1,
        "provider": "moonshot",
        "protocol": "chat_completions",
        "model": model,
        "api_base_url": "https://api.moonshot.ai/v1",
        "state": "complete",
        "rounds": rounds,
    }


def test_exported_json_import_family_owner_rule_covers_versioned_kimi() -> None:
    """TASK-19170: the import-side exact-owner rule for complete
    preserved-thinking checkpoints follows the versioned-kimi family; a
    matching kimi-k2.6 checkpoint restores, a mismatched one is dropped."""
    database = CharactersRAGDB(":memory:", "kimi-family-json-import")
    try:
        history = [
            {"role": "user", "content": "Think"},
            {
                "role": "assistant",
                "content": "visible answer",
                "provider_continuation_json": json.dumps(
                    _kimi_family_checkpoint("visible answer")
                ),
            },
            {
                "role": "assistant",
                "content": "other visible answer",
                "provider_continuation_json": json.dumps(
                    _kimi_family_checkpoint("does not match owner")
                ),
            },
            # Pre-19170 durable shape: complete, ends with a tool round --
            # exempt from the exact-owner rule (its final round's content is
            # never the visible answer).
            {
                "role": "assistant",
                "content": "tool visible answer",
                "provider_continuation_json": json.dumps(
                    _kimi_family_checkpoint("ignored", post_tool_only=True)
                ),
            },
        ]
        payload_json, _ = generate_chat_history_content(history, None, None)

        conversation_id, _character_id = load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(payload_json.encode())
        )

        rows = database.get_messages_for_conversation(conversation_id)
        assert [row["content"] for row in rows] == [
            "Think",
            "visible answer",
            "other visible answer",
            "tool visible answer",
        ]
        restored = parse_provider_continuation_json(
            rows[1]["provider_continuation_json"]
        )
        assert restored.model == "kimi-k2.6"
        assert restored.rounds[-1].reasoning_blocks == ("PRIVATE-FAMILY-REASONING",)
        assert rows[2]["provider_continuation_json"] is None
        tool_shape = parse_provider_continuation_json(
            rows[3]["provider_continuation_json"]
        )
        assert tool_shape.rounds[-1].calls[0].state == "completed"
    finally:
        database.close_connection()


def test_exported_json_import_drops_invalid_private_with_safe_warning(
    monkeypatch,
) -> None:
    database = CharactersRAGDB(":memory:", "invalid-json-continuation")
    try:
        warnings = []

        class WarningLogger:
            @staticmethod
            def warning(message, *args):
                warnings.append(message.format(*args))

        monkeypatch.setattr(character_chat_module, "logger", WarningLogger())
        checkpoint = _checkpoint()
        checkpoint["schema_version"] = 99
        payload = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Imported",
            "history": [
                {
                    "role": "assistant",
                    "content": "visible answer",
                    "_private": {"provider_continuation": checkpoint},
                }
            ],
        }

        conversation_id, _ = load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(payload).encode())
        )

        assert conversation_id is not None
        rows = database.get_messages_for_conversation(conversation_id)
        assert len(rows) == 1
        assert rows[0]["content"] == "visible answer"
        assert rows[0]["provider_continuation_json"] is None
        diagnostic = "\n".join(warnings)
        assert "Exact tool continuation was discarded for message 1." in diagnostic
        assert "PRIVATE-JSON-REASONING-CANARY" not in diagnostic
        assert "PRIVATE-JSON-ARGUMENT-CANARY" not in diagnostic
    finally:
        database.close_connection()


def test_exported_json_import_bounds_messages_before_database_writes(
    monkeypatch,
) -> None:
    database = CharactersRAGDB(":memory:", "bounded-json-import")
    try:
        monkeypatch.setattr(
            "tldw_chatbook.Character_Chat.Character_Chat_Lib._MAX_EXPORTED_HISTORY_MESSAGES",
            1,
        )
        payload = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Bounded",
            "history": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        }

        result = load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(payload).encode())
        )

        assert result == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()


def test_malformed_json_export_input_does_not_log_private_payload(caplog) -> None:
    malformed = {
        "provider_continuation_json": json.dumps(_checkpoint()),
        "api_key": "PRIVATE-LOG-CREDENTIAL-CANARY",
    }

    with caplog.at_level(logging.WARNING):
        payload_json, _ = generate_chat_history_content(
            [malformed], "conversation-1", None
        )

    assert json.loads(payload_json)["history"] == []
    log_text = caplog.text
    assert "Unexpected item format" in log_text
    for canary in (
        "PRIVATE-JSON-REASONING-CANARY",
        "PRIVATE-JSON-ARGUMENT-CANARY",
        "PRIVATE-LOG-CREDENTIAL-CANARY",
    ):
        assert canary not in log_text


def test_exported_json_requires_exact_marker_without_stealing_legacy_shape() -> None:
    database = CharactersRAGDB(":memory:", "json-format-discriminator")
    try:
        collision = {
            "conversation_name": "Must stay legacy",
            "char_name": "Legacy owner",
            "history": [{"role": "assistant", "content": "visible answer"}],
        }
        unknown = {
            **collision,
            "format": "tldw_chat_history",
            "format_version": 999,
            "history": [
                {
                    "role": "assistant",
                    "content": "visible answer",
                    "_private": {"provider_continuation": _checkpoint()},
                }
            ],
        }

        assert load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(collision).encode())
        ) == (None, None)
        assert load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(unknown).encode())
        ) == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()


def test_unmarked_legacy_collision_never_logs_private_malformed_pair() -> None:
    payload = {
        "conversation_name": "Must stay legacy",
        "char_name": "Legacy owner",
        "history": [
            {
                "role": "assistant",
                "content": "visible answer",
                "_private": {"provider_continuation": "PRIVATE-LEGACY-PAIR-CANARY"},
            }
        ],
    }
    diagnostics = io.StringIO()
    sink_id = character_chat_module.logger.add(diagnostics, format="{message}")
    database = CharactersRAGDB(":memory:", "json-legacy-log-privacy")
    try:
        result = load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(payload).encode())
        )

        assert result == (None, None)
        diagnostic = diagnostics.getvalue()
        assert "Skipping malformed message pair 0 (category=dict)." in diagnostic
        assert "PRIVATE-LEGACY-PAIR-CANARY" not in diagnostic
    finally:
        character_chat_module.logger.remove(sink_id)
        database.close_connection()


@pytest.mark.parametrize("format_version", [True, 1.0])
def test_exported_json_rejects_non_integer_format_version(format_version) -> None:
    payload = {
        "format": "tldw_chat_history",
        "format_version": format_version,
        "conversation_name": "Wrong version type",
        "history": [{"role": "assistant", "content": "visible answer"}],
    }
    database = CharactersRAGDB(":memory:", "json-format-version-type")
    try:
        assert load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(payload).encode())
        ) == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()


def test_exported_json_private_bound_counts_utf8_not_ascii_escapes(monkeypatch) -> None:
    checkpoint = _checkpoint()
    checkpoint["rounds"][0]["reasoning_blocks"] = ["😀"]
    private = {"provider_continuation": checkpoint}
    monkeypatch.setattr(
        character_chat_module,
        "_MAX_EXPORTED_HISTORY_PRIVATE_BYTES",
        len(json.dumps(private, separators=(",", ":"), ensure_ascii=False).encode()),
    )
    payload = {
        "format": "tldw_chat_history",
        "format_version": 1,
        "conversation_name": "Unicode private",
        "history": [
            {
                "role": "assistant",
                "content": "visible answer",
                "_private": private,
            }
        ],
    }
    database = CharactersRAGDB(":memory:", "json-unicode-private-bound")
    try:
        conversation_id, _ = load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(payload).encode())
        )
        assert conversation_id is not None
        row = database.get_messages_for_conversation(conversation_id)[0]
        assert row["content"] == "visible answer"
        assert row["provider_continuation_json"] is not None
    finally:
        database.close_connection()


@pytest.mark.parametrize("private_kind", ["oversize", "deep"])
def test_invalid_private_resource_drops_without_rejecting_visible_json(
    monkeypatch, private_kind
) -> None:
    checkpoint = _checkpoint()
    if private_kind == "oversize":
        monkeypatch.setattr(
            character_chat_module, "_MAX_EXPORTED_HISTORY_PRIVATE_BYTES", 1
        )
        private_value = checkpoint
    else:
        private_value = "PRIVATE-DEEP-CANARY"
        for _ in range(40):
            private_value = [private_value]
    payload = {
        "format": "tldw_chat_history",
        "format_version": 1,
        "conversation_name": "Private resource",
        "history": [
            {
                "role": "assistant",
                "content": "visible answer",
                "_private": {"provider_continuation": private_value},
            }
        ],
    }
    database = CharactersRAGDB(":memory:", "json-private-resource")
    try:
        conversation_id, _ = load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(payload).encode())
        )
        assert conversation_id is not None
        row = database.get_messages_for_conversation(conversation_id)[0]
        assert row["content"] == "visible answer"
        assert row["provider_continuation_json"] is None
    finally:
        database.close_connection()


def test_exported_json_rejects_oversize_path_before_open(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "history.json"
    source.write_bytes(b"{}")
    monkeypatch.setattr(character_chat_module, "_MAX_EXPORTED_HISTORY_FILE_BYTES", 1)
    opened = False

    def forbidden_open(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("oversize input must not be opened")

    monkeypatch.setattr(character_chat_module, "open", forbidden_open, raising=False)
    database = CharactersRAGDB(":memory:", "json-path-bound")
    try:
        assert load_chat_history_from_file_and_save_to_db(
            database, str(source), base_directory=str(tmp_path)
        ) == (None, None)
        assert opened is False
    finally:
        database.close_connection()


def test_exported_json_rejects_aggregate_content_before_database_writes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        character_chat_module, "_MAX_EXPORTED_HISTORY_TOTAL_CONTENT_CHARS", 3
    )
    payload = {
        "format": "tldw_chat_history",
        "format_version": 1,
        "conversation_name": "Bounded",
        "history": [
            {"role": "user", "content": "ab"},
            {"role": "assistant", "content": "cd"},
        ],
    }
    database = CharactersRAGDB(":memory:", "json-content-bound")
    try:
        assert load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(payload).encode())
        ) == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    ("constant", "value"),
    [
        ("_MAX_EXPORTED_HISTORY_TOTAL_ID_CHARS", 0),
        ("_MAX_EXPORTED_HISTORY_JSON_DEPTH", 2),
    ],
)
def test_exported_json_rejects_id_private_and_depth_bounds(
    monkeypatch, constant, value
) -> None:
    monkeypatch.setattr(character_chat_module, constant, value)
    payload = {
        "format": "tldw_chat_history",
        "format_version": 1,
        "conversation_name": "Bounded",
        "history": [
            {
                "id": "message-1",
                "role": "assistant",
                "content": "visible answer",
                "_private": {"provider_continuation": _checkpoint()},
            }
        ],
    }
    database = CharactersRAGDB(":memory:", "json-resource-bound")
    try:
        assert load_chat_history_from_file_and_save_to_db(
            database, io.BytesIO(json.dumps(payload).encode())
        ) == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()


def test_exported_json_bounded_reader_checks_actual_bytes(monkeypatch) -> None:
    monkeypatch.setattr(character_chat_module, "_MAX_EXPORTED_HISTORY_FILE_BYTES", 4)

    class DishonestReader(io.BytesIO):
        def read(self, _size=-1):
            return b"12345"

    database = CharactersRAGDB(":memory:", "json-actual-byte-bound")
    try:
        assert load_chat_history_from_file_and_save_to_db(
            database, DishonestReader()
        ) == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()


def test_import_failure_log_has_safe_operation_context_without_payload() -> None:
    diagnostics = io.StringIO()
    sink_id = character_chat_module.logger.add(diagnostics, format="{message}")
    database = CharactersRAGDB(":memory:", "json-import-log-context")
    try:
        assert load_chat_history_from_file_and_save_to_db(
            database,
            io.BytesIO(b'{"api_key":"PRIVATE-IMPORT-LOG-CANARY"'),
        ) == (None, None)
    finally:
        character_chat_module.logger.remove(sink_id)
        database.close_connection()

    diagnostic = diagnostics.getvalue()
    assert "operation=chat_history_import" in diagnostic
    assert "source=stream" in diagnostic
    assert "category=JSONDecodeError" in diagnostic
    assert "PRIVATE-IMPORT-LOG-CANARY" not in diagnostic


def test_search_and_fts_never_project_private_continuation(
    tmp_path: Path, chachanotes_template_db: Path
) -> None:
    db_path = tmp_path / "privacy.db"
    shutil.copyfile(chachanotes_template_db, db_path)
    db = CharactersRAGDB(str(db_path), "privacy")
    conversation_id = db.add_conversation({"title": "Safe title"})
    assert conversation_id
    assert db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "visible searchable answer",
            "provider_continuation_json": json.dumps(_checkpoint()),
        }
    )

    visible_hits = db.search_messages_by_content("visible searchable")

    assert len(visible_hits) == 1
    serialized = json.dumps(visible_hits, default=str)
    assert "provider_continuation" not in serialized
    assert "PRIVATE-JSON-REASONING-CANARY" not in serialized
    assert "PRIVATE-JSON-ARGUMENT-CANARY" not in serialized
    assert db.search_messages_by_content("REASONING") == []


def test_render_repr_clipboard_and_document_context_use_visible_content_only() -> None:
    checkpoint = parse_provider_continuation_json(_checkpoint())
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="visible answer",
        provider_continuation=checkpoint,
    )

    action = ConsoleMessageActionService().dispatch("copy", message)

    assert action.clipboard_text == "visible answer"
    assert "PRIVATE-JSON-REASONING-CANARY" not in repr(message)
    assert "PRIVATE-JSON-ARGUMENT-CANARY" not in repr(message)

    class FakeDB:
        @staticmethod
        def get_messages_for_conversation(*_args, **_kwargs):
            return [
                {
                    "role": "assistant",
                    "content": "visible answer",
                    "timestamp": "2026-08-12T00:00:00+00:00",
                    "provider_continuation_json": json.dumps(_checkpoint()),
                }
            ]

    generator = DocumentGenerator.__new__(DocumentGenerator)
    generator.db = FakeDB()
    context_messages = generator.get_conversation_context("conversation-1")
    context = generator.format_context_for_llm(context_messages)
    serialized = json.dumps(context_messages) + context
    assert "visible answer" in serialized
    assert "provider_continuation" not in serialized
    assert "PRIVATE-JSON-REASONING-CANARY" not in serialized
    assert "PRIVATE-JSON-ARGUMENT-CANARY" not in serialized
