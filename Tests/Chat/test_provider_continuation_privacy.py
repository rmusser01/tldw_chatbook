"""Explicit JSON inclusion and ordinary-surface continuation privacy."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from tldw_chatbook.Chat.Chat_Functions import generate_chat_history_content
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Chat.document_generator import DocumentGenerator
from tldw_chatbook.Chat.provider_continuation import parse_provider_continuation_json
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
    assert payload["private_data_warning"] == (
        "This JSON contains private provider continuation data."
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
