"""Assistant generation state transport and user-visible export contracts."""

from __future__ import annotations

import io
import json

import pytest

from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_conversation_to_json,
    export_conversation_to_text,
    load_chat_history_from_file_and_save_to_db,
)
from tldw_chatbook.Chat.Chat_Functions import generate_chat_history_content
from tldw_chatbook.Chat.document_generator import DocumentGenerator
from tldw_chatbook.Chat.trajectory_export import (
    build_trajectory_export,
    validate_trajectory_export,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _active_continuation_json() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k2",
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": [],
                    "calls": [
                        {
                            "call_id": "call-1",
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                            "state": "pending",
                        }
                    ],
                }
            ],
        }
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("accepted", "Response accepted on another device; waiting for dispatch."),
        (
            "dispatch_started",
            "Response delivery status is unknown on the source device.",
        ),
        ("complete", "No response was generated."),
    ],
)
def test_text_and_document_context_render_empty_assistant_state_copy(
    tmp_path, state: str, expected: str
) -> None:
    db = CharactersRAGDB(tmp_path / f"text-{state}.db", client_id="export-test")
    try:
        conversation_id = db.add_conversation({"title": "Portable state"})
        assert db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "",
                "assistant_generation_state": state,
            }
        )

        exported = export_conversation_to_text(db, conversation_id)
        context = DocumentGenerator.format_context_for_llm(
            DocumentGenerator.__new__(DocumentGenerator),
            [
                {
                    "role": "assistant",
                    "content": "",
                    "timestamp": "2026-08-22T00:00:00Z",
                    "assistant_generation_state": state,
                }
            ],
        )

        assert exported is not None
        assert expected in exported
        assert expected in context
    finally:
        db.close_connection()


def test_json_and_active_path_import_round_trip_generation_state(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "active-path.db", client_id="export-test")
    try:
        conversation_id = db.add_conversation({"title": "Active path"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "",
                "assistant_generation_state": "accepted",
            }
        )
        assert message_id is not None
        row = db.get_message_by_id(str(message_id))

        ordinary_json = json.loads(
            export_conversation_to_json(db, conversation_id) or "{}"
        )
        active_json, _ = generate_chat_history_content(
            [dict(row)], conversation_id, None, db
        )
        active_payload = json.loads(active_json)

        assert ordinary_json["messages"][0]["assistant_generation_state"] == (
            "accepted"
        )
        assert active_payload["history"][0]["assistant_generation_state"] == (
            "accepted"
        )
        assert "console_dispatch_checkpoints" not in json.dumps(ordinary_json)
        assert "console_dispatch_checkpoints" not in active_json

        imported_id, _ = load_chat_history_from_file_and_save_to_db(
            db, io.BytesIO(active_json.encode("utf-8"))
        )
        assert imported_id is not None
        imported = db.get_messages_for_conversation(imported_id)
        assert imported[0]["assistant_generation_state"] == "accepted"
    finally:
        db.close_connection()


def test_active_path_import_accepts_legacy_missing_generation_state(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "legacy-active-path.db", client_id="import-test")
    try:
        legacy = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Legacy",
            "history": [{"role": "assistant", "content": "legacy answer"}],
        }

        imported_id, _ = load_chat_history_from_file_and_save_to_db(
            db, io.BytesIO(json.dumps(legacy).encode("utf-8"))
        )

        assert imported_id is not None
        imported = db.get_messages_for_conversation(imported_id)
        assert imported[0]["assistant_generation_state"] is None
    finally:
        db.close_connection()


def test_json_export_normalizes_active_continuation_over_stale_state(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "active-authority.db", client_id="export-test")
    try:
        conversation_id = db.add_conversation({"title": "Active authority"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "",
                "provider_continuation_json": _active_continuation_json(),
                "assistant_generation_state": "complete",
            }
        )
        assert message_id is not None
        with db.transaction() as connection:
            connection.execute(
                "UPDATE messages SET assistant_generation_state = 'complete' WHERE id = ?",
                (message_id,),
            )

        ordinary_json = json.loads(
            export_conversation_to_json(db, conversation_id) or "{}"
        )

        assert ordinary_json["messages"][0]["assistant_generation_state"] == (
            "continuation_active"
        )
    finally:
        db.close_connection()


def test_trajectory_projection_round_trips_generation_state_without_checkpoint(
    tmp_path,
) -> None:
    db = CharactersRAGDB(tmp_path / "trajectory.db", client_id="trajectory-test")
    try:
        conversation_id = db.add_conversation({"title": "Trajectory state"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "",
                "assistant_generation_state": "dispatch_started",
            }
        )
        assert message_id is not None

        payload = validate_trajectory_export(
            build_trajectory_export(db, conversation_id)
        )

        assert payload["messages"][0]["assistant_generation_state"] == (
            "dispatch_started"
        )
        assert "console_dispatch_checkpoints" not in json.dumps(payload)
    finally:
        db.close_connection()
