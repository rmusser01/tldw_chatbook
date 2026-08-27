"""Round-trip tests for importable selected-conversation thinking data."""

from __future__ import annotations

import io
import json

import pytest

import tldw_chatbook.Character_Chat.Character_Chat_Lib as character_chat_module
import tldw_chatbook.Chat.Chat_Functions as chat_functions_module
from tldw_chatbook.Chat.Chat_Functions import generate_chat_history_content
from tldw_chatbook.Chat.thinking_blocks import (
    MAX_THINKING_BLOCKS,
    MAX_THINKING_TEXT_BYTES,
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
    preflight_thinking_history_policy,
)
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    load_chat_history_from_file_and_save_to_db,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


THINKING_EXPORT_WARNING = (
    "This conversation export contains model thinking or private provider "
    "continuation. Treat it as sensitive conversation data."
)
UNKNOWN_POLICY_WARNING = "Unknown thinking history policy was reset to Auto."


def test_save_chat_history_passes_the_db_owner_to_content_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_owner = object()
    seen: list[object | None] = []

    def stop_after_recording(
        _history: object,
        _conversation_id: object,
        _media_content: object,
        db_instance: object | None = None,
    ) -> tuple[str, str]:
        seen.append(db_instance)
        raise RuntimeError("stop before filesystem write")

    monkeypatch.setattr(
        chat_functions_module,
        "generate_chat_history_content",
        stop_after_recording,
    )

    assert (
        chat_functions_module.save_chat_history(
            [], "conversation-1", None, db_instance=database_owner
        )
        is None
    )
    assert seen == [database_owner]


def test_selected_json_export_fails_closed_when_conversation_lookup_fails() -> None:
    class FailingConversationDB:
        @staticmethod
        def get_conversation_by_id(_conversation_id: str) -> dict[str, object]:
            raise RuntimeError("PRIVATE-POLICY-LOOKUP-CANARY")

    with pytest.raises(
        ValueError, match="Conversation metadata is unavailable for export"
    ) as caught:
        generate_chat_history_content(
            [{"role": "assistant", "content": "Visible answer"}],
            "conversation-1",
            None,
            db_instance=FailingConversationDB(),  # type: ignore[arg-type]
        )

    assert "PRIVATE-POLICY-LOOKUP-CANARY" not in str(caught.value)


@pytest.mark.parametrize("policy", ["include", "exclude"])
def test_selected_json_export_preserves_db_owned_explicit_policy(policy: str) -> None:
    class ConversationDB:
        @staticmethod
        def get_conversation_by_id(_conversation_id: str) -> dict[str, object]:
            return {"title": "Policy owner", "thinking_history_policy": policy}

    content, _ = generate_chat_history_content(
        [{"role": "assistant", "content": "Visible answer"}],
        "conversation-1",
        None,
        db_instance=ConversationDB(),  # type: ignore[arg-type]
    )

    assert json.loads(content)["thinking_history_policy"] == policy


def test_empty_policy_is_unknown_while_missing_or_null_policy_is_silent_auto() -> None:
    assert preflight_thinking_history_policy(None) == ("auto", None)
    assert preflight_thinking_history_policy("") == (
        "auto",
        UNKNOWN_POLICY_WARNING,
    )


def _thinking(text: str = "DISPLAYABLE-THINKING-CANARY") -> ThinkingEnvelope:
    return ThinkingEnvelope(
        blocks=(
            DisplayableThinkingBlock(
                block_id="displayable-1",
                round_ordinal=0,
                provider="llama.cpp",
                model="local-reasoner",
                protocol="chat_completions",
                source_format="reasoning_content",
                status="complete",
                text=text,
            ),
            ProprietaryThinkingBlock(
                block_id="proprietary-1",
                round_ordinal=1,
                provider="hosted",
                model="private-reasoner",
                protocol="responses",
                source_format="provider_evidence",
                status="complete",
            ),
        )
    )


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


def _export_payload(
    history: list[dict[str, object]],
    *,
    policy: object = "include",
) -> dict[str, object]:
    class ConversationDB:
        @staticmethod
        def get_conversation_by_id(_conversation_id: str) -> dict[str, object]:
            return {
                "title": "Thinking exchange",
                "thinking_history_policy": policy,
            }

    content, _ = generate_chat_history_content(
        history,
        "conversation-1",
        None,
        db_instance=ConversationDB(),  # type: ignore[arg-type]
    )
    return json.loads(content)


def _import_payload(
    database: CharactersRAGDB, payload: dict[str, object]
) -> tuple[str | None, int | None]:
    return load_chat_history_from_file_and_save_to_db(
        database,
        io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
    )


def test_selected_json_exports_structured_thinking_policy_and_shared_warning() -> None:
    canonical = dump_thinking_blocks_json(_thinking())

    payload = _export_payload(
        [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "Visible answer",
                "thinking_blocks_json": canonical,
            },
        ]
    )

    assert payload["thinking_history_policy"] == "include"
    assert payload["sensitive_data_warning"] == THINKING_EXPORT_WARNING
    assistant = payload["history"][1]
    assert assistant["thinking_blocks"] == json.loads(canonical or "null")
    assert assistant["thinking_blocks"]["blocks"][0] == {
        "block_id": "displayable-1",
        "round_ordinal": 0,
        "provider": "llama.cpp",
        "model": "local-reasoner",
        "protocol": "chat_completions",
        "source_format": "reasoning_content",
        "status": "complete",
        "visibility": "displayable",
        "text": "DISPLAYABLE-THINKING-CANARY",
    }
    assert "text" not in assistant["thinking_blocks"]["blocks"][1]
    assert "Proprietary thinking obfuscated - not available" not in json.dumps(payload)


def test_selected_json_without_evidence_exports_auto_without_sensitive_warning() -> None:
    payload = _export_payload(
        [{"role": "assistant", "content": "Visible answer"}], policy=None
    )

    assert payload["thinking_history_policy"] == "auto"
    assert "sensitive_data_warning" not in payload
    assert "thinking_blocks" not in payload["history"][0]


def test_selected_json_rejects_opaque_future_thinking_without_copying_content() -> None:
    future = json.dumps(
        {"version": 2, "blocks": [], "private_future": "FUTURE-THINKING-CANARY"}
    )

    with pytest.raises(
        ValueError,
        match="Upgrade Chatbook before exporting this conversation's thinking data",
    ) as caught:
        _export_payload(
            [
                {
                    "role": "assistant",
                    "content": "Visible answer",
                    "thinking_blocks_json": future,
                }
            ]
        )

    assert "FUTURE-THINKING-CANARY" not in str(caught.value)


def test_selected_json_round_trip_restores_policy_and_canonical_assistant_envelope(
    tmp_path,
) -> None:
    source = CharactersRAGDB(tmp_path / "source.db", "thinking-source")
    target = CharactersRAGDB(tmp_path / "target.db", "thinking-target")
    try:
        conversation_id = source.add_conversation(
            {
                "title": "Thinking source",
                "thinking_history_policy": "include",
            }
        )
        assert conversation_id
        assert source.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "Visible answer",
                "thinking_blocks_json": dump_thinking_blocks_json(_thinking()),
            }
        )
        rows = source.get_messages_for_conversation(conversation_id)
        content, _ = generate_chat_history_content(
            rows,
            str(conversation_id),
            None,
            db_instance=source,
        )

        imported_id, _ = load_chat_history_from_file_and_save_to_db(
            target, io.BytesIO(content.encode("utf-8"))
        )

        assert imported_id is not None
        assert (
            target.get_conversation_by_id(imported_id)["thinking_history_policy"]
            == "include"
        )
        restored = target.get_messages_for_conversation(imported_id)
        assert len(restored) == 1
        assert restored[0]["content"] == "Visible answer"
        assert restored[0]["thinking_blocks_json"] == dump_thinking_blocks_json(
            _thinking()
        )
    finally:
        source.close_connection()
        target.close_connection()


def test_selected_json_unknown_bounded_policy_falls_back_with_content_free_warning(
    monkeypatch,
) -> None:
    warnings: list[str] = []

    class WarningLogger:
        @staticmethod
        def warning(message, *args):
            warnings.append(message.format(*args))

        @staticmethod
        def info(*_args, **_kwargs):
            return None

        @staticmethod
        def error(*_args, **_kwargs):
            return None

    monkeypatch.setattr(character_chat_module, "logger", WarningLogger())
    database = CharactersRAGDB(":memory:", "thinking-policy-fallback")
    try:
        payload = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Unknown policy",
            "thinking_history_policy": "future-policy",
            "history": [{"role": "assistant", "content": "Visible answer"}],
        }

        conversation_id, _ = _import_payload(database, payload)

        assert conversation_id is not None
        assert (
            database.get_conversation_by_id(conversation_id)[
                "thinking_history_policy"
            ]
            == "auto"
        )
        assert warnings == [UNKNOWN_POLICY_WARNING]
        assert "future-policy" not in "\n".join(warnings)
    finally:
        database.close_connection()


def test_selected_json_empty_policy_falls_back_with_unknown_policy_warning(
    monkeypatch,
) -> None:
    warnings: list[str] = []

    class WarningLogger:
        @staticmethod
        def warning(message, *args):
            warnings.append(message.format(*args))

        @staticmethod
        def info(*_args, **_kwargs):
            return None

        @staticmethod
        def error(*_args, **_kwargs):
            return None

    monkeypatch.setattr(character_chat_module, "logger", WarningLogger())
    database = CharactersRAGDB(":memory:", "thinking-empty-policy")
    try:
        payload = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Empty policy",
            "thinking_history_policy": "",
            "history": [{"role": "assistant", "content": "Visible answer"}],
        }

        conversation_id, _ = _import_payload(database, payload)

        assert conversation_id is not None
        assert (
            database.get_conversation_by_id(conversation_id)[
                "thinking_history_policy"
            ]
            == "auto"
        )
        assert warnings == [UNKNOWN_POLICY_WARNING]
    finally:
        database.close_connection()


@pytest.mark.parametrize("policy_fields", [{}, {"thinking_history_policy": None}])
def test_selected_json_missing_or_null_policy_imports_as_silent_auto(
    monkeypatch,
    policy_fields: dict[str, object],
) -> None:
    warnings: list[str] = []

    class WarningLogger:
        @staticmethod
        def warning(message, *args):
            warnings.append(message.format(*args))

        @staticmethod
        def info(*_args, **_kwargs):
            return None

        @staticmethod
        def error(*_args, **_kwargs):
            return None

    monkeypatch.setattr(character_chat_module, "logger", WarningLogger())
    database = CharactersRAGDB(":memory:", "thinking-silent-auto-policy")
    try:
        payload = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Silent Auto policy",
            "history": [{"role": "assistant", "content": "Visible answer"}],
            **policy_fields,
        }

        conversation_id, _ = _import_payload(database, payload)

        assert conversation_id is not None
        assert (
            database.get_conversation_by_id(conversation_id)[
                "thinking_history_policy"
            ]
            == "auto"
        )
        assert warnings == []
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    "policy",
    [42, "x" * 10_000],
)
def test_selected_json_rejects_invalid_policy_before_database_mutation(policy) -> None:
    database = CharactersRAGDB(":memory:", "thinking-policy-reject")
    try:
        payload = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Invalid policy",
            "thinking_history_policy": policy,
            "history": [{"role": "assistant", "content": "Visible answer"}],
        }

        assert _import_payload(database, payload) == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    "role,thinking",
    [
        ("assistant", None),
        ("user", None),
        ("user", json.loads(dump_thinking_blocks_json(_thinking()) or "null")),
        (
            "assistant",
            {"version": 2, "blocks": [], "secret": "INVALID-THINKING-CANARY"},
        ),
        (
            "assistant",
            {
                "version": 1,
                "blocks": [
                    {
                        "block_id": "bad",
                        "round_ordinal": 0,
                        "provider": "",
                        "model": "model",
                        "protocol": "protocol",
                        "source_format": "source",
                        "status": "complete",
                        "visibility": "displayable",
                        "text": "INVALID-THINKING-CANARY",
                    }
                ],
            },
        ),
        ("assistant", _too_many_exchange_blocks()),
        ("assistant", _oversized_exchange_text()),
    ],
)
def test_selected_json_rejects_invalid_thinking_before_database_mutation(
    role, thinking
) -> None:
    database = CharactersRAGDB(":memory:", "thinking-envelope-reject")
    try:
        payload = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Invalid thinking",
            "thinking_history_policy": "auto",
            "history": [
                {"role": role, "content": "Visible content", "thinking_blocks": thinking}
            ],
        }

        assert _import_payload(database, payload) == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()


def test_selected_json_rejects_aggregate_thinking_bytes_before_database_mutation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        character_chat_module, "_MAX_EXPORTED_HISTORY_THINKING_BYTES", 1, raising=False
    )
    thinking = json.loads(dump_thinking_blocks_json(_thinking("😀")) or "null")
    database = CharactersRAGDB(":memory:", "thinking-aggregate-reject")
    try:
        payload = {
            "format": "tldw_chat_history",
            "format_version": 1,
            "conversation_name": "Aggregate thinking",
            "thinking_history_policy": "auto",
            "history": [
                {
                    "role": "assistant",
                    "content": "Visible answer",
                    "thinking_blocks": thinking,
                }
            ],
        }

        assert _import_payload(database, payload) == (None, None)
        assert (
            database.execute_query("SELECT COUNT(*) FROM conversations").fetchone()[0]
            == 0
        )
    finally:
        database.close_connection()
