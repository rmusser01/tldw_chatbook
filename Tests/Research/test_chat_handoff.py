"""Chat handoff insertion (task-16481)."""

import json

from tldw_chatbook.Research_Interop.chat_handoff import (
    insert_research_completion_message,
)


class FakeDB:
    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    def add_message(self, msg_data):
        if self.fail:
            raise RuntimeError("db down")
        self.messages.append(msg_data)
        return f"msg-{len(self.messages)}"


_PAYLOAD = {
    "run_id": "run-1",
    "question": "What is RAG?",
    "chat_handoff": {"conversation_id": "conv-42", "origin": "console"},
    "report_markdown": "Answer citing [1].\n\nSources:\n[1] T — https://t.example/",
    "bundle": {"source_count": 2},
    "verification_summary": {
        "confidence": 0.9,
        "gate": {"relevant": 2, "raw": 5, "fallback": False},
    },
}


def test_inserts_assistant_message_with_report_and_metadata():
    db = FakeDB()

    message_id = insert_research_completion_message(db, _PAYLOAD)

    assert message_id == "msg-1"
    msg = db.messages[0]
    assert msg["conversation_id"] == "conv-42"
    assert msg["sender"] == "assistant"
    assert "Deep research completed for: What is RAG?" in msg["content"]
    assert "Answer citing [1]." in msg["content"]
    block = json.loads(msg["metadata_json"])["deep_research_completion"]
    assert block["run_id"] == "run-1"
    assert block["source_count"] == 2
    assert block["confidence"] == 0.9


def test_missing_conversation_id_returns_none_without_raising():
    payload = dict(_PAYLOAD)
    payload["chat_handoff"] = {}
    db = FakeDB()

    assert insert_research_completion_message(db, payload) is None
    assert db.messages == []


def test_db_failure_returns_none_without_raising():
    db = FakeDB(fail=True)

    assert insert_research_completion_message(db, _PAYLOAD) is None


def test_research_command_registered_in_default_console_registry():
    from tldw_chatbook.Chat.console_command_grammar import (
        RESEARCH_COMMAND_HANDLER_ID,
        RESEARCH_COMMAND_NAME,
        default_console_registry,
    )

    parse = default_console_registry().parse("/research what is RAG?")

    assert parse.kind == "command"
    assert parse.name == RESEARCH_COMMAND_NAME == "research"
    assert parse.args == "what is RAG?"
    assert RESEARCH_COMMAND_HANDLER_ID == "research"
