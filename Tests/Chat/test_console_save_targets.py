"""Pure tests for Console Save-as title and payload derivation."""

from datetime import datetime, timezone

from tldw_chatbook.Chat.console_save_targets import (
    CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS,
    CONSOLE_SAVE_TITLE_MAX_CHARS,
    console_chatbook_artifact_payload,
    console_message_preview,
    derive_console_save_title,
)

FIXED_NOW = datetime(2026, 7, 11, 12, 30, tzinfo=timezone.utc)


def test_derive_console_save_title_includes_conversation_title_and_date():
    title = derive_console_save_title("Draft a release note", now=FIXED_NOW)

    assert title == "Console message — Draft a release note (2026-07-11)"


def test_derive_console_save_title_blank_conversation_falls_back_to_prefix():
    assert derive_console_save_title("", now=FIXED_NOW) == "Console message (2026-07-11)"
    assert derive_console_save_title("   ", now=FIXED_NOW) == "Console message (2026-07-11)"


def test_derive_console_save_title_weaves_role_label_into_prefix():
    title = derive_console_save_title("Chat 1", role_label="Assistant", now=FIXED_NOW)

    assert title == "Console assistant message — Chat 1 (2026-07-11)"


def test_derive_console_save_title_caps_long_conversation_titles():
    long_conversation = "How to configure llama.cpp streaming endpoints " * 5

    title = derive_console_save_title(long_conversation, now=FIXED_NOW)

    assert len(title) <= CONSOLE_SAVE_TITLE_MAX_CHARS
    assert title.startswith("Console message — How to configure")
    assert title.endswith("(2026-07-11)")
    assert "…" in title


def test_derive_console_save_title_collapses_multiline_titles():
    title = derive_console_save_title("Line one\n  Line two", now=FIXED_NOW)

    assert title == "Console message — Line one Line two (2026-07-11)"


def test_console_message_preview_bounds_and_flattens():
    preview = console_message_preview("alpha\nbeta   gamma", max_length=280)
    assert preview == "alpha beta gamma"

    bounded = console_message_preview("x" * 500, max_length=280)
    assert len(bounded) <= 280
    assert bounded.endswith("...")


def test_console_chatbook_artifact_payload_marks_console_saved_artifact():
    payload = console_chatbook_artifact_payload(
        title="Console message — Chat 1 (2026-07-11)",
        message_text="answer body",
        message_role="Assistant",
        conversation_id="conv-9",
        message_id="m2",
        provider="llama_cpp",
        model="test-model",
    )

    assert payload["name"] == "Console message — Chat 1 (2026-07-11)"
    assert "Preview: answer body" in payload["description"]
    assert payload["tags"] == ["console", "artifact"]
    assert payload["categories"] == ["Console", "Artifacts"]
    metadata = payload["metadata"]
    assert metadata["artifact_source"] == "console"
    assert metadata["artifact_kind"] == "assistant-response"
    assert metadata["message_role"] == "Assistant"
    assert metadata["content"] == "answer body"
    assert metadata["content_truncated"] is False
    assert metadata["conversation_id"] == "conv-9"
    assert metadata["message_id"] == "m2"
    assert metadata["provider"] == "llama_cpp"
    assert metadata["model"] == "test-model"


def test_console_chatbook_artifact_payload_omits_blank_optional_metadata():
    payload = console_chatbook_artifact_payload(
        title="Title",
        message_text="body",
        message_role="",
        conversation_id=None,
        message_id="  ",
        provider="",
        model=None,
    )

    metadata = payload["metadata"]
    assert metadata["message_role"] == "Assistant"
    for key in ("conversation_id", "message_id", "provider", "model"):
        assert key not in metadata


def test_console_chatbook_artifact_payload_truncates_oversized_content():
    oversized = "y" * (CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS + 10)

    payload = console_chatbook_artifact_payload(
        title="Title",
        message_text=oversized,
        message_role="Assistant",
    )

    metadata = payload["metadata"]
    assert len(metadata["content"]) == CONSOLE_CHATBOOK_ARTIFACT_CONTENT_MAX_CHARS
    assert metadata["content_truncated"] is True
