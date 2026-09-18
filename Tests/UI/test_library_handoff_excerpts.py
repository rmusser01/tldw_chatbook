"""Source builders must preserve bounded, correctly attributed text (TASK-2376)."""

from dataclasses import replace

import pytest

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationMessageView,
    ConversationReaderState,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _screen_with_conversation(*texts: str) -> LibraryScreen:
    screen = LibraryScreen(_build_test_app())
    state = screen._conversations_state
    state.page_records = ({"id": "chat-a", "title": "Planning", "version": 4},)
    screen._selected_conversation_id = "chat-a"
    state.reader_state = ConversationReaderState(
        selected_id="chat-a",
        selected_version=4,
        loaded_id="chat-a",
        loaded_version=4,
        loaded_generation=2,
        generation=2,
        messages=tuple(
            ConversationMessageView(
                str(i), "user" if i % 2 == 0 else "assistant", "", "1", len(text), text
            )
            for i, text in enumerate(texts)
        ),
        message_total=len(texts),
        complete=True,
    )
    return screen


def test_conversation_builder_starts_with_actual_speaker_labeled_transcript():
    screen = _screen_with_conversation("Keep [this] and café.", "Review on Friday.")
    payload = screen._selected_conversation_handoff_payload()

    assert payload is not None
    assert payload.body.startswith(
        "Conversation excerpt:\nuser: Keep [this] and café.\n\nassistant: Review on Friday."
    )
    assert payload.source_id == "chat-a"
    assert payload.body_truncated is False


@pytest.mark.parametrize(
    "changes",
    [
        {"selected_id": "chat-b", "loaded_id": "chat-b"},
        {"selected_id": "chat-b"},
        {"selected_version": 5},
        {"generation": 3},
        {"complete": False},
        {"loading": True},
        {"unavailable": True},
        {"error": "failed"},
        {"bulk_active": True},
    ],
)
def test_conversation_builder_refuses_unsettled_or_other_reader_text(changes):
    screen = _screen_with_conversation("Never attribute this to another source.")
    screen._conversations_state.reader_state = replace(
        screen._conversations_state.reader_state, **changes
    )
    assert screen._selected_conversation_handoff_payload() is None


@pytest.mark.parametrize(
    "length,truncated", [(2994, False), (2995, True), (50000, True)]
)
def test_conversation_excerpt_is_bounded_and_reports_truncation(length, truncated):
    screen = _screen_with_conversation("x" * length)
    payload = screen._selected_conversation_handoff_payload()

    assert payload is not None
    excerpt = payload.body.split("Conversation excerpt:\n", 1)[1].split(
        "\n\nConversation:", 1
    )[0]
    assert excerpt == "user: " + "x" * min(length, 2994)
    assert payload.body_truncated is truncated


def test_conversation_reports_omitted_later_messages_at_exact_boundary():
    screen = _screen_with_conversation("x" * 2994, "Later message.")
    payload = screen._selected_conversation_handoff_payload()
    assert payload is not None
    assert payload.body_truncated is True
    assert "Later message." not in payload.body


def test_long_speaker_name_cannot_displace_the_actual_message_text():
    screen = _screen_with_conversation("Actual first message.", "Actual reply.")
    state = screen._conversations_state.reader_state
    screen._conversations_state.reader_state = replace(
        state,
        messages=(replace(state.messages[0], sender="Name" * 1000), state.messages[1]),
    )
    payload = screen._selected_conversation_handoff_payload()
    assert payload is not None
    assert "Actual first message." in payload.body
    assert "assistant: Actual reply." in payload.body
    assert "Name" * 1000 not in payload.body
    assert payload.body_truncated is True


def test_empty_conversation_does_not_invent_transcript_content():
    payload = _screen_with_conversation()._selected_conversation_handoff_payload()
    assert payload is not None
    assert payload.body.startswith("Conversation excerpt:\nNo stored messages.")
    assert payload.body_truncated is False


@pytest.mark.parametrize("length,truncated", [(0, False), (500, False), (501, True)])
def test_media_content_precedes_metadata_and_retains_existing_bound(length, truncated):
    screen = LibraryScreen(_build_test_app())
    screen._media_state.detail = {
        "id": 7,
        "title": "Very long title " * 400,
        "content": "x" * length,
        "type": "document",
    }
    payload = screen._selected_media_handoff_payload()
    assert payload is not None
    assert payload.body.startswith(
        "Content excerpt:\n"
        + ("x" * min(length, 500) or "No stored content.")
        + "\n\nMedia:"
    )
    assert payload.body_truncated is truncated
