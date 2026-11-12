"""Transcript-visible store mutations advance the display projection token."""

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

pytestmark = pytest.mark.bootstrap_profile


def _assistant_stream(store, session_id):
    return store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content=""
    )


def test_append_and_stream_bump_the_token():
    store = ConsoleChatStore()
    session = store.ensure_session()
    before = store.display_projection_revision(session.id)

    message = _assistant_stream(store, session.id)
    assert store.display_projection_revision(session.id) > before

    mid = store.display_projection_revision(session.id)
    store.append_stream_chunk(message.id, "hello")
    assert store.display_projection_revision(session.id) > mid


def test_terminal_marks_bump_the_token_even_without_new_content():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = _assistant_stream(store, session.id)
    store.append_stream_chunk(message.id, "done")
    store.messages_for_session(session.id)  # fold happens on read

    before = store.display_projection_revision(session.id)
    store.mark_message_complete(message.id)
    assert store.display_projection_revision(session.id) > before


def test_content_edit_and_delete_bump_the_token():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = _assistant_stream(store, session.id)
    store.append_stream_chunk(message.id, "text")
    store.mark_message_complete(message.id)

    before = store.display_projection_revision(session.id)
    store.update_message_content(message.id, "edited")
    assert store.display_projection_revision(session.id) > before

    mid = store.display_projection_revision(session.id)
    store.delete_message(message.id)
    assert store.display_projection_revision(session.id) > mid


def test_reads_alone_do_not_bump_the_token():
    store = ConsoleChatStore()
    session = store.ensure_session()
    _assistant_stream(store, session.id)
    settled = store.display_projection_revision(session.id)

    store.messages_for_session(session.id)
    store.messages_for_session(session.id)
    assert store.display_projection_revision(session.id) == settled
