"""O(1) transcript accessors on ConsoleChatStore (TASK-24300).

``messages_for_session`` materialises every stream buffer and deep-snapshots
every message. Four call sites used it as a predicate and two more used it to
walk backwards to the newest match, so both shapes paid O(transcript) for an
answer that does not need the transcript. These tests pin the replacements'
agreement with the projection they replace -- an accessor that disagreed with
``messages_for_session`` about emptiness would be worse than the cost it saves.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def _session(store: ConsoleChatStore):
    """Create and return one session on the store."""
    return store.create_session(title="counts")


def test_message_count_and_has_messages_agree_with_the_projection():
    """The O(1) accessors never disagree with the list they replace."""
    store = ConsoleChatStore()
    session = _session(store)

    assert store.message_count(session.id) == 0
    assert store.has_messages(session.id) is False
    assert len(store.messages_for_session(session.id)) == 0

    for index in range(5):
        store.append_message(
            session.id,
            role=(
                ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT
            ),
            content=f"message {index}",
        )
        assert store.message_count(session.id) == index + 1
        assert store.has_messages(session.id) is True
        assert store.message_count(session.id) == len(
            store.messages_for_session(session.id)
        )


def test_transcript_accessors_reject_an_unknown_session():
    """An unknown session raises, matching ``messages_for_session``."""
    store = ConsoleChatStore()

    with pytest.raises(KeyError):
        store.message_count("no-such-session")
    with pytest.raises(KeyError):
        store.has_messages("no-such-session")
    with pytest.raises(KeyError):
        next(store.iter_messages_newest_first("no-such-session"))


def test_iter_messages_newest_first_reverses_the_projection():
    """The lazy walk yields exactly the projection, newest first."""
    store = ConsoleChatStore()
    session = _session(store)
    for index in range(6):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=f"message {index}",
        )

    walked = [message.content for message in store.iter_messages_newest_first(session.id)]
    projected = [message.content for message in store.messages_for_session(session.id)]

    assert walked == list(reversed(projected))


def test_iter_messages_newest_first_stops_early_without_walking_the_rest():
    """Breaking out of the walk does not snapshot the whole transcript.

    This is the property the reverse-scan call sites depend on: they look for
    the most recent message matching a predicate and almost always find it in
    the first few. Counting ``_snapshot`` calls proves the laziness directly --
    asserting only on the yielded values would pass just as well for an
    implementation that built the entire list up front.
    """
    store = ConsoleChatStore()
    session = _session(store)
    for index in range(50):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=f"message {index}",
        )

    snapshots = {"count": 0}
    real_snapshot = store._snapshot

    def counted_snapshot(message):
        snapshots["count"] += 1
        return real_snapshot(message)

    store._snapshot = counted_snapshot  # type: ignore[method-assign]

    for message in store.iter_messages_newest_first(session.id):
        if message.content == "message 49":
            break

    assert snapshots["count"] == 1, (
        f"stopping at the newest message snapshotted {snapshots['count']} "
        "messages; the walk is not lazy."
    )


def test_streaming_text_is_visible_to_the_lazy_walk():
    """A streaming message's buffered text is materialised before it is yielded.

    ``messages_for_session`` materialises stream buffers before snapshotting;
    the lazy walk must do the same per message or a caller inspecting content
    would see a partially-written turn as empty.
    """
    store = ConsoleChatStore()
    session = _session(store)
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(message.id, "streamed body")

    newest = next(store.iter_messages_newest_first(session.id))

    assert newest.content == "streamed body"
    assert newest.content == store.messages_for_session(session.id)[-1].content
