"""Exchange attach lifecycle: dedup, stop-path flush, regen keep-marked,
ephemeral no-persist (task-6, Console Conversation Inspector).

Fixture and fake-persistence idioms are copied mechanically from
Tests/Chat/test_console_chat_store.py's ``set_message_usage`` tests -- the
named precedent for these exact lifecycle transitions (terminal mark,
variant restore, ephemeral, fake persistence):
  * streaming-assistant setup: ``test_set_message_usage_on_a_streaming_
    message_defers_persistence``
  * terminal-mark flush: ``test_set_message_usage_after_a_terminal_mark_
    flushes_to_persistence``
  * variant-restore driver (``begin_variant_stream`` +
    ``mark_message_stopped``): ``test_stopped_regenerate_keeps_the_
    original_answers_usage``
  * ephemeral driver (``create_session(ephemeral=True)``):
    ``test_temporary_session_keeps_override_without_durable_write``
  * extending ``RecordingPersistence`` with a recording subclass:
    ``_UsageUpdatePersistence`` / ``UsagePersistence``
"""
from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture


def _cap(run_tag="r1", seq=0, status="complete"):
    return ExchangeCapture(
        run_tag=run_tag, seq=seq, created_at="t", provider="p", model="m",
        endpoint=None, request={"messages_payload": []},
        response={"content": "x"}, status=status, usage_json=None,
        omitted_keys=())


def _recording_exchange_persistence():
    """A ``RecordingPersistence`` (Tests/Chat/test_console_chat_store.py)
    extended with ``append_message_exchanges``, mirroring how that file's
    own usage tests extend it (e.g. ``_UsageUpdatePersistence``,
    ``UsagePersistence``)."""
    from Tests.Chat.test_console_chat_store import RecordingPersistence

    class RecordingExchangePersistence(RecordingPersistence):
        def __init__(self):
            super().__init__()
            self.appended_exchange_rows = []

        def append_message_exchanges(self, *, message_id, rows):
            self.appended_exchange_rows.extend(rows)
            return True

    return RecordingExchangePersistence()


@pytest.fixture
def store_with_streaming_assistant():
    """Plain store (no persistence) + a still-streaming assistant message --
    mirrors ``test_set_message_usage_on_a_streaming_message_defers_
    persistence``'s setup, minus the persistence adapter (the dedup test
    below never reaches a flush)."""
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "hi")
    return store, message.id


@pytest.fixture
def store_with_fake_persistence():
    """Streaming assistant message backed by a recording persistence fake --
    mirrors ``test_set_message_usage_after_a_terminal_mark_flushes_to_
    persistence``'s setup."""
    persistence = _recording_exchange_persistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "hi")
    return store, message.id, persistence


@pytest.fixture
def store_after_variant_restore():
    """A completed message put through a stopped regenerate -- mirrors
    ``test_stopped_regenerate_keeps_the_original_answers_usage``'s driver
    (``begin_variant_stream`` + ``mark_message_stopped``), which restores
    the pre-regenerate content/status and marks the message in
    ``_variant_restored_message_ids``."""
    persistence = _recording_exchange_persistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "the original answer")
    store.mark_message_complete(message.id)

    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "half of a new ans")
    store.mark_message_stopped(message.id)
    assert message.id in store._variant_restored_message_ids, "precondition"

    return store, message.id


@pytest.fixture
def ephemeral_store():
    """Ephemeral session -- mirrors ``test_temporary_session_keeps_
    override_without_durable_write``'s driver (``create_session(
    ephemeral=True)``); a real persistence adapter is attached but never
    reached because ``persist_session_if_needed`` short-circuits."""
    persistence = _recording_exchange_persistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1", ephemeral=True)
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "hi")
    return store, message.id


def test_attach_dedups_by_run_tag_and_seq(store_with_streaming_assistant):
    store, mid = store_with_streaming_assistant
    store.attach_message_exchanges(mid, [_cap(seq=0)])
    store.attach_message_exchanges(mid, [_cap(seq=0), _cap(seq=1)])
    message = store.get_message(mid)  # use the store's real snapshot accessor
    assert [c.seq for c in message.exchanges] == [0, 1]


def test_attach_replaces_a_stopped_snapshot_with_the_terminal_capture(
        store_with_streaming_assistant):
    """Carried-context refinement: a stop-time snapshot for (run_tag, seq)
    is superseded by a later, non-'stopped' capture for the same key -- the
    closed capture arriving after the stop-time snapshot was attached."""
    store, mid = store_with_streaming_assistant
    store.attach_message_exchanges(mid, [_cap(seq=0, status="stopped")])
    store.attach_message_exchanges(mid, [_cap(seq=0, status="complete")])
    message = store.get_message(mid)
    assert [c.status for c in message.exchanges] == ["complete"]


def test_attach_keeps_the_first_capture_when_neither_side_is_stopped(
        store_with_streaming_assistant):
    """The refinement is scoped to a 'stopped' existing capture -- a repeat
    key otherwise still keeps the FIRST-attached capture."""
    store, mid = store_with_streaming_assistant
    store.attach_message_exchanges(mid, [_cap(seq=0, status="complete")])
    store.attach_message_exchanges(mid, [_cap(seq=0, status="error")])
    message = store.get_message(mid)
    assert [c.status for c in message.exchanges] == ["complete"]


def test_terminal_mark_flushes_exchanges(store_with_fake_persistence):
    store, mid, persistence = store_with_fake_persistence
    store.attach_message_exchanges(mid, [_cap()])
    # drive the message terminal via the store's real terminal-mark API
    store.mark_message_complete(mid)
    assert persistence.appended_exchange_rows  # fake recorded the flush


def test_attach_after_terminal_flushes_immediately(store_with_fake_persistence):
    """Stop-path inversion: stop finalizes first, capture attaches late."""
    store, mid, persistence = store_with_fake_persistence
    # drive terminal FIRST, then attach
    store.mark_message_complete(mid)
    store.attach_message_exchanges(mid, [_cap(status="stopped")])
    assert persistence.appended_exchange_rows


def test_variant_restored_message_keeps_captures_marked_abandoned(
        store_after_variant_restore):
    """CONTRAST with usage (which drops): spec owner decision 6."""
    store, mid = store_after_variant_restore  # mid in _variant_restored_message_ids
    store.attach_message_exchanges(mid, [_cap(run_tag="r2")])
    message = store.get_message(mid)
    assert any(c.run_tag == "r2" for c in message.exchanges)
    # the flush row carries abandoned=True
    rows = store.persistence.appended_exchange_rows
    assert rows
    row = next(r for r in rows if r["run_tag"] == "r2")
    assert row["abandoned"] is True


def test_ephemeral_session_never_persists(ephemeral_store):
    store, mid = ephemeral_store
    store.attach_message_exchanges(mid, [_cap()])
    # drive terminal; assert the persistence fake saw NO exchange append
    store.mark_message_complete(mid)
    assert store.persistence.appended_exchange_rows == []
