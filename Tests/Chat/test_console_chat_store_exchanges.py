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


@pytest.fixture
def store_with_deferred_terminal_persistence():
    """Assistant message with terminal persistence deferred to the citation
    path -- armed via ``defer_terminal_persistence=True`` at append time,
    the same flag ``_append_eligible`` in
    Tests/Chat/test_console_terminal_citation_persistence.py uses (finalizer
    omitted: this only needs the deferred-persistence branch of
    ``mark_message_complete``, not a real citation write). Unlike the other
    fixtures, this message has NO ``persisted_message_id`` yet when
    streaming ends -- ``mark_message_complete`` creates the durable row for
    the first time via ``_persist_new_message(..., terminal_persistence=
    True)`` rather than routing through ``_persist_existing_message``."""
    persistence = _recording_exchange_persistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
        defer_terminal_persistence=True,
    )
    assert message.id in store._terminal_persistence_deferred_ids, "precondition"
    store.append_stream_chunk(message.id, "hi")
    return store, message.id, persistence


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


def test_abandoned_exchange_run_tags_reads_the_native_bookkeeping(
        store_after_variant_restore):
    """task-9: the public accessor the Conversation Inspector's Exchange
    tab uses to resolve a NATIVE (not-yet-persisted) capture's ``abandoned``
    flag -- must reflect the same run_tags ``attach_message_exchanges``
    marks internally, without exposing the private mutable set itself."""
    store, mid = store_after_variant_restore  # mid in _variant_restored_message_ids
    store.attach_message_exchanges(mid, [_cap(run_tag="r2")])

    tags = store.abandoned_exchange_run_tags(mid)

    assert tags == frozenset({"r2"})
    # An immutable snapshot -- mutating the return value must not reach
    # back into the store's own bookkeeping.
    assert isinstance(tags, frozenset)


def test_abandoned_exchange_run_tags_empty_for_an_unknown_message():
    """A message the store has never seen (or one with no abandoned runs
    at all) returns an empty set rather than raising."""
    store = ConsoleChatStore()
    assert store.abandoned_exchange_run_tags("nonexistent") == frozenset()


def test_ephemeral_session_never_persists(ephemeral_store):
    store, mid = ephemeral_store
    store.attach_message_exchanges(mid, [_cap()])
    # drive terminal; assert the persistence fake saw NO exchange append
    store.mark_message_complete(mid)
    assert store.persistence.appended_exchange_rows == []


def test_ephemeral_session_keeps_exchanges_in_memory(ephemeral_store):
    """FINDING 3 (positive pin): the negative test above only proves nothing
    reached the persistence fake -- this proves the captures are genuinely
    HELD in memory (not silently dropped), which is what an ephemeral
    session's own conversation-inspector view would read from."""
    store, mid = ephemeral_store
    store.attach_message_exchanges(mid, [_cap()])
    store.mark_message_complete(mid)
    assert store.get_message(mid).exchanges


def test_deferred_terminal_persistence_flushes_exchanges(
        store_with_deferred_terminal_persistence):
    """FINDING 1: the citation-deferred terminal branch creates the durable
    row via ``_persist_new_message(..., terminal_persistence=True)``
    directly, never routing through ``_persist_existing_message`` -- so
    captures attached during streaming (before the message had a
    ``persisted_message_id`` at all) must still flush at that terminal
    mark, not be silently dropped forever."""
    store, mid, persistence = store_with_deferred_terminal_persistence
    store.attach_message_exchanges(mid, [_cap()])
    store.mark_message_complete(mid)
    assert persistence.appended_exchange_rows


def test_persist_exchanges_only_survives_a_serialization_failure(
        store_with_fake_persistence):
    """FINDING 2: row-building (``capture_to_blob``'s JSON serialization)
    must run INSIDE ``_persist_exchanges_only``'s try, not before it -- a
    malformed capture (a circular reference in ``request``) degrades to the
    same ``exchange_flush_failed`` warning as a writer failure, and must
    never escape past an already-committed terminal mark."""
    from loguru import logger as loguru_logger

    store, mid, persistence = store_with_fake_persistence
    circular: dict = {}
    circular["self"] = circular
    bad_capture = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="p", model="m",
        endpoint=None, request=circular, response={"content": "x"},
        status="complete", usage_json=None, omitted_keys=())
    store.attach_message_exchanges(mid, [bad_capture])

    diagnostics: list[str] = []
    sink_id = loguru_logger.add(
        diagnostics.append,
        level="WARNING",
        format="{extra[message_id]} {extra[error]} {message}",
    )
    try:
        store.mark_message_complete(mid)  # must not raise
    finally:
        loguru_logger.remove(sink_id)

    assert persistence.appended_exchange_rows == []  # never reached the writer
    assert any("exchange_flush_failed" in d for d in diagnostics), diagnostics


def test_append_message_exchanges_service_wrapper_logs_and_returns_false():
    """FINDING 4: real coverage for ``ChatPersistenceService.
    append_message_exchanges`` -- a raising DB must not escape the wrapper,
    the call must report ``False``, and the warning log must carry only
    ``message_id``/``error`` (never row contents or capture bytes)."""
    from loguru import logger as loguru_logger

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

    class _RaisingDb:
        def append_message_exchanges_local(self, message_id, rows):
            raise RuntimeError("disk full")

    service = ChatPersistenceService(_RaisingDb())
    rows = [{
        "run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
        "capture_blob": b"SECRET-CAPTURE-BYTES", "created_at": "t",
    }]

    diagnostics: list[str] = []
    sink_id = loguru_logger.add(
        diagnostics.append,
        level="WARNING",
        format="{extra[message_id]} {extra[error]} {message}",
    )
    try:
        result = service.append_message_exchanges(message_id="msg-1", rows=rows)
    finally:
        loguru_logger.remove(sink_id)

    assert result is False
    assert any(
        "exchange_append_failed" in d and "msg-1" in d for d in diagnostics
    ), diagnostics
    assert not any("SECRET-CAPTURE-BYTES" in d for d in diagnostics), diagnostics
