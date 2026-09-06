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

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat import console_chat_store as store_module
from tldw_chatbook.Chat import console_capture_policy_repository as repository_module
from tldw_chatbook.Chat.console_capture_policy_repository import (
    CapturePolicyWriteResult,
    CapturePolicyWriteStatus,
    ConversationCapturePolicy,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleStagedConversationIdentity,
)
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, ExchangeCapture


def _cap(run_tag="r1", seq=0, status="complete"):
    return ExchangeCapture(
        run_tag=run_tag,
        seq=seq,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request={"messages_payload": []},
        response={"content": "x"},
        status=status,
        usage_json=None,
        omitted_keys=(),
    )


@pytest.mark.parametrize("terminal", ("complete", "stopped", "failed"))
@pytest.mark.parametrize("ephemeral", (False, True))
def test_terminal_generation_preserves_exchanges_with_real_sqlite(
    tmp_path, terminal: str, ephemeral: bool
) -> None:
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_exchange_capture import capture_from_blob
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "terminal.sqlite", client_id="exchange-terminal")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Terminal exchanges", ephemeral=ephemeral)
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        store.append_stream_chunk(message.id, "Retained answer")
        # The real getter materializes the pending streaming row; an empty
        # placeholder deliberately has no durable message ID yet.
        message = store.get_message(message.id)
        persisted_id = message.persisted_message_id
        if not ephemeral:
            assert persisted_id is not None
            before_version = db.get_message_by_id(persisted_id)["version"]
        capture = _cap(status="error" if terminal == "failed" else terminal)
        store.attach_message_exchanges(message.id, [capture])

        result = getattr(store, f"mark_message_{terminal}")(message.id)

        assert result.content == "Retained answer"
        assert result.status == terminal
        assert result.assistant_generation_state == terminal
        assert result.exchanges == (capture,)
        db.close_connection()
        if ephemeral:
            assert session.persisted_conversation_id is None
            assert result.persisted_message_id is None
            with db.transaction() as cursor:
                counts = cursor.execute(
                    "SELECT (SELECT COUNT(*) FROM conversations), "
                    "(SELECT COUNT(*) FROM messages), "
                    "(SELECT COUNT(*) FROM message_exchanges)"
                ).fetchone()
                assert tuple(counts) == (0, 0, 0)
        else:
            row = db.get_message_by_id(persisted_id)
            assert row["content"] == "Retained answer"
            assert row["assistant_generation_state"] == terminal
            assert row["version"] == before_version + 1
            rows = db.get_message_exchanges(persisted_id)
            assert len(rows) == 1
            assert rows[0]["status"] == capture.status
            assert capture_from_blob(rows[0]["capture_blob"]).response == {
                "content": "x"
            }
    finally:
        with db.quiesce_connections(timeout_seconds=2):
            pass
        db.close_connection()
        assert db.registered_connection_count() == 0


def test_capture_policy_state_uses_exact_revisions():
    store = ConsoleChatStore()
    session = store.ensure_session()
    initial = store.capture_policy_state(session.id)

    value, slot_revision, policy_revision = store.set_session_next_capture_detail(
        session.id,
        CaptureDetail.FULL,
        expected_policy_revision=initial.policy_revision,
    )

    assert value is CaptureDetail.FULL
    assert slot_revision == 1
    assert policy_revision == 1
    assert (
        store.consume_session_next_capture_detail(
            session.id, expected_next_revision=slot_revision
        )
        is True
    )
    assert store.capture_policy_state(session.id).next_detail is None


def test_capture_policy_rejects_stale_mutation_without_disclosure():
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.set_session_next_capture_detail(
        session.id, CaptureDetail.SAFE, expected_policy_revision=0
    )

    with pytest.raises(store_module.CapturePolicyStaleError) as raised:
        store.replace_session_capture_override(
            session.id,
            CaptureDetail.FULL,
            expected_policy_revision=0,
        )

    assert str(raised.value) == ""
    assert store.capture_policy_state(session.id).conversation_detail is None


def test_exact_revision_consumption_preserves_concurrently_rearmed_slot():
    store = ConsoleChatStore()
    session = store.ensure_session()
    _, admitted_revision, policy_revision = store.set_session_next_capture_detail(
        session.id, CaptureDetail.FULL, expected_policy_revision=0
    )
    store.set_session_next_capture_detail(
        session.id,
        CaptureDetail.SAFE,
        expected_policy_revision=policy_revision,
    )

    assert (
        store.consume_session_next_capture_detail(
            session.id, expected_next_revision=admitted_revision
        )
        is False
    )
    assert store.capture_policy_state(session.id).next_detail is CaptureDetail.SAFE


def test_capture_policy_hydrates_from_the_existing_repository():
    class Repository:
        @staticmethod
        def read(conversation_id):
            assert conversation_id == "conversation-1"
            return repository_module.CapturePolicyReadResult(
                repository_module.CapturePolicyReadStatus.FOUND,
                ConversationCapturePolicy(
                    conversation_id=conversation_id,
                    detail=CaptureDetail.FULL,
                    capture_enabled=None,
                    pii_redaction_enabled=None,
                    updated_at="2026-08-26T00:00:00Z",
                ),
            )

    store = ConsoleChatStore()
    store.capture_policy_repository = Repository()
    session = store.ensure_session()
    session.persisted_conversation_id = "conversation-1"

    store.hydrate_session_capture_policy(session.id)

    assert (
        store.capture_policy_state(session.id).conversation_detail is CaptureDetail.FULL
    )


def test_unavailable_capture_policy_hydration_publishes_explicit_safe_pending() -> None:
    class Repository:
        @staticmethod
        def read(_conversation_id):
            return repository_module.CapturePolicyReadResult(
                repository_module.CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT,
                None,
            )

    store = ConsoleChatStore()
    store.capture_policy_repository = Repository()
    session = store.ensure_session()
    session.persisted_conversation_id = "conversation-1"
    session.capture_detail_override = CaptureDetail.FULL

    outcome = store.hydrate_session_capture_policy(session.id)

    state = store.capture_policy_state(session.id)
    assert (
        outcome.status
        is repository_module.CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT
    )
    assert state.conversation_detail is CaptureDetail.SAFE
    assert state.save_pending is True


@pytest.mark.parametrize("failed_stage", ("privacy", "detail"))
def test_failed_staged_safe_flush_stays_safe_and_pending_after_publication(
    failed_stage,
):
    class Repository:
        @staticmethod
        def replace_privacy(conversation_id, *, capture_enabled, pii_redaction_enabled):
            assert (conversation_id, capture_enabled, pii_redaction_enabled) == (
                "conversation-1",
                None,
                None,
            )
            return CapturePolicyWriteResult(
                CapturePolicyWriteStatus.UNAVAILABLE
                if failed_stage == "privacy"
                else CapturePolicyWriteStatus.UNCHANGED,
                None,
            )

        @staticmethod
        def replace(conversation_id, detail):
            assert failed_stage == "detail"
            assert (conversation_id, detail) == (
                "conversation-1",
                CaptureDetail.SAFE,
            )
            return CapturePolicyWriteResult(CapturePolicyWriteStatus.UNAVAILABLE, None)

    store = ConsoleChatStore()
    store.capture_policy_repository = Repository()
    session = store.ensure_session()
    store.replace_session_capture_override(
        session.id,
        CaptureDetail.SAFE,
        expected_policy_revision=0,
    )

    store.publish_committed_identity(
        session.id,
        ConsoleStagedConversationIdentity("conversation-1", "Conversation"),
    )

    state = store.capture_policy_state(session.id)
    assert state.conversation_detail is CaptureDetail.SAFE
    assert state.save_pending is True


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
    store_with_streaming_assistant,
):
    """Carried-context refinement: a stop-time snapshot for (run_tag, seq)
    is superseded by a later, non-'stopped' capture for the same key -- the
    closed capture arriving after the stop-time snapshot was attached."""
    store, mid = store_with_streaming_assistant
    store.attach_message_exchanges(mid, [_cap(seq=0, status="stopped")])
    store.attach_message_exchanges(mid, [_cap(seq=0, status="complete")])
    message = store.get_message(mid)
    assert [c.status for c in message.exchanges] == ["complete"]


def test_attach_keeps_the_first_capture_when_neither_side_is_stopped(
    store_with_streaming_assistant,
):
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


def test_flush_derives_capture_detail_from_the_immutable_capture(
    store_with_fake_persistence,
):
    store, mid, persistence = store_with_fake_persistence
    capture = _cap()
    object.__setattr__(capture, "capture_detail", CaptureDetail.FULL)
    store.attach_message_exchanges(mid, [capture])
    store.mark_message_complete(mid)
    assert persistence.appended_exchange_rows[0]["capture_detail"] == "full"


def test_attach_after_terminal_flushes_immediately(store_with_fake_persistence):
    """Stop-path inversion: stop finalizes first, capture attaches late."""
    store, mid, persistence = store_with_fake_persistence
    # drive terminal FIRST, then attach
    store.mark_message_complete(mid)
    store.attach_message_exchanges(mid, [_cap(status="stopped")])
    assert persistence.appended_exchange_rows


def test_variant_restored_message_keeps_captures_marked_abandoned(
    store_after_variant_restore,
):
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
    store_after_variant_restore,
):
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
    store_with_deferred_terminal_persistence,
):
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
    store_with_fake_persistence, monkeypatch
):
    """FINDING 2: row-building (``capture_to_blob``'s JSON serialization)
    must run INSIDE ``_persist_exchanges_only``'s try, not before it -- a
    malformed capture (a circular reference in ``request``) degrades to the
    same ``exchange_flush_failed`` warning as a writer failure, and must
    never escape past an already-committed terminal mark."""
    from loguru import logger as loguru_logger

    store, mid, persistence = store_with_fake_persistence
    canary = "CANARY_FULL_CAPTURE_MUST_NOT_REACH_LOGS"
    bad_capture = ExchangeCapture(
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request={},
        response={"content": "x"},
        status="complete",
        usage_json=None,
        omitted_keys=(),
    )
    store.attach_message_exchanges(mid, [bad_capture])
    monkeypatch.setattr(
        store_module,
        "capture_to_blob",
        lambda _capture: (_ for _ in ()).throw(RuntimeError(canary)),
    )

    diagnostics: list[str] = []
    sink_id = loguru_logger.add(
        diagnostics.append,
        level="WARNING",
        format="{extra} {message}",
    )
    try:
        store.mark_message_complete(mid)  # must not raise
    finally:
        loguru_logger.remove(sink_id)

    assert persistence.appended_exchange_rows == []  # never reached the writer
    assert any("exchange_flush_failed" in d for d in diagnostics), diagnostics
    assert any("RuntimeError" in d for d in diagnostics), diagnostics
    assert canary not in "\n".join(diagnostics)


def test_append_message_exchanges_service_wrapper_logs_and_returns_false():
    """FINDING 4: real coverage for ``ChatPersistenceService.
    append_message_exchanges`` -- a raising DB must not escape the wrapper,
    the call must report ``False``, and the warning log must carry only a
    stable category, ``message_id``, and exception type (never semantic
    exception text, row contents, or capture bytes)."""
    from loguru import logger as loguru_logger

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

    canary = "CANARY_SEMANTIC_REQUEST_RESPONSE_MUST_NOT_REACH_LOGS"
    failure = RuntimeError(
        f"{canary}: request=private-prompt response=private-completion"
    )

    class _RaisingDb:
        def append_message_exchanges_local(self, message_id, rows):
            raise failure

    service = ChatPersistenceService(_RaisingDb())
    rows = [
        {
            "run_tag": "r1",
            "seq": 0,
            "status": "complete",
            "abandoned": False,
            "capture_blob": b"SECRET-CAPTURE-BYTES",
            "created_at": "t",
        }
    ]

    events: list[dict] = []
    sink_id = loguru_logger.add(
        lambda message: events.append(message.record.copy()),
        level="WARNING",
    )
    try:
        result = service.append_message_exchanges(message_id="msg-1", rows=rows)
    finally:
        loguru_logger.remove(sink_id)

    assert result is False
    event = next(e for e in events if e["message"] == "exchange_append_failed")
    assert event["extra"]["message_id"] == "msg-1"
    assert event["extra"]["error_type"] == "RuntimeError"
    assert "error" not in event["extra"]
    serialized_event = repr(event)
    assert canary not in serialized_event
    assert repr(failure) not in serialized_event
    assert "SECRET-CAPTURE-BYTES" not in serialized_event


# --- Blob-compression memoization (Qodo PR #1883 finding 4) ----------------


def test_persist_exchanges_only_compresses_each_capture_once_across_flushes(
    monkeypatch, store_with_fake_persistence
):
    """``_persist_exchanges_only`` runs on EVERY flush of a message with
    exchanges -- e.g. once per tool call in a long agent turn -- and used
    to call ``capture_to_blob`` for every capture on every one of those
    flushes, even ones already compressed and unchanged. Two consecutive
    flushes of the same unchanged captures must compress each capture only
    ONCE (spied via a wrapper around the real ``capture_to_blob``)."""
    import tldw_chatbook.Chat.console_chat_store as store_module

    store, mid, persistence = store_with_fake_persistence
    real_capture_to_blob = store_module.capture_to_blob
    calls: list[str] = []

    def _spy(capture):
        calls.append(f"{capture.run_tag}:{capture.seq}:{capture.status}")
        return real_capture_to_blob(capture)

    monkeypatch.setattr(store_module, "capture_to_blob", _spy)

    store.attach_message_exchanges(mid, [_cap(seq=0), _cap(seq=1)])
    store.mark_message_complete(mid)  # first flush: 2 unseen captures
    assert calls == ["r1:0:complete", "r1:1:complete"]

    # A second flush of the exact same, unchanged captures (e.g. a later
    # metadata-only edit that re-enters ``_persist_exchanges_only``) must
    # not recompress either one.
    message = store.get_message(mid)
    store._persist_exchanges_only(message)

    assert calls == ["r1:0:complete", "r1:1:complete"], (
        "unchanged captures must not be recompressed on a second flush"
    )
    # The writer still received both rows on the second flush (memoization
    # is a compression-cost optimization, not a "skip the write" one).
    assert len(persistence.appended_exchange_rows) == 4


def test_persist_exchanges_only_recompresses_a_superseded_stopped_capture(
    store_with_fake_persistence,
):
    """The one legitimate content change for an existing (run_tag, seq) key
    -- a 'stopped' snapshot superseded by a later non-'stopped' capture for
    the same key (``attach_message_exchanges``'s documented merge rule) --
    is a STATUS change, so it must be treated as a cache MISS: recompressed
    and persisted with the NEW bytes, not served the stale 'stopped' blob."""
    store, mid, persistence = store_with_fake_persistence

    store.attach_message_exchanges(mid, [_cap(seq=0, status="stopped")])
    store.mark_message_complete(mid)  # first flush: the stop-time snapshot
    stopped_rows = [
        r for r in persistence.appended_exchange_rows if r["status"] == "stopped"
    ]
    assert len(stopped_rows) == 1
    stopped_blob = stopped_rows[0]["capture_blob"]
    assert mid in store._exchange_blob_cache
    assert ("r1", 0, "stopped") in store._exchange_blob_cache[mid]

    # Supersede with the run's actual closed outcome for the same key --
    # message is already terminal, so this flushes immediately.
    store.attach_message_exchanges(mid, [_cap(seq=0, status="complete")])

    complete_rows = [
        r for r in persistence.appended_exchange_rows if r["status"] == "complete"
    ]
    assert len(complete_rows) == 1
    complete_blob = complete_rows[0]["capture_blob"]
    assert complete_blob != stopped_blob, (
        "a superseded capture must persist NEW bytes, not the stale stopped blob"
    )
    # The stale 'stopped' cache entry is pruned, not left to linger --
    # ``_exchange_blob_cache`` cannot grow past what is currently live.
    assert ("r1", 0, "stopped") not in store._exchange_blob_cache[mid]
    assert ("r1", 0, "complete") in store._exchange_blob_cache[mid]


def test_exchange_blob_cache_does_not_leak_between_messages(
    store_with_fake_persistence,
):
    """Two different messages that each happen to carry a capture with the
    same (run_tag, seq, status) key must not share a cache entry -- the
    cache is keyed by message id first, so one message's blob can never be
    served to (or invalidated by) another's flush."""
    store, mid, persistence = store_with_fake_persistence
    session_id = store.session_id_for_message(mid)
    other = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(other.id, "hi")

    store.attach_message_exchanges(mid, [_cap(seq=0, status="complete")])
    store.mark_message_complete(mid)
    store.attach_message_exchanges(other.id, [_cap(seq=0, status="complete")])
    store.mark_message_complete(other.id)

    assert mid in store._exchange_blob_cache
    assert other.id in store._exchange_blob_cache
    assert store._exchange_blob_cache[mid] is not store._exchange_blob_cache[other.id]


def test_restore_state_clears_exchange_blob_cache_and_abandoned_run_tags(
    store_with_fake_persistence,
):
    """M2: ``restore_state`` used to clear 25 sibling in-memory maps but
    leave ``_exchange_blob_cache``/``_abandoned_exchange_run_tags``
    untouched -- unlike ``delete_message`` and session-close (the two sites
    this pair's own bound comment used to cite as exhaustive), a restore
    (session switch / restart replay) replaces the in-memory session/message
    set wholesale without going through either of those, so a stale entry
    keyed by a message id no longer present in the restored state could
    linger indefinitely."""
    store, mid, persistence = store_with_fake_persistence
    session_id = store.session_id_for_message(mid)

    store.attach_message_exchanges(mid, [_cap(seq=0, status="complete")])
    store.mark_message_complete(mid)
    assert mid in store._exchange_blob_cache  # precondition

    store._abandoned_exchange_run_tags[mid] = {"r-abandoned"}

    session = store._sessions[session_id]
    store.restore_state(sessions=[replace(session)], active_session_id=session_id)

    assert store._exchange_blob_cache == {}
    assert store._abandoned_exchange_run_tags == {}
