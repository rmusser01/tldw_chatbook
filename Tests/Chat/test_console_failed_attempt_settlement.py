"""No-response failure settlements keep custody through faults and cancellation."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_trace_settlement import _call, _request, db as _db_fixture
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderStreamSignals,
    _settle_trace_response,
)
from tldw_chatbook.Chat.console_trace_models import TraceCallState
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_settlement import (
    ConsoleTraceSettlementCoordinator,
)

pytestmark = pytest.mark.bootstrap_profile
db = _db_fixture


async def test_failed_seal_is_retained_and_retried_through_store_teardown(db):
    class FailOnceRepository(ConsoleTraceRepository):
        failed = False

        def advance_call_state(self, *args, **kwargs):
            if kwargs.get("target") is TraceCallState.ERROR and not self.failed:
                self.failed = True
                raise RuntimeError("fixture write failure")
            return super().advance_call_state(*args, **kwargs)

    repository = FailOnceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _, _, call_id = _call(db, repository)
    handoff = coordinator.prepare_handoff(
        db,
        replace(
            _request(call_id), outcome=TraceCallState.ERROR, response_envelope=None
        ),
    )
    store = ConsoleChatStore(settle_provider_traces_off_thread=True)
    signals = ConsoleProviderStreamSignals()
    signals.bind_trace_settlement_sink(
        lambda item: store.register_provider_trace_settlement("missing-owner", item)
    )
    boundary = SimpleNamespace(prepare_response_settlement=lambda *args: handoff)
    try:
        await _settle_trace_response(
            boundary,
            (),
            outcome=TraceCallState.ERROR,
            usage=None,
            signals=signals.new_usage_call(),
        )
    finally:
        await asyncio.to_thread(store.end_app_runtime)
    assert repository.failed
    assert coordinator.pending_count == 0
    assert store.pending_provider_trace_settlement_work_count() == 0
    with db.transaction() as cursor:
        assert repository.get_call(cursor, call_id).state is TraceCallState.ERROR
        assert repository.get_response_link(cursor, call_id) is None
    # A later canonical save cannot change the trace-owned fingerprint.
    assert handoff.settle("later-assistant")


async def test_cancelled_gateway_keeps_failure_settlement_work_owned(db):
    entered, release = threading.Event(), threading.Event()

    class BlockingRepository(ConsoleTraceRepository):
        def advance_call_state(self, *args, **kwargs):
            if kwargs.get("target") is TraceCallState.ERROR:
                entered.set()
                assert release.wait(5)
            return super().advance_call_state(*args, **kwargs)

    repository = BlockingRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    _, _, call_id = _call(db, repository)
    handoff = coordinator.prepare_handoff(
        db,
        replace(
            _request(call_id), outcome=TraceCallState.ERROR, response_envelope=None
        ),
    )
    owned = []
    signals = ConsoleProviderStreamSignals(
        provider_work_callback=lambda task, cancel: (owned.append(task) or True)
    )
    boundary = SimpleNamespace(prepare_response_settlement=lambda *args: handoff)
    task = asyncio.create_task(
        _settle_trace_response(
            boundary,
            (),
            outcome=TraceCallState.ERROR,
            usage=None,
            signals=signals.new_usage_call(),
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(owned) == 1 and not owned[0].done()
    finally:
        release.set()
        await asyncio.gather(*owned)
    with db.transaction() as cursor:
        assert repository.get_call(cursor, call_id).state is TraceCallState.ERROR


async def test_pending_failed_seal_survives_later_canonical_assistant_save(db):
    class FailOnceRepository(ConsoleTraceRepository):
        failed = False

        def advance_call_state(self, *args, **kwargs):
            if kwargs.get("target") is TraceCallState.ERROR and not self.failed:
                self.failed = True
                raise RuntimeError("fixture write failure")
            return super().advance_call_state(*args, **kwargs)

    repository = FailOnceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session()
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="question", persist=True
    )
    assert session.persisted_conversation_id is not None
    _, _, call_id = _call(
        db, repository, conversation_id=session.persisted_conversation_id
    )
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    handoff = coordinator.prepare_handoff(
        db, _request(call_id, outcome=TraceCallState.ERROR)
    )
    signals = ConsoleProviderStreamSignals()
    signals.bind_trace_settlement_sink(
        lambda item: store.register_provider_trace_settlement(assistant.id, item)
    )
    boundary = SimpleNamespace(prepare_response_settlement=lambda *args: handoff)
    try:
        await _settle_trace_response(
            boundary,
            (),
            outcome=TraceCallState.ERROR,
            usage=None,
            signals=signals.new_usage_call(),
        )
        assert coordinator.pending_count == 1
        assert store.pending_provider_trace_settlement_count(assistant.id) == 1
        store.append_stream_chunk(assistant.id, "later successful answer")
        completed = await asyncio.to_thread(store.mark_message_complete, assistant.id)
        assert completed.persisted_message_id is not None
        # Assert before teardown: a cleanup retry with None must not hide a
        # fingerprint collision when the real assistant ID becomes available.
        assert coordinator.pending_count == 0
        assert store.pending_provider_trace_settlement_work_count() == 0
        with db.transaction() as cursor:
            assert repository.get_call(cursor, call_id).state is TraceCallState.ERROR
            assert repository.get_response_link(cursor, call_id) is None
    finally:
        await asyncio.to_thread(store.end_app_runtime)
