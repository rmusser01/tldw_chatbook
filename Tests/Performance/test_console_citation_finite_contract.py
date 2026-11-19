"""Original citation helper keeps custom cancellation and borrower/error routes."""
import asyncio
import sqlite3
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.thread import _WorkItem
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


@pytest.mark.parametrize("shape", ["memory", "subclass", "custom"])
async def test_original_nonstock_citation_route_keeps_direct_cancellation(tmp_path, shape):
    from tldw_chatbook.Chat.citation_provenance_runtime import (
        CitationProvenanceRuntimePolicy,
    )
    from tldw_chatbook.Chat.citation_trace_repository import CitationTraceRepository
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.citation_counts import read_citation_counts
    database = CharactersRAGDB(":memory:" if shape == "memory" else tmp_path / (shape + ".db"), "citation-contract")
    class CustomRepository(CitationTraceRepository):
        pass
    repository = SimpleNamespace(db=database) if shape == "custom" else (CustomRepository if shape == "subclass" else CitationTraceRepository)(database, policy=CitationProvenanceRuntimePolicy(), identity_context=None, fingerprint_codec=None)
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    calls, held = [], []
    eligible = (("native", "missing", "Reply", "complete"),)
    def reader(actual, entries):
        frame = sys._getframe()
        parent = frame.f_back
        try:
            while parent is not None and parent.f_code is not _WorkItem.run.__code__:
                parent = parent.f_back
            assert parent is not None and parent.f_globals is _WorkItem.run.__globals__
            item = parent.f_locals["self"]
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            held.append(item.future)
        finally:
            del frame, parent
        calls.append((actual, entries, threading.current_thread()))
        entered.set()
        try:
            assert actual is repository and entries is eligible
            assert release.wait(10)
            return {"native": 0}
        finally:
            retired.set()
    task = asyncio.create_task(read_citation_counts(repository, reader, eligible))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.wait({task}, timeout=0.05)
        direct_cancelled = task.done() and task.cancelled()
        assert not retired.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        try:
            if held:
                native_results = await asyncio.gather(asyncio.wrap_future(held[0]), return_exceptions=True)
        finally:
            database.close_connection()
    assert held and held[0].done() and native_results == [{"native": 0}], native_results
    assert retired.is_set()
    assert direct_cancelled
    assert len(calls) == 1 and calls[0][:2] == (repository, eligible)
    assert calls[0][2] is not threading.main_thread()



@pytest.mark.parametrize("borrowed", [False, True])
async def test_original_stock_citation_callback_error_preserves_borrowed_handle(tmp_path, borrowed):
    from tldw_chatbook.Chat.citation_provenance_runtime import (
        CitationProvenanceRuntimePolicy,
    )
    from tldw_chatbook.Chat.citation_trace_repository import CitationTraceRepository
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Console_Modules.citation_counts import read_citation_counts
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    database = CharactersRAGDB(tmp_path / "stock.db", "citation-contract")
    repository = CitationTraceRepository(database, policy=CitationProvenanceRuntimePolicy(), identity_context=None, fingerprint_codec=None)
    eligible = (("native", "missing", "Reply", "complete"),)
    loop = asyncio.get_running_loop()
    old_executor = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    connection, calls = None, []
    error = LookupError("original-citation-query-error")
    def reader(actual, entries):
        result = ChatScreen._read_console_citation_counts(actual, entries)
        assert result == {"native": 0}
        connection = database._local.conn
        assert not _closed(connection)
        calls.append((actual, entries, connection, threading.current_thread()))
        raise error
    try:
        if borrowed:
            connection = await asyncio.to_thread(database.get_connection)
        with pytest.raises(LookupError) as caught:
            await read_citation_counts(repository, reader, eligible)
        assert caught.value is error
        assert len(calls) == 1 and calls[0][:2] == (repository, eligible)
        actual = calls[0][2]
        assert _closed(actual) is not borrowed
        if borrowed:
            assert actual is connection
            assert await asyncio.to_thread(lambda: database.get_connection() is connection)
    finally:
        # This fixture-owned single worker retains the exact borrower even
        # when any preceding producer/result assertion raises.
        try:
            if connection is not None:
                await asyncio.to_thread(database.close_connection)
                assert _closed(connection)
        finally:
            loop._default_executor = old_executor
            executor.shutdown(wait=True)
            database.close_connection()
