"""TASK-21126: the Search/RAG panel's report line over the REAL backend.

``test_library_rag_legacy_chunk_report.py`` pins the renderer's contract
against a stubbed scope service. This file runs the same panel against a
REAL ``RAGAdminScopeService`` -> ``LocalRAGAdminService`` -> file-backed
``MediaDatabase``, because moving the census onto a worker thread is a
change a stubbed async service cannot observe at all: the stub was already
awaitable, so it would have reported "off the loop" while the production
seam still blocked it.

It also walks the two lifecycle paths the perf burn-down keeps breaking:
unmounting the panel while a census is in flight, and quitting the app in
the same state.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService
from tldw_chatbook.RAG_Admin.rag_admin_scope_service import RAGAdminScopeService
from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
    LibrarySearchRagPanel,
)

REPORT_LINE_ID = "library-rag-legacy-chunk-line"
RECHUNK_BUTTON_ID = "library-rag-rechunk-legacy"

pytestmark = pytest.mark.asyncio


def _seed(db: MediaDatabase, *, legacy: int, stamped: int) -> None:
    conn = db.get_connection()
    now = "2026-08-23T00:00:00Z"
    conn.execute("PRAGMA foreign_keys = OFF")
    total = legacy + stamped
    conn.executemany(
        "INSERT INTO Media (id, title, type, content, content_hash, uuid, "
        "last_modified, version, client_id, deleted) "
        "VALUES (?,?,?,?,?,?,?,1,'test',0)",
        [
            (i, f"doc {i}", "document", "b", f"h{i}", f"m{i}", now)
            for i in range(1, total + 1)
        ],
    )
    rows = []
    n = 0
    for media_id in range(1, total + 1):
        version = None if media_id <= legacy else "parity-1@385afa95"
        for index in range(2):
            n += 1
            rows.append((n, media_id, f"c{n}", index, "words", f"u{n}", now, version))
    conn.executemany(
        "INSERT INTO UnvectorizedMediaChunks (id, media_id, chunk_text, "
        "chunk_index, chunk_type, uuid, last_modified, chunk_engine_version, "
        "deleted, version, client_id) VALUES (?,?,?,?,?,?,?,?,0,1,'test')",
        rows,
    )
    conn.commit()


class _RealBackendHost(ConsolidatedCSSApp):
    """Mount one Search/RAG panel over a real local RAG admin backend."""

    def __init__(self, scope_service: RAGAdminScopeService) -> None:
        super().__init__()
        self._scope_service = scope_service

    @property
    def rag_admin_scope_service(self) -> RAGAdminScopeService:
        return self._scope_service

    def compose(self) -> ComposeResult:
        with Vertical(id="canvas-slot"):
            yield LibrarySearchRagPanel(
                LibraryRagPanelState.from_values(
                    source_counts={"media": 3}, query="", mode="search"
                ),
                id="library-search-rag-panel",
            )


def _scope_service(db: MediaDatabase) -> RAGAdminScopeService:
    return RAGAdminScopeService(
        local_service=LocalRAGAdminService(db, chunking_service=object()),
        server_service=None,
    )


async def _settle(pilot, predicate, attempts: int = 120):
    for _ in range(attempts):
        if predicate():
            await pilot.pause()
            return True
        await pilot.pause()
    return False


@pytest.fixture()
def media_db(tmp_path):
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="test")
    yield db
    db.close_connection()


async def test_panel_renders_the_real_census_line(media_db):
    """End to end: the number on screen is the number in the database."""
    _seed(media_db, legacy=4, stamped=2)
    app = _RealBackendHost(_scope_service(media_db))
    async with app.run_test() as pilot:
        line = app.query_one(f"#{REPORT_LINE_ID}", Static)
        assert await _settle(pilot, lambda: line.display is True), str(line.renderable)
        assert str(line.renderable) == "Chunked by an older engine: 4 items"
        assert app.query_one(f"#{RECHUNK_BUTTON_ID}").display is True


async def test_panel_shows_nothing_for_a_fully_stamped_library(media_db):
    """The honest empty state -- no line, no re-chunk control, no zero."""
    _seed(media_db, legacy=0, stamped=3)
    app = _RealBackendHost(_scope_service(media_db))
    async with app.run_test() as pilot:
        for _ in range(30):
            await pilot.pause()
        assert app.query_one(f"#{REPORT_LINE_ID}", Static).display is False
        assert app.query_one(f"#{RECHUNK_BUTTON_ID}").display is False


async def test_panel_shows_nothing_on_a_first_run_empty_library(media_db):
    app = _RealBackendHost(_scope_service(media_db))
    async with app.run_test() as pilot:
        for _ in range(30):
            await pilot.pause()
        assert app.query_one(f"#{REPORT_LINE_ID}", Static).display is False


def _instrument_slow_census(service: RAGAdminScopeService, seconds: float):
    """Slow the real backend down and expose entry/exit as loop events.

    ``entered`` is set from the census's own thread the moment it starts,
    so a test can prove it interrupted a census that was genuinely IN
    FLIGHT -- and, because it is delivered via ``call_soon_threadsafe``, it
    can only be observed at all if the event loop is still running while
    the census runs. ``finished`` proves the query completed rather than
    being wedged by whatever the test did to the UI meanwhile.
    """
    local = service.local_service
    inner = local.get_template_diagnostics
    entered = asyncio.Event()
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_diagnostics():
        loop.call_soon_threadsafe(entered.set)
        time.sleep(seconds)
        try:
            return inner()
        finally:
            loop.call_soon_threadsafe(finished.set)

    local.get_template_diagnostics = slow_diagnostics
    return entered, finished


async def test_unmounting_mid_census_neither_raises_nor_paints(media_db):
    """Panel unmount while the worker thread is still in the census.

    The offload means the SELECT outlives the widget by design; what must
    not happen is a stalled UI, an exception reaching the app, or a dead
    widget being written to.

    The teeth here are the heartbeat: it counts loop ticks DURING the
    census, so this test also fails if the census ever goes back to
    running inline (that is the mutation it is checked against).
    """
    _seed(media_db, legacy=7, stamped=1)
    service = _scope_service(media_db)
    entered, finished = _instrument_slow_census(service, 0.30)

    ticks = 0
    stop = False

    async def heartbeat():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.01)

    app = _RealBackendHost(service)
    async with app.run_test() as pilot:
        beat = asyncio.create_task(heartbeat())
        await asyncio.wait_for(entered.wait(), timeout=5)
        ticks_at_entry = ticks
        panel = app.query_one("#library-search-rag-panel")
        await panel.remove()
        assert list(app.query("#library-search-rag-panel")) == []
        # The census completes on its own thread with no widget left.
        await asyncio.wait_for(finished.wait(), timeout=5)
        stop = True
        await beat
        assert ticks - ticks_at_entry >= 15, (
            f"the event loop stalled during the census: {ticks - ticks_at_entry} ticks"
        )
        assert list(app.query(f"#{REPORT_LINE_ID}")) == []


async def test_quitting_mid_census_exits_cleanly(media_db):
    """App quit while the census thread is running must not hang or raise."""
    _seed(media_db, legacy=2, stamped=2)
    service = _scope_service(media_db)
    entered, _finished = _instrument_slow_census(service, 0.30)

    app = _RealBackendHost(service)
    started = time.perf_counter()
    async with app.run_test() as pilot:
        await asyncio.wait_for(entered.wait(), timeout=5)
        app.exit()
    elapsed = time.perf_counter() - started
    assert app.return_value is None
    # The census must not have been awaited on the way out.
    assert elapsed < 5.0, elapsed
