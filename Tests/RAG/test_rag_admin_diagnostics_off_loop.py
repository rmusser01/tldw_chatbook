"""TASK-21126: the legacy-chunk census must not run on the event loop.

``RAGAdminScopeService._maybe_await(service.get_template_diagnostics())``
evaluated its argument before the first suspension point, so the LOCAL
backend's synchronous census SELECT -- measured 119 ms at 200k live chunk
rows and 701 ms at 1M -- ran to completion on the event loop, once per
Library Search/RAG panel show.

Every test here runs against the REAL ``LocalRAGAdminService`` over a real
file-backed ``MediaDatabase``. That is deliberate: a synchronous test double
makes an "it runs off the loop" assertion pass while moving zero work (the
trap recorded in ``lessons-testing-evidence.md``), and only the real service
can also show that the work still HAPPENS and the answer is still right.

Mutation-checked: reverting ``get_template_diagnostics`` to
``await self._maybe_await(service.get_template_diagnostics())`` turns
``test_census_runs_on_a_worker_thread`` and
``test_event_loop_keeps_running_during_a_slow_census`` red.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService
from tldw_chatbook.RAG_Admin.rag_admin_scope_service import RAGAdminScopeService

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
            rows.append(
                (n, media_id, f"c{n}", index, "words", f"u{n}", now, version)
            )
    conn.executemany(
        "INSERT INTO UnvectorizedMediaChunks (id, media_id, chunk_text, "
        "chunk_index, chunk_type, uuid, last_modified, chunk_engine_version, "
        "deleted, version, client_id) VALUES (?,?,?,?,?,?,?,?,0,1,'test')",
        rows,
    )
    conn.commit()


def _service(db: MediaDatabase) -> RAGAdminScopeService:
    """A real scope service over a real local backend.

    ``chunking_service`` is stubbed to a sentinel only so
    ``_require_chunking_service`` succeeds without importing the ~15k-LOC
    chunking engine; nothing in the diagnostics path calls it.
    """
    local = LocalRAGAdminService(db, chunking_service=object())
    return RAGAdminScopeService(local_service=local, server_service=None)


@pytest.fixture()
def media_db(tmp_path):
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="test")
    yield db
    db.close_connection()


async def test_census_runs_on_a_worker_thread(media_db):
    """The SELECT executes off the loop thread -- and still executes."""
    _seed(media_db, legacy=3, stamped=4)
    threads: list[int] = []
    real_get_connection = media_db.get_connection

    def recording_get_connection():
        threads.append(threading.get_ident())
        return real_get_connection()

    # Shadow on the INSTANCE, not the class: it catches the callee that
    # reaches this object by any route.
    media_db.get_connection = recording_get_connection
    loop_thread = threading.get_ident()

    payload = await _service(media_db).get_template_diagnostics(mode="local")

    assert threads, "the census never touched the media DB at all"
    assert loop_thread not in threads, (
        "the census ran on the event-loop thread: " f"{threads} vs {loop_thread}"
    )
    # The work still happened AND produced the right answer.
    assert payload["legacy_chunk_report"] == "Chunked by an older engine: 3 items"


async def test_event_loop_keeps_running_during_a_slow_census(media_db):
    """A slow backend must not starve the loop.

    The census is deliberately slowed with a real ``time.sleep`` inside the
    synchronous backend method, because that is exactly what a large media
    DB does: blocking CPU/IO in a sync call. A heartbeat coroutine counts
    its own ticks while the fetch is in flight.
    """
    _seed(media_db, legacy=1, stamped=1)
    service = _service(media_db)
    local = service.local_service
    inner = local.get_template_diagnostics

    def slow_diagnostics():
        time.sleep(0.30)
        return inner()

    local.get_template_diagnostics = slow_diagnostics

    ticks = 0
    stop = False

    async def heartbeat():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.01)

    beat = asyncio.create_task(heartbeat())
    started = time.perf_counter()
    payload = await service.get_template_diagnostics(mode="local")
    elapsed = time.perf_counter() - started
    stop = True
    await beat

    assert elapsed >= 0.30, "the slow backend was not actually invoked"
    # A blocked loop yields ~1 tick (the one before the call). 30 slots of
    # 10 ms fit in 300 ms; require a clear majority to survive scheduler
    # jitter on a loaded machine.
    assert ticks >= 15, f"the event loop stalled: only {ticks} heartbeat ticks"
    assert payload["legacy_chunk_report"] == "Chunked by an older engine: 1 items"


async def test_memory_backed_db_still_reports_the_real_counts():
    """`:memory:` stays on the calling thread -- never a silent zero.

    ``MediaDatabase`` connections are thread-local, so a worker thread
    opening ``:memory:`` gets a DIFFERENT, EMPTY database and the census
    would report nothing at all. This is the interleaving hazard the
    offload could have introduced; the backend declares itself unsafe and
    the scope service runs it inline instead.
    """
    db = MediaDatabase(":memory:", client_id="test")
    try:
        _seed(db, legacy=2, stamped=1)
        local = LocalRAGAdminService(db, chunking_service=object())
        assert local.diagnostics_are_thread_safe() is False
        payload = await RAGAdminScopeService(
            local_service=local, server_service=None
        ).get_template_diagnostics(mode="local")
        assert payload["legacy_chunk_report"] == "Chunked by an older engine: 2 items"
    finally:
        db.close_connection()


async def test_file_backed_db_declares_itself_thread_safe(media_db):
    assert LocalRAGAdminService(
        media_db, chunking_service=object()
    ).diagnostics_are_thread_safe() is True


async def test_no_media_db_is_thread_safe_and_reports_nothing():
    local = LocalRAGAdminService(None, chunking_service=object())
    assert local.diagnostics_are_thread_safe() is True
    payload = await RAGAdminScopeService(
        local_service=local, server_service=None
    ).get_template_diagnostics(mode="local")
    assert "legacy_chunk_report" not in payload
    assert payload["backend"] == "local"


async def test_a_fully_stamped_library_reports_nothing(media_db):
    """Empty state: nothing older-engine, so no line and no button."""
    _seed(media_db, legacy=0, stamped=5)
    payload = await _service(media_db).get_template_diagnostics(mode="local")
    assert "legacy_chunk_report" not in payload


async def test_an_empty_library_reports_nothing(media_db):
    """First run: no media at all."""
    payload = await _service(media_db).get_template_diagnostics(mode="local")
    assert "legacy_chunk_report" not in payload


async def test_a_failing_census_still_yields_a_usable_payload(media_db):
    """Error path: the census raising must not lose the diagnostics dict.

    ``get_template_diagnostics`` already guards the report; what this pins
    is that the guard still works when the call is made from a worker
    thread (``to_thread`` re-raises into the awaiting coroutine, so a
    swallowed-there exception would surface here as a raise).
    """
    _seed(media_db, legacy=2, stamped=0)
    local = _service(media_db).local_service

    def exploding(_db):
        raise RuntimeError("no such column: chunk_engine_version")

    local.count_chunks_by_engine_version = exploding
    payload = await RAGAdminScopeService(
        local_service=local, server_service=None
    ).get_template_diagnostics(mode="local")
    assert "legacy_chunk_report" not in payload
    assert payload["capability"] == "native"


async def test_a_raising_backend_propagates_unchanged(media_db):
    """A failure OUTSIDE the guarded report still reaches the caller."""
    local = _service(media_db).local_service

    def exploding():
        raise ValueError("Local chunking template backend is unavailable.")

    local.get_template_diagnostics = exploding
    with pytest.raises(ValueError, match="unavailable"):
        await RAGAdminScopeService(
            local_service=local, server_service=None
        ).get_template_diagnostics(mode="local")


async def test_concurrent_censuses_never_tear_against_a_live_writer(media_db):
    """Interleaving: the offload puts reads and writes on different threads.

    Before this change the census could not overlap anything -- it held the
    loop. Now twelve of them run on executor threads while the loop thread
    inserts new legacy media between batches. Under WAL each reader takes a
    consistent snapshot, so every answer must be a count the database
    genuinely held at some instant: never a torn value, never a lock error.
    """
    _seed(media_db, legacy=4, stamped=1)
    service = _service(media_db)
    conn = media_db.get_connection()
    now = "2026-08-23T00:00:00Z"

    async def census() -> str:
        payload = await service.get_template_diagnostics(mode="local")
        return payload.get("legacy_chunk_report", "")

    def add_legacy_media(media_id: int) -> None:
        conn.execute(
            "INSERT INTO Media (id, title, type, content, content_hash, uuid, "
            "last_modified, version, client_id, deleted) "
            "VALUES (?,?,?,?,?,?,?,1,'test',0)",
            (media_id, f"doc {media_id}", "document", "b", f"h{media_id}",
             f"m{media_id}", now),
        )
        conn.execute(
            "INSERT INTO UnvectorizedMediaChunks (id, media_id, chunk_text, "
            "chunk_index, chunk_type, uuid, last_modified, "
            "chunk_engine_version, deleted, version, client_id) "
            "VALUES (?,?,?,0,'words',?,?,NULL,0,1,'test')",
            (1000 + media_id, media_id, "c", f"u{1000 + media_id}", now),
        )
        conn.commit()

    results: list[str] = []
    for round_index in range(4):
        pending = asyncio.gather(*(census() for _ in range(3)))
        add_legacy_media(100 + round_index)
        results.extend(await pending)

    counts = [int(line.split(": ")[1].split(" ")[0]) for line in results]
    # Only counts the database genuinely held (4 at seed, +1 per round).
    assert set(counts) <= {4, 5, 6, 7, 8}, counts
    # A count can never go backwards, and the writer must actually have
    # raced -- a run that saw one value throughout proved nothing.
    assert counts == sorted(counts), counts
    assert len(set(counts)) >= 2, counts


async def test_an_unknown_backend_is_not_moved_off_the_loop():
    """Opt-in only: a backend that does not declare safety runs inline.

    Guards the blast radius of the change -- a test double, a Mock, or any
    future sync backend keeps the pre-existing behaviour exactly.
    """
    calls: list[int] = []

    class _Unknown:
        def get_template_diagnostics(self):
            calls.append(threading.get_ident())
            return {"capability": "stub"}

    loop_thread = threading.get_ident()
    payload = await RAGAdminScopeService(
        local_service=_Unknown(), server_service=None
    ).get_template_diagnostics(mode="local")
    assert calls == [loop_thread]
    assert payload == {"capability": "stub", "backend": "local"}


async def test_an_async_backend_is_awaited_not_threaded():
    calls: list[int] = []

    class _AsyncBackend:
        async def get_template_diagnostics(self):
            calls.append(threading.get_ident())
            return {"capability": "server"}

    loop_thread = threading.get_ident()
    payload = await RAGAdminScopeService(
        local_service=None, server_service=_AsyncBackend()
    ).get_template_diagnostics(mode="server")
    assert calls == [loop_thread]
    assert payload == {"capability": "server", "backend": "server"}
