"""TASK-21127: the four lifecycle paths a connection refactor keeps breaking.

The held-connection change is exercised elsewhere on happy paths; this file
walks the ones that have historically broken in this burn-down -- quit with work
in flight, a database error mid-run, a cancelled run, and the empty/first-run
case -- against a FILE-BACKED store, which is the shape production uses and the
one the engine tests (``:memory:``) never touch.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from tldw_chatbook.Research_Interop.local_research_engine import LocalResearchEngine
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService


def _pipeline(question: str):
    """The two injectable seams, faked; everything else runs real."""

    def search_fn(query, params):
        return (
            {
                "results": [
                    {"title": "One", "url": "https://one.example/"},
                    {"title": "Two", "url": "https://two.example/"},
                ],
                "warnings": [],
            },
            {"sub_questions": ["sub q1"], "main_goal": question},
        )

    async def analyze_fn(wsr, sqd, params, cancel_event=None):
        return {
            "final_answer": {
                "text": "Answer citing [1].",
                "evidence": [
                    {
                        "id": 1,
                        "url": "https://one.example/",
                        "title": "One",
                        "content": "c1",
                        "original_content": "o1",
                        "reasoning": "r1",
                        "chunk_index": 1,
                    }
                ],
                "confidence": 0.8,
                "chunks": [],
                "citation_verification": {
                    "markers_total": 1,
                    "markers_resolved": 1,
                    "unknown_marker_ids": [],
                    "quotes_checked": 0,
                    "quotes_verified": 0,
                    "quotes_misquoted": 0,
                    "uncited_sentences": 0,
                },
            },
            "relevant_results": {"1": {}},
            "web_search_results_dict": wsr,
        }

    return search_fn, analyze_fn


# --- 1. quit while a run is in flight ---------------------------------------


@pytest.mark.asyncio
async def test_close_during_a_live_run_lets_the_run_finish(tmp_path):
    """Shutdown must not hand the engine a closed database mid-run.

    ``close()`` re-arms the store, so the run's next operation reopens
    transparently instead of raising ``ProgrammingError: Cannot operate on a
    closed database`` -- the TASK-21101 signature this whole lifecycle gate
    exists to prevent.
    """
    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="quit mid-run", autonomy_mode="autonomous")
    search_fn, analyze_fn = _pipeline(run["query"])
    closed = threading.Event()

    def _closing_search(query, params):
        # Fires from the engine's worker thread, mid-run, the way a user
        # quitting the app would.
        closer = threading.Thread(target=service.close)
        closer.start()
        closer.join(10)
        closed.set()
        return search_fn(query, params)

    engine = LocalResearchEngine(
        service, search_fn=_closing_search, analyze_fn=analyze_fn
    )
    final = await engine.execute_run(run["id"])

    assert closed.is_set(), "the probe never closed the store"
    assert final["status"] == "completed", final.get("error_msg")
    assert service.get_run(run["id"])["status"] == "completed"
    service.close()


# --- 2. a database error mid-run --------------------------------------------


@pytest.mark.asyncio
async def test_a_database_error_mid_run_fails_the_run_and_heals_the_store(tmp_path):
    """A failing DB op must fail the RUN legibly, not poison the connection.

    A transaction abandoned mid-flight would otherwise leave every later
    operation on that thread failing with "cannot start a transaction within a
    transaction".
    """
    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="db error", autonomy_mode="autonomous")
    search_fn, analyze_fn = _pipeline(run["query"])

    real_save = LocalResearchService.save_artifact
    calls = {"n": 0}

    def _exploding_save(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("disk I/O error (injected)")
        return real_save(self, *args, **kwargs)

    LocalResearchService.save_artifact = _exploding_save
    try:
        engine = LocalResearchEngine(
            service, search_fn=search_fn, analyze_fn=analyze_fn
        )
        final = await engine.execute_run(run["id"])
    finally:
        LocalResearchService.save_artifact = real_save

    assert final["status"] == "failed"
    # ``fail_run`` records the message on ``progress_message`` -- the runs table
    # has no error_msg column.
    assert "injected" in str(final.get("progress_message") or "")
    # The store is still usable: the abandoned transaction was rolled back.
    later = service.launch_run(query="after the error")
    assert service.get_run(later["id"])["query"] == "after the error"
    assert service.list_run_events(run["id"])
    service.close()


# --- 3. a cancelled run ------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_between_phases_resolves_once_on_a_file_backed_store(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="cancel me", autonomy_mode="autonomous")
    search_fn, analyze_fn = _pipeline(run["query"])

    def _cancelling_search(query, params):
        service.update_run_progress(run["id"], control_state="cancel_requested")
        return search_fn(query, params)

    engine = LocalResearchEngine(
        service, search_fn=_cancelling_search, analyze_fn=analyze_fn
    )
    final = await engine.execute_run(run["id"])

    assert final["status"] == "cancelled"
    stored = service.get_run(run["id"])
    assert stored["status"] == "cancelled"
    events = [event["event"] for event in service.list_run_events(run["id"])]
    assert events.count("cancelled") == 1, events
    service.close()


# --- 4. the empty / first-run case ------------------------------------------


def test_a_never_used_store_creates_no_file_and_lists_nothing(tmp_path):
    """TASK-21105's lazy-open contract must survive the held-connection change:
    construction resolves the path only."""
    db_path = tmp_path / "research.db"
    service = LocalResearchService(db_path)

    assert not db_path.exists(), "construction created the database file"
    assert list(service.list_runs()) == []
    assert service.list_sessions() == []
    assert db_path.exists(), "first use did not create the database"
    service.close()


def test_close_before_any_use_is_a_no_op(tmp_path):
    db_path = tmp_path / "research.db"
    service = LocalResearchService(db_path)
    service.close()
    assert not db_path.exists()
    # ... and the store still works afterwards.
    assert list(service.list_runs()) == []
    service.close()


@pytest.mark.asyncio
async def test_two_close_calls_race_without_deadlocking(tmp_path):
    """A second closer must wait out the first, not double-close."""
    service = LocalResearchService(tmp_path / "research.db")
    service.launch_run(query="double close")

    await asyncio.gather(
        asyncio.to_thread(service.close), asyncio.to_thread(service.close)
    )
    assert len(list(service.list_runs())) == 1
    service.close()


def test_close_releases_every_connection_including_the_schema_one(tmp_path):
    """``_init_schema`` used to open a connection and leak it in file mode.

    The oracle is the ``-wal`` sidecar: SQLite checkpoints and removes it only
    when the LAST connection to the database closes, so a surviving schema
    connection leaves it on disk. Asserting on the held-connection map alone
    could not see that connection at all.
    """
    db_path = tmp_path / "research.db"
    service = LocalResearchService(db_path)
    service.launch_run(query="wal witness")
    wal = db_path.with_name(db_path.name + "-wal")
    assert wal.exists(), "the store is not in WAL mode; the oracle is vacuous"

    service.close()

    assert not wal.exists(), (
        "a connection to the research database outlived close(): the -wal "
        "sidecar was not checkpointed away"
    )


def test_a_second_store_on_the_same_file_sees_the_first_stores_writes(tmp_path):
    """Held connections must not hide committed data from another opener."""
    db_path = tmp_path / "research.db"
    writer = LocalResearchService(db_path)
    run = writer.launch_run(query="cross-connection")

    reader = LocalResearchService(db_path)
    assert reader.get_run(run["id"])["query"] == "cross-connection"

    writer.update_run_progress(run["id"], progress_message="second write")
    assert reader.get_run(run["id"])["progress_message"] == "second write"
    writer.close()
    reader.close()


def test_a_held_connection_closed_behind_the_stores_back_self_heals(tmp_path):
    """``_begin``'s ProgrammingError arm, exercised directly.

    ``close()`` normally POPS what it closes, so a later operation opens a fresh
    connection and never meets a closed handle -- which is why the end-to-end
    quit-mid-run walk stays green even with this arm removed. The arm is the
    second line of defence, and this is the only scenario that reaches it: a
    connection closed while it is still mapped.
    """
    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="stale handle")

    stale = service._connect()
    stale.close()  # closed, but deliberately left in _connections
    assert service._connections[threading.get_ident()] is stale

    recovered = service.get_run(run["id"])

    assert recovered["query"] == "stale handle"
    assert service._connections[threading.get_ident()] is not stale
    service.close()


def test_a_failed_transaction_leaves_no_open_transaction(tmp_path):
    """The rollback must happen AT the failure, not be deferred to the next
    ``_begin``'s heal: a connection left mid-transaction holds the write lock
    against every other opener of the file until something else touches it."""
    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="rollback now")

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with service._transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE research_runs SET progress_message = ? WHERE id = ?",
                ("partial", run["id"]),
            )
            raise _Boom("mid-transaction failure")

    held = service._connections[threading.get_ident()]
    assert held.in_transaction is False, (
        "the failed transaction was left open; the write lock is still held"
    )
    assert service.get_run(run["id"])["progress_message"] != "partial"
    service.close()


def test_a_nested_immediate_transaction_inside_a_deferred_one_is_refused(tmp_path):
    """Joining a DEFERRED outer would silently downgrade a nested immediate.

    SQLite cannot upgrade a transaction's lock once it has begun, so the nested
    read-then-write would run under the outer's read snapshot and fail
    BUSY_SNAPSHOT -- only ever under contention, so it would pass every
    single-threaded test and then appear in the field as "database is locked",
    the one failure SQLite's busy handler does NOT retry. Refusing makes it
    deterministic and names the mistake.
    """
    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="nesting")

    with pytest.raises(RuntimeError, match="IMMEDIATE"):
        with service._transaction():  # deferred outer
            with service._transaction(immediate=True):  # must be refused
                pass  # pragma: no cover - the guard fires first

    # The outer transaction unwound cleanly: the store is still usable and no
    # transaction was left open on this thread's connection.
    assert service._connections[threading.get_ident()].in_transaction is False
    assert service.get_run(run["id"])["query"] == "nesting"
    service.close()


def test_the_legal_transaction_nestings_are_still_allowed(tmp_path):
    """The guard must refuse ONLY immediate-inside-deferred.

    Deferred-inside-immediate is the shipped hot path -- ``_require_one`` runs
    inside ``save_artifact``'s write transaction on every artifact the engine
    saves -- so over-restricting here would break real behaviour rather than
    protect it.
    """
    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="legal nesting")

    with service._transaction(immediate=True):  # deferred inside immediate
        assert service._require_one("research_runs", run["id"], "run")
        with service._transaction():
            pass
        with service._transaction(immediate=True):  # immediate inside immediate
            pass
    with service._transaction():  # deferred inside deferred
        with service._transaction():
            pass

    # And the real call graph that relies on it still works end to end.
    saved = service.save_artifact(
        run["id"],
        artifact_name="nested.json",
        content_type="application/json",
        content={"ok": True},
    )
    assert saved["content"] == {"ok": True}
    service.close()
