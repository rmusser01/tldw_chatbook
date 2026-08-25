"""Pacing for the ChaChaNotes ``messages_fts`` backfill driver (task-22200).

task-21100 moved the v46 whole-history FTS reinsert off the boot path into
``DB/chachanotes_fts_backfill.py`` -- but the driver ran its ``BEGIN
IMMEDIATE`` chunks in a tight loop with no pause, so an upgrading user's whole
first session contended with a back-to-back write-lock convoy (the
holistic-perf review's finding 22200). task-22200 adds:

- a fixed inter-chunk pause, so the write lock's duty cycle is bounded and a
  foreground ``BEGIN IMMEDIATE`` writer acquires the lock inside the gaps
  instead of racing the next chunk's begin;
- bounded retry-with-backoff when a chunk itself dies on SQLite's plain
  lock-queue timeout (`database is locked` after the 15 s busy handler -- the
  RETRYABLE kind; the non-retryable snapshot-upgrade form only exists for
  DEFERRED read-then-write transactions, which the IMMEDIATE chunk is not),
  so one slow foreground transaction no longer kills the run until next boot;
- a ``should_abort`` seam checked between chunks and inside every sleep
  (sliced), so app shutdown cuts an in-flight pause instead of waiting it out.

The deterministic tests below pin each mechanism through an injected ``sleep``
recorder; ``test_ui_write_latency_stays_bounded_while_a_backfill_is_in_flight``
is the AC's behavioural probe -- a real concurrent writer against a real
in-flight backfill, with the in-flight-ness asserted so the probe cannot pass
vacuously.

Resumability invariants stay where task-21100 put them
(``Tests/DB/test_chachanotes_v47_messages_fts_backfill.py``, including the
real-SIGKILL form); this module re-witnesses only the abort path's frontier.

Fixture note: the backfill window is opened by seeding a CURRENT-schema DB
and then issuing the same ``'delete-all'`` the v46 migration performs, NOT by
replaying the real v45 upgrade like the v47 module does. Deliberate, twice
over: this module tests the driver's pacing against the window STATE (live
rows absent from ``messages_fts_docsize``), which the reset reproduces
exactly -- and at the time of writing the v45 bootstrap seeding path is a
pre-existing dev red anyway (``add_message`` now writes
``assistant_generation_state``, a v48 column, so every
``chachanotes_db_at_version(..., 45)`` fixture that seeds through it fails
with ``no column named assistant_generation_state``; see task-22200's notes).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tldw_chatbook.DB.chachanotes_fts_backfill import (
    _ABORT_POLL_SECONDS,
    _LOCKED_RETRY_BACKOFF_SECONDS,
    ChaChaNotesFTSBackfillError,
    backfill_chachanotes_messages_fts,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _open_with_backfill_window(
    db_path: Path, count: int, *, client_id: str
) -> CharactersRAGDB:
    """A current-schema DB holding ``count`` live messages, index cleared.

    Seeds in one outer transaction (so a few hundred rows stay cheap), then
    reproduces the post-v46-upgrade state with the migration's own reset:
    ``'delete-all'`` empties the index and its ``_docsize`` shadow table
    while every live row stays in ``messages`` -- the exact window the
    backfill driver exists to close.
    """
    db = CharactersRAGDB(db_path, client_id=client_id)
    with db.transaction(immediate=True):
        conversation_id = db.add_conversation(
            {"title": "pacing", "character_id": 1}
        )
        for i in range(count):
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": f"paceneedle{i:04d} body",
                }
            )
    with db.transaction(immediate=True) as conn:
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('delete-all')")
    assert _docsize_count(db) == 0  # the window is open
    return db


def _docsize_count(db: CharactersRAGDB) -> int:
    return db.execute_query(
        "SELECT COUNT(*) FROM messages_fts_docsize"
    ).fetchone()[0]


def _live_count(db: CharactersRAGDB) -> int:
    return db.execute_query(
        "SELECT COUNT(*) FROM messages WHERE deleted = 0"
    ).fetchone()[0]


@pytest.fixture
def upgraded_db(tmp_path: Path):
    """A DB with 10 live messages awaiting backfill (empty index)."""
    instance = _open_with_backfill_window(
        tmp_path / "chachanotes.db", 10, client_id="t22200-test"
    )
    yield instance
    instance.close_connection()


def test_backfill_sleeps_the_configured_pause_between_chunks(upgraded_db):
    """The load-bearing line: every data chunk is followed by exactly one
    pause (10 rows / chunk 4 -> chunks of 4, 4, 2 -> 3 pauses; the terminal
    0-row probe ends the run without sleeping again)."""
    recorded: list[float] = []

    total = backfill_chachanotes_messages_fts(
        upgraded_db, chunk_size=4, pause_seconds=0.25, sleep=recorded.append
    )

    assert total == 10
    assert recorded == [0.25, 0.25, 0.25]
    assert _docsize_count(upgraded_db) == _live_count(upgraded_db)


def test_the_no_op_boot_probe_never_sleeps(upgraded_db):
    """AC #4: the every-boot call on a complete index stays one indexed scan
    -- and must not pick up a pacing sleep either."""
    assert backfill_chachanotes_messages_fts(upgraded_db, chunk_size=4) == 10

    recorded: list[float] = []
    assert (
        backfill_chachanotes_messages_fts(
            upgraded_db, chunk_size=4, pause_seconds=0.25, sleep=recorded.append
        )
        == 0
    )
    assert recorded == []


def test_abort_between_chunks_leaves_the_resumable_frontier(upgraded_db):
    """Stopping is not failing: an abort after the first chunk returns the
    partial count, leaves ``messages_fts_docsize`` as the frontier, and a
    later un-aborted run converges (task-21100's resumability, AC #3)."""
    chunks_done = 0
    original = CharactersRAGDB.backfill_messages_fts

    def counting(self, *args, **kwargs):
        nonlocal chunks_done
        result = original(self, *args, **kwargs)
        chunks_done += 1
        return result

    with patch.object(CharactersRAGDB, "backfill_messages_fts", counting):
        total = backfill_chachanotes_messages_fts(
            upgraded_db,
            chunk_size=4,
            pause_seconds=0.0,
            should_abort=lambda: chunks_done >= 1,
        )

    assert total == 4
    assert _docsize_count(upgraded_db) == 4
    # The frontier is in the database; a fresh run finishes the job.
    assert backfill_chachanotes_messages_fts(upgraded_db, chunk_size=4) == 6
    assert _docsize_count(upgraded_db) == _live_count(upgraded_db)


def test_abort_cuts_an_in_flight_pause_at_the_poll_slice(upgraded_db):
    """A ``stop()`` that only sets flags cannot cut a plain ``time.sleep``
    (documented lesson): with ``should_abort`` provided, the pause must be
    sliced into abort-checked increments so shutdown waits one slice, not the
    whole pause."""
    recorded: list[float] = []
    aborted = {"flag": False}

    def sliced_sleep(seconds: float) -> None:
        recorded.append(seconds)
        if len(recorded) >= 3:
            aborted["flag"] = True  # "shutdown" arrives mid-pause

    total = backfill_chachanotes_messages_fts(
        upgraded_db,
        chunk_size=4,
        pause_seconds=5.0,
        should_abort=lambda: aborted["flag"],
        sleep=sliced_sleep,
    )

    # One chunk committed, then the pause began and was cut at the third
    # slice -- never anywhere near the full 5 s.
    assert total == 4
    assert recorded == [_ABORT_POLL_SECONDS] * 3
    assert sum(recorded) < 5.0
    assert _docsize_count(upgraded_db) == 4


def test_a_locked_chunk_is_retried_with_backoff_and_the_run_completes(
    upgraded_db,
):
    """Contention-aware backoff (AC #1's second half): one foreground
    transaction outliving the chunk's 15 s busy handler must cost a backoff
    sleep and a retry, not the rest of the run."""
    attempts = {"n": 0}
    recorded: list[float] = []
    original = CharactersRAGDB.backfill_messages_fts

    def locked_once(self, *args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(self, *args, **kwargs)

    with patch.object(CharactersRAGDB, "backfill_messages_fts", locked_once):
        total = backfill_chachanotes_messages_fts(
            upgraded_db, chunk_size=10, pause_seconds=0.0, sleep=recorded.append
        )

    assert total == 10
    assert recorded[0] == _LOCKED_RETRY_BACKOFF_SECONDS[0]
    assert _docsize_count(upgraded_db) == _live_count(upgraded_db)


def test_locked_retries_are_bounded_then_wrapped_with_the_partial_count(
    upgraded_db,
):
    """Permanent contention still ends the run the pre-existing way: wrapped,
    with the rows this run really committed, after the whole (bounded)
    backoff schedule."""
    attempts = {"n": 0}
    recorded: list[float] = []
    original = CharactersRAGDB.backfill_messages_fts

    def locked_after_one_chunk(self, *args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return original(self, *args, **kwargs)
        raise sqlite3.OperationalError("database is locked")

    with patch.object(
        CharactersRAGDB, "backfill_messages_fts", locked_after_one_chunk
    ):
        with pytest.raises(ChaChaNotesFTSBackfillError) as excinfo:
            backfill_chachanotes_messages_fts(
                upgraded_db,
                chunk_size=4,
                pause_seconds=0.0,
                sleep=recorded.append,
            )

    assert excinfo.value.rows_indexed == 4
    # One good chunk, then 1 failing call + one retry per backoff step.
    assert attempts["n"] == 2 + len(_LOCKED_RETRY_BACKOFF_SECONDS)
    assert recorded == list(_LOCKED_RETRY_BACKOFF_SECONDS)


def test_a_non_lock_error_still_fails_fast(upgraded_db):
    """The backoff is for lock-queue timeouts only -- any other failure keeps
    task-21100's fail-fast-and-wrap contract, with zero sleeps."""
    recorded: list[float] = []

    def broken(self, *args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    with patch.object(CharactersRAGDB, "backfill_messages_fts", broken):
        with pytest.raises(ChaChaNotesFTSBackfillError) as excinfo:
            backfill_chachanotes_messages_fts(
                upgraded_db, pause_seconds=0.0, sleep=recorded.append
            )

    assert excinfo.value.rows_indexed == 0
    assert recorded == []


#: The stated bound for the AC's latency probe. A foreground ``add_message``
#: colliding with the paced backfill waits at most one in-flight chunk (a few
#: ms of tokenize+write for 8 tiny rows) plus scheduler noise; 2 s is ~100x
#: the locally measured worst case (see the task file's measurements) while
#: staying far below the 15 s busy timeout an unpaced convoy can consume.
UI_WRITE_LATENCY_BOUND_SECONDS = 2.0


def test_ui_write_latency_stays_bounded_while_a_backfill_is_in_flight(
    tmp_path: Path,
):
    """AC #1's probe: a concurrent writer against an in-flight backfill.

    Production shape end to end: ONE shared ``CharactersRAGDB`` (thread-local
    connections), the real driver with its default pacing on a worker thread,
    and the real ``add_message`` (``BEGIN IMMEDIATE``) from the foreground.
    240 rows / chunk 8 = 30 chunks at >=0.05 s pause gives the backfill a
    >=1.5 s floor, so the five foreground writes (~0.3 s) provably land
    mid-run -- asserted via ``thread.is_alive()`` after the last write, which
    is what keeps this probe from passing vacuously against a finished run.
    """
    db = _open_with_backfill_window(
        tmp_path / "chachanotes.db", 240, client_id="t22200-probe"
    )
    try:
        conversation_id = db.add_conversation(
            {"title": "latency probe", "character_id": 1}
        )

        failures: list[BaseException] = []

        def run_backfill() -> None:
            try:
                backfill_chachanotes_messages_fts(
                    db, chunk_size=8, pause_seconds=0.05
                )
            except BaseException as exc:  # pragma: no cover - failure detail
                failures.append(exc)

        backfill_thread = threading.Thread(target=run_backfill, daemon=True)
        backfill_thread.start()

        # Wait for the run to be genuinely in flight (first chunk committed).
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _docsize_count(db) == 0:
            time.sleep(0.005)
        assert _docsize_count(db) > 0, "backfill never started"

        latencies: list[float] = []
        for i in range(5):
            started = time.perf_counter()
            message_id = db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": f"foreground write {i} during the window",
                }
            )
            latencies.append(time.perf_counter() - started)
            assert message_id, "a foreground write failed during the window"
            time.sleep(0.04)

        assert backfill_thread.is_alive(), (
            "backfill finished before the writes -- probe was vacuous; "
            f"latencies so far: {latencies}"
        )
        assert max(latencies) < UI_WRITE_LATENCY_BOUND_SECONDS, latencies

        backfill_thread.join(timeout=60.0)
        assert not backfill_thread.is_alive()
        assert failures == []
        # Everything converged: 240 backfilled + 5 trigger-indexed writes.
        assert _docsize_count(db) == _live_count(db) == 245
    finally:
        db.close_connection()
