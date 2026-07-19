"""Idle-gated `SELECT 1` liveness ping (task-261).

`_get_thread_connection` in CharactersRAGDB / MediaDatabase / PromptsDatabase
used to run a `SELECT 1` liveness ping on EVERY `get_connection()` call,
roughly doubling raw statement counts on query-heavy paths. The ping is now
gated behind `_LIVENESS_PING_IDLE_SECONDS`. These tests run against real
on-disk SQLite databases (no mocks; the seam is the DB) and use sqlite3's
trace callback to count the pings actually executed, proving:

- a burst of recently-used calls issues zero pings;
- a connection idle past the threshold is pinged exactly once, then the
  window resets;
- the transparent reopen of a dead connection (the ping's whole purpose)
  still works.
"""

import time

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

DB_FACTORIES = [
    pytest.param(
        lambda tmp: CharactersRAGDB(tmp / "chachanotes.db", "ping-gate-test"),
        id="chachanotes",
    ),
    pytest.param(
        lambda tmp: MediaDatabase(tmp / "media.db", "ping-gate-test"),
        id="media",
    ),
    pytest.param(
        lambda tmp: PromptsDatabase(tmp / "prompts.db", "ping-gate-test"),
        id="prompts",
    ),
]


def _trace_pings(conn) -> list:
    """Attach a trace callback recording every `SELECT 1` liveness ping."""
    statements: list = []
    conn.set_trace_callback(statements.append)
    return statements


def _ping_count(statements: list) -> int:
    return sum(1 for statement in statements if statement.strip() == "SELECT 1")


def _mark_idle(db) -> None:
    """Backdate the connection's last-used stamp past the ping threshold."""
    db._local.conn_last_used = time.monotonic() - (
        type(db)._LIVENESS_PING_IDLE_SECONDS + 1.0
    )


@pytest.mark.parametrize("factory", DB_FACTORIES)
def test_recently_used_connection_burst_skips_the_ping(tmp_path, factory):
    db = factory(tmp_path)
    try:
        conn = db.get_connection()
        statements = _trace_pings(conn)

        for _ in range(10):
            db.get_connection().execute(
                "SELECT count(*) FROM sqlite_master"
            ).fetchone()

        assert _ping_count(statements) == 0, (
            "a recently-used connection must not be pinged per call"
        )
    finally:
        db.close_connection()


@pytest.mark.parametrize("factory", DB_FACTORIES)
def test_idle_connection_is_pinged_once_then_window_resets(tmp_path, factory):
    db = factory(tmp_path)
    try:
        conn = db.get_connection()
        statements = _trace_pings(conn)

        _mark_idle(db)
        assert db.get_connection() is conn
        assert _ping_count(statements) == 1, (
            "an idle connection must get exactly one liveness ping"
        )

        # The successful call refreshed the window: no further pings.
        db.get_connection().execute("SELECT count(*) FROM sqlite_master").fetchone()
        assert _ping_count(statements) == 1
    finally:
        db.close_connection()


@pytest.mark.parametrize("factory", DB_FACTORIES)
def test_dead_idle_connection_is_transparently_reopened(tmp_path, factory):
    db = factory(tmp_path)
    try:
        dead = db.get_connection()
        # Kill the connection behind the manager's back, then age it past the
        # threshold so the gate re-verifies it.
        dead.close()
        _mark_idle(db)

        fresh = db.get_connection()

        assert fresh is not dead
        row = fresh.execute("SELECT count(*) FROM sqlite_master").fetchone()
        assert row[0] >= 0
    finally:
        db.close_connection()
