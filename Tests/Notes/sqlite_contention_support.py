"""Force SQLite read-to-write contention on a shared ChaChaNotes handle.

TASK-34000.28 and TASK-34000.84. A DEFERRED transaction that reads and then
writes must upgrade its read snapshot to a write lock. SQLite refuses that
upgrade at once, without asking the busy handler:

- with SQLITE_BUSY_SNAPSHOT (517) when another connection committed, or
  restarted the WAL (any finite worker's ``close_connection`` runs
  ``PRAGMA wal_checkpoint(TRUNCATE)``), after the read;
- with SQLITE_BUSY (5) while another connection still holds the write lock.

A transaction that takes the write lock first (``BEGIN IMMEDIATE``) holds the
other connection off instead, or waits for it on the busy timeout. Each
context manager below forces one of those orders on this thread's connection
and yields a list that records what the other connection saw.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


def _statement_kind(sql: str) -> str | None:
    """``read``, ``write`` or ``end`` for one traced statement, else None.

    FTS5 runs its own statements on the connection (``'main'.'notes_fts_config'``
    and the like); they are not the transaction's reads.
    """
    if "'main'." in sql:
        return None
    words = sql.lstrip().upper().split(None, 2)
    if not words:
        return None
    if words[0] in {"SELECT", "WITH"}:
        return "read"
    if words[0] in {"INSERT", "UPDATE", "DELETE", "REPLACE"}:
        return "write"
    if words[0] in {"COMMIT", "END"} or words == ["ROLLBACK"]:
        return "end"
    return None


@contextmanager
def _between_read_and_write(
    db: Any, act: Callable[[], object]
) -> Iterator[list[object]]:
    """Run ``act`` once, when this thread's transaction first writes after a read.

    The trace callback fires as a statement starts, before it takes any lock,
    so ``act`` runs after the transaction's read snapshot and before its first
    write asks to upgrade it.
    """
    connection = db.get_connection()
    outcomes: list[object] = []
    read_seen = False

    def on_statement(sql: str) -> None:
        nonlocal read_seen
        if outcomes:
            return
        kind = _statement_kind(sql)
        if kind == "end":
            read_seen = False
        elif kind == "read" and connection.in_transaction:
            read_seen = True
        elif kind == "write" and read_seen and connection.in_transaction:
            outcomes.append(act())

    connection.set_trace_callback(on_statement)
    try:
        yield outcomes
    finally:
        connection.set_trace_callback(None)


@contextmanager
def wal_restarted_between_read_and_write(db: Any) -> Iterator[list[object]]:
    """Restart the WAL from a second connection between a read and a write.

    N-24's measured cause (TASK-34000.28): another connection's close-time
    ``PRAGMA wal_checkpoint(TRUNCATE)``, no other writer needed. The
    checkpointer waits on nothing (``timeout=0``), so against a transaction
    that already holds the write lock it falls back to a passive checkpoint:
    the yielded result row then starts with 1 (busy).
    """
    checkpointer = sqlite3.connect(db.db_path_str, timeout=0, isolation_level=None)
    # Every frame is backfilled first, so the next reader reads the database
    # file alone -- exactly the state a just-closed worker leaves behind.
    checkpointer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    try:
        with _between_read_and_write(
            db,
            lambda: tuple(
                checkpointer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            ),
        ) as restarts:
            yield restarts
    finally:
        checkpointer.close()


@contextmanager
def write_committed_between_read_and_write(db: Any) -> Iterator[list[object]]:
    """Commit another connection's write, then restart the WAL, between a
    read and a write: both ways another connection makes a read snapshot
    stale (SQLITE_BUSY_SNAPSHOT, 517).

    The commit is TASK-42412's captured order; the restart is TASK-34000.28's
    (a worker's close-time ``wal_checkpoint(TRUNCATE)``). The other connection
    waits on nothing (``timeout=0``). It yields one ``(commit, checkpoint)``
    pair: ``"committed"`` or SQLite's ``database is locked`` error, and the
    checkpoint's result row, whose first column is 1 (busy) when it could not
    restart the WAL.
    """
    writer = sqlite3.connect(db.db_path_str, timeout=0, isolation_level=None)
    writer.execute("CREATE TABLE IF NOT EXISTS contention_probe (n INTEGER)")
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()

    def commit_and_restart() -> object:
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("INSERT INTO contention_probe (n) VALUES (1)")
            writer.execute("COMMIT")
            committed: object = "committed"
        except sqlite3.OperationalError as error:
            if writer.in_transaction:
                writer.rollback()
            committed = error
        restart = tuple(writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
        return committed, restart

    try:
        with _between_read_and_write(db, commit_and_restart) as outcomes:
            yield outcomes
    finally:
        writer.close()


@contextmanager
def write_lock_held_elsewhere(
    db: Any, *, grace: float = 0.2, bound: float = 30.0
) -> Iterator[list[object]]:
    """Hold the write lock from another connection while the block runs.

    The hold ends ``grace`` seconds after this thread asks for the write lock
    (``BEGIN IMMEDIATE``, or a write statement), so a transaction that waits on
    its busy timeout gets the lock then. A DEFERRED transaction that already
    read fails that write at once with SQLITE_BUSY (5) instead. The hold also
    ends when the block exits, and after ``bound`` seconds at most (a safety
    stop, far above any case's setup).

    Yields one entry for this thread's first ask: ``"asked"`` when the other
    connection still held the lock at that moment, ``"asked-after-release"``
    when it had already let go (the hold proved nothing, so ``held_off``
    rejects it).
    """
    connection = db.get_connection()
    holding, asked, released, done = (threading.Event() for _ in range(4))
    outcomes: list[object] = []

    def on_statement(sql: str) -> None:
        if asked.is_set():
            return
        upper = sql.lstrip().upper()
        if upper.startswith("BEGIN IMMEDIATE") or (
            _statement_kind(sql) == "write" and connection.in_transaction
        ):
            outcomes.append("asked-after-release" if released.is_set() else "asked")
            asked.set()

    def hold() -> None:
        holder = sqlite3.connect(db.db_path_str, isolation_level=None)
        try:
            holder.execute("BEGIN IMMEDIATE")
            holding.set()
            deadline = time.monotonic() + bound
            while not done.is_set() and time.monotonic() < deadline:
                if asked.wait(0.01):
                    done.wait(grace)  # let the transaction under test block
                    break
            released.set()
            holder.rollback()
        finally:
            holder.close()

    holder = threading.Thread(target=hold, name="write-lock-holder", daemon=True)
    holder.start()
    try:
        if not holding.wait(10):
            raise AssertionError("the other connection never took the write lock")
        connection.set_trace_callback(on_statement)
        try:
            yield outcomes
        finally:
            connection.set_trace_callback(None)
    finally:
        done.set()
        holder.join(10)
    if holder.is_alive():
        raise AssertionError("the write-lock holder did not finish")


def held_off(mode: Callable[..., Any], outcomes: list[object]) -> bool:
    """Whether the forced contention ran and the transaction held it off.

    For the commit order: the other connection's write was refused ("database
    is locked") and its WAL restart fell back to passive (busy flag 1), so the
    transaction held the write lock across its read and its write. For the
    held order: this thread asked for the write lock while the other
    connection still held it (``"asked"``); an ask after the hold had already
    ended (``"asked-after-release"``) does not count.
    """
    if len(outcomes) != 1:
        return False
    (outcome,) = outcomes
    if mode is write_committed_between_read_and_write:
        committed, restart = outcome
        return (
            type(committed) is sqlite3.OperationalError
            and str(committed) == "database is locked"
            and restart[0] == 1
        )
    return outcome == "asked"


#: The two orders, for ``pytest.mark.parametrize``.
CONTENTION_MODES = (write_committed_between_read_and_write, write_lock_held_elsewhere)
CONTENTION_IDS = ("commit-517", "held-5")
