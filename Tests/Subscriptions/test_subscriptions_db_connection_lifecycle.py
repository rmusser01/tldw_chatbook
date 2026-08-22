"""The subscriptions DB must settle its connections and its `-wal` (task-19562).

Two separate defects from the same lane, both about a connection nobody ever
closed:

* **`busy_timeout` was never set** -- the connection inherited Python's 5 s
  default. The task demanded this be *measured, not assumed*, so
  `test_busy_timeout_is_set_explicitly` pins the value and
  `test_a_writer_collision_really_waits` reproduces the collision it was
  worried about. The measurement (1.0 s lock held -> 1.07 s wait, then the
  lock acquired) confirmed the lane's PLAUSIBLE rating. What it also showed
  is that this pragma is not itself a *fix*: 5000 is what the connection
  already had, and lowering it would only turn a stall into an earlier
  `database is locked`. It is pinned so it cannot drift silently.

* **the connection was never closed and the `-wal` never checkpointed.**
  `SubscriptionsDB` keeps thread-local connections, so every worker thread
  that touched it opened one and nothing ever closed any of them. Measured
  on a real database: 4.1 MB of `-wal` after 300 inserts, 0 bytes after one
  `wal_checkpoint(TRUNCATE)`. One half of that concern was **refuted by
  measurement** and is recorded rather than quietly fixed: a child process
  that wrote a 4.1 MB `-wal` and exited normally left only `subs.db`
  behind, with the exit hook suppressed exactly as with it enabled --
  CPython finalizes the connections and SQLite removes the `-wal` on last
  close. What remains real is the *standing* 4 MB a long-running app
  carries, and the `os._exit(0)` signal path, which no `atexit` hook can
  reach (task-19561).
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from tldw_chatbook.DB.Subscriptions_DB import BUSY_TIMEOUT_MS, SubscriptionsDB

pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path):
    """A file-backed database -- `:memory:` has no `-wal` to test."""
    database = SubscriptionsDB(tmp_path / "subs.db", "test")
    yield database
    try:
        database.close()
    except Exception:
        pass


def _wal_bytes(tmp_path) -> int:
    wal = tmp_path / "subs.db-wal"
    return wal.stat().st_size if wal.exists() else 0


def test_busy_timeout_is_set_explicitly(db):
    """The value, pinned.

    On its own this assertion is weak and says so: 5000 is also what
    `sqlite3.connect(timeout=5.0)` gives you for free, so it would pass with
    the pragma deleted. `test_busy_timeout_survives_a_connector_default_change`
    is the one that can actually fail.
    """
    assert db.conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS


def test_busy_timeout_survives_a_connector_default_change(tmp_path, monkeypatch):
    """The pragma must win over whatever the connector passes.

    This is the whole point of writing the value down instead of inheriting
    it: the lock-wait budget is a property of THIS database, not a leftover
    of `sqlite3.connect`'s default. Red with the pragma removed -- the
    connection then reports the connector's 0 ms and a contended write
    raises `database is locked` immediately instead of waiting.
    """
    from tldw_chatbook.DB import base_db

    original = base_db.connect_private_sqlite

    def connect_with_no_wait(*args, **kwargs):
        kwargs["timeout"] = 0
        return original(*args, **kwargs)

    monkeypatch.setattr(base_db, "connect_private_sqlite", connect_with_no_wait)

    database = SubscriptionsDB(tmp_path / "nowait.db", "test")
    try:
        assert (
            database.conn.execute("PRAGMA busy_timeout").fetchone()[0]
            == BUSY_TIMEOUT_MS
        )
    finally:
        database.close()


def test_journal_mode_is_wal_so_readers_never_wait_on_writers(db):
    """The narrowing that shaped the fix: the exposure is writer-vs-writer.

    Under WAL a reader is never blocked by a writer, so the read-only
    service methods cannot be *stalled* by a concurrent write -- they only
    block the loop for their own duration, which is what the offload fixes.
    """
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_a_writer_collision_really_waits(db):
    """The measurement the AC asked for, as a test rather than a claim."""
    holder_ready = threading.Event()
    release = threading.Event()

    def holder():
        held = db._get_connection()
        held.execute("BEGIN IMMEDIATE")
        holder_ready.set()
        release.wait(10)
        held.rollback()
        held.close()

    worker = threading.Thread(target=holder, daemon=True)
    worker.start()
    assert holder_ready.wait(5)

    second = db._get_connection()
    started = time.monotonic()
    threading.Timer(0.3, release.set).start()
    try:
        second.execute("BEGIN IMMEDIATE")
        waited = time.monotonic() - started
        second.rollback()
    finally:
        second.close()
        worker.join(5)

    assert waited >= 0.25, (
        "the second writer did not wait for the lock at all -- either the "
        f"collision was not real or busy_timeout is 0 (waited {waited:.3f}s)"
    )


def test_close_checkpoints_and_truncates_the_wal(db, tmp_path):
    """The leak's visible half: a `-wal` left behind with content in it.

    A second connection is deliberately held open throughout. Without one,
    this test would be inert: SQLite checkpoints and *deletes* the `-wal`
    by itself when the LAST connection to a database closes, so
    `_wal_bytes() == 0` would pass with the checkpoint removed -- it would
    be measuring sqlite's own cleanup rather than ours. The held connection
    is exactly the leaked worker connection this task is about, and it is
    what keeps the file on disk for the assertion to mean something.
    """
    other = db._get_connection()
    try:
        for index in range(300):
            db.add_subscription(
                name=f"s{index}", type="rss", source=f"https://e.invalid/{index}.xml"
            )
        assert _wal_bytes(tmp_path) > 0, "nothing was written to the WAL to settle"

        db.close()

        assert (tmp_path / "subs.db-wal").exists(), (
            "the -wal vanished, so this assertion is measuring sqlite's "
            "last-connection cleanup, not close()'s checkpoint"
        )
        assert _wal_bytes(tmp_path) == 0, (
            "close() left content in the -wal; it was never checkpointed"
        )
    finally:
        other.close()


def test_close_forgets_the_connection_it_closed(db):
    """A registry that reports closed connections would be worse than none."""
    db.conn  # open one on this thread
    assert threading.get_ident() in db._connections
    db.close()
    assert threading.get_ident() not in db._connections


def test_the_connection_registry_sees_worker_threads(db):
    """`threading.local` is invisible from outside; the registry is not.

    This is what makes the leak countable at all: before task-19562 nothing
    could observe that a worker thread had opened a connection.
    """
    def touch():
        db.conn.execute("SELECT 1").fetchone()

    worker = threading.Thread(target=touch)
    worker.start()
    worker.join(5)

    assert len(db._connections) >= 1
    assert worker.ident in db._connections


def test_close_all_connections_settles_the_file_and_reports_the_rest(db, tmp_path):
    """Shutdown: checkpoint for everyone, close what this thread may close."""
    opened = threading.Event()

    def worker():
        db.conn.execute("SELECT 1").fetchone()
        opened.set()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(5)
    assert opened.is_set()

    for index in range(300):
        db.add_subscription(
            name=f"s{index}", type="rss", source=f"https://e.invalid/{index}.xml"
        )

    remaining = db.close_all_connections()

    assert _wal_bytes(tmp_path) == 0, "shutdown left an un-checkpointed -wal"
    assert remaining == 1, (
        "the worker thread's connection should be reported as still open -- "
        f"sqlite3 refuses a cross-thread close, got {remaining}"
    )
    assert threading.get_ident() not in db._connections


def test_a_cross_thread_close_is_refused_by_sqlite(db):
    """Why `close_all_connections` reports instead of closing.

    Pinned as a fact about the runtime, not an opinion about the design: if
    a future Python allows this, the reporting compromise can be revisited
    and this test is what will say so.
    """
    made: dict[str, sqlite3.Connection] = {}

    def maker():
        made["conn"] = db._get_connection()

    thread = threading.Thread(target=maker)
    thread.start()
    thread.join(5)

    with pytest.raises(sqlite3.ProgrammingError):
        made["conn"].close()


def test_checkpoint_declines_inside_an_open_transaction(db):
    """It must never commit a caller's half-finished unit of work."""
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source) VALUES (?,?,?)",
            ("mid", "rss", "https://e.invalid/mid.xml"),
        )
        assert db.checkpoint_wal() is False


def test_an_in_memory_database_has_nothing_to_checkpoint(tmp_path):
    """No `-wal` exists, and the connection IS the database."""
    memory_db = SubscriptionsDB(":memory:", "test")
    assert memory_db.checkpoint_wal() is False
    memory_db.close()


def test_a_writable_database_registers_for_the_exit_hook(tmp_path):
    """The hook can only settle databases it knows about."""
    from tldw_chatbook.DB import Subscriptions_DB as module

    database = SubscriptionsDB(tmp_path / "registered.db", "test")
    try:
        assert database in set(module._OPEN_SUBSCRIPTIONS_DBS)
    finally:
        database.close()


def test_an_in_memory_database_does_not_register(tmp_path):
    """It has no `-wal`, and it ceases to exist with its connection."""
    from tldw_chatbook.DB import Subscriptions_DB as module

    database = SubscriptionsDB(":memory:", "test")
    try:
        assert database not in set(module._OPEN_SUBSCRIPTIONS_DBS)
    finally:
        database.close()


def test_the_exit_hook_settles_a_registered_database(tmp_path):
    """The hook body, exercised directly.

    Measured caveat recorded in the hook's own docstring: on a clean
    process exit CPython finalizes the connections and SQLite removes the
    `-wal` by itself, so this hook is not what saves the file there. It is
    tested here for what it actually does -- settle at a defined moment,
    with a defined error path -- rather than assumed to be doing more.
    """
    from tldw_chatbook.DB import Subscriptions_DB as module

    database = SubscriptionsDB(tmp_path / "hooked.db", "test")
    other = database._get_connection()  # keeps the -wal on disk to inspect
    try:
        for index in range(300):
            database.add_subscription(
                name=f"s{index}", type="rss", source=f"https://e.invalid/{index}.xml"
            )
        wal = tmp_path / "hooked.db-wal"
        assert wal.stat().st_size > 0

        module._checkpoint_open_databases_at_exit()

        assert wal.exists() and wal.stat().st_size == 0
    finally:
        other.close()
        database.close()


def test_the_exit_hook_never_raises(tmp_path, monkeypatch):
    """A diagnostic must not be what breaks the exit."""
    from tldw_chatbook.DB import Subscriptions_DB as module

    database = SubscriptionsDB(tmp_path / "boom.db", "test")

    def explode():
        raise RuntimeError("shutdown is a bad time to raise")

    monkeypatch.setattr(database, "close_all_connections", explode)
    module._checkpoint_open_databases_at_exit()  # must not raise
    database.close()


def test_the_exit_hook_stays_silent_when_the_database_file_is_gone(
    tmp_path, capsys, monkeypatch
):
    """A settle at teardown must not try to log to a closed sink.

    The incident this pins: the first version of the hook hit a temporary
    database directory that pytest had already removed, and the resulting
    `logger.warning` raised `ValueError: I/O operation on closed file`,
    printing a logging traceback on every exit. The diagnostic became the
    failure.
    """
    from tldw_chatbook.DB import Subscriptions_DB as module

    target = tmp_path / "vanishing.db"
    database = SubscriptionsDB(target, "test")
    database.conn.execute("SELECT 1").fetchone()
    database.close()
    for leftover in tmp_path.glob("vanishing.db*"):
        leftover.unlink()

    monkeypatch.setattr(module, "_INTERPRETER_EXITING", False)
    module._checkpoint_open_databases_at_exit()

    assert module._INTERPRETER_EXITING is True, (
        "the hook must announce the exit so the settle path stops logging"
    )
    assert not target.exists(), "the hook re-created a database it should skip"
