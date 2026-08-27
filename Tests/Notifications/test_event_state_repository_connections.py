"""TASK-21131: held-connection lifecycle for the event-state repository.

The repository used to open (and GC-leak) a fresh private-SQLite connection
per operation. These guards pin the replacement: one held connection per
thread, IMMEDIATE transactions around every read-modify-write, and a
``close()`` that neither strands a busy operation nor wedges the store.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from tldw_chatbook.DB import base_db
from tldw_chatbook.Notifications import event_state_repository as esr_module
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository
from tldw_chatbook.Notifications.server_notification_events import (
    build_server_notification_feed,
)
from tldw_chatbook.runtime_policy.server_parity_models import NormalizedEventRecord


def _event(index: int = 0) -> NormalizedEventRecord:
    return NormalizedEventRecord(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="global",
        event_kind="notification.created",
        entity_ref={"type": "notification", "id": f"n{index}"},
        payload_hash=f"hash-{index}",
        event_id=str(index),
        server_cursor=str(index),
        emitted_at="2026-08-24T01:02:03Z",
        received_at="2026-08-24T01:02:04Z",
        transport_type="sse",
        payload_kind="notification",
        payload={"data": {"id": f"n{index}", "title": f"Notice {index}"}},
    )


_FEED_SCOPE = dict(
    server_profile_id="server-a",
    authenticated_principal_id="user-a",
    stream_instance_id="global",
)


@pytest.fixture
def repo(tmp_path):
    repository = EventStateRepository(tmp_path / "event_state.db", "test")
    yield repository
    repository.close()


@pytest.fixture
def open_counter(monkeypatch):
    """Count every private-SQLite open this repository performs.

    Both seams are wrapped: the repository's own module-level name and the
    one ``BaseDB._get_connection`` uses. Wrapping only one would make the
    counter silently blind to a shape that opened through the other.
    """
    calls: list[str] = []

    for module in (esr_module, base_db):
        original = module.connect_private_sqlite

        def counted(owner_id, database, *args, _original=original, **kwargs):
            calls.append(str(database))
            return _original(owner_id, database, *args, **kwargs)

        monkeypatch.setattr(module, "connect_private_sqlite", counted)
    return calls


# --------------------------------------------------------------------------
# The connection is held, not reopened per operation
# --------------------------------------------------------------------------


def test_repeated_operations_open_no_further_connections(repo, open_counter):
    repo.record_event_and_advance_processed_cursor(_event(0))
    open_counter.clear()

    for index in range(1, 11):
        repo.record_event_and_advance_processed_cursor(_event(index))
        repo.list_events(payload_kind="notification", limit=10)
        repo.get_processed_cursor(
            source_authority="server",
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            stream_name="notifications",
            stream_instance_id="global",
        )

    assert open_counter == [], (
        f"30 operations on a held connection opened {len(open_counter)} new connections"
    )


def test_feed_build_opens_no_connections_after_warmup(repo, open_counter):
    for index in range(5):
        repo.record_event_and_advance_processed_cursor(_event(index))
    open_counter.clear()

    feed = build_server_notification_feed(
        repo, limit=20, mark_presented=True, **_FEED_SCOPE
    )

    assert feed["total"] == 5
    assert feed["replay"]["state"] == "available"
    assert open_counter == [], (
        f"one feed build opened {len(open_counter)} connections; the whole "
        "point of the held connection is that it opens none"
    )


def test_each_thread_holds_its_own_connection(repo, open_counter):
    repo.list_events(limit=1)
    open_counter.clear()
    seen: dict[str, int] = {}
    errors: list[Exception] = []

    def worker(name: str) -> None:
        try:
            for _ in range(5):
                repo.list_events(limit=1)
            seen[name] = id(repo._held_connection())
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(f"w{i}",), name=f"esr-w{i}")
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert len(set(seen.values())) == 2, "two threads shared one connection"
    assert len(open_counter) == 2, (
        f"two worker threads should open exactly two connections, got "
        f"{len(open_counter)}"
    )


def test_held_connection_keeps_wal_normal_autocommit_and_foreign_keys(repo):
    repo.list_events(limit=1)
    conn = repo._held_connection()

    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    assert conn.isolation_level is None, (
        "a held connection must be in true autocommit or the explicit "
        "BEGIN IMMEDIATE raises 'cannot start a transaction within a "
        "transaction'"
    )
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000, (
        "IMMEDIATE contention is only safe because the busy handler retries"
    )


def test_schema_initialization_closes_its_own_connection(tmp_path, monkeypatch):
    """The schema pass owns a short-lived connection and must close it.

    It cannot use the held connection (it runs inside ``_ensure_schema``'s
    lock, which ``_get_connection`` re-enters), so it is the one place that
    still opens a connection of its own -- and the one place that can leak
    one again.
    """
    created: list[sqlite3.Connection] = []

    for module in (esr_module, base_db):
        original = module.connect_private_sqlite

        def recorded(owner_id, database, *args, _original=original, **kwargs):
            conn = _original(owner_id, database, *args, **kwargs)
            created.append(conn)
            return conn

        monkeypatch.setattr(module, "connect_private_sqlite", recorded)

    repository = EventStateRepository(tmp_path / "event_state.db", "test")
    try:
        repository.list_events(limit=1)

        assert len(created) == 2, (
            "expected exactly one schema connection plus one held connection, "
            f"got {len(created)}"
        )
        schema_conn, held_conn = created
        with pytest.raises(sqlite3.ProgrammingError):
            schema_conn.execute("SELECT 1")
        assert held_conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        repository.close()


# --------------------------------------------------------------------------
# close(): re-arms, and never closes a connection under live work
# --------------------------------------------------------------------------


def test_close_rearms_a_file_backed_store(repo):
    repo.record_event_and_advance_processed_cursor(_event(0))
    repo.close()

    rows = repo.list_events(payload_kind="notification", limit=10)
    assert len(rows) == 1, "close() must re-arm, not wedge, a file store"


def test_close_rearms_a_memory_store_with_its_schema():
    repository = EventStateRepository(":memory:", "test")
    try:
        repository.record_event_and_advance_processed_cursor(_event(0))
        repository.close()

        # The in-memory database died with its connection. The re-armed
        # store must rebuild the schema rather than query tables that no
        # longer exist.
        assert repository.list_events(limit=10) == []
        repository.record_event_and_advance_processed_cursor(_event(1))
        assert len(repository.list_events(limit=10)) == 1
    finally:
        repository.close()


def test_close_releases_an_idle_worker_threads_connection(repo):
    """Shutdown must reach connections it does not own.

    Home's feed refresh runs on ``asyncio.to_thread`` workers, so most held
    connections belong to threads that are long gone by shutdown. With
    sqlite3's default same-thread guard a cross-thread ``close()`` raises
    instead of closing, and those connections would stay open for the life
    of the process.
    """
    worker_conn: dict[str, sqlite3.Connection] = {}

    def worker() -> None:
        repo.list_events(limit=1)
        worker_conn["conn"] = repo._held_connection()

    thread = threading.Thread(target=worker, name="esr-idle")
    thread.start()
    thread.join(10)

    assert "conn" in worker_conn
    assert len(repo._held) == 1

    repo.close()  # from the MAIN thread

    assert repo._held == {}, "close() left a foreign thread's connection mapped"
    # Match the message: a connection that was merely *refused* to this
    # thread also raises ProgrammingError, which would pass vacuously.
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        worker_conn["conn"].execute("SELECT 1")


def test_close_leaves_an_in_flight_operations_connection_open(repo, monkeypatch):
    """Quit with a feed in flight: the running operation must still finish."""
    entered = threading.Event()
    release = threading.Event()
    original = EventStateRepository._dedupe_exists

    def parked(conn, dedupe_key):
        entered.set()
        assert release.wait(10), "probe never released"
        return original(conn, dedupe_key)

    monkeypatch.setattr(
        EventStateRepository, "_dedupe_exists", staticmethod(parked), raising=True
    )

    outcome: list[object] = []

    def worker() -> None:
        try:
            outcome.append(repo.record_event_and_advance_processed_cursor(_event(7)))
        except Exception as exc:  # noqa: BLE001 - asserted below
            outcome.append(exc)

    thread = threading.Thread(target=worker, name="esr-inflight")
    thread.start()
    assert entered.wait(10), "worker never reached the parked probe"

    repo.close()  # shutdown, from another thread, mid-transaction

    release.set()
    thread.join(10)
    assert not thread.is_alive()

    assert len(outcome) == 1
    assert not isinstance(outcome[0], Exception), (
        f"close() broke an in-flight operation: {outcome[0]!r}"
    )
    assert outcome[0].is_duplicate is False


def test_liveness_ping_revives_an_externally_closed_connection(repo, monkeypatch):
    monkeypatch.setattr(EventStateRepository, "_LIVENESS_PING_IDLE_SECONDS", 0.0)
    repo.record_event_and_advance_processed_cursor(_event(0))

    # Something else closed the handle out from under us.
    repo._held_connection().close()

    rows = repo.list_events(payload_kind="notification", limit=10)
    assert len(rows) == 1


# --------------------------------------------------------------------------
# Transactions: atomic read-modify-write, and errors that stay legible
# --------------------------------------------------------------------------


def test_concurrent_record_of_one_event_yields_exactly_one_insert(repo):
    """A held connection per thread makes the dedupe TOCTOU a live race."""
    repo.list_events(limit=1)  # warm the schema before the barrier
    workers = 12
    barrier = threading.Barrier(workers)
    results: list[object] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        # Open THIS thread's connection before the barrier. Connection
        # creation serialises on the held-connection lock, so a thread that
        # opens after the barrier is handed a window in which every earlier
        # writer has already committed -- which makes the race the test is
        # trying to force disappear.
        repo.list_events(limit=1)
        barrier.wait(10)
        try:
            result = repo.record_event_and_advance_processed_cursor(_event(0))
        except Exception as exc:  # noqa: BLE001 - asserted below
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(result)

    threads = [
        threading.Thread(target=worker, name=f"esr-race{i}") for i in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert errors == [], f"concurrent recording raised: {errors!r}"
    inserted = [result for result in results if not result.is_duplicate]
    assert len(inserted) == 1, (
        f"{len(inserted)} of {workers} writers each believed they inserted the "
        "event: the dedupe probe and the insert did not share a transaction"
    )
    assert len(repo.list_events(payload_kind="notification", limit=50)) == 1


def test_concurrent_remember_event_yields_exactly_one_insert(repo):
    repo.list_events(limit=1)
    workers = 12
    barrier = threading.Barrier(workers)
    results: list[object] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        repo.list_events(limit=1)  # open this thread's connection first
        barrier.wait(10)
        try:
            result = repo.remember_event(_event(3))
        except Exception as exc:  # noqa: BLE001 - asserted below
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(result)

    threads = [
        threading.Thread(target=worker, name=f"esr-remember{i}") for i in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert errors == [], f"concurrent remember_event raised: {errors!r}"
    inserted = [result for result in results if not result.is_duplicate]
    assert len(inserted) == 1, (
        f"{len(inserted)} of {workers} writers each believed they inserted the "
        "dedupe record"
    )


def test_transaction_error_is_not_masked_by_a_failing_rollback(repo):
    repo.list_events(limit=1)
    entry = repo._held_entry()
    real = entry.conn

    class _RollbackExplodes:
        def __getattr__(self, name):
            return getattr(real, name)

        def rollback(self):
            raise sqlite3.OperationalError("rollback exploded")

    entry.conn = _RollbackExplodes()
    try:
        with pytest.raises(ValueError, match="body failed"):
            with repo.transaction():
                raise ValueError("body failed")
    finally:
        entry.conn = real


def test_a_failed_transaction_leaves_no_open_transaction(repo):
    repo.list_events(limit=1)
    conn = repo._held_connection()

    with pytest.raises(ValueError):
        with repo.transaction():
            raise ValueError("body failed")

    assert conn.in_transaction is False, (
        "a failed body left the write transaction open; the next operation "
        "would fail with 'cannot start a transaction within a transaction'"
    )
    # ...and the store is still usable.
    repo.record_event_and_advance_processed_cursor(_event(0))
    assert len(repo.list_events(payload_kind="notification", limit=10)) == 1


def test_feed_on_a_first_run_store_is_empty_and_opens_one_connection(
    repo, open_counter
):
    """Empty/first-run walk: nothing recorded yet, nothing to present."""
    feed = build_server_notification_feed(
        repo, limit=20, mark_presented=True, **_FEED_SCOPE
    )

    assert feed["items"] == []
    assert feed["total"] == 0
    assert feed["replay"]["state"] == "empty"
    assert feed["replay"]["server_refetch_required"] is False
    assert len(open_counter) == 2, (
        "a first-run feed build should open the schema connection plus one "
        f"held connection, got {len(open_counter)}"
    )


def test_an_unopenable_database_raises_legibly_and_recovers_when_replaced(tmp_path):
    """Error walk: a corrupt store must not wedge into an un-retryable state."""
    db_path = tmp_path / "event_state.db"
    db_path.write_bytes(b"not a database\n" * 64)
    db_path.chmod(0o600)

    repository = EventStateRepository(db_path, "test")
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not a database"):
            repository.list_events(limit=1)
        assert repository._held == {}, "a failed open left a connection registered"

        db_path.unlink()  # the user removed the corrupt file

        assert repository.list_events(limit=1) == [], (
            "the schema pass did not retry after its first failure"
        )
    finally:
        repository.close()


def test_mark_presented_for_an_unknown_event_leaves_the_store_usable(repo):
    """Error path: the KeyError arm rolls back rather than wedging."""
    repo.record_event_and_advance_processed_cursor(_event(0))

    with pytest.raises(KeyError):
        repo.mark_event_presented_and_advance_high_water(
            event_key="does-not-exist", cursor="9"
        )

    feed = build_server_notification_feed(
        repo, limit=20, mark_presented=True, **_FEED_SCOPE
    )
    assert feed["total"] == 1


def test_a_failing_close_is_recorded_rather_than_discarded(repo, monkeypatch):
    """A swallowed `conn.close()` error left an untracked handle to debug.

    Best-effort teardown is deliberate -- the entry is already out of `_held`
    and re-closing a handle whose close raised would keep a broken connection
    reachable. What must not happen is losing the fact that it happened.
    """
    repo.record_event_and_advance_processed_cursor(_event(0))

    recorded: list[tuple] = []
    monkeypatch.setattr(
        esr_module.logger, "debug", lambda *args, **kwargs: recorded.append(args)
    )

    class _RefusesToClose:
        """`sqlite3.Connection.close` is read-only, so stand in for the handle."""

        def close(self):
            # A real sqlite3 message can carry the database path; use one.
            raise sqlite3.OperationalError("unable to close /private/db.sqlite")

    held = list(repo._held.values())
    assert held, "no held connection to close, so the assertion would be vacuous"
    real_conns = []
    for entry in held:
        real_conns.append(entry.conn)
        entry.conn = _RefusesToClose()

    try:
        repo.close()
    finally:
        for conn in real_conns:
            conn.close()

    rendered = " ".join(str(part) for call in recorded for part in call)
    assert "close failed" in rendered, (
        f"a failing close was discarded silently: {recorded}"
    )
    assert "OperationalError" in rendered, (
        f"the failure type was not recorded: {recorded}"
    )
    # Type name ONLY: a sqlite3 message can carry the database path.
    assert "/private/db.sqlite" not in rendered, (
        f"the close diagnostic leaked the database path: {recorded}"
    )
