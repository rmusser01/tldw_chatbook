"""Resumed character reads retire fresh native handles and preserve borrowers."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

_route, outcome = sys.argv[1:]
db = CharactersRAGDB(Path.home() / "resume-character.db", "resume-lifetime-test")
character_id = db.add_character_card({"name": "  Resumed character  "})
db.close_connection()
assert getattr(db._local, "conn", None) is None
host = SimpleNamespace(app_instance=SimpleNamespace(chachanotes_db=db))
original = db.get_character_card_by_id
entered, release = threading.Event(), threading.Event()
observed = []
borrowed = []


def worker_leases(worker):
    with storage._lock:
        return tuple(
            lease for lease in storage._live_leases
            if lease.resource_thread is worker and lease.resource_path == db.db_path
        )


def read_card(*args, **kwargs):
    card = original(*args, **kwargs)
    connection = db.get_connection()
    observed.append((connection, threading.current_thread()))
    assert db._connection_quiescence.is_registered(connection)
    assert worker_leases(threading.current_thread())
    entered.set()
    if outcome == "cancel" and not release.wait(10):
        raise AssertionError("native worker was not released")
    if outcome == "failure":
        raise RuntimeError("injected failure after the real character query")
    return card


def open_borrower():
    connection = db.get_connection()
    borrowed.append(connection)
    connection.execute("BEGIN").close()
    connection.execute(
        "UPDATE character_cards SET description=? WHERE id=?",
        ("borrowed pending edit", character_id),
    ).close()
    assert connection.in_transaction


def check_worker_outcome():
    assert len(observed) == 1
    connection, worker = observed[0]
    assert worker is threading.current_thread()
    if outcome == "borrowed":
        assert connection is borrowed[0]
        assert getattr(db._local, "conn", None) is connection
        assert db._connection_quiescence.is_registered(connection)
        assert worker_leases(worker)
        assert connection.in_transaction
        cursor = connection.execute(
            "SELECT description FROM character_cards WHERE id=?", (character_id,)
        )
        try:
            assert cursor.fetchone()[0] == "borrowed pending edit"
        finally:
            cursor.close()
        connection.rollback()
        assert not connection.in_transaction
        db.close_connection()
    else:
        # These assertions run after the real worker invocation has unwound,
        # before test cleanup can close any leaked native connection.
        assert getattr(db._local, "conn", None) is None
        assert not db._connection_quiescence.is_registered(connection)
        assert not worker_leases(worker), "resumed-character worker retained a lease"
    # Bypass subclass convenience methods and check the native SQLite handle on
    # its owning thread. A thread-affinity error is not evidence of closure.
    try:
        cursor = sqlite3.Connection.execute(connection, "SELECT 1")
    except sqlite3.ProgrammingError as error:
        assert "closed" in str(error).lower()
    else:
        cursor.close()
        raise AssertionError("resumed-character native handle remains open")
    assert not db._connection_quiescence.is_registered(connection)
    assert not worker_leases(worker)


async def main():
    loop = asyncio.get_running_loop()
    # asyncio.run owns shutdown; one worker gives deterministic reuse and lets
    # the observation below run only after an independently cancelled call ends.
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    task = None
    try:
        if outcome == "borrowed":
            await asyncio.to_thread(open_borrower)
        db.get_character_card_by_id = read_card
        task = asyncio.create_task(
            ChatScreen._resolve_resumed_character_name(host, character_id)
        )
        if outcome == "cancel":
            async with asyncio.timeout(10):
                while not entered.is_set():
                    await asyncio.sleep(.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("UI lookup cancellation was swallowed")
            connection, worker = observed[0]
            assert not release.is_set()
            assert db._connection_quiescence.is_registered(connection)
            assert worker_leases(worker), "cancelled UI dropped a live worker lease"
            release.set()
        else:
            result = await task
            assert result == ("" if outcome == "failure" else "Resumed character")
        # Queued on the same sole worker: this is native completion evidence,
        # unlike the getter's return event or an elapsed sleep.
        await asyncio.to_thread(check_worker_outcome)
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(db.close_connection)
        db.get_character_card_by_id = original


try:
    asyncio.run(main())
finally:
    db.close_connection()
print("retired and reopened")
"""


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel", "borrowed"])
def test_resumed_character_lookup_owns_only_its_fresh_native_connection(
    tmp_path, outcome
):
    _run(tmp_path, "character-resume", outcome, script=_SCRIPT)
