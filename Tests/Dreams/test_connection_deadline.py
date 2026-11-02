import time

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB


@pytest.mark.parametrize("revived", [False, True])
@pytest.mark.parametrize("pin_configure", [False, True])
def test_deadline_opens_fresh_or_revived_connection(
    tmp_path, monkeypatch, revived, pin_configure
):
    db = DreamsDB(tmp_path / "deadline.sqlite", "deadline")
    try:
        if revived:
            old = db._held_connection()
            old.close()
            db._thread_local.conn_last_used = 0
        else:
            db.close()
        if pin_configure:
            original = db._open_connection
            monkeypatch.setattr(db, "_open_connection", lambda **kwargs: original())
        with db.connection(operation_deadline=time.monotonic() + 5) as conn:
            assert conn.execute("SELECT 42").fetchone()[0] == 42
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        db.close_all()


def test_expired_setup_closes_native_handle_and_recovers(tmp_path, monkeypatch):
    import sqlite3
    import threading

    from tldw_chatbook.DB.base_db import BaseDB

    db = DreamsDB(tmp_path / "setup-timeout.sqlite", "deadline")
    db.close()
    allocated = []
    entered = threading.Event()
    release = threading.Event()
    original_open = BaseDB._get_connection
    original_lineage = db._validate_schema_lineage

    def capture(*args, **kwargs):
        conn = original_open(*args, **kwargs)
        allocated.append(conn)
        return conn

    def pause(conn):
        entered.set()
        assert release.wait(5)
        return original_lineage(conn)

    monkeypatch.setattr(BaseDB, "_get_connection", capture)
    monkeypatch.setattr(db, "_validate_schema_lineage", pause)
    outcome = []
    deadline = time.monotonic() + 0.5

    def open_connection():
        try:
            db._held_connection(operation_deadline=deadline)
        except BaseException as exc:
            outcome.append(exc)
        outcome.append(getattr(db._thread_local, "conn", None))

    worker = threading.Thread(target=open_connection)
    worker.start()
    try:
        assert entered.wait(5)
        while time.monotonic() < deadline:
            threading.Event().wait(min(0.01, deadline - time.monotonic()))
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert isinstance(outcome[0], TimeoutError)
        assert outcome[1] is None
        assert len(allocated) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            allocated[0].execute("SELECT 1")
        monkeypatch.setattr(db, "_validate_schema_lineage", original_lineage)
        with db.connection(operation_deadline=time.monotonic() + 5) as conn:
            assert conn.execute("SELECT 42").fetchone()[0] == 42
    finally:
        release.set()
        worker.join(5)
        db.close_all()


def test_revival_retains_closed_object_until_close_all(tmp_path):
    import sqlite3

    db = DreamsDB(tmp_path / "retained.sqlite", "deadline")
    try:
        old = db._held_connection()
        old.close()
        db._thread_local.conn_last_used = 0
        new = db._held_connection(operation_deadline=time.monotonic() + 5)
        assert new is not old
        assert old in db._opened
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            old.execute("SELECT 1")
        db.close_all()
        assert old not in db._opened
    finally:
        db.close_all()
