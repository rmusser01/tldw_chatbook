"""BaseDB vacuum/integrity connections close on every path (TASK-38606).

``BaseDB.vacuum`` and ``BaseDB.check_integrity`` open a short-lived
connection and used to call ``.close()`` only on the success path, so any
raised ``sqlite3.Error`` (locked or corrupt file) leaked the handle until
GC. These tests spy on the connection object and force failures at each
step to prove the close happens on success AND error paths, without
leaking and without changing the methods' outward contracts (vacuum
re-raises; check_integrity returns False).
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tldw_chatbook.DB.base_db import BaseDB


class _SpyCursor:
    def __init__(self, connection: _SpyConnection) -> None:
        self._connection = connection

    def execute(self, statement: str) -> _SpyCursor:
        if self._connection.fail_on == statement:
            raise sqlite3.OperationalError(f"forced failure on {statement!r}")
        return self

    def fetchone(self) -> Any:
        return self._connection.fetchone_result


class _SpyConnection:
    """Records closes; optionally raises on one exact statement."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.fetchone_result: Any = ("ok",)
        self.close_calls = 0

    def execute(self, statement: str) -> None:
        if self.fail_on == statement:
            raise sqlite3.OperationalError(f"forced failure on {statement!r}")

    def cursor(self) -> _SpyCursor:
        return _SpyCursor(self)

    def close(self) -> None:
        self.close_calls += 1


class _ProbeDB(BaseDB):
    """Minimal concrete BaseDB; schema init is irrelevant to this lifecycle."""

    def _initialize_schema(self) -> None:  # pragma: no cover - abstract stub
        pass

    def __init__(self, path: Any) -> None:
        super().__init__(path, initialize_schema=False)


@pytest.fixture()
def probe_db(tmp_path):
    return _ProbeDB(tmp_path / "probe.sqlite")


def test_vacuum_closes_connection_on_success(probe_db, monkeypatch):
    conn = _SpyConnection()
    monkeypatch.setattr(probe_db, "_get_connection", lambda **_: conn)
    probe_db.vacuum()
    assert conn.close_calls == 1


def test_vacuum_closes_connection_when_vacuum_raises(probe_db, monkeypatch):
    conn = _SpyConnection(fail_on="VACUUM")
    monkeypatch.setattr(probe_db, "_get_connection", lambda **_: conn)
    with pytest.raises(sqlite3.OperationalError, match="VACUUM"):
        probe_db.vacuum()
    assert conn.close_calls == 1


def test_check_integrity_closes_connection_on_success(probe_db, monkeypatch):
    conn = _SpyConnection()
    monkeypatch.setattr(probe_db, "_get_connection", lambda **_: conn)
    assert probe_db.check_integrity() is True
    assert conn.close_calls == 1


def test_check_integrity_closes_connection_when_pragma_raises(
    probe_db, monkeypatch
):
    conn = _SpyConnection(fail_on="PRAGMA integrity_check")
    monkeypatch.setattr(probe_db, "_get_connection", lambda **_: conn)
    assert probe_db.check_integrity() is False
    assert conn.close_calls == 1


def test_check_integrity_closes_connection_when_open_fails(
    probe_db, monkeypatch
):
    # _get_connection itself raising must not attempt a close (no handle
    # exists) and must still report failure, not raise.
    def refuse(**_: Any) -> sqlite3.Connection:
        raise sqlite3.OperationalError("cannot open")

    monkeypatch.setattr(probe_db, "_get_connection", refuse)
    assert probe_db.check_integrity() is False


def test_check_integrity_close_failure_returns_false_and_retains_native_owner(
    probe_db, monkeypatch
):
    from tldw_chatbook.DB import private_sqlite

    class CloseOnce(sqlite3.Connection):
        close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise sqlite3.OperationalError("controlled native close refusal")
            super().close()

    connect = probe_db._get_connection
    connection = connect(factory=CloseOnce)
    monkeypatch.setattr(probe_db, "_get_connection", lambda **_: connection)
    try:
        assert probe_db.check_integrity() is False
        assert connection.close_calls == 1
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert connection in private_sqlite._ordinary_connections
    finally:
        connection.close()
    assert connection.close_calls == 2
    assert connection not in private_sqlite._ordinary_connections
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
