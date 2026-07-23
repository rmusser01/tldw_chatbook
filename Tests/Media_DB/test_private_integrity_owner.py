from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

import tldw_chatbook.DB.private_sqlite as private_sqlite
from tldw_chatbook.DB import Client_Media_DB_v2 as media_db
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


def _create_private_database(path: Path) -> None:
    path.parent.chmod(0o700)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE integrity_probe (value TEXT)")
        connection.execute("INSERT INTO integrity_probe VALUES ('ok')")
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)


def test_integrity_check_uses_registered_read_only_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "media.db"
    _create_private_database(database)
    calls: list[tuple[str, Path, bool]] = []

    def tracking_connect(owner_id, target, *, read_only=False, **kwargs):
        calls.append((owner_id, Path(target), read_only))
        connection = connect_private_sqlite(
            owner_id,
            target,
            read_only=read_only,
            **kwargs,
        )
        try:
            connection.execute("CREATE TABLE forbidden_write (value TEXT)")
        except sqlite3.OperationalError:
            pass
        else:
            raise AssertionError("integrity connection was writable")
        return connection

    monkeypatch.setattr(media_db, "connect_private_sqlite", tracking_connect)

    assert media_db.check_database_integrity(database) is True
    assert calls == [("db.media.integrity", database, True)]


def test_integrity_check_invalid_memory_target_preserves_bool_contract() -> None:
    assert media_db.check_database_integrity(":memory:") is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX trust contract")
def test_integrity_check_rejects_unsafe_parent_and_preserves_false_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    database = unsafe_parent / "media.db"
    _create_private_database(database)
    unsafe_parent.chmod(0o777)
    raw_calls: list[object] = []

    def forbidden_raw_connect(*args, **kwargs):
        raw_calls.append((args, kwargs))
        raise AssertionError("raw SQLite was reached")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", forbidden_raw_connect)
    try:
        assert media_db.check_database_integrity(database) is False
    finally:
        os.chmod(unsafe_parent, 0o700)

    assert raw_calls == []
