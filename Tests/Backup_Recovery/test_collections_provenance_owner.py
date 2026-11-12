"""Populated, source-bound Collections v4 provenance migration outcomes."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.recovery_core import core_adapters


def _adapter():
    return next(a for a in core_adapters() if a.owner_id == "db.library_collections")


def _legacy(path):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/collections_v4.json").read_text()
    )
    with closing(sqlite3.connect(path)) as connection:
        for statement in fixture["ddl"]:
            connection.execute(statement)
        connection.execute("INSERT INTO schema_version VALUES (4)")
        connection.execute(
            "INSERT INTO library_collections (collection_id,name,created_at,updated_at) VALUES ('collection', 'Before upgrade', '2000-01-01', '2000-01-02')"
        )
        connection.execute("""INSERT INTO collection_capture_items
            (authority_key,capture_id,submitted_url,canonical_url,domain,title,status,favorite,
             processing_state,created_at,updated_at,revision)
            VALUES ('local:test','captured','http://intranet/article',
            'http://intranet/article','intranet','Original','saved',0,'failed',
            '2000-01-01','2000-01-02',7)""")
        connection.execute(
            "INSERT INTO collection_capture_tags VALUES ('local:test',1,'science','Science')"
        )
        connection.execute(
            "INSERT INTO collection_capture_item_tags VALUES ('local:test','captured',1)"
        )
        connection.commit()
        return _snapshot(connection)


def _snapshot(connection):
    tables = (
        "library_collections",
        "collection_capture_items",
        "collection_capture_tags",
        "collection_capture_item_tags",
    )
    return {
        table: (
            [r[1] for r in connection.execute(f"PRAGMA table_info({table})")],
            connection.execute(f"SELECT * FROM {table}").fetchall(),
        )
        for table in tables
    }


def _preserved(connection, before):
    for table, (columns, rows) in before.items():
        projection = ",".join(f'"{column}"' for column in columns)
        assert (
            connection.execute(f"SELECT {projection} FROM {table}").fetchall() == rows
        )
    assert connection.execute(
        "SELECT url_provenance FROM collection_capture_items"
    ).fetchall() == [("unknown",)]
    assert connection.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
        5,
    )
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("route", ["runtime", "installed"])
def test_populated_v4_migrates_and_reopens_without_inventing_authority(tmp_path, route):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    path = tmp_path / "collections.sqlite"
    before = _legacy(path)
    assert _adapter().validate(path) == ()
    if route == "installed":
        assert validate_candidate(_adapter(), path, Event(), migrate=True) == ()
    for _ in range(2):
        db = LibraryCollectionsDB(path)
        try:
            with db.connection() as connection:
                # Native row factory belongs to the store; comparison preserves it.
                for table, (columns, rows) in before.items():
                    projection = ",".join(f'"{column}"' for column in columns)
                    assert [
                        tuple(r)
                        for r in connection.execute(f"SELECT {projection} FROM {table}")
                    ] == rows
                assert (
                    connection.execute(
                        "SELECT url_provenance FROM collection_capture_items"
                    ).fetchone()[0]
                    == "unknown"
                )
        finally:
            db.close()
        with closing(sqlite3.connect(path)) as connection:
            _preserved(connection, before)
        assert _adapter().validate(path) == ()


def test_v4_runtime_failure_rolls_back_provenance_and_original_history(
    tmp_path, monkeypatch
):
    path = tmp_path / "collections.sqlite"
    _legacy(path)
    with closing(sqlite3.connect(path)) as connection:
        before = list(connection.iterdump())
    acquired = []
    original = LibraryCollectionsDB._get_connection

    def capture(self):
        connection = original(self)
        acquired.append(connection)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(LibraryCollectionsDB, "_get_connection", capture)
        patch.setattr(
            LibraryCollectionsDB,
            "_REVIEW_SET_SCHEMA_DDL",
            ("SELECT missing_after_provenance",),
        )
        try:
            with pytest.raises(sqlite3.Error):
                LibraryCollectionsDB(path)
        finally:
            for connection in acquired:
                connection.close()
    with closing(sqlite3.connect(path)) as connection:
        assert list(connection.iterdump()) == before
    db = LibraryCollectionsDB(path)
    db.close()
    assert _adapter().validate(path) == ()


@pytest.mark.parametrize("damage", ["future", "empty", "hybrid", "foreign_column"])
def test_installed_collections_refuses_unproven_lineage_without_mutation(
    tmp_path, damage
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    path = tmp_path / "collections.sqlite"
    _legacy(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            {
                "future": "UPDATE schema_version SET version=6",
                "empty": "DELETE FROM schema_version",
                "hybrid": "UPDATE schema_version SET version=5",
                "foreign_column": "ALTER TABLE collection_capture_items ADD COLUMN foreign_field TEXT",
            }[damage]
        )
        connection.commit()
        before = list(connection.iterdump())
    assert validate_candidate(_adapter(), path, Event(), migrate=True)
    with closing(sqlite3.connect(path)) as connection:
        assert list(connection.iterdump()) == before


def test_collections_native_capture_relocation_keeps_migrated_unknown_and_history(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    source = tmp_path / "collections.sqlite"
    expected = _legacy(source)
    db = LibraryCollectionsDB(source)
    db.close()
    before = source.read_bytes()
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = application_authority(tmp_path, source, monkeypatch)
    adapter = _adapter()
    destination = stage / "snapshot.sqlite"
    item = StorageItem(
        "db.library_collections",
        "profile:test:db.library_collections",
        source,
        "included",
        (),
    )
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((source,), stage),
    ):
        adapter.capture(item, destination, Event())
        assert adapter.validate(destination) == ()
        adapter.relocate(destination, {"unused": tmp_path / "new-path"})
    assert source.read_bytes() == before
    with closing(sqlite3.connect(destination)) as connection:
        _preserved(connection, expected)
