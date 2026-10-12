"""Installed Dreams recovery owns historical schemas and finite native leases."""
import sqlite3
from contextlib import closing
from threading import Event

import pytest

from Tests.Dreams.test_schema_lineage import LAYOUTS, historical_store, snapshot
from tldw_chatbook.DB.Dreams_DB import DreamsDB


def _adapter():
    from tldw_chatbook.DB.recovery_dreams_guardian import recovery_adapters
    return next(owner for owner in recovery_adapters() if owner.owner_id == "db.dreams")


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda row: f"v{row['version']}-{row['source_commit'][:8]}-{row['statements'][-2]}")
def test_installed_dreams_migration_preserves_populated_history(tmp_path, layout):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    path = tmp_path / "dreams.sqlite"
    historical_store(path, layout)
    with closing(sqlite3.connect(path)) as conn:
        before = snapshot(conn)
    adapter = _adapter()
    assert adapter.validate(path) == ()
    assert validate_candidate(adapter, path, Event(), migrate=True) == ()
    assert adapter.validate(path) == ()
    with closing(sqlite3.connect(path)) as conn:
        for table, (columns, rows) in before.items():
            if table == "schema_version":
                assert {row[0] for row in rows} <= {row[0] for row in conn.execute("SELECT version FROM schema_version")}
                continue
            projection = ",".join(f'"{column}"' for column in columns)
            assert conn.execute(f'SELECT {projection} FROM "{table}"').fetchall() == rows
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 6
        migrated = list(conn.iterdump())
    reopened = DreamsDB(path)
    try:
        with reopened.connection() as conn:
            assert list(conn.iterdump()) == migrated
    finally:
        reopened.close_all()


@pytest.mark.parametrize("mutation", ["DELETE FROM schema_version", "INSERT INTO schema_version VALUES (0)",
    "INSERT INTO schema_version VALUES (99)", "ALTER TABLE dream_stories ADD COLUMN foreign_field TEXT"])
def test_dreams_recovery_refuses_invalid_lineage_without_mutation(tmp_path, mutation):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    path = tmp_path / "dreams.sqlite"
    historical_store(path, LAYOUTS[-1])
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(mutation)
        conn.commit()
        before = list(conn.iterdump())
    assert validate_candidate(_adapter(), path, Event(), migrate=True)
    with closing(sqlite3.connect(path)) as conn:
        assert list(conn.iterdump()) == before


def test_dreams_participant_preserves_borrower_and_drains_after_explicit_close(tmp_path):
    import time

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    db = DreamsDB(tmp_path / "dreams.sqlite")
    try:
        connection = db._held_connection()
        participant = _repository_participant(db)
        assert participant.owner_id == "db.dreams"
        assert connection in participant.connections
        connection.execute("BEGIN")
        participant.close_admission()
        try:
            with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
                db.list_profile()
            db.close()
            assert connection.in_transaction
            assert not participant.drain(time.monotonic() + 0.02)
            connection.rollback()
            db.close()
            assert participant.drain(time.monotonic() + 0.1)
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        finally:
            participant.resume()
        assert db.list_profile() == []
    finally:
        db.close_all()


def test_optional_dreams_default_absence_and_explicit_custom_path(tmp_path):
    adapter = _adapter()
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    config = {"paths": {"data_dir": str(tmp_path / "data")}, "general": {"users_name": "owner"}}
    (tmp_path / "data/owner").mkdir(parents=True, mode=0o700)
    (tmp_path / "custom").mkdir(mode=0o700)
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(tmp_path / "config.toml", "dreams-test")
    item, = adapter.discover(config)
    assert item.path == tmp_path / "data/owner/dreams.sqlite"
    assert item.status == "unused"
    custom = tmp_path / "custom/dreams.sqlite"
    config["database"] = {"dreams_db_path": str(custom)}
    item, = adapter.discover(config)
    assert item.path == custom and item.status == "missing_required"
    db = DreamsDB(custom)
    db.close()
    item, = adapter.discover(config)
    assert item.path == custom and item.status == "included"
    assert adapter.activation_required is True




def test_dreams_native_capture_and_relocation_preserve_history(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    source = tmp_path / "dreams.db"
    historical_store(source, LAYOUTS[0])
    db = DreamsDB(source)
    with db.connection() as conn:
        expected = tuple(conn.iterdump())
    db.close()
    before = source.read_bytes()
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = application_authority(tmp_path, source, monkeypatch)
    adapter = _adapter()
    item = StorageItem("db.dreams", "profile:test:db.dreams", source, "included", ())
    destination = stage / "snapshot.db"
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((source,), stage),
    ):
        adapter.capture(item, destination, Event())
        assert adapter.validate(destination) == ()
        adapter.relocate(destination, {"unused": tmp_path / "new-path"})
    assert source.read_bytes() == before
    with closing(sqlite3.connect(destination)) as conn:
        assert tuple(conn.iterdump()) == expected


def test_dreams_migration_authority_is_limited_to_fixed_objects():
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

    with closing(sqlite3.connect(":memory:")) as conn:
        gate = _Restrictions(conn)
        gate.migrating = True
        gate.migration_owner = "db.dreams"
        for table in ("dream_stories", "dream_daily_usage", "guardian_rules", "subscriptions"):
            assert gate.authorize(sqlite3.SQLITE_UPDATE, table, "name", "main", None) == sqlite3.SQLITE_DENY
            assert gate.authorize(sqlite3.SQLITE_DELETE, table, None, "main", None) == sqlite3.SQLITE_DENY
        assert gate.authorize(sqlite3.SQLITE_CREATE_TABLE, "foreign_table", None, "main", None) == sqlite3.SQLITE_DENY
        assert gate.authorize(sqlite3.SQLITE_ALTER_TABLE, "main", "subscriptions", None, None) == sqlite3.SQLITE_DENY
        assert gate.authorize(sqlite3.SQLITE_FUNCTION, None, "load_extension", None, None) == sqlite3.SQLITE_DENY
        assert gate.authorize(sqlite3.SQLITE_READ, "foreign_table", "data", "attached", None) == sqlite3.SQLITE_DENY
        gate.migrating = False
        assert gate.authorize(sqlite3.SQLITE_READ, "sqlite_temp_master", "sql", "temp", None) == sqlite3.SQLITE_DENY


def test_installed_dreams_failed_migration_rolls_back_then_reopens(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import (
        _Restrictions,
        validate_candidate,
    )

    path = tmp_path / "dreams.sqlite"
    historical_store(path, LAYOUTS[0])
    with closing(sqlite3.connect(path)) as conn:
        original = list(conn.iterdump())
    authorize = _Restrictions.authorize
    denied = []
    def fail_after_rebuild(self, action, first, second, database, source):
        if self.migrating and self.migration_owner == "db.dreams" and action == sqlite3.SQLITE_ALTER_TABLE and second == "dream_stories":
            denied.append(True)
            return sqlite3.SQLITE_DENY
        return authorize(self, action, first, second, database, source)
    with monkeypatch.context() as patch:
        patch.setattr(_Restrictions, "authorize", fail_after_rebuild)
        assert validate_candidate(_adapter(), path, Event(), migrate=True)
    assert denied
    with closing(sqlite3.connect(path)) as conn:
        assert list(conn.iterdump()) == original
    assert validate_candidate(_adapter(), path, Event(), migrate=True) == ()
    db = DreamsDB(path)
    db.close_all()


@pytest.mark.parametrize("borrowed", [False, True])
def test_finite_dreams_worker_retires_only_new_native_acquisition(tmp_path, borrowed):
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

    path = tmp_path / "dreams.sqlite"
    def probe():
        db = DreamsDB(path) if borrowed else None
        captured = []
        def read():
            nonlocal db
            if db is None:
                db = DreamsDB(path)
            captured.append(db._held_connection())
            return db.list_profile()
        try:
            assert run_finite_local_worker(read) == []
            if borrowed:
                assert captured[0].execute("SELECT 1").fetchone()[0] == 1
            else:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    captured[0].execute("SELECT 1")
        finally:
            if db is not None:
                db.close_all()
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(probe).result(timeout=5)
        assert executor.submit(lambda: "origin worker still alive").result(timeout=5)
