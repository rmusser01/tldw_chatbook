"""Installed optional Guardian schemas, paths and migration authority."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from Tests.Guardian.test_crisis_provenance import _legacy
from tldw_chatbook.DB.Guardian_DB import GuardianDB


def _catalog(path):
    with closing(sqlite3.connect(path)) as conn:
        return tuple(row[0] for row in conn.execute("SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"))


def _adapter():
    from tldw_chatbook.DB.recovery_dreams_guardian import recovery_adapters
    return next(a for a in recovery_adapters() if a.owner_id == "db.guardian")


def test_native_guardian_catalogs_match_installed_policy(tmp_path):
    old = tmp_path / "legacy.db"
    _legacy(old)
    v1 = _catalog(old)
    migrated = GuardianDB(old)
    migrated.close()
    v2_migrated = _catalog(old)
    fresh_path = tmp_path / "fresh.db"
    fresh = GuardianDB(fresh_path)
    fresh.close()
    v2_fresh = _catalog(fresh_path)
    receipt = {"source_v1": "0c941ba2ec984d960ba83376aeace0e830452bbc exact DDL fixture",
        "catalogs": [[1, v1], [2, v2_migrated], [2, v2_fresh]]}
    evidence = Path("/Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-wave2/pr2921-current-integration")
    measured = evidence / "guardian-native-catalogs-v1-v2.json"
    if measured.exists():
        assert json.loads(measured.read_text()) == json.loads(json.dumps(receipt))
    else:
        measured.write_text(json.dumps(receipt, indent=2))
    adapter = _adapter()
    for version, catalog in receipt["catalogs"]:
        assert (version, tuple(catalog)) in adapter.schema_policy().schema_sql
    assert adapter.validate(fresh_path) == ()
    assert adapter.validate(old) == ()


def test_installed_guardian_recovery_migrates_populated_v1(tmp_path):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    path = tmp_path / "legacy.db"
    _legacy(path)
    adapter = _adapter()
    assert validate_candidate(adapter, path, Event(), migrate=True) == ()
    db = GuardianDB(path)
    try:
        with db.connection() as conn:
            assert conn.execute("SELECT crisis_provenance FROM guardian_alerts WHERE id=90").fetchone()[0] == -1
            assert conn.execute("SELECT updated_at FROM guardian_rules WHERE id=90").fetchone()[0].startswith("2000-01-01")
    finally:
        db.close()


def test_optional_guardian_default_absence_and_explicit_custom_path(tmp_path):
    adapter = _adapter()
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    config = {"paths": {"data_dir": str(tmp_path / "data")}, "general": {"users_name": "owner"}}
    (tmp_path / "data/owner").mkdir(parents=True, mode=0o700)
    (tmp_path / "custom").mkdir(mode=0o700)
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(tmp_path / "config.toml", "guardian-test")
    item, = adapter.discover(config)
    assert item.path == tmp_path / "data/owner/guardian.sqlite"
    assert item.status == "unused"
    custom = tmp_path / "custom/guardian.sqlite"
    config["database"] = {"guardian_db_path": str(custom)}
    item, = adapter.discover(config)
    assert item.path == custom and item.status == "missing_required"
    db = GuardianDB(custom)
    db.close()
    item, = adapter.discover(config)
    assert item.path == custom and item.status == "included"
    assert adapter.activation_required is True


def test_guardian_maintenance_counts_native_handles_and_preserves_borrower(tmp_path):
    import time

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    db = GuardianDB(tmp_path / "guardian.db")
    try:
        connection = db._held_connection()
        participant = _repository_participant(db)
        assert participant.owner_id == "db.guardian"
        assert connection in participant.connections
        connection.execute("BEGIN")
        participant.close_admission()
        try:
            with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
                db.list_rules()
            db.close()
            assert connection.in_transaction
            assert not participant.drain(time.monotonic() + 0.02)
            connection.rollback()
            db.close()
            assert participant.drain(time.monotonic() + 0.1)
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
        finally:
            participant.resume()
        assert db.list_rules()
    finally:
        db.close()


@pytest.mark.parametrize("damage", ["future", "empty", "hybrid"])
def test_installed_guardian_refuses_bad_lineage_without_mutation(tmp_path, damage):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    path = tmp_path / "guardian.db"
    _legacy(path)
    with closing(sqlite3.connect(path)) as conn:
        if damage == "future":
            conn.execute("UPDATE schema_version SET version=99")
        elif damage == "empty":
            conn.execute("DELETE FROM schema_version")
        else:
            conn.execute("UPDATE schema_version SET version=2")
        conn.commit()
        before = list(conn.iterdump())
    assert validate_candidate(_adapter(), path, Event(), migrate=True)
    with closing(sqlite3.connect(path)) as conn:
        assert list(conn.iterdump()) == before


def test_guardian_migration_authority_denies_other_domain_writes():
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions
    with closing(sqlite3.connect(":memory:")) as conn:
        gate = _Restrictions(conn)
        gate.migrating = True
        gate.migration_owner = "db.guardian"
        for table in ("guardian_rules", "subscriptions", "dream_interest_profile"):
            assert gate.authorize(sqlite3.SQLITE_UPDATE, table, "name", "main", None) == sqlite3.SQLITE_DENY
            assert gate.authorize(sqlite3.SQLITE_DELETE, table, None, "main", None) == sqlite3.SQLITE_DENY
        assert gate.authorize(sqlite3.SQLITE_ALTER_TABLE, "main", "subscriptions", None, None) == sqlite3.SQLITE_DENY
        assert gate.authorize(sqlite3.SQLITE_FUNCTION, None, "load_extension", None, None) == sqlite3.SQLITE_DENY


def test_guardian_native_capture_and_relocation_preserve_history(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    source = tmp_path / "guardian.db"
    _legacy(source)
    db = GuardianDB(source)
    with db.connection() as conn:
        expected = tuple(conn.iterdump())
    db.close()
    before = source.read_bytes()
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = application_authority(tmp_path, source, monkeypatch)
    adapter = _adapter()
    item = StorageItem("db.guardian", "profile:test:db.guardian", source, "included", ())
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
