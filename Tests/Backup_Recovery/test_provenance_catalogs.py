"""Native fresh and migrated provenance schemas must be recovery-qualified."""

import json
import sqlite3
from contextlib import closing

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.recovery_core import core_adapters
from tldw_chatbook.DB.recovery_core_schema import CORE_SCHEMAS


@pytest.mark.parametrize("lineage", ["fresh", "v4", "v4-recovery"])
def test_collections_provenance_catalog_matches_native_constructor(
    tmp_path, lineage, monkeypatch
):
    path = tmp_path / "collections.db"
    if lineage.startswith("v4"):
        old_sql = next(
            sql
            for owner, version, schema in CORE_SCHEMAS
            if owner == "db.library_collections"
            for sql in schema
            if sql.startswith("CREATE TABLE collection_capture_items")
        )
        with closing(sqlite3.connect(path)) as conn:
            for statement in LibraryCollectionsDB._LEGACY_SCHEMA_DDL:
                conn.execute(statement)
            for statement in LibraryCollectionsDB._CAPTURE_SCHEMA_DDL:
                conn.execute(
                    old_sql
                    if "CREATE TABLE IF NOT EXISTS collection_capture_items"
                    in statement
                    else statement
                )
            for statement in LibraryCollectionsDB._REVIEW_SET_SCHEMA_DDL:
                conn.execute(statement)
            conn.execute("INSERT INTO schema_version VALUES (4)")
            conn.commit()
    if lineage == "v4-recovery":
        from threading import Event

        from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

        adapter = next(
            a for a in core_adapters() if a.owner_id == "db.library_collections"
        )
        from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

        denied = []
        original = _Restrictions.authorize

        def audit(self, *args):
            result = original(self, *args)
            if result == sqlite3.SQLITE_DENY:
                denied.append(args)
            return result

        monkeypatch.setattr(_Restrictions, "authorize", audit)
        assert validate_candidate(adapter, path, Event(), migrate=True) == (), denied
    db = LibraryCollectionsDB(path)
    try:
        with db.connection() as conn:
            actual = tuple(
                row[0]
                for row in conn.execute(
                    "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                )
            )
        (tmp_path / "native-schema.json").write_text(json.dumps(actual, indent=2))
        adapter = next(
            a for a in core_adapters() if a.owner_id == "db.library_collections"
        )
        assert (5, actual) in adapter.schema_policy().schema_sql
        assert adapter.validate(path) == ()
    finally:
        db.close()


@pytest.mark.parametrize("owner", ["db.library_collections", "db.subscriptions"])
def test_provenance_migration_gate_refuses_unrelated_authority(owner):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

    with closing(sqlite3.connect(":memory:")) as connection:
        restrictions = _Restrictions(connection)
        restrictions.migrating = True
        restrictions.migration_owner = owner
        for table in ("guardian_rules", "dream_interest_profile", "conversations"):
            assert (
                restrictions.authorize(
                    sqlite3.SQLITE_UPDATE, table, "name", "main", None
                )
                == sqlite3.SQLITE_DENY
            )
            assert (
                restrictions.authorize(sqlite3.SQLITE_DELETE, table, None, "main", None)
                == sqlite3.SQLITE_DENY
            )
        assert (
            restrictions.authorize(
                sqlite3.SQLITE_ALTER_TABLE, "main", "subscriptions", None, None
            )
            == sqlite3.SQLITE_DENY
        )
        assert (
            restrictions.authorize(
                sqlite3.SQLITE_FUNCTION, None, "load_extension", None, None
            )
            == sqlite3.SQLITE_DENY
        )
