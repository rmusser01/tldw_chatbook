"""Fleet progress follows shipped native receipts without broad recovery authority."""

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_chachanotes_native76_compatibility import (
    _checkpoint_state,
    _dump,
    _shipped76,
)
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.recovery_core import core_adapters

pytestmark = pytest.mark.bootstrap_profile

# The chain lands every candidate at the current schema: the literal 80 went
# stale at v81 and each later bump (TASK-42516).
CURRENT = CharactersRAGDB._CURRENT_SCHEMA_VERSION


def _owner():
    return next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("route", ["recovery76", "recovery77", "constructor77"])
def test_fleet_upgrade_preserves_native_receipts_and_predecessor_rows(
    tmp_path, dictionary, route
):
    path = tmp_path / "installed.sqlite"
    _shipped76(path, dictionary=dictionary)
    if route.endswith("77"):
        with chachanotes_db_at_version(path, 77):
            pass
    before, messages, indexes = _checkpoint_state(path)
    if route.startswith("recovery"):
        assert validate_candidate(_owner(), path, Event(), migrate=True) == ()
    else:
        db = CharactersRAGDB(path, "fleet-upgrade")
        db.close_connection()
    after, actual_messages, actual_indexes = _checkpoint_state(path)
    if route == "recovery76":
        for row in after:
            assert row.pop("agent_chat_start_attempt_id") is None
    # v86 (ADR-286) added the mix-in column; a predecessor row has none.
    for row in after:
        assert row.pop("mixin_snapshot_schema_version") is None
    assert after == before
    assert actual_messages == messages
    assert all(index in actual_indexes for index in indexes)
    assert validate_candidate(_owner(), path, Event(), migrate=False) == ()
    with closing(sqlite3.connect(path)) as connection:
        # The chain (constructor and recovery routes alike) lands these
        # candidates at the current head.
        assert connection.execute(
            "SELECT version FROM db_schema_version"
        ).fetchone() == (CURRENT,)
        assert connection.execute(
            "SELECT count(*) FROM fleet_progress_messages"
        ).fetchone() == (0,)
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


def test_installed_fleet_migration_refuses_foreign_ddl_and_rolls_back(
    tmp_path, monkeypatch
):
    path = tmp_path / "rollback.sqlite"
    _shipped76(path)
    with chachanotes_db_at_version(path, 77):
        pass
    before = _dump(path)
    read = Path.read_text

    def foreign_ddl(file, *args, **kwargs):
        source = read(file, *args, **kwargs)
        if file.name == "chachanotes_v77_to_v78_fleet_progress.sql":
            source += "\nCREATE TABLE foreign_progress(secret TEXT);\n"
        return source

    monkeypatch.setattr(Path, "read_text", foreign_ddl)
    assert validate_candidate(_owner(), path, Event(), migrate=True) == (
        "sqlite_validation_unavailable",
    )
    assert _dump(path) == before


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("stamp", [77, 78, 79, 80, CURRENT])
def test_current_fleet_catalog_accepts_only_its_complete_matching_stamp(
    tmp_path, shared, stamp
):
    from tldw_chatbook.DB.recovery_operations import recovery_adapters
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    path = tmp_path / "catalog.sqlite"
    if shared:
        subscriptions = SubscriptionsDB(path)
        subscriptions.close()
    db = CharactersRAGDB(path, "fleet-stamp")
    db.get_connection().execute("UPDATE db_schema_version SET version=?", (stamp,))
    db.get_connection().commit()
    db.close_connection()
    owner = (
        next(a for a in recovery_adapters() if a.owner_id == "db.subscriptions")
        if shared
        else _owner()
    )
    # Only the current catalog's own complete stamp validates read-only; 80
    # (ADR-216's browse-order head when this was written) is now a stale one.
    assert validate_candidate(owner, path, Event(), migrate=False) == (
        () if stamp == CURRENT else ("unsupported_schema_version",)
    )
