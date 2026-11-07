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
    assert after == before
    assert actual_messages == messages
    assert all(index in actual_indexes for index in indexes)
    assert validate_candidate(_owner(), path, Event(), migrate=False) == ()
    with closing(sqlite3.connect(path)) as connection:
        # ADR-224 moved the head to v79; the chain (constructor and recovery
        # routes alike) now lands these candidates there.
        assert connection.execute(
            "SELECT version FROM db_schema_version"
        ).fetchone() == (80,)
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
@pytest.mark.parametrize("stamp", [77, 78, 79, 80])
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
    # ADR-216: the current catalog is the v80 browse-order one; only its
    # own complete stamp validates read-only.
    assert validate_candidate(owner, path, Event(), migrate=False) == (
        () if stamp == 80 else ("unsupported_schema_version",)
    )
