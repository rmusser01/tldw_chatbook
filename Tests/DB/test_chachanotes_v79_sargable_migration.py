"""ADR-224: sargable v78->v79 upgrade keeps recovery validation's authority.

Mirrors ``test_chachanotes_v78_fleet_progress_migration.py``: the restore
path must be able to migrate a genuinely-v78 candidate to v79 under the
restricted connection (the authorizer admits exactly the installed .sql
file's actions -- normalization UPDATEs, the two indexes, the recreated
sync trigger), refuse foreign DDL, and the resulting catalog must validate
cleanly at the v79 head, for both the primary and the shared-file hybrid
(Subscriptions + ChaChaNotes) shapes.
"""

from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from Tests.ChaChaNotesDB.test_sargable_timestamps import (
    CANONICAL_RE,
    seed_mixed_conversations,
    seed_mixed_flashcards,
)
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.recovery_core import core_adapters

pytestmark = pytest.mark.bootstrap_profile


def _owner():
    return next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")


def _v78_with_mixed_data(path: Path, *, seed: bool) -> None:
    with chachanotes_db_at_version(path, 78, client_id="sargable-recovery") as db:
        if seed:
            seed_mixed_conversations(db)
            seed_mixed_flashcards(db)


@pytest.mark.parametrize("seed", [False, True])
def test_recovery_migration_to_v79_normalizes_and_validates(tmp_path, seed):
    path = tmp_path / "installed.sqlite"
    _v78_with_mixed_data(path, seed=seed)
    assert validate_candidate(_owner(), path, Event(), migrate=True) == ()
    with closing(_connect(path)) as connection:
        assert connection.execute(
            "SELECT version FROM db_schema_version"
        ).fetchone() == (79,)
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        if seed:
            values = [
                row[0]
                for row in connection.execute(
                    "SELECT CAST(last_modified AS TEXT) FROM conversations "
                    "WHERE character_id = 1"
                )
            ]
            cards = [
                row[0]
                for row in connection.execute(
                    "SELECT CAST(next_review AS TEXT) FROM flashcards "
                    "WHERE next_review IS NOT NULL"
                )
            ]
            assert all(CANONICAL_RE.match(v) for v in values + cards)
            assert [
                row[0]
                for row in connection.execute(
                    "SELECT id FROM conversations WHERE character_id = 1 "
                    "AND deleted = 0 AND scope_type = 'global' AND archived = 0 "
                    "ORDER BY last_modified DESC, id DESC"
                )
            ] == [
                "c-new",
                "c-space-later",
                "a-canon-earlier",
                "z-tie-space",
                "a-tie-canonical",
                "c-oldest",
            ]
    assert validate_candidate(_owner(), path, Event(), migrate=False) == ()


def test_installed_sargable_migration_refuses_foreign_ddl_and_rolls_back(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_chachanotes_native76_compatibility import _dump

    path = tmp_path / "rollback.sqlite"
    _v78_with_mixed_data(path, seed=True)
    before = _dump(path)
    read = Path.read_text

    def foreign_ddl(file, *args, **kwargs):
        source = read(file, *args, **kwargs)
        if file.name == ("chachanotes_v78_to_v79_sargable_timestamp_normalization.sql"):
            source += "\nCREATE TABLE foreign_sargable(secret TEXT);\n"
        return source

    monkeypatch.setattr(Path, "read_text", foreign_ddl)
    assert validate_candidate(_owner(), path, Event(), migrate=True) == (
        "sqlite_validation_unavailable",
    )
    assert _dump(path) == before


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("stamp", [77, 78, 79])
def test_current_sargable_catalog_accepts_only_its_stamp(tmp_path, shared, stamp):
    from tldw_chatbook.DB.recovery_operations import recovery_adapters
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    path = tmp_path / "catalog.sqlite"
    if shared:
        subscriptions = SubscriptionsDB(path)
        subscriptions.close()
    db = CharactersRAGDB(path, "sargable-stamp")
    db.get_connection().execute("UPDATE db_schema_version SET version=?", (stamp,))
    db.get_connection().commit()
    db.close_connection()
    owner = (
        next(a for a in recovery_adapters() if a.owner_id == "db.subscriptions")
        if shared
        else _owner()
    )
    assert validate_candidate(owner, path, Event(), migrate=False) == (
        () if stamp == 79 else ("unsupported_schema_version",)
    )


def _connect(path: Path):
    import sqlite3

    return sqlite3.connect(str(path))
