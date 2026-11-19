"""Installed recovery preserves operational records in disposable DB candidates."""

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from Tests.Chat.test_console_trace_runtime import (
    make_database as make_database,  # noqa: PLC0414 - shared pytest fixture
)
from Tests.DB.test_console_prompt_mixins import attempt, saved_turn
from tldw_chatbook.DB.console_prompt_mixins import read_attempt, write_attempt

pytestmark = pytest.mark.bootstrap_profile


def test_recovery_candidate_retains_capture_off_attempt(
    make_database, tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_core import core_adapters

    db = make_database(tmp_path / "original.sqlite", "mixin-restore")
    cid, ids = saved_turn(db)
    with db.transaction() as cursor:
        write_attempt(cursor, cid, batch=attempt(), native_message_ids=ids)
        assert (
            cursor.execute("SELECT COUNT(*) FROM message_exchanges").fetchone()[0] == 0
        )
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    source = Path(db.db_path)
    db.close()
    adapter = next(
        owner for owner in core_adapters() if owner.owner_id == "db.chachanotes.primary"
    )
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "capture"
    stage.mkdir(mode=0o700)
    candidate = stage / "restore.sqlite"
    source_bytes = source.read_bytes()
    item = StorageItem(adapter.owner_id, "fixture:mixins", source, "included", ())
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 2) as session,
        session.capture_scope((source,), stage),
    ):
        adapter.capture(item, candidate, Event())
    assert validate_candidate(adapter, candidate, Event(), migrate=True) == ()
    with closing(sqlite3.connect(candidate)) as connection:
        assert (
            read_attempt(connection.cursor(), cid, attempt_id="attempt").snapshot
            == attempt().snapshot
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert Path(db.db_path).read_bytes() == source_bytes


@pytest.mark.parametrize("version", [81, 85])
def test_historical_candidate_migrates_without_touching_source(tmp_path, version):
    from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite
    from tldw_chatbook.DB.recovery_core import core_adapters

    source = tmp_path / "old.sqlite"
    with chachanotes_db_at_version(source, version) as db:
        cid, _ = saved_turn(db)
    before = source.read_bytes()
    candidate = tmp_path / "candidate.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", source, candidate)
    adapter = next(
        owner for owner in core_adapters() if owner.owner_id == "db.chachanotes.primary"
    )
    assert validate_candidate(adapter, candidate, Event(), migrate=True) == ()
    with closing(sqlite3.connect(candidate)) as connection:
        assert (
            connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 86
        )
        assert (
            read_attempt(connection.cursor(), cid, attempt_id="legacy").status
            == "legacy"
        )
    assert source.read_bytes() == before
