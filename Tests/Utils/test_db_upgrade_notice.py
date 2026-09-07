"""The pre-boot "upgrading database..." notice (task-21100, AC #2).

Pending ChaChaNotes migrations replay inside ``TldwCli.__init__`` before
anything can paint, so the entry points print a terminal line first. These
tests exercise the probe against tmp databases only -- the ``db_path``
parameter exists precisely so no test ever resolves the live profile path.
"""

from __future__ import annotations

import io
from pathlib import Path

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Utils.db_upgrade_notice import print_db_upgrade_notice_if_pending


def test_notice_prints_for_a_database_behind_the_current_schema(tmp_path: Path):
    db_path = tmp_path / "chachanotes.db"
    with chachanotes_db_at_version(db_path, 45):
        pass

    stream = io.StringIO()
    assert print_db_upgrade_notice_if_pending(db_path, stream=stream) is True
    printed = stream.getvalue()
    assert "Upgrading database" in printed
    assert "v45" in printed
    assert f"v{CharactersRAGDB._CURRENT_SCHEMA_VERSION}" in printed


def test_notice_stays_silent_for_an_up_to_date_database(tmp_path: Path):
    db_path = tmp_path / "chachanotes.db"
    CharactersRAGDB(db_path, client_id="notice-test").close_connection()

    stream = io.StringIO()
    assert print_db_upgrade_notice_if_pending(db_path, stream=stream) is False
    assert stream.getvalue() == ""


def test_notice_stays_silent_for_a_fresh_install(tmp_path: Path):
    stream = io.StringIO()
    assert (
        print_db_upgrade_notice_if_pending(tmp_path / "missing.db", stream=stream)
        is False
    )
    assert stream.getvalue() == ""


def test_notice_swallows_an_unreadable_file(tmp_path: Path):
    """The courtesy line must never become a boot failure of its own."""
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"not a sqlite database")
    stream = io.StringIO()
    assert print_db_upgrade_notice_if_pending(garbage, stream=stream) is False
    assert stream.getvalue() == ""
