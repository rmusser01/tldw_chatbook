"""Language schema follows the current strict backup catalog owner."""

import sqlite3
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.recovery_core import core_adapters


@pytest.mark.parametrize("version", [80, 81])
def test_note_language_backup_catalog_and_upgrade(tmp_path, monkeypatch, version):
    path = tmp_path / "candidate.db"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", version)
        db = CharactersRAGDB(path, client_id="recovery-language")
        db.get_connection().execute(
            "INSERT INTO notes(id,title,content,client_id) VALUES ('note','Title','exact source','test')"
        )
        db.get_connection().commit()
        db.close_connection()
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
        ).fetchone() == (81,)
    db = CharactersRAGDB(path, client_id="verify")
    try:
        row = db.get_note_by_id("note")
        assert row["content"] == "exact source"
        assert row["content_language"] is None
        assert db._get_db_version(db.get_connection()) == 81
        db.get_connection().execute("CREATE TABLE unrecognized(value TEXT)")
        db.get_connection().commit()
    finally:
        db.close_connection()
    assert validate_candidate(owner, path, Event(), migrate=False) == (
        "unsupported_schema",
    )
