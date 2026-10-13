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


@pytest.mark.parametrize("note_version", [80, 81])
@pytest.mark.parametrize("subscription_version", [2, 3])
def test_shared_subscription_provenance_and_note_language_catalogs(
    tmp_path, monkeypatch, note_version, subscription_version
):
    """Both native owners share exact catalogs without borrowing DDL authority."""
    import json
    from contextlib import closing

    from tldw_chatbook.DB.recovery_operations import recovery_adapters
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    path = tmp_path / "shared.db"
    subscriptions = SubscriptionsDB(path)
    subscriptions.add_subscription(
        "Historical source", "rss", "https://example.invalid/feed"
    )
    subscriptions.close()
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", note_version)
        notes = CharactersRAGDB(path, client_id="shared-language")
        with notes.transaction() as cursor:
            cursor.execute(
                "INSERT INTO notes(id,title,content,client_id) VALUES (?,?,?,?)",
                ("note", "Source", '\ufeff{"value":1}\r\n', "shared-language"),
            )
        notes.close()
    # ADR-207's v2/v3 DDL is identical. Restore the predecessor stamp and
    # actual historical data rather than constructing a policy-shaped catalog.
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE schema_version SET version=?", (subscription_version,)
        )
        connection.execute(
            "UPDATE subscriptions SET processing_options=?", ('{"keep":"literal"}',)
        )
    owner = next(a for a in recovery_adapters() if a.owner_id == "db.subscriptions")
    assert owner.validate(path) == ()
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT version FROM schema_version").fetchone() == (
            3,
        )
        assert connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
        ).fetchone() == (note_version,)
        options = json.loads(
            connection.execute(
                "SELECT processing_options FROM subscriptions"
            ).fetchone()[0]
        )
        assert options == (
            {"keep": "literal", "url_provenance": "unknown"}
            if subscription_version == 2
            else {"keep": "literal"}
        )
    # The canonical native constructor, not the Subscriptions adapter,
    # advances the embedded note side and retains source bytes.
    notes = CharactersRAGDB(path, client_id="shared-upgrade")
    try:
        row = notes.get_note_by_id("note")
        assert (row["content"], row["content_language"]) == (
            '\ufeff{"value":1}\r\n',
            None,
        )
        assert notes._get_db_version(notes.get_connection()) == 81
    finally:
        notes.close()
    assert owner.validate(path) == ()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE unrecognized(value TEXT)")
    assert validate_candidate(owner, path, Event(), migrate=False) == (
        "unsupported_schema",
    )
