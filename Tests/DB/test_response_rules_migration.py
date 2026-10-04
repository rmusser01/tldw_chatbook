"""The installed schema must create local-only tables and migrate atomically."""

import shutil
import sqlite3

import pytest
from contextlib import closing

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_installed_response_rules_are_local_only(chachanotes_template_db, tmp_path):
    path = tmp_path / "rules.sqlite"
    shutil.copyfile(chachanotes_template_db, path)
    db = CharactersRAGDB(path, "rules-migration")
    try:
        with db.transaction() as cursor:
            names = {
                r[0]
                for r in cursor.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table'"
                )
            }
            assert {
                "console_response_rule_revisions",
                "console_response_rule_bindings",
                "console_response_rule_drafts",
                "console_response_rule_fixtures",
                "console_response_rule_validations",
                "console_response_rule_assessments",
                "console_machine_followup_receipts",
            } <= names
            assert db._CURRENT_SCHEMA_VERSION == 77
            assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        db.close()


@pytest.mark.parametrize("version", [75, 76])
def test_historical_migrates_using_installed_sql_without_losing_history(
    tmp_path, chachanotes_template_db, version
):
    from tldw_chatbook.DB.recovery_core_schema import CHACHANOTES_V75_SCHEMA

    path = tmp_path / "historical.sqlite"
    shutil.copyfile(chachanotes_template_db, path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "DROP TRIGGER IF EXISTS console_response_rules_conversation_cleanup"
        )
        names = [
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND (name LIKE 'console_response_rule_%' OR name='console_machine_followup_receipts')"
            )
        ]
        for name in names:
            connection.execute('DROP TABLE "' + name + '"')
        connection.execute(
            "UPDATE db_schema_version SET version=? WHERE schema_name='rag_char_chat_schema'",
            (version,),
        )
        actual = tuple(
            r[0]
            for r in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert actual == CHACHANOTES_V75_SCHEMA
    db = CharactersRAGDB(path, "rules-migration")
    try:
        assert db._get_db_version(db.get_connection()) == 77
        with db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_response_rule_bindings"
                ).fetchone()[0]
                == 0
            )
    finally:
        db.close()
