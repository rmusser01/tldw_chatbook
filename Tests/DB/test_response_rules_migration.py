"""The installed schema must create local-only tables and migrate atomically."""

import shutil

import pytest

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
            assert db._CURRENT_SCHEMA_VERSION == 81
            assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        db.close()


@pytest.mark.parametrize("version", [76, 77, 78, 79, 80])
def test_historical_migrates_using_installed_sql_without_losing_history(
    tmp_path, version
):
    from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
    from tldw_chatbook.DB.recovery_core_schema import CHACHANOTES_V81_SCHEMAS

    path = tmp_path / "historical.sqlite"
    with chachanotes_db_at_version(path, version) as predecessor:
        from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

        persistence = ChatPersistenceService(predecessor)
        conversation_id = persistence.create_conversation(
            conversation_title="retained conversation"
        )
        message_id = persistence.create_message(
            conversation_id=conversation_id, sender="user", content="retained body"
        )
    db = CharactersRAGDB(path, "rules-migration")
    try:
        assert db._get_db_version(db.get_connection()) == 81
        with db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT content FROM messages WHERE id=?", (message_id,)
                ).fetchone()[0]
                == "retained body"
            )
            catalog = tuple(
                row[0]
                for row in cursor.execute(
                    "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                )
            )
            assert catalog in CHACHANOTES_V81_SCHEMAS
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_response_rule_bindings"
                ).fetchone()[0]
                == 0
            )
            assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        db.close()
    reopened = CharactersRAGDB(path, "rules-reopen")
    try:
        assert reopened._get_db_version(reopened.get_connection()) == 81
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "index,query,params",
    [
        (
            "idx_machine_followup_receipts_conversation",
            "SELECT * FROM console_machine_followup_receipts WHERE conversation_id=?",
            ("chat",),
        ),
        (
            "idx_response_rule_assessments_source",
            "SELECT * FROM console_response_rule_assessments WHERE conversation_id=? AND message_id=?",
            ("chat", "answer"),
        ),
        (
            "idx_response_rule_drafts_scope",
            "SELECT * FROM console_response_rule_drafts WHERE scope_kind=? AND scope_id=?",
            ("chat", "chat"),
        ),
    ],
)
def test_rule_lookup_indexes_without_statistics(
    chachanotes_template_db, tmp_path, index, query, params
):
    path = tmp_path / "index.sqlite"
    shutil.copyfile(chachanotes_template_db, path)
    db = CharactersRAGDB(path, "rule-indexes")
    try:
        with db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT name FROM sqlite_schema WHERE name='sqlite_stat1'"
                ).fetchall()
                == []
            )
            plan = cursor.execute("EXPLAIN QUERY PLAN " + query, params).fetchall()
            assert any(index in row[3] for row in plan), tuple(
                tuple(row) for row in plan
            )
    finally:
        db.close()
