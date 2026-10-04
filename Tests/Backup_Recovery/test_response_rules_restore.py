"""Imported activations are inert; rollback keeps the original SQLite state."""

import sqlite3
from contextlib import closing
from dataclasses import replace
from threading import Event

import pytest

from Tests.Chat.response_rules_fixtures import inputs
from Tests.Chat.response_rules_store_fixtures import activate
from Tests.Chat.response_rules_store_fixtures import (
    rule_store as rule_store,  # noqa: PLC0414 - pytest fixture registration
)
from tldw_chatbook.Chat.response_rules.evaluator import aggregate_checks
from tldw_chatbook.Chat.response_rules.recovery import prepare_imported_rules


def test_restore_imports_inactive_without_pending_work(rule_store):
    store, db, origin = rule_store
    binding = activate(store, origin)
    store.save_assessment(
        replace(aggregate_checks(origin, (), inputs=inputs()), state="pending")
    )
    with db.transaction() as cursor:
        prepare_imported_rules(cursor)
    assert store.list_bindings(binding.scope)[0].state == "disabled"
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_response_rule_assessments WHERE state='pending'"
            ).fetchone()[0]
            == 0
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_machine_followup_receipts"
            ).fetchone()[0]
            == 0
        )


def test_internal_rollback_preserves_prior_binding_state(rule_store):
    store, db, origin = rule_store
    binding = activate(store, origin)
    with pytest.raises(RuntimeError), db.transaction() as cursor:
        prepare_imported_rules(cursor)
        raise RuntimeError("rollback")
    assert store.list_bindings(binding.scope) == (binding,)


def test_installed_recovery_policy_migrates_only_disposable_candidate(
    rule_store, tmp_path
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_core import core_adapters

    _, db, _ = rule_store
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    assert db._CURRENT_SCHEMA_VERSION in owner.schema_policy().versions
    assert validate_candidate(owner, db.db_path, Event(), migrate=False) == ()


def test_restricted_import_preparation_disables_only_qualified_selected_payload(
    rule_store, tmp_path
):
    from tldw_chatbook.Chat.response_rules.recovery import prepare_imported_candidate
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

    store, db, origin = rule_store
    binding = activate(store, origin)
    candidate = tmp_path / "import.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", db.db_path, candidate)
    prepare_imported_candidate(candidate, Event())
    assert store.list_bindings(binding.scope) == (binding,)
    with closing(sqlite3.connect(candidate)) as connection:
        assert (
            connection.execute(
                "SELECT state FROM console_response_rule_bindings"
            ).fetchone()[0]
            == "disabled"
        )


def test_restricted_import_preparation_refuses_an_unknown_table(rule_store, tmp_path):
    from tldw_chatbook.Chat.response_rules.recovery import prepare_imported_candidate
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

    _, db, _ = rule_store
    candidate = tmp_path / "import.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", db.db_path, candidate)
    with closing(sqlite3.connect(candidate)) as connection, connection:
        connection.execute("CREATE TABLE foreign_payload(value TEXT)")
    with pytest.raises(ValueError, match="unsupported_schema"):
        prepare_imported_candidate(candidate, Event())


def test_restricted_v75_restore_migrates_only_disposable_copy(rule_store, tmp_path):
    import shutil

    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.Chat.response_rules.recovery import prepare_imported_candidate
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite
    from tldw_chatbook.DB.recovery_core import core_adapters

    _, db, _ = rule_store
    source = tmp_path / "historical.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", db.db_path, source)
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DROP TRIGGER console_response_rules_conversation_cleanup")
        names = [
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND (name LIKE 'console_response_rule_%' OR name='console_machine_followup_receipts')"
            )
        ]
        for name in names:
            connection.execute('DROP TABLE "' + name + '"')
        connection.execute(
            "UPDATE db_schema_version SET version=75 WHERE schema_name='rag_char_chat_schema'"
        )
    before = source.read_bytes()
    candidate = tmp_path / "working-import.sqlite"
    shutil.copyfile(source, candidate)
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    assert validate_candidate(owner, candidate, Event(), migrate=True) == ()
    prepare_imported_candidate(candidate, Event())
    with closing(sqlite3.connect(candidate)) as connection:
        assert (
            connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 76
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM console_response_rule_bindings"
            ).fetchone()[0]
            == 0
        )
    assert source.read_bytes() == before
