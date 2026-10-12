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


@pytest.mark.parametrize("version", [76, 77, 78, 79, 80])
def test_restricted_historical_restore_migrates_only_disposable_copy(
    rule_store, tmp_path, version
):
    import shutil

    from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.Chat.response_rules.recovery import prepare_imported_candidate
    from tldw_chatbook.DB.recovery_core import core_adapters

    source = tmp_path / "historical.sqlite"
    with chachanotes_db_at_version(source, version):
        pass
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
            == 81
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM console_response_rule_bindings"
            ).fetchone()[0]
            == 0
        )
    assert source.read_bytes() == before


@pytest.mark.parametrize("lineage", range(4))
def test_every_accepted_v80_catalog_migrates_to_its_exact_v81_variant(
    tmp_path, lineage
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.recovery_core_schema import (
        CHACHANOTES_V80_SCHEMAS,
        CHACHANOTES_V81_SCHEMAS,
    )

    path = tmp_path / "retained-lineage.sqlite"
    with closing(sqlite3.connect(path)) as connection, connection:
        from tldw_chatbook.DB.canvas_payload_validation import (
            install_canvas_revision_payload_validator,
        )

        install_canvas_revision_payload_validator(connection)
        catalog = CHACHANOTES_V80_SCHEMAS[lineage]
        # Virtual tables create their native shadow tables themselves.
        declarations = sorted(
            catalog,
            key=lambda sql: (
                not sql.startswith("CREATE VIRTUAL TABLE"),
                not sql.startswith("CREATE TABLE"),
            ),
        )
        for sql in declarations:
            if sql.startswith("CREATE TABLE sqlite_sequence"):
                continue
            try:
                connection.execute(sql)
            except sqlite3.OperationalError as error:
                if "already exists" not in str(error):
                    raise
                assert connection.execute(
                    "SELECT 1 FROM sqlite_schema WHERE sql=?", (sql,)
                ).fetchone()
        connection.execute(
            "INSERT INTO db_schema_version(schema_name,version) VALUES ('rag_char_chat_schema',80)"
        )
        connection.execute(
            "INSERT INTO conversations(id,root_id,title,client_id) VALUES ('retained','retained','saved lineage','rules')"
        )
    path.chmod(0o600)
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as connection:
        catalog = tuple(
            r[0]
            for r in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert catalog == CHACHANOTES_V81_SCHEMAS[lineage]
        assert connection.execute(
            "SELECT title FROM conversations WHERE id='retained'"
        ).fetchone() == ("saved lineage",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("damage", ["future", "hybrid", "changed_index"])
def test_response_rule_candidate_refusal_never_rewrites_rejected_input(
    rule_store, tmp_path, damage
):
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.private_sqlite import copy_private_sqlite
    from tldw_chatbook.DB.recovery_core import core_adapters

    _, db, _ = rule_store
    path = tmp_path / "rejected.sqlite"
    copy_private_sqlite("recovery.core.chachanotes", db.db_path, path)
    with closing(sqlite3.connect(path)) as connection, connection:
        if damage == "future":
            connection.execute(
                "UPDATE db_schema_version SET version=82 WHERE schema_name='rag_char_chat_schema'"
            )
        elif damage == "hybrid":
            connection.execute("DROP INDEX idx_response_rule_drafts_scope")
        else:
            connection.execute("DROP INDEX idx_response_rule_drafts_scope")
            connection.execute(
                "CREATE INDEX idx_response_rule_drafts_scope ON console_response_rule_drafts(scope_id,scope_kind)"
            )
    before = path.read_bytes()
    owner = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    expected = (
        "unsupported_schema_version" if damage == "future" else "unsupported_schema"
    )
    assert expected in validate_candidate(owner, path, Event(), migrate=True)
    assert path.read_bytes() == before
