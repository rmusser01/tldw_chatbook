"""Exact call-source pin migration preserves existing nullable boundaries."""

import sqlite3

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_redaction import CREDENTIAL_FILTER_VERSION
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


def test_v68_upgrade_preserves_old_boundaries_and_checks_exact_call_source(
    tmp_path, monkeypatch
):
    path = tmp_path / "source-pin.sqlite"
    with monkeypatch.context() as historical_target:
        historical_target.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        old = CharactersRAGDB(path, "source-pin-old")
        conversation = old.add_conversation({"title": "source pin"})
        source = old.add_message(
            {"conversation_id": conversation, "sender": "user", "content": "source"}
        )
        other = old.add_message(
            {"conversation_id": conversation, "sender": "user", "content": "other"}
        )
        foreign_conversation = old.add_conversation({"title": "foreign source"})
        foreign_source = old.add_message(
            {
                "conversation_id": foreign_conversation,
                "sender": "user",
                "content": "foreign",
            }
        )
        repository = ConsoleTraceRepository()
        policy = FrozenTracePolicy(
            new_opaque_id(), CREDENTIAL_FILTER_VERSION, False, None
        )
        with old.transaction() as cursor:
            revisions = dict(
                cursor.execute(
                    "SELECT source_message_id, revision_id FROM console_trace_semantic_revisions"
                ).fetchall()
            )
            repository.ensure_policy(cursor, policy)
            segment = repository.create_segment(cursor)
            owner = repository.attach_owner(
                cursor, conversation_id=conversation, root_segment_id=segment.segment_id
            )
            call = repository.reserve_call(
                cursor,
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                turn_id=source,
                run_id=new_opaque_id(),
                call_sequence=0,
                idempotency_key=new_opaque_id(),
                policy_id=policy.policy_id,
            )
            old_event = repository.append_event(
                cursor,
                segment_id=segment.segment_id,
                sequence=0,
                event_type="call_boundary",
                call_id=call.call_id,
            )
        old.close()

    db = CharactersRAGDB(path, "source-pin-new")
    try:
        assert (
            db._get_db_version(db.get_connection())
            == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        )
        with db.transaction() as cursor:
            assert (
                repository.get_latest_call_boundary(cursor, segment.segment_id)
                == old_event
            )
            pinned = repository.append_event(
                cursor,
                segment_id=segment.segment_id,
                sequence=1,
                event_type="call_boundary",
                call_id=call.call_id,
                semantic_revision_id=revisions[source],
            )
            assert pinned.semantic_revision_id == revisions[source]
        for event_type, revision in (
            ("call_boundary", revisions[other]),
            ("call_boundary", revisions[foreign_source]),
            ("call_outcome", revisions[source]),
            ("usage", revisions[source]),
        ):
            with pytest.raises(sqlite3.IntegrityError), db.transaction() as cursor:
                repository.append_event(
                    cursor,
                    segment_id=segment.segment_id,
                    sequence=2,
                    event_type=event_type,
                    call_id=call.call_id,
                    semantic_revision_id=revision,
                )
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as cursor:
            cursor.execute(
                "UPDATE console_trace_events SET semantic_revision_id = ? WHERE event_id = ?",
                (revisions[other], pinned.event_id),
            )
        assert not db.get_connection().execute("PRAGMA foreign_key_check").fetchall()
    finally:
        db.close()


def _schema(connection):
    return tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        )
    )


@pytest.mark.parametrize("fault", ["after_ddl", "version_update"])
def test_source_pin_migration_failure_restores_v68_and_can_retry(
    tmp_path, monkeypatch, fault
):
    path = tmp_path / "rollback.sqlite"
    with chachanotes_db_at_version(path, 68) as db:
        connection = db.get_connection()
        before = _schema(connection)
        execute = db._execute_migration_statements

        def fail_migration(cursor, script, label):
            execute(cursor, script, label)
            if fault == "after_ddl":
                raise sqlite3.OperationalError("injected post-DDL failure")
            cursor.execute(
                "CREATE TRIGGER reject_source_pin_version BEFORE UPDATE OF version "
                "ON db_schema_version WHEN NEW.version = 69 BEGIN "
                "SELECT RAISE(ABORT, 'injected version-update failure'); END"
            )

        with monkeypatch.context() as injected:
            injected.setattr(db, "_execute_migration_statements", fail_migration)
            with pytest.raises(SchemaError):
                db._migrate_from_v68_to_v69(connection)
        assert db._get_db_version(connection) == 68
        assert _schema(connection) == before
        db._migrate_from_v68_to_v69(connection)
        assert db._get_db_version(connection) == 69
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


@pytest.mark.parametrize("damage", ["missing_shape_guard", "renamed_source_column"])
def test_source_pin_migration_rejects_malformed_predecessor_without_changes(
    tmp_path, damage
):
    with chachanotes_db_at_version(tmp_path / "malformed.sqlite", 68) as db:
        with db.transaction() as cursor:
            if damage == "missing_shape_guard":
                cursor.execute("DROP TRIGGER console_trace_events_shape_guard")
            else:
                cursor.execute(
                    "ALTER TABLE console_trace_semantic_revisions "
                    "RENAME COLUMN source_message_id TO unexpected_source_message_id"
                )
        connection = db.get_connection()
        before = _schema(connection)
        with pytest.raises(SchemaError):
            db._migrate_from_v68_to_v69(connection)
        assert db._get_db_version(connection) == 68
        assert _schema(connection) == before


def test_source_pin_migration_requires_v68_and_fresh_schema_matches_upgrade(tmp_path):
    with chachanotes_db_at_version(tmp_path / "upgrade.sqlite", 68) as db:
        connection = db.get_connection()
        db._migrate_from_v68_to_v69(connection)
        upgraded_schema = _schema(connection)
        with pytest.raises(SchemaError, match="requires schema version 68"):
            db._migrate_from_v68_to_v69(connection)
        assert _schema(connection) == upgraded_schema

    with chachanotes_db_at_version(tmp_path / "fresh.sqlite", 69) as fresh:
        assert fresh._get_db_version(fresh.get_connection()) == 69
        assert _schema(fresh.get_connection()) == upgraded_schema
