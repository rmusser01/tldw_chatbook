"""Real SQLite voice import grants, rollback and GC epoch boundaries."""

import sqlite3
import pytest

from Tests.Chat.test_console_voice_trace_repository import (
    _completed_pair,
    _post_dispatch_import,
)
from Tests.Chat.test_console_trace_graph_gc import _finish_legacy_migration
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_maintenance import TraceGarbageCollector
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    ConfirmedPreCommitTraceImportError,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _authorized(connection, call_id):
    return connection.execute(
        "SELECT console_voice_trace_import_authorized(?)", (call_id,)
    ).fetchone()[0]


def test_managed_exact_grant_is_transaction_local_and_not_inherited_on_reopen(tmp_path):
    db = CharactersRAGDB(tmp_path / "grant.db", "voice-grant")
    foreign = sqlite3.connect(":memory:")
    try:
        connection = db.get_connection()
        grant = db._voice_trace_import_authorization_for_repository(connection)
        assert _authorized(connection, "call") == 0
        with pytest.raises(RuntimeError, match="transaction_required"):
            with grant._authorize(("call",)):
                pytest.fail("grant outside transaction")
        with pytest.raises(RuntimeError, match="connection_mismatch"):
            db._voice_trace_import_authorization_for_repository(foreign)
        with db.transaction(immediate=True):
            with pytest.raises(LookupError):
                with grant._authorize(("call",)):
                    assert _authorized(connection, "call") == 1
                    assert _authorized(connection, "foreign-call") == 0
                    with pytest.raises(RuntimeError, match="already_active"):
                        with grant._authorize(("other",)):
                            pytest.fail("nested grant")
                    raise LookupError("fixture")
            assert _authorized(connection, "call") == 0
        with db.quiesce_connections(timeout_seconds=0.5):
            pass
        reopened = db.get_connection()
        assert reopened is not connection
        assert _authorized(reopened, "call") == 0
        assert (
            db._voice_trace_import_authorization_for_repository(reopened) is not grant
        )
        with pytest.raises(RuntimeError, match="connection_mismatch"):
            db._voice_trace_import_authorization_for_repository(connection)
    finally:
        foreign.close()
        db.close_connection()


def test_insert_phase_exception_clears_exact_grant_and_rolls_back_graph():
    db = CharactersRAGDB(":memory:", "voice-grant-rollback")
    try:
        conversation, user, assistant = _completed_pair(db)
        request = _post_dispatch_import(
            db,
            conversation_id=conversation,
            user_message_id=user,
            assistant_message_id=assistant,
        )
        connection = db.get_connection()
        connection.execute("""CREATE TEMP TRIGGER fail_voice_second_insert
            BEFORE INSERT ON console_trace_calls WHEN NEW.call_sequence = 1
            BEGIN SELECT RAISE(ABORT, 'fixture insertion failure'); END""")
        with pytest.raises(ConfirmedPreCommitTraceImportError):
            ConsoleTraceRepository().import_post_dispatch_trace(db, request)
        for call in request.calls:
            assert _authorized(connection, call.call_id) == 0
        for table in (
            "console_trace_calls",
            "console_trace_owners",
            "console_trace_segments",
            "console_trace_events",
        ):
            assert (
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
            )
        connection.execute("DROP TRIGGER fail_voice_second_insert")
        assert (
            not ConsoleTraceRepository()
            .import_post_dispatch_trace(db, request)
            .already_imported
        )
    finally:
        db.close_connection()


def test_voice_import_invalidates_existing_gc_mark_and_retry_is_read_only():
    db = CharactersRAGDB(":memory:", "voice-gc")
    try:
        conversation, user, assistant = _completed_pair(db)
        request = _post_dispatch_import(
            db,
            conversation_id=conversation,
            user_message_id=user,
            assistant_message_id=assistant,
        )
        _finish_legacy_migration(db)
        collector = TraceGarbageCollector(db)
        marked = collector.mark(request_id="before-voice")
        repository = ConsoleTraceRepository()
        imported = repository.import_post_dispatch_trace(db, request)
        connection = db.get_connection()
        epoch = repository.get_graph_epoch(connection.cursor())
        assert epoch > marked.marked_epoch
        swept = collector.sweep(request_id="before-voice")
        assert swept.status == "stale_epoch" and swept.swept_epoch is None
        count = connection.execute(
            "SELECT count(*) FROM console_trace_events"
        ).fetchone()[0]
        assert repository.import_post_dispatch_trace(db, request).already_imported
        assert repository.get_graph_epoch(connection.cursor()) == epoch
        assert (
            connection.execute("SELECT count(*) FROM console_trace_events").fetchone()[
                0
            ]
            == count
        )
        grant = db._voice_trace_import_authorization_for_repository(connection)
        with db.transaction(immediate=True) as cursor:
            with grant._authorize(imported.call_ids):
                with pytest.raises(
                    sqlite3.IntegrityError, match="trace GC deletion authorization"
                ):
                    cursor.execute(
                        "DELETE FROM console_trace_segments WHERE segment_id = ?",
                        (imported.segment_id,),
                    )
                with pytest.raises(
                    sqlite3.IntegrityError, match="semantic mutation authorization"
                ):
                    cursor.execute(
                        "UPDATE messages SET content = 'unauthorized' WHERE id = ?",
                        (user,),
                    )
        assert (
            connection.execute("SELECT count(*) FROM console_trace_calls").fetchone()[0]
            == 2
        )
    finally:
        db.close_connection()
