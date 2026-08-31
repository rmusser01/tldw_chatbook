"""v43 message_exchanges: local-only, idempotent upsert, cascade delete
(Console Conversation Inspector, task-5).

Local-only means: no sync_log rows are ever written for this table (same
precedent as the v29->v30 usage_json / v39->v40 transcript_annotations
local-only additions), and a hard delete of the parent message cascades
straight through via the FK -- there is no soft-delete/version bookkeeping
for these rows.
"""

import json
import sqlite3
import traceback
from dataclasses import asdict
from contextlib import contextmanager

import pytest
from loguru import logger

from Tests.ChaChaNotesDB.historical_bootstrap import (
    chachanotes_db_at_version,
    open_current_chachanotes_from_legacy,
)

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
    capture_from_storage,
    capture_to_blob,
)
from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_SANITIZER_UNAVAILABLE,
    CredentialSanitizationResult,
)

# Matches CharactersRAGDB._SCHEMA_NAME, per the sibling migration tests
# (e.g. Tests/DB/test_chachanotes_message_usage_migration.py).
SCHEMA_NAME = "rag_char_chat_schema"
CAPTURE_DETAIL_INDEX = "idx_message_exchanges_capture_detail"


@pytest.fixture
def db():
    database = CharactersRAGDB(":memory:", client_id="message-exchanges-test")
    yield database
    database.close_connection()


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _seed_message(db) -> str:
    """Create a conversation + message via the DB's real public API and
    return the message id, mirroring the seeding helper used by the v30
    usage_json round-trip test."""
    conv_id = db.add_conversation({"title": "t"})
    msg_id = db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "hi"}
    )
    return msg_id


def test_append_and_read_round_trip(db):
    mid = _seed_message(db)
    rows = [
        {
            "run_tag": "r1",
            "seq": 0,
            "status": "complete",
            "abandoned": False,
            "capture_blob": b"blob0",
            "created_at": "2026-08-18T00:00:00Z",
        },
        {
            "run_tag": "r1",
            "seq": 1,
            "status": "stopped",
            "abandoned": False,
            "capture_blob": b"blob1",
            "created_at": "2026-08-18T00:00:01Z",
        },
    ]
    assert db.append_message_exchanges_local(mid, rows) == 2
    stored = db.get_message_exchanges(mid)
    assert [(r["run_tag"], r["seq"], r["capture_blob"]) for r in stored] == [
        ("r1", 0, b"blob0"),
        ("r1", 1, b"blob1"),
    ]


def test_lowest_exchange_write_error_boundary_is_content_free(db, monkeypatch):
    message_id = _seed_message(db)
    canaries = (
        "SEMANTIC-EXCHANGE-ERROR-CANARY",
        "/private/exchange/error/path/canary",
        "QUJD" * 1200,
    )

    class FailingCursor:
        def execute(self, _query, _params):
            raise sqlite3.OperationalError(" | ".join(canaries))

    @contextmanager
    def failing_transaction(*_args, **_kwargs):
        yield FailingCursor()

    monkeypatch.setattr(db, "transaction", failing_transaction)
    diagnostics: list[str] = []
    sink_id = logger.add(
        diagnostics.append,
        level="ERROR",
        format="{extra[message_id]} {extra[error_type]} {message}",
    )
    try:
        with pytest.raises(Exception) as raised:
            db.append_message_exchanges_local(
                message_id,
                [
                    {
                        "run_tag": "canary",
                        "seq": 0,
                        "status": "complete",
                        "abandoned": False,
                        "capture_detail": "full",
                        "capture_blob": canaries[-1].encode(),
                        "created_at": "t",
                    }
                ],
            )
    finally:
        logger.remove(sink_id)

    assert type(raised.value).__name__ == "CharactersRAGDBError"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    exception_graph: list[BaseException] = []
    pending: list[BaseException] = [raised.value]
    while pending:
        error = pending.pop()
        exception_graph.append(error)
        if error.__cause__ is not None:
            pending.append(error.__cause__)
        if error.__context__ is not None:
            pending.append(error.__context__)
    rendered_traceback = "".join(
        traceback.format_exception(raised.type, raised.value, raised.tb)
    )
    boundary_text = " ".join(
        [
            *(f"{error!s} {error!r}" for error in exception_graph),
            *diagnostics,
            rendered_traceback,
        ]
    )
    assert message_id in boundary_text
    assert "OperationalError" in boundary_text
    assert "message_exchange_write_failed" in boundary_text
    for canary in canaries:
        assert canary not in boundary_text


def test_full_capture_column_matches_blob_provenance(db):
    mid = _seed_message(db)
    capture = ExchangeCapture(
        run_tag="full",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request={},
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=(),
        capture_detail=CaptureDetail.FULL,
    )
    db.append_message_exchanges_local(
        mid,
        [
            {
                "run_tag": capture.run_tag,
                "seq": capture.seq,
                "status": capture.status,
                "abandoned": False,
                "capture_detail": capture.capture_detail.value,
                "capture_blob": capture_to_blob(capture),
                "created_at": capture.created_at,
            }
        ],
    )
    stored = db.get_message_exchanges(mid)[0]
    assert stored["capture_detail"] == "full"
    assert (
        capture_from_storage(
            stored["capture_blob"], stored["capture_detail"]
        ).capture_detail
        is CaptureDetail.FULL
    )


@pytest.mark.parametrize("detail", [CaptureDetail.SAFE, CaptureDetail.FULL])
def test_first_durable_exchange_write_contains_only_sanitized_capture(db, detail):
    message_id = _seed_message(db)
    credential = "sk-live-first-write-canary"
    capture = ExchangeCapture(
        run_tag="canary",
        seq=0,
        created_at="2026-08-29T00:00:00Z",
        provider="openai",
        model="ordinary-model",
        endpoint=f"https://user:{credential}@example.test/v1?token={credential}",
        request={
            "messages_payload": [
                {
                    "role": "user",
                    "content": f"ordinary request then bearer {credential}",
                }
            ]
        },
        response={
            "content": "ordinary response",
            "tool_calls": [{"function": {"arguments": f"token={credential}"}}],
        },
        status="complete",
        usage_json=None,
        omitted_keys=(),
        capture_detail=detail,
    )

    db.append_message_exchanges_local(
        message_id,
        [
            {
                "run_tag": capture.run_tag,
                "seq": capture.seq,
                "status": capture.status,
                "abandoned": False,
                "capture_detail": capture.capture_detail.value,
                "capture_blob": capture_to_blob(capture),
                "created_at": capture.created_at,
            }
        ],
    )

    row = db.get_message_exchanges(message_id)[0]
    restored = capture_from_storage(row["capture_blob"], row["capture_detail"])
    persisted = json.dumps(asdict(restored), default=str)
    assert credential not in persisted
    assert "ordinary request" in persisted
    assert "ordinary response" in persisted
    assert "capture.credential_redacted" in restored.omitted_keys


def test_capture_serializer_fails_closed_when_credential_filter_is_unavailable(
    db, monkeypatch
):
    message_id = _seed_message(db)
    credential = "sk-live-sanitizer-failure-canary"
    capture = ExchangeCapture(
        run_tag="canary",
        seq=0,
        created_at="2026-08-29T00:00:00Z",
        provider=credential,
        model=credential,
        endpoint=f"https://example.test/?token={credential}",
        request={"messages_payload": [{"content": credential}]},
        response={"content": credential},
        status="complete",
        usage_json=None,
        omitted_keys=(),
        capture_detail=CaptureDetail.SAFE,
    )

    def fail_closed(_self, _value):
        return CredentialSanitizationResult(
            available=False,
            value=None,
            omission_reason_code=CREDENTIAL_SANITIZER_UNAVAILABLE,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_exchange_capture.CredentialSanitizer.sanitize",
        fail_closed,
    )
    blob = capture_to_blob(capture)
    db.append_message_exchanges_local(
        message_id,
        [
            {
                "run_tag": capture.run_tag,
                "seq": capture.seq,
                "status": capture.status,
                "abandoned": False,
                "capture_detail": capture.capture_detail.value,
                "capture_blob": blob,
                "created_at": capture.created_at,
            }
        ],
    )

    row = db.get_message_exchanges(message_id)[0]
    restored = capture_from_storage(row["capture_blob"], row["capture_detail"])
    assert credential not in json.dumps(asdict(restored), default=str)
    assert restored.request == {"omitted": True}
    assert restored.response == {"omitted": True}
    assert "capture" in restored.omitted_keys


def test_upsert_idempotent_and_updates_in_place(db):
    mid = _seed_message(db)
    row = {
        "run_tag": "r1",
        "seq": 0,
        "status": "complete",
        "abandoned": False,
        "capture_blob": b"v1",
        "created_at": "t",
    }
    db.append_message_exchanges_local(mid, [row])
    db.append_message_exchanges_local(
        mid, [{**row, "capture_blob": b"v2", "abandoned": True}]
    )
    stored = db.get_message_exchanges(mid)
    assert len(stored) == 1
    assert stored[0]["capture_blob"] == b"v2" and stored[0]["abandoned"]


def test_no_sync_log_rows_written(db):
    mid = _seed_message(db)
    with db.transaction() as cursor:
        before = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
    # Self-validating: seeding a conversation + message fires the
    # conversations/messages sync_log triggers, so this must be nonzero --
    # otherwise an `after == before` comparison could pass vacuously (e.g.
    # if sync_log were broken/empty for an unrelated reason).
    assert before > 0
    db.append_message_exchanges_local(
        mid,
        [
            {
                "run_tag": "r1",
                "seq": 0,
                "status": "complete",
                "abandoned": False,
                "capture_blob": b"b",
                "created_at": "t",
            }
        ],
    )
    with db.transaction() as cursor:
        after = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
    assert after == before


def test_hard_delete_cascades(db):
    mid = _seed_message(db)
    db.append_message_exchanges_local(
        mid,
        [
            {
                "run_tag": "r1",
                "seq": 0,
                "status": "complete",
                "abandoned": False,
                "capture_blob": b"b",
                "created_at": "t",
            }
        ],
    )
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM messages WHERE id = ?", (mid,))
        count = cursor.execute("SELECT COUNT(*) FROM message_exchanges").fetchone()[0]
    assert count == 0


def test_full_capture_queries_use_capture_detail_index_without_stats(db):
    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'"
        ).fetchone()
        is None
    ), (
        "the plan must match production's no-ANALYZE state, not a test-only "
        "sqlite_stat1-assisted plan"
    )

    conversation_id = db.add_conversation({"title": "target"})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "captured",
        }
    )
    db.append_message_exchanges_local(
        message_id,
        [
            {
                "run_tag": "full",
                "seq": 0,
                "status": "complete",
                "abandoned": False,
                "capture_detail": "full",
                "capture_blob": b"full",
                "created_at": "t",
            }
        ],
    )

    queries = (
        """
        SELECT exchange.message_id, exchange.run_tag, exchange.seq
          FROM message_exchanges AS exchange
          JOIN messages AS message ON message.id = exchange.message_id
         WHERE message.conversation_id = ?
           AND exchange.capture_detail = 'full'
        """,
        """
        DELETE FROM message_exchanges
         WHERE capture_detail = 'full'
           AND message_id IN (
               SELECT id FROM messages WHERE conversation_id = ?
           )
        """,
    )
    for query in queries:
        plan = " | ".join(
            str(row[-1])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN " + query, (conversation_id,)
            )
        )
        assert plan, "an empty query plan would make the index assertion vacuous"
        assert CAPTURE_DETAIL_INDEX in plan


def test_full_exchange_purge_scopes_by_conversation_including_deleted_messages(db):
    conversation_id = db.add_conversation({"title": "target"})
    other_conversation_id = db.add_conversation({"title": "other"})
    message_ids = [
        db.add_message(
            {"conversation_id": conversation_id, "sender": "assistant", "content": name}
        )
        for name in ("active", "off-path", "abandoned", "soft-deleted")
    ]
    other_message_id = db.add_message(
        {
            "conversation_id": other_conversation_id,
            "sender": "assistant",
            "content": "other",
        }
    )
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET deleted = 1, usage_json = ? WHERE id = ?",
            ('{"total_tokens":7}', message_ids[-1]),
        )

    for index, message_id in enumerate((*message_ids, other_message_id)):
        db.append_message_exchanges_local(
            message_id,
            [
                {
                    "run_tag": f"safe-{index}",
                    "seq": 0,
                    "status": "complete",
                    "abandoned": index == 2,
                    "capture_detail": "safe",
                    "capture_blob": f"safe-{index}".encode(),
                    "created_at": "t",
                },
                {
                    "run_tag": f"full-{index}",
                    "seq": 0,
                    "status": "complete",
                    "abandoned": index == 2,
                    "capture_detail": "full",
                    "capture_blob": f"full-{index}".encode(),
                    "created_at": "t",
                },
            ],
        )

    assert db.list_full_exchange_keys_for_conversation(conversation_id) == {
        (message_id, f"full-{index}", 0) for index, message_id in enumerate(message_ids)
    }
    assert db.delete_full_exchanges_for_conversation(conversation_id) == 4

    for index, message_id in enumerate(message_ids):
        assert [row["run_tag"] for row in db.get_message_exchanges(message_id)] == [
            f"safe-{index}"
        ]
    assert len(db.get_message_exchanges(other_message_id)) == 2
    with db.transaction() as cursor:
        row = cursor.execute(
            "SELECT deleted, usage_json FROM messages WHERE id = ?", (message_ids[-1],)
        ).fetchone()
    assert tuple(row) == (1, '{"total_tokens":7}')


def test_full_exchange_delete_rolls_back_atomically(db):
    message_id = _seed_message(db)
    rows = [
        {
            "run_tag": run_tag,
            "seq": 0,
            "status": "complete",
            "abandoned": False,
            "capture_detail": "full",
            "capture_blob": blob,
            "created_at": "t",
        }
        for run_tag, blob in (("first", b"one"), ("second", b"two"))
    ]
    db.append_message_exchanges_local(message_id, rows)
    conversation_id = db.get_message_by_id(message_id)["conversation_id"]
    with db.transaction() as cursor:
        cursor.execute(
            """
            CREATE TEMP TRIGGER fail_second_full_delete
            BEFORE DELETE ON message_exchanges
            WHEN OLD.run_tag = 'second'
            BEGIN
                SELECT RAISE(ABORT, 'injected delete failure');
            END
            """
        )

    with pytest.raises(Exception, match="injected delete failure"):
        db.delete_full_exchanges_for_conversation(conversation_id)

    assert [
        (row["run_tag"], row["capture_blob"])
        for row in db.get_message_exchanges(message_id)
    ] == [("first", b"one"), ("second", b"two")]


def test_full_exchange_delete_rolls_back_when_staged_inventory_changed(db):
    message_id = _seed_message(db)
    db.append_message_exchanges_local(
        message_id,
        [
            {
                "run_tag": "full",
                "seq": 0,
                "status": "complete",
                "abandoned": False,
                "capture_detail": "full",
                "capture_blob": b"full",
                "created_at": "t",
            }
        ],
    )
    conversation_id = db.get_message_by_id(message_id)["conversation_id"]

    with pytest.raises(Exception, match="inventory changed"):
        db.delete_full_exchanges_for_conversation(conversation_id, expected_count=2)

    assert [row["run_tag"] for row in db.get_message_exchanges(message_id)] == ["full"]


def test_schema_version_is_at_least_43(db):
    # Mirrors the house sibling-version test pattern (a local `_version()`
    # helper against db_schema_version -- there is no public accessor).
    #
    # task-19554: this used to be `== 43` and was designated the repo's one
    # exact current-version pin. That made every LATER migration edit this
    # file, which owns only v42->v43. It now asserts at-or-past its own
    # version, and the exact pin lives with the newest migration --
    # `Tests/DB/test_chachanotes_console_library_policy_migration.py`'s
    # `test_real_v47_fixture_gains_exact_v48_local_schema_and_seed_rows`.
    assert _version(db.get_connection()) >= 43


def test_migrate_from_v42_to_v43_requires_version_42(tmp_path):
    # Mirrors the version pre-check idiom in
    # test_chachanotes_default_assistant_enrichment_migration.py::
    # test_migrate_from_v31_to_v32_requires_version_31: a fresh database
    # lands on the CURRENT schema (>= 43), so calling the v42->v43 step
    # directly against it must reject rather than silently re-run.
    from tldw_chatbook.DB.ChaChaNotes_DB import SchemaError

    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="version-test")
    conn = db.get_connection()
    with pytest.raises(SchemaError):
        db._migrate_from_v42_to_v43(conn)
    db.close_connection()


def test_upgrade_path_from_v42_recreates_the_table(tmp_path):
    """A genuine v42 database must, on reopen, run
    _migrate_from_v42_to_v43 and land on the current version with the
    table back. (task-19554: the landing version is the CURRENT one, not
    43 -- a stamped-back DB replays every later step too.)"""
    db_path = tmp_path / "chachanotes.db"
    with chachanotes_db_at_version(db_path, 42, client_id="upgrade-test") as db:
        assert _version(db.get_connection()) == 42
        tables = {
            row[0]
            for row in db.get_connection().execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "message_exchanges" not in tables

    reopened = open_current_chachanotes_from_legacy(
        db_path, client_id="upgrade-test-reopen"
    )
    reopened_connection = reopened.get_connection()
    assert _version(reopened_connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    tables = {
        row[0]
        for row in reopened_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "message_exchanges" in tables
    # And the recreated table is genuinely usable, not left half-migrated.
    mid = _seed_message(reopened)
    assert (
        reopened.append_message_exchanges_local(
            mid,
            [
                {
                    "run_tag": "r1",
                    "seq": 0,
                    "status": "complete",
                    "abandoned": False,
                    "capture_blob": b"b",
                    "created_at": "t",
                }
            ],
        )
        == 1
    )
    reopened.close_connection()
