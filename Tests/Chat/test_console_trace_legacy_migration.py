"""Bounded, resumable legacy exchange normalization."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
    capture_to_blob,
)
from tldw_chatbook.Chat.console_trace_legacy import LegacyTraceNormalizer
from tldw_chatbook.Chat.console_trace_maintenance import LegacyTraceMaintenance
from tldw_chatbook.Chat.console_trace_projection import ConsoleTraceProjection
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db() -> CharactersRAGDB:
    database = CharactersRAGDB(":memory:", "legacy-trace-maintenance-test")
    yield database
    database.close_connection()


def _message(db: CharactersRAGDB, conversation_id: str, content: str) -> str:
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "role": "assistant",
            "content": content,
        }
    )
    assert message_id is not None
    return message_id


def _capture(seq: int) -> ExchangeCapture:
    return ExchangeCapture(
        run_tag="legacy-run",
        seq=seq,
        created_at=f"2026-08-28T12:00:0{seq}+00:00",
        provider="openai",
        model="gpt-test",
        endpoint=None,
        request={"messages_payload": [{"role": "user", "content": f"row-{seq}"}]},
        response={"content": f"answer-{seq}"},
        status="complete",
        usage_json=None,
        omitted_keys=(),
        capture_detail=CaptureDetail.FULL,
    )


def _insert_exchange(
    db: CharactersRAGDB,
    *,
    message_id: str,
    capture: ExchangeCapture,
    blob: bytes | None = None,
) -> None:
    db.append_message_exchanges_local(
        message_id,
        [
            {
                "run_tag": capture.run_tag,
                "seq": capture.seq,
                "status": capture.status,
                "abandoned": False,
                "capture_detail": capture.capture_detail.value,
                "capture_blob": capture_to_blob(capture) if blob is None else blob,
                "created_at": capture.created_at,
            }
        ],
    )


def test_batches_resume_delete_only_verified_rows_and_finish_idempotently(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy migration"})
    assert conversation_id is not None
    message_ids: list[str] = []
    for seq in range(3):
        message_id = _message(db, conversation_id, f"answer-{seq}")
        message_ids.append(message_id)
        _insert_exchange(db, message_id=message_id, capture=_capture(seq))
    maintenance = LegacyTraceMaintenance(db, max_rows=1)

    first = maintenance.run_batch()
    second = maintenance.run_batch()
    third = maintenance.run_batch()
    finished = maintenance.run_batch()

    assert [first.processed_rows, second.processed_rows, third.processed_rows] == [
        1,
        1,
        1,
    ]
    assert third.logical_complete is True
    assert finished.processed_rows == 0
    assert finished.logical_complete is True
    with db.transaction() as cursor:
        assert (
            cursor.execute("SELECT COUNT(*) FROM message_exchanges").fetchone()[0] == 0
        )
        state = cursor.execute(
            """SELECT status, processed_rows, last_exchange_id
                 FROM console_trace_migration_state
                WHERE migration_name = 'legacy_exchange_normalization'"""
        ).fetchone()
        assert tuple(state[:2]) == ("logical_complete", 3)
        assert state[2] is not None
    normalizer = LegacyTraceNormalizer(db)
    assert all(normalizer.read_calls(message_id) for message_id in message_ids)


def test_corrupt_capture_becomes_explicit_snapshot_omission(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy corruption"})
    assert conversation_id is not None
    message_id = _message(db, conversation_id, "unavailable")
    capture = _capture(0)
    _insert_exchange(db, message_id=message_id, capture=capture, blob=b"not-zlib")

    result = LegacyTraceMaintenance(db).run_batch()

    assert result.processed_rows == 1
    calls = LegacyTraceNormalizer(db).read_calls(message_id)
    assert calls[0].capture.request == {"legacy_omission": "legacy_capture_unavailable"}
    assert calls[0].capture.omitted_keys == ("legacy_capture_unavailable",)
    assert "legacy_capture_unavailable" in calls[0].uncertainty_codes


def test_active_provider_run_defers_without_claiming_or_advancing(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy busy"})
    assert conversation_id is not None
    message_id = _message(db, conversation_id, "answer")
    _insert_exchange(db, message_id=message_id, capture=_capture(0))
    maintenance = LegacyTraceMaintenance(db, provider_active=lambda: True)

    result = maintenance.run_batch()

    assert result.admitted is False
    assert result.processed_rows == 0
    assert db.get_message_exchanges(message_id)


def test_other_maintenance_lease_defers_batch(db: CharactersRAGDB) -> None:
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            """UPDATE console_trace_maintenance_state
                  SET state = 'marking', lease_id = 'lease', lease_owner = 'test',
                      lease_expires_at = '2099-01-01T00:00:00+00:00',
                      marked_epoch = 0
                WHERE singleton_id = 1"""
        )

    result = LegacyTraceMaintenance(db).run_batch()

    assert result.admitted is False
    assert result.processed_rows == 0


def test_unexpected_failure_rolls_back_normalized_rows_and_checkpoint(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy rollback"})
    assert conversation_id is not None
    message_id = _message(db, conversation_id, "answer")
    _insert_exchange(db, message_id=message_id, capture=_capture(0))
    real = LegacyTraceNormalizer(db)

    class _FailAfterNormalize:
        def normalize_exchange(self, cursor, row):
            real.normalize_exchange(cursor, row)
            raise RuntimeError("injected_failure")

    maintenance = LegacyTraceMaintenance(db, normalizer=_FailAfterNormalize())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="injected_failure"):
        maintenance.run_batch()

    with db.transaction() as cursor:
        assert (
            cursor.execute("SELECT COUNT(*) FROM message_exchanges").fetchone()[0] == 1
        )
        assert (
            cursor.execute("SELECT COUNT(*) FROM console_trace_calls").fetchone()[0]
            == 0
        )
        state = cursor.execute(
            """SELECT status, last_exchange_id, processed_rows
                 FROM console_trace_migration_state
                WHERE migration_name = 'legacy_exchange_normalization'"""
        ).fetchone()
        assert tuple(state) == ("pending", None, 0)


def test_elapsed_time_limit_yields_after_current_verified_row(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy time bound"})
    assert conversation_id is not None
    for seq in range(2):
        message_id = _message(db, conversation_id, f"answer-{seq}")
        _insert_exchange(db, message_id=message_id, capture=_capture(seq))
    ticks = iter((0.0, 0.101))
    maintenance = LegacyTraceMaintenance(db, clock=lambda: next(ticks))

    result = maintenance.run_batch()

    assert result.processed_rows == 1
    assert result.logical_complete is False


def test_partial_migration_dual_read_keeps_normalized_and_legacy_calls_visible(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy dual read"})
    assert conversation_id is not None
    message_id = _message(db, conversation_id, "answer")
    _insert_exchange(db, message_id=message_id, capture=_capture(0))
    _insert_exchange(db, message_id=message_id, capture=_capture(1))
    normalizer = LegacyTraceNormalizer(db)
    maintenance = LegacyTraceMaintenance(
        db,
        normalizer=normalizer,
        max_rows=1,
    )

    first = maintenance.run_batch()
    projection = ConsoleTraceProjection(
        legacy_reader=db.get_message_exchanges,
        normalized_reader=normalizer.read_calls,
        normalized_reads_enabled=True,
    )

    assert first.processed_rows == 1
    assert first.logical_complete is False
    calls = projection.read_calls(message_id)
    assert [(call.capture.seq, call.source) for call in calls] == [
        (0, "normalized"),
        (1, "legacy"),
    ]


def test_new_legacy_row_after_logical_completion_reopens_checkpoint(
    db: CharactersRAGDB,
) -> None:
    conversation_id = db.add_conversation({"title": "legacy writer overlap"})
    assert conversation_id is not None
    first_message_id = _message(db, conversation_id, "answer-0")
    _insert_exchange(db, message_id=first_message_id, capture=_capture(0))
    maintenance = LegacyTraceMaintenance(db)

    assert maintenance.run_batch().logical_complete is True
    second_message_id = _message(db, conversation_id, "answer-1")
    _insert_exchange(db, message_id=second_message_id, capture=_capture(1))

    reopened = maintenance.run_batch()

    assert reopened.processed_rows == 1
    assert reopened.logical_complete is True
    assert LegacyTraceNormalizer(db).read_calls(second_message_id)
