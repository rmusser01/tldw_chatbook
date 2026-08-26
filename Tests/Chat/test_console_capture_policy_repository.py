"""Local per-conversation capture-detail repository contract."""
from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
from tldw_chatbook.Chat import console_capture_policy_repository as repository_module
from tldw_chatbook.Chat.console_capture_policy_repository import (
    CapturePolicyWriteStatus,
    ConsoleCapturePolicyRepository,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db():
    database = CharactersRAGDB(":memory:", client_id="capture-policy-test")
    yield database
    database.close_connection()


def _conversation(db: CharactersRAGDB) -> str:
    return db.add_conversation({"title": "capture policy"})


def test_missing_row_means_inherit_and_replace_round_trips(db) -> None:
    conversation_id = _conversation(db)
    repository = ConsoleCapturePolicyRepository(db)
    assert repository.read(conversation_id).status is repository_module.CapturePolicyReadStatus.ABSENT
    safe = repository.replace(conversation_id, CaptureDetail.SAFE)
    assert safe.status is CapturePolicyWriteStatus.STORED
    assert safe.policy is not None and safe.policy.detail is CaptureDetail.SAFE
    full = repository.replace(conversation_id, CaptureDetail.FULL)
    assert full.status is CapturePolicyWriteStatus.STORED
    found = repository.read(conversation_id)
    assert found.status is repository_module.CapturePolicyReadStatus.FOUND
    assert found.policy is not None and found.policy.detail is CaptureDetail.FULL


def test_inherit_deletes_the_local_row(db) -> None:
    conversation_id = _conversation(db)
    repository = ConsoleCapturePolicyRepository(db)
    repository.replace(conversation_id, CaptureDetail.SAFE)
    assert repository.replace(conversation_id, None).status is CapturePolicyWriteStatus.DELETED
    assert repository.read(conversation_id).status is repository_module.CapturePolicyReadStatus.ABSENT
    assert repository.replace(conversation_id, None).status is CapturePolicyWriteStatus.UNCHANGED


def test_missing_or_deleted_conversation_refuses_writes(db) -> None:
    repository = ConsoleCapturePolicyRepository(db)
    assert repository.replace("missing", CaptureDetail.SAFE).status is CapturePolicyWriteStatus.MISSING_CONVERSATION
    conversation_id = _conversation(db)
    with db.transaction() as cursor:
        cursor.execute("UPDATE conversations SET deleted = 1 WHERE id = ?", (conversation_id,))
    assert repository.replace(conversation_id, CaptureDetail.SAFE).status is CapturePolicyWriteStatus.MISSING_CONVERSATION


def test_cascade_delete_corrupt_value_and_no_sync_delta_fail_closed(db) -> None:
    conversation_id = _conversation(db)
    repository = ConsoleCapturePolicyRepository(db)
    with db.transaction() as cursor:
        before = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
    repository.replace(conversation_id, CaptureDetail.FULL)
    with db.transaction() as cursor:
        after = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
        assert after == before
        cursor.execute("PRAGMA ignore_check_constraints = ON")
        cursor.execute(
            "UPDATE console_conversation_capture_policy SET capture_detail = 'bad' WHERE conversation_id = ?",
            (conversation_id,),
        )
        cursor.execute("PRAGMA ignore_check_constraints = OFF")
    corrupt = repository.read(conversation_id)
    assert corrupt.status is repository_module.CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT
    assert corrupt.policy is None
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        assert cursor.execute(
            "SELECT COUNT(*) FROM console_conversation_capture_policy"
        ).fetchone()[0] == 0


def test_schema_unavailable_is_not_reported_as_absent(db) -> None:
    conversation_id = _conversation(db)
    repository = ConsoleCapturePolicyRepository(db)
    with db.transaction() as cursor:
        cursor.execute("DROP TABLE console_conversation_capture_policy")

    result = repository.read(conversation_id)

    assert result.status is repository_module.CapturePolicyReadStatus.UNAVAILABLE_OR_CORRUPT
    assert result.policy is None
