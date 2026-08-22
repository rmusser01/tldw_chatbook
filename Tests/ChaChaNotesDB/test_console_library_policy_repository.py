from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.Chat.console_library_policy_repository import (
    ConsoleLibraryPolicyRepository,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _candidate(*, allowed: bool) -> ConsoleLibraryPolicyCandidate:
    return ConsoleLibraryPolicyCandidate(
        auto_retrieve=(
            ConsoleAutoRetrieve.AUTOMATIC if allowed else ConsoleAutoRetrieve.NEVER
        ),
        assistant_access=(
            ConsoleAssistantLibraryAccess.ALLOWED
            if allowed
            else ConsoleAssistantLibraryAccess.BLOCKED
        ),
    )


def _add_conversation(db: CharactersRAGDB, title: str = "policy") -> str:
    conversation_id = db.add_conversation({"title": title})
    assert conversation_id is not None
    return conversation_id


def test_read_distinguishes_valid_absent_corrupt_and_error_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = CharactersRAGDB(tmp_path / "policy.sqlite", client_id="policy-test")
    repository = ConsoleLibraryPolicyRepository(db)
    valid_id = _add_conversation(db, "valid")
    absent_id = _add_conversation(db, "absent")
    corrupt_id = _add_conversation(db, "corrupt")

    assert repository.insert(valid_id, _candidate(allowed=True)).status is (
        ConsoleLibraryPolicyWriteStatus.COMMITTED
    )
    assert repository.insert(corrupt_id, _candidate(allowed=False)).status is (
        ConsoleLibraryPolicyWriteStatus.COMMITTED
    )
    connection = db.get_connection()
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(
        "UPDATE console_conversation_library_policy "
        "SET schema_version = ?, policy_revision = ? WHERE conversation_id = ?",
        (2, 0, corrupt_id),
    )
    connection.commit()

    valid = repository.read(valid_id)
    assert valid.snapshot.source == "durable"
    assert valid.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert valid.snapshot.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
    assert valid.snapshot.policy_revision == 1
    assert valid.durable_policy is not None

    absent = repository.read(absent_id)
    assert absent.durable_policy is None
    assert (
        absent.snapshot.auto_retrieve,
        absent.snapshot.assistant_access,
        absent.snapshot.source,
        absent.snapshot.error_code,
    ) == (
        ConsoleAutoRetrieve.NEVER,
        ConsoleAssistantLibraryAccess.BLOCKED,
        "missing",
        None,
    )

    corrupt = repository.read(corrupt_id)
    assert corrupt.durable_policy is None
    assert (
        corrupt.snapshot.auto_retrieve,
        corrupt.snapshot.assistant_access,
        corrupt.snapshot.source,
        corrupt.snapshot.error_code,
    ) == (
        ConsoleAutoRetrieve.NEVER,
        ConsoleAssistantLibraryAccess.BLOCKED,
        "unavailable",
        "corrupt_policy",
    )

    def unavailable() -> object:
        raise RuntimeError("private database detail")

    monkeypatch.setattr(db, "get_connection", unavailable)
    error = repository.read(valid_id)
    assert error.durable_policy is None
    assert (
        error.snapshot.auto_retrieve,
        error.snapshot.assistant_access,
        error.snapshot.source,
        error.snapshot.error_code,
    ) == (
        ConsoleAutoRetrieve.NEVER,
        ConsoleAssistantLibraryAccess.BLOCKED,
        "unavailable",
        "policy_read_error",
    )


def test_conditional_insert_reports_race_winner_without_candidate_publication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "race.sqlite"
    first_db = CharactersRAGDB(path, client_id="first-policy-writer")
    conversation_id = _add_conversation(first_db)
    second_db = CharactersRAGDB(path, client_id="second-policy-writer")
    first = ConsoleLibraryPolicyRepository(first_db)
    second = ConsoleLibraryPolicyRepository(second_db)

    winner = first.insert(conversation_id, _candidate(allowed=False))
    loser = second.insert(conversation_id, _candidate(allowed=True))

    assert winner.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert loser.status is ConsoleLibraryPolicyWriteStatus.CONFLICT
    assert loser.snapshot.source == "durable"
    assert loser.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert loser.snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
    assert loser.snapshot.policy_revision == 1


def test_compare_and_swap_commits_one_revision_and_reports_stale_conflict(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "cas.sqlite", client_id="policy-cas")
    conversation_id = _add_conversation(db)
    repository = ConsoleLibraryPolicyRepository(db)
    repository.insert(conversation_id, _candidate(allowed=False))

    committed = repository.compare_and_swap(
        conversation_id, 1, _candidate(allowed=True)
    )
    stale = repository.compare_and_swap(
        conversation_id, 1, _candidate(allowed=False)
    )

    assert committed.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert committed.snapshot.policy_revision == 2
    assert committed.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert stale.status is ConsoleLibraryPolicyWriteStatus.CONFLICT
    assert stale.snapshot == committed.snapshot


def test_insert_integrity_failure_without_a_race_winner_is_unavailable(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "insert-error.sqlite", client_id="policy")
    conversation_id = _add_conversation(db)
    db.get_connection().execute(
        """
        CREATE TRIGGER fail_policy_insert
        BEFORE INSERT ON console_conversation_library_policy
        BEGIN SELECT RAISE(ABORT, 'injected policy failure'); END
        """
    )
    db.get_connection().commit()

    result = ConsoleLibraryPolicyRepository(db).insert(
        conversation_id,
        _candidate(allowed=True),
    )

    assert result.status is ConsoleLibraryPolicyWriteStatus.UNAVAILABLE
    assert result.snapshot.source == "unavailable"


@pytest.mark.parametrize("operation", ["insert", "compare_and_swap"])
def test_writes_reject_missing_or_soft_deleted_conversations(
    tmp_path: Path, operation: str
) -> None:
    db = CharactersRAGDB(tmp_path / f"missing-{operation}.sqlite", client_id="policy")
    repository = ConsoleLibraryPolicyRepository(db)
    deleted_id = _add_conversation(db)
    assert db.soft_delete_conversation(deleted_id, expected_version=1) is True

    if operation == "insert":
        missing = repository.insert("does-not-exist", _candidate(allowed=True))
        deleted = repository.insert(deleted_id, _candidate(allowed=True))
    else:
        missing = repository.compare_and_swap(
            "does-not-exist", 1, _candidate(allowed=True)
        )
        deleted = repository.compare_and_swap(
            deleted_id, 1, _candidate(allowed=True)
        )

    assert missing.status is ConsoleLibraryPolicyWriteStatus.MISSING_CONVERSATION
    assert deleted.status is ConsoleLibraryPolicyWriteStatus.MISSING_CONVERSATION
    assert missing.snapshot.source == "missing"
    assert deleted.snapshot.source == "missing"
    row_count = db.get_connection().execute(
        "SELECT COUNT(*) FROM console_conversation_library_policy"
    ).fetchone()[0]
    assert row_count == 0


def test_soft_delete_retains_policy_restore_reuses_it_and_hard_purge_cascades(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "lifecycle.sqlite", client_id="policy-lifecycle")
    conversation_id = _add_conversation(db)
    repository = ConsoleLibraryPolicyRepository(db)
    committed = repository.insert(conversation_id, _candidate(allowed=True))
    assert committed.status is ConsoleLibraryPolicyWriteStatus.COMMITTED

    assert db.soft_delete_conversation(conversation_id, expected_version=1) is True
    retained = db.get_connection().execute(
        "SELECT policy_revision FROM console_conversation_library_policy "
        "WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    assert tuple(retained) == (1,)

    assert db.restore_conversation(conversation_id, expected_version=2) is True
    restored = repository.read(conversation_id)
    assert restored.snapshot == committed.snapshot

    with db.transaction() as cursor:
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    assert repository.read(conversation_id).snapshot.source == "missing"
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM console_conversation_library_policy "
        "WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()[0] == 0
