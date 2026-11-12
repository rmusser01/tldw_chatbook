"""ChaChaNotes v73 -> v74: compaction attempts record why they failed.

TASK-33621.3. A failed Console compaction was billed and ledgered as
``status='failed'`` with no reason anywhere, so neither the user nor support
could tell a lineage fault from a provider error. V74 adds a content-free
``failure_reason`` column; these tests pin the column on a fresh database, the
upgrade of a genuine v73 database with a row already in the ledger, and the
repository contract that writes and reads it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Chat.console_context_repository import (
    AuxiliaryAttemptStart,
    AuxiliaryAttemptStatus,
    ConsoleContextRepository,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(console_auxiliary_attempts)")
    }


def _start(repository: ConsoleContextRepository, conversation_id: str, op: str) -> None:
    repository.start_auxiliary_attempt(
        AuxiliaryAttemptStart(
            operation_id=op,
            conversation_id=conversation_id,
            purpose="conversation_compaction",
            provider="openai",
            model="gpt-test",
            requested_output_cap=100,
            estimated_input_tokens=1_000,
            started_at="2026-09-30T00:00:00+00:00",
        )
    )


def test_fresh_database_has_the_failure_reason_column(tmp_path: Path) -> None:
    # A fresh database is built at the CURRENT version, whatever later
    # migrations have bumped it to, and still carries v74's column. Pinning
    # the literal (77, then 80) went red on every bump (TASK-33621.27).
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh")
    try:
        connection = db.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION >= 74
        assert "failure_reason" in _columns(connection)
    finally:
        db.close_connection()


def test_v73_database_upgrades_and_keeps_its_ledger(tmp_path: Path) -> None:
    path = tmp_path / "genuine-v73.sqlite"
    with chachanotes_db_at_version(path, 73, client_id="v73-fixture") as historical:
        connection = historical.get_connection()
        assert _version(connection) == 73
        assert "failure_reason" not in _columns(connection)
        conversation_id = historical.add_conversation({"title": "Before v74"})
        with historical.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO console_auxiliary_attempts(
                    operation_id, conversation_id, purpose, provider, model,
                    requested_output_cap, estimated_input_tokens, status,
                    started_at, finished_at
                ) VALUES ('old-op', ?, 'conversation_compaction', 'openai',
                          'gpt-test', 100, 1000, 'failed',
                          '2026-09-29T00:00:00+00:00', '2026-09-29T00:00:01+00:00')
                """,
                (conversation_id,),
            )

    upgraded = CharactersRAGDB(path, client_id="v74-open")
    try:
        connection = upgraded.get_connection()
        assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert "failure_reason" in _columns(connection)
        repository = ConsoleContextRepository(upgraded)
        old = repository.get_auxiliary_attempt("old-op")
        assert old is not None
        assert old["status"] == "failed"
        assert old["failure_reason"] is None

        _start(repository, conversation_id, "new-op")
        assert repository.finish_auxiliary_attempt(
            "new-op",
            status=AuxiliaryAttemptStatus.FAILED,
            finished_at="2026-09-30T00:00:01+00:00",
            failure_reason="memory_commit_failed",
        )
        listed = repository.list_auxiliary_attempts(conversation_id)
        assert {row["operation_id"]: row["failure_reason"] for row in listed} == {
            "old-op": None,
            "new-op": "memory_commit_failed",
        }
    finally:
        upgraded.close_connection()


def test_repository_refuses_a_reason_on_success_or_free_text(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "contract.db", client_id="contract")
    try:
        conversation_id = db.add_conversation({"title": "Contract"})
        repository = ConsoleContextRepository(db)
        _start(repository, conversation_id, "op-1")
        with pytest.raises(ValueError, match="failure_reason"):
            repository.finish_auxiliary_attempt(
                "op-1",
                status=AuxiliaryAttemptStatus.SUCCEEDED,
                finished_at="2026-09-30T00:00:01+00:00",
                failure_reason="invalid_summary_output",
            )
        with pytest.raises(ValueError, match="failure_reason"):
            repository.finish_auxiliary_attempt(
                "op-1",
                status=AuxiliaryAttemptStatus.FAILED,
                finished_at="2026-09-30T00:00:01+00:00",
                failure_reason="The summary said: secret transcript text",
            )
        # The schema enforces the same shape for any writer that bypasses
        # the repository.
        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE console_auxiliary_attempts SET failure_reason = ? "
                    "WHERE operation_id = 'op-1'",
                    ("Free Text!",),
                )
        assert repository.get_auxiliary_attempt("op-1")["failure_reason"] is None
    finally:
        db.close_connection()


_TERMINAL_FAILURES = ("failed", "cancelled", "stale", "timed_out")


@pytest.mark.parametrize(
    ("status", "reason", "accepted"),
    [
        # Every non-successful terminal status the compaction service records
        # a reason on (cancelled, timed_out and stale included).
        *((status, "memory_commit_failed", True) for status in _TERMINAL_FAILURES),
        ("failed", "a", True),
        ("failed", "a" * 64, True),
        # Rows written before v74, and every success, carry no reason.
        ("succeeded", None, True),
        ("failed", None, True),
        # The repository's reason shape is ``[a-z][a-z_]{0,63}``.
        ("failed", "_", False),
        ("failed", "_bad", False),
        ("failed", "Bad", False),
        ("failed", "", False),
        ("failed", "a" * 65, False),
        # A success, or a still-started attempt, never carries a reason.
        ("succeeded", "invalid_summary_output", False),
        ("started", "invalid_summary_output", False),
    ],
)
def test_schema_enforces_the_repository_reason_contract(
    tmp_path: Path, status: str, reason: str | None, accepted: bool
) -> None:
    """A writer that bypasses the repository meets the same contract."""

    db = CharactersRAGDB(tmp_path / "check.db", client_id="check")
    try:
        conversation_id = db.add_conversation({"title": "Check"})

        def insert() -> None:
            with db.transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO console_auxiliary_attempts(
                        operation_id, conversation_id, purpose, provider,
                        model, requested_output_cap, estimated_input_tokens,
                        status, started_at, failure_reason
                    ) VALUES ('op', ?, 'conversation_compaction', 'openai',
                              'gpt-test', 100, 1000, ?,
                              '2026-09-30T00:00:00+00:00', ?)
                    """,
                    (conversation_id, status, reason),
                )

        if accepted:
            insert()
            row = ConsoleContextRepository(db).get_auxiliary_attempt("op")
            assert row is not None and row["failure_reason"] == reason
        else:
            with pytest.raises(sqlite3.IntegrityError):
                insert()
    finally:
        db.close_connection()
