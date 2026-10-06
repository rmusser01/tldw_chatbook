"""Finite unread reads retire new handles, never a caller's database owner."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(params=["unread_token", "unread_ids_for"])
def reader(request):
    """Exercise both shared readers with fixed, independently checked inputs."""
    return request.param


@pytest.fixture
def database(tmp_path):
    """Retire the actual fixture owner only after each executor has joined."""
    db = CharactersRAGDB(tmp_path / "marks.sqlite", client_id="read-ownership")
    try:
        yield db
    finally:
        with db.quiesce_connections(timeout_seconds=5):
            pass
        assert db.registered_connection_count() == 0


def _read(service, reader):
    argument = "unread" if reader == "unread_token" else ("unread", "read", "unread")
    return getattr(service, reader)(argument)


def test_worker_unread_read_retires_its_new_handle(database, reader):
    """Removing operation ownership leaks a real handle after worker exit."""
    service = ConversationLocalMarksService(database)
    token = service.mark_unread("unread")
    expected = token if reader == "unread_token" else frozenset({"unread"})
    assert database.registered_connection_count() == 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Repeated cold callbacks must not accumulate or borrow their own leak.
        for _ in range(3):
            assert executor.submit(_read, service, reader).result(timeout=5) == expected
            assert database.registered_connection_count() == 1
    assert database.registered_connection_count() == 1


def test_failed_worker_unread_read_retires_its_new_handle(database, reader):
    """A real SQL failure must propagate without stranding its acquired handle."""
    service = ConversationLocalMarksService(database)
    with database.transaction() as cursor:
        cursor.execute("DROP TABLE conversation_local_marks")

    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(sqlite3.OperationalError, match="no such table"),
    ):
        executor.submit(_read, service, reader).result(timeout=5)
    assert database.registered_connection_count() == 1


@pytest.mark.parametrize("fail_sql", [False, True])
def test_unread_read_preserves_borrowed_transaction(database, reader, fail_sql):
    """Overeager cleanup must not close, commit or roll back a native borrower."""
    service = ConversationLocalMarksService(database)
    token = service.mark_unread("unread")
    expected = token if reader == "unread_token" else frozenset({"unread"})

    def borrow():
        connection = database.get_connection()
        try:
            connection.execute("BEGIN")
            connection.execute(
                "UPDATE conversation_local_marks SET updated_at = 'borrowed'"
            )
            if fail_sql:
                connection.execute("DROP TABLE conversation_local_marks")
                with pytest.raises(sqlite3.OperationalError, match="no such table"):
                    _read(service, reader)
            else:
                assert _read(service, reader) == expected
                assert (
                    connection.execute(
                        "SELECT updated_at FROM conversation_local_marks"
                    ).fetchone()[0]
                    == "borrowed"
                )
            assert connection.in_transaction
            assert database.get_connection() is connection
            assert connection.execute("SELECT 1").fetchone()[0] == 1
        finally:
            connection.rollback()
            database.close_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(borrow).result(timeout=5)
    assert database.registered_connection_count() == 1
    assert (
        database.get_connection()
        .execute("SELECT updated_at FROM conversation_local_marks")
        .fetchone()[0]
        != "borrowed"
    )


class _CustomDatabase(CharactersRAGDB):
    """A real installed subclass whose connection lifetime is caller-owned."""


@pytest.mark.parametrize("owner", ["memory", "custom"])
def test_unread_read_preserves_memory_and_custom_owners(tmp_path, reader, owner):
    """Broadening the exact-file guard must not retire excluded owners."""
    db = (
        CharactersRAGDB(":memory:", client_id="memory-read-owner")
        if owner == "memory"
        else _CustomDatabase(tmp_path / "custom.sqlite", client_id="custom-read-owner")
    )
    try:
        service = ConversationLocalMarksService(db)
        token = service.mark_unread("unread")
        expected = token if reader == "unread_token" else frozenset({"unread"})

        def read_owned():
            if owner == "memory":
                # A new thread has its own schema-empty memory database. Even
                # this failed read must leave that owner alive, not erase it.
                with pytest.raises(sqlite3.OperationalError, match="no such table"):
                    _read(service, reader)
            else:
                assert _read(service, reader) == expected
            connection = getattr(db._local, "conn", None)
            assert connection is not None
            assert connection.execute("SELECT 1").fetchone()[0] == 1

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(read_owned).result(timeout=5)
        assert db.registered_connection_count() == 2
    finally:
        # These are fixture-owned handles; all physical callbacks have stopped.
        with db.quiesce_connections(timeout_seconds=5):
            pass
        assert db.registered_connection_count() == 0
