"""Finite Console readers leave no newly opened native worker connection."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from Tests.Chat.test_citation_trace_repository import _persist, _repository
from Tests.Chat.test_local_marks_read_ownership import _CustomDatabase
from Tests.UI.test_console_citation_sources import _bare_screen, _message
from Tests.UI.test_console_retrieval_controller import _controller as retrieval_owner
from Tests.UI.test_console_review_selection_controller import (
    _controller as review_owner,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    resolve_scope_for_session,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def database(tmp_path):
    """Retire this exact owner only after each asyncio runner joins its workers."""
    db = CharactersRAGDB(tmp_path / "finite-console.sqlite", "finite-console-test")
    db.add_conversation({"id": "conversation", "title": "Finite readers"})
    try:
        yield db
    finally:
        with db.quiesce_connections(timeout_seconds=5):
            pass
        assert db.registered_connection_count() == 0


@pytest.mark.parametrize(
    "reader",
    [
        "cached_scope",
        "fresh_scope",
        "world_books",
        "annotations",
        "annotation_browser",
        "citations",
    ],
)
@pytest.mark.parametrize("sql_failure", [False, True])
def test_console_read_retires_its_worker_handle_after_results_or_sql_failure(
    database, reader, sql_failure
):
    """An ordinary to_thread read retains a cold cache after its worker exits.

    Args:
        database: Real file-backed owner with its original main-thread cache.
        reader: Actual production callback family from the shutdown trace.
        sql_failure: Make its real SQL fail after worker handle acquisition.
    """
    if reader in {"cached_scope", "fresh_scope"}:
        app = SimpleNamespace(chachanotes_db=database)
        session = SimpleNamespace(persisted_conversation_id="conversation")

        async def read():
            result = await resolve_scope_for_session(
                app, session, use_cache=reader == "cached_scope"
            )
            return result.effective.state

        table = "conversations"
        expected = "empty" if sql_failure and reader == "fresh_scope" else "unscoped"
    elif reader == "world_books":
        manager = WorldBookManager(database)
        book = manager.create_world_book("Atlas")
        manager.create_world_book_entry(book, keys=["atlas"], content="Synthetic")
        manager.associate_world_book_with_conversation("conversation", book)
        owner, _ = retrieval_owner()
        owner.app_instance.chachanotes_db = database
        owner._current_conversation_id = lambda: "conversation"

        async def read():
            await owner.refresh_active_world_books_summary()
            return owner._active_world_books_summary

        table = "world_books"
        expected = {
            "world_books": []
            if sql_failure
            else [{"name": "Atlas", "enabled": True, "entry_count": 1}],
            "source": "local",
        }
    elif reader in {"annotations", "annotation_browser"}:
        database.add_message(
            {
                "id": "message",
                "conversation_id": "conversation",
                "sender": "assistant",
                "content": "Synthetic",
                "client_id": database.client_id,
            }
        )
        database.upsert_transcript_annotation(
            conversation_id="conversation",
            row_key="message:message",
            message_id="message",
            quote_text="Synthetic",
            comment="Retained note",
        )
        messages = [SimpleNamespace(id="native", persisted_message_id="message")]
        if reader == "annotations":
            owner = review_owner(native_messages_accessor=lambda: messages)
            owner.annotation_loaded_conversation = "conversation"

            async def read():
                await owner._load_console_annotation_previews(
                    database, object(), "conversation"
                )
                return owner.annotation_previews

            expected = {} if sql_failure else {"native": ("Retained note",)}
        else:
            comments = []

            async def dismiss(modal):
                comments.extend(note["comment"] for note in modal._notes.values())
                return False

            store = SimpleNamespace(
                persistence=SimpleNamespace(db=database),
                active_session_id="session",
                _sessions={
                    "session": SimpleNamespace(persisted_conversation_id="conversation")
                },
            )
            owner = SimpleNamespace(
                _ensure_console_chat_controller=lambda: SimpleNamespace(store=store),
                _message=SimpleNamespace(_native_console_messages=lambda: messages),
                app=SimpleNamespace(push_screen_wait=dismiss),
                notify=lambda *_args, **_kwargs: None,
            )

            async def read():
                comments.clear()
                owner._console_review_notes_inflight = True
                await ChatScreen._console_review_notes_flow(owner, "native")
                assert owner._console_review_notes_inflight is False
                return comments

            expected = [] if sql_failure else ["Retained note"]

        table = "transcript_annotations"
    else:
        repository = _repository(database)
        _persist(database, repository)
        message = _message("native", persisted_message_id="message-1")
        owner = _bare_screen([message], repository, app_db=database)
        signature = owner._console_citation_signature([message])
        owner._console_citation_input_signature = signature
        owner._console_citation_request_generation = 1

        async def read():
            await owner._discover_console_citation_counts(repository, signature, 1)
            return owner._console_citation_counts

        table = "rag_message_trace_owners"
        expected = {"native": 0 if sql_failure else 1}

    if sql_failure:
        # All names above are literal test-owned schema identifiers, not input.
        database.get_connection().execute(
            f"ALTER TABLE {table} RENAME TO unavailable_reader_table"
        )
    assert database.registered_connection_count() == 1
    for _ in range(2):
        assert asyncio.run(read()) == expected
        assert database.registered_connection_count() == 1
    assert database.get_connection().execute("SELECT 1").fetchone()[0] == 1


def test_projection_read_preserves_borrowed_chat_transaction(database):
    """Shared projection must not close or settle its caller's transaction."""
    service = ChatPersistenceService(database)

    def borrow():
        connection = database.get_connection()
        try:
            connection.execute("BEGIN")
            connection.execute(
                "UPDATE conversations SET title = 'Borrowed' WHERE id = 'conversation'"
            )
            assert service.project_workspace_membership("conversation") is None
            assert database.get_connection() is connection
            assert connection.in_transaction
            assert connection.execute("SELECT title FROM conversations").fetchone()[
                0
            ] == ("Borrowed")
        finally:
            connection.rollback()
            database.close_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(borrow).result(timeout=5)
    assert database.registered_connection_count() == 1
    assert database.get_conversation_by_id("conversation")["title"] == "Finite readers"


@pytest.mark.parametrize("owner", ["memory", "custom"])
def test_projection_read_preserves_excluded_chat_owner(tmp_path, owner):
    """Exact-file ownership must not retire memory or subclass caches."""
    db = (
        CharactersRAGDB(":memory:", "projection-memory")
        if owner == "memory"
        else _CustomDatabase(tmp_path / "projection-custom.sqlite", "projection-custom")
    )
    try:
        db.add_conversation({"id": "conversation", "title": "Excluded owner"})
        service = ChatPersistenceService(db)

        def read_owned():
            if owner == "memory":
                with pytest.raises(CharactersRAGDBError, match="no such table"):
                    service.project_workspace_membership("conversation")
            else:
                assert service.project_workspace_membership("conversation") is None
            connection = getattr(db._local, "conn", None)
            assert connection is not None
            assert connection.execute("SELECT 1").fetchone()[0] == 1

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(read_owned).result(timeout=5)
        assert db.registered_connection_count() == 2
    finally:
        with db.quiesce_connections(timeout_seconds=5):
            pass
        assert db.registered_connection_count() == 0
