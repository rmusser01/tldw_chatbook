"""Finite History reads retire new worker handles, never warm borrowers."""

from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_activity_switcher import (
    _projection_controller,
    _ReceiptSnapshot,
)
from Tests.UI.test_console_workspace_controller import _workspace_controller
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.bootstrap_profile


def _controller(database):
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=_ReceiptSnapshot(),
        ),
        local_chat_conversation_service=ChatConversationService(database),
    )
    controller = _projection_controller(app)
    controller._console_browser_workspace_labels = dict
    return controller


@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("sql_error", [False, True], ids=["success", "sql-error"])
def test_history_page_retires_only_its_new_native_worker_handle(
    tmp_path, monkeypatch, borrowed, sql_error
):
    database = CharactersRAGDB(tmp_path / "history-owned.sqlite", "history-owned")
    conversation_id = database.add_conversation({"title": "Native history"})
    controller = _controller(database)
    original_get = database.get_connection
    original_close = database.close_connection
    main_thread = get_ident()
    worker_connections, closes = [], []

    def observed_get():
        connection = original_get()
        if get_ident() != main_thread:
            worker_connections.append(connection)
        return connection

    def observed_close():
        closes.append(get_ident())
        return original_close()

    monkeypatch.setattr(database, "get_connection", observed_get)
    monkeypatch.setattr(database, "close_connection", observed_close)
    if sql_error:

        def failed_query(*_args, **_kwargs):
            with database.transaction() as connection:
                connection.execute("SELECT * FROM missing_history_fixture_table")

        monkeypatch.setattr(database, "search_conversations_page", failed_query)

    async def exercise():
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )

        def borrow():
            transaction = database.transaction()
            transaction.__enter__()
            return transaction, original_get()

        transaction, warm = (
            await asyncio.to_thread(borrow) if borrowed else (None, None)
        )
        count = database.registered_connection_count()
        try:
            page = await controller.load_console_session_switcher_history(
                query="", offset=0, limit=50
            )
            if sql_error:
                assert page.entries == () and page.error == "History is unavailable."
            else:
                assert [entry.conversation_id for entry in page.entries] == [
                    conversation_id
                ]
            assert worker_connections
            assert database.registered_connection_count() == count
            if borrowed:
                assert all(connection is warm for connection in worker_connections)
                assert warm.execute("SELECT 1").fetchone()[0] == 1
                assert warm.in_transaction
                assert closes == []
            else:
                for connection in worker_connections:
                    with pytest.raises(
                        sqlite3.ProgrammingError, match="closed database"
                    ):
                        connection.execute("SELECT 1")
                assert len(closes) == 1 and closes[0] != main_thread
        finally:
            if transaction is not None:
                await asyncio.to_thread(transaction.__exit__, None, None, None)
            await asyncio.to_thread(original_close)

    try:
        asyncio.run(exercise())
    finally:
        original_close()


@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm"])
def test_flat_browser_read_retires_only_its_own_handle_without_checkpoint(
    tmp_path, monkeypatch, borrowed
):
    database = CharactersRAGDB(tmp_path / "browser-owned.sqlite", "browser-owned")
    conversation_id = database.add_conversation({"title": "Native saved page"})
    controller = _workspace_controller(
        app_instance=SimpleNamespace(
            local_chat_conversation_service=ChatConversationService(database)
        )
    )
    original_close = database.close_connection
    close_modes = []

    def observed_close(*, checkpoint=True):
        close_modes.append(checkpoint)
        return original_close(checkpoint=checkpoint)

    monkeypatch.setattr(database, "close_connection", observed_close)

    async def exercise():
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )
        transaction = database.transaction()
        if borrowed:
            await asyncio.to_thread(transaction.__enter__)
        try:
            rows, total, error = await controller._persisted_console_browser_rows()
            assert [row.conversation_id for row in rows] == [conversation_id]
            assert total == 1 and error == ""
            assert database.registered_connection_count() == (2 if borrowed else 1)
            assert close_modes == ([] if borrowed else [False, False])
            if borrowed:
                assert await asyncio.to_thread(
                    lambda: database.get_connection().in_transaction
                )
        finally:
            if borrowed:
                await asyncio.to_thread(transaction.__exit__, None, None, None)
            await asyncio.to_thread(original_close)

    try:
        asyncio.run(exercise())
    finally:
        original_close()
    assert database.registered_connection_count() == 0


@pytest.mark.parametrize("sql_error", [False, True], ids=["success", "sql-error"])
def test_cancelled_history_waiter_does_not_close_a_running_native_read(
    tmp_path, monkeypatch, sql_error
):
    database = CharactersRAGDB(tmp_path / "history-cancel.sqlite", "history-cancel")
    database.add_conversation({"title": "Held history"})
    controller = _controller(database)
    entered, release, retired = Event(), Event(), Event()
    captured = []
    original_search = database.search_conversations_page
    original_close = database.close_connection
    close_threads = []
    callback_threads = []

    def held_search(*args, **kwargs):
        connection = database.get_connection()
        captured.append(connection)
        callback_threads.append(get_ident())
        entered.set()
        assert release.wait(5), "History native read was not released"
        if sql_error:
            connection.execute("SELECT * FROM missing_history_cancel_table")
        return original_search(*args, **kwargs)

    def observed_close():
        result = original_close()
        close_threads.append(get_ident())
        retired.set()
        return result

    monkeypatch.setattr(database, "search_conversations_page", held_search)
    monkeypatch.setattr(database, "close_connection", observed_close)

    async def exercise():
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )

        async def wait(event):
            async with asyncio.timeout(2):
                while not event.is_set():
                    await asyncio.sleep(0.01)

        task = asyncio.create_task(
            controller.load_console_session_switcher_history(
                query="", offset=0, limit=50
            )
        )
        try:
            await wait(entered)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not retired.is_set()
            assert captured[0].execute("SELECT 1").fetchone()[0] == 1
            release.set()
            await wait(retired)
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                captured[0].execute("SELECT 1")
            assert close_threads == callback_threads
            assert database.registered_connection_count() == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(original_close)

    try:
        asyncio.run(exercise())
    finally:
        original_close()
