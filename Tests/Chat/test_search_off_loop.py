"""Tests for task-283: move debounced-search DB work off the event loop.

Cover the retained Console search leaves the perf audit (§P1 B4) identified as
running sync sqlite/FTS work on the event loop when run_worker(coroutine)
fires after a debounce:

  * Console browser search -- ChatConversationScopeService.list_conversations
    (chat_conversation_scope_service.py) and the raw-service branch in
    ChatScreen._persisted_console_browser_rows (chat_screen.py).

For each, a file-backed DB runs the leaf via asyncio.to_thread (assert via
threading.get_ident()) and a per-connection ``:memory:`` DB stays inline.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_screen_navigation import _build_test_app


MAIN_THREAD_IDENT = threading.get_ident


# ---------------------------------------------------------------------------
# Console browser search: ChatConversationScopeService.list_conversations
# ---------------------------------------------------------------------------


class _CountingLocalConversationService:
    def __init__(self, *, is_memory_db: bool):
        self.db = SimpleNamespace(is_memory_db=is_memory_db)
        self.calls = 0
        self.thread_idents: list[int] = []

    def list_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.thread_idents.append(threading.get_ident())
        return {"items": [], "pagination": {"total": 0}}


class _CountingServerConversationService:
    def __init__(self):
        self.calls = 0
        self.thread_idents: list[int] = []

    async def list_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.thread_idents.append(threading.get_ident())
        return {"items": [], "pagination": {"total": 0}}


@pytest.mark.asyncio
async def test_scope_service_list_conversations_threads_file_backed_local_db():
    local = _CountingLocalConversationService(is_memory_db=False)
    service = ChatConversationScopeService(local_service=local, server_service=None)
    caller_thread = threading.get_ident()

    await service.list_conversations(mode="local", query="q")

    assert local.calls == 1
    assert local.thread_idents[0] != caller_thread


@pytest.mark.asyncio
async def test_scope_service_list_conversations_stays_inline_for_memory_backed_db():
    local = _CountingLocalConversationService(is_memory_db=True)
    service = ChatConversationScopeService(local_service=local, server_service=None)
    caller_thread = threading.get_ident()

    await service.list_conversations(mode="local", query="q")

    assert local.calls == 1
    assert local.thread_idents[0] == caller_thread


@pytest.mark.asyncio
async def test_scope_service_list_conversations_never_threads_server_mode():
    server = _CountingServerConversationService()
    service = ChatConversationScopeService(local_service=None, server_service=server)
    caller_thread = threading.get_ident()

    result = await service.list_conversations(mode="server", query="q")

    assert server.calls == 1
    assert server.thread_idents[0] == caller_thread
    assert result == {"items": [], "pagination": {"total": 0}}


# ---------------------------------------------------------------------------
# Console browser search: ChatScreen._persisted_console_browser_rows'
# raw-service branch (bypasses the scope service entirely)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persisted_console_browser_rows_threads_raw_file_backed_service():
    app = _build_test_app()
    raw_service = _CountingLocalConversationService(is_memory_db=False)
    app.chat_conversation_scope_service = None
    app.local_chat_conversation_service = raw_service
    screen = ChatScreen(app)
    caller_thread = threading.get_ident()

    await screen._persisted_console_browser_rows("query")

    assert raw_service.calls >= 1
    assert all(ident != caller_thread for ident in raw_service.thread_idents)


@pytest.mark.asyncio
async def test_persisted_console_browser_rows_stays_inline_for_memory_backed_service():
    app = _build_test_app()
    raw_service = _CountingLocalConversationService(is_memory_db=True)
    app.chat_conversation_scope_service = None
    app.local_chat_conversation_service = raw_service
    screen = ChatScreen(app)
    caller_thread = threading.get_ident()

    await screen._persisted_console_browser_rows("query")

    assert raw_service.calls >= 1
    assert all(ident == caller_thread for ident in raw_service.thread_idents)


@pytest.mark.asyncio
async def test_refresh_console_conversation_browser_search_discards_result_when_token_changes_in_flight():
    """AC#2: the existing cancellation-token guard still protects correctness.

    asyncio.to_thread cannot be interrupted mid-flight by a newer debounce
    firing (exclusive=True only cancels a coroutine, not an in-flight
    thread call) -- the token re-check after the await is what discards a
    now-stale result instead of letting it clobber a newer search's rows.
    """
    app = _build_test_app()
    screen = ChatScreen(app)
    screen._console_conversation_browser_query = "hello"
    screen._console_conversation_browser_search_token = 1
    screen._console_conversation_browser_rows = ()
    screen._console_conversation_browser_total = None
    screen._console_conversation_browser_error = ""

    async def fake_persisted_rows(query):
        # Simulate a newer debounce firing while this search was in
        # flight -- the exact race an in-flight thread call can't avoid.
        screen._console_conversation_browser_search_token = 2
        return [SimpleNamespace(row_key="conv-x")], 1, ""

    screen._persisted_console_browser_rows = fake_persisted_rows
    screen._sync_console_workspace_context = lambda: None
    screen.call_after_refresh = lambda fn: None
    screen._filter_console_browser_rows_for_query = lambda rows, query: rows
    screen._merge_console_browser_rows = lambda a, b: tuple(a) + tuple(b)
    screen._native_console_browser_rows = lambda: ()
    screen._membership_console_browser_rows = lambda: ()

    await screen._refresh_console_conversation_browser_search("hello", 1)

    # The persisted-rows result must NOT have been applied: total stays
    # None/unset and the rows list holds only what was already staged
    # before persisted rows returned (empty, since it starts as ()).
    assert screen._console_conversation_browser_total is None
    assert screen._console_conversation_browser_rows == ()
