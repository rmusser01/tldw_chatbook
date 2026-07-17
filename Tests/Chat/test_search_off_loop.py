"""Tests for task-283: move debounced-search DB work off the event loop.

Covers the three search leaves the perf audit (§P1 B4) identified as
running sync sqlite/FTS work on the event loop when run_worker(coroutine)
fires after a debounce:

  * Console browser search -- ChatConversationScopeService.list_conversations
    (chat_conversation_scope_service.py) and the raw-service branch in
    ChatScreen._persisted_console_browser_rows (chat_screen.py).
  * CCP conversation search -- perform_ccp_conversation_search's DB leaf
    (conv_char_events.py).
  * Media search -- perform_media_search_and_display's search_media_db call
    (media_events.py).

For each: a file-backed DB runs the leaf via asyncio.to_thread (assert via
threading.get_ident()); a per-connection ``:memory:`` DB stays inline
(thread-affinity hazard); and where a staleness guard exists/was added, a
result that becomes stale mid-flight is discarded rather than clobbering a
newer search's output.
"""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger as _real_logger
from textual.css.query import QueryError

from tldw_chatbook.Chat.chat_conversation_scope_service import ChatConversationScopeService
from tldw_chatbook.Event_Handlers import conv_char_events, media_events
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_screen_navigation import _build_test_app


MAIN_THREAD_IDENT = threading.get_ident


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeInput:
    def __init__(self, value: str = ""):
        self.value = value
        self.disabled = False


class _FakeCheckbox:
    def __init__(self, value: bool):
        self.value = value


class _FakeSelect:
    def __init__(self, value: Any):
        self.value = value


class _FakeListView:
    def __init__(self):
        self.items: list[Any] = []

    async def clear(self) -> None:
        self.items = []

    async def append(self, item: Any) -> None:
        self.items.append(item)


class _SelectorApp(SimpleNamespace):
    """Minimal query_one-capable stand-in, not a real Textual App/Screen."""

    def query_one(self, selector: Any, widget_type: Any = None) -> Any:
        widgets = self.__dict__.setdefault("_widgets", {})
        widget = widgets.get(selector)
        if widget is None:
            raise QueryError(f"No match for {selector!r}")
        return widget


def _selector_app(**widgets: Any) -> _SelectorApp:
    app = _SelectorApp()
    app._widgets = widgets
    return app


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


# ---------------------------------------------------------------------------
# CCP conversation search
# ---------------------------------------------------------------------------


class _CountingCcpDb:
    def __init__(self, *, is_memory_db: bool, on_call=None):
        self.is_memory_db = is_memory_db
        self.calls = 0
        self.thread_idents: list[int] = []
        self._on_call = on_call

    def _record(self):
        self.calls += 1
        self.thread_idents.append(threading.get_ident())
        if self._on_call is not None:
            self._on_call()

    def list_all_active_conversations(self, *, limit=200):
        self._record()
        return [{"id": "conv-1", "title": "Alpha", "character_id": None}]

    def get_conversations_for_character(self, *, character_id, limit=200):
        self._record()
        return []

    def search_conversations_by_title(self, *, title_query, character_id, limit=200):
        self._record()
        return []

    def search_conversations_by_content(self, term, *, limit=200):
        return []

    def search_keywords(self, term, *, limit=10):
        return []

    def get_conversations_for_keyword(self, keyword_id, *, limit=200):
        return []

    def get_character_card_by_id(self, character_id):
        return None


def _ccp_app(db: Any) -> _SelectorApp:
    app = _selector_app(**{
        "#conv-char-search-input": _FakeInput(""),
        "#conv-char-keyword-search-input": _FakeInput(""),
        "#conv-char-tags-search-input": _FakeInput(""),
        "#conv-char-search-include-character-checkbox": _FakeCheckbox(True),
        "#conv-char-search-all-characters-checkbox": _FakeCheckbox(True),
        "#conv-char-character-select": _FakeSelect(None),
        "#conv-char-search-results-list": _FakeListView(),
    })
    app.notes_service = SimpleNamespace(_get_db=lambda user_id: db)
    app.notes_user_id = "default_user"
    app.loguru_logger = _real_logger
    app._ccp_conversation_search_generation = 0
    return app


@pytest.mark.asyncio
async def test_ccp_search_threads_file_backed_db():
    db = _CountingCcpDb(is_memory_db=False)
    app = _ccp_app(db)
    caller_thread = threading.get_ident()

    await conv_char_events.perform_ccp_conversation_search(app)

    assert db.calls >= 1
    assert all(ident != caller_thread for ident in db.thread_idents)
    results_list = app.query_one("#conv-char-search-results-list", None)
    assert len(results_list.items) == 1


@pytest.mark.asyncio
async def test_ccp_search_stays_inline_for_memory_backed_db():
    db = _CountingCcpDb(is_memory_db=True)
    app = _ccp_app(db)
    caller_thread = threading.get_ident()

    await conv_char_events.perform_ccp_conversation_search(app)

    assert db.calls >= 1
    assert all(ident == caller_thread for ident in db.thread_idents)


@pytest.mark.asyncio
async def test_ccp_search_discards_stale_generation():
    def bump_generation():
        app._ccp_conversation_search_generation += 1

    db = _CountingCcpDb(is_memory_db=False, on_call=bump_generation)
    app = _ccp_app(db)

    await conv_char_events.perform_ccp_conversation_search(app)

    results_list = app.query_one("#conv-char-search-results-list", None)
    # The generation was bumped (simulating a newer search starting) while
    # this search's DB work was off-thread -- its results must be dropped,
    # leaving the list exactly where the pre-search clear() left it.
    assert results_list.items == []


# ---------------------------------------------------------------------------
# Media search
# ---------------------------------------------------------------------------


class _CountingMediaDb:
    def __init__(self, *, is_memory_db: bool, on_call=None):
        self.is_memory_db = is_memory_db
        self.calls = 0
        self.thread_idents: list[int] = []
        self._on_call = on_call

    def search_media_db(self, **kwargs):
        self.calls += 1
        self.thread_idents.append(threading.get_ident())
        if self._on_call is not None:
            self._on_call()
        return [{"id": 1, "title": "Doc", "type": "document"}], 1


def _media_app(db: Any, *, type_slug: str = "all-media") -> _SelectorApp:
    app = _selector_app(**{
        f"#media-list-view-{type_slug}": _FakeListView(),
    })
    app.media_db = db
    app.media_current_page = 1
    app.loguru_logger = _real_logger
    app._media_search_generation = {}
    return app


def _stub_media_window_module(monkeypatch) -> None:
    """Work around an unrelated, pre-existing dead import.

    ``perform_media_search_and_display`` does ``from ..UI.MediaWindow
    import MediaWindow`` inside a ``try/except QueryError`` -- but
    ``tldw_chatbook/UI/MediaWindow.py`` no longer exists (renamed to
    ``MediaWindow_v2.py``/``MediaWindowV88.py`` well before task-283), so
    the import always raises ``ModuleNotFoundError``, which that narrow
    ``except QueryError`` does NOT catch. In production this is swallowed
    by the function's outer broad ``except Exception`` (media search
    always falls into the "Error loading" branch instead of ever reaching
    the DB call this task threads) -- a real, pre-existing bug, but fixing
    it is out of scope for task-283 (STOP-don't-improvise per site; flagged
    in the task's Implementation Notes instead). Stub the module so this
    test can actually exercise the threaded search_media_db call.
    """
    stub_module = types.ModuleType("tldw_chatbook.UI.MediaWindow")
    stub_module.MediaWindow = type("MediaWindow", (), {})
    monkeypatch.setitem(sys.modules, "tldw_chatbook.UI.MediaWindow", stub_module)


@pytest.mark.asyncio
async def test_media_search_threads_file_backed_db(monkeypatch):
    _stub_media_window_module(monkeypatch)
    db = _CountingMediaDb(is_memory_db=False)
    app = _media_app(db)
    caller_thread = threading.get_ident()

    await media_events.perform_media_search_and_display(app, "all-media", "term", "")

    assert db.calls == 1
    assert db.thread_idents[0] != caller_thread
    results_list = app.query_one("#media-list-view-all-media", None)
    assert len(results_list.items) == 1


@pytest.mark.asyncio
async def test_media_search_stays_inline_for_memory_backed_db(monkeypatch):
    _stub_media_window_module(monkeypatch)
    db = _CountingMediaDb(is_memory_db=True)
    app = _media_app(db)
    caller_thread = threading.get_ident()

    await media_events.perform_media_search_and_display(app, "all-media", "term", "")

    assert db.calls == 1
    assert db.thread_idents[0] == caller_thread


@pytest.mark.asyncio
async def test_media_search_discards_stale_generation(monkeypatch):
    _stub_media_window_module(monkeypatch)

    def bump_generation():
        app._media_search_generation["all-media"] += 1

    db = _CountingMediaDb(is_memory_db=False, on_call=bump_generation)
    app = _media_app(db)

    await media_events.perform_media_search_and_display(app, "all-media", "term", "")

    results_list = app.query_one("#media-list-view-all-media", None)
    # A newer search's generation bump while this one's DB call was
    # off-thread must discard these results (list stays at whatever the
    # pre-search clear() left, i.e. empty).
    assert results_list.items == []
