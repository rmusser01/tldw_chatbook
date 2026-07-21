"""P2g-2 Task 3: `ChatScreen` caches the "what's in play" world-book summary
on native Console session change and feeds it into the Console inspector
build with ZERO DB I/O on recompose.

Mirrors `Tests/UI/test_console_dictionaries_screen.py` (the chat-dictionary
inspector read wiring), but drives a REAL `CharactersRAGDB` +
`WorldBookManager` instead of a fake scope service -- world-book
summarization (`summarize_active_world_books`, P2g-2 Task 1) always reads the
DB directly rather than going through an app-level scope service.

`refresh_active_world_books_summary()` is the ONLY place allowed to call
`summarize_active_world_books`; a bare `_build_console_inspector_state()`
call (the same path every Console recompose/refresh goes through) must read
only the cached `self._active_world_books_summary` set by the last refresh.
"""

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_display_state import ConsoleDisplayRow
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _active_native_session(console: ChatScreen):
    store = console._ensure_console_chat_store()
    return next(s for s in store.sessions() if s.id == store.active_session_id)


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "console_worldbook_inspector.db", "test-client")
    yield db
    db.close_connection()


@pytest.mark.asyncio
async def test_refresh_caches_summary_and_build_projects_world_book_rows(wb_db):
    wb_db.add_conversation({"id": "conv-wb", "title": "C"})
    manager = WorldBookManager(wb_db)
    book_a = manager.create_world_book("Alpha")
    manager.create_world_book_entry(book_a, keys=["a"], content="x")
    manager.create_world_book_entry(book_a, keys=["b"], content="y")
    book_b = manager.create_world_book("Beta")
    manager.create_world_book_entry(book_b, keys=["c"], content="z")
    manager.associate_world_book_with_conversation("conv-wb", book_a)
    manager.associate_world_book_with_conversation("conv-wb", book_b)

    app = _build_test_app()
    app.chachanotes_db = wb_db

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = "conv-wb"

        await screen.refresh_active_world_books_summary()

        rows = screen._console_world_book_inspector_rows()
        row_texts = [row.text for row in rows]
        assert any("Alpha" in text for text in row_texts)
        assert any("Beta" in text for text in row_texts)

        inspector_state = screen._build_console_inspector_state(None)
        assert inspector_state.world_book_rows == rows
        inspector_row_texts = [row.text for row in inspector_state.world_book_rows]
        assert any("Alpha" in text for text in inspector_row_texts)
        assert any("Beta" in text for text in inspector_row_texts)


@pytest.mark.asyncio
async def test_build_console_inspector_state_never_re_queries_the_db(wb_db, monkeypatch):
    wb_db.add_conversation({"id": "conv-wb", "title": "C"})
    manager = WorldBookManager(wb_db)
    book_a = manager.create_world_book("Alpha")
    manager.create_world_book_entry(book_a, keys=["a"], content="x")
    manager.associate_world_book_with_conversation("conv-wb", book_a)

    app = _build_test_app()
    app.chachanotes_db = wb_db

    calls: list[tuple] = []
    from tldw_chatbook.Character_Chat import world_info_resolver as wir

    real_summarize = wir.summarize_active_world_books

    def _tracking_summarize(db, conversation_id, char_data):
        calls.append((conversation_id, char_data))
        return real_summarize(db, conversation_id, char_data)

    monkeypatch.setattr(
        "tldw_chatbook.Character_Chat.world_info_resolver.summarize_active_world_books",
        _tracking_summarize,
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = "conv-wb"

        await screen.refresh_active_world_books_summary()
        assert len(calls) == 1

        # A bare, recompose-equivalent build must read only the cache -- no
        # re-invocation of the DB-backed summarize call.
        screen._build_console_inspector_state(None)
        screen._build_console_inspector_state(None)
        screen._build_console_inspector_state(None)

        assert len(calls) == 1


@pytest.mark.asyncio
async def test_no_active_chat_renders_no_active_chat_row(wb_db):
    app = _build_test_app()
    app.chachanotes_db = wb_db

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        # Fresh default native session has no persisted conversation yet.
        assert _active_native_session(screen).persisted_conversation_id is None

        await screen.refresh_active_world_books_summary()

        rows = screen._console_world_book_inspector_rows()
        assert rows == (ConsoleDisplayRow("No active chat", ""),)


@pytest.mark.asyncio
async def test_conversation_with_no_books_renders_no_world_books_in_play_row(wb_db):
    wb_db.add_conversation({"id": "conv-empty", "title": "C"})

    app = _build_test_app()
    app.chachanotes_db = wb_db

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = "conv-empty"

        await screen.refresh_active_world_books_summary()

        rows = screen._console_world_book_inspector_rows()
        assert rows == (ConsoleDisplayRow("No world books in play", ""),)


@pytest.mark.asyncio
async def test_sync_native_console_chat_ui_refreshes_world_books_on_scope_change(wb_db):
    wb_db.add_conversation({"id": "conv-wb", "title": "C"})
    manager = WorldBookManager(wb_db)
    book_a = manager.create_world_book("Alpha")
    manager.create_world_book_entry(book_a, keys=["a"], content="x")
    manager.associate_world_book_with_conversation("conv-wb", book_a)

    app = _build_test_app()
    app.chachanotes_db = wb_db

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = "conv-wb"

        await screen._sync_native_console_chat_ui()

        rows = screen._console_world_book_inspector_rows()
        assert any("Alpha" in row.text for row in rows)
