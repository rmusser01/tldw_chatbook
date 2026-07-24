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

P2g-2 Task 4 (below): the World Books inspector block's Attach/Detach
actions + workers -- a real-DB round trip mirroring
`Tests/UI/test_console_dictionaries_attach.py` (the chat-dictionary Console
attach/detach test), driving the REAL `WorldBookManager` over the REAL db
rather than a fake.
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
from tldw_chatbook.Widgets.Persona_Widgets.world_book_picker import WorldBookPicker


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


# --- P2g-2 Task 4: Attach/Detach actions + workers --------------------------


@pytest.mark.asyncio
async def test_console_worldbook_attach_then_detach_round_trips_through_real_db(
    wb_db, monkeypatch
):
    """A real-DB round trip: the attach worker persists a genuine
    ``conversation_world_books`` junction row (not a stub), the inspector
    cache reflects it, and the detach worker removes it again.
    """
    conv_id = wb_db.add_conversation({"title": "Attach flow"})
    manager = WorldBookManager(wb_db)
    book_id = manager.create_world_book("Standalone Lore")
    manager.create_world_book_entry(book_id, keys=["a"], content="x")

    app = _build_test_app()
    app.chachanotes_db = wb_db

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = conv_id

        async def _fake_push_screen_wait(picker):
            return book_id if isinstance(picker, WorldBookPicker) else None

        monkeypatch.setattr(
            screen.app_instance,
            "push_screen_wait",
            _fake_push_screen_wait,
            raising=False,
        )

        await screen.refresh_active_world_books_summary()
        assert manager.get_world_books_for_conversation(conv_id, enabled_only=False) == []

        # --- Attach ---
        await screen._console_worldbook_attach_worker()
        await pilot.pause()

        attached = manager.get_world_books_for_conversation(conv_id, enabled_only=False)
        assert [b["id"] for b in attached] == [book_id]
        rows = screen._console_world_book_inspector_rows()
        assert any("Standalone Lore" in row.text for row in rows)
        assert screen._console_worldbook_dialog_active is False

        # --- Detach (same monkeypatched picker returns the same id) ---
        await screen._console_worldbook_detach_worker()
        await pilot.pause()

        assert manager.get_world_books_for_conversation(conv_id, enabled_only=False) == []
        rows_after = screen._console_world_book_inspector_rows()
        assert rows_after == (ConsoleDisplayRow("No world books in play", ""),)
        assert screen._console_worldbook_dialog_active is False


@pytest.mark.asyncio
async def test_console_worldbook_attach_notifies_and_noops_without_a_conversation(
    wb_db, monkeypatch
):
    """No active native-session conversation -> the attach worker must not
    touch the DB. The attach action is *disabled* with no conversation (see
    the gating test below), so this drives the worker directly -- exercising
    the same production guard the button-press handler defers to.
    """
    app = _build_test_app()
    app.chachanotes_db = wb_db

    calls: list = []
    monkeypatch.setattr(app, "notify", lambda *a, **k: calls.append((a, k)))

    push_calls: list = []

    async def _fake_push_screen_wait(picker):
        push_calls.append(picker)
        return None

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        assert _active_native_session(screen).persisted_conversation_id is None

        monkeypatch.setattr(
            screen.app_instance,
            "push_screen_wait",
            _fake_push_screen_wait,
            raising=False,
        )

        await screen._console_worldbook_attach_worker()
        await pilot.pause()

        assert push_calls == []  # never even opened the picker
        assert any("conversation" in str(a).lower() for a, _k in calls)
        assert screen._console_worldbook_dialog_active is False


@pytest.mark.asyncio
async def test_console_world_book_inspector_actions_gate_on_conversation_and_attached(
    wb_db,
):
    """``_console_world_book_inspector_actions`` gates Attach on a
    conversation id being active and Detach on >=1 currently-attached book --
    read purely from the cache + conversation id, never the DB.
    """
    conv_id = wb_db.add_conversation({"title": "Gate flow"})
    manager = WorldBookManager(wb_db)
    book_id = manager.create_world_book("Gate Lore")
    manager.create_world_book_entry(book_id, keys=["a"], content="x")

    app = _build_test_app()
    app.chachanotes_db = wb_db

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")

        # No conversation yet: Attach disabled, Detach disabled (no summary).
        actions = screen._console_world_book_inspector_actions()
        attach = next(
            a for a in actions if a.widget_id == "console-inspector-worldbooks-attach"
        )
        detach = next(
            a for a in actions if a.widget_id == "console-inspector-worldbooks-detach"
        )
        assert attach.enabled is False
        assert detach.enabled is False

        # Conversation active, but nothing attached yet: Attach enabled,
        # Detach still disabled.
        _active_native_session(screen).persisted_conversation_id = conv_id
        await screen.refresh_active_world_books_summary()
        actions = screen._console_world_book_inspector_actions()
        attach = next(
            a for a in actions if a.widget_id == "console-inspector-worldbooks-attach"
        )
        detach = next(
            a for a in actions if a.widget_id == "console-inspector-worldbooks-detach"
        )
        assert attach.enabled is True
        assert detach.enabled is False

        # After attaching: Detach becomes enabled too.
        manager.associate_world_book_with_conversation(conv_id, book_id)
        await screen.refresh_active_world_books_summary()
        actions = screen._console_world_book_inspector_actions()
        detach = next(
            a for a in actions if a.widget_id == "console-inspector-worldbooks-detach"
        )
        assert detach.enabled is True
