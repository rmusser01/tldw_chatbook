"""Tests for the Console Inspector "Retrieval scope" row (task-9).

Covers: row placement (below the Sources tray, a sibling in the Inspector
rail body -- the task-400 placement-test pattern), unscoped/scoped display
states, zero-DB-on-recompose, Edit/Narrow opening the picker modal wired to
the real listers (fake app-level seams), save-persisted writing through
(honest notify on a corrupt-metadata refusal) and refreshing the row + the
Inspector run-recipe line, save-unpersisted holding the scope on the
session, and Clear working in both branches.
"""
from __future__ import annotations

import time

import pytest
from textual.widgets import Static

from Tests.UI.test_console_native_chat_flow import _static_plain_text
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.rag_scope import (
    RagScope,
    ScopeItem,
    read_conversation_scope,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console.console_retrieval_scope_row import (
    CLEAR_BTN_ID,
    EDIT_BTN_ID,
    LABEL_ID,
    NARROW_BTN_ID,
    ROW_ID,
)
from tldw_chatbook.Widgets.Console.console_scope_picker_modal import (
    ConsoleScopePickerModal,
)


async def _open_console_inspector(console, pilot) -> None:
    """Open the persistent Inspector rail and wait for measurable layout.

    Mirrors ``test_console_internals_decomposition.py``'s helper of the
    same name.
    """
    right_rail = console.query_one("#console-right-rail")
    if getattr(right_rail, "display", False) and right_rail.region.width > 0:
        return
    await pilot.click("#console-inspector-rail-open")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        right_rail = console.query_one("#console-right-rail")
        if getattr(right_rail, "display", False) and right_rail.region.width > 0:
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for Console Inspector rail to open")


class _SpyMediaReadingScopeService:
    def __init__(self, items=None):
        self.items = items or []
        self.calls: list[dict] = []

    async def search_media(self, *, mode=None, query=None, limit=20, offset=0, **filters):
        self.calls.append({"mode": mode, "query": query, "limit": limit, "offset": offset})
        return {"items": self.items, "total": len(self.items), "offset": offset, "limit": limit}


class _SpyNotesScopeService:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls: list[dict] = []

    async def search_notes(self, *, scope, query, limit=10, user_id=None, **kwargs):
        self.calls.append({"query": query, "limit": limit})
        return self.rows


async def _open_inspector_and_get_row(console, pilot):
    await _open_console_inspector(console, pilot)
    return console.query_one(f"#{ROW_ID}")


@pytest.mark.asyncio
async def test_retrieval_scope_row_sits_directly_below_sources_tray_above_run_inspector():
    """Placement pinned: sibling of the tray in the Inspector rail body,
    never nested inside it or inside ConsoleRunInspector (task-400-style
    placement test)."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)

        rail_body = console.query_one("#console-inspector-rail-body")
        tray = console.query_one("#console-staged-context-tray")
        row = console.query_one(f"#{ROW_ID}")
        inspector = console.query_one("#console-run-inspector")

        assert tray.parent is rail_body
        assert row.parent is rail_body
        assert inspector.parent is rail_body
        assert tray.region.y < row.region.y < inspector.region.y


@pytest.mark.asyncio
async def test_retrieval_scope_row_default_state_is_unscoped():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        row = await _open_inspector_and_get_row(console, pilot)

        label = row.query_one(f"#{LABEL_ID}", Static)
        assert _static_plain_text(label) == "Scope: everything"
        assert list(row.query(f"#{NARROW_BTN_ID}"))
        assert not list(row.query(f"#{EDIT_BTN_ID}"))
        assert not list(row.query(f"#{CLEAR_BTN_ID}"))


@pytest.mark.asyncio
async def test_retrieval_scope_row_reflects_held_scope_for_unpersisted_session():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        row = await _open_inspector_and_get_row(console, pilot)

        session = console._active_native_console_session()
        assert session is not None
        session.rag_scope_holder.set(
            RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", "n1")),
                updated_at="2026-01-01T00:00:00Z",
            )
        )
        console._sync_console_retrieval_scope_row()
        await pilot.pause()

        row = console.query_one(f"#{ROW_ID}")
        label = row.query_one(f"#{LABEL_ID}", Static)
        assert _static_plain_text(label) == "Scope: 2 items"
        assert list(row.query(f"#{EDIT_BTN_ID}"))
        assert list(row.query(f"#{CLEAR_BTN_ID}"))
        assert not list(row.query(f"#{NARROW_BTN_ID}"))


@pytest.mark.asyncio
async def test_retrieval_scope_row_zero_db_on_forced_recompose():
    """A persisted-but-not-yet-cached conversation id must render the safe
    "everything" default on recompose WITHOUT ever touching the DB -- the
    brief's two read triggers are modal-open and after-save, never
    compose/recompose."""

    class _RaisingDB:
        def __init__(self):
            self.calls: list[str] = []

        def get_conversation_by_id(self, *args, **kwargs):
            self.calls.append("get_conversation_by_id")
            raise AssertionError("compose/recompose must never touch the DB")

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        row = await _open_inspector_and_get_row(console, pilot)
        assert _static_plain_text(row.query_one(f"#{LABEL_ID}", Static)) == "Scope: everything"

        session = console._active_native_console_session()
        session.persisted_conversation_id = "conv-not-cached"
        raising_db = _RaisingDB()
        app.chachanotes_db = raising_db

        console.refresh(recompose=True)
        await pilot.pause()
        for _ in range(20):
            await pilot.pause(0.02)

        assert raising_db.calls == []
        row = console.query_one(f"#{ROW_ID}")
        assert _static_plain_text(row.query_one(f"#{LABEL_ID}", Static)) == "Scope: everything"


@pytest.mark.asyncio
async def test_narrow_button_opens_modal_with_real_listers_wired():
    app = _build_test_app()
    media_spy = _SpyMediaReadingScopeService()
    notes_spy = _SpyNotesScopeService()
    app.media_reading_scope_service = media_spy
    app.notes_scope_service = notes_spy
    app.media_db = None
    app.chachanotes_db = None
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_inspector_and_get_row(console, pilot)

        await pilot.click(f"#{NARROW_BTN_ID}")
        await pilot.pause()
        for _ in range(20):
            await pilot.pause(0.02)

        modals = [s for s in host.screen_stack if isinstance(s, ConsoleScopePickerModal)]
        assert len(modals) == 1
        modal = modals[0]
        assert modal._target_label == "Chat 1"
        assert modal._universe is None
        # Real listers wired all the way through to the app-level seams
        # (the fake spies above), not a placeholder/no-op.
        assert media_spy.calls


@pytest.mark.asyncio
async def test_edit_button_seeds_modal_from_held_scope():
    app = _build_test_app()
    app.media_reading_scope_service = _SpyMediaReadingScopeService()
    app.notes_scope_service = _SpyNotesScopeService()
    app.media_db = None
    app.chachanotes_db = None
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_inspector_and_get_row(console, pilot)
        session = console._active_native_console_session()
        session.rag_scope_holder.set(
            RagScope(items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z")
        )
        console._sync_console_retrieval_scope_row()
        await pilot.pause()

        await pilot.click(f"#{EDIT_BTN_ID}")
        await pilot.pause()
        for _ in range(20):
            await pilot.pause(0.02)

        modals = [s for s in host.screen_stack if isinstance(s, ConsoleScopePickerModal)]
        assert len(modals) == 1
        modal = modals[0]
        assert ("media", "m1") in modal._selected


@pytest.mark.asyncio
async def test_save_unpersisted_stores_in_holder_and_refreshes_row():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        row = await _open_inspector_and_get_row(console, pilot)
        session = console._active_native_console_session()
        assert session.persisted_conversation_id is None
        scope = RagScope(items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z")

        await console._apply_console_retrieval_scope_save(session, scope)
        await pilot.pause()

        assert session.rag_scope_holder.scope == scope
        row = console.query_one(f"#{ROW_ID}")
        assert _static_plain_text(row.query_one(f"#{LABEL_ID}", Static)) == "Scope: 1 items"


@pytest.mark.asyncio
async def test_save_persisted_writes_through_refreshes_row_and_run_recipe():
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            row = await _open_inspector_and_get_row(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            conversation_id = store.persist_session_if_needed(session.id)
            assert conversation_id is not None
            scope = RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", "n1")),
                updated_at="2026-01-01T00:00:00Z",
            )

            await console._apply_console_retrieval_scope_save(session, scope)
            await pilot.pause()

            assert read_conversation_scope(db, conversation_id) == scope
            row = console.query_one(f"#{ROW_ID}")
            assert (
                _static_plain_text(row.query_one(f"#{LABEL_ID}", Static))
                == "Scope: 2 items"
            )
            assert "scope 2 items" in _visible_text(console)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_save_persisted_corrupt_metadata_surfaces_honest_notify():
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        notifications: list[tuple[str, dict]] = []
        app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            row = await _open_inspector_and_get_row(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            conversation_id = store.persist_session_if_needed(session.id)
            record = db.get_conversation_by_id(conversation_id)
            db.update_conversation(
                conversation_id,
                {"metadata": "{not valid json"},
                expected_version=record["version"],
            )
            scope = RagScope(
                items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z"
            )

            await console._apply_console_retrieval_scope_save(session, scope)
            await pilot.pause()

            assert notifications, "expected an honest notify on write refusal"
            message, kwargs = notifications[-1]
            assert "corrupt" in message.lower()
            assert kwargs.get("severity") == "error"
            # The write was genuinely refused -- metadata is untouched, and
            # the row must not have flipped to a false "saved" scoped state.
            raw = db.get_conversation_by_id(conversation_id)
            assert raw["metadata"] == "{not valid json"
            row = console.query_one(f"#{ROW_ID}")
            assert (
                _static_plain_text(row.query_one(f"#{LABEL_ID}", Static))
                == "Scope: everything"
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_clear_button_unpersisted_session():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        row = await _open_inspector_and_get_row(console, pilot)
        session = console._active_native_console_session()
        session.rag_scope_holder.set(
            RagScope(items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z")
        )
        console._sync_console_retrieval_scope_row()
        await pilot.pause()

        await pilot.click(f"#{CLEAR_BTN_ID}")
        await pilot.pause()
        for _ in range(20):
            await pilot.pause(0.02)

        assert session.rag_scope_holder.scope is None
        row = console.query_one(f"#{ROW_ID}")
        assert _static_plain_text(row.query_one(f"#{LABEL_ID}", Static)) == "Scope: everything"
        assert list(row.query(f"#{NARROW_BTN_ID}"))


@pytest.mark.asyncio
async def test_clear_button_persisted_session():
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            row = await _open_inspector_and_get_row(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            conversation_id = store.persist_session_if_needed(session.id)
            scope = RagScope(
                items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z"
            )
            await console._apply_console_retrieval_scope_save(session, scope)
            await pilot.pause()
            row = console.query_one(f"#{ROW_ID}")
            assert list(row.query(f"#{CLEAR_BTN_ID}"))

            await pilot.click(f"#{CLEAR_BTN_ID}")
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.02)

            assert read_conversation_scope(db, conversation_id) is None
            row = console.query_one(f"#{ROW_ID}")
            assert (
                _static_plain_text(row.query_one(f"#{LABEL_ID}", Static))
                == "Scope: everything"
            )
    finally:
        db.close_connection()
