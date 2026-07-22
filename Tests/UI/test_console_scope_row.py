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

from Tests.UI.test_console_native_chat_flow import (
    StaticConversationTreeService,
    _static_plain_text,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_display_state import ConsoleRetrievalScopeState
from tldw_chatbook.Chat.rag_scope import (
    SCOPE_EMPTY_NOTICE_TEMPLATE,
    RagScope,
    ScopeItem,
    read_conversation_scope,
    write_conversation_scope,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_chatbook.Widgets.Console.console_control_bar import ConsoleScopeChip
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

SCOPE_CHIP_ID = "console-scope-chip"


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
    compose/recompose. Task-10: the header chip shares this exact
    zero-DB compose path (built from the same
    ``_build_console_retrieval_scope_state()`` call), so it must stay
    hidden through the forced recompose too."""

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
        assert console.query_one(f"#{SCOPE_CHIP_ID}").display is False

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
        assert console.query_one(f"#{SCOPE_CHIP_ID}").display is False


@pytest.mark.asyncio
async def test_retrieval_scope_row_empty_state_matches_chip_rendering():
    """task-10 review finding 1: row/chip renderer divergence on EMPTY.

    ``compose()`` branched only on ``is_scoped`` and never checked
    ``is_empty``/``cause`` -- the fields ``ConsoleRetrievalScopeState``
    gained for task-10's header chip. The chip's own
    ``_scope_chip_render`` DOES render EMPTY ("Scope: empty" + alert +
    cause tooltip via ``SCOPE_EMPTY_NOTICE_TEMPLATE``); since row and chip
    are documented as "one state, two renderers" of the exact same
    snapshot (``ConsoleRetrievalScopeState``'s own docstring), they must
    not diverge -- an EMPTY snapshot must not silently render as the
    "everything" default the way an ``is_scoped=False`` check alone would
    produce.

    EMPTY is not reachable from the real conversation-only
    ``_build_console_retrieval_scope_state`` path today (that builder is
    zero-DB by contract; detecting a fully-deleted scope needs a DB
    existence check) -- mirrors the chip's own EMPTY test
    (``test_scope_chip_empty_state_action_required_styling_and_cause_tooltip``)
    by driving the row directly with an EMPTY state via ``sync_state``.
    EMPTY becomes reachable once Phase 3 of the rag-scope-narrowing
    program wires conversation/workspace intersection into the display
    path.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        row = await _open_inspector_and_get_row(console, pilot)

        row.sync_state(ConsoleRetrievalScopeState.empty(cause="deleted-items"))
        await pilot.pause()

        row = console.query_one(f"#{ROW_ID}")
        label = row.query_one(f"#{LABEL_ID}", Static)
        assert _static_plain_text(label) == "Scope: empty"
        assert str(label.tooltip) == SCOPE_EMPTY_NOTICE_TEMPLATE.format(
            cause="deleted-items"
        )
        # Same action-available affordance as the plain-unscoped row (an
        # empty *evaluated* scope still lets the user narrow to something
        # that actually resolves, or clear it outright).
        assert list(row.query(f"#{NARROW_BTN_ID}"))
        assert not list(row.query(f"#{EDIT_BTN_ID}"))
        assert not list(row.query(f"#{CLEAR_BTN_ID}"))


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
async def test_scope_saved_unpersisted_then_first_send_flush_refreshes_row():
    """task-9 review finding 1: narrow-then-first-send must not leave the
    Inspector row stale.

    Saving a scope on an UNPERSISTED session holds it in
    ``session.rag_scope_holder``. The bug: when the first message send
    later persists the session (``ConsoleChatStore.append_message(...,
    persist=True)`` -> ``persist_session_if_needed``, which flushes the
    holder to the DB and empties it), nothing told
    ``ChatScreen._console_retrieval_scope_cache`` about the new
    conversation id -- the row then rendered "everything" even though the
    scope was persisted correctly. This drives the REAL message-send path
    (``store.append_message(..., persist=True)``, exactly what
    ``ConsoleChatController.submit_draft`` calls) with no modal-open/resume
    read trigger in between.
    """
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
            assert session.persisted_conversation_id is None
            scope = RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", "n1")),
                updated_at="2026-01-01T00:00:00Z",
            )

            # 1. Narrow the scope FIRST, while still unpersisted.
            await console._apply_console_retrieval_scope_save(session, scope)
            assert session.rag_scope_holder.scope == scope
            assert session.persisted_conversation_id is None

            # 2. Persist via the real message-send path (no modal-open, no
            # resume in between) -- mirrors
            # ``ConsoleChatController.submit_draft``'s
            # ``store.append_message(..., persist=True)`` call.
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="hello",
                persist=True,
            )
            await pilot.pause()

            assert session.persisted_conversation_id is not None
            conversation_id = session.persisted_conversation_id
            # The scope persisted correctly...
            assert read_conversation_scope(db, conversation_id) == scope
            # ...and the row must already reflect it (no modal-open/resume).
            state = console._build_console_retrieval_scope_state()
            assert state.is_scoped
            assert state.item_count == 2
            row = console.query_one(f"#{ROW_ID}")
            assert (
                _static_plain_text(row.query_one(f"#{LABEL_ID}", Static))
                == "Scope: 2 items"
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_save_persisted_writes_through_refreshes_row_and_run_recipe():
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            await _open_inspector_and_get_row(console, pilot)
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
            await _open_inspector_and_get_row(console, pilot)
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
async def test_save_persisted_conflict_error_surfaces_specific_notify():
    """task-9 review finding 4: a version-conflict write (``ConflictError``
    -- e.g. a concurrent dictionary-attach write racing the same
    conversation row, real per PR #734 docs) must not be lumped in with the
    generic "corrupted metadata" wording."""
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        notifications: list[tuple[str, dict]] = []
        app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            await _open_inspector_and_get_row(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            conversation_id = store.persist_session_if_needed(session.id)

            def _raise_conflict(*args, **kwargs):
                raise ConflictError(
                    "Version mismatch", entity="conversations", entity_id=conversation_id
                )

            db.update_conversation = _raise_conflict
            scope = RagScope(
                items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z"
            )

            await console._apply_console_retrieval_scope_save(session, scope)
            await pilot.pause()

            assert notifications, "expected an honest notify on write conflict"
            message, kwargs = notifications[-1]
            assert "concurrently" in message.lower()
            assert "corrupt" not in message.lower()
            assert kwargs.get("severity") == "error"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_save_persisted_conversation_not_found_surfaces_specific_notify():
    """task-9 review finding 4: a ``ValueError`` (conversation not found --
    e.g. deleted out from under the session) must not be lumped in with the
    generic "corrupted metadata" wording either."""
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        notifications: list[tuple[str, dict]] = []
        app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            await _open_inspector_and_get_row(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            conversation_id = store.persist_session_if_needed(session.id)
            db.soft_delete_conversation(conversation_id, expected_version=1)
            scope = RagScope(
                items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z"
            )

            await console._apply_console_retrieval_scope_save(session, scope)
            await pilot.pause()

            assert notifications, "expected an honest notify when the conversation is gone"
            message, kwargs = notifications[-1]
            assert "corrupt" not in message.lower()
            assert kwargs.get("severity") == "error"
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
            await _open_inspector_and_get_row(console, pilot)
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


@pytest.mark.asyncio
async def test_resume_console_workspace_conversation_refreshes_row_and_chip():
    """task-10 review finding 2: resuming a scoped saved conversation must
    show the scoped row/chip immediately, not "Scope: everything" until the
    user touches Edit/Narrow/save.

    ``_resume_console_workspace_conversation`` already warms the
    retrieval-scope cache (``_warm_console_retrieval_scope_cache``) for the
    resumed conversation id, but nothing downstream refreshed the MOUNTED
    row/chip: its own ``_sync_native_console_chat_ui()`` call ->
    ``_sync_console_control_bar()`` only pushes into
    ``#console-control-bar``/``#console-run-inspector-state`` -- never the
    sibling retrieval-scope row, and never
    ``ConsoleControlBar.sync_scope_chip`` (deliberately a separate method
    from the control bar's general ``sync_state``, per its own docstring).
    Drives the real production coroutine directly (the same pattern the
    save/clear tests above already use for
    ``_apply_console_retrieval_scope_save``), not a UI pixel-click, so this
    proves the actual resume seam rather than a rebuilt double.
    """
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        conversation_id = db.add_conversation({"title": "Resumed scoped chat"})
        assert conversation_id
        scope = RagScope(
            items=(ScopeItem("media", "m1"), ScopeItem("note", "n1")),
            updated_at="2026-01-01T00:00:00Z",
        )
        write_conversation_scope(db, conversation_id, scope)
        app.chat_conversation_scope_service = StaticConversationTreeService(
            {
                conversation_id: {
                    "conversation": {
                        "id": conversation_id,
                        "title": "Resumed scoped chat",
                        "workspace_id": None,
                    },
                    "root_threads": [],
                }
            }
        )
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            row = await _open_inspector_and_get_row(console, pilot)
            # Pre-condition: the currently active (not-yet-resumed) session
            # is unscoped -- matches the row/chip's default compose-time
            # state, so a pass here can't be a coincidental leftover.
            assert (
                _static_plain_text(row.query_one(f"#{LABEL_ID}", Static))
                == "Scope: everything"
            )
            assert console.query_one(f"#{SCOPE_CHIP_ID}").display is False

            resumed = await console._resume_console_workspace_conversation(
                conversation_id
            )
            await pilot.pause()

            assert resumed is True
            session = console._active_native_console_session()
            assert session is not None
            assert session.persisted_conversation_id == conversation_id

            row = console.query_one(f"#{ROW_ID}")
            assert (
                _static_plain_text(row.query_one(f"#{LABEL_ID}", Static))
                == "Scope: 2 items"
            )
            chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
            assert chip.display is True
            assert _static_plain_text(chip) == "Scope: 2"
    finally:
        db.close_connection()


# --- task-10: header "Scope" chip -------------------------------------
#
# The chip lives in the Console header's chip row (``#console-scope-chip``,
# a sibling of "Sources: 0 staged" / "RAG: off") and renders from the exact
# same ``ConsoleRetrievalScopeState`` snapshot as the Inspector row above --
# see ``ChatScreen._sync_console_retrieval_scope_row`` and
# ``ConsoleControlBar.sync_scope_chip``/``_scope_chip_render``.


@pytest.mark.asyncio
async def test_scope_chip_hidden_when_unscoped():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)

        chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
        assert chip.display is False


@pytest.mark.asyncio
async def test_scope_chip_shows_count_and_tooltip_when_scoped():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)

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

        chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
        assert chip.display is True
        assert _static_plain_text(chip) == "Scope: 2"
        # Phase-3 will widen this to the intersection breakdown
        # ("conversation A ∩ workspace B → N") once workspace-level
        # scoping resolves through the same seam; today it is
        # conversation-only.
        assert str(chip.tooltip) == "conversation 2 items"
        assert not chip.has_class("console-chip-alert")


@pytest.mark.asyncio
async def test_scope_chip_refreshes_on_modal_save():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
        session = console._active_native_console_session()
        assert session.persisted_conversation_id is None
        scope = RagScope(items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z")

        await console._apply_console_retrieval_scope_save(session, scope)
        await pilot.pause()

        chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
        assert chip.display is True
        assert _static_plain_text(chip) == "Scope: 1"


@pytest.mark.asyncio
async def test_scope_chip_refreshes_on_persist_transition_flush():
    """Mirrors ``test_scope_saved_unpersisted_then_first_send_flush_refreshes_row``:
    the real message-send path (``store.append_message(..., persist=True)``)
    must refresh the chip too, with no modal-open/resume read trigger in
    between."""
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            await _open_console_inspector(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            scope = RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", "n1")),
                updated_at="2026-01-01T00:00:00Z",
            )

            await console._apply_console_retrieval_scope_save(session, scope)
            assert console.query_one(f"#{SCOPE_CHIP_ID}").display is True

            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="hello",
                persist=True,
            )
            await pilot.pause()

            chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
            assert chip.display is True
            assert _static_plain_text(chip) == "Scope: 2"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_scope_chip_click_opens_picker_modal():
    """Same handler seam as the Inspector row's Edit button (task-9):
    activating the chip -- via ``_on_click`` or Enter/Space while focused,
    exactly like ``ConsoleApprovalsChip`` -- posts ``OpenRequested``, which
    ``ChatScreen`` routes straight into
    ``_open_console_retrieval_scope_picker``. Driven via focus + Enter
    here (the codebase's established activation pattern for these header
    chips, e.g. the approvals chip's own contract tests) rather than a
    pixel-coordinate ``pilot.click``, which is flaky against this
    viewport's clipped/overflowing chip row."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
        session = console._active_native_console_session()
        session.rag_scope_holder.set(
            RagScope(items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z")
        )
        console._sync_console_retrieval_scope_row()
        await pilot.pause()

        chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
        chip.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        for _ in range(20):
            await pilot.pause(0.02)

        modals = [s for s in host.screen_stack if isinstance(s, ConsoleScopePickerModal)]
        assert len(modals) == 1


@pytest.mark.asyncio
async def test_scope_chip_empty_state_action_required_styling_and_cause_tooltip():
    """EMPTY is not reachable from the real conversation-only path yet --
    ``_build_console_retrieval_scope_state`` is zero-DB by contract, and
    detecting a fully-deleted scope (or, once Phase 3 lands, a
    no-overlap conversation/workspace intersection) needs a DB existence
    check. This drives the renderer directly with an EMPTY state object
    (monkeypatching the builder), exactly as the brief calls for, ahead of
    that Phase-3 wiring."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)

        console._build_console_retrieval_scope_state = lambda: ConsoleRetrievalScopeState.empty(
            cause="deleted-items"
        )
        console._sync_console_retrieval_scope_row()
        await pilot.pause()

        chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
        assert chip.display is True
        assert _static_plain_text(chip) == "Scope: empty"
        assert chip.has_class("console-chip-alert")
        assert "deleted-items" in str(chip.tooltip)
