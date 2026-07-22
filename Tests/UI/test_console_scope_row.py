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
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import (
    StaticConversationTreeService,
    _static_plain_text,
)
from Tests.UI.test_destination_shells import _wait_for_selector
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
from tldw_chatbook.Workspaces.registry_service import WorkspaceNotFound

SCOPE_CHIP_ID = "console-scope-chip"


class _AlwaysExistsMediaDB:
    """Minimal ``media_db`` stub for the row/chip's dangling-drop existence
    check (task-13 wires real ``resolve_effective_scope`` -- with its own
    ``existing_ids`` dangling-drop step -- into the display layer;
    previously the row/chip never touched existence at all). Every
    candidate id "exists" -- these tests are about row/chip rendering
    given an already-resolved scope, not about dangling-drop correctness
    itself (covered thoroughly by ``Tests/RAG/
    test_scope_pipeline_enforcement.py``), so a permissive stub keeps them
    focused on their own concern rather than requiring a fully seeded
    ``MediaDatabase`` in every test.
    """

    def execute_query(self, query, params):
        import json as _json

        (ids_json,) = params
        ids = _json.loads(ids_json)
        return _AlwaysExistsRows(ids)


class _AlwaysExistsRows:
    def __init__(self, ids):
        self._ids = ids

    def fetchall(self):
        return [(i,) for i in self._ids]


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
    app.media_db = _AlwaysExistsMediaDB()
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
        note_id = db.add_note(title="N1", content="body")
        app = _build_test_app()
        app.chachanotes_db = db
        app.media_db = _AlwaysExistsMediaDB()
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            row = await _open_inspector_and_get_row(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            assert session.persisted_conversation_id is None
            scope = RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", note_id)),
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
        note_id = db.add_note(title="N1", content="body")
        app = _build_test_app()
        app.chachanotes_db = db
        app.media_db = _AlwaysExistsMediaDB()
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            await _open_inspector_and_get_row(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            conversation_id = store.persist_session_if_needed(session.id)
            assert conversation_id is not None
            scope = RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", note_id)),
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
        app.media_db = _AlwaysExistsMediaDB()
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
    effective-scope cache (``_resolve_console_effective_scope_state``) for
    the resumed conversation id, but nothing downstream refreshed the
    MOUNTED row/chip: its own ``_sync_native_console_chat_ui()`` call ->
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
        note_id = db.add_note(title="N1", content="body")
        app = _build_test_app()
        app.chachanotes_db = db
        app.media_db = _AlwaysExistsMediaDB()
        conversation_id = db.add_conversation({"title": "Resumed scoped chat"})
        assert conversation_id
        scope = RagScope(
            items=(ScopeItem("media", "m1"), ScopeItem("note", note_id)),
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
    app.media_db = _AlwaysExistsMediaDB()
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
        note_id = db.add_note(title="N1", content="body")
        app = _build_test_app()
        app.chachanotes_db = db
        app.media_db = _AlwaysExistsMediaDB()
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            await _open_console_inspector(console, pilot)
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            scope = RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", note_id)),
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


# --- task-13: workspace-level scope entry point + intersection ---------
#
# The "Scope" affordance beside the workspace row in the Session area opens
# the SAME `ConsoleScopePickerModal`, targeting the active workspace instead
# of the active conversation. Saving writes through `set_workspace_scope`;
# the conversation-target picker's `universe` narrows to the workspace's own
# items once one is set (D3); the Inspector row/header chip render the
# EFFECTIVE (conversation ∩ workspace) state, including the intersection-
# breakdown tooltip and the `no-workspace-overlap` EMPTY state.

WORKSPACE_SCOPE_BTN_ID = "console-workspace-rag-scope-open"


@pytest.mark.asyncio
async def test_workspace_rag_scope_button_disabled_when_registry_unavailable():
    """No real workspace context (no registry service at all) -> the
    affordance is gated off, per the task brief's Default-workspace
    decision: only a REAL registry row can be scoped."""
    app = _build_test_app()
    app.workspace_registry_service = None
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, f"#{WORKSPACE_SCOPE_BTN_ID}")

        button = console.query_one(f"#{WORKSPACE_SCOPE_BTN_ID}", Button)
        assert button.disabled is True


@pytest.mark.asyncio
async def test_workspace_rag_scope_button_enabled_for_real_default_workspace():
    """The built-in Default workspace HAS a real registry row
    (``DEFAULT_WORKSPACE_ID``) once ``ensure_default_workspace`` has run
    (real in normal Console operation) -- so it IS scopable, per the task
    brief's explicit Default-workspace decision."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, f"#{WORKSPACE_SCOPE_BTN_ID}")

        button = console.query_one(f"#{WORKSPACE_SCOPE_BTN_ID}", Button)
        assert button.disabled is False


@pytest.mark.asyncio
async def test_workspace_rag_scope_button_opens_modal_with_universe_none():
    """Workspace target: the modal offers the FULL library (``universe=
    None``), names the workspace in the title, and seeds nothing when no
    workspace scope exists yet."""
    app = _build_test_app()
    app.media_reading_scope_service = _SpyMediaReadingScopeService()
    app.notes_scope_service = _SpyNotesScopeService()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, f"#{WORKSPACE_SCOPE_BTN_ID}")
        registry = app.workspace_registry_service
        active = registry.get_active_workspace()
        assert active is not None

        await pilot.click(f"#{WORKSPACE_SCOPE_BTN_ID}")
        await pilot.pause()
        for _ in range(20):
            await pilot.pause(0.02)

        modals = [s for s in host.screen_stack if isinstance(s, ConsoleScopePickerModal)]
        assert len(modals) == 1
        modal = modals[0]
        assert modal._universe is None
        assert modal._target_label == f"workspace '{active.name}'"


@pytest.mark.asyncio
async def test_workspace_rag_scope_save_persists_via_registry():
    """Save writes through ``LocalWorkspaceRegistryService.set_workspace_
    scope`` for the active workspace's id."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, f"#{WORKSPACE_SCOPE_BTN_ID}")
        registry = app.workspace_registry_service
        active = registry.get_active_workspace()
        scope = RagScope(
            items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z"
        )

        await console._apply_console_workspace_scope_save(active.workspace_id, scope)
        await pilot.pause()

        assert registry.get_workspace_scope(active.workspace_id) == scope


@pytest.mark.asyncio
async def test_workspace_rag_scope_save_catches_workspace_not_found():
    """``WorkspaceNotFound`` (the workspace was archived/deleted between
    opening the picker and saving) is caught deliberately, per the task
    brief, with an honest notify -- never an unhandled crash."""
    app = _build_test_app()
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, f"#{WORKSPACE_SCOPE_BTN_ID}")
        registry = app.workspace_registry_service

        def _raise_not_found(workspace_id, scope):
            raise WorkspaceNotFound(workspace_id)

        registry.set_workspace_scope = _raise_not_found
        scope = RagScope(
            items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z"
        )

        await console._apply_console_workspace_scope_save("workspace-default", scope)
        await pilot.pause()

        assert notifications, "expected an honest notify on WorkspaceNotFound"
        message, kwargs = notifications[-1]
        assert "no longer exists" in message.lower()
        assert kwargs.get("severity") == "error"


@pytest.mark.asyncio
async def test_conversation_picker_universe_is_workspace_scope_when_set():
    """D3: the CONVERSATION-target picker restricts its universe to the
    active workspace's own items once the workspace has a scope."""
    app = _build_test_app()
    app.media_reading_scope_service = _SpyMediaReadingScopeService()
    app.notes_scope_service = _SpyNotesScopeService()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        registry = app.workspace_registry_service
        active = registry.get_active_workspace()
        registry.set_workspace_scope(
            active.workspace_id,
            RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", "n1")),
                updated_at="2026-01-01T00:00:00Z",
            ),
        )
        await _open_inspector_and_get_row(console, pilot)

        await pilot.click(f"#{NARROW_BTN_ID}")
        await pilot.pause()
        for _ in range(20):
            await pilot.pause(0.02)

        modals = [s for s in host.screen_stack if isinstance(s, ConsoleScopePickerModal)]
        assert len(modals) == 1
        modal = modals[0]
        assert modal._universe == frozenset({("media", "m1"), ("note", "n1")})


@pytest.mark.asyncio
async def test_chip_tooltip_shows_intersection_breakdown_when_both_levels_set():
    """Both a conversation scope AND the workspace scope active -> the
    chip's effective count is the intersection, and its tooltip shows the
    full breakdown ("conversation A ∩ workspace B → N")."""
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        note_id = db.add_note(title="N1", content="body")
        app = _build_test_app()
        app.chachanotes_db = db
        app.media_db = _AlwaysExistsMediaDB()
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            await _open_console_inspector(console, pilot)
            registry = app.workspace_registry_service
            active = registry.get_active_workspace()
            registry.set_workspace_scope(
                active.workspace_id,
                RagScope(
                    items=(ScopeItem("media", "m1"), ScopeItem("media", "m2")),
                    updated_at="2026-01-01T00:00:00Z",
                ),
            )
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            store.persist_session_if_needed(session.id)
            conv_scope = RagScope(
                items=(ScopeItem("media", "m1"), ScopeItem("note", note_id)),
                updated_at="2026-01-01T00:00:00Z",
            )

            await console._apply_console_retrieval_scope_save(session, conv_scope)
            await pilot.pause()

            chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
            assert chip.display is True
            # conversation {m1, note} ∩ workspace {m1, m2} -> {m1}.
            assert _static_plain_text(chip) == "Scope: 1"
            assert str(chip.tooltip) == "conversation 2 ∩ workspace 2 → 1"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_chip_tooltip_shows_workspace_only_breakdown():
    """A workspace scope with NO conversation scope -> single-level
    "workspace N items" tooltip (not the conversation-flavored default)."""
    app = _build_test_app()
    app.media_db = _AlwaysExistsMediaDB()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector(console, pilot)
        registry = app.workspace_registry_service
        active = registry.get_active_workspace()
        registry.set_workspace_scope(
            active.workspace_id,
            RagScope(items=(ScopeItem("media", "m1"),), updated_at="2026-01-01T00:00:00Z"),
        )
        session = console._active_native_console_session()

        await console._resolve_console_effective_scope_state(session)
        console._sync_console_retrieval_scope_row()
        await pilot.pause()

        chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
        assert chip.display is True
        assert _static_plain_text(chip) == "Scope: 1"
        assert str(chip.tooltip) == "workspace 1 items"


@pytest.mark.asyncio
async def test_no_workspace_overlap_renders_empty_state_on_row_and_chip():
    """Fork-into-scoped-workspace: a conversation scope disjoint from the
    workspace scope resolves EMPTY with cause "no-workspace-overlap" --
    diagnosed on both the Inspector row (tooltip) and the header chip
    (alert styling + tooltip), matching the deleted-items EMPTY case."""
    db = CharactersRAGDB(":memory:", "test-client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        app.media_db = _AlwaysExistsMediaDB()
        host = ConsoleHarness(app)
        async with host.run_test(size=(240, 64)) as pilot:
            console = host.screen_stack[-1]
            await _open_inspector_and_get_row(console, pilot)
            registry = app.workspace_registry_service
            active = registry.get_active_workspace()
            registry.set_workspace_scope(
                active.workspace_id,
                RagScope(
                    items=(ScopeItem("media", "m-ws"),), updated_at="2026-01-01T00:00:00Z"
                ),
            )
            store = console._ensure_console_chat_store()
            session = console._active_native_console_session()
            store.persist_session_if_needed(session.id)
            conv_scope = RagScope(
                items=(ScopeItem("media", "m-conv"),), updated_at="2026-01-01T00:00:00Z"
            )

            await console._apply_console_retrieval_scope_save(session, conv_scope)
            await pilot.pause()

            row = console.query_one(f"#{ROW_ID}")
            label = row.query_one(f"#{LABEL_ID}", Static)
            assert _static_plain_text(label) == "Scope: empty"
            assert "no-workspace-overlap" in str(label.tooltip)

            chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
            assert chip.display is True
            assert _static_plain_text(chip) == "Scope: empty"
            assert chip.has_class("console-chip-alert")
            assert "no-workspace-overlap" in str(chip.tooltip)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_switch_between_resumed_sessions_refreshes_stale_workspace_scope():
    """Task-13 review finding 2: ``_apply_console_workspace_scope_save``
    only refreshes the ACTIVE session's row/chip (its own docstring says
    so -- "only the currently active session's row/chip are mounted to
    refresh"). Switching to a DIFFERENT, already-resumed native session in
    the same workspace via ``_activate_native_console_session`` (the
    shared tab-click / Ctrl+K / Alt+1..9 activation path) must not keep
    serving that session's stale ``_console_effective_scope_cache`` entry
    -- it must reflect the workspace's CURRENT scope, even though the
    workspace scope changed while that session's own tab was inactive.

    Unlike ``_resume_console_workspace_conversation`` (which warms this
    cache itself, per ``test_resume_console_workspace_conversation_
    refreshes_row_and_chip`` above), ``_activate_native_console_session``
    previously never touched it at all -- a real display-staleness gap,
    not a made-up scenario.
    """
    app = _build_test_app()
    app.media_db = _AlwaysExistsMediaDB()
    host = ConsoleHarness(app)
    async with host.run_test(size=(240, 64)) as pilot:
        console = host.screen_stack[-1]
        await _open_inspector_and_get_row(console, pilot)
        registry = app.workspace_registry_service
        active = registry.get_active_workspace()
        store = console._ensure_console_chat_store()

        first_session = console._active_native_console_session()
        assert first_session is not None
        second_session = store.create_session(title="Second")
        assert second_session.workspace_id == first_session.workspace_id

        # An earlier resolve (resume, a prior scope-picker save, or an
        # earlier activation) cached `first_session`'s effective scope
        # against the workspace's OLD scope.
        registry.set_workspace_scope(
            active.workspace_id,
            RagScope(
                items=(ScopeItem("media", "m-old"),),
                updated_at="2026-01-01T00:00:00Z",
            ),
        )
        await console._resolve_console_effective_scope_state(first_session)
        stale = console._console_effective_scope_cache[first_session.id]
        assert stale.item_count == 1

        # The workspace scope is edited while `second_session` (not
        # `first_session`) is active -- `_apply_console_workspace_scope_
        # save` only refreshes the ACTIVE session, so this write alone
        # leaves `first_session`'s cache entry stale.
        registry.set_workspace_scope(
            active.workspace_id,
            RagScope(
                items=(
                    ScopeItem("media", "m-new-1"),
                    ScopeItem("media", "m-new-2"),
                ),
                updated_at="2026-01-02T00:00:00Z",
            ),
        )

        await console._activate_native_console_session(first_session.id)
        await pilot.pause()

        assert console._active_native_console_session().id == first_session.id
        row = console.query_one(f"#{ROW_ID}")
        assert (
            _static_plain_text(row.query_one(f"#{LABEL_ID}", Static))
            == "Scope: 2 items"
        ), "switching back to an already-resumed session must not keep the stale cached scope"
        chip = console.query_one(f"#{SCOPE_CHIP_ID}", ConsoleScopeChip)
        assert chip.display is True
        assert _static_plain_text(chip) == "Scope: 2"
