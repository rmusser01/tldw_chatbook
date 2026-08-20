"""Library media Trash view (task-4025): widget, handlers, restore seam.

The Trash view is the third ``_library_media_view`` value of the Browse ▸
Media canvas. These tests mirror ``test_library_multiselect_media.py``'s
structure: pilot tests for the canvas widget's rendered DOM, plain handler
tests over a SimpleNamespace fake, and real-DB tests (file-backed
``MediaDatabase``, never ``:memory:`` -- the restore worker hops threads via
``isolate_in_worker=True``) for the restore path, including a chunked item.
"""

import types
from types import SimpleNamespace

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Static

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media import LocalMediaReadingService, MediaReadingScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.library_media_state import (
    LIBRARY_MEDIA_TRASH_EMPTY_COPY,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP,
    LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP,
    MediaBrowseScope,
    build_library_media_trash_state,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
from tldw_chatbook.Widgets.Library.library_media_trash_canvas import (
    LibraryMediaTrashCanvas,
)


# ---------------------------------------------------------------------------
# Canvas widget pilot tests
# ---------------------------------------------------------------------------


def _trash_state(**kwargs):
    records = kwargs.pop(
        "records",
        [
            {
                "id": "11",
                "title": "First trashed",
                "type": "pdf",
                "trash_date": "2026-08-11T11:00:00+00:00",
            },
            {"id": "12", "title": "Second trashed", "type": "video"},
        ],
    )
    kwargs.setdefault("total", len(records or ()))
    return build_library_media_trash_state(records, **kwargs)


class _TrashCanvasApp(ConsolidatedCSSApp):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def compose(self):
        yield LibraryMediaTrashCanvas(
            canvas=self._state, id="library-media-trash-canvas"
        )


@pytest.mark.asyncio
async def test_trash_canvas_renders_heading_rows_and_enabled_restore():
    app = _TrashCanvasApp(_trash_state())
    async with app.run_test() as pilot:
        await pilot.pause()
        title = app.query_one("#library-media-trash-title", Static)
        assert "Trash (2)" in str(title.render())
        assert app.query_one("#library-media-trash-back", Button)
        rows = list(app.query(".library-media-trash-row"))
        assert len(rows) == 2
        # Selected-row grammar: leading "▸ " on the auto-selected first row.
        assert str(rows[0].label).startswith("▸ First trashed")
        assert str(rows[1].label).startswith("  Second trashed")
        restore = app.query_one("#library-media-trash-restore", Button)
        assert restore.disabled is False
        assert str(restore.label) == "Restore"
        assert restore.tooltip == LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP


@pytest.mark.asyncio
async def test_trash_canvas_empty_state_disables_restore_with_reason():
    app = _TrashCanvasApp(_trash_state(records=[], total=0))
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#library-media-trash-status", Static)
        assert LIBRARY_MEDIA_TRASH_EMPTY_COPY in str(status.render())
        restore = app.query_one("#library-media-trash-restore", Button)
        assert restore.disabled is True
        # F-018 + the non-colour disabled marker.
        assert str(restore.label) == "○ Restore"
        assert restore.tooltip == LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP


@pytest.mark.asyncio
async def test_trash_canvas_loading_state_never_claims_empty():
    app = _TrashCanvasApp(_trash_state(records=None, total=0))
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#library-media-trash-status", Static)
        rendered = str(status.render())
        assert "Loading Trash…" in rendered
        assert LIBRARY_MEDIA_TRASH_EMPTY_COPY not in rendered
        restore = app.query_one("#library-media-trash-restore", Button)
        assert restore.disabled is True
        assert (
            restore.tooltip == LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP
        )


@pytest.mark.asyncio
async def test_trash_canvas_truncation_status_and_notice_render():
    records = [
        {"id": str(i), "title": f"T{i}", "type": "pdf"} for i in range(3)
    ]
    app = _TrashCanvasApp(
        _trash_state(records=records, total=9, notice="Restored 'A doc'.")
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#library-media-trash-status", Static)
        assert "showing 3 of 9" in str(status.render())
        notice = app.query_one("#library-media-trash-notice", Static)
        assert "Restored 'A doc'." in str(notice.render())
        title = app.query_one("#library-media-trash-title", Static)
        assert "Trash (9)" in str(title.render())


@pytest.mark.asyncio
async def test_trash_canvas_error_state_shows_error_not_empty_copy():
    app = _TrashCanvasApp(
        _trash_state(records=[], total=0, error="Could not load Trash.")
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#library-media-trash-status", Static)
        rendered = str(status.render())
        assert "Could not load Trash." in rendered
        assert LIBRARY_MEDIA_TRASH_EMPTY_COPY not in rendered


@pytest.mark.asyncio
async def test_trash_canvas_error_state_restore_tooltip_names_the_failure():
    """PR-1505 review: with a failed fetch (``canvas.error`` set) the
    disabled Restore's tooltip must state the TRUE blocker -- the load
    failure -- never the empty-trash copy (F-018 accuracy: an unreadable
    Trash is not an empty Trash)."""
    app = _TrashCanvasApp(
        _trash_state(records=[], total=0, error="Could not load Trash.")
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        restore = app.query_one("#library-media-trash-restore", Button)
        assert restore.disabled is True
        assert str(restore.label) == "○ Restore"
        assert restore.tooltip != LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP
        assert restore.tooltip == LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_ERROR_TOOLTIP


# ---------------------------------------------------------------------------
# Fold reachability (PR-1505 review): with a full page of trashed items the
# list must scroll -- the repo's L3a clipping lesson (LibraryExportCanvas's
# docstring: a plain Vertical canvas clips content past the fold) applies
# here too, and the whole-branch live smoke only ever had ONE trash row, so
# this was never exercised. Measured in the REAL LibraryHarness (real app
# stylesheet, real #library-canvas host) -- a canvas mounted alone in a bare
# App is not measured against the tier that wins live (task-14900's lesson).
# ---------------------------------------------------------------------------


def _forty_trash_items():
    return [
        {
            "id": f"trash-{index}",
            "title": f"Trashed item {index:02d}",
            "type": "document",
            "trash_date": "2026-08-10T12:00:00+00:00",
        }
        for index in range(1, 41)
    ]


def _compositor_text(svg: str) -> str:
    """Rejoin an exported-screenshot SVG's `<text>` nodes into plain text.

    The established compositor-honest idiom (see
    ``test_console_fleet_discoverability._compositor_text``): scroll-clipped
    or off-screen content never becomes a `<text>` node at all, unlike a
    widget's ``.renderable``/``region``, which exist regardless of paint.
    """
    import re
    from html import unescape

    joined = "".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    return unescape(joined).replace("\xa0", " ")


async def _open_media_trash_with_items(host, pilot, trash_items):
    """Drive the real screen into the Trash view over a seeded trash page."""
    from Tests.UI.test_library_shell import (
        _active_library_screen,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)

    async def _list_media_trash(**kwargs):
        return {
            "items": [dict(item) for item in trash_items],
            "pagination": {"total_items": len(trash_items)},
        }

    # The shared StaticLibraryMediaScopeService fake predates the trash
    # seam; give this instance the same async surface the real scope
    # service exposes.
    host.app_instance.media_reading_scope_service.list_media_trash = (
        _list_media_trash
    )

    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-trash-open")
    screen.query_one("#library-media-trash-open").press()
    await _wait_for_selector(
        screen, pilot, f"#library-media-trash-row-{len(trash_items) - 1}"
    )
    await pilot.pause()
    return screen


async def _assert_trash_rows_and_restore_reachable(host, pilot):
    screen = await _open_media_trash_with_items(
        host, pilot, _forty_trash_items()
    )
    trash_list = screen.query_one("#library-media-trash-list")
    restore = screen.query_one("#library-media-trash-restore", Button)
    last_row = screen.query_one("#library-media-trash-row-39", Button)
    screen_height = host.size.height

    # The Restore toolbar stays inside the terminal: 40 auto-height rows
    # must not push it below the fold.
    assert restore.region.height > 0
    assert restore.region.y + restore.region.height <= screen_height

    # The list owns the vertical scroll -- its content overflows the pane
    # and the overflow is reachable rather than clipped.
    assert trash_list.max_scroll_y > 0

    # Scroll reachability: after scroll_end the LAST row is actually
    # painted (compositor-honest -- clipped content never paints).
    trash_list.scroll_end(animate=False)
    await pilot.pause()
    await pilot.pause()
    painted = _compositor_text(host.export_screenshot(simplify=True))
    assert "Trashed item 40" in painted

    # Keyboard reachability: from the top, focusing the last row
    # auto-scrolls it into view (Screen.set_focus -> scroll_visible).
    trash_list.scroll_home(animate=False)
    await pilot.pause()
    painted_top = _compositor_text(host.export_screenshot(simplify=True))
    assert "Trashed item 40" not in painted_top
    last_row.focus()
    await pilot.pause()
    await pilot.pause()
    painted = _compositor_text(host.export_screenshot(simplify=True))
    assert "Trashed item 40" in painted


@pytest.mark.asyncio
async def test_trash_page_past_fold_reachable_wide_split_layout():
    """Wide (>= the compact breakpoint) on a 24-row terminal: the 40-row
    trash page scrolls; the last row and Restore both stay reachable."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _seed_conversations,
        _two_conversations,
        _two_media_items,
    )

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=(170, 24)) as pilot:
        await _assert_trash_rows_and_restore_reachable(host, pilot)


@pytest.mark.asyncio
async def test_trash_page_past_fold_reachable_stacked_layout():
    """Below the breakpoint (the ``library-notes-compact`` stacked regime),
    same 24-row terminal: identical reachability contract."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _seed_conversations,
        _two_conversations,
        _two_media_items,
    )

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)
    async with host.run_test(size=(100, 24)) as pilot:
        await _assert_trash_rows_and_restore_reachable(host, pilot)


# ---------------------------------------------------------------------------
# The media list toolbar's "Trash" entry point (AC#1 reachability)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_list_toolbar_offers_trash_outside_select_mode():
    """AC#1: the Trash surface is reachable from the Media canvas -- the
    list toolbar carries a "Trash" action (hidden in select mode like
    "Export…"; Select's toolbar acts on the selection, never navigates)."""
    import dataclasses as _dataclasses

    from tldw_chatbook.Widgets.Library.library_media_canvas import (
        LibraryMediaCanvas,
    )
    from tldw_chatbook.Library.library_media_state import build_library_media_state

    list_state = build_library_media_state(
        [{"id": "1", "title": "One", "type": "pdf"}]
    )

    class _ListApp(ConsolidatedCSSApp):
        def __init__(self, state):
            super().__init__()
            self._state = state

        def compose(self):
            yield LibraryMediaCanvas(canvas=self._state, id="library-media-canvas")

    app = _ListApp(list_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        trash_btn = app.query_one("#library-media-trash-open", Button)
        assert trash_btn.display is True
        assert str(trash_btn.label) == "Trash"

    select_state = _dataclasses.replace(list_state, select_mode=True)
    app = _ListApp(select_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        trash_btn = app.query_one("#library-media-trash-open", Button)
        assert trash_btn.display is False


@pytest.mark.asyncio
async def test_confirm_copies_and_receipt_point_at_trash():
    """AC#3 (ADR-055 Pattern A): both delete confirmations and the receipt
    name the durable recovery path -- Trash -- instead of the old
    "there's no Trash view" copy."""
    import dataclasses as _dataclasses

    from tldw_chatbook.Widgets.Library.library_media_canvas import (
        LibraryMediaCanvas,
    )
    from tldw_chatbook.Library.library_media_state import build_library_media_state
    from tldw_chatbook.Library.library_media_viewer_state import (
        build_library_media_viewer_state,
    )
    from tldw_chatbook.Widgets.Library.library_media_viewer import (
        LibraryMediaViewer,
    )

    confirming_state = _dataclasses.replace(
        build_library_media_state(
            [{"id": "1", "title": "One", "type": "pdf"}],
            select_mode=True,
            selected_ids=frozenset({"1"}),
        ),
        confirming_bulk_delete=True,
    )
    receipt_state = _dataclasses.replace(
        build_library_media_state([{"id": "1", "title": "One", "type": "pdf"}]),
        delete_receipt_count=2,
    )

    class _ConfirmApp(ConsolidatedCSSApp):
        def __init__(self, state):
            super().__init__()
            self._state = state

        def compose(self):
            yield LibraryMediaCanvas(canvas=self._state, id="library-media-canvas")

    app = _ConfirmApp(confirming_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        copy = str(
            app.query_one(
                "#library-media-bulk-delete-confirm-copy", Static
            ).renderable
        )
        assert "restore later from Trash" in copy
        assert "no Trash view" not in copy

    app = _ConfirmApp(receipt_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        receipt = str(
            app.query_one(
                "#library-media-bulk-delete-receipt-copy", Static
            ).renderable
        )
        assert receipt == "✓ deleted · 2 items · in Trash"

    class _ViewerApp(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaViewer(
                build_library_media_viewer_state(
                    {"id": "1", "title": "One", "type": "pdf", "content": "x"}
                ),
                confirming_delete=True,
                id="library-media-viewer",
            )

    app = _ViewerApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        copy = str(
            app.query_one("#library-media-delete-confirm-copy", Static).renderable
        )
        assert "restore later from Trash" in copy
        assert "no Trash view" not in copy


# ---------------------------------------------------------------------------
# Handler tests (SimpleNamespace fakes, mirroring test_library_multiselect_media)
# ---------------------------------------------------------------------------


def _bind_trash_mutation_seams(fake):
    """Give direct restore fakes the production mutation boundary shape."""
    events = []
    scope = MediaBrowseScope()
    fake._mutation_events = events
    fake._library_media_browse_controller = SimpleNamespace(
        mutation_refresh_scope=scope,
        begin_mutation=lambda: events.append(("begin",)) or scope,
        reconcile_committed_mutation=lambda **kwargs: events.append(
            ("reconcile", kwargs)
        ),
        request=lambda requested, **kwargs: events.append(
            ("request", requested, kwargs)
        ),
        request_facets=lambda **kwargs: events.append(("facets", kwargs)),
    )
    fake._library_media_mutation_scope = None
    fake._library_media_mutation_authority = None
    fake._library_media_lifecycle_generation = 0
    fake._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    fake._library_media_type_choices_visible = False
    fake._sync_library_media_browse_state = lambda *_args: events.append(("sync",))
    fake._sync_library_media_viewer_mutation_gate = lambda: None
    fake._begin_library_media_mutation = types.MethodType(
        LibraryScreen._begin_library_media_mutation, fake
    )
    fake._library_media_backing_id = types.MethodType(
        LibraryScreen._library_media_backing_id, fake
    )
    fake._required_library_media_backing_id = types.MethodType(
        LibraryScreen._required_library_media_backing_id, fake
    )
    fake._library_media_mutation_summary = types.MethodType(
        LibraryScreen._library_media_mutation_summary, fake
    )
    fake._complete_library_media_mutation = types.MethodType(
        LibraryScreen._complete_library_media_mutation, fake
    )
    if fake._library_media_bulk_delete_in_flight:
        fake._begin_library_media_mutation()
    return fake


def _trash_view_fake(
    *,
    records=None,
    total=0,
    selected_id="",
    view="trash",
    in_flight=False,
):
    from tldw_chatbook.Library.row_selection import RowSelection

    notified = []
    refresh_calls = []
    worker_calls = []
    after_refresh = []
    fake = SimpleNamespace(
        _library_media_select_mode=False,
        _library_media_row_selection=RowSelection("media"),
        _library_media_confirming_bulk_delete=False,
        _library_media_bulk_delete_in_flight=in_flight,
        _library_media_delete_receipt_ids=("stale-receipt-id",),
        _library_media_view=view,
        _library_media_trash_records=records,
        _library_media_trash_total=total,
        _library_media_trash_error="",
        _library_media_trash_notice="",
        _selected_media_trash_id=selected_id,
        app_instance=SimpleNamespace(
            notify=lambda msg, **k: notified.append((msg, k))
        ),
        _notified=notified,
        _refresh_calls=refresh_calls,
        _worker_calls=worker_calls,
        _after_refresh=after_refresh,
        is_mounted=True,
        refresh=lambda **k: refresh_calls.append(k),
        run_worker=lambda coro, **k: worker_calls.append((coro, k)),
        call_after_refresh=lambda fn, *a: after_refresh.append(fn),
        _focus_library_media_trash_entry=lambda: None,
        _arm_library_list_entry_focus=lambda: None,
    )
    fake._exit_library_media_select_mode = types.MethodType(
        LibraryScreen._exit_library_media_select_mode, fake
    )
    fake._notify_library_media_selection_discarded = types.MethodType(
        LibraryScreen._notify_library_media_selection_discarded, fake
    )
    fake._build_library_media_trash_state = types.MethodType(
        LibraryScreen._build_library_media_trash_state, fake
    )
    fake._exit_library_media_trash = types.MethodType(
        LibraryScreen._exit_library_media_trash, fake
    )
    return _bind_trash_mutation_seams(fake)


def test_trash_open_enters_view_resets_state_and_kicks_fetch():
    """Entering Trash clears the stale receipt (the Trash view IS the
    durable path the receipt points at), resets the fetch/session state,
    recomposes, and schedules exactly one fetch worker."""
    fake = _trash_view_fake(view="list", records=("stale",), total=7)
    fake._library_media_trash_notice = "stale notice"

    async def _load():
        return None

    fake._load_library_media_trash = _load
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_open(fake, event)

    assert fake._library_media_view == "trash"
    assert fake._library_media_delete_receipt_ids == ()
    assert fake._library_media_trash_records is None
    assert fake._library_media_trash_total == 0
    assert fake._library_media_trash_notice == ""
    assert fake._refresh_calls == [{"recompose": True}]
    assert len(fake._worker_calls) == 1
    _coro, kwargs = fake._worker_calls[0]
    assert kwargs.get("exclusive") is True
    assert kwargs.get("group") == "library_media_trash_load"
    _coro.close()


def test_trash_row_press_updates_selection():
    fake = _trash_view_fake(
        records=(
            {"id": "1", "title": "A", "type": "pdf"},
            {"id": "2", "title": "B", "type": "pdf"},
        ),
        total=2,
    )
    event = SimpleNamespace(
        button=SimpleNamespace(media_id="2"), stop=lambda: None
    )

    LibraryScreen.handle_library_media_trash_row(fake, event)

    assert fake._selected_media_trash_id == "2"
    # ``_sync_library_canvas`` falls back to a full recompose on a fake
    # without a mounted canvas -- either way the view re-renders.
    assert fake._refresh_calls == [{"recompose": True}]


def test_trash_back_returns_to_list_and_drops_session_state():
    fake = _trash_view_fake(
        records=({"id": "1", "title": "A", "type": "pdf"},),
        total=1,
        selected_id="1",
    )
    fake._library_media_trash_notice = "Restored 'A'."
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_back(fake, event)

    assert fake._library_media_view == "list"
    assert fake._library_media_trash_records is None
    assert fake._library_media_trash_total == 0
    assert fake._library_media_trash_notice == ""
    assert fake._selected_media_trash_id == ""
    assert fake._refresh_calls == [{"recompose": True}]


def test_trash_restore_reads_resolved_selection_and_claims_shared_flag():
    """The restore worker joins the ONE bulk-delete interlock (PR-1473's
    one-flag rule): same flag, same exclusive worker group -- it is the
    fourth mutator of the shared records/counts state."""
    fake = _trash_view_fake(
        records=(
            {"id": "5", "title": "A", "type": "pdf"},
            {"id": "6", "title": "B", "type": "pdf"},
        ),
        total=2,
    )
    restore_calls = []

    async def _restore(media_id):
        restore_calls.append(media_id)

    fake._restore_library_media_from_trash = _restore
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_restore(fake, event)

    assert fake._library_media_bulk_delete_in_flight is True
    assert len(fake._worker_calls) == 1
    coro, kwargs = fake._worker_calls[0]
    assert kwargs.get("exclusive") is True
    assert kwargs.get("group") == "library_media_bulk_delete"
    import asyncio

    asyncio.run(coro)
    # No explicit selection -> the state builder's first-row fallback,
    # exactly what the "▸" marker shows.
    assert restore_calls == ["5"]


def test_trash_restore_refused_while_delete_or_undo_in_flight():
    fake = _trash_view_fake(
        records=({"id": "5", "title": "A", "type": "pdf"},),
        total=1,
        in_flight=True,
    )
    fake.run_worker = lambda coro, **k: pytest.fail("must not start a worker")
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_restore(fake, event)

    assert fake._library_media_bulk_delete_in_flight is True


def test_trash_restore_noop_when_trash_empty():
    fake = _trash_view_fake(records=(), total=0)
    fake.run_worker = lambda coro, **k: pytest.fail("must not start a worker")
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_trash_restore(fake, event)

    assert fake._library_media_bulk_delete_in_flight is False


def test_escape_gate_only_passes_in_trash_view():
    """check_action: ``library_media_trash_back`` fires only while the
    media canvas genuinely shows its Trash view; the viewer's own gate is
    False there (disjoint-gate contract of the escape chain)."""
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_MEDIA,
    )

    in_trash = SimpleNamespace(
        _library_selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_view="trash",
        _library_media_bulk_delete_in_flight=False,
    )
    assert (
        LibraryScreen.check_action(in_trash, "library_media_trash_back", ())
        is True
    )
    assert (
        LibraryScreen.check_action(in_trash, "library_media_viewer_back", ())
        is False
    )

    in_list = SimpleNamespace(
        _library_selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_view="list",
    )
    assert (
        LibraryScreen.check_action(in_list, "library_media_trash_back", ())
        is False
    )

    other_row = SimpleNamespace(
        _library_selected_row_id="browse-notes",
        _library_media_view="trash",
    )
    assert (
        LibraryScreen.check_action(other_row, "library_media_trash_back", ())
        is False
    )


# ---------------------------------------------------------------------------
# The restore seam against a REAL (file-backed) MediaDatabase -- never
# :memory: (the worker hops threads via ``isolate_in_worker=True``; a fresh
# :memory: connection on a new thread is a distinct, empty database).
# ---------------------------------------------------------------------------


def _restore_fake(*, db, trash_records, media_records, media_count):
    local_service = LocalMediaReadingService(db)
    scope_service = MediaReadingScopeService(local_service, None)
    notified = []
    refresh_calls = []
    after_refresh = []
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            media_reading_scope_service=scope_service,
            notify=lambda msg, **k: notified.append((msg, k)),
        ),
        _notified=notified,
        _refresh_calls=refresh_calls,
        _after_refresh=after_refresh,
        _local_source_records={"media": tuple(media_records)},
        _local_source_counts={"media": media_count},
        _library_media_trash_records=tuple(trash_records),
        _library_media_trash_total=len(trash_records),
        _library_media_trash_error="",
        _library_media_trash_notice="",
        _selected_media_trash_id="",
        # The real caller (``handle_library_media_trash_restore``) claims
        # the shared flag BEFORE scheduling this coroutine.
        _library_media_bulk_delete_in_flight=True,
        is_mounted=True,
        refresh=lambda **k: refresh_calls.append(k),
        call_after_refresh=lambda fn, *a: after_refresh.append(fn),
        _focus_library_media_trash_entry=lambda: None,
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _source_record_id=LibraryScreen._source_record_id,
    )
    fake._notify_library_media_delete_warning = types.MethodType(
        LibraryScreen._notify_library_media_delete_warning, fake
    )
    return _bind_trash_mutation_seams(fake)


@pytest.mark.asyncio
async def test_restore_via_real_db_moves_item_back_and_updates_counts(tmp_path):
    """AC#2: restore flips ``is_trash`` back through the existing seam
    (``restore_media_item`` -> ``restore_from_trash``, never raw SQL), the
    trash list loses the row, the media records/rail count gain it back in
    place, a notice (never a receipt) names it, and the shared interlock
    clears in the ``finally``."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4025-restore"
    )
    keep_id, _, _ = db.add_media_with_keywords(
        title="Keep", content="keep", media_type="article", keywords=[]
    )
    trashed_id, _, _ = db.add_media_with_keywords(
        title="Trashed Doc", content="body", media_type="document", keywords=[]
    )
    assert db.mark_as_trash(trashed_id) is True

    fake = _restore_fake(
        db=db,
        trash_records=(
            {"id": str(trashed_id), "title": "Trashed Doc", "type": "document"},
        ),
        media_records=({"id": str(keep_id), "title": "Keep"},),
        media_count=1,
    )

    await LibraryScreen._restore_library_media_from_trash(fake, str(trashed_id))

    row = db.get_media_by_id(trashed_id)
    assert row is not None
    assert not row["is_trash"]

    assert fake._library_media_trash_records == ()
    assert fake._library_media_trash_total == 0
    restored_ids = {
        LibraryScreen._source_record_id(r)
        for r in fake._local_source_records["media"]
    }
    assert str(trashed_id) in restored_ids
    assert fake._local_source_counts["media"] == 2
    assert fake._library_media_trash_notice == "Restored 'Trashed Doc'."
    assert fake._notified == []
    assert fake._refresh_calls == [{"recompose": True}]
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._mutation_events[0] == ("begin",)
    assert any(event[0] == "reconcile" for event in fake._mutation_events)
    assert any(event[0] == "request" for event in fake._mutation_events)
    assert any(event[0] == "facets" for event in fake._mutation_events)

    db.close_connection()


@pytest.mark.asyncio
async def test_restore_via_real_db_chunked_item_keeps_chunks_and_url(tmp_path):
    """Chunked variant (the batch's real-DB requirement) + the
    url-canonicalization decision pin: Trash-surface restore is an
    ``is_trash`` flag flip on the existing row -- it must leave the stored
    chunks AND the row's url byte-identical (task-4026's one-directional
    url-canonicalization edge lives only in ``add_media_with_keywords``'s
    restore-by-re-import path, which this surface never takes)."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4025-restore-chunked"
    )
    url = "https://example.com/task-4025/chunked.txt"
    media_id, _, _ = db.add_media_with_keywords(
        title="Chunked Doc",
        content="chunked content",
        media_type="document",
        keywords=[],
        url=url,
        chunks=[
            {"text": "chunk one", "chunk_type": "text"},
            {"text": "chunk two", "chunk_type": "text"},
        ],
    )
    assert db.mark_as_trash(media_id) is True

    def _chunk_texts():
        return {
            r["chunk_text"]
            for r in db.execute_query(
                "SELECT chunk_text FROM UnvectorizedMediaChunks "
                "WHERE media_id = ? AND deleted = 0",
                (media_id,),
            ).fetchall()
        }

    assert _chunk_texts() == {"chunk one", "chunk two"}

    fake = _restore_fake(
        db=db,
        trash_records=(
            {"id": str(media_id), "title": "Chunked Doc", "type": "document"},
        ),
        media_records=(),
        media_count=0,
    )

    await LibraryScreen._restore_library_media_from_trash(fake, str(media_id))

    row = db.get_media_by_id(media_id)
    assert row is not None
    assert not row["is_trash"]
    assert row["url"] == url
    assert row["content"] == "chunked content"
    assert _chunk_texts() == {"chunk one", "chunk two"}
    assert fake._local_source_counts["media"] == 1

    db.close_connection()


@pytest.mark.asyncio
async def test_restore_failure_warns_keeps_row_and_clears_flag(tmp_path):
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4025-restore-fail"
    )
    trash_records = ({"id": "999999", "title": "Ghost", "type": "document"},)
    fake = _restore_fake(
        db=db,
        trash_records=trash_records,
        media_records=(),
        media_count=0,
    )

    await LibraryScreen._restore_library_media_from_trash(fake, "999999")

    assert fake._library_media_trash_records == trash_records
    assert fake._local_source_records["media"] == ()
    assert fake._local_source_counts["media"] == 0
    assert fake._library_media_trash_notice == ""
    assert len(fake._notified) == 1
    assert "Could not restore" in fake._notified[0][0]
    assert fake._library_media_bulk_delete_in_flight is False

    db.close_connection()


@pytest.mark.asyncio
async def test_load_trash_fetches_page_through_real_seam(tmp_path):
    """AC#1: the fetch worker lists every ``is_trash=1`` item through the
    existing ``list_media_trash`` seam and maps items + total into the
    view's session state (trash_date included, for the "trashed <age>"
    secondary)."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4025-load"
    )
    active_id, _, _ = db.add_media_with_keywords(
        title="Active", content="a", media_type="article", keywords=[]
    )
    trashed_a, _, _ = db.add_media_with_keywords(
        title="Trash A", content="b", media_type="document", keywords=[]
    )
    trashed_b, _, _ = db.add_media_with_keywords(
        title="Trash B", content="c", media_type="video", keywords=[]
    )
    assert db.mark_as_trash(trashed_a) is True
    assert db.mark_as_trash(trashed_b) is True

    fake = _restore_fake(
        db=db, trash_records=(), media_records=(), media_count=1
    )
    fake._library_media_trash_records = None
    fake._library_media_view = "trash"
    # ``_load_library_media_trash``'s completion path calls the module's
    # ``_sync_library_canvas``, which falls back to ``refresh`` on a fake.
    await LibraryScreen._load_library_media_trash(fake)

    records = fake._library_media_trash_records
    assert records is not None
    listed_ids = {LibraryScreen._source_record_id(r) for r in records}
    assert listed_ids == {str(trashed_a), str(trashed_b)}
    assert str(active_id) not in listed_ids
    assert fake._library_media_trash_total == 2
    assert all(r.get("trash_date") for r in records)
    assert fake._library_media_trash_error == ""

    db.close_connection()


# ---------------------------------------------------------------------------
# The FTS decision (AC#4's explicit question), pinned against a real DB:
# trashed items are EXCLUDED from search results. The FTS5 index itself
# retains the trashed row's content (``mark_as_trash`` documents "does not
# affect FTS" -- an index-level property that makes restore instant), but
# every reachable query path filters ``is_trash = 0`` at query time:
# ``search_media_db`` (the Library keyword-search seam AND the RAG
# ``search_media_fts5`` leg both route through it with the default
# ``include_trash=False``) and ``rag_service``'s direct media_fts query
# (hard-coded ``m.is_trash = 0``). These tests pin both directions so the
# exclusion can never silently regress.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trashed_item_excluded_from_search_until_restored(tmp_path):
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4025-fts"
    )
    active_id, _, _ = db.add_media_with_keywords(
        title="Active zebra notes",
        content="the zebra grazes quietly",
        media_type="document",
        keywords=[],
    )
    trashed_id, _, _ = db.add_media_with_keywords(
        title="Trashed zebra notes",
        content="the zebra sleeps soundly",
        media_type="document",
        keywords=[],
    )
    assert db.mark_as_trash(trashed_id) is True
    service = LocalMediaReadingService(db)

    # The Library keyword-search seam (library_local_rag_search_service ->
    # scope service -> this): the trashed item must not surface.
    hits = service.search_media(query="zebra", limit=10)
    hit_ids = {int(item["id"]) for item in hits["items"]}
    assert hit_ids == {active_id}

    # The DB layer's own default, which the RAG fts5 leg also uses.
    rows, total = db.search_media_db(search_query="zebra")
    assert {int(r["id"]) for r in rows} == {active_id}
    assert total == 1

    # Symmetric: restore brings it straight back into search results (the
    # index never dropped it, so no re-index is needed).
    assert db.restore_from_trash(trashed_id) is True
    hits = service.search_media(query="zebra", limit=10)
    hit_ids = {int(item["id"]) for item in hits["items"]}
    assert hit_ids == {active_id, trashed_id}

    db.close_connection()
