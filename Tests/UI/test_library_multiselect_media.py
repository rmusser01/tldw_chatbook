import asyncio
import dataclasses
import types
from types import SimpleNamespace

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media import LocalMediaReadingService, MediaReadingScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_media_state import (
    LibraryMediaCanvasState,
    LibraryMediaRow,
    MediaBrowseScope,
    build_library_media_state,
)
from tldw_chatbook.Library.library_pager_state import build_library_pager_display
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_INGEST_MEDIA,
)
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas


def _bind_media_mutation_seams(fake):
    """Give direct method fakes the production mutation boundary shape."""
    if not hasattr(fake, "_library_media_bulk_delete_in_flight"):
        fake._library_media_bulk_delete_in_flight = True
    events = []
    scope = MediaBrowseScope()
    controller = SimpleNamespace(
        applied_scope=scope,
        retained_items=(),
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
    fake._mutation_events = events
    fake._library_media_browse_controller = controller
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


def _media_fake(
    select_mode, *, confirming_bulk_delete=False, bulk_delete_in_flight=False
):
    notified = []
    fake = SimpleNamespace(
        _library_media_select_mode=select_mode,
        _library_media_row_selection=RowSelection("media"),
        _library_media_confirming_bulk_delete=confirming_bulk_delete,
        # task-3020 AC1: default False -- most tests exercise a single
        # press, never a double one, and the guard would otherwise reject
        # every scripted confirm press unconditionally.
        _library_media_bulk_delete_in_flight=bulk_delete_in_flight,
        _library_media_selection_notice="",
        app_instance=SimpleNamespace(
            notify=lambda msg, **k: notified.append((msg, k))
        ),
        _notified=notified,
        _opened=[],
        _refreshed=0,
        _viewer_opened=[],
        _footer_registrations=0,
    )
    # task-3020 AC2: arming/cancelling the bulk-delete confirmation now
    # explicitly re-registers the footer (the canvas-scoped sync these
    # handlers otherwise use deliberately skips the footer widget) -- a
    # bare counting stub is enough here; the actual shortcut-set content
    # is covered by the real-screen tests in test_screen_navigation.py.
    fake._register_footer_shortcuts = lambda: setattr(
        fake, "_footer_registrations", fake._footer_registrations + 1
    )
    # These are real LibraryScreen instance methods (not module-level
    # helpers like ``_apply_library_row_toggle``), so handlers that call
    # ``self._exit_library_media_select_mode(...)`` need them actually
    # bound to this fake -- ``types.MethodType`` reuses the REAL
    # implementation rather than a hand-rolled stub duplicating its logic.
    fake._exit_library_media_select_mode = types.MethodType(
        LibraryScreen._exit_library_media_select_mode, fake
    )
    fake._clear_library_media_selection_for_scope_change = types.MethodType(
        LibraryScreen._clear_library_media_selection_for_scope_change, fake
    )
    fake._notify_library_media_selection_discarded = types.MethodType(
        LibraryScreen._notify_library_media_selection_discarded, fake
    )
    fake._cancel_library_media_bulk_delete = types.MethodType(
        LibraryScreen._cancel_library_media_bulk_delete, fake
    )
    return _bind_media_mutation_seams(fake)


def test_row_press_in_select_mode_toggles_not_opens():
    fake = _media_fake(select_mode=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    fake._open_library_media_viewer = lambda mid: fake._viewer_opened.append(mid)
    event = SimpleNamespace(button=SimpleNamespace(media_id="7"), stop=lambda: None)
    LibraryScreen.handle_library_media_row(fake, event)
    assert fake._library_media_row_selection.is_selected("7")
    assert fake._viewer_opened == []  # viewer NOT opened
    assert fake._refreshed == 1


def test_row_press_normal_mode_opens_viewer():
    fake = _media_fake(select_mode=False)
    fake._open_library_media_viewer = lambda mid: fake._viewer_opened.append(mid)
    event = SimpleNamespace(button=SimpleNamespace(media_id="7"), stop=lambda: None)
    LibraryScreen.handle_library_media_row(fake, event)
    assert fake._viewer_opened == ["7"]
    assert not fake._library_media_row_selection.is_selected("7")


@pytest.mark.asyncio
async def test_export_selected_builds_ids_scope():
    fake = _media_fake(select_mode=True)
    fake._library_media_row_selection.select_all(["3", "1", "2"])

    async def _open(scope):
        fake._opened.append(scope)

    fake._open_library_export_canvas = _open
    event = SimpleNamespace(stop=lambda: None)
    await LibraryScreen.handle_library_media_export_selected(fake, event)
    assert fake._opened == [ExportScope(kind="media", ids=("1", "2", "3"))]


@pytest.mark.asyncio
async def test_export_selected_empty_is_noop_not_whole_source():
    # An empty selection must NOT fall through to a whole-source export
    # (empty ids == whole source in resolve_export_selections).
    fake = _media_fake(select_mode=True)

    async def _open(scope):
        fake._opened.append(scope)

    fake._open_library_export_canvas = _open
    event = SimpleNamespace(stop=lambda: None)
    await LibraryScreen.handle_library_media_export_selected(fake, event)
    assert fake._opened == []


def _select_mode_canvas_state() -> LibraryMediaCanvasState:
    rows = (
        LibraryMediaRow(
            media_id="1",
            title="First item",
            media_type="video",
            secondary="video · today",
            checked=False,
        ),
        LibraryMediaRow(
            media_id="2",
            title="Second item",
            media_type="audio",
            secondary="audio · today",
            checked=False,
        ),
    )
    return LibraryMediaCanvasState(
        rows=rows,
        type_options=("All", "audio", "video"),
        active_type="All",
        status_copy="",
        empty_copy="No media in your Library yet. Import something to see it here.",
        selected_id="",
        preview_lines=(),
        count=len(rows),
        select_mode=True,
        selected_count=0,
    )


class _MediaCanvasApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_select_mode_canvas_state(), id="library-media-canvas"
        )


@pytest.mark.asyncio
async def test_canvas_select_mode_renders_action_row_and_disables_export():
    app = _MediaCanvasApp()
    async with app.run_test() as pilot:
        select_all_btn = pilot.app.query_one("#library-media-select-all", Button)
        assert select_all_btn is not None
        export_selected_btn = pilot.app.query_one(
            "#library-media-export-selected", Button
        )
        assert export_selected_btn.disabled is True


def _empty_select_mode_state() -> LibraryMediaCanvasState:
    # Select mode is active but the list rendered zero rows (e.g. a background
    # refresh emptied it). "Done" must stay pressable so the user can exit.
    return LibraryMediaCanvasState(
        rows=(),
        type_options=("All",),
        active_type="All",
        status_copy="",
        empty_copy="No media in your Library yet.",
        selected_id="",
        preview_lines=(),
        count=0,
        select_mode=True,
        selected_count=0,
    )


class _EmptySelectModeCanvasApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_empty_select_mode_state(), id="library-media-canvas"
        )


class _FreshEmptyMediaCanvasApp(ConsolidatedCSSApp):
    def __init__(
        self,
        *,
        media_type: str | None,
        delete_receipt_count: int = 0,
        loading: bool = False,
        error_copy: str = "",
    ) -> None:
        super().__init__()
        self.media_type = media_type
        self.delete_receipt_count = delete_receipt_count
        self.loading = loading
        self.error_copy = error_copy

    def compose(self):
        copy = (
            f"No media of type '{self.media_type}'."
            if self.media_type is not None
            else "No media in your Library yet. Import something to see it here."
        )
        yield LibraryMediaCanvas(
            canvas=LibraryMediaCanvasState(
                rows=(),
                type_options=(None, "video"),
                active_type=self.media_type,
                status_copy="",
                empty_copy=copy,
                selected_id="",
                preview_lines=(),
                count=0,
                delete_receipt_count=self.delete_receipt_count,
            ),
            pager=build_library_pager_display(
                applied_page=1,
                requested_page=1,
                page_size=20,
                row_count=0,
                total=0,
                freshness="fresh",
                loading=self.loading,
                error_copy=self.error_copy,
            ),
            id="library-media-canvas",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media_type", "action_id", "label"),
    [
        (None, "library-media-empty-import", "Import media"),
        ("video", "library-media-empty-clear-type", "Show all types"),
    ],
    ids=["source-empty", "filtered-zero"],
)
async def test_media_fresh_zero_distills_to_one_recovery_action(
    media_type: str | None, action_id: str, label: str
):
    app = _FreshEmptyMediaCanvasApp(media_type=media_type)

    async with app.run_test() as pilot:
        action = pilot.app.query_one(f"#{action_id}", Button)
        assert str(action.label) == label
        assert action.disabled is False
        assert action in pilot.app.screen.focus_chain
        assert not pilot.app.query("#library-media-pager")
        assert not pilot.app.query("#library-media-select-toggle")
        assert not pilot.app.query("#library-media-export")
        assert not pilot.app.query("#library-media-detail-empty")
        assert len(pilot.app.query(".library-canvas-action")) == 1


@pytest.mark.asyncio
async def test_media_fresh_zero_keeps_committed_delete_recovery_authority():
    app = _FreshEmptyMediaCanvasApp(media_type=None, delete_receipt_count=1)

    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-media-bulk-delete-undo", Button)
        assert pilot.app.query_one("#library-media-trash-open", Button)
        assert pilot.app.query_one("#library-media-pager")
        assert not pilot.app.query("#library-media-empty-import")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loading", "error_copy", "retry_visible"),
    [
        (True, "", False),
        (False, "Filter wasn't applied; showing previous results.", True),
    ],
    ids=["loading", "error"],
)
async def test_media_retained_zero_keeps_request_recovery_authority(
    loading: bool, error_copy: str, retry_visible: bool
):
    app = _FreshEmptyMediaCanvasApp(
        media_type=None,
        loading=loading,
        error_copy=error_copy,
    )

    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-media-pager")
        assert bool(pilot.app.query("#library-media-retry")) is retry_visible
        assert not pilot.app.query("#library-media-empty-import")


@pytest.mark.asyncio
async def test_media_empty_import_uses_existing_library_destination():
    calls = []

    async def select_row(row_id: str) -> None:
        calls.append(row_id)

    fake = SimpleNamespace(
        _select_library_rail_row=select_row,
    )

    await LibraryScreen.handle_library_media_empty_import(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert calls == [LIBRARY_ROW_INGEST_MEDIA]


def test_media_empty_clear_type_requests_applied_scope_page_one():
    calls = []
    fake = SimpleNamespace(
        _library_media_bulk_delete_in_flight=False,
        _request_library_media_type=lambda media_type, **kwargs: calls.append(
            (media_type, kwargs)
        ),
    )

    LibraryScreen.handle_library_media_empty_clear_type(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert calls == [(None, {"focus_identity": "#library-media-empty-clear-type"})]


@pytest.mark.asyncio
async def test_done_toggle_stays_enabled_at_zero_rows_in_select_mode():
    # Regression: the Select/Done toggle must NOT be disabled at 0 rendered rows
    # while select mode is active, or the user is stuck with no way to exit.
    app = _EmptySelectModeCanvasApp()
    async with app.run_test() as pilot:
        toggle = pilot.app.query_one("#library-media-select-toggle", Button)
        assert toggle.disabled is False
        assert str(toggle.label) == "Done"


def _filtered_select_mode_state() -> LibraryMediaCanvasState:
    # Two media types; filtering to "video" renders ONE row while
    # ``count`` (the pre-filter total across all types) stays at 3.
    records = [
        {
            "media_id": "1",
            "title": "A video",
            "type": "video",
            "last_modified": "2026-07-10T00:00:00Z",
        },
        {
            "media_id": "2",
            "title": "An audio",
            "type": "audio",
            "last_modified": "2026-07-10T00:00:00Z",
        },
        {
            "media_id": "3",
            "title": "More audio",
            "type": "audio",
            "last_modified": "2026-07-10T00:00:00Z",
        },
    ]
    return build_library_media_state(
        records,
        active_type="video",
        select_mode=True,
    )


class _FilteredMediaCanvasApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_filtered_select_mode_state(), id="library-media-canvas"
        )


@pytest.mark.asyncio
async def test_select_all_label_uses_rendered_count_not_total_count():
    """The "Select all N shown" label must count the rendered rows, not
    ``canvas.count`` (the pre-filter total across all media types). With a
    media-type filter active, ``count`` (3) overstates the one rendered row.
    """
    state = _filtered_select_mode_state()
    assert len(state.rows) == 1
    assert state.count == 3  # guards the fixture: total > rendered
    app = _FilteredMediaCanvasApp()
    async with app.run_test() as pilot:
        select_all_btn = pilot.app.query_one("#library-media-select-all", Button)
        label = str(select_all_btn.label)
        assert f"Select all {len(state.rows)} shown" == label
        assert str(state.count) not in label


class _MediaCanvasSelectedApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=dataclasses.replace(_select_mode_canvas_state(), selected_count=1),
            id="library-media-canvas",
        )


@pytest.mark.asyncio
async def test_export_selected_tooltip_follows_its_disabled_state():
    """F-018: "Export selected" disabled with zero selection says WHY;
    with a selection the tooltip describes the action."""
    app = _MediaCanvasApp()
    async with app.run_test() as pilot:
        export_btn = pilot.app.query_one("#library-media-export-selected", Button)
        assert export_btn.disabled is True
        assert "select" in str(export_btn.tooltip).lower()

    async with _MediaCanvasSelectedApp().run_test() as pilot:
        export_btn = pilot.app.query_one("#library-media-export-selected", Button)
        assert export_btn.disabled is False
        assert "export" in str(export_btn.tooltip).lower()


# ---------------------------------------------------------------------------
# task-2853: the Select-mode toolbar ships "Delete selected" alongside
# "Export selected", a bulk-delete confirmation that replaces the toolbar in
# place, and the preview pane never shows an item outside select mode.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_selected_button_renders_and_disables_at_zero_selection():
    app = _MediaCanvasApp()
    async with app.run_test() as pilot:
        delete_btn = pilot.app.query_one("#library-media-delete-selected", Button)
        assert delete_btn.disabled is True
        assert "select" in str(delete_btn.tooltip).lower()

    async with _MediaCanvasSelectedApp().run_test() as pilot:
        delete_btn = pilot.app.query_one("#library-media-delete-selected", Button)
        assert delete_btn.disabled is False
        assert "trash" in str(delete_btn.tooltip).lower()


def _confirming_bulk_delete_state() -> LibraryMediaCanvasState:
    return dataclasses.replace(
        _select_mode_canvas_state(), selected_count=2, confirming_bulk_delete=True
    )


class _MediaCanvasConfirmingApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_confirming_bulk_delete_state(), id="library-media-canvas"
        )


@pytest.mark.asyncio
async def test_confirming_bulk_delete_swaps_toolbar_for_confirm_row():
    """While confirming, the normal select toolbar (Select all/Clear/Export
    selected/Delete selected) is replaced by a confirm copy naming the
    count plus Delete/Cancel -- mirroring the single-item viewer delete's
    own in-place armed-button pattern (never a modal)."""
    app = _MediaCanvasConfirmingApp()
    async with app.run_test() as pilot:
        confirm_copy = pilot.app.query_one(
            "#library-media-bulk-delete-confirm-copy", Static
        )
        assert "2" in str(confirm_copy.renderable)
        assert "trash" in str(confirm_copy.renderable).lower()

        confirm_btn = pilot.app.query_one(
            "#library-media-bulk-delete-confirm", Button
        )
        cancel_btn = pilot.app.query_one("#library-media-bulk-delete-cancel", Button)
        assert confirm_btn is not None and cancel_btn is not None

        # The normal toolbar's controls are gone while confirming.
        with pytest.raises(NoMatches):
            pilot.app.query_one("#library-media-select-all", Button)
        with pytest.raises(NoMatches):
            pilot.app.query_one("#library-media-export-selected", Button)
        with pytest.raises(NoMatches):
            pilot.app.query_one("#library-media-delete-selected", Button)

        # "N selected" stays visible for context.
        count_static = pilot.app.query_one(
            "#library-media-selected-count", Static
        )
        assert "2" in str(count_static.renderable)


def _delete_receipt_state() -> LibraryMediaCanvasState:
    """Outside select mode (a full-success delete exits it) with a
    receipt from the just-completed bulk delete."""
    return dataclasses.replace(
        _select_mode_canvas_state(), select_mode=False, delete_receipt_count=2
    )


class _MediaCanvasReceiptApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_delete_receipt_state(), id="library-media-canvas"
        )


@pytest.mark.asyncio
async def test_delete_receipt_renders_count_with_undo_and_dismiss():
    """task-4022 AC2: a completed bulk delete's receipt renders OUTSIDE
    select mode (a full success exits it) naming the count, with an Undo
    affordance right at the point of action -- mirroring the ingest
    queue's own done-row grammar."""
    app = _MediaCanvasReceiptApp()
    async with app.run_test() as pilot:
        receipt_copy = pilot.app.query_one(
            "#library-media-bulk-delete-receipt-copy", Static
        )
        assert "2" in str(receipt_copy.renderable)
        assert "deleted" in str(receipt_copy.renderable).lower()

        undo_btn = pilot.app.query_one("#library-media-bulk-delete-undo", Button)
        dismiss_btn = pilot.app.query_one(
            "#library-media-bulk-delete-receipt-dismiss", Button
        )
        assert undo_btn is not None and dismiss_btn is not None


@pytest.mark.asyncio
async def test_delete_receipt_absent_when_count_zero():
    """Regression guard: the normal state (no receipt) renders none of
    the receipt row's widgets."""
    app = _MediaCanvasApp()  # delete_receipt_count defaults to 0
    async with app.run_test() as pilot:
        with pytest.raises(NoMatches):
            pilot.app.query_one("#library-media-bulk-delete-receipt-copy", Static)
        with pytest.raises(NoMatches):
            pilot.app.query_one("#library-media-bulk-delete-undo", Button)


def _select_mode_with_preview_state() -> LibraryMediaCanvasState:
    """Select mode active with a stale ``selected_id``/preview left over
    from before Select was entered -- the exact UAT repro shape (LIB-05)."""
    return dataclasses.replace(
        _select_mode_canvas_state(),
        selected_id="1",
        preview_lines=("First item", "Type: video", "Updated: today"),
    )


class _MediaCanvasSelectModePreviewApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_select_mode_with_preview_state(), id="library-media-canvas"
        )


@pytest.mark.asyncio
async def test_preview_pane_hidden_while_select_mode_active():
    """task-2853 AC4: the preview pane never shows an item outside the
    current selection context -- while Select mode is active it is hidden
    entirely, even when ``canvas.selected_id``/``preview_lines`` carry a
    stale normal-mode preview (the UAT's "bottom preview pane meanwhile
    shows a previously-selected different item" finding)."""
    app = _MediaCanvasSelectModePreviewApp()
    async with app.run_test() as pilot:
        preview = pilot.app.query_one("#library-media-preview")
        assert preview.display is False


@pytest.mark.asyncio
async def test_preview_pane_visible_outside_select_mode_with_selection():
    """Regression guard: the AC4 fix must not hide the preview OUTSIDE
    select mode -- only while actively selecting."""
    state = dataclasses.replace(
        _select_mode_with_preview_state(), select_mode=False
    )

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaCanvas(canvas=state, id="library-media-canvas")

    async with _App().run_test() as pilot:
        preview = pilot.app.query_one("#library-media-preview")
        assert preview.display is True


# ---------------------------------------------------------------------------
# task-2853: screen-level handler behavior (fake ``self``, same idiom as the
# row-press tests above).
# ---------------------------------------------------------------------------


def test_row_press_blocked_while_confirming_bulk_delete():
    """A row's checkbox must not change while the bulk-delete confirmation
    is showing -- otherwise the confirmed count could silently drift from
    what the user actually confirmed."""
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake._library_media_row_selection.toggle("7")  # pre-armed selection
    event = SimpleNamespace(button=SimpleNamespace(media_id="7"), stop=lambda: None)
    LibraryScreen.handle_library_media_row(fake, event)
    # Still selected -- untouched, not toggled off.
    assert fake._library_media_row_selection.is_selected("7")
    assert fake._refreshed == 0


def test_delete_selected_arms_confirmation():
    fake = _media_fake(select_mode=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    fake._library_media_row_selection.select_all(["1", "2"])
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_delete_selected(fake, event)
    assert fake._library_media_confirming_bulk_delete is True
    assert fake._refreshed == 1  # canvas sync fallback (fake has no widgets)
    # task-3020 AC2: the footer must be explicitly re-registered here --
    # live-verification caught that the canvas-scoped sync above leaves it
    # showing the plain list's stale "esc focus rail" hint otherwise.
    assert fake._footer_registrations == 1


def test_delete_selected_noop_when_nothing_selected():
    fake = _media_fake(select_mode=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_delete_selected(fake, event)
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._refreshed == 0


def test_bulk_delete_cancel_clears_confirming_flag():
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_bulk_delete_cancel(fake, event)
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._footer_registrations == 1
    assert fake._refreshed == 1


def test_bulk_delete_confirm_reads_selection_and_kicks_worker():
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake._library_media_row_selection.select_all(["3", "1"])
    delete_calls = []

    async def _delete(ids):
        delete_calls.append(ids)

    fake._delete_library_media_selection = _delete
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append(coro)
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_bulk_delete_confirm(fake, event)
    assert len(worker_calls) == 1
    asyncio.run(worker_calls[0])
    assert delete_calls == [("1", "3")]  # sorted, read synchronously


async def _noop_delete(ids):
    """Stand-in for ``_delete_library_media_selection`` in tests that only
    care about the confirm handler's own guard/dispatch behavior, never
    the actual delete -- mirrors ``test_bulk_delete_confirm_reads_
    selection_and_kicks_worker``'s ``_delete`` stub but returns a fresh
    coroutine per call so multiple presses can each be inspected/closed
    independently."""
    return None


def test_bulk_delete_confirm_second_press_while_in_flight_is_noop():
    """task-3020 AC1: a fast double-press on the confirm button must not
    launch a second delete worker over the same selection -- the confirm
    row stays visible/enabled until the FIRST worker's own completion
    recompose swaps it away, so without a synchronous guard a second
    press before that would hand off a second worker over the identical
    frozen id tuple (harmless for the idempotent delete itself, but its
    OWN rail-count decrement would double-count)."""
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake._library_media_row_selection.select_all(["1", "2"])
    fake._delete_library_media_selection = _noop_delete
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_bulk_delete_confirm(fake, event)
    assert len(worker_calls) == 1
    assert fake._library_media_bulk_delete_in_flight is True

    # Second press before the first worker has had any chance to run
    # (and clear the flag in its own ``finally``) -- must be a complete
    # no-op: no second worker, no second coroutine even constructed.
    LibraryScreen.handle_library_media_bulk_delete_confirm(fake, event)
    assert len(worker_calls) == 1

    worker_calls[0][0].close()  # avoid a "coroutine never awaited" warning


def test_bulk_delete_confirm_uses_exclusive_worker_group():
    """task-3020 AC1: belt-and-suspenders -- the worker is scheduled
    ``exclusive=True`` in its own named group, matching this screen's
    other single-flight workers (e.g. ``library_note_save``)."""
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake._library_media_row_selection.select_all(["1"])
    fake._delete_library_media_selection = _noop_delete
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_bulk_delete_confirm(fake, event)

    assert len(worker_calls) == 1
    _coro, kwargs = worker_calls[0]
    assert kwargs.get("exclusive") is True
    assert kwargs.get("group") == "library_media_bulk_delete"
    worker_calls[0][0].close()


def test_bulk_delete_confirm_allowed_again_after_in_flight_flag_clears():
    """Regression guard: the guard must not be a one-shot lockout -- once
    a prior delete's own worker completes (clearing the flag in its
    ``finally``, see ``test_delete_selection_soft_deletes_via_real_db_
    and_updates_records_and_counts``), a legitimate follow-up bulk delete
    dispatches normally."""
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake._library_media_row_selection.select_all(["1"])
    fake._delete_library_media_selection = _noop_delete
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_bulk_delete_confirm(fake, event)
    assert len(worker_calls) == 1
    worker_calls[0][0].close()

    # Simulate the first worker's own completion clearing the guard.
    fake._library_media_bulk_delete_in_flight = False
    fake._library_media_confirming_bulk_delete = True
    fake._library_media_row_selection.select_all(["2"])

    LibraryScreen.handle_library_media_bulk_delete_confirm(fake, event)
    assert len(worker_calls) == 2
    worker_calls[1][0].close()


def test_bulk_delete_confirm_empty_selection_is_noop():
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    fake.run_worker = lambda coro, **k: pytest.fail("must not start a worker")
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_bulk_delete_confirm(fake, event)
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._refreshed == 1


def test_select_toggle_off_with_selection_notifies_discard_and_resets_confirm():
    """task-2853 AC4: leaving Select mode without acting states the discard
    explicitly. Also guards against a stranded ``confirming_bulk_delete``
    flag surviving an exit via "Done" mid-confirmation."""
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    fake._library_media_row_selection.select_all(["1", "2"])
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_select_toggle(fake, event)
    assert fake._library_media_select_mode is False
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._library_media_row_selection.count == 0
    assert len(fake._notified) == 1
    message, kwargs = fake._notified[0]
    assert "2" in message and "discard" in message.lower()
    assert kwargs.get("severity") == "information"


def test_select_toggle_off_with_empty_selection_does_not_notify():
    fake = _media_fake(select_mode=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_select_toggle(fake, event)
    assert fake._notified == []


def test_select_toggle_on_does_not_notify():
    fake = _media_fake(select_mode=False)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_select_toggle(fake, event)
    assert fake._library_media_select_mode is True
    assert fake._notified == []


def test_type_filter_change_exits_select_mode_and_notifies_discard():
    """The type-filter change also silently reset select mode before
    task-2853 -- it goes through the same shared exit path, so it cannot
    strand ``confirming_bulk_delete`` either.

    (rebase note: task-14902 retired the per-press cycle -- the chooser
    press now only opens the choice strip and is inert while the
    bulk-delete confirmation is armed (task-2853 AC3), and the actual
    filter change moved to the strip's pick handler. The pinned outcome
    is unchanged; it is asserted at the pick seam, the one place the
    filter can change now.)"""
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    fake.call_after_refresh = lambda *a, **k: None
    fake._focus_library_control = lambda *a, **k: None
    fake._library_media_row_selection.select_all(["9"])
    fake._library_media_type_filter = "All"
    fake._library_media_type_choices_visible = False
    fake._library_media_browse_controller = SimpleNamespace(
        type_options=("All", "video")
    )
    fake._library_media_type_options = (
        LibraryScreen._library_media_type_options.__get__(fake)
    )
    # Pressing the chooser under an armed confirm is inert: no strip, no
    # filter drift, the confirmation stays exactly as armed.
    press = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_type_filter_pressed(fake, press)
    assert fake._library_media_type_choices_visible is False
    assert fake._library_media_type_filter == "All"
    assert fake._library_media_confirming_bulk_delete is True
    # A strip pick applies the value and routes through the shared exit
    # helper -- the original task-2853 pin, one seam over.
    fake._library_media_type_choices_visible = True
    fake._request_library_media_type = (
        lambda *_args, **_kwargs: fake._clear_library_media_selection_for_scope_change()
    )
    pick = SimpleNamespace(
        stop=lambda: None,
        option=SimpleNamespace(choice_value="video"),
    )
    LibraryScreen.handle_library_media_type_choice(fake, pick)
    assert fake._library_media_type_filter == "video"
    assert fake._library_media_type_choices_visible is False
    assert fake._library_media_select_mode is False
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._library_media_row_selection.count == 0
    assert len(fake._notified) == 1
    assert fake._notified[0][0] == "Selection cleared."


# ---------------------------------------------------------------------------
# task-2853 AC3: the actual soft-delete, driven against a REAL (file-backed)
# MediaDatabase through the exact seam the single-item viewer delete uses --
# never :memory: (the bulk delete hops to a worker thread via
# ``isolate_in_worker=True``, and a fresh connection to :memory: on a new
# thread is a distinct, empty database -- see
# Tests/Library/test_library_export_roundtrip.py's module docstring for the
# same trap).
# ---------------------------------------------------------------------------


def _bulk_delete_fake(*, db, records, counts, selected_ids):
    local_service = LocalMediaReadingService(db)
    scope_service = MediaReadingScopeService(local_service, None)
    selection = RowSelection("media")
    selection.select_all(selected_ids)
    notified = []
    refresh_calls = []
    entry_focus_arm_calls = []
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            media_reading_scope_service=scope_service,
            notify=lambda msg, **k: notified.append((msg, k)),
        ),
        _notified=notified,
        _refresh_calls=refresh_calls,
        _entry_focus_arm_calls=entry_focus_arm_calls,
        _local_source_records={"media": tuple(records)},
        _local_source_counts=dict(counts),
        _library_media_row_selection=selection,
        _library_media_select_mode=True,
        _library_media_confirming_bulk_delete=True,
        # task-3020 AC1: a real caller (``handle_library_media_bulk_
        # delete_confirm``) always sets this True BEFORE scheduling the
        # worker this coroutine's caller is standing in for -- start it
        # True here too, so the tests below can assert the ``finally``
        # actually clears it on every completion path.
        # P1 re-critique finding 3: this ONE flag now also guards Undo --
        # see its declaration on ``LibraryScreen`` for why two independent
        # flags (one per direction) let the two race on shared state.
        _library_media_bulk_delete_in_flight=True,
        # task-4022 AC2: the receipt starts empty -- a real
        # ``handle_library_media_delete_selected`` already cleared any
        # earlier one when it armed this confirmation.
        _library_media_delete_receipt_ids=(),
        is_mounted=True,
        refresh=lambda **k: refresh_calls.append(k),
        # review round 2: pin that a full-success bulk delete re-arms
        # keyboard entry focus (task-2856 AC1's "return to a list canvas"
        # convention); a bare recording stub is enough here since the
        # assertion only cares WHETHER it was called, not what it does --
        # that behavior is already covered by task-2856's own tests.
        _arm_library_list_entry_focus=lambda: entry_focus_arm_calls.append(True),
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _source_record_id=LibraryScreen._source_record_id,
    )
    fake._library_media_backing_id = types.MethodType(
        LibraryScreen._library_media_backing_id, fake
    )
    fake._notify_library_media_delete_warning = types.MethodType(
        LibraryScreen._notify_library_media_delete_warning, fake
    )
    return _bind_media_mutation_seams(fake)


@pytest.mark.parametrize(
    "media_id",
    ("local:media:local:media:1", "local:media:0", "0", "-1", True),
)
@pytest.mark.asyncio
async def test_invalid_mutation_ids_never_reach_delete_service(tmp_path, media_id):
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="invalid-id")
    fake = _bulk_delete_fake(
        db=db,
        records=(),
        counts={"media": 0},
        selected_ids=[],
    )
    calls = []

    async def delete_media_item(**kwargs):
        calls.append(kwargs)

    fake.app_instance.media_reading_scope_service = SimpleNamespace(
        delete_media_item=delete_media_item
    )

    await LibraryScreen._delete_library_media_selection(fake, (media_id,))

    assert calls == []
    with pytest.raises(ValueError, match="positive backing id"):
        LibraryScreen._required_library_media_backing_id(fake, media_id)


@pytest.mark.asyncio
async def test_delete_selection_soft_deletes_via_real_db_and_updates_records_and_counts(
    tmp_path,
):
    """Full success: both targeted items are moved to trash in the REAL
    DB (``MediaDatabase.mark_as_trash``, the same soft-deletion the
    single-item viewer delete already uses -- never raw SQL), dropped
    from the in-place list/rail-count bookkeeping, and select mode exits
    since the confirmed action fully completed. Applied-page canonical
    identities stay canonical in selection/receipt state while the DB gets
    positive backing ids."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-2853-bulk-delete"
    )
    keep_id, _, _ = db.add_media_with_keywords(
        title="Keep", content="keep", media_type="article", keywords=[]
    )
    delete_a_id, _, _ = db.add_media_with_keywords(
        title="Delete A", content="a", media_type="article", keywords=[]
    )
    delete_b_id, _, _ = db.add_media_with_keywords(
        title="Delete B", content="b", media_type="article", keywords=[]
    )
    keep_identity = f"local:media:{keep_id}"
    delete_a_identity = f"local:media:{delete_a_id}"
    delete_b_identity = f"local:media:{delete_b_id}"
    records = (
        {"id": keep_identity, "title": "Keep"},
        {"id": delete_a_identity, "title": "Delete A"},
        {"id": delete_b_identity, "title": "Delete B"},
    )
    fake = _bulk_delete_fake(
        db=db,
        records=records,
        counts={"media": 3},
        selected_ids=[delete_a_identity, delete_b_identity],
    )

    await LibraryScreen._delete_library_media_selection(
        fake, (delete_a_identity, delete_b_identity)
    )

    assert db.get_media_by_id(delete_a_id, include_trash=True)["is_trash"] in {
        1,
        True,
    }
    assert db.get_media_by_id(delete_b_id, include_trash=True)["is_trash"] in {
        1,
        True,
    }
    assert not db.get_media_by_id(keep_id, include_trash=True)["is_trash"]

    remaining_ids = {r["id"] for r in fake._local_source_records["media"]}
    assert remaining_ids == {keep_identity}
    assert fake._local_source_counts["media"] == 1

    assert fake._library_media_row_selection.count == 0
    assert fake._library_media_select_mode is False
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._notified == []
    # task-4022 AC2: a full success leaves a receipt naming exactly the
    # ids that were actually deleted, ready for Undo.
    assert fake._library_media_delete_receipt_ids == (
        delete_a_identity,
        delete_b_identity,
    )
    # AC3's rail count lives on the SHELL input (built from
    # ``_local_source_counts`` in ``_build_library_shell_input``), which a
    # canvas-scoped sync deliberately skips (see ``_sync_library_canvas``'s
    # own docstring) -- only a full ``refresh(recompose=True)`` repaints
    # it, mirroring ``_delete_library_media_item``'s own tail.
    assert fake._refresh_calls == [{"recompose": True}]
    # review round 2: a full-success bulk delete exits Select mode --
    # exactly the "return to a list canvas" transition task-2856 AC1's
    # entry-focus convention targets (mirrors ``_exit_library_media_
    # viewer``'s own tail). Without this a keyboard-only user is left
    # with nothing focused once the confirm row's "Delete" button (which
    # had focus) is gone from the DOM after the recompose above.
    assert fake._entry_focus_arm_calls == [True]
    # task-3020 AC1: the in-flight guard is cleared once the worker
    # actually completes, so a legitimate follow-up bulk delete is never
    # left permanently blocked.
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._mutation_events[0] == ("begin",)
    assert any(event[0] == "reconcile" for event in fake._mutation_events)
    assert any(event[0] == "request" for event in fake._mutation_events)
    assert any(event[0] == "facets" for event in fake._mutation_events)

    db.close_connection()


@pytest.mark.asyncio
async def test_delete_selection_partial_failure_keeps_select_mode_and_warns(
    tmp_path,
):
    """One id in the batch does not exist in the real DB (already gone) --
    the batch must not abort: the real id is still deleted, the missing
    one is reported, and select mode stays active with only the failed id
    still selected so the user can see/retry it."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-2853-bulk-delete-partial"
    )
    real_id, _, _ = db.add_media_with_keywords(
        title="Real", content="r", media_type="article", keywords=[]
    )
    missing_id = "999999"
    records = (
        {"id": str(real_id), "title": "Real"},
        {"id": missing_id, "title": "Ghost"},
    )
    fake = _bulk_delete_fake(
        db=db,
        records=records,
        counts={"media": 2},
        selected_ids=[str(real_id), missing_id],
    )

    await LibraryScreen._delete_library_media_selection(
        fake, (str(real_id), missing_id)
    )

    assert db.get_media_by_id(real_id, include_trash=True)["is_trash"] in {1, True}

    remaining_ids = {r["id"] for r in fake._local_source_records["media"]}
    assert remaining_ids == {missing_id}
    assert fake._local_source_counts["media"] == 1

    assert fake._library_media_row_selection.ids == frozenset({missing_id})
    assert fake._library_media_select_mode is True
    assert fake._library_media_confirming_bulk_delete is False
    # task-4022 AC2: even a partial batch leaves a receipt for the subset
    # that DID succeed -- the user can still undo the real item, and the
    # missing one is separately reported below.
    assert fake._library_media_delete_receipt_ids == (str(real_id),)
    assert len(fake._notified) == 1
    message, kwargs = fake._notified[0]
    assert "1" in message
    assert kwargs.get("severity") == "warning"
    # A partial failure still touched the list/rail counts (one item DID
    # get deleted) -- the rail must still repaint, same as full success.
    assert fake._refresh_calls == [{"recompose": True}]
    # task-3020 AC3 (review round 2 superseded): a partial failure keeps
    # Select mode ACTIVE (the failed id is still checked, waiting for a
    # retry) -- it is not a "return to a list" transition the way exiting
    # Select mode entirely is, but entry focus IS now armed here too, so
    # it lands on that still-checked row instead of leaving nothing
    # focused once the confirm row's "Delete" button (which had focus) is
    # gone from the DOM after the recompose above.
    assert fake._entry_focus_arm_calls == [True]
    assert fake._mutation_events[0] == ("begin",)
    assert any(event[0] == "reconcile" for event in fake._mutation_events)
    assert any(event[0] == "request" for event in fake._mutation_events)
    assert any(event[0] == "facets" for event in fake._mutation_events)
    assert fake._library_media_bulk_delete_in_flight is False

    db.close_connection()


@pytest.mark.asyncio
async def test_delete_selection_service_unavailable_keeps_selection_and_warns():
    """No ``media_reading_scope_service`` on the app (or no
    ``delete_media_item`` seam) must fail closed: nothing is removed from
    the list, the whole batch is reported as failed, and select mode is
    left active."""
    entry_focus_arm_calls = []
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            media_reading_scope_service=None,
            notify=lambda msg, **k: fake._notified.append((msg, k)),
        ),
        _notified=[],
        _local_source_records={"media": ({"id": "1", "title": "A"},)},
        _local_source_counts={"media": 1},
        _library_media_row_selection=RowSelection("media"),
        _library_media_select_mode=True,
        _library_media_confirming_bulk_delete=True,
        _library_media_bulk_delete_in_flight=True,
        is_mounted=True,
        refresh=lambda **k: None,
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _source_record_id=LibraryScreen._source_record_id,
        _entry_focus_arm_calls=entry_focus_arm_calls,
        _arm_library_list_entry_focus=lambda: entry_focus_arm_calls.append(True),
    )
    fake._library_media_row_selection.select_all(["1"])
    fake._notify_library_media_delete_warning = types.MethodType(
        LibraryScreen._notify_library_media_delete_warning, fake
    )
    _bind_media_mutation_seams(fake)

    await LibraryScreen._delete_library_media_selection(fake, ("1",))

    assert len(fake._local_source_records["media"]) == 1
    assert fake._local_source_counts["media"] == 1
    assert fake._library_media_select_mode is True
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._library_media_row_selection.ids == frozenset({"1"})
    # task-4022 AC2: nothing succeeded, so there is no receipt to show.
    assert fake._library_media_delete_receipt_ids == ()
    assert len(fake._notified) == 1
    assert fake._notified[0][1].get("severity") == "warning"
    # task-3020 AC1/AC3: even a total failure clears the in-flight guard
    # and arms entry focus onto the still-checked (failed) row.
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._entry_focus_arm_calls == [True]


# ---------------------------------------------------------------------------
# task-4022 AC2: Undo for a bulk-delete receipt, driven against the SAME
# real (file-backed) MediaDatabase + MediaReadingScopeService seam as the
# delete tests above (``restore_media_item`` -> ``MediaDatabase.
# restore_from_trash``, never raw SQL).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undo_restores_items_via_real_db_and_updates_records_and_counts(
    tmp_path,
):
    """Full success: every id in the receipt is un-trashed in the REAL DB,
    reinserted into the in-place list/rail-count bookkeeping, and the
    receipt itself is cleared. The receipt carries canonical identities while
    the restore service receives positive backing ids."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4022-undo"
    )
    keep_id, _, _ = db.add_media_with_keywords(
        title="Keep", content="keep", media_type="article", keywords=[]
    )
    undo_a_id, _, _ = db.add_media_with_keywords(
        title="Undo A", content="a", media_type="article", keywords=[]
    )
    undo_b_id, _, _ = db.add_media_with_keywords(
        title="Undo B", content="b", media_type="article", keywords=[]
    )
    assert db.mark_as_trash(undo_a_id) is True
    assert db.mark_as_trash(undo_b_id) is True

    fake = _bulk_delete_fake(
        db=db,
        records=({"id": str(keep_id), "title": "Keep"},),
        counts={"media": 1},
        selected_ids=[],
    )
    undo_a_identity = f"local:media:{undo_a_id}"
    undo_b_identity = f"local:media:{undo_b_id}"
    fake._library_media_delete_receipt_ids = (undo_a_identity, undo_b_identity)

    await LibraryScreen._undo_library_media_bulk_delete(
        fake, (undo_a_identity, undo_b_identity)
    )

    assert not db.get_media_by_id(undo_a_id, include_trash=True)["is_trash"]
    assert not db.get_media_by_id(undo_b_id, include_trash=True)["is_trash"]

    # ``restore_media_item`` returns the raw DB row, whose "id" is an int
    # (unlike the manually-seeded "Keep" record above, which used a str) --
    # ``str()`` here mirrors how ``_source_record_id``/the state builder
    # normalize both shapes for display and dedup.
    restored_ids = {str(r["id"]) for r in fake._local_source_records["media"]}
    assert restored_ids == {str(keep_id), str(undo_a_id), str(undo_b_id)}
    assert fake._local_source_counts["media"] == 3

    assert fake._library_media_delete_receipt_ids == ()
    assert fake._notified == []
    assert fake._refresh_calls == [{"recompose": True}]
    assert fake._library_media_bulk_delete_in_flight is False
    # Review round 1 (Important #2): ``refresh(recompose=True)`` destroys
    # and remounts the receipt row, taking the focused "Undo" button with
    # it -- entry focus must be re-armed the same way the delete path's own
    # tail already does.
    assert fake._entry_focus_arm_calls == [True]
    assert fake._mutation_events[0] == ("begin",)
    assert any(event[0] == "reconcile" for event in fake._mutation_events)
    assert any(event[0] == "request" for event in fake._mutation_events)
    assert any(event[0] == "facets" for event in fake._mutation_events)

    db.close_connection()


@pytest.mark.asyncio
async def test_undo_reinserts_and_reselects_when_item_matches_active_scope(tmp_path):
    """A one-item receipt returns to Reader when its summary is in scope."""
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="undo-visible")
    media_id, _, _ = db.add_media_with_keywords(
        title="Visible", content="body", media_type="article", keywords=[]
    )
    assert db.mark_as_trash(media_id) is True
    fake = _bulk_delete_fake(db=db, records=(), counts={"media": 0}, selected_ids=[])
    fake._library_media_browse_controller.applied_scope = MediaBrowseScope(
        media_type="article"
    )
    selections = []
    def select_restored(*args, **kwargs):
        fake._mutation_events.append(("select", args[0]))
        selections.append((args, kwargs))

    fake._select_library_media_reader_row = select_restored

    identity = f"local:media:{media_id}"
    await LibraryScreen._undo_library_media_bulk_delete(fake, (identity,))

    assert selections == [((identity, "Visible"), {"immediate": True})]
    reconcile_index = next(
        index
        for index, event in enumerate(fake._mutation_events)
        if event[0] == "reconcile"
    )
    select_index = fake._mutation_events.index(("select", identity))
    assert reconcile_index < select_index
    db.close_connection()


@pytest.mark.asyncio
async def test_undo_succeeds_with_restored_outside_current_filter_message(tmp_path):
    """A filtered-out restore keeps Reader selection stable and explains why."""
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="undo-filtered")
    media_id, _, _ = db.add_media_with_keywords(
        title="Visible", content="body", media_type="article", keywords=[]
    )
    assert db.mark_as_trash(media_id) is True
    fake = _bulk_delete_fake(db=db, records=(), counts={"media": 0}, selected_ids=[])
    fake._library_media_browse_controller.applied_scope = MediaBrowseScope(
        query="different"
    )
    selections = []
    fake._select_library_media_reader_row = lambda *args, **kwargs: selections.append(
        (args, kwargs)
    )

    await LibraryScreen._undo_library_media_bulk_delete(
        fake, (f"local:media:{media_id}",)
    )

    assert selections == []
    assert fake._notified == [("Restored outside the current filter.", {})]
    db.close_connection()


@pytest.mark.asyncio
async def test_undo_partial_failure_narrows_receipt_and_warns(tmp_path):
    """One id in the receipt no longer exists in the real DB (e.g.
    permanently purged some other way) -- the real id is still restored,
    the missing one is reported, and the receipt narrows to just the
    still-failed id so a retry is possible."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4022-undo-partial"
    )
    real_id, _, _ = db.add_media_with_keywords(
        title="Real", content="r", media_type="article", keywords=[]
    )
    assert db.mark_as_trash(real_id) is True
    missing_id = "999999"

    fake = _bulk_delete_fake(
        db=db, records=(), counts={"media": 0}, selected_ids=[]
    )
    fake._library_media_delete_receipt_ids = (str(real_id), missing_id)

    await LibraryScreen._undo_library_media_bulk_delete(
        fake, (str(real_id), missing_id)
    )

    assert not db.get_media_by_id(real_id, include_trash=True)["is_trash"]

    restored_ids = {str(r["id"]) for r in fake._local_source_records["media"]}
    assert restored_ids == {str(real_id)}
    assert fake._local_source_counts["media"] == 1

    assert fake._library_media_delete_receipt_ids == (missing_id,)
    assert len(fake._notified) == 1
    message, kwargs = fake._notified[0]
    assert "1" in message
    assert kwargs.get("severity") == "warning"
    # Review round 1 (Important #2): a PARTIAL failure still recomposes
    # (the receipt narrows and re-renders with a new "Undo" button
    # instance), so entry focus must be armed here too, not only on full
    # success.
    assert fake._entry_focus_arm_calls == [True]

    db.close_connection()


@pytest.mark.asyncio
async def test_undo_does_not_duplicate_a_record_already_present(tmp_path):
    """Defensive: if the target id is somehow already back in the cached
    list (e.g. a background refresh raced ahead of Undo), restoring it
    again must not insert a second copy."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4022-undo-dupe-guard"
    )
    media_id, _, _ = db.add_media_with_keywords(
        title="Already back", content="x", media_type="article", keywords=[]
    )
    assert db.mark_as_trash(media_id) is True
    assert db.restore_from_trash(media_id) is True  # e.g. a racing refresh

    fake = _bulk_delete_fake(
        db=db,
        records=({"id": str(media_id), "title": "Already back"},),
        counts={"media": 1},
        selected_ids=[],
    )
    fake._library_media_delete_receipt_ids = (str(media_id),)

    await LibraryScreen._undo_library_media_bulk_delete(fake, (str(media_id),))

    matching = [
        r for r in fake._local_source_records["media"] if r["id"] == str(media_id)
    ]
    assert len(matching) == 1
    assert fake._local_source_counts["media"] == 1

    db.close_connection()


@pytest.mark.asyncio
async def test_delete_confirm_refused_while_undo_in_flight_keeps_state_consistent(
    tmp_path,
):
    """P1 re-critique finding 3 (Bug): Undo and a fresh bulk delete used
    to be guarded by two INDEPENDENT in-flight flags and scheduled into
    two different exclusive worker groups, so nothing stopped a delete
    press from starting while an Undo triggered moments earlier was still
    awaiting its own real per-item service calls. Both mutate the same
    shared state (``_local_source_records``, ``_local_source_counts``,
    ``_library_media_delete_receipt_ids``), so whichever finished LAST
    could clobber the other's writes with a stale snapshot -- the
    interleaving the reviewer described: "Undo starts restoring and
    awaits -> user starts a new bulk delete -> delete mutates shared
    state -> Undo completes and writes receipt/count/list from stale
    snapshots, clobbering the newer delete."

    Locks the fix: the two button handlers now share ONE in-flight flag,
    checked synchronously as the very first line of both handlers --
    BEFORE either even reads selection/receipt state -- so a press while
    the other is in flight is refused outright (a no-op, not queued
    behind stale state) and the shared state it would have touched is
    left untouched. This is sound because the flag transition (False ->
    True) and the refusal check both happen synchronously on the UI
    thread with no ``await`` between them: there is no window where a
    second press can observe a stale False before the first press's True
    is visible.
    """
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-4022-review-p1-race"
    )
    keep_id, _, _ = db.add_media_with_keywords(
        title="Keep", content="keep", media_type="article", keywords=[]
    )
    fresh_delete_id, _, _ = db.add_media_with_keywords(
        title="Fresh delete target",
        content="fresh",
        media_type="article",
        keywords=[],
    )
    receipt_a_id, _, _ = db.add_media_with_keywords(
        title="Receipt A", content="a", media_type="article", keywords=[]
    )
    receipt_b_id, _, _ = db.add_media_with_keywords(
        title="Receipt B", content="b", media_type="article", keywords=[]
    )
    assert db.mark_as_trash(receipt_a_id) is True
    assert db.mark_as_trash(receipt_b_id) is True

    fake = _bulk_delete_fake(
        db=db,
        records=(
            {"id": str(keep_id), "title": "Keep"},
            {"id": str(fresh_delete_id), "title": "Fresh delete target"},
        ),
        counts={"media": 2},
        selected_ids=[str(fresh_delete_id)],
    )
    # A real, pre-existing receipt from an earlier bulk delete is still
    # showing (Undo/Dismiss visible) AND Select mode is separately active
    # with a fresh selection armed for a brand-new bulk delete -- exactly
    # the state a partial failure (or simply not dismissing the receipt
    # yet) leaves behind, and the state the interleaving needs both
    # affordances live at once.
    fake._library_media_delete_receipt_ids = (str(receipt_a_id), str(receipt_b_id))
    fake._library_media_bulk_delete_in_flight = False
    fake._undo_library_media_bulk_delete = types.MethodType(
        LibraryScreen._undo_library_media_bulk_delete, fake
    )
    fake._delete_library_media_selection = types.MethodType(
        LibraryScreen._delete_library_media_selection, fake
    )
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))

    # 1. Undo is pressed first -- synchronously claims the shared flag and
    #    hands its worker coroutine to ``run_worker`` (captured here, not
    #    yet run -- mirroring how a real worker wouldn't have executed a
    #    single line of the coroutine body at the instant the button
    #    handler returns).
    LibraryScreen.handle_library_media_bulk_delete_undo(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert len(worker_calls) == 1
    assert fake._library_media_bulk_delete_in_flight is True

    # 2. Before that worker has run AT ALL, the user presses "Delete" on
    #    the fresh selection. Must be refused: no second worker
    #    scheduled, and -- because the guard is the very first line of
    #    the handler -- the selection/receipt/records are never even
    #    read, let alone mutated by this refused attempt.
    LibraryScreen.handle_library_media_bulk_delete_confirm(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert len(worker_calls) == 1, "the delete press must not schedule a second worker"
    assert fake._library_media_row_selection.count == 1  # untouched
    assert fake._library_media_delete_receipt_ids == (
        str(receipt_a_id),
        str(receipt_b_id),
    )  # untouched

    # 3. Only now does Undo's own worker actually run, against the real DB.
    undo_coro, undo_kwargs = worker_calls[0]
    assert undo_kwargs.get("group") == "library_media_bulk_delete"
    await undo_coro

    assert not db.get_media_by_id(receipt_a_id, include_trash=True)["is_trash"]
    assert not db.get_media_by_id(receipt_b_id, include_trash=True)["is_trash"]
    # The refused delete attempt never touched its target.
    assert not db.get_media_by_id(fresh_delete_id, include_trash=True)["is_trash"]

    restored_ids = {str(r["id"]) for r in fake._local_source_records["media"]}
    assert restored_ids == {
        str(keep_id),
        str(fresh_delete_id),
        str(receipt_a_id),
        str(receipt_b_id),
    }
    assert fake._local_source_counts["media"] == 4
    assert fake._library_media_delete_receipt_ids == ()
    assert fake._library_media_bulk_delete_in_flight is False

    # 4. Only now that the flag has cleared is the still-armed delete
    #    selection allowed through -- proving the refusal above was a
    #    transient no-op, not a permanent lockout.
    LibraryScreen.handle_library_media_bulk_delete_confirm(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert len(worker_calls) == 2
    delete_coro, delete_kwargs = worker_calls[1]
    assert delete_kwargs.get("group") == "library_media_bulk_delete"
    await delete_coro

    assert db.get_media_by_id(fresh_delete_id, include_trash=True)["is_trash"] in {
        1,
        True,
    }
    # ``str()`` here mirrors ``test_undo_restores_items_via_real_db_and_
    # updates_records_and_counts``'s own note: the two restored-by-Undo
    # records carry the raw DB row's int "id", while the manually-seeded
    # "Keep" record above used a str.
    remaining_ids = {str(r["id"]) for r in fake._local_source_records["media"]}
    assert remaining_ids == {str(keep_id), str(receipt_a_id), str(receipt_b_id)}
    assert fake._local_source_counts["media"] == 3
    assert fake._library_media_bulk_delete_in_flight is False

    db.close_connection()


async def _noop_undo(ids):
    """Stand-in for ``_undo_library_media_bulk_delete`` -- mirrors
    ``_noop_delete``'s role for the confirm-button tests above."""
    return None


def _undo_fake(*, receipt_ids, undo_in_flight=False):
    """Handler-level fake for the Undo/Dismiss BUTTON handlers -- mirrors
    ``_media_fake`` (never touches a real DB; the coroutine itself is
    covered by the real-DB tests above).

    ``undo_in_flight`` seeds the SAME shared
    ``_library_media_bulk_delete_in_flight`` flag the delete-confirm
    handler uses (P1 re-critique finding 3) -- kept as a distinctly-named
    kwarg here since every caller in this file is specifically about the
    Undo button, not the delete button.
    """
    notified = []
    fake = SimpleNamespace(
        _library_media_delete_receipt_ids=receipt_ids,
        _library_media_bulk_delete_in_flight=undo_in_flight,
        app_instance=SimpleNamespace(
            notify=lambda msg, **k: notified.append((msg, k))
        ),
        _notified=notified,
        _undo_library_media_bulk_delete=_noop_undo,
    )
    return _bind_media_mutation_seams(fake)


def test_undo_button_kicks_worker_with_receipt_ids():
    fake = _undo_fake(receipt_ids=("1", "2"))
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_bulk_delete_undo(fake, event)

    assert len(worker_calls) == 1
    coro, kwargs = worker_calls[0]
    assert kwargs.get("exclusive") is True
    # P1 re-critique finding 3: SAME group as the delete-confirm worker
    # (see ``test_bulk_delete_confirm_uses_exclusive_worker_group``) --
    # no longer a separate "..._undo" group, so the two can never run
    # concurrently even as a defensive backstop behind the shared flag.
    assert kwargs.get("group") == "library_media_bulk_delete"
    assert fake._library_media_bulk_delete_in_flight is True
    coro.close()


def test_undo_button_noop_when_receipt_empty():
    fake = _undo_fake(receipt_ids=())
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_bulk_delete_undo(fake, event)

    assert worker_calls == []
    assert fake._library_media_bulk_delete_in_flight is False


def test_undo_button_second_press_while_in_flight_is_noop():
    fake = _undo_fake(receipt_ids=("1",), undo_in_flight=True)
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_bulk_delete_undo(fake, event)

    assert worker_calls == []


def test_undo_button_refused_while_delete_confirm_in_flight():
    """P1 re-critique finding 3: the shared flag blocks Undo when it's a
    fresh bulk DELETE that's in flight, not just another Undo -- the
    mirror image of ``test_bulk_delete_confirm_second_press_while_in_
    flight_is_noop``, now proven from the Undo side too."""
    fake = _undo_fake(receipt_ids=("1", "2"), undo_in_flight=True)
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_bulk_delete_undo(fake, event)

    assert worker_calls == []
    # The receipt is untouched -- the handler returned before reading it.
    assert fake._library_media_delete_receipt_ids == ("1", "2")


def test_dismiss_clears_receipt_without_restoring():
    fake = _undo_fake(receipt_ids=("1", "2"))
    fake.refresh = lambda **k: setattr(
        fake, "_refreshed", getattr(fake, "_refreshed", 0) + 1
    )
    event = SimpleNamespace(stop=lambda: None)

    LibraryScreen.handle_library_media_bulk_delete_receipt_dismiss(fake, event)

    assert fake._library_media_delete_receipt_ids == ()
    assert fake._refreshed == 1  # canvas sync fallback (fake has no widgets)


@pytest.mark.asyncio
async def test_single_item_delete_also_arms_entry_focus_on_success(tmp_path):
    """review round 2 sibling fix: the single-item viewer delete
    (``_delete_library_media_item``) has the IDENTICAL "return to a list
    canvas" transition on success (``_library_media_view`` flips from
    "viewer" to "list") that established the entry-focus convention in
    the first place (task-2856 AC1, ``_exit_library_media_viewer``) -- it
    was simply missed when this method was written before that
    convention existed. Pin it fixed too, not just the bulk path."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-2853-single-delete"
    )
    media_id, _, _ = db.add_media_with_keywords(
        title="Solo", content="s", media_type="article", keywords=[]
    )
    local_service = LocalMediaReadingService(db)
    scope_service = MediaReadingScopeService(local_service, None)
    entry_focus_arm_calls = []
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(media_reading_scope_service=scope_service),
        _local_source_records={"media": ({"id": str(media_id), "title": "Solo"},)},
        # task-3020 AC5: the single-item viewer delete now decrements the
        # rail count in place too, mirroring the bulk path.
        _local_source_counts={"media": 1},
        _library_media_view="viewer",
        _library_media_detail={"id": str(media_id)},
        _library_media_highlights=[{"id": "h1"}],
        _library_media_editing_analysis=True,
        _library_media_content_query="solo",
        _library_media_content_match_index=1,
        _selected_media_id=str(media_id),
        _library_media_confirming_delete=True,
        is_mounted=True,
        refresh=lambda **k: None,
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _source_record_id=LibraryScreen._source_record_id,
        _entry_focus_arm_calls=entry_focus_arm_calls,
        _arm_library_list_entry_focus=lambda: entry_focus_arm_calls.append(True),
    )
    fake._library_media_backing_id = types.MethodType(
        LibraryScreen._library_media_backing_id, fake
    )
    fake._notify_library_media_delete_warning = types.MethodType(
        LibraryScreen._notify_library_media_delete_warning, fake
    )
    _bind_media_mutation_seams(fake)

    await LibraryScreen._delete_library_media_item(fake, str(media_id))

    assert db.get_media_by_id(media_id, include_trash=True)["is_trash"] in {1, True}
    assert fake._library_media_view == "list"
    assert fake._entry_focus_arm_calls == [True]
    # task-3020 AC5: rail count decremented in place, like the bulk path.
    assert fake._local_source_counts["media"] == 0
    assert fake._mutation_events[0] == ("begin",)
    assert any(event[0] == "reconcile" for event in fake._mutation_events)
    assert any(event[0] == "request" for event in fake._mutation_events)
    assert any(event[0] == "facets" for event in fake._mutation_events)

    db.close_connection()


# ---------------------------------------------------------------------------
# task-14901 (ADR-055): single media delete is one-item bulk. It adopts the
# SAME receipt/Undo seam as "Delete selected" -- the shared
# ``_library_media_bulk_delete_in_flight`` flag, the shared exclusive worker
# group, ``_library_media_delete_receipt_ids``, and
# ``_undo_library_media_bulk_delete`` -- instead of confirm-then-silence.
# No second undo path is forked.
# ---------------------------------------------------------------------------


async def _noop_delete_item(media_id):
    """Stand-in for ``_delete_library_media_item`` -- mirrors ``_noop_undo``'s
    role for the handler-level tests (the coroutine body is covered by the
    real-DB tests below)."""
    return None


def _single_delete_confirm_fake(
    *, selected_media_id="7", in_flight=False, receipt_ids=()
):
    """Handler-level fake for the single-item viewer delete-confirm button."""
    fake = SimpleNamespace(
        _selected_media_id=selected_media_id,
        _library_media_confirming_delete=True,
        _library_media_bulk_delete_in_flight=in_flight,
        _library_media_delete_receipt_ids=receipt_ids,
        _delete_library_media_item=_noop_delete_item,
        refresh=lambda **k: None,
    )
    return _bind_media_mutation_seams(fake)


def test_single_delete_confirm_claims_shared_flag_and_group():
    """The confirm press claims the SAME in-flight flag and exclusive worker
    group as the bulk delete/Undo pair -- one interlock across all three
    mutators of the shared list/count/receipt state."""
    fake = _single_delete_confirm_fake()
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))

    LibraryScreen.handle_library_media_delete_confirm(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert len(worker_calls) == 1
    coro, kwargs = worker_calls[0]
    assert kwargs.get("exclusive") is True
    assert kwargs.get("group") == "library_media_bulk_delete"
    assert fake._library_media_bulk_delete_in_flight is True
    coro.close()


def test_single_delete_confirm_refused_while_bulk_or_undo_in_flight():
    """A single-item confirm while a bulk delete OR an Undo is still running
    is refused outright (PR-1473's one-flag rule): no second worker, and the
    receipt it would eventually overwrite is never touched."""
    fake = _single_delete_confirm_fake(in_flight=True, receipt_ids=("1", "2"))
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))

    LibraryScreen.handle_library_media_delete_confirm(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert worker_calls == []
    assert fake._library_media_delete_receipt_ids == ("1", "2")


def test_single_delete_confirm_empty_id_does_not_claim_flag():
    """The no-selected-id early-out never claims the shared flag -- a later
    legitimate delete/Undo must not find it stuck True."""
    fake = _single_delete_confirm_fake(selected_media_id="")
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))

    LibraryScreen.handle_library_media_delete_confirm(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert worker_calls == []
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._library_media_confirming_delete is False


def test_single_delete_arm_supersedes_stale_receipt():
    """Arming the viewer confirm clears any receipt still showing from an
    earlier delete -- mirroring ``handle_library_media_delete_selected``'s
    arm-time clear, so a completed receipt always reflects only what just
    happened."""
    fake = SimpleNamespace(
        _library_media_confirming_delete=False,
        _library_media_delete_receipt_ids=("9",),
        refresh=lambda **k: None,
    )

    LibraryScreen.handle_library_media_delete(
        fake, SimpleNamespace(stop=lambda: None)
    )

    assert fake._library_media_confirming_delete is True
    assert fake._library_media_delete_receipt_ids == ()


def _single_delete_worker_fake(*, db, records, counts, selected_media_id):
    """Real-DB fake for ``_delete_library_media_item`` -- the single-item
    sibling of ``_bulk_delete_fake``, seeded with the viewer state the
    method resets on success."""
    local_service = LocalMediaReadingService(db)
    scope_service = MediaReadingScopeService(local_service, None)
    notified = []
    refresh_calls = []
    entry_focus_arm_calls = []
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            media_reading_scope_service=scope_service,
            notify=lambda msg, **k: notified.append((msg, k)),
        ),
        _notified=notified,
        _refresh_calls=refresh_calls,
        _entry_focus_arm_calls=entry_focus_arm_calls,
        _local_source_records={"media": tuple(records)},
        _local_source_counts=dict(counts),
        _library_media_view="viewer",
        _library_media_detail={"id": selected_media_id},
        _library_media_highlights=[],
        _library_media_editing_analysis=False,
        _library_media_content_query="",
        _library_media_content_match_index=0,
        _selected_media_id=selected_media_id,
        _library_media_confirming_delete=True,
        # The real confirm handler sets the shared flag BEFORE scheduling
        # this coroutine -- start True so the ``finally`` clear is provable.
        _library_media_bulk_delete_in_flight=True,
        # The real arm handler already cleared any stale receipt.
        _library_media_delete_receipt_ids=(),
        is_mounted=True,
        refresh=lambda **k: refresh_calls.append(k),
        _arm_library_list_entry_focus=lambda: entry_focus_arm_calls.append(True),
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _source_record_id=LibraryScreen._source_record_id,
    )
    fake._library_media_backing_id = types.MethodType(
        LibraryScreen._library_media_backing_id, fake
    )
    fake._notify_library_media_delete_warning = types.MethodType(
        LibraryScreen._notify_library_media_delete_warning, fake
    )
    return _bind_media_mutation_seams(fake)


@pytest.mark.asyncio
async def test_single_delete_leaves_receipt_and_undo_restores_via_bulk_seam(
    tmp_path,
):
    """Full cycle against the REAL DB: a confirmed single delete trashes the
    item and leaves a one-id receipt (rendered as "✓ deleted · 1 item" with
    Undo/Dismiss), and pressing Undo restores it through the EXACT same
    ``_undo_library_media_bulk_delete`` coroutine the bulk receipt uses --
    record back in the list, rail count back up, receipt cleared. The item
    carries ``chunks`` so the restore path is proven against a chunked row
    (the task-4022 Critical was invisible because every test omitted them).
    """
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-14901-single-receipt"
    )
    media_id, _, _ = db.add_media_with_keywords(
        title="Solo",
        content="solo body",
        media_type="article",
        keywords=[],
        chunks=[{"text": "solo body", "chunk_type": "text"}],
    )
    fake = _single_delete_worker_fake(
        db=db,
        records=({"id": str(media_id), "title": "Solo"},),
        counts={"media": 1},
        selected_media_id=str(media_id),
    )

    await LibraryScreen._delete_library_media_item(fake, str(media_id))

    assert db.get_media_by_id(media_id, include_trash=True)["is_trash"] in {1, True}
    assert fake._library_media_view == "list"
    assert fake._local_source_records["media"] == ()
    assert fake._local_source_counts["media"] == 0
    # task-14901: the receipt names exactly the one deleted id, ready for
    # the existing Undo/Dismiss handlers -- no silence, no second seam.
    assert fake._library_media_delete_receipt_ids == (str(media_id),)
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._notified == []

    # Undo exactly as the receipt row's button would: handler claims the
    # shared flag, then the SAME bulk-undo coroutine runs (the interlock
    # test's capture-then-await pattern).
    fake._undo_library_media_bulk_delete = types.MethodType(
        LibraryScreen._undo_library_media_bulk_delete, fake
    )
    worker_calls = []
    fake.run_worker = lambda coro, **k: worker_calls.append((coro, k))
    LibraryScreen.handle_library_media_bulk_delete_undo(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert len(worker_calls) == 1
    assert fake._library_media_bulk_delete_in_flight is True
    undo_coro, undo_kwargs = worker_calls[0]
    assert undo_kwargs.get("group") == "library_media_bulk_delete"
    await undo_coro

    assert not db.get_media_by_id(media_id, include_trash=True)["is_trash"]
    restored_ids = {str(r["id"]) for r in fake._local_source_records["media"]}
    assert restored_ids == {str(media_id)}
    assert fake._local_source_counts["media"] == 1
    assert fake._library_media_delete_receipt_ids == ()
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._notified == []

    db.close_connection()


@pytest.mark.asyncio
async def test_single_delete_failure_leaves_no_receipt_and_clears_flag(tmp_path):
    """A failed single delete (id not in the real DB) must warn quietly,
    leave NO receipt (nothing succeeded), and still clear the shared flag so
    a follow-up delete/Undo is never permanently blocked."""
    db = MediaDatabase(
        db_path=str(tmp_path / "media.db"), client_id="task-14901-single-fail"
    )
    fake = _single_delete_worker_fake(
        db=db,
        records=({"id": "424242", "title": "Ghost"},),
        counts={"media": 1},
        selected_media_id="424242",
    )

    await LibraryScreen._delete_library_media_item(fake, "424242")

    assert fake._library_media_delete_receipt_ids == ()
    assert fake._library_media_bulk_delete_in_flight is False
    assert fake._library_media_view == "viewer"
    assert len(fake._notified) == 1
    assert fake._notified[0][1].get("severity") == "warning"

    db.close_connection()
