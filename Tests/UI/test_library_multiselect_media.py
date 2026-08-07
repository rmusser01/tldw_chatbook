import asyncio
import dataclasses
import types
from types import SimpleNamespace

import pytest
from textual.app import App
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
    build_library_media_state,
)
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas


def _media_fake(select_mode, *, confirming_bulk_delete=False):
    notified = []
    fake = SimpleNamespace(
        _library_media_select_mode=select_mode,
        _library_media_row_selection=RowSelection("media"),
        _library_media_confirming_bulk_delete=confirming_bulk_delete,
        app_instance=SimpleNamespace(
            notify=lambda msg, **k: notified.append((msg, k))
        ),
        _notified=notified,
        _opened=[],
        _refreshed=0,
        _viewer_opened=[],
    )
    # These two are real LibraryScreen instance methods (not module-level
    # helpers like ``_apply_library_row_toggle``), so handlers that call
    # ``self._exit_library_media_select_mode(...)`` need them actually
    # bound to this fake -- ``types.MethodType`` reuses the REAL
    # implementation rather than a hand-rolled stub duplicating its logic.
    fake._exit_library_media_select_mode = types.MethodType(
        LibraryScreen._exit_library_media_select_mode, fake
    )
    fake._notify_library_media_selection_discarded = types.MethodType(
        LibraryScreen._notify_library_media_selection_discarded, fake
    )
    return fake


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


class _MediaCanvasApp(App):
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


class _EmptySelectModeCanvasApp(App):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_empty_select_mode_state(), id="library-media-canvas"
        )


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


class _FilteredMediaCanvasApp(App):
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


class _MediaCanvasSelectedApp(App):
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


class _MediaCanvasConfirmingApp(App):
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


def _select_mode_with_preview_state() -> LibraryMediaCanvasState:
    """Select mode active with a stale ``selected_id``/preview left over
    from before Select was entered -- the exact UAT repro shape (LIB-05)."""
    return dataclasses.replace(
        _select_mode_canvas_state(),
        selected_id="1",
        preview_lines=("First item", "Type: video", "Updated: today"),
    )


class _MediaCanvasSelectModePreviewApp(App):
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

    class _App(App):
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


def test_type_filter_cycle_exits_select_mode_and_notifies_discard():
    """The type-filter cycle also silently reset select mode before
    task-2853 -- it now goes through the same shared exit path, so it
    cannot strand ``confirming_bulk_delete`` either."""
    fake = _media_fake(select_mode=True, confirming_bulk_delete=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    fake._library_media_row_selection.select_all(["9"])
    fake._library_media_type_filter = "All"
    fake._build_library_media_state = lambda: SimpleNamespace(
        type_options=("All", "video")
    )
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_media_type_filter_pressed(fake, event)
    assert fake._library_media_type_filter == "video"
    assert fake._library_media_select_mode is False
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._library_media_row_selection.count == 0
    assert len(fake._notified) == 1


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
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(
            media_reading_scope_service=scope_service,
            notify=lambda msg, **k: notified.append((msg, k)),
        ),
        _notified=notified,
        _refresh_calls=refresh_calls,
        _local_source_records={"media": tuple(records)},
        _local_source_counts=dict(counts),
        _library_media_row_selection=selection,
        _library_media_select_mode=True,
        _library_media_confirming_bulk_delete=True,
        is_mounted=True,
        refresh=lambda **k: refresh_calls.append(k),
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _source_record_id=LibraryScreen._source_record_id,
    )
    fake._notify_library_media_delete_warning = types.MethodType(
        LibraryScreen._notify_library_media_delete_warning, fake
    )
    return fake


@pytest.mark.asyncio
async def test_delete_selection_soft_deletes_via_real_db_and_updates_records_and_counts(
    tmp_path,
):
    """Full success: both targeted items are moved to trash in the REAL
    DB (``MediaDatabase.mark_as_trash``, the same soft-deletion the
    single-item viewer delete already uses -- never raw SQL), dropped
    from the in-place list/rail-count bookkeeping, and select mode exits
    since the confirmed action fully completed."""
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
    records = (
        {"id": str(keep_id), "title": "Keep"},
        {"id": str(delete_a_id), "title": "Delete A"},
        {"id": str(delete_b_id), "title": "Delete B"},
    )
    fake = _bulk_delete_fake(
        db=db,
        records=records,
        counts={"media": 3},
        selected_ids=[str(delete_a_id), str(delete_b_id)],
    )

    await LibraryScreen._delete_library_media_selection(
        fake, (str(delete_a_id), str(delete_b_id))
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
    assert remaining_ids == {str(keep_id)}
    assert fake._local_source_counts["media"] == 1

    assert fake._library_media_row_selection.count == 0
    assert fake._library_media_select_mode is False
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._notified == []
    # AC3's rail count lives on the SHELL input (built from
    # ``_local_source_counts`` in ``_build_library_shell_input``), which a
    # canvas-scoped sync deliberately skips (see ``_sync_library_canvas``'s
    # own docstring) -- only a full ``refresh(recompose=True)`` repaints
    # it, mirroring ``_delete_library_media_item``'s own tail.
    assert fake._refresh_calls == [{"recompose": True}]

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
    assert len(fake._notified) == 1
    message, kwargs = fake._notified[0]
    assert "1" in message
    assert kwargs.get("severity") == "warning"
    # A partial failure still touched the list/rail counts (one item DID
    # get deleted) -- the rail must still repaint, same as full success.
    assert fake._refresh_calls == [{"recompose": True}]

    db.close_connection()


@pytest.mark.asyncio
async def test_delete_selection_service_unavailable_keeps_selection_and_warns():
    """No ``media_reading_scope_service`` on the app (or no
    ``delete_media_item`` seam) must fail closed: nothing is removed from
    the list, the whole batch is reported as failed, and select mode is
    left active."""
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
        is_mounted=True,
        refresh=lambda **k: None,
        _run_library_service_call=LibraryScreen._run_library_service_call,
        _source_record_id=LibraryScreen._source_record_id,
    )
    fake._library_media_row_selection.select_all(["1"])
    fake._notify_library_media_delete_warning = types.MethodType(
        LibraryScreen._notify_library_media_delete_warning, fake
    )

    await LibraryScreen._delete_library_media_selection(fake, ("1",))

    assert len(fake._local_source_records["media"]) == 1
    assert fake._local_source_counts["media"] == 1
    assert fake._library_media_select_mode is True
    assert fake._library_media_confirming_bulk_delete is False
    assert fake._library_media_row_selection.ids == frozenset({"1"})
    assert len(fake._notified) == 1
    assert fake._notified[0][1].get("severity") == "warning"
