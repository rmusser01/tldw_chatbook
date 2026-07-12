from types import SimpleNamespace
import pytest
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_media_state import (
    LibraryMediaCanvasState,
    LibraryMediaRow,
    build_library_media_state,
)
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas


def _media_fake(select_mode):
    return SimpleNamespace(
        _library_media_select_mode=select_mode,
        _library_media_row_selection=RowSelection("media"),
        _opened=[],
        _refreshed=0,
        _viewer_opened=[],
    )


def test_row_press_in_select_mode_toggles_not_opens():
    fake = _media_fake(select_mode=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    fake._open_library_media_viewer = lambda mid: fake._viewer_opened.append(mid)
    event = SimpleNamespace(button=SimpleNamespace(media_id="7"), stop=lambda: None)
    LibraryScreen.handle_library_media_row(fake, event)
    assert fake._library_media_row_selection.is_selected("7")
    assert fake._viewer_opened == []          # viewer NOT opened
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
    async def _open(scope): fake._opened.append(scope)
    fake._open_library_export_canvas = _open
    event = SimpleNamespace(stop=lambda: None)
    await LibraryScreen.handle_library_media_export_selected(fake, event)
    assert fake._opened == [ExportScope(kind="media", ids=("1", "2", "3"))]


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
        empty_copy="No media in your Library yet. Ingest something to see it here.",
        selected_id="",
        preview_lines=(),
        count=len(rows),
        select_mode=True,
        selected_count=0,
    )


class _MediaCanvasApp(App):
    def compose(self):
        yield LibraryMediaCanvas(canvas=_select_mode_canvas_state(), id="library-media-canvas")


@pytest.mark.asyncio
async def test_canvas_select_mode_renders_action_row_and_disables_export():
    app = _MediaCanvasApp()
    async with app.run_test() as pilot:
        select_all_btn = pilot.app.query_one("#library-media-select-all", Button)
        assert select_all_btn is not None
        export_selected_btn = pilot.app.query_one("#library-media-export-selected", Button)
        assert export_selected_btn.disabled is True


def _filtered_select_mode_state() -> LibraryMediaCanvasState:
    # Two media types; filtering to "video" renders ONE row while
    # ``count`` (the pre-filter total across all types) stays at 3.
    records = [
        {"media_id": "1", "title": "A video", "type": "video", "last_modified": "2026-07-10T00:00:00Z"},
        {"media_id": "2", "title": "An audio", "type": "audio", "last_modified": "2026-07-10T00:00:00Z"},
        {"media_id": "3", "title": "More audio", "type": "audio", "last_modified": "2026-07-10T00:00:00Z"},
    ]
    return build_library_media_state(
        records,
        active_type="video",
        select_mode=True,
    )


class _FilteredMediaCanvasApp(App):
    def compose(self):
        yield LibraryMediaCanvas(canvas=_filtered_select_mode_state(), id="library-media-canvas")


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
