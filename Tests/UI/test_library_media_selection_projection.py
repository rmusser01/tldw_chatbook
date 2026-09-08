"""Local Reader identity survives Items projection and page arrival."""

from types import SimpleNamespace

import pytest

from Tests.UI.library_media_rows import summary_row
from tldw_chatbook.Library.library_media_reader_state import (
    begin_selection,
    enter_external_detail,
    settle_success,
)
from tldw_chatbook.Library.library_media_state import (
    MediaBrowseScope,
    build_media_browse_result,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _apply_page(screen, ids=(1, 2)):
    browse = screen._library_media_browse_controller
    browse.applied_result = build_media_browse_result(
        MediaBrowseScope(),
        {
            "items": [summary_row(id=identity) for identity in ids],
            "total": len(ids),
            "limit": 20,
            "offset": 0,
        },
    )
    browse.retained_items = browse.applied_result.items


def _select_reader(screen, identity=2, *, external=False):
    state = screen._media_state
    if external:
        state.reader_session = enter_external_detail(
            state.reader_session, identity, "Server target"
        )
    else:
        pending = begin_selection(
            state.reader_session, f"local:media:{identity}", identity, "Local target"
        )
        state.reader_session = settle_success(
            pending, pending.request_generation, pending.selected_id
        )


@pytest.mark.parametrize("view", ("viewer", "list"))
@pytest.mark.parametrize("external", (False, True), ids=("local", "external"))
def test_items_projection_uses_reader_only_for_a_local_viewer(view, external):
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    _apply_page(screen)
    _select_reader(screen, external=external)
    state = screen._media_state
    state.view = view
    state.selected_media_id = "local:media:1"
    session = state.reader_session

    projected = screen._media_controller._build_library_media_state()

    expected = "local:media:2" if view == "viewer" and not external else "local:media:1"
    assert projected.selected_id == expected
    assert [row.media_id for row in projected.rows if row.selected] == [expected]
    assert state.selected_media_id == "local:media:1"
    assert state.reader_session is session


def test_cold_page_projection_does_not_lose_the_reader_selection_on_arrival():
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    _select_reader(screen)
    state = screen._media_state
    state.view = "viewer"
    state.selected_media_id = "2"
    session = state.reader_session

    initial = screen._media_controller._build_library_media_state()
    assert initial.rows == ()
    assert initial.selected_id == ""
    # The existing screen projection writes the empty Items selection while
    # the initial page is pending; the Reader remains the selected authority.
    state.selected_media_id = initial.selected_id
    _apply_page(screen)
    arrived = screen._media_controller._build_library_media_state()

    assert arrived.selected_id == "local:media:2"
    assert [row.media_id for row in arrived.rows if row.selected] == ["local:media:2"]
    assert state.reader_session is session
    assert session.selected_id == session.loaded_id == "local:media:2"


def test_pending_reader_projection_selects_the_request_not_the_loaded_item():
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    _apply_page(screen)
    _select_reader(screen, identity=1)
    state = screen._media_state
    state.view = "viewer"
    state.selected_media_id = "local:media:1"
    state.reader_session = begin_selection(
        state.reader_session, "local:media:2", 2, "Next item"
    )
    session = state.reader_session

    projected = screen._media_controller._build_library_media_state()

    assert projected.selected_id == "local:media:2"
    assert [row.media_id for row in projected.rows if row.loading] == ["local:media:2"]
    assert [row.media_id for row in projected.rows if row.loaded] == ["local:media:1"]
    assert state.reader_session is session
    assert session.selected_id == "local:media:2"
    assert session.loaded_id == "local:media:1"


def test_local_viewer_without_a_reader_selection_keeps_the_list_anchor():
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    _apply_page(screen)
    state = screen._media_state
    state.view = "viewer"
    state.selected_media_id = "local:media:2"
    assert state.reader_session.selected_id is None

    projected = screen._media_controller._build_library_media_state()

    assert projected.selected_id == "local:media:2"
    assert state.reader_session.selected_id is None


@pytest.mark.parametrize("ids", ((), (1, 2)), ids=("empty-page", "different-page"))
def test_off_page_reader_keeps_the_existing_items_membership_fallback(ids):
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    _apply_page(screen, ids)
    _select_reader(screen, identity=3)
    state = screen._media_state
    state.view = "viewer"
    state.selected_media_id = "local:media:3"
    session = state.reader_session

    projected = screen._media_controller._build_library_media_state()

    expected = "local:media:1" if ids else ""
    assert projected.selected_id == expected
    assert [row.media_id for row in projected.rows if row.selected] == (
        [expected] if expected else []
    )
    assert state.reader_session is session
    assert session.selected_id == session.loaded_id == "local:media:3"
