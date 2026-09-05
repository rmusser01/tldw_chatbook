"""Media browse replacements must not strand their entry-focus guard."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens import library_screen as screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback


@pytest.mark.parametrize("replacement_accepted", [True, False])
def test_replaced_entry_callback_releases_guard(monkeypatch, replacement_accepted):
    canvas = PostRecomposeCallback()
    entry_focus_calls = []
    control_focus_calls = []
    screen = SimpleNamespace(
        _library_selected_row_id=screen_module.LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_view="list",
        _library_media_browse_controller=SimpleNamespace(applied_scope=None),
        _build_library_media_state=lambda: SimpleNamespace(selected_id="media-1"),
        focused=None,
        _library_pending_list_entry_media_return=None,
        _library_pending_list_entry_focus=True,
        _library_media_return_candidate=lambda receipt: False,
        _library_list_entry_focus_generation=1,
        _library_notes_restoring_focus=False,
        _focus_library_list_entry_if_current=entry_focus_calls.append,
        _focus_library_control=control_focus_calls.append,
    )
    accepted = True

    def sync_canvas(_screen, kind, *, then):
        assert kind == "media"
        # Production queues only the latest intent; a suppressed projection
        # does not run any callback against the outgoing children.
        canvas.queue_after_recompose(then if accepted else None)
        return accepted

    monkeypatch.setattr(screen_module, "_sync_library_canvas", sync_canvas)
    LibraryScreen._sync_library_media_browse_state(screen, None)
    assert screen._library_notes_restoring_focus
    old_callback = canvas._post_recompose_callback

    screen._library_pending_list_entry_focus = False
    accepted = replacement_accepted
    LibraryScreen._sync_library_media_browse_state(screen, "#library-media-type-filter")
    assert not screen._library_notes_restoring_focus
    assert canvas._post_recompose_callback is not old_callback
    if canvas._post_recompose_callback is not None:
        canvas._post_recompose_callback()
    assert entry_focus_calls == []
    assert control_focus_calls == (
        ["#library-media-type-filter"] if replacement_accepted else []
    )


def test_suppressed_entry_sync_releases_guard(monkeypatch):
    screen = SimpleNamespace(
        _library_selected_row_id=screen_module.LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_view="list",
        _library_media_browse_controller=SimpleNamespace(applied_scope=None),
        _build_library_media_state=lambda: SimpleNamespace(selected_id="media-1"),
        focused=None,
        _library_pending_list_entry_media_return=None,
        _library_pending_list_entry_focus=True,
        _library_media_return_candidate=lambda receipt: False,
        _library_list_entry_focus_generation=1,
        _library_notes_restoring_focus=False,
    )
    monkeypatch.setattr(
        screen_module, "_sync_library_canvas", lambda *args, **kwargs: False
    )
    LibraryScreen._sync_library_media_browse_state(screen, None)
    assert not screen._library_notes_restoring_focus
