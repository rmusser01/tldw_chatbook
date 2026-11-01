"""TASK-34000.18 final review (C-1): the Reader's controls act on the item
the Reader SHOWS even while the Media canvas is in its LIST view.

In the three-pane shell (160x45) the Reader keeps painting, with live
controls, after Trash › "‹ Media" puts the canvas back in the list view.
In that view the list highlight is free to move -- a type filter that drops
the Reader's item moves it to row 0 (TASK-15457), and the adopt seam used
to do the same under a DIRTY form -- while the Reader still shows the item
the user opened. ``reader_action_target`` used to answer with the list
selection whenever the view was not ``"viewer"``, so:

1. a dirty Edit metadata form on item 1, a filter that drops item 1, Trash
   and "‹ Media", then Save wrote item 1's form onto item 2 -- silently;
2. with no dirty form, item 1 open, Trash and "‹ Media", then the type
   chooser set to video: Edit metadata prefilled item 1 but Save wrote
   item 2;
3. in that same state, Move to trash confirmed ``Move "Interview
   Recording" to Trash?`` and deleted item 2.

Now the target is the loaded item whatever the view, the trash confirm's
title derives from that same identity, and the adopt seam keeps the
selection on the pinned item while a form is dirty in the list view too.
The TASK-15457 pin (``test_library_canvas_sync_defects.py``) is unchanged:
with no dirty form the list view's selection still follows the canvas.

Real app stylesheet (``LibraryProductionCSSHarness``), the production-shaped
exact-page fake (every write recorded), real keys, clicks and native Paste
filter preparation, painted text through the compositor. One boot per test;
three boots. Slow filter typing is separately tracked in TASK-37360.
"""

from __future__ import annotations

import pytest
from textual.events import Paste
from textual.widgets import Button, Input, OptionList

from Tests.UI.library_quit_guard_support import _record_toasts
from Tests.UI.test_library_media_editor_dirty_escape import (
    ROW_1,
    SIZE,
    _host,
    _open_edit_form,
    _open_row_1,
    _painted,
)
from Tests.UI.test_library_media_editor_dirty_escape_extended import (
    ROW_0,
    _browse_settled,
    _highlighted,
    _rows,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: Item 1's backing id as the fake records it (``update_media_item`` /
#: ``delete_media_item`` receive the resolved int).
ITEM_1 = 1
ITEM_1_TITLE = "Interview Recording"


async def _trash_round_trip(screen, pilot) -> None:
    """Trash, then "‹ Media": the canvas is back in the LIST view while the
    three-pane shell still paints the Reader."""
    screen.query_one("#library-media-trash-open", Button).press()
    await _wait_for_condition(
        pilot, lambda: screen._media_state.view == "trash", message="Trash never opened"
    )
    back = await _wait_for_selector(screen, pilot, "#library-media-trash-back")
    back.press()
    await _wait_for_condition(
        pilot, lambda: screen._media_state.view == "list", message="Trash never closed"
    )
    for _ in range(8):
        await pilot.pause()
    assert screen._media_state.view == "list"
    assert screen._media_state.reader_session.loaded_id == ROW_1


async def _filter_type_to_video(screen, pilot) -> None:
    """The type chooser set to video: item 1 (audio) leaves the page and the
    TASK-15457 fallback moves the list selection to row 0 (item 2)."""
    screen.query_one("#library-media-type-filter", Button).press()
    chooser = await _wait_for_selector(screen, pilot, "#library-media-type-choices")
    assert isinstance(chooser, OptionList)
    chooser.highlighted = next(
        index
        for index, option in enumerate(chooser.options)
        if getattr(option, "choice_value", None) == "video"
    )
    chooser.action_select()
    await _wait_for_condition(
        pilot,
        lambda: [str(row.media_id) for row in _rows(screen)] == [ROW_0],
        message=lambda: f"type filter never settled: {[str(r.media_id) for r in _rows(screen)]}",
    )
    for _ in range(6):
        await pilot.pause()
    # The pinned TASK-15457 contract, unchanged: no dirty form, so the list
    # moved on and the selection followed the canvas to row 0 ...
    assert screen._media_state.selected_media_id == ROW_0
    assert _highlighted(screen) == [ROW_0]
    # ... while the Reader still shows item 1.
    assert screen._media_state.reader_session.loaded_id == ROW_1
    title = screen.query_one("#library-media-viewer-title")
    assert ITEM_1_TITLE in _painted(screen.app, title.region)


async def _focus_filter_box(screen, pilot) -> None:
    await pilot.click("#library-media-filter")
    await _wait_for_condition(
        pilot,
        lambda: getattr(screen.focused, "id", None) == "library-media-filter",
        message=lambda: f"the Items filter never took focus: {screen.focused!r}",
    )


async def _save_edit_form(screen, pilot, service) -> None:
    await pilot.press("end")
    await pilot.press("exclamation_mark")
    await _wait_for_condition(
        pilot,
        lambda: screen.query_one("#library-media-edit-title", Input).value
        == f"{ITEM_1_TITLE}!",
        message="typed char never landed",
    )
    screen.query_one("#library-media-edit-save", Button).press()
    await _wait_for_condition(
        pilot, lambda: bool(service.update_calls), message="Save never wrote"
    )
    for _ in range(6):
        await pilot.pause()


async def test_a_dirty_form_filtered_off_the_page_saves_to_the_reader_item_after_trash_back(
    monkeypatch,
):
    """Route 1: the dirty form is pinned to item 1 through the filter and the
    Trash round trip; in the list view the selection keeps naming item 1
    (no row wears the cue: item 1 is off the page) and Save writes item 1,
    with no toast. Before the fix the adopt seam moved the selection to row
    0 in the list view and Save wrote ``{'media_id': 2, ...}``."""
    host, app, service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        await _open_edit_form(screen, pilot)
        await pilot.press("end")
        await pilot.press("exclamation_mark")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-media-edit-title", Input).value
            == f"{ITEM_1_TITLE}!",
            message="typed char never landed",
        )
        # Prepare a query keeping ONLY the other item through the focused
        # field's native Paste event. Slow keys across filter recomposition
        # can replace the selected prefix (TASK-37360); this case pins the
        # Reader's write identity after the actual filtered browse settles.
        await _focus_filter_box(screen, pilot)
        pilot.app.post_message(Paste("Product"))
        await _browse_settled(screen, pilot, query="Product", rows=1)
        assert [str(row.media_id) for row in _rows(screen)] == [ROW_0]

        await _trash_round_trip(screen, pilot)

        # The pinned item holds the selection in the list view too, and the
        # off-page item gets no cue (owner Q2 (b)).
        assert screen._media_state.selected_media_id == ROW_1
        assert _highlighted(screen) == []
        assert screen._media_state.editing is True
        assert (
            screen.query_one("#library-media-edit-title", Input).value
            == f"{ITEM_1_TITLE}!"
        )

        screen.query_one("#library-media-edit-save", Button).press()
        await _wait_for_condition(
            pilot, lambda: bool(service.update_calls), message="Save never wrote"
        )
        for _ in range(6):
            await pilot.pause()

        assert [call["media_id"] for call in service.update_calls] == [ITEM_1], (
            service.update_calls
        )
        assert service.update_calls[-1]["title"] == f"{ITEM_1_TITLE}!"
        assert toasts == [], toasts


async def test_edit_metadata_save_in_the_list_view_writes_the_reader_item_not_row_0(
    monkeypatch,
):
    """Route 2: no dirty form at the scope change. Item 1 open, Trash and
    "‹ Media", the type chooser set to video (row 0 = item 2 is highlighted,
    the Reader shows item 1): Edit metadata prefills item 1 and Save writes
    item 1. Before the fix ``update_calls == [{'media_id': 2, ...}]``."""
    host, app, service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        await _trash_round_trip(screen, pilot)
        await _filter_type_to_video(screen, pilot)

        field = await _open_edit_form(screen, pilot)
        assert field.value == ITEM_1_TITLE, field.value
        await _save_edit_form(screen, pilot, service)

        assert [call["media_id"] for call in service.update_calls] == [ITEM_1], (
            service.update_calls
        )
        assert service.update_calls[-1]["title"] == f"{ITEM_1_TITLE}!"
        assert toasts == [], toasts


async def test_move_to_trash_in_the_list_view_names_and_deletes_the_reader_item(
    monkeypatch,
):
    """Route 3: the same state as route 2; More ▸ Move to trash paints
    ``Move "Interview Recording" to Trash?`` and Delete trashes item 1 --
    the confirm and the write name ONE record. Before the fix the confirm
    named item 1 and ``delete_calls == [{'media_id': 2}]``."""
    host, app, service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        await _trash_round_trip(screen, pilot)
        await _filter_type_to_video(screen, pilot)

        if not screen.query("#library-media-delete"):
            screen.query_one("#library-media-reader-more", Button).press()
        delete = await _wait_for_selector(screen, pilot, "#library-media-delete")
        delete.press()
        confirm = await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")
        for _ in range(4):
            await pilot.pause()
        viewer = screen.query_one("#library-media-viewer")
        painted = " ".join(_painted(host, viewer.region).split())
        assert f'Move "{ITEM_1_TITLE}" to Trash?' in painted, painted

        confirm.press()
        await _wait_for_condition(
            pilot, lambda: bool(service.delete_calls), message="Delete never wrote"
        )
        for _ in range(6):
            await pilot.pause()

        assert [call["media_id"] for call in service.delete_calls] == [ITEM_1], (
            service.delete_calls
        )
        assert toasts == [], toasts
