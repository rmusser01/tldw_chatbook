"""TASK-34000.22 fix round 1: the slower siblings of
``test_library_media_editor_dirty_escape.py`` (in the census too).

The task review found that the first round's refusals were not the whole
story. A dirty authoring form could still be lost or stranded by:

- the Items filter: a typed query's select-first, a 0-result page's Reader
  reset and Clear's restore all moved or reset the Reader under the form
  (I-2) -- now the Reader's item is PINNED while a form is dirty: the list
  filters as usual, the Reader and its draft stay, the selection keeps
  naming the Reader's item (Task 1's hold paints no row when it is off the
  page), and the central guard at ``_select_library_media_reader_row``
  refuses every automatic move;
- a destination moved behind the draft (Export…, C-1): the Media rail
  press that brings the user back must never be refused and lands with
  the form and its text intact;
- the exits the first round added without pins (I-3): another row's press,
  the Up/Down walk, Tab across rows (silent backstop) and "‹ Back";
- a refusal for a form on the OTHER Reader tab switches to that tab (M-4);
- Escape over an armed delete confirm cancels the confirm first (M-5).

Real keys and clicks throughout; painted/mounted widgets and the selection
asserted; one boot per test.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.library_quit_guard_support import _record_toasts
from Tests.UI.test_library_media_editor_dirty_escape import (
    ROW_1,
    SIZE,
    _host,
    _open_analysis_form,
    _open_edit_form,
    _open_row_1,
    _type,
    _wait_for_text,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_INGEST_EXPORT,
)
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY,
    LIBRARY_MEDIA_EDIT_DIRTY_VETO_COPY,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

ROW_0 = "local:media:2"


def _rows(screen) -> list[Button]:
    return [
        button
        for button in screen.query("#library-media-canvas Button")
        if str(getattr(button, "id", "") or "").startswith("library-media-row-")
    ]


def _highlighted(screen) -> list[str]:
    return [
        str(row.media_id)
        for row in _rows(screen)
        if getattr(row, "_library_media_selected", False)
    ]


def _reader_holds_the_dirty_analysis(screen, area=None) -> None:
    """The Reader still shows item 1's dirty analysis form with its text.
    ``area`` pins the SAME mounted TextArea (nothing recomposed it); None
    where the gesture's own one-step recompose is expected (the armed
    delete confirm swaps the toolbar) and only the kept text matters."""
    state = screen._media_state
    assert state.view == "viewer"
    assert state.reader_session.loaded_id == ROW_1
    assert state.selected_media_id == ROW_1
    assert state.editing_analysis is True
    assert state.analysis_draft == "Key points"
    mounted = screen.query_one("#library-media-analysis-edit-text", TextArea)
    if area is not None:
        assert mounted is area
    assert mounted.text == "Key points"


async def _dirty_analysis(screen, pilot) -> TextArea:
    await _open_row_1(screen, pilot)
    area = await _open_analysis_form(screen, pilot)
    await _type(pilot, "Key points")
    await _wait_for_text(screen, pilot, "Key points")
    return area


async def _browse_settled(screen, pilot, *, query: str, rows: int) -> None:
    # ``query`` is matched by suffix: the first settled query recomposes the
    # Items pane, so the characters typed after it land in a fresh box.
    browse = screen._library_media_browse_controller
    await _wait_for_condition(
        pilot,
        lambda: browse.applied_scope is not None
        and browse.applied_scope.query.endswith(query)
        and not browse.loading
        and browse.applied_scope == browse.requested_scope
        and len(_rows(screen)) == rows,
        message=lambda: (
            f"query {query!r} never settled to {rows} rows: "
            f"{browse.applied_scope!r} rows={len(_rows(screen))}"
        ),
    )
    for _ in range(4):
        await pilot.pause()


async def test_a_media_rail_press_returns_to_a_dirty_form_behind_another_destination(
    monkeypatch,
):
    """C-1 ruling: whatever moved the destination off Media (here, the exact
    state the old Export… opener left behind), the Media rail press is never
    refused and lands on the Reader with the form and its text intact."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        area = await _dirty_analysis(screen, pilot)
        # The trapped state: the destination moved while the form stayed set.
        screen._set_library_destination_with_conversation_fence(LIBRARY_ROW_INGEST_EXPORT)
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        for _ in range(6):
            await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        assert toasts == [], toasts
        _reader_holds_the_dirty_analysis(screen, area)


async def test_the_items_filter_never_moves_the_reader_off_a_dirty_form(monkeypatch):
    """I-2: a typed query, a 0-result query and Clear -- real keys in the
    Items filter -- leave the Reader on the dirty form: the list filters, the
    Reader's item and text stay, the selection keeps naming the Reader's
    item, no row is highlighted while it is off the page, and nothing is
    said (nothing was refused: the filter did what it does)."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        area = await _dirty_analysis(screen, pilot)
        assert [str(r.media_id) for r in _rows(screen)] == [ROW_0, ROW_1]

        # A query that keeps ONLY the other item on the page.
        await pilot.click("#library-media-filter")
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == "library-media-filter",
            message="the Items filter never took focus",
        )
        await _type(pilot, "Product")
        await _browse_settled(screen, pilot, query="Product", rows=1)
        assert [str(r.media_id) for r in _rows(screen)] == [ROW_0]
        _reader_holds_the_dirty_analysis(screen, area)
        assert screen._media_state.highlight_held_by_reader is True
        assert _highlighted(screen) == [], _highlighted(screen)

        # A 0-result query: the Reader is not reset under the form.
        await _type(pilot, "zzz")
        await _browse_settled(screen, pilot, query="zzz", rows=0)
        _reader_holds_the_dirty_analysis(screen, area)

        # Clear: the restore does not move the Reader either; the full list
        # is back and the Reader's own row wears the cue again.
        await pilot.click("#library-media-filter-clear")
        await _browse_settled(screen, pilot, query="", rows=2)
        _reader_holds_the_dirty_analysis(screen, area)
        assert _highlighted(screen) == [ROW_1]
        assert toasts == [], toasts


async def test_row_press_keys_tab_and_back_refuse_or_keep_a_dirty_form(monkeypatch):
    """I-3 (100x30: the Library pane is collapsed, the Items list and the
    "‹ Back" control both render). Another row's press and an Up/Down walk
    onto another item each refuse with one toast; Shift+Tab onto the other
    row moves focus there and selects nothing, silently -- the ONE guard at
    ``_select_library_media_reader_row`` refuses the focus move's settle
    select (fix round 2 replaced the ``on_descendant_focus`` clause with it);
    Back refuses with one toast. The selection never leaves the Reader's
    item and the text stays. Mutation-checked: dropping that central guard
    lets the refused click's queued focus bubble select the other row."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        area = await _dirty_analysis(screen, pilot)
        rows = _rows(screen)
        assert [str(r.media_id) for r in rows] == [ROW_0, ROW_1]
        other = rows[0]

        await pilot.click(f"#{other.id}")
        for _ in range(8):
            await pilot.pause()
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY], toasts
        assert screen.focused is other
        _reader_holds_the_dirty_analysis(screen, area)

        # Down from row 0 lands on the Reader's own row: no refusal, focus
        # moves; Up from there targets the other item: refused, focus stays.
        await pilot.press("down")
        for _ in range(4):
            await pilot.pause()
        assert len(toasts) == 1, toasts
        assert screen.focused is rows[1]
        await pilot.press("up")
        for _ in range(4):
            await pilot.pause()
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY] * 2, toasts
        assert screen.focused is rows[1]
        _reader_holds_the_dirty_analysis(screen, area)

        # Shift+Tab onto the other row: focus lands there, the select seam's
        # guard refuses the settle select, nothing is said.
        await pilot.press("shift+tab")
        for _ in range(6):
            await pilot.pause()
        assert len(toasts) == 2, toasts
        assert screen.focused is rows[0], screen.focused
        _reader_holds_the_dirty_analysis(screen, area)

        back = await _wait_for_selector(screen, pilot, "#library-media-back")
        assert isinstance(back, Button)
        await pilot.click("#library-media-back")
        for _ in range(6):
            await pilot.pause()
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY] * 3, toasts
        _reader_holds_the_dirty_analysis(screen, area)


async def test_a_refusal_switches_to_the_tab_holding_the_draft(monkeypatch):
    """M-4: a dirty metadata draft (Info tab) refused from the Analysis tab
    brings the Info tab -- and the form with its text, Save and Discard
    changes -- back on screen with the toast."""
    host, app, _service = _host()
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
            == "Interview Recording!",
            message="the typed character never landed",
        )
        screen.query_one("#library-media-reader-select-analysis", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._media_state.reader_session.mode == "analysis"
            and not screen.query("#library-media-edit-title"),
            message="the Analysis tab never showed",
        )
        screen.set_focus(None)

        await pilot.press("escape")
        title = await _wait_for_selector(screen, pilot, "#library-media-edit-title")
        for _ in range(4):
            await pilot.pause()

        assert toasts == [LIBRARY_MEDIA_EDIT_DIRTY_VETO_COPY], toasts
        assert screen._media_state.reader_session.mode == "info"
        assert screen._media_state.editing is True
        assert isinstance(title, Input) and title.value == "Interview Recording!"
        assert str(screen.query_one("#library-media-edit-cancel", Button).label) == (
            "Discard changes"
        )


async def test_escape_cancels_an_armed_delete_before_the_dirty_veto(monkeypatch):
    """M-5 / R2-5: an armed "Move to trash" confirm over a form that then
    becomes dirty -- the chip reads ``esc cancel`` (what the key does), Escape
    cancels the confirm with no toast and the text stays; and once the form
    is dirty, Move to trash itself (the button) refuses with the one toast
    rather than arming a confirm (which a dirty metadata form would not
    even render)."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        area = await _open_analysis_form(screen, pilot)
        # Arm the confirm while the form is still clean ...
        screen.query_one("#library-media-reader-more", Button).press()
        delete = await _wait_for_selector(screen, pilot, "#library-media-delete")
        delete.press()
        await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")
        assert screen._media_state.confirming_delete is True
        # ... then make it dirty under the armed confirm.
        await pilot.click("#library-media-analysis-edit-text")
        await _type(pilot, "Key points")
        await _wait_for_text(screen, pilot, "Key points")
        assert screen._library_media_editor_dirty() == "analysis"
        assert ("esc", "cancel") in screen._footer_shortcut_registration[1], (
            screen._footer_shortcut_registration
        )
        screen.set_focus(None)

        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.confirming_delete
            and not screen.query("#library-media-delete-confirm"),
            message="Escape never cancelled the armed delete",
        )
        for _ in range(4):
            await pilot.pause()
        assert toasts == [], toasts
        _reader_holds_the_dirty_analysis(screen)

        # R2-5: Move to trash on a dirty form is refused, not armed.
        if not screen.query("#library-media-delete"):
            screen.query_one("#library-media-reader-more", Button).press()
        delete = await _wait_for_selector(screen, pilot, "#library-media-delete")
        delete.press()
        await _wait_for_condition(pilot, lambda: bool(toasts), message="no refusal")
        for _ in range(4):
            await pilot.pause()
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY], toasts
        assert screen._media_state.confirming_delete is False
        assert not screen.query("#library-media-delete-confirm")
        _reader_holds_the_dirty_analysis(screen)
