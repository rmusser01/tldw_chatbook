"""TASK-34000.25, the slower arms and the negative controls (not PR-gated).

The PR-gated core is ``test_library_rail_switch_keeps_reader_and_note.py``;
this file runs with the full ``Tests/UI`` suite. Each test is one real
``LibraryHarness`` boot over a real ChaChaNotes database (see the core
file's docstring for the base-commit failure story).

Covered here:

- the Info tab survives the round trip too (AC#2's "tab");
- a REAL external edit still produces the conflict, the conflict-state note
  is NOT retained (the rail press is vetoed) and keeps its Overwrite / Reload
  choices (AC#3's "unless");
- ``n`` inside the Find input types an ``n`` and creates nothing; the footer
  advertises ``n note`` only in a settled local Reader; a server detail
  refuses the key with the button's reason (AC#4's gate);
- an untouched blank is still garbage-collected by the round trip, and a
  vetoed title is not retained and shows the veto (AC#1's "existing flush
  and veto rules");
- ‹ Back still returns to the list (at 100x30, where the control renders),
  and Conversations ▸ Media with nothing loaded shows the list.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.library_quit_guard_support import (
    _armed_editor,
    _new_blank_note,
    _scaled_autosave,
    _type,
    _type_at_end,
    _until,
)
from Tests.UI.test_library_media_reader_flow import _reader_key_fake
from Tests.UI.test_library_rail_switch_keeps_reader_and_note import (
    LOADED_ID,
    _assert_editor_surfaces_painted,
    MEDIA_ROW,
    NOTE_BODY,
    NOTE_TITLE,
    SCROLL_Y,
    SIZE,
    VIEWER_TITLE,
    _host,
    _library,
    _loaded_rows,
    _open_media_item,
    _rail,
    _reader_scroll_y,
    _saved,
    _scroll_reader,
    _select_reader_mode,
    _session,
)
from Tests.UI.test_library_shell import _wait_for_selector
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
)
from tldw_chatbook.Widgets.Library.library_media_viewer import (
    MEDIA_TAKE_NOTE_EXTERNAL_REASON,
)

COMPACT = (100, 30)


def _footer_keys(screen) -> list[str]:
    return [key for key, _label in screen._library_route_shortcuts_for_current_state()]


# --- AC#2: the Info tab comes back too -----------------------------------------


@pytest.mark.asyncio
async def test_media_item_left_on_the_info_tab_comes_back_on_info(tmp_path):
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_media_item(screen, pilot)
        await _select_reader_mode(screen, pilot, "info")

        await _rail(screen, pilot, "notes")
        await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
        await _rail(screen, pilot, "media")

        assert screen.query(VIEWER_TITLE)
        assert screen._media_state.reader_session.loaded_id == LOADED_ID
        assert screen._media_state.reader_session.mode == "info"
        assert screen.query("#library-media-reader-select-info")
        assert _loaded_rows(screen) == [MEDIA_ROW.lstrip("#")]
    profile.db.close_connection()


# --- AC#3: a real external edit still conflicts, and that note is not retained


@pytest.mark.asyncio
async def test_a_real_external_edit_still_conflicts_and_the_conflict_note_is_not_retained(
    tmp_path, monkeypatch
):
    _scaled_autosave(monkeypatch)
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _new_blank_note(screen, pilot)
        note_id = _session(screen).note_id
        await _type_at_end(pilot, screen.query_one(NOTE_BODY, TextArea), "alpha")
        await _until(pilot, lambda: _saved(screen), "the first autosave")

        # The rule is unchanged: a write the session did not make bumps the
        # row, and the next save of OUR text reports the conflict.
        row = profile.note(note_id)
        assert profile.db.update_note(
            note_id, {"content": "edited elsewhere"}, expected_version=row["version"]
        )
        assert profile.note(note_id)["version"] == row["version"] + 1
        await _type_at_end(pilot, screen.query_one(NOTE_BODY, TextArea), " beta")
        await _until(
            pilot,
            lambda: _session(screen).in_conflict,
            "the conflict after the external edit",
        )
        region = screen.query_one("#library-note-conflict-region")
        assert region.display is True
        assert screen.query_one("#library-note-conflict-overwrite", Button)
        assert screen.query_one("#library-note-conflict-reload", Button)
        assert "Conflict" in screen._notes_controller._library_note_status_line()
        assert "changed elsewhere" in str(
            screen.query_one("#library-note-conflict-copy").renderable
        )

        # A conflict-state note is NOT retained: the rail press takes the
        # flush branch, which is vetoed, so the user stays with the choices.
        await _rail(screen, pilot, "media")
        await pilot.pause()
        assert screen._library_selected_row_id == "browse-notes"
        assert screen.query(NOTE_BODY)
        assert _session(screen).note_id == note_id
        assert _session(screen).in_conflict
        assert screen.query_one("#library-note-conflict-overwrite", Button)
        assert screen.query_one("#library-note-conflict-reload", Button)
        assert screen.query_one(NOTE_BODY, TextArea).text == "alpha beta"
        # The DB keeps the external text until the user chooses.
        assert profile.note(note_id)["content"] == "edited elsewhere"
    profile.db.close_connection()


# --- AC#4: the gate ---------------------------------------------------------


@pytest.mark.asyncio
async def test_n_inside_the_find_input_types_and_the_footer_drops_the_chip(tmp_path):
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        body = await _open_media_item(screen, pilot)
        screen.set_focus(body.scroller)
        await pilot.pause()
        assert "n" in _footer_keys(screen), _footer_keys(screen)
        assert screen.check_action("library_media_take_note", ()) is True
        rows_before = len(profile.rows())

        await pilot.press("ctrl+f")
        find = await _wait_for_selector(screen, pilot, "#library-media-content-search")
        await _until(pilot, lambda: screen.focused is find, "the Find input to take focus")
        assert screen.check_action("library_media_take_note", ()) is False
        assert "n" not in _footer_keys(screen), _footer_keys(screen)

        await pilot.press("n")
        await pilot.pause()
        assert screen.query_one("#library-media-content-search", Input).value == "n"
        assert screen._library_selected_row_id == "browse-media"
        assert screen.query(VIEWER_TITLE)
        assert len(profile.rows()) == rows_before

        # And the Items filter box, the other text field beside the Reader.
        screen.query_one("#library-media-filter", Input).focus()
        await pilot.pause()
        assert screen.check_action("library_media_take_note", ()) is False
        await pilot.press("n")
        await pilot.pause()
        assert screen.query_one("#library-media-filter", Input).value == "n"
        assert len(profile.rows()) == rows_before
    profile.db.close_connection()


def test_take_note_gate_mirrors_the_reader_action_keys():
    """Unit: the same fences as l/c/t, plus local-only and not-typing."""
    plain = _reader_key_fake()
    plain.focused = None
    assert LibraryScreen.check_action(plain, "library_media_take_note", ()) is True
    for fake in (
        _reader_key_fake(view="list"),
        _reader_key_fake(substate=True),
        _reader_key_fake(pending=True),
        _reader_key_fake(external=True),
    ):
        fake.focused = None
        assert LibraryScreen.check_action(fake, "library_media_take_note", ()) is False
    typing = _reader_key_fake()
    typing.focused = Input()
    assert LibraryScreen.check_action(typing, "library_media_take_note", ()) is False


@pytest.mark.asyncio
async def test_the_note_button_is_disabled_with_its_reason_for_a_server_detail():
    """The viewer composes ``○ Note`` disabled with the one reason string.

    Composed directly, as ``test_library_crit10_viewer`` does for Find: a
    server detail is a round trip the Library harness has no fixture for.
    """
    from Tests.UI.test_library_crit10_viewer import _media_host
    from tldw_chatbook.Library.library_media_viewer_state import (
        build_library_media_viewer_state,
    )
    from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer

    state = build_library_media_viewer_state(
        {
            "media_id": "server:1",
            "title": "Server item",
            "type": "article",
            "content": "text",
        }
    )
    external = LibraryMediaViewer(state, external_detail=True)
    local = LibraryMediaViewer(state, external_detail=False)
    async with _media_host().run_test(size=(235, 52)) as pilot:
        await pilot.app.screen.mount(external)
        await pilot.pause()
        note = external.query_one("#library-media-take-note", Button)
        assert note.disabled is True
        assert note.tooltip == MEDIA_TAKE_NOTE_EXTERNAL_REASON
        assert str(note.label).startswith("○")
        await external.remove()
        await pilot.app.screen.mount(local)
        await pilot.pause()
        live = local.query_one("#library-media-take-note", Button)
        assert live.disabled is False
        assert live.tooltip == "Take a note from this document (n)"
        assert str(live.label) == "Note"


# --- AC#1: the existing flush and veto rules ---------------------------------


@pytest.mark.asyncio
async def test_an_untouched_blank_is_still_garbage_collected_by_the_round_trip(
    tmp_path,
):
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        rows_before = profile.rows()
        await _new_blank_note(screen, pilot)
        await _rail(screen, pilot, "media")
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        await _rail(screen, pilot, "notes")
        await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")

        assert not screen.query(NOTE_BODY), "the untouched blank was retained"
        assert screen._notes_state.view == "list"
        assert profile.rows() == rows_before, "the untouched blank survived in the DB"
    profile.db.close_connection()


@pytest.mark.asyncio
async def test_a_vetoed_title_follows_the_rule_a_list_note_already_has(
    tmp_path, monkeypatch
):
    """AC#1: the New-note note follows the LIST note's existing rules.

    Across the retained reader rows a dirty edit is carried, not flushed --
    that is how a note opened from the list has always behaved on this tree
    -- and the re-armed autosave reports the veto at the editor on return,
    moving no focus; a REAL exit (Import) is vetoed and the user stays with
    the status. Nothing invalid ever reaches the database.
    """
    _scaled_autosave(monkeypatch)
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _new_blank_note(screen, pilot)
        note_id = _session(screen).note_id
        await _type_at_end(pilot, screen.query_one(NOTE_BODY, TextArea), "body")
        await _until(pilot, lambda: _saved(screen), "the first autosave")
        saved_version = profile.note(note_id)["version"]
        screen.query_one(NOTE_TITLE, Input).focus()
        await pilot.pause()
        # A trailing space is the title veto ("begins or ends with whitespace").
        await _type(pilot, "Draft ")
        assert _session(screen).dirty

        await _rail(screen, pilot, "media")
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        await _rail(screen, pilot, "notes")
        await _until(
            pilot, lambda: bool(screen.query(NOTE_BODY)), "the editor to be retained"
        )
        assert _session(screen).note_id == note_id
        # The autosaved seed ("Untitled") is already in the Input; the typed
        # tail with its trailing space is what the veto is about.
        assert screen.query_one(NOTE_TITLE, Input).value.endswith("Draft ")
        await _until(
            pilot,
            lambda: screen._notes_state.autosave_state == "validation",
            "the re-armed autosave to report the veto",
        )
        assert "whitespace" in screen._notes_controller._library_note_status_line()
        assert profile.note(note_id)["version"] == saved_version
        assert profile.note(note_id)["title"] == "Untitled"

        # A real exit is still vetoed: the user stays with the status.
        screen.query_one("#library-row-ingest-import-media", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen._library_selected_row_id == "browse-notes"
        assert screen.query(NOTE_BODY)
        assert screen._notes_state.autosave_state == "validation"
        assert profile.note(note_id)["version"] == saved_version
    profile.db.close_connection()


# --- ‹ Back, and a Media entry with nothing loaded ----------------------------


@pytest.mark.asyncio
async def test_back_in_the_reader_still_returns_to_the_list(tmp_path):
    host, profile = _host(tmp_path)
    async with host.run_test(size=COMPACT) as pilot:
        screen = await _library(host, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        row = await _wait_for_selector(screen, pilot, MEDIA_ROW)
        row.press()
        await _wait_for_selector(screen, pilot, VIEWER_TITLE)
        await _until(
            pilot,
            lambda: screen._media_state.reader_session.pending_request is None
            and screen._media_state.reader_session.loaded_id == LOADED_ID,
            "the Reader to settle",
        )
        back = await _wait_for_selector(screen, pilot, "#library-media-back")
        back.press()
        await _until(
            pilot, lambda: screen._media_state.view == "list", "Back to return to the list"
        )
        await pilot.pause()
        assert screen.query_one("#library-media-list")
        # The rail round trip after Back is a fresh entry: nothing to restore.
        await _rail(screen, pilot, "notes")
        await _rail(screen, pilot, "media")
        assert screen._media_state.view == "list"
    profile.db.close_connection()


@pytest.mark.asyncio
async def test_conversations_to_media_with_nothing_loaded_shows_the_list(tmp_path):
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _rail(screen, pilot, "conversations")
        await pilot.pause()
        await _rail(screen, pilot, "media")
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        assert screen._media_state.view == "list"
        assert screen._media_state.reader_session.loaded_id is None
        assert _loaded_rows(screen) == []
    profile.db.close_connection()


# --- the n note's own round trip at the narrow review size --------------------


@pytest.mark.asyncio
async def test_n_note_round_trip_at_120x36_restores_the_scroll(tmp_path):
    host, profile = _host(tmp_path)
    async with host.run_test(size=(120, 36)) as pilot:
        screen = await _library(host, pilot)
        body = await _open_media_item(screen, pilot)
        await _scroll_reader(screen, pilot, body, SCROLL_Y)
        screen.set_focus(body.scroller)
        await pilot.pause()
        await pilot.press("n")
        await _armed_editor(screen, pilot)
        await _rail(screen, pilot, "media")
        assert screen._media_state.reader_session.loaded_id == LOADED_ID
        await _until(
            pilot,
            lambda: _reader_scroll_y(screen) == SCROLL_Y,
            f"the reading position to be restored (now {_reader_scroll_y(screen)})",
            timeout=10.0,
        )
        assert isinstance(
            screen.query_one("#library-media-viewer-content", LibraryMediaContentBody),
            LibraryMediaContentBody,
        )
    profile.db.close_connection()


# --- fix round 1: the retained editor keeps the mode it was left on ---------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("button", "mode"),
    [("#library-note-preview", "preview"), ("#library-note-context", "context")],
)
async def test_a_retained_note_comes_back_on_the_mode_it_was_left_on(
    tmp_path, monkeypatch, button, mode
):
    """Preview and Info survive the round trip with exactly their surfaces
    painted in the first frame (the gated file pins Edit)."""
    _scaled_autosave(monkeypatch)
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _new_blank_note(screen, pilot)
        await _type_at_end(pilot, screen.query_one(NOTE_BODY, TextArea), "s02 paper notes")
        await _until(pilot, lambda: _saved(screen), "the autosave to land")
        screen.query_one(button, Button).press()
        await _until(
            pilot,
            lambda: _painted_region(screen, mode),
            f"the {mode} region to paint",
        )
        _assert_editor_surfaces_painted(screen, mode)

        await _rail(screen, pilot, "media")
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        await _rail(screen, pilot, "notes")
        await _until(
            pilot,
            lambda: bool(screen.query("#library-note-title")),
            "the note editor to be open again after the round trip",
            timeout=5.0,
        )
        await pilot.pause()
        _assert_editor_surfaces_painted(screen, mode)
    profile.db.close_connection()


def _painted_region(screen, mode: str) -> bool:
    selector = {"preview": "#library-note-preview-region", "context": "#library-note-context-region"}[mode]
    try:
        widget = screen.query_one(selector)
    except Exception:  # noqa: BLE001 -- not composed yet
        return False
    return widget in screen._compositor.visible_widgets and widget.region.height > 0


# --- fix round 1: the primary toolbar with five buttons ----------------------


def _primary_buttons(reader):
    return [
        reader.query_one(selector, Button)
        for selector in (
            "#library-media-reader-find",
            "#library-media-take-note",
            "#library-media-read-later",
            "#library-media-use-in-chat",
            "#library-media-reader-more",
        )
    ]


@pytest.mark.asyncio
async def test_more_open_cue_fits_the_narrowest_reader_and_wide_readers_keep_one_row(
    tmp_path,
):
    """100x30: the five actions stack into two rows so ``More ▴`` is whole
    inside the Reader; 120x36 and 160x45: one row, as before the fifth
    button (the threshold is derived from the labels, not a size)."""
    from Tests.UI.test_library_media_reader_shell import _painted_text_in_region

    for size, stacked in (((100, 30), True), ((120, 36), False), ((160, 45), False)):
        size_dir = tmp_path / f"{size[0]}x{size[1]}"
        size_dir.mkdir()
        host, profile = _host(size_dir)
        async with host.run_test(size=size) as pilot:
            screen = await _library(host, pilot)
            await _open_media_item(screen, pilot)
            reader = screen.query_one("#library-media-viewer")
            buttons = _primary_buttons(reader)
            rows = {button.region.y for button in buttons}
            assert (len(rows) == 2) is stacked, (size, [(b.id, b.region) for b in buttons])
            for button in buttons:
                assert reader.content_region.contains_region(button.region), (size, button.id, button.region)
            screen.query_one("#library-media-reader-more", Button).press()
            await _until(
                pilot,
                lambda: "\u25b4" in str(screen.query_one("#library-media-reader-more", Button).label),
                "More to open",
            )
            await pilot.pause()
            more = screen.query_one("#library-media-reader-more", Button)
            assert reader.content_region.contains_region(more.region), (size, more.region, reader.content_region)
            painted = _painted_text_in_region(host, more.region)
            assert "More \u25b4" in painted, (size, painted)
            assert screen.query_one("#library-media-reader-more-actions")
        profile.db.close_connection()
