"""The explicit paths still route to the vetoed field (TASK-34000.10, AC#3), and
the fix-round-1 pins: terminal focus loss never tidies (C-1), the compact mark
(I-3), a steady hint row (M-5), the mark cleared on discard and switch (M-6),
the Info twin tidy (M-7).

Companion to ``test_library_note_autosave_veto_in_place.py`` (the gated core);
not in the PR gate census. Both arms are green on base: the explicit Save and
the refused exit are the paths TASK-34000.10 leaves routing and focusing.
They are pinned here so the focus-independent autosave guard can never be
"fixed" by removing the explicit routing too.
"""

from __future__ import annotations

import asyncio

import pytest
from textual import events as events_module
from textual.widgets import Input, Static

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _OTHER_TITLE,
    _TITLE,
    SIZE,
    _library,
    _library_app,
    _new_blank_note,
    _open_note,
    _scaled_autosave,
    _settings_without_splash,
    _type,
    _until,
)
from Tests.UI.test_library_note_autosave_veto_in_place import (
    _INVALID,
    _open_info,
    _set_long_title,
    _vetoed,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


# --- AC#3: an explicit Save, or leaving the note, still routes ---------------


async def test_an_explicit_save_still_routes_to_the_vetoed_field(
    tmp_path, monkeypatch
):
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _open_note(screen, pilot, _TITLE, profile.note_id)
                _set_long_title(screen)
                await _open_info(pilot, screen)
                await _vetoed(pilot, screen)

                await pilot.click("#library-note-save")
                await _until(
                    pilot,
                    lambda: getattr(screen.focused, "id", None) == "library-note-title",
                    "Save to route to the vetoed title",
                )
                assert screen._notes_state.context is False
                assert screen.query_one("#library-note-editor-region").display
                assert screen._notes_state.autosave_state == "validation"
                assert screen.query_one("#library-note-title").has_class(_INVALID)
                assert profile.note(profile.note_id)["title"] == _TITLE
    finally:
        profile.db.close_connection()
    assert events == []


async def test_leaving_the_note_with_a_veto_standing_still_routes(
    tmp_path, monkeypatch
):
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _open_note(screen, pilot, _TITLE, profile.note_id)
                _set_long_title(screen)
                await _open_info(pilot, screen)
                await _vetoed(pilot, screen)

                # Escape from Info closes Info (focus to the Info button); that
                # is not an exit, and the standing autosave veto still routes
                # nothing. The second Escape is the attempt to leave the note.
                await pilot.press("escape")
                await _until(
                    pilot,
                    lambda: screen._notes_state.context is False
                    and getattr(screen.focused, "id", None) == "library-note-context",
                    "Escape to close Info",
                )
                await pilot.pause(0.3)
                assert getattr(screen.focused, "id", None) == "library-note-context"
                assert screen._notes_state.autosave_state == "validation"

                await pilot.press("escape")
                await _until(
                    pilot,
                    lambda: getattr(screen.focused, "id", None) == "library-note-title",
                    "Escape's refused exit to route to the vetoed title",
                )
                assert screen._notes_state.view == "editor", "Escape left the note"
                assert screen._notes_state.context is False
                assert screen.query_one("#library-note-editor-region").display
                assert profile.note(profile.note_id)["title"] == _TITLE
    finally:
        profile.db.close_connection()
    assert events == []


# --- Fix round 1 ---------------------------------------------------------------

_LONG_TITLE = "x" * 301
_DUPLICATE_VETO = "Keywords contain a case-insensitive duplicate — remove one to save."


def _painted_text_in_region(app, region) -> str:
    strips = list(app.screen._compositor.render_strips())
    rows = (
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, region.bottom)
    )
    return " ".join(" ".join(rows).split())


def _painted_color_of(app, widget, text: str):
    """The foreground colour the compositor painted ``text`` with, inside ``widget``."""
    strip = list(app.screen._compositor.render_strips())[widget.region.y]
    strip = strip.crop(widget.region.x, widget.region.right)
    for segment in strip:
        if text in segment.text:
            return segment.style.color
    raise AssertionError(f"{text!r} is not painted inside {widget.id}: {strip.text!r}")


async def test_losing_terminal_focus_mid_word_does_not_tidy_the_field(
    tmp_path, monkeypatch
):
    """C-1: Textual clears focus when the TERMINAL loses focus (``AppBlur`` ->
    ``set_focus(None)``), which blurs the Input too. The tidy must run only
    for a focus move the user made inside the app: Cmd-Tab mid-word and back
    used to glue the next keys on ("Meeting withAlice", "okx").
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _new_blank_note(screen, pilot)
                title = screen.query_one("#library-note-title", Input)
                keywords = screen.query_one("#library-note-keywords", Input)
                await _until(pilot, lambda: screen.focused is title, "the title caret")

                await _type(pilot, "Meeting with ")
                app.post_message(events_module.AppBlur())
                await _until(pilot, lambda: screen.focused is None, "AppBlur to clear focus")
                assert app.app_focus is False
                app.post_message(events_module.AppFocus())
                await _until(pilot, lambda: screen.focused is title, "AppFocus to restore focus")
                assert title.value == "Meeting with ", "losing the terminal tidied the title"
                await _type(pilot, "Alice")
                assert title.value == "Meeting with Alice"

                await pilot.press("tab")
                await _until(pilot, lambda: screen.focused is keywords, "Tab to keywords")
                await _type(pilot, "ok, ")
                app.post_message(events_module.AppBlur())
                await _until(pilot, lambda: screen.focused is None, "AppBlur to clear focus")
                app.post_message(events_module.AppFocus())
                await _until(pilot, lambda: screen.focused is keywords, "AppFocus to restore focus")
                assert keywords.value == "ok, ", "losing the terminal tidied the keywords"
                await _type(pilot, "x")
                assert keywords.value == "ok, x"
    finally:
        profile.db.close_connection()
    assert events == []


async def test_the_compact_mark_is_painted_without_a_border_at_80x24(
    tmp_path, monkeypatch
):
    """I-3: under 120 columns the editor Inputs have no border, so the outline
    could not show. The compact mark tints the field's text with the readable
    error colour; pinned by the painted cells, not the class alone.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=(80, 24)) as pilot:
                screen = await _library(app, pilot)
                await _new_blank_note(screen, pilot)
                title = screen.query_one("#library-note-title", Input)
                keywords = screen.query_one("#library-note-keywords", Input)
                await _until(pilot, lambda: screen.focused is title, "the title caret")
                await pilot.press("tab")
                await _until(pilot, lambda: screen.focused is keywords, "Tab to keywords")
                await _type(pilot, "ok, x, X")
                await pilot.pause()
                before = _painted_color_of(app, keywords, "ok, x, X")

                await _vetoed(pilot, screen)
                await pilot.pause()
                await pilot.pause(0.3)
                assert keywords.has_class(_INVALID)
                assert keywords.region.height == 1, "compact Inputs are borderless, one row"
                after = _painted_color_of(app, keywords, "ok, x, X")
                assert after != before, "the compact mark painted nothing"
                assert after.triplet == keywords.styles.color.rich_color.triplet, (
                    f"painted {after!r}, the field's error colour is {keywords.styles.color!r}"
                )
                hint = screen.query_one("#library-note-keywords-hint", Static)
                assert hint.display and hint.region.height == 1
                assert _painted_text_in_region(app, hint.region).startswith(
                    "Keywords contain a case-insensitive duplicate"
                )
    finally:
        profile.db.close_connection()
    assert events == []


async def test_typing_in_the_body_under_a_standing_title_veto_keeps_the_hint_steady(
    tmp_path, monkeypatch
):
    """M-5: the mark used to clear on EVERY keystroke, so each body key hid
    the hint row and the next veto re-showed it -- the body shifted up and
    down while the user typed. The mark clears only when the vetoed field
    changes.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                _set_long_title(screen)
                await pilot.click("#library-note-body")
                await _until(pilot, lambda: screen.focused is body, "the body to take focus")
                await _vetoed(pilot, screen)
                await pilot.pause()
                await pilot.pause(0.3)
                hint = screen.query_one("#library-note-title-hint", Static)
                assert hint.display
                hint_region = hint.region
                body_y = body.region.y

                for key in "abc":
                    await pilot.press(key)
                    await pilot.pause()
                    assert hint.display, f"the hint row vanished after {key!r}"
                    assert hint.region == hint_region, "the hint row moved"
                    assert body.region.y == body_y, "the body shifted"
                    assert screen._notes_state.validation_field == "title"
                    status = _painted_text_in_region(
                        app, screen.query_one("#library-note-work-authority").region
                    )
                    assert "Fix the marked field" in status, status
                    assert "save automatically" not in status, status
                # Fix round 2: the next autosave is refused again; sample the
                # layout THROUGH it, every frame, including the "Saving…" one
                # that used to paint a one-row status and move the body.
                seen_y: set[int] = set()
                seen_hint: set = set()
                try:
                    async with asyncio.timeout(10.0):
                        while screen._notes_state.autosave_state != "validation":
                            seen_y.add(body.region.y)
                            seen_hint.add(hint.region)
                            await pilot.pause(0.005)
                except TimeoutError as exc:
                    raise AssertionError("no refused autosave followed the keys") from exc
                seen_y.add(body.region.y)
                seen_hint.add(hint.region)
                assert seen_y == {body_y}, f"the body moved through the refused autosave: {sorted(seen_y)}"
                assert seen_hint == {hint_region}, f"the hint row moved: {seen_hint}"
                await pilot.pause(0.3)
                assert hint.display and hint.region == hint_region
                assert body.region.y == body_y

                # Editing the vetoed field itself clears its mark.
                await pilot.click("#library-note-title")
                await _until(
                    pilot,
                    lambda: getattr(screen.focused, "id", None) == "library-note-title",
                    "the title to take focus",
                )
                await pilot.press("backspace")
                await pilot.pause()
                assert screen._notes_state.validation_field == ""
                assert not hint.display
    finally:
        profile.db.close_connection()
    assert events == []


async def test_discarding_a_new_note_with_a_veto_standing_clears_the_mark(
    tmp_path, monkeypatch
):
    """M-6 (discard): the mark is state on the screen, not on the note; a
    discarded new note must not hand its mark to the next note opened.

    The reachable discard with a mark standing: a new note whose only edit
    is a whitespace title ("  ") -- the autosave vetoes it (the whitespace
    veto marks the title), and Escape discards the note as empty rather than
    refusing to leave (review finding #7, TASK-34000.1). "Discard new note"
    itself is offered only while the note is untouched, and an untouched
    note has no veto.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _new_blank_note(screen, pilot)
                title = screen.query_one("#library-note-title", Input)
                await _until(pilot, lambda: screen.focused is title, "the title caret")
                await _type(pilot, "  ")
                await _vetoed(pilot, screen)
                await pilot.pause(0.3)
                assert screen._notes_state.validation_field == "title"
                assert screen.query_one("#library-note-title-hint", Static).display

                await pilot.press("escape")
                await _until(
                    pilot, lambda: screen._notes_state.view == "list", "Escape to discard the empty note"
                )
                assert screen._notes_state.validation_field == ""

                await _open_note(screen, pilot, _TITLE, profile.note_id)
                assert screen._notes_state.validation_field == ""
                assert not screen.query_one("#library-note-title").has_class(_INVALID)
                assert not screen.query_one("#library-note-title-hint", Static).display
    finally:
        profile.db.close_connection()
    assert events == []


async def test_opening_another_note_clears_a_stale_mark(tmp_path, monkeypatch):
    """M-6 (switch): a mark left on the state is cleared when another note
    loads, so it can never be painted onto a note it does not belong to.
    The mark is set on the state directly: every user path that leaves a
    note with a veto standing is itself vetoed, so this is the belt that
    backs the braces."""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _open_note(screen, pilot, _TITLE, profile.note_id)
                screen._notes_state.mark_vetoed_field("title", "stale reason")
                screen._apply_library_note_presentation_state()
                await pilot.pause()
                assert screen.query_one("#library-note-title-hint", Static).display

                await _open_note(screen, pilot, _OTHER_TITLE, profile.other_id)
                assert screen._notes_state.validation_field == ""
                assert not screen.query_one("#library-note-title").has_class(_INVALID)
                assert not screen.query_one("#library-note-title-hint", Static).display
    finally:
        profile.db.close_connection()
    assert events == []


async def test_leaving_the_info_keywords_box_tidies_it(tmp_path, monkeypatch):
    """M-7: the Info twin ``#library-note-context-keywords`` edits the same
    keywords draft and gets the same tidy on the way out."""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _open_info(pilot, screen)
                context_keywords = screen.query_one("#library-note-context-keywords", Input)
                await pilot.click("#library-note-context-keywords")
                await _until(
                    pilot, lambda: screen.focused is context_keywords, "the Info keywords box"
                )
                await _type(pilot, "ok, x, X")
                assert context_keywords.value == "ok, x, X"
                await pilot.press("tab")
                await _until(pilot, lambda: screen.focused is not context_keywords, "Tab out")
                assert context_keywords.value == "ok, x", "leaving the Info keywords did not tidy"
                assert screen.query_one("#library-note-keywords", Input).value == "ok, x"
                await _until(
                    pilot,
                    lambda: {
                        row["keyword"]
                        for row in profile.db.get_keywords_for_note(profile.note_id)
                    }
                    == {"ok", "x"},
                    "the tidied keywords to be saved",
                    timeout=10.0,
                )
    finally:
        profile.db.close_connection()
    assert events == []
