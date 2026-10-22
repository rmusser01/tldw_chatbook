"""An autosave veto never moves the user; the field is marked; leaving tidies it.

TASK-34000.10 (review N-07). A title ending in a space, or keywords with a
case-insensitive duplicate, vetoes every autosave (ADR-027: the save never
strips or rewrites). The screen then handled the vetoed AUTOSAVE like a veto
on an explicit Save: it routed to the field, switched the pane and focused
it, so the next keys landed in the wrong field
(``verify/nl-v-n-07/06-after-eggs-160x45.txt``: title "Groceries  eggs",
body "milk"; ``12-kw-after-body2-160x45.txt``: body text in Keywords, on
Info). TASK-34000.1 guarded the half where an editor field has focus; this
pins the rest:

- a vetoed autosave never routes, wherever focus is -- here, on the Info
  region (AC#1, AC#5 (a));
- the vetoed field is outlined with the coordinator's reason under it, and
  the status names the way out without promising a save (AC#2, AC#5 (b));
- leaving Title or Keywords tidies the edge space or the duplicate as a
  visible draft edit, and the note then saves (AC#4, AC#5 (c)).

This is the lean core that gates pull requests (``scripts/ui_pr_gate_census.txt``).
The two explicit-path pins -- Save, and leaving the note, still route to the
vetoed field (AC#3) -- are green on base and live in
``test_library_note_autosave_veto_in_place_extended.py``, which is not gated.

The title-while-typing case stays pinned by TASK-34000.1's
``test_a_refused_save_keeps_the_typist_in_place_and_ctrl_q_asks``.

Same harness as ``test_library_quit_guard.py``: the real ``TldwCli``, real
keys and clicks, a real ChaChaNotes database. The mark and the status are
read from the compositor's painted cells; a save is proven by the row.

A trailing-space title cannot be used for the focus-on-Info arms: Tab or a
click out of the title tidies it before any veto lands (AC#4). Those arms use
the one title veto the tidy cannot remove, a title over the length limit.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, Static, TextArea

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
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

#: Same reason as the quit-guard files: full-app boots need the bootstrap
#: profile locally (``raw_source_selection_changed`` under the per-test redirect).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

_LONG_TITLE = "x" * 301
_DUPLICATE_VETO = "Keywords contain a case-insensitive duplicate — remove one to save."
_FIX_THE_FIELD = "Fix the marked field"
_PROMISE = "save automatically"
_INVALID = "library-note-field-invalid"


def _painted_text_in_region(app, region) -> str:
    """Return only compositor cells painted inside ``region``, one line."""
    strips = list(app.screen._compositor.render_strips())
    rows = (
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, region.bottom)
    )
    return " ".join(" ".join(rows).split())


def _painted(app, screen, selector: str) -> str:
    return _painted_text_in_region(app, screen.query_one(selector).region)


async def _vetoed(pilot, screen) -> None:
    await _until(
        pilot,
        lambda: screen._notes_state.autosave_state == "validation",
        "the autosave to be vetoed",
        timeout=10.0,
    )


async def _open_info(pilot, screen):
    """Click Info; return the widget that then has focus (the Info region)."""
    await pilot.click("#library-note-context")
    await _until(
        pilot,
        lambda: screen.query_one("#library-note-context-region").display
        and screen.focused is not None
        and screen.focused.id == "library-note-context-region",
        "Info to open and take focus",
    )
    await pilot.pause()
    return screen.focused


def _set_long_title(screen) -> None:
    """The length veto, through the Input so ``Input.Changed`` drives the draft."""
    screen.query_one("#library-note-title", Input).value = _LONG_TITLE


# --- AC#1 / AC#5 (a): a vetoed autosave never routes, wherever focus is -----


async def test_an_autosave_veto_with_focus_on_info_keeps_focus_and_pane(
    tmp_path, monkeypatch
):
    """Focus on the Info region (not an editor field) when the veto lands.

    Before the fix this routed to Edit and focused the title -- the hole
    TASK-34000.1's editor-field guard left open, reachable mid-task because
    the max wait fires while the user is reading Info.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _open_note(screen, pilot, _TITLE, profile.note_id)
                _set_long_title(screen)
                focused = await _open_info(pilot, screen)
                preview_before = screen._notes_state.preview

                await _vetoed(pilot, screen)
                await pilot.pause()
                await pilot.pause(0.3)  # the unguarded veto focused after a refresh

                assert screen.focused is focused, (
                    "a vetoed autosave moved focus to "
                    f"{getattr(screen.focused, 'id', None)!r}"
                )
                assert screen._notes_state.context is True, "the pane left Info"
                assert screen.query_one("#library-note-context-region").display
                assert screen._notes_state.preview is preview_before
                assert screen.query_one("#library-note-title").has_class(_INVALID)
                assert screen._notes_state.validation_field == "title"
                assert profile.note(profile.note_id)["title"] == _TITLE
    finally:
        profile.db.close_connection()
    assert events == []


# --- AC#2 / AC#5 (b): the field is marked; the status does not promise -------


async def test_a_duplicate_keyword_veto_marks_the_field_and_drops_the_promise(
    tmp_path, monkeypatch
):
    """Review capture 12, as a user reaches it: "ok, x, X" typed in Keywords,
    and the autosave fires while the cursor is still in the field.

    Fix round 1 (I-2): the first version injected the duplicate through the
    Input's ``Changed`` while the body had focus -- a state no user reaches,
    because Tab, a click and a programmatic focus move all tidy the field on
    the way out (AC#4). The veto now lands in the field the user is typing
    in: focus stays there, the field is outlined with the reason under it,
    the pane stays Edit, and the status names the fix with ONE instruction
    (M-4: the meta line's "Retry Save" is suppressed while a veto stands).
    Then Tab tidies the duplicate, the mark clears, and the note saves.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _new_blank_note(screen, pilot)
                note_id = screen._library_note_session.snapshot.note_id
                title = screen.query_one("#library-note-title", Input)
                keywords = screen.query_one("#library-note-keywords", Input)
                body = screen.query_one("#library-note-body", TextArea)
                await _until(pilot, lambda: screen.focused is title, "the title caret")
                await pilot.press("tab")
                await _until(pilot, lambda: screen.focused is keywords, "Tab to keywords")
                await _type(pilot, "ok, x, X")
                await _vetoed(pilot, screen)  # the debounce fires in the field
                await pilot.pause()
                await pilot.pause(0.3)

                assert screen.focused is keywords, (
                    f"focus moved to {getattr(screen.focused, 'id', None)!r}"
                )
                assert screen._notes_state.context is False, "the pane flipped to Info"
                assert screen.query_one("#library-note-editor-region").display
                assert keywords.value == "ok, x, X"
                assert keywords.has_class(_INVALID), "the keywords field is not marked"
                hint = screen.query_one("#library-note-keywords-hint", Static)
                assert hint.display, "no reason under the keywords field"
                assert _painted_text_in_region(app, hint.region) == _DUPLICATE_VETO
                status = _painted(app, screen, "#library-note-work-authority")
                assert _FIX_THE_FIELD in status, status
                assert _PROMISE not in status, status
                # M-4: one instruction. The meta row clips at this width
                # ("Next: Retry S…"), so the Static's own text is checked too.
                meta_widget = screen.query_one("#library-note-status", Static)
                meta_text = getattr(meta_widget.renderable, "plain", str(meta_widget.renderable))
                assert _DUPLICATE_VETO in meta_text, meta_text
                assert "Retry Save" not in meta_text, meta_text
                assert "Retry" not in _painted(app, screen, "#library-note-status")

                # Leaving the field tidies it: the duplicate goes, the mark
                # clears, the note saves.
                await pilot.press("tab")
                await _until(pilot, lambda: screen.focused is body, "Tab to the body")
                assert keywords.value == "ok, x", "leaving Keywords did not tidy them"
                assert not keywords.has_class(_INVALID), "the mark outlived the veto"
                assert not hint.display

                def _stored_keywords() -> set[str]:
                    return {
                        row["keyword"] for row in profile.db.get_keywords_for_note(note_id)
                    }

                await _until(
                    pilot,
                    lambda: _stored_keywords() == {"ok", "x"}
                    and not screen._library_note_session.snapshot.dirty,
                    "the tidied keywords to be saved",
                    timeout=10.0,
                )
                assert screen._notes_state.autosave_state != "validation"
    finally:
        profile.db.close_connection()
    assert events == []


# --- AC#4 / AC#5 (c): leaving the field tidies it, and the note saves --------


async def test_leaving_the_title_with_a_trailing_space_tidies_it_and_the_note_saves(
    tmp_path, monkeypatch
):
    """The review's title case, up to the Tab: the space goes, the note saves.

    The keywords twin (Tab out of "ok, x, X" stores exactly {ok, x}) is the
    second half of the duplicate-keyword arm above.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _new_blank_note(screen, pilot)
                note_id = screen._library_note_session.snapshot.note_id
                title = screen.query_one("#library-note-title", Input)
                keywords = screen.query_one("#library-note-keywords", Input)
                await _until(pilot, lambda: screen.focused is title, "the title caret")
                await _type(pilot, "Groceries ")
                assert title.value == "Groceries "
                await pilot.press("tab")
                await _until(pilot, lambda: screen.focused is keywords, "Tab to keywords")
                assert title.value == "Groceries", "leaving the title did not tidy it"
                await _until(
                    pilot,
                    lambda: profile.note(note_id)["title"] == "Groceries"
                    and not screen._library_note_session.snapshot.dirty,
                    "the tidied title to be saved",
                    timeout=10.0,
                )
                assert not title.has_class(_INVALID)

    finally:
        profile.db.close_connection()
    assert events == []
