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
    """Review capture 12: a keyword veto landing while the body is typed.

    Leaving the keywords box by Tab tidies a duplicate (AC#4), so the
    duplicate reaches the autosave the one way it still can while the body
    has focus: through the keywords Input's own ``Changed`` while the user
    is in the body. The veto then lands mid-sentence; the keys keep landing
    in the body, the keywords field is outlined with the reason under it,
    the pane stays Edit, and the status names the fix instead of promising
    a save that cannot happen.
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
                body = screen.query_one("#library-note-body", TextArea)
                await _until(pilot, lambda: screen.focused is title, "the title caret")
                await pilot.press("tab")
                await _until(pilot, lambda: screen.focused is keywords, "Tab to keywords")
                await _type(pilot, "ok, x")
                await pilot.press("tab")
                await _until(pilot, lambda: screen.focused is body, "Tab to the body")
                await _type(pilot, "milk")
                keywords.value = "ok, x, X"  # the duplicate, while the body has focus
                await _vetoed(pilot, screen)
                await _type(pilot, " eggs")
                # Typing cleared the veto; the next autosave is refused again.
                await _vetoed(pilot, screen)
                await pilot.pause()
                await pilot.pause(0.3)

                assert screen.focused is body, (
                    f"focus moved to {getattr(screen.focused, 'id', None)!r}"
                )
                assert screen._notes_state.context is False, "the pane flipped to Info"
                assert screen.query_one("#library-note-editor-region").display
                assert "milk eggs" in _painted(app, screen, "#library-note-body")
                assert body.text == "milk eggs"
                assert keywords.value == "ok, x, X", "keys landed in Keywords"
                assert keywords.has_class(_INVALID), "the keywords field is not marked"
                hint = screen.query_one("#library-note-keywords-hint", Static)
                assert hint.display, "no reason under the keywords field"
                assert _painted_text_in_region(app, hint.region) == _DUPLICATE_VETO
                status = _painted(app, screen, "#library-note-work-authority")
                assert _FIX_THE_FIELD in status, status
                assert _PROMISE not in status, status
    finally:
        profile.db.close_connection()
    assert events == []


# --- AC#4 / AC#5 (c): leaving the field tidies it, and the note saves --------


async def test_leaving_the_title_with_a_trailing_space_tidies_it_and_the_note_saves(
    tmp_path, monkeypatch
):
    """The review's title case, up to the Tab: the space goes, the note saves.

    Twin arm: keywords "ok, x, X" → Tab → "ok, x", and the stored keyword
    rows are exactly {ok, x}.
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

                await _type(pilot, "ok, x, X")
                assert keywords.value == "ok, x, X"
                await pilot.press("tab")
                await _until(
                    pilot,
                    lambda: screen.focused is screen.query_one("#library-note-body"),
                    "Tab to the body",
                )
                assert keywords.value == "ok, x", "leaving Keywords did not tidy them"

                def _stored_keywords() -> set[str]:
                    return {row["keyword"] for row in profile.db.get_keywords_for_note(note_id)}

                await _until(
                    pilot,
                    lambda: _stored_keywords() == {"ok", "x"}
                    and not screen._library_note_session.snapshot.dirty,
                    "the tidied keywords to be saved",
                    timeout=10.0,
                )
                assert screen._notes_state.autosave_state != "validation"
                assert not keywords.has_class(_INVALID)
    finally:
        profile.db.close_connection()
    assert events == []
