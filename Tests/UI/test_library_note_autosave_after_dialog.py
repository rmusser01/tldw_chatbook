"""Autosave survives a dialog, and the status promises it only while armed.

TASK-34000.9 (review N-06). The Library route is reusable (TASK-31521), so
pushing ANY modal -- F1 help, Move note / Add to folder, the command palette
-- suspends the screen, and ``on_screen_suspend`` stops the autosave timer so
nothing fires against a hidden screen. Nothing re-armed it at resume: a note
typed into right before F1 stayed "Unsaved changes · Next: Keep editing;
changes save automatically." for 20 s over a stored row that never changed
(``verify/nl-v-n-06/07-after-20s-160x45.txt``).

This is the lean core that gates pull requests (``scripts/ui_pr_gate_census.txt``):
- F1 help and the command palette inside the autosave window still save
  (AC#1, AC#4);
- the painted status promises autosave only while a timer is armed (AC#2);
- two dialog cycles inside one burst arm one timer and save once (AC#3).

The Move note dialog and the 80x24 repeat live in
``test_library_note_autosave_after_dialog_extended.py``, which is not gated.

Same harness as ``test_library_quit_guard.py``: the real ``TldwCli``, the real
keys (F1, Ctrl+P, Escape), a real ChaChaNotes database. A save is proven by
the DATABASE ROW; the status by the compositor's painted cells.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.command import CommandPalette
from textual.widgets import Static

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _DEBOUNCE,
    _MAX_WAIT,
    _TITLE,
    SIZE,
    _library,
    _library_app,
    _no_autosave,
    _open_note,
    _scaled_autosave,
    _settings_without_splash,
    _type_at_end,
    _until,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

#: Same reason as the quit-guard files: full-app boots need the bootstrap
#: profile locally (``raw_source_selection_changed`` under the per-test redirect).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: How long a test waits for a save that should take about one debounce
#: (the scaled pair is 1 s / 2 s). Generous for a loaded machine; before the
#: fix no wait was long enough, because nothing was armed.
_AUTOSAVE_WITHIN = _MAX_WAIT + 6.0

_PROMISE = "changes save automatically"
_NO_PROMISE = "Unsaved changes · Next: Keep editing; press Save to save now."


def _painted_text_in_region(app, region) -> str:
    """Return only compositor cells painted inside ``region``, one line."""
    strips = list(app.screen._compositor.render_strips())
    rows = (
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, region.bottom)
    )
    return " ".join(" ".join(rows).split())


def _painted_status(app, screen) -> str:
    authority = screen.query_one("#library-note-work-authority", Static)
    return _painted_text_in_region(app, authority.region)


async def _autosaved(pilot, screen, profile, body: str) -> bool:
    """Whether the note's row holds ``body`` and the draft is clean, in time."""

    def _landed() -> bool:
        snapshot = screen._library_note_session.snapshot
        return (
            profile.note(profile.note_id)["content"] == body
            and snapshot is not None
            and not snapshot.dirty
            and not snapshot.saving
        )

    try:
        async with asyncio.timeout(_AUTOSAVE_WITHIN):
            while not _landed():
                await pilot.pause(0.05)
    except TimeoutError:
        return False
    return True


async def _open_help(pilot, app, screen) -> None:
    """Press F1; the help panel covers Library and its timer is stopped."""
    await pilot.press("f1")
    await _until(
        pilot, lambda: isinstance(app.screen, WorkbenchHelpPanel), "F1 help to open"
    )
    # TASK-31521 (AC#3): no timer may fire against the suspended screen.
    assert screen._notes_state.autosave_timer is None, "suspend left a timer armed"


async def _open_palette(pilot, app, screen) -> None:
    await pilot.press("ctrl+p")
    await _until(
        pilot, lambda: isinstance(app.screen, CommandPalette), "the palette to open"
    )
    assert screen._notes_state.autosave_timer is None, "suspend left a timer armed"


async def _close_dialog(pilot, app, screen) -> None:
    await pilot.press("escape")
    await _until(pilot, lambda: app.screen is screen, "the dialog to close")


# --- AC#1 / AC#4: the edit is saved after the dialog closes -----------------


async def test_f1_help_inside_the_autosave_window_still_saves_the_edit(
    tmp_path, monkeypatch
):
    """Review capture 07: F1 → Esc inside the debounce; the row must change."""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    saved = False
    row: dict = {}
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _type_at_end(pilot, body, " after help")
                assert screen._library_note_session.snapshot.dirty

                await _open_help(pilot, app, screen)
                await _close_dialog(pilot, app, screen)

                saved = await _autosaved(pilot, screen, profile, _BODY + " after help")
                row = profile.note(profile.note_id)
    finally:
        profile.db.close_connection()

    assert events == []
    assert saved, (
        "the edit typed before F1 was never autosaved after help closed: the "
        f"row still holds {row.get('content')!r}"
    )
    assert row["title"] == _TITLE


async def test_the_command_palette_inside_the_autosave_window_still_saves_the_edit(
    tmp_path, monkeypatch
):
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    saved = False
    row: dict = {}
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _type_at_end(pilot, body, " after palette")

                await _open_palette(pilot, app, screen)
                await _close_dialog(pilot, app, screen)

                saved = await _autosaved(
                    pilot, screen, profile, _BODY + " after palette"
                )
                row = profile.note(profile.note_id)
    finally:
        profile.db.close_connection()

    assert events == []
    assert saved, (
        "the edit typed before Ctrl+P was never autosaved after the palette "
        f"closed: the row still holds {row.get('content')!r}"
    )


# --- AC#2: the promise is painted only while a save is scheduled ------------


async def test_the_status_promises_autosave_only_while_a_timer_is_armed(
    tmp_path, monkeypatch
):
    """Autosave is turned off (one hour), so the only timer is the one a
    keystroke or a resume arms, and no save can land mid-assertion."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _type_at_end(pilot, body, "x")
                await pilot.pause()
                assert screen._notes_state.autosave_timer is not None
                assert _PROMISE in _painted_status(app, screen)

                # The real dialog: the palette stops the timer; closing it
                # must arm one again, and only then is the promise true.
                await _open_palette(pilot, app, screen)
                await _close_dialog(pilot, app, screen)
                await pilot.pause()
                assert screen._notes_state.autosave_timer is not None, (
                    "closing the palette left the dirty note with no timer"
                )
                assert _PROMISE in _painted_status(app, screen)

                # Dirty with nothing armed -- the state a modal leaves while it
                # is up -- must not promise. The suspend hook is called
                # directly so the covered screen's own paint can be read.
                screen.on_screen_suspend()
                assert screen._notes_state.autosave_timer is None
                screen._update_library_note_meta_static(content=body.text)
                await pilot.pause()
                painted = _painted_status(app, screen)
                assert _NO_PROMISE in painted, painted
                assert _PROMISE not in painted, painted

                screen.on_screen_resume()
                await pilot.pause()
                assert screen._notes_state.autosave_timer is not None
                painted = _painted_status(app, screen)
                assert _PROMISE in painted, painted
                assert "press Save" not in painted, painted
                # Nothing was ever saved: the promise is about the future.
                assert profile.note(profile.note_id)["content"] == _BODY
    finally:
        profile.db.close_connection()
    assert events == []


# --- AC#3: resume arms exactly one timer, and it fires once ------------------


async def test_resume_arms_exactly_one_timer_and_it_fires_once(tmp_path, monkeypatch):
    """Two F1 cycles inside one burst: one re-armed timer, one save (one
    version bump), and no timer left behind once it has landed."""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    saved = False
    version_before = profile.note(profile.note_id)["version"]
    version_after = version_before
    timer_after: object = None
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _type_at_end(pilot, body, " twice")

                await _open_help(pilot, app, screen)
                await _close_dialog(pilot, app, screen)
                await _open_help(pilot, app, screen)
                await _close_dialog(pilot, app, screen)

                saved = await _autosaved(pilot, screen, profile, _BODY + " twice")
                # A second armed timer would have fired by now.
                await pilot.pause(_DEBOUNCE + 0.5)
                version_after = profile.note(profile.note_id)["version"]
                timer_after = screen._notes_state.autosave_timer
    finally:
        profile.db.close_connection()

    assert events == []
    assert saved, "the edit typed before two F1 cycles was never autosaved"
    assert version_after == version_before + 1, (
        f"expected exactly one save, the row went {version_before} -> {version_after}"
    )
    assert timer_after is None, "a timer was left armed after the save landed"
