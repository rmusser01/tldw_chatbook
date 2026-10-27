"""Ctrl+S saves in every Library editor, and nowhere else (TASK-34000.40 N-33).

The 2026-10-02 review found four save models on one screen: Ctrl+S saved a
skill, and did nothing in a note (the status sat on "Unsaved changes" until the
2 s autosave), a prompt or a Folder files document. The owner ruled
(ADR-031 refinement 2026-10-03; settled 2026-10-27): Ctrl+S performs the
focused editor's visible commit, flushes an autosaving editor at once, says
why at the editor when it refuses, and is unbound outside an editor.

Every test here drives the production ``TldwCli`` booted into Library and
presses the real keys; the data claims read a real ChaChaNotes database. This
file holds the notes arms; split for the 30 s census budget, the siblings
hold the prompt and Folder files arms (``_extended``), the skill editor
(``_skills``), the two Media forms on the production stylesheet harness
(``_media``), the chip following a save state no key changed
(``test_library_ctrl_s_chip_follows_save_state.py``) and the boot-free
refusal branches (``test_library_ctrl_s_refusals.py``).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, TextArea

# Stubs first: they register the optional modules the app imports below probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _TITLE,
    SIZE,
    _library,
    _library_app,
    _no_autosave,
    _open_note,
    _record_toasts,
    _settings_without_splash,
    _type,
    _until,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: TASK-34000.40 AC#3 (owner ruling "ctrl-s should save in every editor";
#: controller ruling T9-g): a clean editor's Ctrl+S, the note Save's answer.
NO_CHANGES = "Saved — no changes."


def _painted(app, widget) -> str:
    """The text the compositor paints inside ``widget``'s region."""
    region = widget.region
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text.rstrip()
        for y in range(region.y, min(region.bottom, len(strips)))
    )


def _footer(app) -> str:
    return _painted(app, app.screen.query_one(AppFooterStatus))


def _within(widget, selector_id: str) -> bool:
    return widget is not None and any(
        node.id == selector_id for node in widget.ancestors_with_self
    )


def _focused_id(screen) -> str | None:
    """The focused widget's id: a rebuild may replace the widget, not the id."""
    return getattr(screen.focused, "id", None)


def _record_resolved_keys(monkeypatch, app) -> list[str]:
    """Record each key whose NON-priority binding check has completed.

    Review R1 N-1: a check that a key did NOT do something needs a positive
    signal that the key was fully processed, not a fixed pause (which can
    only ever give a false green). ``App._on_key`` resolves non-priority
    bindings after the key has bubbled through every widget, and awaits the
    bound action, so a key listed here has been handled end to end.
    """
    resolved: list[str] = []
    check_bindings = app._check_bindings

    async def recording(key: str, priority: bool = False) -> bool:
        try:
            return await check_bindings(key, priority)
        finally:
            if not priority:
                resolved.append(key)

    monkeypatch.setattr(app, "_check_bindings", recording)
    return resolved


async def _press_processed(pilot, resolved: list[str], key: str) -> None:
    """Press ``key``, then wait until its binding check has completed."""
    before = resolved.count(key)
    await pilot.press(key)
    await _until(
        pilot, lambda: resolved.count(key) > before, f"{key} to be fully processed"
    )


async def _f1_keys(pilot, app, screen) -> set[str]:
    """Open F1, read the keys it lists, close it again.

    Ruling T9-h: F1 lists keys, so no keyless status row ("- : typing in
    field") may reach it -- checked on every opening.
    """
    await pilot.press("f1")
    await _until(
        pilot, lambda: isinstance(app.screen, WorkbenchHelpPanel), "F1 to open help"
    )
    keys = {key for key, _label in app.screen.state.shortcuts}
    assert "" not in keys, f"F1 lists a keyless row: {app.screen.state.shortcuts}"
    await pilot.press("escape")
    await _until(pilot, lambda: app.screen is screen, "help to close")
    return keys


async def _click_into_body(pilot, screen) -> TextArea:
    body = screen.query_one("#library-note-body", TextArea)
    await pilot.click("#library-note-body")
    await _until(pilot, lambda: screen.focused is body, "a click to focus the body")
    return body


def _body_text(screen) -> str | None:
    """The live body's text, re-queried: never a widget held across a save."""
    matches = screen.query("#library-note-body")
    return matches.first().text if matches else None


async def test_ctrl_s_in_the_note_body_flushes_the_autosave_now(tmp_path, monkeypatch):
    """Rule (b): the key writes the draft now, not when the debounce fires.

    Autosave is pushed out to an hour, so the only thing that can write the
    row is the key. The footer chips Ctrl+S only while the note has unsaved
    changes (ruling T9-j): none on the clean note, one after one typed
    character, none again once the key has saved. A second Ctrl+S on the
    clean note says "Saved — no changes." on the editor's own status line --
    no toast -- and the caret never moves.
    """
    _no_autosave(monkeypatch)
    app, profile = _library_app(tmp_path, monkeypatch, [], lambda notes: None)
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _click_into_body(pilot, screen)
            await _until(
                pilot,
                lambda: "esc back to notes" in _footer(app),
                "the body's footer to paint",
            )
            assert "ctrl+s" not in _footer(app), f"clean note chipped: {_footer(app)}"
            await _type(pilot, "c")
            await _until(
                pilot,
                lambda: "ctrl+s save now" in _footer(app),
                "one typed character to chip Ctrl+S",
            )
            await _type(pilot, "trl-s test ")
            await _until(
                pilot,
                lambda: screen._library_note_session.snapshot.dirty,
                "typing to dirty the note",
            )
            assert profile.note(profile.note_id)["content"] == _BODY
            # Ctrl+S is real input: it reaches ``on_key`` like any key, so a
            # pending programmatic landing is withdrawn (standing owner rule).
            screen._library_notes_filter_row_handoff = ("pending", True)

            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: profile.note(profile.note_id)["content"] == _body_text(screen),
                "Ctrl+S to write the typed body to the database",
                timeout=3.0,
            )
            assert "ctrl-s test" in profile.note(profile.note_id)["content"]
            assert screen._library_notes_filter_row_handoff == ("pending", False)
            status = screen.query_one("#library-note-status")
            await _until(
                pilot,
                lambda: _painted(app, status).startswith("Saved"),
                "the status line to say Saved",
            )
            assert _focused_id(screen) == "library-note-body", (
                f"Ctrl+S moved focus to {screen.focused!r}"
            )
            await _until(
                pilot,
                lambda: "ctrl+s" not in _footer(app),
                "the saved, clean note to drop the Ctrl+S chip",
            )

            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: NO_CHANGES in _painted(app, status),
                "a clean note's Ctrl+S to say there were no changes",
            )
            assert _focused_id(screen) == "library-note-body"
            assert not [toast for toast in toasts if "Saved" in toast], toasts

            # A real key still types after both saves (lessons-textual :680).
            await _type(pilot, "z")
            await _until(
                pilot,
                lambda: screen._library_note_session.snapshot.dirty,
                "a key typed after Ctrl+S to reach the body",
            )
    profile.db.close_connection()


async def test_ctrl_s_outside_an_editor_does_nothing_and_is_not_advertised(
    tmp_path, monkeypatch
):
    """Rules (c), (d), (e). The Delete confirm refuses at the editor -- on the
    status line, no toast (review M-3) -- without moving focus, and F1 there
    does not offer the key it refuses (ruling T9-h); the Notes list does
    nothing even with a dirty note open beside it (ruling T9-d), and neither
    the footer nor F1 offers the key there, while the body's footer and F1
    do."""
    _no_autosave(monkeypatch)
    app, profile = _library_app(tmp_path, monkeypatch, [], lambda notes: None)
    toasts = _record_toasts(monkeypatch, app)
    resolved = _record_resolved_keys(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _open_note(screen, pilot, _TITLE, profile.note_id)

            # The Delete confirm: refused, focus kept on Cancel, nothing written.
            await pilot.click("#library-note-context")
            delete = await _until_selector(pilot, screen, "#library-note-delete")
            # Setup, not the subject: Delete sits below Info's fold at 160x45.
            delete.press()
            await _until_selector(pilot, screen, "#library-note-delete-cancel")
            await _until(
                pilot,
                lambda: _focused_id(screen) == "library-note-delete-cancel",
                "focus on Cancel",
            )
            await pilot.press("ctrl+s")
            status = screen.query_one("#library-note-status")
            await _until(
                pilot,
                lambda: "Save locked" in _painted(app, status),
                "the Delete confirm to refuse Ctrl+S at the editor",
            )
            assert _focused_id(screen) == "library-note-delete-cancel"
            assert profile.note(profile.note_id)["content"] == _BODY
            assert not [toast for toast in toasts if "Save locked" in toast], toasts
            assert "ctrl+s" not in await _f1_keys(pilot, app, screen)
            # (F1's push and pop may rebuild the confirm row: same control.)
            assert _focused_id(screen) == "library-note-delete-cancel"
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: not screen._notes_state.confirming_delete,
                "Enter on Cancel to close the confirm",
            )
            assert profile.note(profile.note_id) is not None

            # A dirty note open in the pane; focus moves to the Notes list.
            await pilot.click("#library-note-edit")
            await _until(
                pilot,
                lambda: bool(screen.query("#library-note-body"))
                and screen.query_one("#library-note-body").display,
                "the editor to return",
            )
            await _click_into_body(pilot, screen)
            await _type(pilot, "unsaved ")
            await _until(
                pilot,
                lambda: screen._library_note_session.snapshot.dirty,
                "typing to dirty the note",
            )
            assert "ctrl+s save now" in _footer(app), _footer(app)
            assert "ctrl+s" in await _f1_keys(pilot, app, screen)

            # Shift+F6 lands in the Notes filter; Down from its empty box
            # lands on the folder tree's first row (TASK-34000.28 N-21).
            toasts_before = list(toasts)
            await pilot.press("shift+f6")
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None) == "library-notes-filter",
                "Shift+F6 to move focus to the Notes filter",
            )
            for step in ("filter", "tree"):
                if step == "tree":
                    await pilot.press("down")
                    await _until(
                        pilot,
                        lambda: isinstance(screen.focused, Button)
                        and not _within(screen.focused, "library-note-work-pane"),
                        "Down to move focus onto the folder tree",
                    )
                moved_to = screen.focused
                # Review R1 N-1: wait until the key is fully processed, then
                # check what it did not do (a fixed pause only false-greens).
                await _press_processed(pilot, resolved, "ctrl+s")
                assert profile.note(profile.note_id)["content"] == _BODY, (
                    f"Ctrl+S on {moved_to!r} wrote the note open beside it"
                )
                assert screen._library_note_session.snapshot.dirty
                assert screen.focused is moved_to
                assert toasts == toasts_before, toasts
                footer = _footer(app)
                assert "ctrl+s" not in footer, (step, footer)
            assert "ctrl+s" not in await _f1_keys(pilot, app, screen)
    profile.db.close_connection()


async def _until_selector(pilot, screen, selector: str):
    await _until(
        pilot,
        lambda: bool(screen.query(selector)) and screen.query_one(selector).display,
        f"{selector} to mount",
    )
    return screen.query_one(selector)
