"""TASK-34000.8: the Library Notes editor header keeps Save, Use in Console
and Discard new note visible inside the editor pane at 120 columns and up.

Review finding N-05 (qa/notes-library-ux-review-2026-10-02): from the
shell's 120-column breakpoint up the header was always ONE horizontal strip
-- the save state, then [Edit Preview Info], then [Save Use in Console
Discard new note] -- shaped from the shell's ``compact`` flag alone. Every
header Button kept Textual's 16-cell minimum and the task row reserved 61
cells (TASK-32623), so the strip needed about 126 cells while the editor
pane had 48 at 120x36 and 82 at 160x45: Save and Use in Console were
painted past the pane edge (or off the terminal), the save state was
squeezed to one column ("S"), F6 landed on an unseen Save and the footer
still read "enter save note".

Every test mounts the real ``LibraryScreen`` under ``LibraryHarness``
(``CSS_PATH`` = the app bundle): without the bundle the width rules are
absent and a ``region`` claim measures nothing
(backlog/docs/lessons-textual.md).

On the base (``8a8970dd2e``) the first test is RED at both sizes: at 120x36
Save's region starts at x=126 on a 120-column screen; at 160x45 Use in
Console (x=149, 18 wide) runs past the pane's right edge at 158. The
slower variants (140x40, 200x50 and 235x52 with the rail open, the long
title, the 235 -> 120 -> 235 round trip, the delete prompt while stacked)
live in ``test_library_note_header_fits_pane_extended.py``, outside the
PR lane.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static, TextArea

from Tests.UI.test_library_media_reader_shell import _painted_text_in_region
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

pytestmark = pytest.mark.bootstrap_profile

#: AC#5 names these two; each is RED on the base for its own reason.
GATED_SIZES = [(120, 36), (160, 45)]

#: AC#2's widening walk, from the compact shape that already fit everything.
WIDENING_STEPS = [(120, 36), (140, 40), (160, 45), (200, 50), (235, 52)]

PANE = "#library-note-work-pane"
STATUS = "#library-note-status"
MODE_CONTROLS = "#library-note-mode-controls"
TASK_ACTIONS = "#library-note-task-actions"
SAVE = "#library-note-save"
USE_IN_CONSOLE = "#library-note-use-in-console"
DISCARD = "#library-note-discard-new"
BODY = "#library-note-body"

#: The header controls an open (existing) note shows; Discard is the new
#: note's extra one.
HEADER_CONTROLS = (
    "#library-note-edit",
    "#library-note-preview",
    "#library-note-context",
    SAVE,
    USE_IN_CONSOLE,
)

#: AC#4: the save state may share the row, but never as a one-column strip.
STATUS_MIN_WIDTH = 12


def _notes_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    return LibraryHarness(app)


async def _open_first_note(host, pilot):
    """Open the Notes list, then the first note; return the screen."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await pilot.pause()
    list(screen.query(".library-notes-row").results(Button))[0].press()
    await _wait_for_selector(screen, pilot, BODY)
    await pilot.pause()
    await pilot.pause()
    return screen


async def _new_blank_note(host, pilot):
    """Create an untouched blank note (Discard new note applies); return the screen."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-create-note", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
    screen.query_one("#library-notes-create-blank", Button).press()
    await _wait_for_selector(screen, pilot, BODY)
    await pilot.pause()
    await pilot.pause()
    return screen


def _is_on_screen(screen, widget) -> bool:
    region = widget.region
    return (
        region.width > 0
        and region.height > 0
        and screen.region.contains_region(region)
        and widget in screen._compositor.visible_widgets
    )


def _assert_whole_inside_pane(size, host, screen, selector: str, label: str) -> None:
    """The control has a region inside the pane and paints its whole label."""
    widget = screen.query_one(selector, Button)
    pane = screen.query_one(PANE)
    region = widget.region
    assert region.width > 0 and region.height > 0, (
        f"{size}: {label} has no painted region ({region})"
    )
    assert screen.region.contains_region(region), (
        f"{size}: {label} is off-screen at {region} (screen {screen.region})"
    )
    assert pane.region.contains_region(region), (
        f"{size}: {label} at {region} is outside the editor pane {pane.region}"
    )
    painted = " ".join(_painted_text_in_region(host, region).split())
    assert label in painted, (
        f"{size}: {label!r} is not painted whole inside its region: {painted!r}"
    )


def _assert_status_is_a_readable_word(size, screen) -> None:
    status = screen.query_one(STATUS, Static)
    assert status.region.width >= STATUS_MIN_WIDTH, (
        f"{size}: the save state is a {status.region.width}-column strip "
        f"({status.region})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", GATED_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
async def test_save_and_use_in_console_are_whole_inside_the_pane(size) -> None:
    """AC#1 / AC#5: on an open note, Save and the whole label "Use in Console"
    have a region inside the editor pane and the screen, and the save state
    beside them is a readable word (AC#4).

    RED on the base: at 120x36 Save is at x=126 (screen ends at 119);
    at 160x45 Use in Console (x=149, w=18) runs past the pane edge at 158.
    """
    host = _notes_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note(host, pilot)
        _assert_whole_inside_pane(size, host, screen, SAVE, "Save")
        _assert_whole_inside_pane(size, host, screen, USE_IN_CONSOLE, "Use in Console")
        _assert_status_is_a_readable_word(size, screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", GATED_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
async def test_discard_new_note_is_whole_and_the_mode_row_never_moves(size) -> None:
    """AC#1 / AC#4: on an untouched new note "Discard new note" is whole inside
    the pane; when typing hides it, Edit/Preview/Info keep their x and the
    task row keeps its width (TASK-32623's guarantee at these sizes -- the
    235-column pin stays in ``test_library_notes_reader.py``).
    """
    host = _notes_host()
    async with host.run_test(size=size) as pilot:
        screen = await _new_blank_note(host, pilot)
        discard = screen.query_one(DISCARD, Button)
        await _wait_for_condition(
            pilot,
            lambda: discard.display and discard.visible,
            message="An untouched new note never offered Discard new note.",
        )
        # AC#1: "Discard new note" or its compact "Discard" -- the stacked
        # task row has the whole pane and no more, and at 48 cells the full
        # wording needs 53 (measured), so a one-line row takes the short one.
        discard_label = str(discard.label)
        assert discard_label in {"Discard new note", "Discard"}, discard_label
        _assert_whole_inside_pane(size, host, screen, DISCARD, discard_label)
        mode_controls = screen.query_one(MODE_CONTROLS)
        task_actions = screen.query_one(TASK_ACTIONS)
        x_with_discard = mode_controls.region.x
        width_with_discard = task_actions.region.width

        screen.query_one(BODY, TextArea).text = "typed content"
        await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: not (discard.display and discard.visible),
            message="Discard new note never went away after typing.",
        )
        await pilot.pause()
        assert discard not in screen.focus_chain, "a hidden Discard is still Tab-reachable"
        assert mode_controls.region.x == x_with_discard, (
            f"{size}: Edit/Preview/Info moved from x={x_with_discard} to "
            f"x={mode_controls.region.x} when Discard new note went away"
        )
        assert task_actions.region.width == width_with_discard, (
            f"{size}: the task row shrank from {width_with_discard} to "
            f"{task_actions.region.width} when Discard new note went away"
        )
        _assert_whole_inside_pane(size, host, screen, SAVE, "Save")
        _assert_whole_inside_pane(size, host, screen, USE_IN_CONSOLE, "Use in Console")


@pytest.mark.asyncio
async def test_widening_never_hides_a_header_control() -> None:
    """AC#2: every header control on screen at 119 columns (the compact shape
    that already fit everything) is still on screen at 120 and at every
    wider size the task names; the save state is never a one-column strip.

    RED on the base from the first step: at 120x36 Save and Use in Console
    leave the screen and the save state is one column wide.
    """
    host = _notes_host()
    async with host.run_test(size=(119, 36)) as pilot:
        screen = await _open_first_note(host, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.compact is True,
            message="119 columns never settled on the compact layout.",
        )
        baseline = {
            selector
            for selector in HEADER_CONTROLS
            if _is_on_screen(screen, screen.query_one(selector, Button))
        }
        assert baseline == set(HEADER_CONTROLS), (
            f"sanity: the compact header at 119 columns hides {set(HEADER_CONTROLS) - baseline}"
        )
        for width, height in WIDENING_STEPS:
            await pilot.resize_terminal(width, height)
            await _wait_for_condition(
                pilot,
                lambda: screen._notes_state.compact is False
                and screen.query_one("#library-shell-grid").region.width == width,
                message=f"{width}x{height}: the wide layout never settled.",
            )
            await pilot.pause()
            await pilot.pause()
            on_screen = {
                selector
                for selector in HEADER_CONTROLS
                if _is_on_screen(screen, screen.query_one(selector, Button))
            }
            missing = baseline - on_screen
            assert not missing, (
                f"{width}x{height}: widening hid {sorted(missing)}; regions: "
                + ", ".join(
                    f"{selector}={screen.query_one(selector).region}"
                    for selector in sorted(missing)
                )
                + f"; pane={screen.query_one(PANE).region}"
            )
            _assert_status_is_a_readable_word((width, height), screen)
            _assert_whole_inside_pane(
                (width, height), host, screen, USE_IN_CONSOLE, "Use in Console"
            )


@pytest.mark.asyncio
async def test_tab_and_f6_only_land_on_visible_header_controls() -> None:
    """AC#3: F6 from the list lands on a Save that is on screen; a full Tab
    cycle from the body stops only on controls with a visible region; the
    footer's "enter save note" chip is shown exactly while the focused Save
    has one.

    RED on the base at 120x36: F6 focuses Save, whose region (x=126) is off
    a 120-column screen, and the footer advertises "enter save note" for it.
    """
    size = (120, 36)
    host = _notes_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note(host, pilot)
        screen.query_one("#library-notes-filter").focus()
        await pilot.pause()
        # The screen's own F6 action: the app-global ``f6`` binding is the
        # real app's (``TldwCli``), not this harness's, so other Library
        # focus tests call the action too (``test_library_artifacts_focus``).
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        await pilot.pause()
        focused = screen.focused
        assert focused is not None and focused.id == "library-note-save", (
            f"F6 from the list landed on {focused!r}"
        )
        assert _is_on_screen(screen, focused), (
            f"F6 landed on a Save the user cannot see: {focused.region} "
            f"(screen {screen.region}, pane {screen.query_one(PANE).region})"
        )
        assert ("enter", "save note") in screen._library_notes_footer_shortcuts()

        screen.query_one(BODY, TextArea).focus()
        await pilot.pause()
        stops: list[str] = []
        for _ in range(16):
            await pilot.press("tab")
            await pilot.pause()
            current = screen.focused
            assert current is not None
            stops.append(str(current.id))
            assert _is_on_screen(screen, current), (
                f"Tab landed on {current.id!r} with no visible region "
                f"({current.region}); stops so far: {stops}"
            )
            if current.id == "library-note-save":
                assert ("enter", "save note") in screen._library_notes_footer_shortcuts()
            if current.id == "library-note-body":
                break
        assert "library-note-save" in stops, f"Tab never reached Save: {stops}"
        assert "library-note-use-in-console" in stops, (
            f"Tab never reached Use in Console: {stops}"
        )
        assert "library-note-discard-new" not in stops, (
            f"Tab stopped on Discard new note on an existing note: {stops}"
        )
