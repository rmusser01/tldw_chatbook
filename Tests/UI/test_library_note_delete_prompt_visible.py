"""TASK-34000.13: the Library Notes delete confirmation shows Cancel and
Delete inside Info at every size.

Review finding N-10 (qa/notes-library-ux-review-2026-10-02): Info ▸ Delete
changed only the footer. The prompt, ``#library-note-delete-confirmation``,
is a ``Vertical`` composed as the last child of Info's Danger section, and
only the compact layout gave it a height; on the wide stage it kept
Textual's default ``height: 1fr; overflow: hidden``, which inside a
scrolling parent means "whatever rows are left over" -- one row at 120x36,
so the copy was clipped and both buttons painted on the screen's last row,
outside the Info box. And nothing scrolled the prompt into view: focusing
Cancel asks the pane to scroll to the button, but a squeezed ``1fr`` child
leaves ``max_scroll_y`` at 0, so there was nowhere to scroll to.

Every test mounts the real ``LibraryScreen`` under ``LibraryHarness``
(``CSS_PATH`` = the app bundle): without the bundle the wide and compact
rules are both absent and a ``region`` claim measures nothing
(backlog/docs/lessons-textual.md).

On the base (``0467e105e7``) the first test is RED at both sizes for two
different reasons, which is why both are in the gate:

* 120x36 -- the HEIGHT half: the prompt gets the one leftover row, so
  Cancel/Delete paint at y=35 while Info ends at row 33.
* 80x24 -- the REVEAL half: the compact rule gives the prompt its two rows
  (y=25-26 on a 24-row screen) and Info can scroll to it
  (``max_scroll_y=5``), but nothing does (``scroll_y=0``).

The slower variants (160x30, 160x45, 235x52, a resize while the prompt is
open, the "Linked from (1)" case through a real database, and the DB-level
Cancel/Delete checks) live in
``test_library_note_delete_prompt_visible_extended.py``, outside the PR lane.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_library_media_reader_shell import _painted_text_in_region
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import DELETE_CONFIRM_COPY

pytestmark = pytest.mark.bootstrap_profile

#: AC#4 names these two; each is RED on the base for a different reason.
GATED_SIZES = [(120, 36), (80, 24)]

PROMPT = "#library-note-delete-confirmation"
COPY = "#library-note-delete-confirm-copy"
ACTIONS = "#library-note-delete-actions"
CANCEL = "#library-note-delete-cancel"
CONFIRM = "#library-note-delete-confirm"
INFO = "#library-note-context-region"
DELETE = "#library-note-context-delete"


def _notes_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    return LibraryHarness(app)


async def _open_first_note_in_info(host, pilot):
    """Open the Notes list, the first note, then its Info pane."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await pilot.pause()
    list(screen.query(".library-notes-row").results(Button))[0].press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await pilot.pause()
    screen.query_one("#library-note-context", Button).press()
    await pilot.pause()
    await pilot.pause()
    return screen


async def _open_delete_prompt(screen, pilot) -> None:
    screen.query_one(DELETE, Button).press()
    await pilot.pause()
    await pilot.pause()


def _parts(screen):
    return (
        screen.query_one(INFO),
        screen.query_one(PROMPT),
        screen.query_one(COPY, Static),
        screen.query_one(ACTIONS),
        screen.query_one(CANCEL, Button),
        screen.query_one(CONFIRM, Button),
    )


def _assert_prompt_fully_inside_info(size, host, screen) -> None:
    """The whole prompt -- copy, Cancel and Delete -- paints inside Info."""
    info, prompt, copy, actions, cancel, confirm = _parts(screen)
    assert prompt.display is True
    for name, widget in (("Cancel", cancel), ("Delete", confirm)):
        assert widget.region.height > 0 and widget.region.width > 0, (
            f"{size}: {name} has no painted region ({widget.region})"
        )
        assert screen.region.contains_region(widget.region), (
            f"{size}: {name} is off-screen at {widget.region} "
            f"(screen {screen.region})"
        )
        assert info.region.contains_region(widget.region), (
            f"{size}: {name} at {widget.region} is outside the Info box "
            f"{info.region} (scroll_y={info.scroll_y}, "
            f"max_scroll_y={info.max_scroll_y}, prompt={prompt.region}, "
            f"prompt height={prompt.styles.height})"
        )
    assert prompt.region.height >= copy.region.height + actions.region.height, (
        f"{size}: the prompt box {prompt.region} is shorter than its copy "
        f"{copy.region} plus its buttons {actions.region} -- a leftover-height "
        f"box (height={prompt.styles.height})"
    )
    assert info.region.contains_region(copy.region), (
        f"{size}: the copy at {copy.region} is outside Info {info.region}"
    )
    painted = " ".join(_painted_text_in_region(host, copy.region).split())
    assert DELETE_CONFIRM_COPY in painted, (
        f"{size}: the prompt copy is not painted in full inside Info: {painted!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", GATED_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
async def test_cancel_and_delete_are_visible_as_soon_as_the_prompt_opens(size) -> None:
    """AC#1 / AC#4: Info ▸ Delete shows the full copy and both buttons
    inside the Info box with no manual scrolling.

    RED on the base at 120x36 because the prompt is a ``1fr`` child squeezed
    to one leftover row (Cancel/Delete at y=35, Info rows 15-33); RED at
    80x24 because the two-row compact box sits at y=25-26 of a 24-row screen
    and Info (``max_scroll_y=5``) is never scrolled to it.
    """
    host = _notes_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note_in_info(host, pilot)
        await _open_delete_prompt(screen, pilot)
        _assert_prompt_fully_inside_info(size, host, screen)


@pytest.mark.asyncio
async def test_focus_lands_on_a_visible_cancel_and_tab_moves_between_visible_buttons() -> None:
    """AC#2: the prompt opens with focus on a visible Cancel; Tab moves it
    to a visible Delete and Shift+Tab brings it back, with both buttons
    inside Info the whole time (the task-32132 trap keeps Tab inside)."""
    size = (120, 36)
    host = _notes_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note_in_info(host, pilot)
        await _open_delete_prompt(screen, pilot)
        info, _prompt, _copy, _actions, cancel, confirm = _parts(screen)

        assert screen.focused is cancel, f"focused={screen.focused!r}"
        assert info.region.contains_region(cancel.region), (
            f"Cancel is focused but outside Info: {cancel.region} vs {info.region}"
        )

        await pilot.press("tab")
        await pilot.pause()
        assert screen.focused is confirm, f"Tab went to {screen.focused!r}"
        assert info.region.contains_region(confirm.region), (
            f"Delete is focused but outside Info: {confirm.region} vs {info.region}"
        )

        await pilot.press("shift+tab")
        await pilot.pause()
        assert screen.focused is cancel, f"Shift+Tab went to {screen.focused!r}"
        assert info.region.contains_region(cancel.region)


@pytest.mark.asyncio
async def test_cancel_restores_info_scroll_and_focuses_delete() -> None:
    """AC#3: the task-32268 contract at a size where revealing the prompt
    has to scroll Info -- the prompt is the next child after Delete, inside
    the Info border; Cancel puts Info's scroll back where the reader had it
    and focuses Delete."""
    size = (120, 36)
    host = _notes_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note_in_info(host, pilot)
        info = screen.query_one(INFO)
        delete_button = screen.query_one(DELETE, Button)
        # Where the reader was: scrolled so Delete is visible (at 120x36
        # that is the top -- Info fits until the prompt opens).
        delete_button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        origin = info.scroll_y
        assert info.region.contains_region(delete_button.region)

        await _open_delete_prompt(screen, pilot)
        _info, prompt, _copy, _actions, cancel, _confirm = _parts(screen)
        children = list(info.children)
        assert children.index(prompt) == children.index(delete_button) + 1, (
            "The prompt must be the next thing after Delete"
        )
        assert info.region.contains_region(prompt.region), (
            f"The prompt at {prompt.region} escapes Info {info.region}"
        )
        assert prompt.region.y > delete_button.region.y
        # The whole prompt did not fit the pre-prompt viewport, so the
        # reveal had to move Info; this is the half Cancel must undo.
        assert info.scroll_y > origin, (
            f"sanity: revealing the prompt did not scroll Info "
            f"(scroll_y={info.scroll_y}, origin={origin}, "
            f"max_scroll_y={info.max_scroll_y}); the fixture is too short"
        )

        cancel.press()
        await pilot.pause()
        await pilot.pause()
        assert prompt.display is False
        assert info.scroll_y == origin, (
            f"Cancel left Info at scroll_y={info.scroll_y}, not {origin}"
        )
        assert screen.focused is delete_button, f"focused={screen.focused!r}"
        assert info.region.contains_region(delete_button.region)
