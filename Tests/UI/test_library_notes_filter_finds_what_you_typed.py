"""TASK-34000.12: the Library Notes filter finds what you typed.

Review finding N-09 (notes-library-ux-review-2026-10-02): a first-timer
typed ``meet`` and got nothing although the notes say "meeting"; a
researcher filtered on the keyword the app itself attached to a captured
answer and got nothing; the zero-result line said only "No notes match
“zebra”. Clear the filter." The list filter bound the whole input as one
FTS5 phrase against title and body alone.

Both walks drive the real ``TldwCli`` booted into Library with the
production stylesheet over a real ChaChaNotes database (the quit-guard
profile plus three notes and two keywords), at the review's size, through
the keys a user presses: ``/`` into the filter box, the word, Enter. The
assertions read what the compositor paints inside the notes canvas.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _library,
    _NotesProfile,
    _type,
    _until,
)
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
    _painted_widget_text,
)
from Tests.UI.test_library_shell import _seed_conversations, _wait_for_selector

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The exact zero-result sentence (AC#4). Asserted whole: the old sentence
#: is not a prefix of it, so a substring wait could not pass on the base.
ZERO_RESULT = (
    "No notes match “qqqq” in titles, bodies or keywords. "
    "Try fewer words, or clear the filter."
)


class _FilterProfile(_NotesProfile):
    """The quit-guard profile plus the review's corpus and two keywords."""

    def __init__(self, tmp_path) -> None:
        super().__init__(tmp_path)
        self.meeting_id = self.db.add_note(
            "Meeting notes", "team meeting about size of chunk"
        )
        self.lisbon_id = self.db.add_note(
            "Trip planning: Lisbon", "itinerary and chunk size notes"
        )
        # The keyed note: its keyword appears in NEITHER its title nor its
        # body, so only the keyword leg can find it (fix round 1, I-1).
        self.keyed_id = self.db.add_note("Field trip", "nothing here")
        assert self.meeting_id and self.lisbon_id and self.keyed_id
        for note_id, keyword in ((self.keyed_id, "zebra"), (self.lisbon_id, "homelab")):
            keyword_id = self.db.add_keyword(keyword)
            assert keyword_id is not None
            assert self.db.link_note_to_keyword(note_id, keyword_id) is True


def _app(tmp_path):
    """The production app booted straight into Library over the filter corpus."""
    app = _build_test_app(configured_default="library")
    _seed_conversations(app, [])
    profile = _FilterProfile(tmp_path)
    app.chachanotes_db = profile.db
    app.notes_scope_service = profile.scope_service
    app.notes_service = profile.interop
    return app, profile


async def _notes_list(app, pilot):
    """Click the rail's Notes row -- a real click, so focus lands where it
    does live (``Button.press()`` would not move it)."""
    screen = await _library(app, pilot)
    await pilot.click("#library-row-browse-notes")
    await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
    return screen


async def _filter(screen, pilot, text: str) -> None:
    """``/`` into the filter box (unless it already has focus), type
    ``text``, Enter -- the user's keys. Arriving from the rail's Notes row
    leaves the box focused, so a ``/`` there would be typed literally."""

    def _focused() -> bool:
        return getattr(screen.focused, "id", None) == "library-notes-filter"

    if not _focused():
        await pilot.press("/")
        await _until(pilot, _focused, "`/` to focus the notes filter", timeout=10.0)
    box = screen.query_one("#library-notes-filter", Input)
    assert box.value == "", box.value
    await _type(pilot, text)
    await _until(pilot, lambda: box.value == text, f"the box to hold {text!r}")
    await pilot.press("enter")


def _painted(host, screen, selector: str) -> str:
    """What the compositor paints inside the widget that owns ``selector``.

    Cropped to the owning widget (task-32294): a canvas-wide crop puts the
    pane's border glyph between the two rows a wrapped sentence occupies.
    """
    return _painted_widget_text(host, screen.query_one(selector))


async def _until_painted(host, screen, pilot, selector: str, needle: str, *, timeout: float = 12.0):
    def _found() -> bool:
        return bool(screen.query(selector)) and needle in _painted(host, screen, selector)

    try:
        await _until(pilot, _found, f"{needle!r} to be painted in {selector}", timeout=timeout)
    except AssertionError as exc:
        current = _painted(host, screen, selector) if screen.query(selector) else "<absent>"
        raise AssertionError(f"{exc}; {selector} paints: {current!r}") from None


async def _clear_filter(screen, pilot) -> None:
    """Press Clear filter and wait for the unfiltered list to be back."""
    screen.query_one("#library-notes-filter-clear", Button).press()
    def _cleared() -> bool:
        # Clear recomposes the canvas: for a frame the filter Input is not
        # mounted. A missing node is "not yet", not an error (PR #3234 UI
        # Fast Lane (7): NoMatches on a loaded runner).
        inputs = screen.query("#library-notes-filter")
        return (
            screen._notes_state.filter == ""
            and bool(inputs)
            and inputs.first(Input).value == ""
            and bool(screen.query(".library-notes-tree-note-row"))
        )

    await _until(pilot, _cleared, "the filter to clear and the list to return")
    await pilot.pause(0.1)


async def test_typing_meet_and_enter_lists_the_meeting_notes_and_zebra_finds_the_keyed_note(
    tmp_path,
) -> None:
    """AC#3 and AC#1, painted: a keyword-only word, a keyword+body pair and a partial word each find their note."""
    app, _profile = _app(tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _notes_list(app, pilot)

        # keyword leg first: "zebra" is only a keyword of "Field trip", so on
        # the base this half fails for the keyword reason alone
        await _filter(screen, pilot, "zebra")
        await _until_painted(app, screen, pilot, "#library-notes-status", "filter: zebra · 1 result")
        rows = _painted(app, screen, "#library-notes-list")
        assert "Field trip" in rows, rows
        assert "Meeting notes" not in rows, rows
        assert not screen.query("#library-notes-empty")

        # across legs: one word in a keyword, the other in the body
        await _clear_filter(screen, pilot)
        await _filter(screen, pilot, "homelab itinerary")
        await _until_painted(
            app, screen, pilot, "#library-notes-status", "filter: homelab itinerary · 1 result"
        )
        rows = _painted(app, screen, "#library-notes-list")
        assert "Trip planning: Lisbon" in rows, rows
        assert not screen.query("#library-notes-empty")

        # a partial word
        await _clear_filter(screen, pilot)
        await _filter(screen, pilot, "meet")
        await _until_painted(app, screen, pilot, "#library-notes-status", "filter: meet · 1 result")
        rows = _painted(app, screen, "#library-notes-list")
        assert "Meeting notes" in rows, rows
        assert "Trip planning: Lisbon" not in rows, rows
        assert not screen.query("#library-notes-empty")


async def test_a_zero_result_names_what_was_searched_and_offers_a_next_step(
    tmp_path,
) -> None:
    """AC#4: the whole sentence, and the Clear filter button beside it."""
    app, _profile = _app(tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _notes_list(app, pilot)

        await _filter(screen, pilot, "qqqq")
        await _until_painted(app, screen, pilot, "#library-notes-status", "filter: qqqq · 0 results")
        await _until_painted(app, screen, pilot, "#library-notes-empty", ZERO_RESULT)
        assert _painted(app, screen, "#library-notes-empty") == ZERO_RESULT
        clear = screen.query_one("#library-notes-filter-clear", Button)
        assert str(clear.label) == "Clear filter"
        assert "Clear filter" in _painted(app, screen, "#library-notes-filter-clear")
