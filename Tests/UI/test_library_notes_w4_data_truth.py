"""Library ▸ Notes critique wave 4 -- group ``data-truth`` (tasks 32538, 32542).

Three facts the critique found wrong on screen: the chrome strip's word
count on a long note, the autosave clock's zone, and a structured import
source's repeat detection (the last is pinned in ``Tests/Notes`` and
``Tests/UI/test_library_notes_wave_import_ux.py``).
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import pytest
from textual.widgets import Button, Static, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_notes_session import DatabaseNoteSessionCoordinator
from tldw_chatbook.Library.library_notes_state import build_library_note_editor_state
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)

FACTS = "#library-note-chrome-facts"
SHORT_BODY = "alpha beta gamma"
#: 360 lines of 15 tokens: ~38 KB, 5,400 words -- the shape of the critique's
#: "Very long note — scaling laws digest" (35 KB, 5,404 tokens, 362 lines).
LONG_BODY = "\n".join(
    " ".join(f"w{line}_{index}" for index in range(15)) for line in range(360)
)


def _count(body: str) -> int:
    return LibraryNotesController._note_word_count(body)


def _build_two_note_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[
            {"title": "Markdown showcase", "id": "note-short", "content": SHORT_BODY},
            {"title": "Very long note", "id": "note-long", "content": LONG_BODY},
        ],
    )
    return LibraryHarness(app)


def _facts(screen) -> str:
    return str(screen.query_one(FACTS, Static).renderable)


async def _press_row(screen, pilot, title_fragment: str) -> TextArea:
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await pilot.pause()
    row = next(
        button
        for button in screen.query(".library-notes-row").results(Button)
        if title_fragment in str(button.label)
    )
    row.press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await pilot.pause()
    return screen.query_one("#library-note-body", TextArea)


async def _open_short_then_long(screen, pilot) -> TextArea:
    screen.query_one("#library-row-browse-notes", Button).press()
    await _press_row(screen, pilot, "Markdown showcase")
    await _wait_for_condition(
        pilot,
        lambda: _facts(screen) == f"{_count(SHORT_BODY)} words · 1:1",
        message=f"The short note's strip read {_facts(screen)!r}.",
    )
    screen.query_one("#library-note-back", Button).press()
    return await _press_row(screen, pilot, "Very long note")


# --- task-32538: the strip counts the open note's body ----------------------


@pytest.mark.asyncio
async def test_the_strip_shows_the_long_notes_count_after_opening_a_short_one():
    """Critique #3 read "404 words" on the 5,404-word note -- its own capture
    31 reads "5,404 words · 1:1"; the count was right and the "5," was
    dropped in the reading. Pinned on the production row-press path so a
    stale or truncated count on a long note is a failure, not a misread.
    """
    assert len(LONG_BODY.encode()) > 35_000
    host = _build_two_note_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        body = await _open_short_then_long(screen, pilot)

        expected = f"{_count(LONG_BODY):,} words · 1:1"
        await _wait_for_condition(
            pilot,
            lambda: _facts(screen) == expected,
            message=f"The long note's strip read {_facts(screen)!r}, not {expected!r}.",
        )
        assert _facts(screen).startswith("5,4")
        assert _count(body.text) == _count(LONG_BODY)


@pytest.mark.asyncio
async def test_a_caret_move_never_paints_a_count_fed_for_another_note():
    """A caret repaint reads the last fed count; after the long note opens
    that count is the long note's, never the short note's from before."""
    host = _build_two_note_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        body = await _open_short_then_long(screen, pilot)
        long_count = f"{_count(LONG_BODY):,} words"
        await _wait_for_condition(
            pilot,
            lambda: _facts(screen).startswith(long_count),
            message=f"The long note's strip read {_facts(screen)!r}.",
        )

        body.focus()
        body.move_cursor((200, 3))
        await _wait_for_condition(
            pilot,
            lambda: _facts(screen) == f"{long_count} · 201:4",
            message=f"After a caret move the strip read {_facts(screen)!r}.",
        )
        assert f"{_count(SHORT_BODY)} words" not in _facts(screen)


# --- task-32623: Info and the editor footer format one word count -----------


@pytest.mark.asyncio
async def test_info_and_the_editor_footer_format_the_same_word_count():
    """Critique #4: the footer read "5,453 words", Info read "5454" for the
    same open note -- no thousands separator. Both are built from the exact
    same int in one ``_library_note_presentation_state()`` call
    (``state.word_count`` feeds the footer, ``state.metadata_line`` feeds
    Info), so only the missing separator could make them read differently;
    fixed by formatting Info's copy the same way the footer already does.
    """
    host = _build_two_note_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_short_then_long(screen, pilot)
        expected = f"{_count(LONG_BODY):,} words"
        await _wait_for_condition(
            pilot,
            lambda: _facts(screen).startswith(expected),
            message=f"The long note's strip read {_facts(screen)!r}.",
        )

        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()
        info_text = str(
            screen.query_one("#library-note-context-meta", Static).renderable
        )
        assert expected in info_text, (
            f"Info read {info_text!r}, not matching the footer's {expected!r}."
        )


# --- task-32542: the status clock and Info agree on one zone ----------------


@pytest.fixture
def los_angeles_zone(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is unavailable on this platform")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


SAVE_INSTANT = datetime(2026, 9, 13, 5, 48, tzinfo=timezone.utc)


def test_status_line_and_info_agree_on_the_save_instant(los_angeles_zone):
    """Critique #3 (A 07/18): "Saved 05:48" beside Info's "Modified
    2026-09-12 22:54". One instant, one zone -- the reader's."""
    status = DatabaseNoteSessionCoordinator.saved_status_message(SAVE_INSTANT)
    info = build_library_note_editor_state(
        {"id": "n-1", "last_modified": SAVE_INSTANT.isoformat()},
        now=SAVE_INSTANT,
    ).meta_line

    assert status == "Saved 22:48"
    assert info.startswith("Modified 2026-09-12 22:48 · ")
    assert re.search(r"\b05:48\b", status + info) is None


def test_a_naive_save_instant_is_refused_rather_than_guessed(los_angeles_zone):
    """``astimezone()`` on a naive value assumes *local* time, so a future
    caller handing this a naive UTC clock would re-create task-32542 exactly,
    with no test failing. The seam refuses the input instead of guessing."""
    with pytest.raises(ValueError, match="timezone-aware"):
        DatabaseNoteSessionCoordinator.saved_status_message(
            SAVE_INSTANT.replace(tzinfo=None)
        )
