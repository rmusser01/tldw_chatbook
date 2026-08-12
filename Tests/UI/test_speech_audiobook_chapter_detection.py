"""AudioBook auto-chapter detection must not crash the paste box (task-15478).

`on_text_area_changed` (the handler behind the "Paste Text" content-preview
box) used to gate chapter detection on `query_one("#auto-chapters-switch",
Switch)`. That id was composed nowhere: the "Chapter Settings" collapsible
that used to own it was replaced by `ChapterEditorWidget` in commit
`256911ea6` ("audiobook work", 2025-07-22), and the four query sites were
never updated. The result was a `NoMatches` raised on every keystroke in the
paste box.

The fix keeps auto-detection (its switch defaulted to on, and there is no UI
left to turn it off) but:
  - drops the dead query on all four sites;
  - moves the keystroke-path call behind a debounce timer, since detection
    walks the *entire* pasted text with `ChapterDetector.detect_chapters`
    and pops a notify toast -- unacceptable once per keystroke;
  - moves the detection itself off the event loop entirely (review
    follow-up): `ChapterDetector.detect_chapters` was benchmarked at
    ~19ms/90k words, ~60ms/300k, ~200ms on a 6MB paste -- past the repo's
    100ms worker budget for exactly the large pastes an audiobook feature
    invites -- so it now runs in a `@work(thread=True)` worker with results
    marshaled back via `call_from_thread`, for all four call sites, not
    just the debounced one;
  - only pops the "Detected N chapters" toast when N actually changes since
    the last toast, so a debounced re-paste that keeps re-running detection
    does not spam a toast per settle.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import TextArea

from tldw_chatbook.TTS.audiobook_generator import Chapter
from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield AudioBookGenerationWidget()


def _make_large_book(n_words: int) -> str:
    """Build synthetic book text with a "Chapter N" header every 2000 words.

    Mirrors the shape the reviewer benchmarked ChapterDetector against
    (~19ms/90k words, ~60ms/300k, ~200ms on a 6MB paste): plain prose lines
    interspersed with chapter markers the detector's regexes actually match,
    not degenerate all-blank or all-matching input.
    """
    words_per_line = 12
    lines: list[str] = []
    total = 0
    chapter = 1
    while total < n_words:
        if total % 2000 == 0:
            lines.append(f"Chapter {chapter}")
            chapter += 1
        lines.append(" ".join(["word"] * words_per_line))
        total += words_per_line
    return "\n".join(lines)


def _stub_chapter(number: int) -> Chapter:
    return Chapter(
        number=number,
        title=f"Chapter {number}",
        content="Some words go here for this chapter.",
        start_position=0,
        end_position=0,
    )


@pytest.mark.asyncio
async def test_typing_in_the_paste_box_raises_no_exception():
    """Regression repro: this used to raise NoMatches on every keystroke.

    Simulates real typing (one `TextArea.Changed` message per keypress) in
    the enabled content-preview TextArea and asserts the app is still alive
    and the widget observed the content afterward.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        content_preview = app.query_one("#content-preview", TextArea)
        content_preview.disabled = False
        content_preview.focus()

        for ch in "Hi there":
            await pilot.press(ch if ch != " " else "space")

        await pilot.pause()

        assert widget.content_text == "Hi there"


@pytest.mark.asyncio
async def test_a_burst_of_keystrokes_runs_detection_once_after_it_settles():
    """Detection must run off the keystroke path (task-15478 AC #2).

    A burst of changes inside the debounce window must not call
    `_detect_chapters` synchronously; it should fire exactly once, only
    after the input goes quiet for the debounce interval.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        content_preview = app.query_one("#content-preview", TextArea)
        content_preview.disabled = False

        detect = Mock()
        widget._detect_chapters = detect

        debounce = widget._CHAPTER_DETECT_DEBOUNCE_SECONDS

        # A rapid burst: several changes, each well inside the debounce
        # window of the previous one.
        content_preview.text = "Chapter 1"
        await pilot.pause(debounce / 4)
        content_preview.text = "Chapter 1\nChapter 2"
        await pilot.pause(debounce / 4)
        content_preview.text = "Chapter 1\nChapter 2\nChapter 3"

        # Still inside the window from the last change: no call yet.
        await pilot.pause(debounce / 4)
        assert detect.call_count == 0

        # Let the debounce interval elapse from the *last* change.
        await pilot.pause(debounce + 0.2)

        assert detect.call_count == 1


@pytest.mark.asyncio
async def test_file_import_still_auto_detects_without_the_missing_switch(tmp_path):
    """The three one-shot import paths (file/notes/conversation) are not on
    a keystroke path, so they keep running detection unconditionally --
    matching the removed switch's `value=True` default, with no dead query.

    Detection itself now runs in a background thread worker (review
    follow-up), so this waits for it to actually finish rather than
    asserting immediately after dispatch.
    """
    book_path = tmp_path / "book.txt"
    book_path.write_text("Chapter 1\nOnce upon a time.\n\nChapter 2\nThe end.\n")

    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)

        assert widget.detected_chapters == []

        widget._handle_file_selection(str(book_path))
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert widget.content_text.startswith("Chapter 1")
        assert len(widget.detected_chapters) >= 1


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_large_paste_chapter_detection():
    """The actual perf bug (task-15478 review): chapter detection used to
    run synchronously on the event loop per debounce settle.
    `ChapterDetector.detect_chapters` was benchmarked at ~19ms/90k words,
    ~60ms/300k, ~200ms on a 6MB paste -- well past the repo's 100ms worker
    budget, for exactly the large pastes an audiobook feature invites.

    Same heartbeat-seam pattern as
    `Tests/UI/test_llm_screen_ollama_probe_nonblocking.py` (task-15473): a
    concurrent task ticks every 5ms while detection is in flight. This
    brackets the heartbeat measurement tightly around `_detect_chapters()`
    and its worker's completion only (bypassing the debounce timer and its
    own ~1s of unrelated real-time sleep, which is covered separately by
    `test_a_burst_of_keystrokes_runs_detection_once_after_it_settles` and
    would otherwise dilute the signal -- confirmed empirically: with the
    debounce wait included, a deliberately-reintroduced synchronous
    detection call still cleared >100 heartbeats, because ~1s of real
    `asyncio.sleep` swamped a ~700ms block). Over this tight window, a loop
    frozen by a synchronous call lands at 0 ticks (measured); a correctly
    threaded call clears double digits.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)
        content_preview = app.query_one("#content-preview", TextArea)
        content_preview.disabled = False

        # ~3,000,000 words / ~15MB -- comfortably north of the reviewer's 6MB
        # reference point, for a clear, low-flake margin above the assertion
        # threshold (measured ~650-700ms of detection, ~25-35 heartbeats).
        widget.content_text = _make_large_book(3_000_000)

        heartbeats = 0
        stop = asyncio.Event()

        async def _heartbeat() -> None:
            nonlocal heartbeats
            while not stop.is_set():
                heartbeats += 1
                await asyncio.sleep(0.005)

        hb_task = asyncio.create_task(_heartbeat())
        try:
            widget._detect_chapters()
            # Wait for the threaded detector to actually finish -- this is
            # the window the heartbeat must keep ticking through.
            await app.workers.wait_for_complete()
        finally:
            stop.set()
            await hb_task

        assert widget.detected_chapters
        assert heartbeats >= 10, (
            f"event loop looks starved during chapter detection: only "
            f"{heartbeats} heartbeat ticks landed"
        )


@pytest.mark.asyncio
async def test_notify_only_fires_when_the_chapter_count_changes():
    """Minor fix (task-15478 review): a debounced re-paste can re-run
    detection several times as the user keeps typing; before this, every
    settle popped its own "Detected N chapters" toast even when N hadn't
    moved.
    """
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)

        notify = Mock()
        app.notify = notify

        # Same count twice in a row: only the first call should toast.
        widget._apply_detected_chapters([_stub_chapter(1), _stub_chapter(2)])
        await pilot.pause()
        widget._apply_detected_chapters([_stub_chapter(1), _stub_chapter(2)])
        await pilot.pause()
        assert notify.call_count == 1

        # Count changes: a new toast fires.
        widget._apply_detected_chapters([_stub_chapter(1)])
        await pilot.pause()
        assert notify.call_count == 2
