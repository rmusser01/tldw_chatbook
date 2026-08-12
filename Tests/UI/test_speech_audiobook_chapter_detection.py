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
    and pops a notify toast -- unacceptable once per keystroke.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import TextArea

from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield AudioBookGenerationWidget()


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

        assert widget.content_text.startswith("Chapter 1")
        assert len(widget.detected_chapters) >= 1
