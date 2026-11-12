"""Chapter results belong to the current source, including before debounce."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, TextArea

from Tests.private_profile import private_profile_test
from tldw_chatbook.TTS.audiobook_generator import ChapterDetector
from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget
from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget

OLD_SOURCE = "Chapter 1: Old source\nThis is the old chapter body."
NEW_SOURCE = "Chapter 1: New source\nThis is the newer chapter body."


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield AudioBookGenerationWidget()


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


def _assert_editor(widget: AudioBookGenerationWidget, titles: list[str]) -> None:
    editor = widget.query_one("#chapter-editor-widget", ChapterEditorWidget)
    assert [chapter.title for chapter in widget.detected_chapters] == titles
    assert [chapter.title for chapter in editor.get_chapters()] == titles
    assert [table.row_count for table in editor.query(DataTable)] == [len(titles)]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unchanged", "clear", "replace"])
@private_profile_test
async def test_current_source_owns_pending_chapter_results(
    request, monkeypatch, change
):
    app = _Host()
    entered, release = threading.Event(), threading.Event()
    original_detect = ChapterDetector.detect_chapters

    def detect(content):
        chapters = original_detect(content)
        if content == OLD_SOURCE:
            entered.set()
            assert release.wait(15), "Old-source detector was not released"
        return chapters

    monkeypatch.setattr(ChapterDetector, "detect_chapters", staticmethod(detect))
    async with app.run_test(size=(120, 48)) as pilot:
        widget = app.query_one(AudioBookGenerationWidget)
        widget._import_from_paste()
        area = widget.query_one("#content-preview", TextArea)
        applications: list[list[str]] = []
        notifications: list[str] = []
        original_apply, original_notify = widget._apply_detected_chapters, app.notify

        def apply(chapters, generation):
            original_apply(chapters, generation)
            applications.append([chapter.title for chapter in chapters])

        def notify(message, **kwargs):
            if str(message).startswith("Detected "):
                notifications.append(str(message))
            original_notify(message, **kwargs)

        monkeypatch.setattr(widget, "_apply_detected_chapters", apply)
        monkeypatch.setattr(app, "notify", notify)
        try:
            area.load_text(OLD_SOURCE)
            await _until(entered.is_set)
            debounce = widget._CHAPTER_DETECT_DEBOUNCE_SECONDS
            # Hold any replacement dispatch beyond this rendezvous: the old
            # result must lose authority on edit, not on a later dispatch.
            monkeypatch.setattr(widget, "_CHAPTER_DETECT_DEBOUNCE_SECONDS", 60)
            if change != "unchanged":
                area.focus()
                await pilot.press("f7", "backspace")
                if change == "replace":
                    await pilot.press("n", "e", "w")
                expected = "new" if change == "replace" else ""
                await _until(lambda: widget.content_text == expected)
                assert area.text == expected

            release.set()
            await _until(lambda: ["Old source"] in applications)
            await app.workers.wait_for_complete()
            await pilot.pause()
            if change == "unchanged":
                _assert_editor(widget, ["Old source"])
                assert len(notifications) == 1
            else:
                _assert_editor(widget, [])
                assert notifications == []
                # A later normal TextArea change still goes through the real
                # debounce, detector, result callback and chapter table.
                monkeypatch.setattr(
                    widget, "_CHAPTER_DETECT_DEBOUNCE_SECONDS", debounce
                )
                area.load_text(NEW_SOURCE)
                await _until(lambda: ["New source"] in applications)
                await pilot.pause()
                _assert_editor(widget, ["New source"])
                assert len(notifications) == 1
        finally:
            release.set()
            await app.workers.wait_for_complete()


@pytest.mark.asyncio
@private_profile_test
async def test_slow_detector_cannot_replace_newer_source_chapters(request, monkeypatch):
    app = _Host()
    entered, release = threading.Event(), threading.Event()
    original_detect = ChapterDetector.detect_chapters

    def detect(content):
        chapters = original_detect(content)
        if content == OLD_SOURCE:
            entered.set()
            assert release.wait(15), "Old-source detector was not released"
        return chapters

    monkeypatch.setattr(ChapterDetector, "detect_chapters", staticmethod(detect))
    async with app.run_test(size=(120, 48)) as pilot:
        widget = app.query_one(AudioBookGenerationWidget)
        widget._import_from_paste()
        area = widget.query_one("#content-preview", TextArea)
        applications: list[list[str]] = []
        original_apply = widget._apply_detected_chapters

        def apply(chapters, generation):
            original_apply(chapters, generation)
            applications.append([chapter.title for chapter in chapters])

        monkeypatch.setattr(widget, "_apply_detected_chapters", apply)
        try:
            area.load_text(OLD_SOURCE)
            await _until(entered.is_set)
            monkeypatch.setattr(widget, "_CHAPTER_DETECT_DEBOUNCE_SECONDS", 60)
            area.load_text(NEW_SOURCE)
            await _until(lambda: widget.content_text == NEW_SOURCE)
            widget._detect_chapters()
            await _until(lambda: ["New source"] in applications)
            _assert_editor(widget, ["New source"])
            release.set()
            # Observe the cancelled awaiter's physical worker publishing. A
            # WorkerManager wait alone does not join that cancelled thread.
            await _until(lambda: ["Old source"] in applications)
            await pilot.pause()
            _assert_editor(widget, ["New source"])
        finally:
            release.set()
            await app.workers.wait_for_complete()


@pytest.mark.asyncio
@private_profile_test
async def test_programmatic_import_echo_keeps_current_dispatch(
    request, monkeypatch, tmp_path
):
    app = _Host()
    entered, release = threading.Event(), threading.Event()
    original_detect = ChapterDetector.detect_chapters

    def detect(content):
        chapters = original_detect(content)
        entered.set()
        assert release.wait(15), "Imported-source detector was not released"
        return chapters

    monkeypatch.setattr(ChapterDetector, "detect_chapters", staticmethod(detect))
    source = tmp_path / "book.txt"
    source.write_text(OLD_SOURCE, encoding="utf-8")
    async with app.run_test(size=(120, 48)) as pilot:
        widget = app.query_one(AudioBookGenerationWidget)
        # The import explicitly dispatches after setting content_text and
        # load_text. Its queued Changed echo must not invalidate that result.
        monkeypatch.setattr(widget, "_CHAPTER_DETECT_DEBOUNCE_SECONDS", 60)
        try:
            await widget._handle_file_selection(source)
            await _until(entered.is_set)
            await pilot.pause()
            assert widget.content_text == OLD_SOURCE
            assert widget.query_one("#content-preview", TextArea).text == OLD_SOURCE
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            _assert_editor(widget, ["Old source"])
        finally:
            release.set()
            await app.workers.wait_for_complete()
