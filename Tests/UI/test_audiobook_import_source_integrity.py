"""Full imported text remains editable and authoritative through Generate.

Restoring any import's 1000-character preview slice must fail these tests.
The host consumes the generation request before any synthesis can run.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, DataTable, Select, TextArea

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSAudioBookGenerateEvent,
)
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.TTS.audiobook_generator import ChapterDetector
from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget, FileOpen
from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget


class _Host(App[None]):
    def __init__(self):
        super().__init__()
        self.submissions: list[STTSAudioBookGenerateEvent] = []

    def compose(self) -> ComposeResult:
        yield AudioBookGenerationWidget()

    @on(STTSAudioBookGenerateEvent)
    def capture_submission(self, event: STTSAudioBookGenerateEvent) -> None:
        event.stop()
        self.submissions.append(event)


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


def _source(length: int) -> str:
    prefix = "Chapter 1: Beginning\n"
    tail = "\nChapter 2: Ending\nKEEP_FINAL_SENTINEL"
    return prefix + "a" * (length - len(prefix) - len(tail)) + tail


async def _import_file(app, widget, pilot, path) -> None:
    widget.query_one(Collapsible).collapsed = False
    widget.query_one("#import-source-select", Select).value = "file"
    await pilot.pause()
    widget.query_one("#import-content-btn", Button).press()
    await _until(lambda: isinstance(app.screen, FileOpen))
    await _until(lambda: bool(app.screen.query("EnhancedDirectoryNavigation")))
    await pilot.pause()
    app.screen.dismiss(path)
    await _until(lambda: not widget.query_one("#content-preview", TextArea).disabled)
    await pilot.pause()


async def _generate(app, widget) -> STTSAudioBookGenerateEvent:
    await _until(
        lambda: widget.query_one("#narrator-voice-select", Select).value != Select.BLANK
    )
    widget.query_one("#generate-audiobook-btn", Button).press()
    await _until(lambda: bool(app.submissions))
    return app.submissions[-1]


def _assert_full_source(
    widget, submitted, expected, tail="KEEP_FINAL_SENTINEL"
) -> None:
    # Check each consumer separately: keeping only the model or only the editor
    # complete still permits a later Changed echo or Generate to lose the tail.
    assert submitted.content == expected
    assert widget.content_text == expected
    assert widget.query_one("#content-preview", TextArea).text == expected
    assert [chapter.title for chapter in widget.detected_chapters][-1] == "Ending"
    editor = widget.query_one("#chapter-editor-widget", ChapterEditorWidget)
    assert editor.get_chapters()[-1].content == tail
    table = editor.query_one("#chapter-table", DataTable)
    assert table.row_count == len(widget.detected_chapters)
    assert table.get_row_at(table.row_count - 1)[2] == "Ending"
    assert submitted.chapters[-1].title == "Ending"
    assert submitted.chapters[-1].content == tail


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("length", [900, 1000, 1600])
@private_profile_test
async def test_file_import_generate_retains_complete_source(request, tmp_path, length):
    source = _source(length)
    assert len(source) == length
    path = tmp_path / "book.txt"
    path.write_text(source, encoding="utf-8")
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        widget = app.query_one(AudioBookGenerationWidget)
        await _import_file(app, widget, pilot, path)
        # Let the actual Changed debounce and real threaded detector settle.
        await _until(lambda: widget._chapter_detect_debounce_timer is None)
        await app.workers.wait_for_complete()
        await pilot.pause()
        submitted = await _generate(app, widget)
        _assert_full_source(widget, submitted, source)
        assert path.read_text(encoding="utf-8") == source


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("route", ["notes", "conversation"])
@private_profile_test
async def test_service_import_generate_retains_complete_source(
    request, tmp_path, route
):
    source = _source(1600)
    app = _Host()
    db = CharactersRAGDB(tmp_path / "sources.db", client_id="audiobook-import")
    interop = None
    try:
        if route == "notes":
            selected_id = db.add_note("Imported note", source)
            interop = NotesInteropService(
                base_db_directory=tmp_path,
                api_client_id="audiobook-import",
                global_db_to_use=db,
            )
            app.notes_scope_service = NotesScopeService(interop, None)
            app.notes_user_id = "audiobook-import"
            expected = "# Imported note\n\n" + source + "\n\n\n"
            modal_name = "NoteSelectionDialog"
        else:
            service = ChatConversationService(db)
            selected_id = service.create_conversation(title="Imported conversation")
            db.add_message(
                {"conversation_id": selected_id, "sender": "user", "content": source}
            )
            app.chat_conversation_scope_service = ChatConversationScopeService(
                local_service=service, server_service=None
            )
            expected = source + "\n"
            modal_name = "ConversationSelectionDialog"
        async with app.run_test(size=(120, 48)) as pilot:
            widget = app.query_one(AudioBookGenerationWidget)
            widget.query_one(Collapsible).collapsed = False
            widget.query_one("#import-source-select", Select).value = route
            await pilot.pause()
            widget.query_one("#import-content-btn", Button).press()
            await _until(lambda: type(app.screen).__name__ == modal_name)
            await pilot.pause()
            if route == "notes":
                app.screen.dismiss([selected_id])
            else:
                app.screen.dismiss(
                    {
                        "conversation_id": selected_id,
                        "include_all": True,
                        "include_speakers": False,
                    }
                )
            await _until(
                lambda: not widget.query_one("#content-preview", TextArea).disabled
            )
            await pilot.pause()
            await _until(lambda: widget._chapter_detect_debounce_timer is None)
            await app.workers.wait_for_complete()
            await pilot.pause()
            _assert_full_source(
                widget,
                await _generate(app, widget),
                expected,
                tail="KEEP_FINAL_SENTINEL\n\n\n"
                if route == "notes"
                else "KEEP_FINAL_SENTINEL\n",
            )
            if route == "notes":
                assert db.get_note_by_id(selected_id)["content"] == source
            else:
                assert (
                    db.get_messages_for_conversation(selected_id)[0]["content"]
                    == source
                )
    finally:
        if interop is not None:
            for connection in interop._db_instances.values():
                connection.close_connection()
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("change", ["clear", "replace"])
@private_profile_test
async def test_editing_import_before_detection_finishes_owns_generate(
    request, monkeypatch, tmp_path, change
):
    source = _source(1600)
    path = tmp_path / "book.txt"
    path.write_text(source, encoding="utf-8")
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original_detect = ChapterDetector.detect_chapters

    def detect(content):
        chapters = original_detect(content)
        if content == source:
            entered.set()
            assert release.wait(15), "Imported-source detector was not released"
            returned.set()
        return chapters

    monkeypatch.setattr(ChapterDetector, "detect_chapters", staticmethod(detect))
    app = _Host()
    async with app.run_test(size=(120, 48)) as pilot:
        widget = app.query_one(AudioBookGenerationWidget)
        monkeypatch.setattr(widget, "_CHAPTER_DETECT_DEBOUNCE_SECONDS", 60)
        try:
            await _import_file(app, widget, pilot, path)
            await _until(entered.is_set)
            area = widget.query_one("#content-preview", TextArea)
            area.focus()
            await pilot.press("f7", "backspace")
            if change == "replace":
                await pilot.press("n", "e", "w")
            expected = "new" if change == "replace" else ""
            await _until(lambda: widget.content_text == expected)
            release.set()
            await _until(returned.is_set)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert area.text == expected
            assert widget.detected_chapters == []
            editor = widget.query_one("#chapter-editor-widget", ChapterEditorWidget)
            assert editor.get_chapters() == []
            assert [table.row_count for table in editor.query(DataTable)] == [0]
            if change == "replace":
                assert (await _generate(app, widget)).content == "new"
            else:
                widget.query_one("#generate-audiobook-btn", Button).press()
                await pilot.pause()
                assert app.submissions == []
        finally:
            release.set()
            await app.workers.wait_for_complete()


@pytest.mark.asyncio
@pytest.mark.timeout(40)
@pytest.mark.parametrize("change", ["unchanged", "edit", "replace", "clear"])
@private_profile_test
async def test_pending_file_import_keeps_current_source(
    request, tmp_path, monkeypatch, change
):
    """Blind import publication loses accepted input before Changed can run."""
    initial = "Chapter 1: Initial\nINITIAL_DRAFT"
    imported = _source(1600)
    path = tmp_path / "pending-book.txt"
    original_bytes = imported.encode("utf-8")
    path.write_bytes(original_bytes)
    entered, release = asyncio.Event(), asyncio.Event()
    original_to_thread = asyncio.to_thread

    async def hold_read_result(func, *args, **kwargs):
        result = await original_to_thread(func, *args, **kwargs)
        if func.__name__ == "_read_audiobook_text":
            # The original off-loop UTF-8 read has physically completed.
            assert result == imported
            entered.set()
            await release.wait()
        return result

    app = _Host()
    typing = None
    notifications = []
    async with app.run_test(size=(120, 48)) as pilot:
        widget = app.query_one(AudioBookGenerationWidget)
        widget.query_one(Collapsible).collapsed = False
        widget.query_one("#import-source-select", Select).value = "paste"
        await pilot.pause()
        widget.query_one("#import-content-btn", Button).press()
        area = widget.query_one("#content-preview", TextArea)
        await _until(lambda: not area.disabled)
        area.load_text(initial)
        await _until(lambda: widget.content_text == initial)
        await _until(lambda: widget._chapter_detect_debounce_timer is None)
        await app.workers.wait_for_complete()
        await pilot.pause()
        if change in {"replace", "clear"}:
            area.focus()
            await pilot.press("f7")
            assert area.selected_text == initial

        original_notify = app.notify

        def notify(message, **kwargs):
            notifications.append(str(message))
            original_notify(message, **kwargs)

        monkeypatch.setattr(app, "notify", notify)
        monkeypatch.setattr(asyncio, "to_thread", hold_read_result)
        widget.query_one("#import-source-select", Select).value = "file"
        await pilot.pause()
        widget.query_one("#import-content-btn", Button).press()
        await _until(lambda: isinstance(app.screen, FileOpen))
        await _until(lambda: bool(app.screen.query("EnhancedDirectoryNavigation")))
        await pilot.pause()
        app.screen.dismiss(path)
        expected = imported
        try:
            await _until(entered.is_set)
            assert not area.disabled
            area.focus()
            if change in {"edit", "replace"}:
                expected = "x" + initial if change == "edit" else "x"
                typing = asyncio.create_task(pilot.press("x"))
                # Pilot completion waits for the held parent pump. The live
                # child text, rather than typing.done(), proves key delivery.
                await _until(lambda: area.text == expected)
            elif change == "clear":
                expected = ""
                # Run the existing bound keyboard action on the focused child;
                # bubbling a non-printable Key waits behind the held callback.
                await app.run_action("delete_left", default_namespace=area)
                await _until(lambda: area.text == expected)
            if change != "unchanged":
                assert widget.content_text == initial
        finally:
            release.set()
        if typing is not None:
            await typing
        await pilot.pause()
        await _until(lambda: widget._chapter_detect_debounce_timer is None)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert path.read_bytes() == original_bytes
        assert area.text == expected, "pending import overwrote accepted source input"
        assert widget.content_text == expected
        import_notices = [
            notice for notice in notifications if notice.startswith("Imported ")
        ]
        if change == "unchanged":
            _assert_full_source(widget, await _generate(app, widget), imported)
            assert len(import_notices) == 1
        else:
            assert import_notices == []
            if expected:
                assert (await _generate(app, widget)).content == expected
            else:
                widget.query_one("#generate-audiobook-btn", Button).press()
                await pilot.pause()
                assert app.submissions == []
