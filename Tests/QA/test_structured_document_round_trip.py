"""Release controls through native editors and their existing persistence owners."""

from pathlib import Path

import pytest
from textual.widgets import Button, Select, Static, TextArea

from Tests.UI.test_library_shell import _wait_for_condition
from Tests.Widgets.Library.test_structured_note_editor import (
    close_database_host,
    database_host,
    native_history,
    open_note,
)
from tldw_chatbook.Widgets.document_diagnostics import DocumentDiagnostics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("jsonl", '{"n":900719925474099312345,"s":"a&b <tag> 😀\\u0001"}'),
        ("yaml", "---\nroot:\n    child: a&b <tag> 😀  "),
    ],
)
async def test_database_release_native_history_exact_save(
    tmp_path: Path, language: str, source: str
) -> None:
    host, service, note_id = database_host(tmp_path, source)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        tools = screen.query_one("#library-note-document-tools", DocumentDiagnostics)
        tools.query_one(Select).value = language
        await pilot.pause()
        editor = screen.query_one("#library-note-body", TextArea)
        controller = screen._notes_controller._document_editor
        assert controller is not None
        await native_history(pilot, editor, tools, controller)
        assert editor.text == source
        await screen._save_library_note(explicit=True)
        saved = service.get_note_by_id(host.app_instance.notes_user_id, note_id)
        assert (saved["content"], saved["content_language"]) == (source, language)
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_already_formatted_status_survives_check_presentation(
    tmp_path: Path,
) -> None:
    host, service, note_id = database_host(tmp_path, '{\n  "a": 1\n}')
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        controller = screen._notes_controller._document_editor
        assert controller is not None
        tools = screen.query_one("#library-note-document-tools", DocumentDiagnostics)
        await _wait_for_condition(
            pilot,
            lambda: controller.result is not None and controller.result.format_allowed,
            message="Complete check",
        )
        tools.query_one(".document-format", Button).press()
        await pilot.pause(0.4)
        assert "Already formatted" in str(
            tools.query_one(".document-check-status", Static).content
        )
    await close_database_host(host, service)
