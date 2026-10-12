"""Real Library actions write exact canonical source through the native picker."""

import json

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_shell import _wait_for_condition
from Tests.Widgets.Library.test_structured_note_editor import database_host, open_note
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import InputBar

SOURCE = '{"x":9007199254740993}\n'


@pytest.mark.asyncio
@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("source", [SOURCE, "\ufeffa\r\nb\nc\r"])
async def test_current_canonical_source_and_portable_note_through_file_dialog(
    tmp_path, portable, source
):
    host, service, note_id = database_host(tmp_path, source)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = await open_note(host, pilot, note_id)
            screen.query_one("#library-note-context", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: screen.query_one("#library-note-context-region").display,
                message="Info actions visible",
            )
            await pilot.pause()
            action = "portable" if portable else "source"
            button = screen.query_one(f"#library-note-context-export-{action}", Button)
            assert button.region.width > 0 and button.region.height > 0
            assert str(button.label) == (
                "Export note" if portable else "Download source"
            )
            (tmp_path / "actions.svg").write_text(host.export_screenshot(simplify=True))
            button.press()
            await _wait_for_condition(
                pilot, lambda: isinstance(host.screen, FileSave), message="Save picker"
            )
            assert len([s for s in host.screen_stack if isinstance(s, FileSave)]) == 1
            suffix = ".note.json" if portable else ".json"
            assert host.screen._default_file == f"Structured{suffix}"
            target = tmp_path / f"download{suffix}"
            await _wait_for_condition(
                pilot,
                lambda: bool(host.screen.query("InputBar Input")),
                message="Picker mounted",
            )
            (tmp_path / "picker.svg").write_text(host.export_screenshot(simplify=True))
            screen._library_note_session.mutate(
                body="later source", content_language="yaml"
            )
            filename = host.screen.query_one(InputBar).query_one(Input)
            filename.value = str(target)
            filename.focus()
            await pilot.pause()
            await pilot.press("enter")
            await _wait_for_condition(pilot, target.exists, message="Export file")
            data = target.read_bytes()
            if portable:
                envelope = json.loads(data)
                assert envelope == {
                    "format_version": "tldw.note.v1",
                    "id": note_id,
                    "title": "Structured",
                    "content": source,
                    "content_language": "json",
                    "keywords": [],
                }
            else:
                assert data == source.encode("utf-8")
    finally:
        await host.app_instance.document_worker.close()
        service.close_all_user_connections()
        host.app_instance.chachanotes_db.close()


@pytest.mark.asyncio
async def test_source_action_keeps_transfer_interlock(tmp_path):
    host, service, note_id = database_host(tmp_path, SOURCE)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = await open_note(host, pilot, note_id)
            operation = screen._notes_controller._begin_library_notes_operation("copy")
            await screen._notes_controller._export_library_note("source")
            assert not isinstance(host.screen, FileSave)
            assert screen._notes_controller._library_notes_operation == operation
    finally:
        await host.app_instance.document_worker.close()
        service.close_all_user_connections()
        host.app_instance.chachanotes_db.close()
