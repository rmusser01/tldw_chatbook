"""Visible Prompt clipboard and file-picker journeys with real Markdown data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual import events
from textual.screen import Screen
from textual.widgets import Button, Static, TextArea

from Tests.UI.test_library_prompt_action_journeys import _activate_more_action
from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _painted_text,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.Prompt_Management.Prompts_Interop import (
    parse_markdown_prompts_from_content,
)
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileNameInput


def _export_host(tmp_path, theme, kind="legacy"):
    db, service = _real_prompt_scope_service(tmp_path)
    fields = {
        "name": "Export [bold] café",
        "author": "Zoë",
        "details": "A reusable message.",
        "system_prompt": "",
        "user_prompt": "Keep [bold] café.",
        "keywords": ["alpha", "beta"],
    }
    if kind != "legacy":
        fields.update(
            artifact_type=kind,
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition={
                "kind": f"block_{kind}",
                "schema_version": 2,
                "lanes": [
                    {"id": "system", "blocks": []},
                    {
                        "id": "user",
                        "blocks": [
                            {
                                "id": "request",
                                "title": "Request",
                                "syntax": "markdown",
                                "content": "Keep [bold] café.",
                            }
                        ],
                    },
                ],
            },
            user_prompt="# Request\n\nKeep [bold] café.",
        )
    prompt_id, _, _ = db.add_prompt(**fields)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    # The host owns the mounted UI; the underlying TldwCli supplies services.
    app.notify = host.notify
    app.copy_to_clipboard = host.copy_to_clipboard
    return db, prompt_id, app, host


def _assert_round_trip(markdown, original):
    parsed = parse_markdown_prompts_from_content(markdown)
    assert len(parsed) == 1
    for key in ("name", "author", "details"):
        assert parsed[0][key] == original[key]
    # The Markdown parser represents an omitted empty lane as None.
    for key in ("system_prompt", "user_prompt"):
        assert (parsed[0][key] or "") == (original[key] or "")
    assert parsed[0]["keywords"] == ["alpha", "beta"]
    if original["prompt_format"] == "structured":
        assert parsed[0]["artifact_type"] == original["artifact_type"]
        assert parsed[0]["prompt_schema_version"] == 2
        definition = parsed[0]["prompt_definition"]
        if isinstance(definition, str):
            definition = json.loads(definition)
        assert definition == json.loads(original["prompt_definition"])


async def _open_export(host, screen, pilot):
    await _activate_more_action(screen, pilot, "export")
    await _wait_for_condition(
        pilot,
        lambda: (
            isinstance(host.screen, FileSave)
            and isinstance(host.screen.focused, FileNameInput)
        ),
        message="Export did not open with filename focus",
    )
    dialog = host.screen
    filename = dialog.query_one(FileNameInput)
    assert filename.value == "Export bold café.md"
    assert "café.md" in _painted_text(host, filename.region)
    return dialog, filename


async def _save_to(host, dialog, filename, pilot, destination):
    await pilot.press("ctrl+a")
    filename.post_message(events.Paste(str(destination)))
    await pilot.pause()
    assert filename.value == str(destination)
    await pilot.press("tab")
    save = dialog.query_one("#select", Button)
    assert dialog.focused is save
    assert "Save" in _painted_text(host, save.region)
    await pilot.press("enter")


async def _returned_to_export(host, screen, pilot):
    await _wait_for_condition(
        pilot,
        lambda: (
            host.screen is screen
            and screen.focused is screen.query_one("#library-prompt-export", Button)
            and "Export" in _painted_text(host, screen.focused.region)
        ),
        message="Export did not return to its readable opener",
    )


async def _inline_result(host, screen, pilot, expected):
    await _wait_for_condition(
        pilot,
        lambda: screen._prompts_state.status == expected,
        message="Prompt result did not reach the inline status",
    )
    status = screen.query_one("#library-prompt-save-status", Static)
    assert expected in _painted_text(host, status.region)
    assert not list(host._notifications)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["legacy", "prompt", "recipe"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_copy_and_export_round_trip_through_visible_actions(
    tmp_path, kind, size, theme
):
    db, prompt_id, _, host = _export_host(tmp_path, theme, kind)
    original = db.fetch_prompt_details(prompt_id)
    destination = tmp_path / "roundtrip.md"
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        fields = tuple(screen.query(TextArea))

        await _activate_more_action(screen, pilot, "copy")
        _assert_round_trip(host.clipboard, original)
        assert screen.focused.id == "library-prompt-copy"
        await _inline_result(
            host, screen, pilot, "Prompt copied to clipboard as markdown!"
        )
        copied = host.clipboard

        await _open_export(host, screen, pilot)
        await pilot.press("escape")
        await _returned_to_export(host, screen, pilot)
        await _inline_result(host, screen, pilot, "Prompt export cancelled.")
        assert not destination.exists()
        assert tuple(screen.query(TextArea)) == fields

        dialog, filename = await _open_export(host, screen, pilot)
        await _save_to(host, dialog, filename, pilot, destination)
        await _wait_for_condition(
            pilot,
            destination.exists,
            message="Keyboard Save did not write the Markdown file",
        )
        await _returned_to_export(host, screen, pilot)
        await _inline_result(
            host, screen, pilot, "Prompt exported successfully to roundtrip.md"
        )
        written = destination.read_text(encoding="utf-8")
        assert written == copied
        _assert_round_trip(written, original)
        assert tuple(screen.query(TextArea)) == fields
        assert db.fetch_prompt_details(prompt_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_copy_and_export_failures_keep_editor_ready_for_retry(
    tmp_path, monkeypatch, size, theme
):
    db, prompt_id, app, host = _export_host(tmp_path, theme)
    original = db.fetch_prompt_details(prompt_id)
    destination = tmp_path / "retry.md"
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        fields = tuple(screen.query(TextArea))
        clipboard = app.copy_to_clipboard

        def reject_copy(text):
            raise RuntimeError("private failure detail must not be shown")

        for adapter, expected in (
            (None, "Clipboard copy is unavailable in this runtime."),
            (reject_copy, "Error copying prompt: RuntimeError"),
        ):
            app.copy_to_clipboard = adapter
            host._notifications.clear()
            await _activate_more_action(screen, pilot, "copy")
            await _inline_result(host, screen, pilot, expected)
            assert screen.focused.id == "library-prompt-copy"
            assert tuple(screen.query(TextArea)) == fields
        app.copy_to_clipboard = clipboard
        await _activate_more_action(screen, pilot, "copy")
        _assert_round_trip(host.clipboard, original)

        write_text = Path.write_text

        def fail_destination(path, *args, **kwargs):
            if path == destination:
                raise OSError("private failure detail must not be shown")
            return write_text(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "write_text", fail_destination)
            host._notifications.clear()
            dialog, filename = await _open_export(host, screen, pilot)
            await _save_to(host, dialog, filename, pilot, destination)
            await _inline_result(host, screen, pilot, "Error exporting prompt: OSError")
            await _returned_to_export(host, screen, pilot)
            assert not destination.exists()
            assert tuple(screen.query(TextArea)) == fields
        dialog, filename = await _open_export(host, screen, pilot)
        await _save_to(host, dialog, filename, pilot, destination)
        await _wait_for_condition(
            pilot,
            destination.exists,
            message="Export retry did not create the chosen file",
        )
        await _returned_to_export(host, screen, pilot)
        _assert_round_trip(destination.read_text(encoding="utf-8"), original)
        assert db.fetch_prompt_details(prompt_id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["other-prompt", "browse-list", "other-screen"])
async def test_export_result_after_navigation_preserves_current_status(tmp_path, route):
    db, prompt_id, _, host = _export_host(tmp_path, "textual-dark")
    other_id, _, _ = db.add_prompt(
        name="Another prompt", author="", details="", user_prompt="Other body"
    )
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        if route == "other-prompt":
            await _open_prompt_editor(screen, pilot, other_id)
        elif route == "other-screen":
            await host.push_screen(Screen())
            await pilot.pause()
        else:
            await screen._select_library_rail_row("browse-prompts")
            await pilot.pause()
        current_status = screen._prompts_state.status
        host._notifications.clear()
        destination = tmp_path / "earlier.md"
        screen._write_library_prompt_export_file(
            destination, "Earlier prompt", "", "", "", "Earlier body", "", prompt_id
        )
        await pilot.pause()
        assert destination.exists()
        assert screen._prompts_state.status == current_status
        notes = list(host._notifications)
        assert [note.message for note in notes] == [
            "Prompt exported successfully to earlier.md"
        ]
        assert notes[0].severity == "information"
