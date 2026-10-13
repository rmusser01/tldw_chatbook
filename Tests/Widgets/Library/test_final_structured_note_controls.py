"""Actual import selection/controller and rendered logical Problems regressions."""

import json
from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, OptionList

from Tests.UI.Library_Modules.test_library_note_import_controller import (
    _controller,
    _FolderRepository,
    _plan,
)
from tldw_chatbook.Document_Editing.models import (
    CheckResult,
    Diagnostic,
    DocumentRequest,
    RequestKey,
)
from tldw_chatbook.Notes.note_import_discovery import discover_import_sources
from tldw_chatbook.Notes.note_import_parsers import parse_import_sources
from tldw_chatbook.Notes.note_import_planner import classify_import_batch
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)
from tldw_chatbook.Widgets.document_diagnostics import DocumentDiagnostics
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
)


class ImportHost(App):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.handler_owner = SimpleNamespace(
            _library_note_import_controller=controller,
            _notify_library_note_import_failure=lambda: pytest.fail("Mode rejected"),
        )

    def compose(self) -> ComposeResult:
        yield LibraryNoteImportCanvas(self.controller.presentation_snapshot)

    @on(LibraryNoteImportCanvas.SourceModeToggled)
    def source_mode(self, event):
        LibraryNotesController.handle_library_note_import_source_mode(
            self.handler_owner, event
        )

    @on(LibraryNoteImportCanvas.CheckRequested)
    async def check(self, event):
        event.stop()
        await self.controller.check()


@pytest.mark.asyncio
@pytest.mark.parametrize("extension", ["json", "yaml", "jsonl", "md", "txt"])
@pytest.mark.parametrize("source_mode", [False, True])
async def test_rendered_import_choice_reaches_real_parser(
    tmp_path, extension, source_mode
):
    envelope = {"title": "Container", "content": "literal"}
    if extension == "json":
        envelope.update(format_version="tldw.note.v1", content_language="yaml")
    source = "\ufeff" + json.dumps(envelope) + "\r\n"
    path = tmp_path / f"Data.{extension}"
    path.write_bytes(source.encode("utf-8"))
    controller = _controller(plan=_plan(path), calls=[], repository=_FolderRepository())
    controller._discover = discover_import_sources
    controller._parse = parse_import_sources
    controller._classify = lambda batch, bounds, **kw: classify_import_batch(
        batch, bounds
    )
    controller.accept_selected_path(path)
    controller.set_destination("Inbox")
    host = ImportHost(controller)
    async with host.run_test(size=(100, 35)) as pilot:
        canvas = host.query_one(LibraryNoteImportCanvas)
        controller._publish_snapshot = canvas.sync_state
        assert not controller.snapshot.source_mode
        if source_mode:
            host.query_one("#note-import-source-mode", Button).press()
            await pilot.pause()
            assert controller.snapshot.source_mode
        host.query_one("#note-import-check", Button).press()
        for _ in range(50):
            await pilot.pause(0.02)
            if controller.snapshot.plan is not None:
                break
        plan = controller.snapshot.plan
        assert plan is not None
        payload = plan.items[0].payloads[0]
        if source_mode or extension == "jsonl":
            assert payload.content == source
            assert payload.content_language == {"md": "markdown", "txt": "text"}.get(
                extension, extension
            )
            assert payload.title == "Data"
        elif extension in {"json", "yaml"}:
            assert payload.content == "literal"
            assert payload.title == "Container"
        with pytest.raises(ValueError, match="before checking"):
            controller.set_source_mode(not source_mode)
        assert controller.snapshot.approved_plan is None


class ProblemsHost(App):
    def compose(self) -> ComposeResult:
        yield DocumentDiagnostics()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "offset", "label"),
    [
        ("[\r0,\r]", 5, "3:1"),
        ("[\r\n0,\r\n]", 7, "3:1"),
        ("é😀\rZ", 3, "2:1"),
        ("é😀\r\nZ", 4, "2:1"),
        ("é😀\nZ", 1, "1:2"),
    ],
)
async def test_rendered_problems_use_logical_newlines(source, offset, label):
    request = DocumentRequest(RequestKey("a", "n", 1, 1, 1, 1), source, "json")
    result = CheckResult.for_request(
        request,
        state="checked",
        complete=True,
        diagnostics=(
            Diagnostic("syntax", "error", "Invalid source", offset, offset + 1),
        ),
        format_allowed=False,
        truncated=False,
    )
    host = ProblemsHost()
    async with host.run_test() as pilot:
        controls = host.query_one(DocumentDiagnostics)
        controls.present(request, result, "", True)
        await pilot.pause()
        prompt = str(
            controls.query_one(".document-problem-list", OptionList)
            .get_option_at_index(0)
            .prompt
        )
        assert f"error {label} [syntax]" in prompt
        assert request.text == source
