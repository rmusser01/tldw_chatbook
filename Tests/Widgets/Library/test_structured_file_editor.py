"""Structured file controls retain the existing disk save authority."""

import pytest
from textual.widgets import Select, TextArea

from Tests.UI.test_library_file_notes_workspace import (
    _CssTrueWorkspaceHarness,
    _wait_until,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)


@pytest.mark.asyncio
async def test_invalid_file_saves_and_override_is_temporary(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text('{"a":}', encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(
        root=tmp_path, replica=None, poll_interval=60, autosave_delay=60
    )
    host = _CssTrueWorkspaceHarness(workspace)
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        tools = workspace.query_one("#file-notes-document-tools")
        await _wait_until(
            pilot,
            lambda: "error"
            in str(tools.query_one(".document-check-status").renderable),
            "Check",
        )
        editor = workspace.query_one("#file-notes-editor", TextArea)
        editor.focus()
        await pilot.press("end", "space")
        assert await workspace.flush_pending_work()
        assert path.read_text() == editor.text
        tools.query_one(Select).value = "yaml"
        await pilot.pause()
        assert editor.text == path.read_text()
        assert (
            "temporary"
            in str(tools.query_one(".document-language-scope").renderable).lower()
        )
        assert await workspace.open_path("draft.json")
        assert tools.query_one(Select).value == "json"
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("json", '{"a":1}'),
        ("jsonl", '{"n":900719925474099312345,"s":"a&b <tag> 😀\\u0001"}'),
        ("yaml", "---\nroot:\n    child: a&b <tag> 😀  "),
    ],
)
async def test_file_native_history_and_real_save(tmp_path, language, source):
    from Tests.Widgets.Library.test_structured_note_editor import native_history
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    path = tmp_path / f"draft.{language}"
    path.write_text(source)
    workspace = LibraryFileNotesWorkspace(
        root=tmp_path, replica=None, poll_interval=60, autosave_delay=60
    )
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path(path.name)
        editor = workspace.query_one("#file-notes-editor", TextArea)
        tools = workspace.query_one("#file-notes-document-tools")
        await native_history(pilot, editor, tools, workspace._document_editor)
        assert await workspace.flush_pending_work()
        assert path.read_text() == editor.text == source
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
async def test_file_representation_excerpt_and_hash_conflict(tmp_path):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    path = tmp_path / "draft.json"
    original = b'\xef\xbb\xbf{\r\n"a":1\r\n}\r\n'
    path.write_bytes(original)
    large = tmp_path / "large.json"
    large.write_text(" " * 200_001)
    workspace = LibraryFileNotesWorkspace(
        root=tmp_path, replica=None, poll_interval=60, autosave_delay=60
    )
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        request = workspace._document_editor.request
        assert request.newline == "\r\n" and request.has_bom and not request.is_excerpt
        editor = workspace.query_one("#file-notes-editor", TextArea)
        editor.focus()
        await pilot.press("end", "space")
        path.write_bytes(b'{"external":true}')
        old = workspace._document_editor
        assert not await workspace.flush_pending_work()
        assert workspace.save_state == "conflict"
        assert old.request is None
        assert path.read_bytes() == b'{"external":true}'
        workspace._clear_open_document()
        assert await workspace.open_path("large.json")
        await _wait_until(
            pilot,
            lambda: workspace._document_editor.result is not None,
            "Excerpt check",
        )
        assert workspace.current_document.is_excerpt
        assert editor.read_only
        assert not workspace._document_editor.result.complete
        assert not await workspace._document_editor.format()
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
async def test_successful_format_preserves_bom_crlf_and_native_noop(tmp_path):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    path = tmp_path / "draft.json"
    path.write_bytes(b'\xef\xbb\xbf{"a":1}\r\n')
    workspace = LibraryFileNotesWorkspace(
        root=tmp_path, replica=None, poll_interval=60, autosave_delay=60
    )
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        controller = workspace._document_editor
        await _wait_until(
            pilot,
            lambda: controller.result is not None and controller.result.format_allowed,
            "Ready",
        )
        controller.editor.focus()
        await pilot.press("end")
        assert controller.editor.cursor_location == (0, 7)
        assert await controller.format()
        await pilot.pause()
        assert controller.editor.cursor_location == (2, 1)
        assert workspace.save_state == "dirty"
        assert await workspace.flush_pending_work()
        assert path.read_bytes() == b'\xef\xbb\xbf{\r\n  "a": 1\r\n}\r\n'
        await _wait_until(
            pilot,
            lambda: controller.result is not None and controller.result.format_allowed,
            "Ready after edit",
        )
        revision = workspace._document_revision
        assert not await controller.format()
        await pilot.pause()
        assert workspace._document_revision == revision
        assert workspace.save_state == "saved"
        controller.editor.focus()
        await pilot.press("ctrl+z")
        assert controller.editor.text == '{"a":1}\n'
        await pilot.press("ctrl+y")
        assert controller.editor.text == '{\n  "a": 1\n}\n'
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
async def test_actual_readonly_lease_and_hidden_source_reacquire(tmp_path):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    (tmp_path / "draft.json").write_text('{"a":1}')
    workspace = LibraryFileNotesWorkspace(root=tmp_path, replica=None, poll_interval=60)
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        controller = workspace._document_editor
        readonly = workspace._acquire_editor_read_only(workspace._session_binding)
        assert controller.request is None
        assert workspace._document_editor is None
        assert not await controller.format()
        readonly.release()
        assert workspace._document_editor is not controller
        current = workspace._document_editor
        workspace.display = False
        workspace.synchronize_document_editor()
        assert current.request is None
        workspace.display = True
        workspace.synchronize_document_editor()
        assert workspace._document_editor is not current
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
async def test_file_compact_tools_focus_and_visual_capture(tmp_path, size):
    from pathlib import Path

    from textual.widgets import Collapsible, Static

    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    (tmp_path / "draft.json").write_text('{"a":}')
    workspace = LibraryFileNotesWorkspace(root=tmp_path, replica=None, poll_interval=60)
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        controller = workspace._document_editor
        controller.editor.focus()
        await _wait_until(pilot, lambda: controller.result is not None, "Check")
        tools = workspace.query_one("#file-notes-document-tools")
        status = tools.query_one(".document-check-status", Static)
        problems = tools.query_one(Collapsible)
        await pilot.pause()
        import os

        root = Path(os.environ.get("TLDW_I2_VISUAL_EVIDENCE", str(tmp_path)))
        svg = host.export_screenshot(title=f"Synthetic File Notes {size[0]}x{size[1]}")
        (root / f"files-{size[0]}.svg").write_text(svg)
        assert controller.editor.has_focus
        assert status.region.height > 0
        assert problems.region.height > 0
        assert status.region.bottom <= size[1]
        assert problems.region.bottom <= size[1]
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [b'{\r"a":1\r}', b'{\r\n"a":1\n}'])
async def test_unqualified_file_newlines_stay_readonly(tmp_path, raw):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    path = tmp_path / "draft.json"
    path.write_bytes(raw)
    workspace = LibraryFileNotesWorkspace(root=tmp_path, replica=None, poll_interval=60)
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        assert workspace.current_document.read_only_reason == "mixed-newlines"
        assert workspace._editor_widget.read_only
        assert not await workspace._document_editor.format()
        assert path.read_bytes() == raw
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
async def test_closed_app_service_reports_failed_check_without_touching_file(tmp_path):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    path = tmp_path / "draft.json"
    path.write_text('{"a":1}')
    workspace = LibraryFileNotesWorkspace(root=tmp_path, replica=None, poll_interval=60)
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    await host.document_worker.close()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        await _wait_until(
            pilot,
            lambda: workspace._document_editor.result is not None,
            "Failed service",
        )
        assert workspace._document_editor.result.state == "failed"
        assert not await workspace._document_editor.format()
        assert path.read_text() == '{"a":1}'
    await workspace.shutdown()


@pytest.mark.asyncio
async def test_native_typing_retires_success_and_real_check_waits_for_idle(tmp_path):
    from time import monotonic
    from unittest.mock import patch

    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    (tmp_path / "draft.json").write_text('{"a":1}')
    workspace = LibraryFileNotesWorkspace(root=tmp_path, replica=None, poll_interval=60)
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        controller = workspace._document_editor
        await _wait_until(pilot, lambda: controller.result is not None, "Initial check")
        calls = []
        real_check = host.document_worker.check

        async def observe(request):
            calls.append(monotonic())
            return await real_check(request)

        controller.editor.focus()
        began = monotonic()
        with patch.object(host.document_worker, "check", observe):
            await pilot.press("end", "space")
            assert controller.result is None
            await _wait_until(
                pilot, lambda: controller.result is not None, "Debounced actual check"
            )
        assert len(calls) == 1
        assert calls[0] - began >= 0.38
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
async def test_file_native_format_preserves_nonzero_viewport_and_caret(tmp_path):
    from textual.widgets import Button

    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    source = "{\n" + ",\n".join(f'"k{index}":{index}' for index in range(100)) + "\n}"
    (tmp_path / "draft.json").write_text(source)
    workspace = LibraryFileNotesWorkspace(root=tmp_path, replica=None, poll_interval=60)
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.json")
        controller = workspace._document_editor
        await _wait_until(pilot, lambda: controller.result is not None, "Check")
        editor = controller.editor
        editor.focus()
        await pilot.press(*(["down"] * 95), "end", "left")
        caret = editor.cursor_location
        scroll = editor.scroll_offset
        assert scroll.y > 0
        button = workspace.query_one(".document-format", Button)
        button.focus()
        await pilot.press("enter")
        await _wait_until(pilot, lambda: editor.text != source, "Format")
        assert editor.cursor_location[0] == caret[0]
        assert editor.scroll_offset == scroll
        assert editor.has_focus
        await pilot.press("ctrl+z")
        assert editor.text == source
        await pilot.press("ctrl+y")
        assert editor.text != source
    await workspace.shutdown()
    await host.document_worker.close()


@pytest.mark.asyncio
async def test_jsonl_hidden_bom_is_diagnostic_and_cannot_be_formatted(tmp_path):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    path = tmp_path / "draft.jsonl"
    raw = b"\xef\xbb\xbf{}\r\n"
    path.write_bytes(raw)
    workspace = LibraryFileNotesWorkspace(root=tmp_path, replica=None, poll_interval=60)
    host = _CssTrueWorkspaceHarness(workspace)
    host.document_worker = DocumentWorker()
    async with host.run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "Initialize")
        assert await workspace.open_path("draft.jsonl")
        controller = workspace._document_editor
        await _wait_until(pilot, lambda: controller.result is not None, "BOM check")
        assert controller.request.has_bom
        assert controller.result.diagnostics[0].code == "syntax"
        assert not controller.result.format_allowed
        assert not await controller.format()
        assert path.read_bytes() == raw
    await workspace.shutdown()
    await host.document_worker.close()
