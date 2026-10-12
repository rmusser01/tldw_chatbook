"""Structured editing through the mounted Library and its real SQLite save port."""

from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import Button, Select, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Document_Editing.editor_controller import DocumentEditorController
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.document_diagnostics import DocumentDiagnostics


def database_host(tmp_path: Path, source: str = '{"a":1}'):
    app = _build_test_app()
    app.library_new_profile_admission = False
    app.notes_user_id = "structured-test"
    db = CharactersRAGDB(str(tmp_path / "notes.db"), app.notes_user_id)
    app.chachanotes_db = db
    service = NotesInteropService(tmp_path, app.notes_user_id, global_db_to_use=db)
    app.notes_service = service
    app.notes_scope_service = NotesScopeService(service, None)
    note_id = service.add_note(
        app.notes_user_id, "Structured", source, content_language="json"
    )
    return LibraryProductionCSSHarness(app), service, note_id


async def close_database_host(
    host: LibraryProductionCSSHarness, service: NotesInteropService
) -> None:
    """Retire the exact fixture's worker and all its SQLite thread handles."""
    worker = host.app_instance.document_worker
    await worker.close()
    assert worker._process is None
    assert worker._task is None
    service.close_all_user_connections()
    db = host.app_instance.chachanotes_db
    with db.quiesce_connections(timeout_seconds=2):
        assert db.registered_connection_count() == 0


async def open_note(
    host: LibraryProductionCSSHarness, pilot: Pilot, note_id: str
) -> LibraryScreen:
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_condition(
        pilot,
        lambda: screen._library_selected_row_id == "browse-notes",
        message="Notes route",
    )
    screen._begin_library_note_load(note_id)
    await _wait_for_selector(screen, pilot, "#library-note-body")
    return screen


@pytest.mark.asyncio
async def test_invalid_source_saves_and_language_changes_only_metadata(tmp_path):
    host, service, note_id = database_host(tmp_path, '{"a":}')
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        controls = screen.query_one("#library-note-document-tools")
        await _wait_for_condition(
            pilot,
            lambda: "error"
            in str(controls.query_one(".document-check-status").renderable),
            message="Syntax result",
        )
        body = screen.query_one("#library-note-body", TextArea)
        body.focus()
        await pilot.press("end", "space")
        status = str(controls.query_one(".document-check-status").renderable)
        assert "Pending" in status
        assert "read-only" not in status
        await pilot.pause(0.6)
        assert "error" in str(controls.query_one(".document-check-status").renderable)
        controls.query_one(Select).value = "yaml"
        await pilot.pause()
        source = body.text
        assert screen._library_note_session.snapshot.draft.content_language == "yaml"
        await screen._save_library_note(explicit=True)
        saved = service.get_note_by_id(host.app_instance.notes_user_id, note_id)
        assert saved["content"] == source
        assert saved["content_language"] == "yaml"
    await close_database_host(host, service)


async def native_history(
    pilot: Pilot,
    editor: TextArea,
    tools: DocumentDiagnostics,
    controller: DocumentEditorController,
) -> str:
    """Actual typing/buttons/history, including independent formatting undo."""
    original = editor.text
    editor.focus()
    await pilot.pause()
    assert editor.has_focus
    await pilot.press(*(["down"] * editor.document.line_count), "end")
    assert editor.cursor_location == editor.document.end
    await pilot.press("space")
    before = editor.text
    from textual.widgets import Button

    await _wait_for_condition(
        pilot,
        lambda: controller.result is not None and controller.result.format_allowed,
        message=lambda: f"Qualified check: request={controller.request!r}, result={controller.result!r}",
    )
    button = tools.query_one(".document-format", Button)
    button.focus()
    await pilot.press("enter")
    await _wait_for_condition(
        pilot, lambda: editor.text != before, message="Format transaction"
    )
    formatted = editor.text
    assert editor.has_focus
    await pilot.press(*(["down"] * editor.document.line_count), "end", "space")
    assert editor.text != formatted
    await pilot.press("ctrl+z")
    assert editor.text == formatted
    await pilot.press("ctrl+z")
    assert editor.text == before
    await pilot.press("ctrl+y")
    assert editor.text == formatted
    await pilot.press("ctrl+z")
    assert editor.text == before
    await pilot.press("ctrl+z")
    assert editor.text == original
    return formatted


@pytest.mark.asyncio
async def test_database_native_history_and_real_save(tmp_path):
    host, service, note_id = database_host(tmp_path)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        tools = screen.query_one("#library-note-document-tools")
        body = screen.query_one("#library-note-body", TextArea)
        await native_history(
            pilot, body, tools, screen._notes_controller._document_editor
        )
        await screen._save_library_note(explicit=True)
        assert (
            service.get_note_by_id(host.app_instance.notes_user_id, note_id)["content"]
            == body.text
        )
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_note_switch_and_authority_loss_retire_pending_results(tmp_path):
    from tldw_chatbook.Document_Editing.models import CheckResult

    host, service, note_a = database_host(tmp_path)
    note_b = service.add_note(
        host.app_instance.notes_user_id,
        "B",
        "a&b <tag>",
        content_language="future-mode",
    )
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_a)
        controller = screen._notes_controller._document_editor
        request = controller.request
        result = CheckResult.for_request(request, "checked", True, (), False, True)
        screen._begin_library_note_load(note_b)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_note_session.snapshot is not None
            and screen._library_note_session.snapshot.note_id == note_b,
            message="B opened",
        )
        await _wait_for_selector(screen, pilot, "#library-note-body")
        controller._receive(result)
        current = screen._notes_controller._document_editor
        assert current is not controller
        assert controller.request is None
        assert current.request.language == "future-mode"
        screen._library_screen_suspended = True
        screen._notes_controller.synchronize_document_editor()
        assert current.request is None
        screen._library_screen_suspended = False
        screen._notes_controller.synchronize_document_editor()
        assert screen._notes_controller._document_editor is not current
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_language_change_invalidates_pending_format_with_success_control(
    tmp_path,
):
    from dataclasses import replace

    from tldw_chatbook.Document_Editing.engine import format_document

    host, service, note_id = database_host(tmp_path)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        controller = screen._notes_controller._document_editor
        request = controller.request
        result = format_document(request)
        bridge = controller.bridge
        bridge.capture_request(request)
        before = bridge.editor.text
        changed_key = replace(
            request.key, language_revision=request.key.language_revision + 1
        )
        assert bridge.apply_format(result, changed_key, editable=True) is False
        assert bridge.editor.text == before
        assert bridge.apply_format(result, request.key, editable=True) is True
        await pilot.pause()
        assert screen._library_note_session.snapshot.body == result.replacement
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_metadata_only_conflict_keeps_body_and_retires_format(tmp_path):
    host, service, note_id = database_host(tmp_path)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        tools = screen.query_one("#library-note-document-tools")
        tools.query_one(Select).value = "yaml"
        await pilot.pause()
        snapshot = screen._library_note_session.snapshot
        old = screen._notes_controller._document_editor
        assert service.update_note(
            host.app_instance.notes_user_id,
            note_id,
            {"content_language": "jsonl"},
            snapshot.version,
        )
        await screen._save_library_note(explicit=True)
        assert screen._library_note_session.snapshot.in_conflict
        assert old.request is None
        assert screen._notes_controller._document_editor is None
        assert screen.query_one("#library-note-body", TextArea).text == '{"a":1}'
        screen.query_one("#library-note-conflict-overwrite", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_note_session.snapshot.in_conflict
            and not screen._library_note_session.snapshot.saving
            and service.get_note_by_id(host.app_instance.notes_user_id, note_id)[
                "content_language"
            ]
            == "yaml",
            message="Conflict overwrite persisted",
        )
        saved = service.get_note_by_id(host.app_instance.notes_user_id, note_id)
        assert saved["content"] == '{"a":1}'
        assert saved["content_language"] == "yaml"
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_synchronous_source_readonly_noop_and_local_authority_retirement(
    tmp_path,
):
    from tldw_chatbook.Document_Editing.engine import format_document

    host, service, note_id = database_host(tmp_path)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        controller = screen._notes_controller._document_editor
        body = controller.editor
        request = controller.request
        result = format_document(request)
        controller.bridge.capture_request(request)
        body.insert(" ")
        assert not controller.bridge.apply_format(result, request.key, True)
        await pilot.pause()
        controller.bridge.capture_request(controller.request)
        body.read_only = True
        assert not controller.bridge.apply_format(
            format_document(controller.request), controller.request.key, True
        )
        body.read_only = False
        # Actual canonical owner identity includes the local principal, not a selected remote profile.
        host.app_instance.notes_user_id = "other-principal"
        screen._notes_controller.synchronize_document_editor()
        assert controller.request is None
        host.app_instance.notes_user_id = "structured-test"
        screen._notes_controller.synchronize_document_editor()
        assert screen._notes_controller._document_editor is not controller
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_logical_diagnostic_navigation_and_bounded_states(tmp_path):
    from textual.widgets import Collapsible, OptionList, Static

    from tldw_chatbook.Document_Editing.models import CheckResult, Diagnostic

    source = "😀\tcombining e\u0301\r\nend"
    host, service, note_id = database_host(tmp_path, source)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        controller = screen._notes_controller._document_editor
        request = controller.request
        tools = screen.query_one("#library-note-document-tools")
        body = controller.editor
        await _wait_for_condition(
            pilot,
            lambda: controller.result is not None,
            message="Initial check settled",
        )
        body.focus()
        diagnostics = (
            Diagnostic("syntax", "error", "Invalid source", 1, 2),
            Diagnostic("syntax", "error", "EOF", len(source), len(source)),
        )
        result = CheckResult.for_request(
            request, "checked", True, diagnostics, False, False
        )
        controller._receive(result)
        assert body.has_focus
        assert controller.navigate(0)
        assert body.cursor_location == (0, 1)
        assert controller.navigate(1)
        assert body.cursor_location == body.document.end
        tools.query_one(Collapsible).collapsed = False
        await pilot.pause()
        assert tools.query_one(".document-problem-list", OptionList).option_count == 2
        tools.query_one(".document-problem-list", OptionList).focus()
        await pilot.pause()
        assert host.focused is tools.query_one(".document-problem-list", OptionList)
        from unittest.mock import patch

        with patch.object(
            controller, "navigate", wraps=controller.navigate
        ) as navigation:
            await pilot.press("home", "enter")
            assert navigation.call_count == 1, (
                tools.query_one(".document-problem-list", OptionList).highlighted,
                host.focused,
            )
        assert body.has_focus
        hundred = tuple(
            Diagnostic("syntax", "error", "Invalid source", 0, 0) for _ in range(100)
        )
        controller._receive(
            CheckResult.for_request(request, "checked", False, hundred, True, False)
        )
        assert "Further diagnostics omitted" in str(
            tools.query_one(".document-truncation", Static).renderable
        )
        assert "No syntax errors" not in str(
            tools.query_one(".document-check-status", Static).renderable
        )
        for state in ("unsupported", "failed", "resource_limited"):
            controller._receive(
                CheckResult.for_request(request, state, False, (), False, False)
            )
            assert tools.query_one(".document-format", Button).disabled
        assert body.text == source
    await close_database_host(host, service)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
async def test_notes_compact_tools_focus_and_visual_capture(tmp_path, size):
    from textual.widgets import Collapsible, Static

    host, service, note_id = database_host(tmp_path, '{"a":}')
    async with host.run_test(size=size) as pilot:
        screen = await open_note(host, pilot, note_id)
        controller = screen._notes_controller._document_editor
        body = controller.editor
        body.focus()
        await _wait_for_condition(
            pilot, lambda: controller.result is not None, message="Check"
        )
        assert body.has_focus
        tools = screen.query_one("#library-note-document-tools")
        status = tools.query_one(".document-check-status", Static)
        problems = tools.query_one(Collapsible)
        await pilot.pause()
        import os

        root = Path(os.environ.get("TLDW_I2_VISUAL_EVIDENCE", str(tmp_path)))
        svg = host.export_screenshot(title=f"Synthetic Notes {size[0]}x{size[1]}")
        (root / f"notes-{size[0]}.svg").write_text(svg)
        import json

        print(
            json.dumps(
                {
                    "size": size,
                    "body": str(body.region),
                    "tools": str(tools.region),
                    "problems": str(problems.region),
                    "classes": sorted(problems.classes),
                    "tools_height": str(tools.styles.height),
                    "children": [
                        (
                            str(c),
                            str(c.region),
                            str(c.styles.margin),
                            str(c.styles.padding),
                            str(c.styles.min_height),
                        )
                        for c in tools.children
                    ],
                    "problems_height": str(problems.styles.height),
                    "problems_min_height": str(problems.styles.min_height),
                }
            )
        )
        rows = screen._compositor.render_strips()
        visible_body = "".join(
            rows[y].crop(body.content_region.x, body.content_region.right).text
            for y in range(
                body.content_region.y, min(body.content_region.bottom, size[1] - 1)
            )
        )
        assert body.text in visible_body
        assert body.content_region.height >= 1
        assert body.region.y < size[1] - 1
        before = body.text
        await pilot.press("ctrl+end", "space")
        assert body.text == before + " "
        assert screen._library_note_session.snapshot.body == body.text
        await _wait_for_condition(
            pilot, lambda: controller.result is not None, message="Edited check"
        )
        problems.query_one("CollapsibleTitle").focus()
        await pilot.press("enter")
        option_list = tools.query_one(".document-problem-list")
        option_list.focus()
        await pilot.press("home", "enter")
        await pilot.pause()
        assert problems.collapsed
        assert body.has_focus and body.content_region.height >= 1
        assert status.region.height > 0
        assert problems.region.height > 0
        assert status.region.bottom <= size[1]
        assert problems.region.bottom <= size[1]
    await close_database_host(host, service)


@pytest.mark.asyncio
@pytest.mark.parametrize("language", [None, "future-mode", "default", "json"])
async def test_main_preview_preserves_known_and_unknown_source(tmp_path, language):
    from textual.widgets import Markdown, Static

    source = "**Markdown**\na&b <tag>  "
    host, service, note_id = database_host(tmp_path, source)
    stored = service.get_note_by_id(host.app_instance.notes_user_id, note_id)
    assert service.update_note(
        host.app_instance.notes_user_id,
        note_id,
        {"content_language": language},
        stored["version"],
    )
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        screen.query_one("#library-note-preview", Button).press()
        await pilot.pause()
        markdown = screen.query_one("#library-note-preview-body", Markdown)
        raw = screen.query_one("#library-note-source-preview", Static)
        assert raw.display == (language is not None)
        assert markdown.display == (language is None)
        if language is None:
            assert markdown.source == source
        else:
            assert str(raw.renderable) == source
        assert screen.query_one("#library-note-body", TextArea).text == source
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_manual_check_uses_resource_limits_and_never_vetoes_save(tmp_path):
    host, service, note_id = database_host(tmp_path, " " * (1024 * 1024 + 1))
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        controller = screen._notes_controller._document_editor
        await _wait_for_condition(
            pilot, lambda: controller.result is not None, message="Initial limit"
        )
        assert controller.result.state == "resource_limited"
        controller.check()
        await _wait_for_condition(
            pilot, lambda: controller.result is not None, message="Manual limit"
        )
        assert controller.result.state == "resource_limited"
        assert not await controller.format()
        await screen._save_library_note(explicit=True)
        assert (
            service.get_note_by_id(host.app_instance.notes_user_id, note_id)["content"]
            == controller.editor.text
        )
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_native_language_chooser_keeps_its_own_options(tmp_path):
    host, service, note_id = database_host(tmp_path, '{"a":}')
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        controller = screen._notes_controller._document_editor
        await _wait_for_condition(
            pilot, lambda: controller.result is not None, message="Syntax check"
        )
        selector = screen.query_one("#library-note-document-tools").query_one(Select)
        selector.focus()
        await pilot.press("enter", "up", "enter")
        assert selector.value == "yaml"
        assert screen._library_note_session.snapshot.draft.content_language == "yaml"
        assert controller.editor.text == '{"a":}'
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_database_crlf_formats_without_file_mapping(tmp_path):
    source = '{\r\n"a":1\r\n}\r\n'
    host, service, note_id = database_host(tmp_path, source)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        controller = screen._notes_controller._document_editor
        await _wait_for_condition(
            pilot, lambda: controller.result is not None, message="CRLF check"
        )
        assert controller.editor.text == source
        assert await controller.format()
        await pilot.pause()
        assert controller.editor.text == '{\r\n  "a": 1\r\n}\r\n'
        await _wait_for_condition(
            pilot, lambda: controller.result is not None, message="CRLF formatted check"
        )
        assert not await controller.format()
        await screen._save_library_note(explicit=True)
        assert (
            service.get_note_by_id(host.app_instance.notes_user_id, note_id)["content"]
            == controller.editor.text
        )
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_real_pending_format_cannot_cross_note_navigation(tmp_path):
    import asyncio
    from unittest.mock import patch

    host, service, note_a = database_host(tmp_path)
    note_b = service.add_note(
        host.app_instance.notes_user_id, "B", '{"b":2}', content_language="json"
    )
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_a)
        controller = screen._notes_controller._document_editor
        await _wait_for_condition(
            pilot, lambda: controller.result is not None, message="Qualified A"
        )
        started, release = asyncio.Event(), asyncio.Event()
        real_format = host.app_instance.document_worker.format

        async def paused(request):
            result = await real_format(request)
            started.set()
            await release.wait()
            return result

        with patch.object(host.app_instance.document_worker, "format", paused):
            pending = asyncio.create_task(controller.format())
            await asyncio.wait_for(started.wait(), 3)
            screen._begin_library_note_load(note_b)
            assert await asyncio.wait_for(pending, 1) is False
            release.set()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_note_session.snapshot is not None
                and screen._library_note_session.snapshot.note_id == note_b,
                message="B opened",
            )
            await _wait_for_selector(screen, pilot, "#library-note-body")
            assert screen.query_one("#library-note-body", TextArea).text == '{"b":2}'
            assert controller.request is None
    await close_database_host(host, service)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ['{\n"a":1\n}', '{\r"a":1\r}'])
async def test_database_supported_native_representations(tmp_path, source):
    host, service, note_id = database_host(tmp_path, source)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        body = screen.query_one("#library-note-body", TextArea)
        assert body.text == source
        assert not body.read_only
        assert screen._notes_controller._document_editor is not None
        await screen._save_library_note(explicit=True)
        assert (
            service.get_note_by_id(host.app_instance.notes_user_id, note_id)["content"]
            == source
        )
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_database_lossy_representation_preserves_canonical_source(tmp_path):
    from textual.widgets import Input, Static

    source = '{\r\n"a":1\n}'
    host, service, note_id = database_host(tmp_path, source)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        body = screen.query_one("#library-note-body", TextArea)
        controls = screen.query_one("#library-note-document-tools")
        assert body.read_only and body.disabled and not body.display
        assert screen._notes_controller._document_editor is None
        assert "Unsupported native representation" in str(
            controls.query_one(".document-check-status", Static).renderable
        )
        assert controls.query_one(".document-format", Button).disabled
        assert (
            str(screen.query_one("#library-note-exact-source", Static).renderable)
            == source
        )
        screen.query_one("#library-note-title", Input).value = "Retained title"
        controls.query_one(Select).value = "markdown"
        await pilot.pause()
        assert (
            screen._library_note_session.snapshot.draft.content_language == "markdown"
        )
        # Native queued event and pre-recompose flush cannot import its lossy view.
        screen.handle_library_note_body_changed(TextArea.Changed(body))
        screen._commit_library_note_widgets_before_recompose()
        assert screen._read_library_note_editor_fields()[1] == source
        await screen._save_library_note(explicit=True)
        saved = service.get_note_by_id(host.app_instance.notes_user_id, note_id)
        assert saved["content"] == source and saved["title"] == "Retained title"
        assert saved["content_language"] == "markdown"
        screen._notes_controller.synchronize_document_editor()
        assert screen._notes_controller._document_editor is None
        assert body.read_only
        other = service.add_note(
            host.app_instance.notes_user_id, "Other", '{"b":2}', content_language="json"
        )
        screen._begin_library_note_load(other)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_note_session.snapshot is not None
            and screen._library_note_session.snapshot.note_id == other,
            message="Other note",
        )
        await _wait_for_selector(screen, pilot, "#library-note-body")
        screen.handle_library_note_body_changed(TextArea.Changed(body))
        assert screen._library_note_session.snapshot.body == '{"b":2}'
    await close_database_host(host, service)


@pytest.mark.asyncio
async def test_hidden_database_host_permanently_retires_and_reacquires(tmp_path):
    host, service, note_id = database_host(tmp_path)
    async with host.run_test(size=(120, 40)) as pilot:
        screen = await open_note(host, pilot, note_id)
        old = screen._notes_controller._document_editor
        await _wait_for_condition(
            pilot, lambda: old.result is not None, message="Active check"
        )
        pane = old.editor.parent
        pane.display = False
        screen._notes_controller.synchronize_document_editor()
        assert old.retired and old.request is None
        assert screen._notes_controller._document_editor is None
        pane.display = True
        screen._notes_controller.synchronize_document_editor()
        current = screen._notes_controller._document_editor
        assert current is not old
        await _wait_for_condition(
            pilot, lambda: current.result is not None, message="Restored check"
        )
        assert await current.format()
        assert not await old.format()
    await close_database_host(host, service)
