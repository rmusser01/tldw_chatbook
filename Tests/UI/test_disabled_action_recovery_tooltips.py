from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Audio.transcription_history import TranscriptionEntry
from tldw_chatbook.UI.STTS_Window import STTSWindow
from tldw_chatbook.UI.Views.RAGSearch import search_rag_window as search_rag_module
from tldw_chatbook.UI.Views.RAGSearch.search_rag_window import SearchRAGWindow
from tldw_chatbook.Widgets.chunking_templates_widget import ChunkingTemplatesWidget
from tldw_chatbook.Widgets.template_selector import (
    TemplatePreviewWidget,
    TemplateSelectorDialog,
)
from tldw_chatbook.Widgets.transcription_history_viewer import TranscriptionHistoryViewer


class _WidgetHost(App):
    def __init__(self, widget):
        super().__init__()
        self.widget_under_test = widget

    def compose(self) -> ComposeResult:
        yield self.widget_under_test


class _ScreenHost(App):
    def __init__(self, screen):
        super().__init__()
        self.screen_under_test = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


class _FakeScopeService:
    def __init__(self, templates=None):
        self.templates = list(templates or [])

    async def list_templates(self, *, mode, **kwargs):
        return list(self.templates)


class _FakeAppInstance:
    def __init__(self):
        self.notifications = []

    def notify(self, message, *args, **kwargs):
        self.notifications.append((message, kwargs))

    def get_authoritative_runtime_source(self):
        return "local"


class _ChunkingTemplatesTestApp(App):
    def __init__(self, templates=None):
        super().__init__()
        self.current_runtime_backend = "local"
        self.rag_admin_scope_service = _FakeScopeService(templates)
        self.notify = Mock()
        self.push_screen = Mock()
        self.push_screen_wait = Mock()
        self.media_db = None
        self.widget = ChunkingTemplatesWidget(app_instance=self)

    def compose(self) -> ComposeResult:
        yield self.widget


def _normalized_template(name: str, *, is_builtin: bool = False) -> dict:
    return {
        "record_id": f"local:chunking_template:{name}",
        "backend": "local",
        "backing_template_name": name,
        "name": name,
        "description": f"{name} description",
        "template_json": '{"chunking": {"method": "words"}}',
        "template": {"chunking": {"method": "words"}},
        "is_builtin": is_builtin,
        "tags": [],
        "created_at": "2026-04-20T00:00:00",
        "updated_at": "2026-04-20T00:00:00",
        "version": 1,
        "backing_id": name,
    }


def _assert_button_tooltips(root, expected_tooltips: dict[str, str]) -> None:
    for button_id, expected_tooltip in expected_tooltips.items():
        button = root.query_one(f"#{button_id}", Button)
        assert str(button.tooltip) == expected_tooltip


def _static_text(static: Static) -> str:
    return str(static.renderable)


@pytest.mark.asyncio
async def test_search_rag_missing_embeddings_dependency_exposes_phase_five_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr(search_rag_module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setitem(search_rag_module.DEPENDENCIES_AVAILABLE, "embeddings_rag", False)
    monkeypatch.setattr(
        "tldw_chatbook.Utils.widget_helpers.alert_embeddings_not_available",
        lambda widget: None,
    )

    widget = SearchRAGWindow(_FakeAppInstance())
    app = _WidgetHost(widget)

    async with app.run_test() as pilot:
        await pilot.pause()

        recovery = widget.query_one("#search-rag-dependency-missing", Static)
        recovery_text = _static_text(recovery)
        assert "Dependency missing" in recovery_text
        assert "Unavailable: Search/RAG queries." in recovery_text
        assert "Why: Missing optional dependencies: embeddings_rag." in recovery_text
        assert 'Next: Install with pip install -e ".[embeddings_rag]" and restart.' in recovery_text
        assert "Recovery: Settings > RAG." in recovery_text
        assert "Owner: optional dependency." in recovery_text

        search_input = widget.query_one("#search-query-input", Input)
        search_button = widget.query_one("#search-button", Button)
        assert search_input.disabled is True
        assert search_button.disabled is True
        assert "Search/RAG queries" in str(search_button.tooltip)
        assert 'pip install -e ".[embeddings_rag]"' in str(search_button.tooltip)


@pytest.mark.asyncio
async def test_stts_missing_speech_dependencies_expose_phase_five_recovery(monkeypatch):
    import tldw_chatbook.UI.STTS_Window as stts_module

    monkeypatch.setattr(stts_module, "check_tts_deps", lambda: None)
    monkeypatch.setattr(stts_module, "check_stt_deps", lambda: None)
    monkeypatch.setitem(stts_module.DEPENDENCIES_AVAILABLE, "tts_processing", False)
    monkeypatch.setitem(stts_module.DEPENDENCIES_AVAILABLE, "stt_processing", False)

    widget = STTSWindow(_FakeAppInstance())
    app = _WidgetHost(widget)

    async with app.run_test() as pilot:
        await pilot.pause()

        recovery = widget.query_one("#speech-capability-status", Static)
        recovery_text = _static_text(recovery)
        assert "Dependency missing" in recovery_text
        assert "Unavailable: Local speech providers." in recovery_text
        assert "Why: Missing optional dependencies: local_tts, transcription_faster_whisper, speech_recording." in recovery_text
        assert (
            'Next: Install with pip install "tldw_chatbook[local_tts,transcription_faster_whisper,speech_recording]" '
            "and restart."
        ) in recovery_text
        assert "Recovery: Settings > Speech." in recovery_text
        assert "Owner: optional dependency." in recovery_text
        assert 'pip install "tldw_chatbook[local_tts,transcription_faster_whisper,speech_recording]"' in str(
            recovery.tooltip
        )


@pytest.mark.asyncio
async def test_transcription_history_disabled_actions_explain_selection_requirement(monkeypatch):
    monkeypatch.setattr(TranscriptionHistoryViewer, "on_mount", lambda self: None)
    monkeypatch.setattr(TranscriptionHistoryViewer, "on_select_changed", lambda self, event: None)
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.transcription_history_viewer.get_transcription_history",
        lambda: SimpleNamespace(is_encrypted=lambda: False),
    )

    app = _WidgetHost(TranscriptionHistoryViewer())

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "copy-btn": "Select a transcription entry before copying text.",
                "export-btn": "Select a transcription entry before exporting it.",
                "delete-btn": "Select a transcription entry before deleting it.",
            },
        )

        app.widget_under_test._show_entry_details(
            TranscriptionEntry(
                id="entry-1",
                timestamp=datetime(2026, 4, 20, 9, 0),
                transcript="The selected transcript text.",
                duration=4.2,
                word_count=4,
                language="en",
                provider="test",
            )
        )

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "copy-btn": "Copy the selected transcription text.",
                "export-btn": "Export the selected transcription entry.",
                "delete-btn": "Delete the selected transcription entry.",
            },
        )


@pytest.mark.asyncio
async def test_template_preview_actions_explain_selection_requirement():
    app = _WidgetHost(TemplatePreviewWidget())

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "create-task-btn": "Select an evaluation template before creating a task.",
                "export-template-btn": "Select an evaluation template before exporting it.",
            },
        )

        app.widget_under_test.update_preview(
            {
                "name": "QA Template",
                "description": "Answer quality checks.",
                "category": "quality",
                "difficulty": "medium",
                "task_type": "qa",
            }
        )

        _assert_button_tooltips(
            app.widget_under_test,
            {
                "create-task-btn": "Create an evaluation task from this template.",
                "export-template-btn": "Export this evaluation template.",
            },
        )


@pytest.mark.asyncio
async def test_template_selector_select_action_explains_selection_requirement(monkeypatch):
    monkeypatch.setattr(TemplateSelectorDialog, "_load_templates", lambda self: None)
    app = _ScreenHost(TemplateSelectorDialog())

    async with app.run_test() as pilot:
        await pilot.pause()

        _assert_button_tooltips(
            app.screen_under_test,
            {"select-button": "Select an evaluation template before continuing."},
        )

        app.screen_under_test._on_template_selected(
            {
                "name": "QA Template",
                "description": "Answer quality checks.",
                "category": "quality",
                "difficulty": "medium",
                "task_type": "qa",
            }
        )

        _assert_button_tooltips(
            app.screen_under_test,
            {"select-button": "Use the selected evaluation template."},
        )


@pytest.mark.asyncio
async def test_chunking_template_actions_explain_missing_and_builtin_selection():
    custom_template = _normalized_template("custom-template")
    builtin_template = _normalized_template("builtin-template", is_builtin=True)
    app = _ChunkingTemplatesTestApp(templates=[custom_template, builtin_template])

    async with app.run_test() as pilot:
        await pilot.pause()
        widget = app.widget

        _assert_button_tooltips(
            widget,
            {
                "edit-template-btn": "Select a custom chunking template before editing.",
                "duplicate-template-btn": "Select a chunking template before duplicating it.",
                "delete-template-btn": "Select a custom chunking template before deleting.",
                "export-template-btn": "Select a chunking template before exporting it.",
            },
        )

        widget.selected_template_record_id = custom_template["record_id"]
        await pilot.pause()

        _assert_button_tooltips(
            widget,
            {
                "edit-template-btn": "Edit the selected custom chunking template.",
                "duplicate-template-btn": "Duplicate the selected chunking template.",
                "delete-template-btn": "Delete the selected custom chunking template.",
                "export-template-btn": "Export the selected chunking template.",
            },
        )

        widget.selected_template_record_id = builtin_template["record_id"]
        await pilot.pause()

        _assert_button_tooltips(
            widget,
            {
                "edit-template-btn": "Built-in chunking templates cannot be edited; duplicate it first.",
                "duplicate-template-btn": "Duplicate the selected chunking template.",
                "delete-template-btn": "Built-in chunking templates cannot be deleted.",
                "export-template-btn": "Export the selected chunking template.",
            },
        )
