"""Layout-contract tests for the redesigned Notes workbench screen."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Input, ListView, Select, TextArea

from tldw_chatbook.UI.Screens.notes_screen import NotesScreen
from tldw_chatbook.UI.Screens.notes_scope_models import NotesScreenState


def _mock_app_instance() -> Mock:
    app = Mock()
    service = Mock()
    service.list_notes.return_value = []
    service.get_keywords_for_note.return_value = []
    app.notes_service = service
    app.notes_user_id = "default_user"
    app.notes_scope_service = Mock()
    app.server_notes_workspace_service = Mock()
    app.notify = Mock()
    app.open_study_screen = Mock()
    app.open_notes_workspace = Mock()
    app.push_screen = Mock()
    app.push_screen_wait = AsyncMock(return_value=True)
    app.call_from_thread = Mock()
    app.loguru_logger = Mock()
    app.current_selected_note_id = None
    app.current_selected_note_version = None
    app.current_selected_note_title = ""
    app.current_selected_note_content = ""
    return app


class NotesWorkbenchHarness(App[None]):
    def __init__(self, screen: NotesScreen):
        super().__init__()
        self.screen_under_test = screen
        mock_instance = screen.app_instance
        self.notes_service = mock_instance.notes_service
        self.notes_user_id = "default_user"
        self.notes_scope_service = mock_instance.notes_scope_service
        self.server_notes_workspace_service = mock_instance.server_notes_workspace_service
        self.notify = Mock()
        self.call_from_thread = Mock()
        self.loguru_logger = Mock()
        self.current_selected_note_id = None
        self.current_selected_note_version = None
        self.current_selected_note_title = ""
        self.current_selected_note_content = ""

    def on_mount(self) -> None:
        self.push_screen(self.screen_under_test)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(90, 32), (140, 42), (180, 50)])
async def test_notes_workbench_three_panes_visible_at_supported_sizes(size):
    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()

        navigator = screen.query_one("#notes-navigator-pane")
        editor_pane = screen.query_one("#notes-editor-pane")
        inspector = screen.query_one("#notes-inspector-pane")

        assert navigator.region.x < editor_pane.region.x < inspector.region.x
        for pane in (navigator, editor_pane, inspector):
            assert pane.region.width > 0
            assert pane.region.height > 0
        assert editor_pane.region.width > navigator.region.width
        assert editor_pane.region.width > inspector.region.width

        strip = screen.query_one("#notes-mode-strip")
        assert strip.region.height <= 2
        for chip in screen.query(".notes-mode-chip"):
            assert chip.region.width > 0
            assert chip.region.x >= strip.region.x
            assert chip.region.x + chip.region.width <= strip.region.x + strip.region.width


@pytest.mark.asyncio
async def test_notes_workbench_exposes_feature_parity_selectors():
    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()

        for selector, widget_type in (
            ("#notes-search-input", Input),
            ("#notes-keyword-filter-input", Input),
            ("#notes-sort-select", Select),
            ("#notes-sort-order-button", Button),
            ("#notes-template-select", Select),
            ("#notes-create-from-template-button", Button),
            ("#notes-create-new-button", Button),
            ("#notes-import-button", Button),
            ("#notes-list-view", ListView),
            ("#server-notes-list-view", ListView),
            ("#workspaces-list-view", ListView),
            ("#notes-title-input", Input),
            ("#notes-editor-area", TextArea),
            ("#notes-save-button", Button),
            ("#notes-preview-toggle", Button),
            ("#notes-sync-button", Button),
            ("#notes-keywords-area", TextArea),
            ("#notes-save-current-button", Button),
            ("#notes-use-in-chat-button", Button),
            ("#notes-export-markdown-button", Button),
            ("#notes-export-text-button", Button),
            ("#notes-copy-markdown-button", Button),
            ("#notes-copy-text-button", Button),
            ("#notes-delete-button", Button),
            ("#notes-sidebar-emoji-button", Button),
        ):
            assert screen.query_one(selector, widget_type) is not None, selector


@pytest.mark.asyncio
async def test_notes_mode_strip_switches_regions_and_preserves_editor_text():
    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()

        notes_region = screen.query_one("#notes-mode-region-notes")
        sync_region = screen.query_one("#notes-mode-region-sync")
        templates_region = screen.query_one("#notes-mode-region-templates")

        assert notes_region.display is True
        assert sync_region.display is False
        assert templates_region.display is False

        editor = screen.query_one("#notes-editor-area", TextArea)
        screen._suspend_dirty_tracking = True
        editor.load_text("unsaved draft content")
        screen._suspend_dirty_tracking = False

        screen.query_one("#notes-mode-sync", Button).press()
        await pilot.pause()
        assert notes_region.display is False
        assert sync_region.display is True
        assert screen.query_one("#notes-mode-sync", Button).has_class("is-active")
        assert screen.query_one("#notes-open-sync-modal", Button) is not None

        screen.query_one("#notes-mode-templates", Button).press()
        await pilot.pause()
        assert sync_region.display is False
        assert templates_region.display is True

        screen.query_one("#notes-mode-notes", Button).press()
        await pilot.pause()
        assert notes_region.display is True
        assert screen.query_one("#notes-editor-area", TextArea).text == "unsaved draft content"


@pytest.mark.asyncio
async def test_notes_create_from_template_creates_local_note():
    mock_instance = _mock_app_instance()
    mock_instance.notes_service.add_note.return_value = "note-template-1"
    mock_instance.notes_service.get_note_by_id.return_value = {
        "id": "note-template-1",
        "title": "Template Note",
        "content": "Body",
        "version": 1,
        "keywords": [],
    }
    screen = NotesScreen(mock_instance)
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()

        screen.query_one("#notes-create-from-template-button", Button).press()
        await pilot.pause()

    assert mock_instance.notes_service.add_note.called
    assert screen.state.selected_note_id == "note-template-1"


@pytest.mark.asyncio
async def test_notes_sort_order_button_toggles_and_refreshes():
    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()

        assert screen.state.sort_ascending is False
        button = screen.query_one("#notes-sort-order-button", Button)
        button.press()
        await pilot.pause()
        assert screen.state.sort_ascending is True
        assert "Oldest" in str(button.label)
        button.press()
        await pilot.pause()
        assert screen.state.sort_ascending is False
        assert "Newest" in str(button.label)


@pytest.mark.asyncio
async def test_notes_active_mode_round_trips_through_save_restore():
    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        screen._set_state(active_mode="sync")
        saved = screen.save_state()

    assert saved["notes_state"]["active_mode"] == "sync"

    restored_screen = NotesScreen(_mock_app_instance())
    restored_screen.restore_state(saved)
    assert restored_screen.state.active_mode == "sync"


def test_notes_state_defaults_include_notes_mode():
    assert NotesScreenState().active_mode == "notes"
