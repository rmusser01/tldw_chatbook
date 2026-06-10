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
        assert screen.query_one("#quick-sync-btn", Button) is not None

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


@pytest.mark.asyncio
async def test_notes_sync_pane_runs_sync_through_service():
    from tldw_chatbook.Widgets.Note_Widgets.notes_workbench_panes import NotesSyncPane
    from textual.widgets import Input

    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        screen.query_one("#notes-mode-sync", Button).press()
        await pilot.pause()

        pane = screen.query_one("#notes-sync-pane", NotesSyncPane)

        sync_button = pane.query_one("#quick-sync-btn", Button)
        sync_region = screen.query_one("#notes-mode-region-sync").region
        assert sync_button.region.height > 0
        assert (
            sync_button.region.y + sync_button.region.height
            <= sync_region.y + sync_region.height
        ), "Sync Now must be visible without scrolling"

        class FakeProgress:
            total_files = 2
            processed_files = 2
            conflicts = []
            errors = []
            created_notes = ["a"]
            updated_notes = []
            created_files = []
            updated_files = ["b"]

        sync_service = Mock()
        sync_service.sync_folder = AsyncMock(return_value=("session-1", FakeProgress()))
        pane.sync_service = sync_service

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            pane.query_one("#sync-folder-input", Input).value = tmp_dir
            pane.query_one("#quick-sync-btn", Button).press()
            await pilot.pause(0.2)

        assert sync_service.sync_folder.await_count == 1
        kwargs = sync_service.sync_folder.await_args.kwargs
        assert kwargs["user_id"] == "default_user"
        activity_text = " ".join(
            str(child.render()) for child in pane.query_one("#activity-section").children
        )
        assert "Sync complete" in activity_text
        assert "1 notes created" in activity_text


@pytest.mark.asyncio
async def test_notes_sync_button_switches_to_sync_mode_without_modal():
    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()

        screen.query_one("#notes-sync-button", Button).press()
        await pilot.pause()

        assert screen.state.active_mode == "sync"
        assert screen.query_one("#notes-mode-region-sync").display is True


@pytest.mark.asyncio
async def test_notes_rails_collapse_to_console_style_handles():
    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()

        navigator = screen.query_one("#notes-navigator-pane")
        inspector = screen.query_one("#notes-inspector-pane")
        navigator_handle = screen.query_one("#notes-navigator-rail-handle")
        inspector_handle = screen.query_one("#notes-inspector-rail-handle")
        editor_pane = screen.query_one("#notes-editor-pane")

        assert navigator.display is True
        assert inspector.display is True
        assert navigator_handle.display is False
        assert inspector_handle.display is False
        editor_width_open = editor_pane.region.width

        screen.query_one("#notes-navigator-rail-collapse", Button).press()
        screen.query_one("#notes-inspector-rail-collapse", Button).press()
        await pilot.pause()

        assert screen.state.left_sidebar_collapsed is True
        assert screen.state.right_sidebar_collapsed is True
        assert navigator.display is False
        assert inspector.display is False
        assert navigator_handle.display is True
        assert inspector_handle.display is True
        assert editor_pane.region.width > editor_width_open

        screen.query_one("#notes-navigator-rail-open", Button).press()
        screen.query_one("#notes-inspector-rail-open", Button).press()
        await pilot.pause()

        assert navigator.display is True
        assert inspector.display is True
        assert navigator_handle.display is False
        assert inspector_handle.display is False


@pytest.mark.asyncio
async def test_notes_rail_collapse_state_round_trips_through_save_restore():
    screen = NotesScreen(_mock_app_instance())
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        screen.query_one("#notes-navigator-rail-collapse", Button).press()
        await pilot.pause()
        saved = screen.save_state()

    assert saved["notes_state"]["left_sidebar_collapsed"] is True

    restored_screen = NotesScreen(_mock_app_instance())
    restored_screen.restore_state(saved)
    restored_app = NotesWorkbenchHarness(restored_screen)
    async with restored_app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        assert restored_screen.query_one("#notes-navigator-pane").display is False
        assert restored_screen.query_one("#notes-navigator-rail-handle").display is True


@pytest.mark.asyncio
async def test_notes_templates_pane_lists_previews_and_creates():
    from tldw_chatbook.Widgets.Note_Widgets.notes_workbench_panes import NotesTemplatesPane
    from textual.widgets import ListView

    mock_instance = _mock_app_instance()
    mock_instance.notes_service.add_note.return_value = "tpl-note-1"
    mock_instance.notes_service.get_note_by_id.return_value = {
        "id": "tpl-note-1",
        "title": "Templated",
        "content": "Body",
        "version": 1,
        "keywords": [],
    }
    screen = NotesScreen(mock_instance)
    app = NotesWorkbenchHarness(screen)
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        screen.query_one("#notes-mode-templates", Button).press()
        await pilot.pause()

        pane = screen.query_one("#notes-templates-pane", NotesTemplatesPane)
        template_list = pane.query_one("#notes-templates-list", ListView)
        assert len(template_list.children) > 0

        template_list.index = 0
        await pilot.pause()
        assert pane.selected_template_key is not None

        pane.query_one("#notes-templates-create-button", Button).press()
        await pilot.pause(0.2)

        assert mock_instance.notes_service.add_note.called
        assert screen.state.active_mode == "notes"
        assert screen.state.selected_note_id == "tpl-note-1"
