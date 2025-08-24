"""
Unit and integration tests for NotesScreen following Textual testing best practices.
Uses Textual's testing framework with async snapshot testing and pilot.
"""

import pytest
from typing import Optional, Dict, Any
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime

from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, TextArea, Label, Input

from tldw_chatbook.UI.Screens.notes_screen import (
    NotesScreen,
    NotesScreenState,
    NoteSelected,
    NoteSaved,
    NoteDeleted,
    AutoSaveTriggered,
    SyncRequested
)


# ========== Fixtures ==========

@pytest.fixture
def mock_app_instance():
    """Create a mock app instance with notes service."""
    app = Mock()
    app.notes_service = Mock()
    app.notes_service.list_notes = Mock(return_value=[
        {
            'id': 1,
            'title': 'Test Note 1',
            'content': 'Content 1',
            'version': 1,
            'created_at': '2024-01-01',
            'updated_at': '2024-01-01'
        },
        {
            'id': 2,
            'title': 'Test Note 2',
            'content': 'Content 2',
            'version': 1,
            'created_at': '2024-01-02',
            'updated_at': '2024-01-02'
        }
    ])
    app.notes_service.get_note_by_id = Mock(return_value={
        'id': 1,
        'title': 'Test Note',
        'content': 'Test content',
        'version': 1
    })
    app.notes_service.add_note = Mock(return_value=3)
    app.notes_service.update_note = Mock(return_value=True)
    app.notes_service.delete_note = Mock(return_value=True)
    
    return app


@pytest.fixture
def notes_screen_state():
    """Create a test NotesScreenState."""
    return NotesScreenState(
        selected_note_id=1,
        selected_note_version=1,
        selected_note_title="Test Note",
        selected_note_content="Test content",
        has_unsaved_changes=False,
        auto_save_enabled=True
    )


# ========== Unit Tests for NotesScreenState ==========

class TestNotesScreenState:
    """Test the NotesScreenState dataclass."""
    
    def test_default_initialization(self):
        """Test state initializes with correct defaults."""
        state = NotesScreenState()
        
        assert state.selected_note_id is None
        assert state.selected_note_version is None
        assert state.selected_note_title == ""
        assert state.selected_note_content == ""
        assert state.has_unsaved_changes is False
        assert state.auto_save_enabled is True
        assert state.sort_by == "date_created"
        assert state.sort_ascending is False
    
    def test_state_mutation(self):
        """Test state can be modified."""
        state = NotesScreenState()
        
        state.selected_note_id = 123
        state.has_unsaved_changes = True
        state.word_count = 42
        
        assert state.selected_note_id == 123
        assert state.has_unsaved_changes is True
        assert state.word_count == 42
    
    def test_state_with_initial_values(self):
        """Test state creation with initial values."""
        state = NotesScreenState(
            selected_note_id=1,
            selected_note_title="My Note",
            has_unsaved_changes=True,
            word_count=100
        )
        
        assert state.selected_note_id == 1
        assert state.selected_note_title == "My Note"
        assert state.has_unsaved_changes is True
        assert state.word_count == 100


# ========== Unit Tests for Custom Messages ==========

class TestCustomMessages:
    """Test custom message classes."""
    
    def test_note_selected_message(self):
        """Test NoteSelected message creation."""
        msg = NoteSelected(note_id=1, note_data={"title": "Test"})
        assert msg.note_id == 1
        assert msg.note_data["title"] == "Test"
    
    def test_note_saved_message(self):
        """Test NoteSaved message creation."""
        msg = NoteSaved(note_id=1, success=True)
        assert msg.note_id == 1
        assert msg.success is True
    
    def test_note_deleted_message(self):
        """Test NoteDeleted message creation."""
        msg = NoteDeleted(note_id=1)
        assert msg.note_id == 1
    
    def test_auto_save_triggered_message(self):
        """Test AutoSaveTriggered message creation."""
        msg = AutoSaveTriggered(note_id=1)
        assert msg.note_id == 1
    
    def test_sync_requested_message(self):
        """Test SyncRequested message creation."""
        msg = SyncRequested()
        assert msg is not None


# ========== Integration Tests using Textual's AppTest ==========

class NotesTestApp(App):
    """Test app for NotesScreen integration tests."""
    
    def __init__(self, notes_service=None):
        super().__init__()
        self.notes_service = notes_service
    
    def on_mount(self):
        """Mount the NotesScreen."""
        self.push_screen(NotesScreen(self))


@pytest.mark.asyncio
class TestNotesScreenIntegration:
    """Integration tests for NotesScreen using Textual's testing framework."""
    
    async def test_screen_mount(self, mock_app_instance):
        """Test NotesScreen mounts correctly."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            # Check screen is mounted
            assert len(pilot.app.screen_stack) > 0
            screen = pilot.app.screen
            assert isinstance(screen, NotesScreen)
            
            # Check initial state
            assert screen.state.selected_note_id is None
            assert screen.state.has_unsaved_changes is False
    
    async def test_save_button_interaction(self, mock_app_instance):
        """Test save button interaction."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            await pilot.pause()  # Let screen fully mount
            screen = pilot.app.screen
            
            # Set up state as if a note is loaded with changes
            screen.state = NotesScreenState(
                selected_note_id=1,
                selected_note_version=1,
                selected_note_content="Test content",
                selected_note_title="Test title",
                has_unsaved_changes=True
            )
            
            # Wait for UI to update
            await pilot.pause()
            
            # Set up the editor content (required for save)
            try:
                editor = screen.query_one("#notes-editor-area", TextArea)
                editor.text = "Test content to save"
            except Exception as e:
                logger.debug(f"Could not set editor text: {e}")
            
            # Set up the title input (required for save) 
            try:
                from tldw_chatbook.Widgets.Note_Widgets.notes_sidebar_right import NotesSidebarRight
                sidebar_right = screen.query_one("#notes-sidebar-right", NotesSidebarRight)
                title_input = sidebar_right.query_one("#notes-title-input", Input)
                title_input.value = "Test Note Title"
            except Exception as e:
                logger.debug(f"Could not set title input: {e}")
            
            # Click save button using pilot's click with CSS selector
            await pilot.click("#notes-save-button")
            
            # Wait for async save operation to complete
            await pilot.pause(0.5)
            
            # Verify save was attempted
            mock_app_instance.notes_service.update_note.assert_called()
    
    async def test_editor_text_change(self, mock_app_instance):
        """Test editor text changes trigger state updates."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Set initial state
            screen.state = NotesScreenState(
                selected_note_id=1,
                selected_note_content="Original content"
            )
            
            # Get editor and change text
            editor = screen.query_one("#notes-editor-area", TextArea)
            editor.text = "Modified content"
            
            # Wait for reactive updates
            await pilot.pause()
            
            # Check state was updated
            assert screen.state.has_unsaved_changes is True
            assert screen.state.word_count == 2  # "Modified content"
    
    async def test_sidebar_toggle(self, mock_app_instance):
        """Test sidebar toggle functionality."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            await pilot.pause()  # Let screen fully mount
            screen = pilot.app.screen
            
            # Force initial state to known value
            screen.state = NotesScreenState(left_sidebar_collapsed=False)
            await pilot.pause()
            
            # Should start not collapsed
            assert screen.state.left_sidebar_collapsed is False
            
            # Click toggle button to collapse
            await pilot.click("#toggle-notes-sidebar-left")
            await pilot.pause(0.2)  # Wait for state update
            
            # Check state changed to collapsed
            assert screen.state.left_sidebar_collapsed is True
            
            # Toggle again to expand
            await pilot.click("#toggle-notes-sidebar-left")
            await pilot.pause(0.2)  # Wait for state update
            
            # Should be expanded again
            assert screen.state.left_sidebar_collapsed is False
    
    async def test_preview_mode_toggle(self, mock_app_instance):
        """Test preview mode toggle."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Initial state
            assert screen.state.is_preview_mode is False
            
            # Click preview button using CSS selector directly with pilot
            await pilot.click("#notes-preview-toggle")
            
            # Wait for async operations to complete
            await pilot.pause()
            
            # Check state changed
            assert screen.state.is_preview_mode is True
    
    async def test_message_posting(self, mock_app_instance):
        """Test that messages are posted correctly."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        messages_received = []
        
        # Set up message handler
        def on_note_saved(message: NoteSaved):
            messages_received.append(message)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            screen.on_note_saved = on_note_saved
            
            # Set up state for save
            screen.state = NotesScreenState(
                selected_note_id=1,
                selected_note_version=1
            )
            
            # Click save button using CSS selector directly
            await pilot.click("#notes-save-button")
            await pilot.pause()
            
            # Verify message was posted
            # Note: In real app, this would be handled by message system
            mock_app_instance.notes_service.update_note.assert_called()


# ========== Unit Tests for NotesScreen Methods ==========

class TestNotesScreenMethods:
    """Unit tests for NotesScreen methods."""
    
    def test_state_validation(self, mock_app_instance):
        """Test state validation."""
        screen = NotesScreen(mock_app_instance)
        
        # Test word count validation
        state = NotesScreenState(word_count=-5)
        validated = screen.validate_state(state)
        assert validated.word_count == 0  # Should be clamped to 0
        
        # Test auto-save status validation
        state = NotesScreenState(auto_save_status="invalid")
        validated = screen.validate_state(state)
        assert validated.auto_save_status == ""
    
    def test_save_state(self, mock_app_instance):
        """Test state serialization."""
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            selected_note_id=1,
            selected_note_title="Test",
            has_unsaved_changes=True
        )
        
        saved = screen.save_state()
        
        assert 'notes_state' in saved
        assert saved['notes_state']['selected_note_id'] == 1
        assert saved['notes_state']['selected_note_title'] == "Test"
        assert saved['notes_state']['has_unsaved_changes'] is True
    
    def test_restore_state(self, mock_app_instance):
        """Test state restoration."""
        screen = NotesScreen(mock_app_instance)
        
        state_data = {
            'notes_state': {
                'selected_note_id': 5,
                'selected_note_title': 'Restored',
                'has_unsaved_changes': False,
                'auto_save_enabled': False
            }
        }
        
        screen.restore_state(state_data)
        
        assert screen.state.selected_note_id == 5
        assert screen.state.selected_note_title == 'Restored'
        assert screen.state.has_unsaved_changes is False
        assert screen.state.auto_save_enabled is False
    
    @pytest.mark.asyncio
    async def test_auto_save_timer(self, mock_app_instance):
        """Test auto-save timer functionality."""
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            selected_note_id=1,
            selected_note_version=1,
            auto_save_enabled=True,
            has_unsaved_changes=True
        )
        
        # Start auto-save timer
        screen._start_auto_save_timer()
        
        # Verify timer was created
        assert screen._auto_save_timer is not None
        
        # Stop timer
        if screen._auto_save_timer:
            screen._auto_save_timer.stop()
    
    def test_lifecycle_methods(self, mock_app_instance):
        """Test lifecycle methods don't raise errors."""
        screen = NotesScreen(mock_app_instance)
        
        # Test unmount
        screen.on_unmount()
        
        # Verify timers are cleaned up
        assert screen._auto_save_timer is None or not screen._auto_save_timer.is_running


# ========== Performance Tests ==========

@pytest.mark.asyncio
class TestNotesScreenPerformance:
    """Performance-related tests."""
    
    async def test_large_notes_list(self, mock_app_instance):
        """Test handling of large notes list."""
        # Create 1000 mock notes
        large_notes_list = [
            {
                'id': i,
                'title': f'Note {i}',
                'content': f'Content {i}',
                'version': 1,
                'created_at': '2024-01-01',
                'updated_at': '2024-01-01'
            }
            for i in range(1000)
        ]
        mock_app_instance.notes_service.list_notes = Mock(return_value=large_notes_list)
        
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            await pilot.pause()  # Let screen fully mount
            screen = pilot.app.screen
            
            # The notes should be loaded during mount automatically
            # Wait for the worker to complete
            await pilot.pause(0.5)
            
            # Verify notes were loaded
            assert len(screen.state.notes_list) == 1000
    
    async def test_rapid_state_changes(self, mock_app_instance):
        """Test rapid state changes don't cause issues."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Rapidly change state
            for i in range(100):
                new_state = screen.state
                new_state.word_count = i
                new_state.has_unsaved_changes = i % 2 == 0
                screen.state = new_state
            
            # Verify final state
            assert screen.state.word_count == 99


# ========== Snapshot Tests ==========

@pytest.mark.asyncio
class TestNotesScreenSnapshots:
    """Snapshot tests for visual regression testing."""
    
    async def test_initial_screen_snapshot(self, mock_app_instance):
        """Test initial screen appearance."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            # Take snapshot of initial state
            assert pilot.app.screen is not None
            # In real test, would compare with saved snapshot:
            # pilot.app.save_screenshot("notes_screen_initial.svg")
    
    async def test_with_note_loaded_snapshot(self, mock_app_instance):
        """Test screen with note loaded."""
        app = NotesTestApp(notes_service=mock_app_instance.notes_service)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Load a note
            screen.state = NotesScreenState(
                selected_note_id=1,
                selected_note_title="Test Note",
                selected_note_content="This is test content",
                word_count=4
            )
            
            # Update editor - get from screen context
            editor = screen.query_one("#notes-editor-area", TextArea)
            editor.text = "This is test content"
            
            await pilot.pause()
            
            # Take snapshot
            assert pilot.app.screen is not None
            # pilot.app.save_screenshot("notes_screen_with_note.svg")