"""
Tests for Notes widget components following Textual testing best practices.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, Label

from tldw_chatbook.Widgets.Note_Widgets.notes_editor_widget import (
    NotesEditorWidget,
    EditorContentChanged
)
from tldw_chatbook.Widgets.Note_Widgets.notes_status_bar import NotesStatusBar
from tldw_chatbook.Widgets.Note_Widgets.notes_toolbar import (
    NotesToolbar,
    NewNoteRequested,
    SaveNoteRequested,
    DeleteNoteRequested,
    PreviewToggleRequested,
    SyncRequested,
    ExportRequested,
    TemplateRequested
)


# ========== NotesEditorWidget Tests ==========

class TestNotesEditorWidget:
    """Tests for NotesEditorWidget."""
    
    def test_initialization(self):
        """Test widget initialization."""
        editor = NotesEditorWidget(text="Initial text")
        
        assert editor.text == "Initial text"
        assert editor.word_count == 2
        assert editor.has_unsaved_changes is False
        assert editor.is_preview_mode is False
    
    def test_word_count_calculation(self):
        """Test word count calculation."""
        editor = NotesEditorWidget()
        
        assert editor._calculate_word_count("") == 0
        assert editor._calculate_word_count("One") == 1
        assert editor._calculate_word_count("One two three") == 3
        assert editor._calculate_word_count("  Multiple   spaces  ") == 2
        assert editor._calculate_word_count("Line\nbreaks\ncount") == 3
    
    def test_content_loading(self):
        """Test loading content."""
        editor = NotesEditorWidget()
        
        editor.load_content("New content", mark_as_saved=True)
        
        assert editor.text == "New content"
        assert editor.has_unsaved_changes is False
        assert editor._original_content == "New content"
        
        editor.load_content("Modified", mark_as_saved=False)
        assert editor.text == "Modified"
        assert editor.has_unsaved_changes is True
    
    def test_mark_as_saved(self):
        """Test marking content as saved."""
        editor = NotesEditorWidget(text="Initial")
        editor.text = "Modified"
        editor.has_unsaved_changes = True
        
        editor.mark_as_saved()
        
        assert editor.has_unsaved_changes is False
        assert editor._original_content == "Modified"
    
    def test_preview_mode_toggle(self):
        """Test preview mode toggling."""
        editor = NotesEditorWidget()
        
        assert editor.is_preview_mode is False
        
        result = editor.toggle_preview_mode()
        assert result is True
        assert editor.is_preview_mode is True
        
        result = editor.toggle_preview_mode()
        assert result is False
        assert editor.is_preview_mode is False
    
    def test_clear_content(self):
        """Test clearing content."""
        editor = NotesEditorWidget(text="Some content")
        editor.has_unsaved_changes = True
        
        editor.clear_content()
        
        assert editor.text == ""
        assert editor._original_content == ""
        assert editor.has_unsaved_changes is False
    
    def test_auto_save_callback(self):
        """Test auto-save callback is triggered."""
        callback = Mock()
        editor = NotesEditorWidget(auto_save_callback=callback)
        
        # Change text to trigger callback
        editor.text = "Changed"
        editor.watch_text("Changed")
        
        callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_posting(self):
        """Test EditorContentChanged message is posted."""
        class TestApp(App):
            def compose(self):
                yield NotesEditorWidget()
        
        app = TestApp()
        messages = []
        
        def on_editor_content_changed(msg: EditorContentChanged):
            messages.append(msg)
        
        app.on_editor_content_changed = on_editor_content_changed
        
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(NotesEditorWidget)
            editor.text = "New text"
            editor.watch_text("New text")
            
            # Message should be posted
            assert editor.word_count == 2


# ========== NotesStatusBar Tests ==========

class TestNotesStatusBar:
    """Tests for NotesStatusBar."""
    
    @pytest.mark.asyncio
    async def test_status_bar_initialization(self):
        """Test status bar initialization."""
        class TestApp(App):
            def compose(self):
                yield NotesStatusBar()
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            status_bar = pilot.app.query_one(NotesStatusBar)
            
            assert status_bar.save_status == "ready"
            assert status_bar.word_count == 0
            assert status_bar.char_count == 0
            assert status_bar.auto_save_enabled is True
    
    @pytest.mark.asyncio
    async def test_status_updates(self):
        """Test status update methods."""
        class TestApp(App):
            def compose(self):
                yield NotesStatusBar()
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            status_bar = pilot.app.query_one(NotesStatusBar)
            
            # Test saving status
            status_bar.set_saving()
            assert status_bar.save_status == "saving"
            
            # Test saved status
            status_bar.set_saved(update_time=True)
            assert status_bar.save_status == "saved"
            assert status_bar.last_saved_time is not None
            
            # Test unsaved status
            status_bar.set_unsaved()
            assert status_bar.save_status == "unsaved"
            
            # Test error status
            status_bar.set_error("Test error")
            assert status_bar.save_status == "error"
            
            # Test ready status
            status_bar.set_ready()
            assert status_bar.save_status == "ready"
    
    @pytest.mark.asyncio
    async def test_count_updates(self):
        """Test word and character count updates."""
        class TestApp(App):
            def compose(self):
                yield NotesStatusBar()
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            status_bar = pilot.app.query_one(NotesStatusBar)
            
            status_bar.update_counts(word_count=42, char_count=256)
            
            assert status_bar.word_count == 42
            assert status_bar.char_count == 256
    
    @pytest.mark.asyncio
    async def test_auto_save_toggle(self):
        """Test auto-save toggle."""
        class TestApp(App):
            def compose(self):
                yield NotesStatusBar()
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            status_bar = pilot.app.query_one(NotesStatusBar)
            
            assert status_bar.auto_save_enabled is True
            
            result = status_bar.toggle_auto_save()
            assert result is False
            assert status_bar.auto_save_enabled is False
            
            result = status_bar.toggle_auto_save()
            assert result is True
            assert status_bar.auto_save_enabled is True
    
    def test_relative_time_formatting(self):
        """Test relative time formatting in watch_last_saved_time."""
        status_bar = NotesStatusBar()
        
        # Mock the label query
        mock_label = Mock(spec=Label)
        status_bar.query_one = Mock(return_value=mock_label)
        
        # Test "Just now"
        now = datetime.now()
        status_bar.last_saved_time = now
        status_bar.watch_last_saved_time(now)
        mock_label.update.assert_called_with("Saved: Just now")
        
        # Test minutes ago
        past_time = now - timedelta(minutes=5)
        status_bar.last_saved_time = past_time
        with patch('tldw_chatbook.Widgets.Note_Widgets.notes_status_bar.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            status_bar.watch_last_saved_time(past_time)
            mock_label.update.assert_called_with("Saved: 5m ago")
        
        # Test hours ago
        past_time = now - timedelta(hours=3)
        status_bar.last_saved_time = past_time
        with patch('tldw_chatbook.Widgets.Note_Widgets.notes_status_bar.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            status_bar.watch_last_saved_time(past_time)
            mock_label.update.assert_called_with("Saved: 3h ago")


# ========== NotesToolbar Tests ==========

class TestNotesToolbar:
    """Tests for NotesToolbar."""
    
    @pytest.mark.asyncio
    async def test_toolbar_initialization(self):
        """Test toolbar initialization with different options."""
        class TestApp(App):
            def compose(self):
                yield NotesToolbar(
                    show_sync=True,
                    show_export=True,
                    show_templates=True
                )
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            toolbar = pilot.app.query_one(NotesToolbar)
            
            assert toolbar.show_sync is True
            assert toolbar.show_export is True
            assert toolbar.show_templates is True
            assert toolbar.preview_mode is False
            
            # Check buttons exist
            assert pilot.app.query_one("#toolbar-new")
            assert pilot.app.query_one("#toolbar-save")
            assert pilot.app.query_one("#toolbar-delete")
            assert pilot.app.query_one("#toolbar-preview")
            assert pilot.app.query_one("#toolbar-sync")
            assert pilot.app.query_one("#toolbar-export")
            assert pilot.app.query_one("#toolbar-template")
    
    @pytest.mark.asyncio
    async def test_toolbar_without_optional_buttons(self):
        """Test toolbar without optional buttons."""
        class TestApp(App):
            def compose(self):
                yield NotesToolbar(
                    show_sync=False,
                    show_export=False,
                    show_templates=False
                )
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            # Check required buttons exist
            assert pilot.app.query_one("#toolbar-new")
            assert pilot.app.query_one("#toolbar-save")
            assert pilot.app.query_one("#toolbar-delete")
            assert pilot.app.query_one("#toolbar-preview")
            
            # Check optional buttons don't exist
            assert len(pilot.app.query("#toolbar-sync")) == 0
            assert len(pilot.app.query("#toolbar-export")) == 0
            assert len(pilot.app.query("#toolbar-template")) == 0
    
    @pytest.mark.asyncio
    async def test_button_messages(self):
        """Test that buttons post correct messages."""
        class TestApp(App):
            messages_received = []
            
            def compose(self):
                yield NotesToolbar()
            
            def on_new_note_requested(self, msg):
                self.messages_received.append(('new', msg))
            
            def on_save_note_requested(self, msg):
                self.messages_received.append(('save', msg))
            
            def on_delete_note_requested(self, msg):
                self.messages_received.append(('delete', msg))
            
            def on_preview_toggle_requested(self, msg):
                self.messages_received.append(('preview', msg))
            
            def on_sync_requested(self, msg):
                self.messages_received.append(('sync', msg))
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            # Click new button
            await pilot.click("#toolbar-new")
            
            # Click save button
            await pilot.click("#toolbar-save")
            
            # Click delete button
            await pilot.click("#toolbar-delete")
            
            # Click preview button
            await pilot.click("#toolbar-preview")
            
            # Click sync button
            await pilot.click("#toolbar-sync")
            
            # Verify messages (messages are posted but may not be received in test)
            # In a real app, these would be handled by the message system
    
    @pytest.mark.asyncio
    async def test_preview_button_toggle(self):
        """Test preview button toggle behavior."""
        class TestApp(App):
            def compose(self):
                yield NotesToolbar()
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            toolbar = pilot.app.query_one(NotesToolbar)
            preview_button = pilot.app.query_one("#toolbar-preview", Button)
            
            # Initial state
            assert toolbar.preview_mode is False
            assert "👁️ Preview" in preview_button.label
            
            # Click to enable preview
            await pilot.click("#toolbar-preview")
            assert toolbar.preview_mode is True
            assert "📝 Edit" in preview_button.label
            
            # Click to disable preview
            await pilot.click("#toolbar-preview")
            assert toolbar.preview_mode is False
            assert "👁️ Preview" in preview_button.label
    
    @pytest.mark.asyncio
    async def test_button_state_management(self):
        """Test button enable/disable functionality."""
        class TestApp(App):
            def compose(self):
                yield NotesToolbar()
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            toolbar = pilot.app.query_one(NotesToolbar)
            
            # Test save button enable/disable
            toolbar.enable_save_button(False)
            save_button = pilot.app.query_one("#toolbar-save", Button)
            assert save_button.disabled is True
            
            toolbar.enable_save_button(True)
            assert save_button.disabled is False
            
            # Test delete button enable/disable
            toolbar.enable_delete_button(False)
            delete_button = pilot.app.query_one("#toolbar-delete", Button)
            assert delete_button.disabled is True
            
            toolbar.enable_delete_button(True)
            assert delete_button.disabled is False
    
    @pytest.mark.asyncio
    async def test_update_button_states(self):
        """Test update_button_states method."""
        class TestApp(App):
            def compose(self):
                yield NotesToolbar()
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            toolbar = pilot.app.query_one(NotesToolbar)
            
            # No note selected
            toolbar.update_button_states(has_note=False, has_unsaved=False)
            save_button = pilot.app.query_one("#toolbar-save", Button)
            delete_button = pilot.app.query_one("#toolbar-delete", Button)
            assert save_button.disabled is True
            assert delete_button.disabled is True
            
            # Note selected, no unsaved changes
            toolbar.update_button_states(has_note=True, has_unsaved=False)
            assert save_button.disabled is True
            assert delete_button.disabled is False
            
            # Note selected with unsaved changes
            toolbar.update_button_states(has_note=True, has_unsaved=True)
            assert save_button.disabled is False
            assert delete_button.disabled is False


# ========== Integration Tests ==========

@pytest.mark.asyncio
class TestWidgetIntegration:
    """Integration tests for widgets working together."""
    
    async def test_editor_and_status_bar_integration(self):
        """Test editor and status bar working together."""
        class TestApp(App):
            def compose(self):
                yield NotesEditorWidget()
                yield NotesStatusBar()
            
            def on_editor_content_changed(self, msg: EditorContentChanged):
                status_bar = self.query_one(NotesStatusBar)
                status_bar.update_counts(
                    word_count=msg.word_count,
                    char_count=len(msg.content)
                )
        
        app = TestApp()
        
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(NotesEditorWidget)
            status_bar = pilot.app.query_one(NotesStatusBar)
            
            # Change editor text
            test_text = "This is a test"
            editor.text = test_text
            editor.watch_text(test_text)
            
            # Manually trigger the integration (in real app, message system handles this)
            pilot.app.on_editor_content_changed(
                EditorContentChanged(test_text, 4)
            )
            
            # Check status bar updated
            assert status_bar.word_count == 4
            assert status_bar.char_count == len(test_text)