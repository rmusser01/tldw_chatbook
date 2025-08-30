"""Toolbar widget for notes screen actions."""

from typing import Optional
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button
from textual.message import Message
from textual import on


# Custom messages for toolbar actions
class NewNoteRequested(Message):
    """Message when new note is requested."""
    pass


class SaveNoteRequested(Message):
    """Message when save is requested."""
    pass


class DeleteNoteRequested(Message):
    """Message when delete is requested."""
    pass


class PreviewToggleRequested(Message):
    """Message when preview toggle is requested."""
    pass


class SyncRequested(Message):
    """Message when sync is requested."""
    pass


class ExportRequested(Message):
    """Message when export is requested."""
    def __init__(self, format: str = "markdown") -> None:
        super().__init__()
        self.format = format


class TemplateRequested(Message):
    """Message when template is requested."""
    def __init__(self, template_name: str) -> None:
        super().__init__()
        self.template_name = template_name


class NotesToolbar(Horizontal):
    """
    Toolbar for notes screen with action buttons.
    Uses message passing for all actions.
    """
    
    DEFAULT_CSS = """
    NotesToolbar {
        height: 3;
        padding: 0 1;
        align: center middle;
        background: $panel;
    }
    
    NotesToolbar Button {
        margin: 0 1;
        min-width: 10;
    }
    
    NotesToolbar Button.primary {
        background: $primary;
    }
    
    NotesToolbar Button.danger {
        background: $error;
    }
    
    NotesToolbar Button.toggle {
        background: $secondary;
    }
    
    NotesToolbar Button.toggle.active {
        background: $primary;
    }
    
    .toolbar-separator {
        width: 1;
        margin: 0 1;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        *,
        show_sync: bool = True,
        show_export: bool = True,
        show_templates: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize the toolbar.
        
        Args:
            show_sync: Whether to show sync button
            show_export: Whether to show export button
            show_templates: Whether to show templates button
        """
        super().__init__(**kwargs)
        self.show_sync = show_sync
        self.show_export = show_export
        self.show_templates = show_templates
        self.preview_mode = False
    
    def compose(self) -> ComposeResult:
        """Compose the toolbar buttons."""
        # File operations
        yield Button(
            "📝 New",
            id="toolbar-new",
            variant="default",
            tooltip="Create a new note"
        )
        yield Button(
            "💾 Save",
            id="toolbar-save",
            variant="primary",
            tooltip="Save the current note"
        )
        yield Button(
            "🗑️ Delete",
            id="toolbar-delete",
            variant="error",
            classes="danger",
            tooltip="Delete the current note"
        )
        
        # Separator
        yield Button("|", disabled=True, classes="toolbar-separator")
        
        # View operations
        yield Button(
            "👁️ Preview",
            id="toolbar-preview",
            variant="default",
            classes="toggle",
            tooltip="Toggle preview mode"
        )
        
        if self.show_sync:
            yield Button(
                "🔄 Sync",
                id="toolbar-sync",
                variant="default",
                tooltip="Sync notes with files"
            )
        
        if self.show_export:
            yield Button(
                "📤 Export",
                id="toolbar-export",
                variant="default",
                tooltip="Export note to file"
            )
        
        if self.show_templates:
            yield Button(
                "📋 Template",
                id="toolbar-template",
                variant="default",
                tooltip="Apply a template"
            )
    
    @on(Button.Pressed, "#toolbar-new")
    def handle_new_button(self, event: Button.Pressed) -> None:
        """Handle new note button."""
        event.stop()
        logger.debug("New note requested from toolbar")
        self.post_message(NewNoteRequested())
    
    @on(Button.Pressed, "#toolbar-save")
    def handle_save_button(self, event: Button.Pressed) -> None:
        """Handle save button."""
        event.stop()
        logger.debug("Save requested from toolbar")
        self.post_message(SaveNoteRequested())
    
    @on(Button.Pressed, "#toolbar-delete")
    def handle_delete_button(self, event: Button.Pressed) -> None:
        """Handle delete button."""
        event.stop()
        logger.debug("Delete requested from toolbar")
        self.post_message(DeleteNoteRequested())
    
    @on(Button.Pressed, "#toolbar-preview")
    def handle_preview_button(self, event: Button.Pressed) -> None:
        """Handle preview toggle button."""
        event.stop()
        
        # Toggle the button state
        button = event.button
        self.preview_mode = not self.preview_mode
        
        if self.preview_mode:
            button.add_class("active")
            button.label = "📝 Edit"
            button.tooltip = "Switch to edit mode"
        else:
            button.remove_class("active")
            button.label = "👁️ Preview"
            button.tooltip = "Toggle preview mode"
        
        logger.debug(f"Preview mode toggled: {self.preview_mode}")
        self.post_message(PreviewToggleRequested())
    
    @on(Button.Pressed, "#toolbar-sync")
    def handle_sync_button(self, event: Button.Pressed) -> None:
        """Handle sync button."""
        event.stop()
        logger.debug("Sync requested from toolbar")
        self.post_message(SyncRequested())
    
    @on(Button.Pressed, "#toolbar-export")
    def handle_export_button(self, event: Button.Pressed) -> None:
        """Handle export button."""
        event.stop()
        logger.debug("Export requested from toolbar")
        # Could show a menu here for format selection
        self.post_message(ExportRequested("markdown"))
    
    @on(Button.Pressed, "#toolbar-template")
    def handle_template_button(self, event: Button.Pressed) -> None:
        """Handle template button."""
        event.stop()
        logger.debug("Template requested from toolbar")
        # Could show a menu here for template selection
        self.post_message(TemplateRequested("default"))
    
    def enable_save_button(self, enabled: bool = True) -> None:
        """Enable or disable the save button."""
        try:
            save_button = self.query_one("#toolbar-save", Button)
            save_button.disabled = not enabled
        except Exception:
            pass
    
    def enable_delete_button(self, enabled: bool = True) -> None:
        """Enable or disable the delete button."""
        try:
            delete_button = self.query_one("#toolbar-delete", Button)
            delete_button.disabled = not enabled
        except Exception:
            pass
    
    def set_preview_mode(self, preview: bool) -> None:
        """Set the preview mode state."""
        self.preview_mode = preview
        try:
            preview_button = self.query_one("#toolbar-preview", Button)
            if preview:
                preview_button.add_class("active")
                preview_button.label = "📝 Edit"
                preview_button.tooltip = "Switch to edit mode"
            else:
                preview_button.remove_class("active")
                preview_button.label = "👁️ Preview"
                preview_button.tooltip = "Toggle preview mode"
        except Exception:
            pass
    
    def update_button_states(
        self,
        has_note: bool = False,
        has_unsaved: bool = False
    ) -> None:
        """
        Update button states based on current context.
        
        Args:
            has_note: Whether a note is currently selected
            has_unsaved: Whether there are unsaved changes
        """
        # Save button enabled if there's a note with unsaved changes
        self.enable_save_button(has_note and has_unsaved)
        
        # Delete button enabled if there's a note
        self.enable_delete_button(has_note)