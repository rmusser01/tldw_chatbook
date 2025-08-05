# note_creation_modal.py
"""
Note Creation Modal
-------------------

Modal dialog for creating notes from chat messages with customizable
title, keywords, and content.
"""

from typing import Optional, Callable, Dict, Any, Tuple
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Input, TextArea, Static
from textual.binding import Binding
from textual.validation import ValidationResult, Validator
from loguru import logger


class NotEmptyValidator(Validator):
    """Validator to ensure a field is not empty."""
    
    def validate(self, value: str) -> ValidationResult:
        """Check if value is not empty after stripping whitespace."""
        if value.strip():
            return self.success()
        return self.failure("This field cannot be empty")


class NoteCreationModal(ModalScreen[Optional[Dict[str, Any]]]):
    """Modal for creating/editing a note with title, keywords, and content."""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]
    
    # CSS for the dialog
    DEFAULT_CSS = """
    NoteCreationModal {
        align: center middle;
    }
    
    NoteCreationModal > Container {
        background: $surface;
        border: thick $primary;
        padding: 2;
        width: 80;
        height: auto;
        max-height: 90%;
    }
    
    NoteCreationModal .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    NoteCreationModal .field-label {
        margin-top: 1;
        margin-bottom: 0;
        color: $text;
    }
    
    NoteCreationModal .title-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    NoteCreationModal .keywords-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    NoteCreationModal .content-textarea {
        width: 100%;
        height: 15;
        margin-bottom: 1;
    }
    
    NoteCreationModal .help-text {
        color: $text-muted;
        margin-top: 0;
        margin-bottom: 0;
    }
    
    NoteCreationModal .button-container {
        height: auto;
        align: center middle;
        padding: 1;
        margin-top: 1;
    }
    
    NoteCreationModal .action-button {
        width: 15;
        margin: 0 1;
    }
    
    NoteCreationModal .error-message {
        color: $error;
        margin-top: 0;
        margin-bottom: 1;
        display: none;
    }
    
    NoteCreationModal .error-message.visible {
        display: block;
    }
    """
    
    def __init__(self, 
                 initial_title: str = "",
                 initial_content: str = "",
                 initial_keywords: str = "",
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 **kwargs):
        """
        Initialize the note creation modal.
        
        Args:
            initial_title: Pre-populated title
            initial_content: Pre-populated content
            initial_keywords: Pre-populated keywords (comma-separated)
            callback: Function to call with the result
        """
        super().__init__(**kwargs)
        self.initial_title = initial_title
        self.initial_content = initial_content
        self.initial_keywords = initial_keywords
        self.callback = callback
        self.title_validator = NotEmptyValidator()
        self.content_validator = NotEmptyValidator()
        
    def compose(self) -> ComposeResult:
        """Compose the dialog UI."""
        with Container(classes="note-dialog"):
            yield Label("Create Note", classes="dialog-title")
            
            # Error message (initially hidden)
            yield Static("", id="error-message", classes="error-message")
            
            # Title field
            yield Label("Title:", classes="field-label")
            yield Input(
                value=self.initial_title,
                placeholder="Enter note title...",
                id="note-title-input",
                classes="title-input",
                validators=[self.title_validator]
            )
            
            # Keywords field
            yield Label("Keywords:", classes="field-label")
            yield Static("Separate multiple keywords with commas", classes="help-text")
            yield Input(
                value=self.initial_keywords,
                placeholder="e.g., important, project-x, todo",
                id="note-keywords-input",
                classes="keywords-input"
            )
            
            # Content field
            yield Label("Content:", classes="field-label")
            yield TextArea(
                self.initial_content,
                id="note-content-textarea",
                classes="content-textarea",
                language="markdown"
            )
            
            # Buttons
            with Horizontal(classes="button-container"):
                yield Button("Save", id="save-button", classes="action-button", variant="success")
                yield Button("Cancel", id="cancel-button", classes="action-button", variant="error")
    
    def on_mount(self) -> None:
        """Focus the title input when the modal is mounted."""
        title_input = self.query_one("#note-title-input", Input)
        title_input.focus()
        # Select all text for easy replacement
        # Set cursor to select all text
        title_input.cursor_position = 0
        title_input.selection = (0, len(title_input.value))
    
    @on(Button.Pressed, "#save-button")
    def handle_save(self, event: Button.Pressed) -> None:
        """Handle save button press."""
        logger.debug("Save button pressed in note creation modal")
        event.stop()
        
        # Get values
        title_input = self.query_one("#note-title-input", Input)
        keywords_input = self.query_one("#note-keywords-input", Input)
        content_textarea = self.query_one("#note-content-textarea", TextArea)
        error_message = self.query_one("#error-message", Static)
        
        title = title_input.value.strip()
        keywords_str = keywords_input.value.strip()
        content = content_textarea.text.strip()
        
        # Validate
        if not title:
            error_message.update("Title cannot be empty")
            error_message.add_class("visible")
            title_input.focus()
            return
        
        if not content:
            error_message.update("Content cannot be empty")
            error_message.add_class("visible")
            content_textarea.focus()
            return
        
        # Parse keywords
        keywords = []
        if keywords_str:
            keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        
        # Prepare result
        result = {
            "title": title,
            "content": content,
            "keywords": keywords
        }
        
        logger.debug(f"Note data prepared: title='{title}', keywords={keywords}, content_length={len(content)}")
        
        # Dismiss with result
        self.dismiss(result)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self, event: Button.Pressed) -> None:
        """Handle cancel button press."""
        logger.debug("Cancel button pressed in note creation modal")
        event.stop()
        self.dismiss(None)
    
    @on(Input.Changed)
    def clear_error_on_input_change(self, event: Input.Changed) -> None:
        """Clear error message when user starts typing."""
        error_message = self.query_one("#error-message", Static)
        if error_message.has_class("visible"):
            error_message.remove_class("visible")
            error_message.update("")
    
    @on(TextArea.Changed)
    def clear_error_on_textarea_change(self, event: TextArea.Changed) -> None:
        """Clear error message when user edits content."""
        error_message = self.query_one("#error-message", Static)
        if error_message.has_class("visible"):
            error_message.remove_class("visible")
            error_message.update("")
    
    def action_dismiss(self) -> None:
        """Handle escape key."""
        logger.debug("Note creation modal dismissed via escape")
        self.dismiss(None)