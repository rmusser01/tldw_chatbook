# document_generation_modal.py
# Description: Modal dialog for generating different document types from chat conversations
#
"""
Document Generation Modal
-------------------------

Provides a modal dialog for users to generate different document types
from the current chat conversation including:
- Timeline
- Study Guide
- Briefing Document  
- Original Note
"""

from typing import Optional, Callable, Dict, Any
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Center, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static, LoadingIndicator
from textual.reactive import reactive
from textual.binding import Binding
from loguru import logger


class DocumentGenerationModal(ModalScreen):
    """Modal for selecting document generation options."""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]
    
    # CSS for the dialog
    DEFAULT_CSS = """
    DocumentGenerationModal {
        align: center middle;
    }
    
    DocumentGenerationModal > Container {
        background: $surface;
        border: thick $primary;
        padding: 2;
        width: 70;
        height: 90%;
        max-height: 90%;
    }
    
    DocumentGenerationModal .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    DocumentGenerationModal .dialog-subtitle {
        text-align: center;
        margin-bottom: 2;
        color: $text-muted;
    }
    
    DocumentGenerationModal .options-container {
        padding: 1;
        height: 1fr;
        overflow-y: auto;
    }
    
    DocumentGenerationModal .option-card {
        background: $boost;
        border: solid $primary-lighten-2;
        padding: 1 2;
        margin-bottom: 1;
        height: auto;
    }
    
    DocumentGenerationModal .option-card:hover {
        background: $primary-lighten-3;
        border: solid $primary;
    }
    
    DocumentGenerationModal .option-title {
        text-style: bold;
        margin-bottom: 0;
    }
    
    DocumentGenerationModal .option-description {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    DocumentGenerationModal .generate-button {
        width: 100%;
        margin-top: 1;
    }
    
    DocumentGenerationModal .loading-container {
        align: center middle;
        height: 10;
        padding: 2;
    }
    
    DocumentGenerationModal .loading-message {
        text-align: center;
        margin-top: 1;
    }
    
    DocumentGenerationModal .close-button {
        width: 20;
        margin-top: 1;
    }
    
    DocumentGenerationModal .button-container {
        height: auto;
        align: center middle;
        padding: 1;
    }
    """
    
    # Track loading state
    is_loading = reactive(False)
    loading_message = reactive("")
    
    def __init__(self, 
                 message_content: str,
                 conversation_context: Optional[Dict[str, Any]] = None,
                 callback: Optional[Callable[[str, str], None]] = None,
                 **kwargs):
        """
        Initialize the document generation modal.
        
        Args:
            message_content: The specific message content to use
            conversation_context: Additional context from the conversation
            callback: Function to call with (document_type, generated_content)
        """
        super().__init__(**kwargs)
        self.message_content = message_content
        self.conversation_context = conversation_context or {}
        self.callback = callback
        
    def compose(self) -> ComposeResult:
        """Compose the dialog UI."""
        with Container(classes="document-dialog"):
            yield Label("Generate Document", classes="dialog-title")
            yield Static("Choose a document type to generate from this conversation", 
                        classes="dialog-subtitle")
            
            # Loading state
            with Container(id="loading-container", classes="loading-container"):
                yield LoadingIndicator()
                yield Label("", id="loading-message", classes="loading-message")
            
            # Document options
            with ScrollableContainer(id="options-container", classes="options-container"):
                # Timeline option
                with Container(classes="option-card"):
                    yield Label("📅 Timeline", classes="option-title")
                    yield Static("Create a chronological timeline of events and key points", 
                                classes="option-description")
                    yield Button("Generate Timeline", 
                               id="timeline-button", 
                               classes="generate-button",
                               variant="primary")
                
                # Study Guide option
                with Container(classes="option-card"):
                    yield Label("📚 Study Guide", classes="option-title")
                    yield Static("Create a comprehensive study guide with key concepts and learning objectives", 
                                classes="option-description")
                    yield Button("Generate Study Guide", 
                               id="study-guide-button", 
                               classes="generate-button",
                               variant="primary")
                
                # Briefing Document option
                with Container(classes="option-card"):
                    yield Label("📋 Briefing Document", classes="option-title")
                    yield Static("Create an executive briefing with key points and actionable insights", 
                                classes="option-description")
                    yield Button("Generate Briefing", 
                               id="briefing-button", 
                               classes="generate-button",
                               variant="primary")
                
                # Original Note option
                with Container(classes="option-card"):
                    yield Label("📝 Note from Response", classes="option-title")
                    yield Static("Create a note from this specific message (original functionality)", 
                                classes="option-description")
                    yield Button("Create Note", 
                               id="note-button", 
                               classes="generate-button",
                               variant="success")
            
            # Close button
            with Container(classes="button-container"):
                yield Button("Close", id="close-button", classes="close-button", variant="error")
    
    def watch_is_loading(self, loading: bool) -> None:
        """React to loading state changes."""
        loading_container = self.query_one("#loading-container")
        options_container = self.query_one("#options-container")
        
        loading_container.display = loading
        options_container.display = not loading
    
    def watch_loading_message(self, message: str) -> None:
        """Update loading message."""
        loading_label = self.query_one("#loading-message", Label)
        loading_label.update(message)
    
    @on(Button.Pressed, "#timeline-button")
    def handle_timeline(self, event: Button.Pressed) -> None:
        """Generate timeline document."""
        logger.debug("Timeline generation requested")
        event.stop()
        self._trigger_generation("timeline")
    
    @on(Button.Pressed, "#study-guide-button")
    def handle_study_guide(self, event: Button.Pressed) -> None:
        """Generate study guide document."""
        logger.debug("Study guide generation requested")
        event.stop()
        self._trigger_generation("study_guide")
    
    @on(Button.Pressed, "#briefing-button")
    def handle_briefing(self, event: Button.Pressed) -> None:
        """Generate briefing document."""
        logger.debug("Briefing document generation requested")
        event.stop()
        self._trigger_generation("briefing")
    
    @on(Button.Pressed, "#note-button")
    def handle_note(self, event: Button.Pressed) -> None:
        """Create original note."""
        logger.debug("Original note creation requested")
        event.stop()
        self._trigger_generation("note")
    
    @on(Button.Pressed, "#close-button")
    def handle_close(self, event: Button.Pressed) -> None:
        """Close the modal."""
        logger.debug("Document generation modal closed")
        event.stop()
        self.dismiss()
    
    def _trigger_generation(self, document_type: str):
        """Trigger document generation for the specified type."""
        try:
            # Don't set loading state here - just dismiss with the document type
            # The actual generation will happen in the parent after modal dismisses
            logger.debug(f"Dismissing modal with document_type: {document_type}")
            self.dismiss(document_type)
            
        except Exception as e:
            logger.error(f"Error triggering {document_type} generation: {e}")
            # Notify error but don't dismiss
            if hasattr(self.app, 'notify'):
                self.app.notify(f"Failed to generate {document_type}: {str(e)}", severity="error")
    
    def action_dismiss(self):
        """Handle escape key."""
        logger.debug("Document generation modal dismissed via escape")
        self.dismiss()