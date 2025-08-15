"""Base components for media ingestion UI."""

from typing import TYPE_CHECKING, Optional, List, Dict, Any, AsyncIterator
from pathlib import Path
import asyncio
import time
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import (
    Static, Button, Input, Label, Select, Checkbox, 
    TextArea, RadioSet, RadioButton, ProgressBar, Collapsible
)
from textual.widget import Widget
from textual.reactive import reactive
from textual import on, work
from textual.message import Message

from .models import ProcessingStatus, BaseMediaFormData

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ProcessingStatusUpdate(Message):
    """Message for processing status updates."""
    def __init__(self, status: ProcessingStatus):
        super().__init__()
        self.status = status


class BaseIngestTab(Container):
    """
    Base class for media ingestion tabs.
    
    Provides common functionality:
    - Form state management
    - Validation
    - Status updates
    - File selection
    - Processing coordination
    """
    
    # CSS classes for styling
    DEFAULT_CSS = """
    BaseIngestTab {
        height: 100%;
        width: 100%;
    }
    
    .ingest-main-scroll {
        height: 100%;
        width: 100%;
        padding: 1 2;
    }
    
    .status-dashboard {
        height: auto;
        width: 100%;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary;
        background: $surface;
    }
    
    .status-message {
        width: 1fr;
        content-align: left middle;
    }
    
    .file-counter {
        width: auto;
        content-align: right middle;
        margin-left: 1;
    }
    
    .progress-bar {
        height: 1;
        width: 100%;
        margin-top: 1;
    }
    
    .hidden {
        display: none;
    }
    
    .form-section {
        width: 100%;
        margin-bottom: 2;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .form-input {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .form-textarea {
        min-height: 5;
        max-height: 10;
        width: 100%;
        margin-bottom: 1;
    }
    
    .form-checkbox {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .form-select {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .error-display {
        color: $error;
        margin-top: 1;
        padding: 1;
        border: solid $error;
    }
    
    .submit-button {
        width: auto;
        margin-top: 2;
    }
    
    .mode-toggle {
        height: auto;
        width: 100%;
        margin-bottom: 2;
    }
    """
    
    # Reactive state
    form_data = reactive({})
    validation_errors = reactive({})
    processing_status = reactive(ProcessingStatus(state="idle"))
    simple_mode = reactive(True)
    is_valid = reactive(False)
    can_submit = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', media_type: str, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_type = media_type
        self.selected_files: List[Path] = []
        self.selected_urls: List[str] = []
        self._processing_worker = None
        
        logger.debug(f"[{media_type}] BaseIngestTab initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the base ingestion interface."""
        with VerticalScroll(classes="ingest-main-scroll"):
            # Status dashboard
            yield from self.create_status_dashboard()
            
            # Mode toggle
            yield from self.create_mode_toggle()
            
            # File/URL selection
            with Container(classes="form-section"):
                yield Label("Source Selection", classes="section-title")
                yield from self.create_source_selector()
            
            # Basic metadata
            with Container(classes="form-section"):
                yield Label("Metadata", classes="section-title")
                yield from self.create_metadata_fields()
            
            # Media-specific options (implemented by subclasses)
            with Container(classes="form-section media-options"):
                yield from self.create_media_options()
            
            # Advanced options (collapsible)
            # Collect the advanced options widgets
            advanced_widgets = list(self.create_advanced_options())
            # Create the collapsible with the widgets as children
            yield Collapsible(
                "Advanced Options",
                *advanced_widgets,
                collapsed=True,
                id="advanced-options"
            )
            
            # Submit button
            yield Button("Process Files", id="submit", classes="submit-button", disabled=True)
    
    def create_status_dashboard(self) -> ComposeResult:
        """Create the status dashboard."""
        with Container(id="status-dashboard", classes="status-dashboard"):
            with Horizontal():
                yield Static("Ready", id="status-message", classes="status-message")
                yield Static("", id="file-counter", classes="file-counter hidden")
            
            yield ProgressBar(id="progress-bar", classes="progress-bar hidden", total=100)
            yield Static("", id="current-operation", classes="hidden")
            yield Static("", id="error-display", classes="error-display hidden")
    
    def create_mode_toggle(self) -> ComposeResult:
        """Create simple/advanced mode toggle."""
        with Container(classes="mode-toggle"):
            yield Label("Processing Mode", classes="section-title")
            with RadioSet(id="mode-selector"):
                yield RadioButton("Simple", value=True, id="simple")
                yield RadioButton("Advanced", id="advanced")
    
    def create_source_selector(self) -> ComposeResult:
        """Create file/URL input controls."""
        yield Button("Select Files", id="select-files", classes="form-input")
        yield Static("No files selected", id="file-list", classes="form-input")
        
        yield Label("Or enter URLs (one per line):", classes="form-label")
        yield TextArea("", id="url-input", classes="form-textarea")
    
    def create_metadata_fields(self) -> ComposeResult:
        """Create basic metadata input fields."""
        yield Label("Title (optional):", classes="form-label")
        yield Input(placeholder="Leave blank to use file name", id="title", classes="form-input")
        
        yield Label("Author (optional):", classes="form-label")
        yield Input(placeholder="Content author", id="author", classes="form-input")
        
        yield Label("Keywords (comma-separated):", classes="form-label")
        yield Input(placeholder="keyword1, keyword2, ...", id="keywords", classes="form-input")
    
    def create_media_options(self) -> ComposeResult:
        """Create media-specific options. Override in subclasses."""
        yield Label(f"{self.media_type.title()} Options", classes="section-title")
        yield Static("Media-specific options will be added by subclass")
    
    def create_advanced_options(self) -> ComposeResult:
        """Create advanced options. Override in subclasses."""
        yield Static("Advanced options will be added by subclass")
    
    # Event handlers
    @on(RadioSet.Changed, "#mode-selector")
    def handle_mode_change(self, event: RadioSet.Changed):
        """Handle mode toggle."""
        self.simple_mode = event.pressed.id == "simple"
        self.update_mode_visibility()
    
    @on(Button.Pressed, "#select-files")
    async def handle_file_selection(self):
        """Handle file selection button."""
        # This would normally open a file picker
        # For now, we'll just show a placeholder
        self.notify("File picker would open here", severity="information")
    
    @on(Input.Changed)
    @on(TextArea.Changed)
    def handle_input_change(self, event):
        """Handle form input changes."""
        # Update form data
        widget_id = event.input.id if hasattr(event, 'input') else event.text_area.id
        value = event.value
        
        self.form_data = {**self.form_data, widget_id: value}
        self.validate_form()
        self.update_submit_button()
    
    @on(Button.Pressed, "#submit")
    async def handle_submit(self):
        """Handle form submission."""
        if not self.can_submit:
            return
        
        # Validate form data
        if not self.validate_form():
            self.show_validation_errors()
            return
        
        # Start processing
        self._processing_worker = self.run_worker(
            self.process_media(),
            exclusive=True
        )
    
    # Validation methods
    def validate_form(self) -> bool:
        """Validate the form data."""
        errors = {}
        
        # Check for at least one source
        if not self.selected_files and not self.form_data.get('url-input', '').strip():
            errors['source'] = "Please select files or enter URLs"
        
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        return self.is_valid
    
    def show_validation_errors(self):
        """Display validation errors."""
        if not self.validation_errors:
            return
        
        error_display = self.query_one("#error-display")
        error_text = "\n".join(f"• {error}" for error in self.validation_errors.values())
        error_display.update(error_text)
        error_display.remove_class("hidden")
    
    def update_submit_button(self):
        """Update submit button state."""
        submit_button = self.query_one("#submit")
        
        # Enable if we have sources and no errors
        has_sources = bool(self.selected_files) or bool(self.form_data.get('url-input', '').strip())
        self.can_submit = has_sources and self.is_valid
        submit_button.disabled = not self.can_submit
    
    def update_mode_visibility(self):
        """Update visibility based on mode."""
        advanced_options = self.query_one("#advanced-options")
        advanced_options.collapsed = self.simple_mode
    
    # Processing methods
    @work(exclusive=True)
    async def process_media(self):
        """Process the media files. Override in subclasses."""
        try:
            # Update status
            self.update_status(ProcessingStatus(state="processing", message="Starting processing..."))
            
            # Get validated form data
            form_data = self.get_validated_form_data()
            
            # Process files (implement in subclass)
            async for status in self.process_media_impl(form_data):
                self.update_status(status)
                
                if status.state == "error":
                    break
            
            # Final status
            if self.processing_status.state != "error":
                self.update_status(ProcessingStatus(
                    state="complete",
                    message="Processing complete!",
                    progress=1.0
                ))
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.update_status(ProcessingStatus(
                state="error",
                message=f"Error: {str(e)}",
                error=str(e)
            ))
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Implement actual processing. Override in subclasses."""
        yield ProcessingStatus(state="error", error="Not implemented")
    
    def get_validated_form_data(self) -> Dict[str, Any]:
        """Get validated form data."""
        # Collect all form data
        data = dict(self.form_data)
        
        # Add files and URLs
        data['files'] = self.selected_files
        urls_text = data.get('url-input', '').strip()
        data['urls'] = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        return data
    
    def update_status(self, status: ProcessingStatus):
        """Update processing status."""
        self.processing_status = status
        
        # Update UI elements
        self.query_one("#status-message").update(status.message)
        
        # Update progress bar
        progress_bar = self.query_one("#progress-bar")
        if status.state == "processing":
            progress_bar.remove_class("hidden")
            progress_bar.progress = status.progress * 100
        elif status.state in ["complete", "error", "idle"]:
            progress_bar.add_class("hidden")
        
        # Update file counter
        if status.total_files > 0:
            counter = self.query_one("#file-counter")
            counter.update(f"{status.files_processed}/{status.total_files} files")
            counter.remove_class("hidden")
        
        # Update current operation
        if status.current_operation:
            op_display = self.query_one("#current-operation")
            op_display.update(f"Current: {status.current_operation}")
            op_display.remove_class("hidden")
        
        # Show errors
        if status.error:
            error_display = self.query_one("#error-display")
            error_display.update(f"Error: {status.error}")
            error_display.remove_class("hidden")