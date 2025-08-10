# tldw_chatbook/Widgets/Media_Ingest/base_media_ingest_window.py
"""
Base media ingestion window with common functionality and patterns.
This implements the architecture designed in New-Ingestion-UX-Plan.md.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional, AsyncIterator
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from loguru import logger

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import Static, Button, Input, Label, RadioSet, RadioButton
from textual.widget import Widget
from textual.reactive import reactive
from textual import on, work

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MediaFormData(BaseModel):
    """Base model for media ingestion form data."""
    title: Optional[str] = Field(None, description="Optional title override")
    author: Optional[str] = Field(None, description="Optional author override")
    keywords: Optional[str] = Field(None, description="Comma-separated keywords")
    files: List[Path] = Field(default_factory=list, description="Selected files")
    urls: List[str] = Field(default_factory=list, description="Media URLs")


class ProcessingStatus(BaseModel):
    """Status update during processing."""
    state: str = Field(..., description="idle, processing, complete, error")
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_file: Optional[str] = None
    files_processed: int = 0
    total_files: int = 0
    message: str = ""
    error: Optional[str] = None


class BaseMediaIngestWindow(Container):
    """
    Base class for all media ingestion windows.
    
    Implements common patterns:
    - Single source of truth for form data
    - Reactive state management
    - Progressive disclosure (simple/advanced modes)
    - Proper input visibility
    - Validation and error handling
    - Status updates and progress tracking
    """
    
    # Core reactive state
    form_data = reactive({})
    validation_errors = reactive({})
    processing_status = reactive(ProcessingStatus(state="idle"))
    simple_mode = reactive(False)  # TEMP: Start in advanced mode for access
    
    # Form state derived properties
    is_valid = reactive(True)
    can_submit = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_type = self.get_media_type()
        
        logger.debug(f"[{self.media_type}] BaseMediaIngestWindow initialized.")
    
    def get_media_type(self) -> str:
        """Return the media type (e.g., 'video', 'audio', 'pdf').
        
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement get_media_type()")
    
    def get_form_data_model(self) -> type[BaseModel]:
        """Return the Pydantic model for this media type's form data.
        
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement get_form_data_model()")
    
    def create_media_specific_options(self) -> ComposeResult:
        """Create media-specific options (transcription, chunking, etc.).
        
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement create_media_specific_options()")
        # This yield statement is never reached but prevents syntax errors
        yield  # type: ignore
    
    async def process_media(self, validated_data: dict) -> AsyncIterator[ProcessingStatus]:
        """Process the media files. Yield status updates.
        
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement process_media()")
        # This yield statement is never reached but prevents syntax errors  
        yield ProcessingStatus(state="error", error="Not implemented")  # type: ignore
    
    def compose(self) -> ComposeResult:
        """Compose the base media ingestion interface."""
        # Single scrolling container
        with VerticalScroll(classes="ingest-main-scroll"):
            # Status dashboard
            yield from self.create_status_dashboard()
            # Mode toggle  
            yield from self.create_mode_toggle()
            
            # Essential fields (always visible)
            with Container(classes="essential-section"):
                yield Label("Essential Information", classes="section-title")
                yield from self.create_file_selector()
                yield from self.create_basic_metadata()
            
            # Media-specific options (implemented by subclasses)
            with Container(classes="media-options-section"):
                yield from self.create_media_specific_options()
            
            # Process button
            yield from self.create_process_button()
    
    def create_status_dashboard(self) -> ComposeResult:
        """Create the status dashboard component."""
        from textual.widgets import ProgressBar
        
        with Container(id="status-dashboard", classes="status-dashboard"):
            # Main status row
            with Horizontal(classes="status-main-row"):
                yield Static("Ready to process files", id="status-message", classes="status-message")
                yield Static("", id="file-counter", classes="file-counter hidden")
                yield Static("", id="time-display", classes="time-display hidden")
            
            # Progress bar (initially hidden)
            yield ProgressBar(id="progress-bar", classes="progress-bar hidden")
            
            # Current operation (initially hidden)
            yield Static("", id="current-operation", classes="current-operation hidden")
            
            # Error display (initially hidden)
            yield Static("", id="error-display", classes="error-display hidden")
    
    def create_mode_toggle(self) -> ComposeResult:
        """Create the simple/advanced mode toggle."""
        with Container(classes="mode-toggle-container"):
            yield Static(f"{self.media_type.title()} Processing Mode", classes="mode-title")
            with RadioSet(id="mode-toggle", classes="mode-toggle"):
                yield RadioButton("Simple Mode", value=False, id="simple-mode")
                yield RadioButton("Advanced Mode", value=True, id="advanced-mode")
    
    def create_file_selector(self) -> ComposeResult:
        """Create the file selection component."""
        yield Label("Select Files", classes="form-label-primary")
        
        # Action buttons
        with Horizontal(classes="file-selection-row"):
            yield Button("Browse Files", id="browse-files", variant="primary", classes="file-button")
            yield Button("Clear All", id="clear-files", variant="default", classes="file-button")
            yield Button("Add URLs", id="add-urls", variant="default", classes="file-button")
        
        # File list display
        yield Container(id="file-list-display", classes="file-list-display")
        
        # URL input (always visible)
        with Container(id="url-input-section", classes="url-input-section"):
            yield Label("Enter URLs (one per line):")
            from textual.widgets import TextArea
            yield TextArea(
                text=f"# Enter {self.media_type} URLs here (one per line)...",
                id="urls-textarea",
                classes="url-textarea"
            )
            with Horizontal(classes="url-actions"):
                yield Button("Add URLs", id="confirm-urls", variant="primary")
                yield Button("Cancel", id="cancel-urls", variant="default")
    
    def create_basic_metadata(self) -> ComposeResult:
        """Create basic metadata fields (title, author, keywords)."""
        yield Label("Metadata (Optional)", classes="form-label-primary")
        
        # Title and Author row
        with Horizontal(classes="metadata-row"):
            # Left column - Title
            with Vertical(classes="metadata-col"):
                yield Label("Title (Optional):", classes="form-label")
                yield Input(
                    placeholder="Auto-detected from file",
                    id="title-input",
                    classes="form-input"
                )
            
            # Right column - Author  
            with Vertical(classes="metadata-col"):
                yield Label("Author (Optional):", classes="form-label")
                yield Input(
                    placeholder="Leave blank if unknown",
                    id="author-input",
                    classes="form-input"
                )
        
        # Keywords
        yield Label("Keywords (Optional):", classes="form-label")
        yield Input(
            placeholder="Comma-separated tags (e.g., education, tutorial, science)",
            id="keywords-input",
            classes="form-input"
        )
    
    def create_process_button(self) -> ComposeResult:
        """Create the process button with proper state management."""
        with Container(classes="process-button-section"):
            yield Button(
                f"Process {self.media_type.title()}",
                id="process-button",
                variant="primary",
                disabled=True,
                classes="process-button"
            )
    
    # Event Handlers
    
    @on(RadioSet.Changed, "#mode-toggle")
    def handle_mode_change(self, event):
        """Handle simple/advanced mode toggle."""
        self.simple_mode = event.pressed.id == "simple-mode"
        logger.debug(f"[{self.media_type}] Mode changed to: {'simple' if self.simple_mode else 'advanced'}")
    
    @on(Input.Changed)
    def handle_input_change(self, event):
        """Handle all input field changes with validation."""
        field_id = event.input.id
        value = event.value
        
        # Update form data
        self.form_data = {**self.form_data, field_id: value}
        
        # Validate field
        error = self.validate_field(field_id, value)
        
        # Update validation state
        errors = dict(self.validation_errors)
        if error:
            errors[field_id] = error
            event.input.add_class("error")
        else:
            errors.pop(field_id, None)
            event.input.remove_class("error")
        
        self.validation_errors = errors
        
        # Update submit button state
        self.update_submit_state()
    
    @on(Button.Pressed, "#browse-files")
    async def handle_browse_files(self):
        """Open file browser for file selection."""
        try:
            from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
            
            # Get appropriate file filters for this media type
            filters = self.get_file_filters()
            
            files = await self.app.push_screen_wait(FileOpen(filters=filters))
            if files:
                self.add_files(files)
                logger.debug(f"[{self.media_type}] Added {len(files)} files")
        except Exception as e:
            logger.error(f"[{self.media_type}] Error browsing files: {e}")
            self.app.notify(f"Error selecting files: {e}", severity="error")
    
    @on(Button.Pressed, "#clear-files")
    def handle_clear_files(self):
        """Clear all selected files."""
        self.form_data = {**self.form_data, "files": [], "urls": []}
        self.update_file_display()
        self.update_submit_state()
        logger.debug(f"[{self.media_type}] Cleared all files")
    
    @on(Button.Pressed, "#add-urls")
    def handle_show_url_input(self):
        """Show URL input section."""
        url_section = self.query_one("#url-input-section")
        url_section.remove_class("hidden")
        
        # Focus the textarea
        try:
            self.query_one("#urls-textarea").focus()
        except:
            pass
    
    @on(Button.Pressed, "#confirm-urls")
    def handle_add_urls(self):
        """Add URLs from textarea."""
        try:
            textarea = self.query_one("#urls-textarea")
            urls_text = textarea.value.strip()
            
            if urls_text:
                # Parse URLs (one per line), filtering out comments
                urls = [url.strip() for url in urls_text.split('\n') 
                       if url.strip() and not url.strip().startswith('#')]
                self.add_urls(urls)
                
                # Clear and hide URL input
                textarea.value = f"# Enter {self.media_type} URLs here (one per line)..."
                self.query_one("#url-input-section").add_class("hidden")
                
                logger.debug(f"[{self.media_type}] Added {len(urls)} URLs")
            else:
                self.app.notify("Please enter at least one URL", severity="warning")
        except Exception as e:
            logger.error(f"[{self.media_type}] Error adding URLs: {e}")
            self.app.notify(f"Error adding URLs: {e}", severity="error")
    
    @on(Button.Pressed, "#cancel-urls")
    def handle_cancel_urls(self):
        """Cancel URL input."""
        textarea = self.query_one("#urls-textarea")
        textarea.value = f"# Enter {self.media_type} URLs here (one per line)..."
        self.query_one("#url-input-section").add_class("hidden")
    
    @on(Button.Pressed, "#process-button")
    def handle_process(self):
        """Start processing the media."""
        if self.can_submit:
            self.start_processing()
        else:
            self.app.notify("Please correct the form errors before submitting", severity="warning")
    
    # State Management Methods
    
    def validate_field(self, field_id: str, value: str) -> Optional[str]:
        """Validate a single field. Override in subclasses for specific validation."""
        # Basic validation for common fields
        if field_id == "title-input":
            if value and len(value.strip()) < 2:
                return "Title must be at least 2 characters"
        elif field_id == "keywords-input":
            if value and len(value.strip()) < 2:
                return "Keywords must be at least 2 characters"
        # Add more field validations as needed
        return None
    
    def update_submit_state(self):
        """Update the submit button enabled/disabled state."""
        # Check if we have files or URLs
        files = self.form_data.get("files", [])
        urls = self.form_data.get("urls", [])
        has_media = len(files) > 0 or len(urls) > 0
        
        # Check for validation errors
        has_errors = bool(self.validation_errors)
        
        # Update reactive state
        self.can_submit = has_media and not has_errors
        
        # Update button
        try:
            button = self.query_one("#process-button")
            button.disabled = not self.can_submit
        except:
            pass
    
    def add_files(self, files: List[Path]):
        """Add files to the selection."""
        current_files = self.form_data.get("files", [])
        new_files = current_files + files
        self.form_data = {**self.form_data, "files": new_files}
        self.update_file_display()
        self.update_submit_state()
    
    def add_urls(self, urls: List[str]):
        """Add URLs to the selection."""
        current_urls = self.form_data.get("urls", [])
        new_urls = current_urls + urls
        self.form_data = {**self.form_data, "urls": new_urls}
        self.update_file_display()
        self.update_submit_state()
    
    def update_file_display(self):
        """Update the file list display."""
        try:
            display = self.query_one("#file-list-display")
            display.remove_children()
            
            files = self.form_data.get("files", [])
            urls = self.form_data.get("urls", [])
            
            if not files and not urls:
                display.mount(Static("No files or URLs selected", classes="empty-message"))
            else:
                # Show file count summary
                total_items = len(files) + len(urls)
                summary = f"Selected: {len(files)} files, {len(urls)} URLs ({total_items} total)"
                display.mount(Static(summary, classes="file-summary"))
                
                # Show individual items (limit display to prevent UI overflow)
                max_display = 10
                items_shown = 0
                
                for file_path in files[:max_display - items_shown]:
                    display.mount(Static(f"📁 {file_path.name}", classes="file-item"))
                    items_shown += 1
                
                for url in urls[:max_display - items_shown]:
                    display.mount(Static(f"🔗 {url[:50]}{'...' if len(url) > 50 else ''}", classes="url-item"))
                    items_shown += 1
                
                if total_items > max_display:
                    display.mount(Static(f"... and {total_items - max_display} more items", classes="more-items"))
        except Exception as e:
            logger.error(f"[{self.media_type}] Error updating file display: {e}")
    
    @work(exclusive=True)
    async def start_processing(self):
        """Start the media processing workflow."""
        try:
            # Validate form data
            form_model = self.get_form_data_model()
            validated_data = form_model(**self.form_data).dict()
            
            # Update status to processing
            self.processing_status = ProcessingStatus(
                state="processing",
                message="Starting processing...",
                total_files=len(validated_data.get("files", [])) + len(validated_data.get("urls", []))
            )
            
            # Process media (implemented by subclasses)
            async for status_update in self.process_media(validated_data):
                self.processing_status = status_update
            
            # Success
            self.processing_status = ProcessingStatus(
                state="complete",
                progress=1.0,
                message="Processing completed successfully!"
            )
            
            self.app.notify("Media processing completed!", severity="information")
            
        except ValidationError as e:
            error_msg = f"Form validation failed: {', '.join([error['msg'] for error in e.errors()])}"
            self.processing_status = ProcessingStatus(
                state="error",
                error=error_msg,
                message="Validation failed"
            )
            logger.error(f"[{self.media_type}] Validation error: {error_msg}")
            self.app.notify(error_msg, severity="error")
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.processing_status = ProcessingStatus(
                state="error",
                error=error_msg,
                message="Processing failed"
            )
            logger.error(f"[{self.media_type}] Processing error: {e}")
            self.app.notify(error_msg, severity="error")
    
    # Watchers
    
    def watch_processing_status(self, status: ProcessingStatus):
        """Update UI when processing status changes."""
        try:
            # Update status message
            status_msg = self.query_one("#status-message")
            status_msg.update(status.message)
            
            # Update progress bar
            progress_bar = self.query_one("#progress-bar")
            if status.state == "processing":
                progress_bar.remove_class("hidden")
                progress_bar.progress = status.progress
            else:
                progress_bar.add_class("hidden")
            
            # Update file counter
            counter = self.query_one("#file-counter")
            if status.total_files > 0:
                counter.update(f"{status.files_processed}/{status.total_files} files")
                counter.remove_class("hidden")
            else:
                counter.add_class("hidden")
            
            # Update current operation
            current_op = self.query_one("#current-operation")
            if status.current_file:
                current_op.update(f"Processing: {status.current_file}")
                current_op.remove_class("hidden")
            else:
                current_op.add_class("hidden")
            
            # Update error display
            error_display = self.query_one("#error-display")
            if status.error:
                error_display.update(f"Error: {status.error}")
                error_display.remove_class("hidden")
            else:
                error_display.add_class("hidden")
            
            # Update process button
            process_btn = self.query_one("#process-button")
            if status.state == "processing":
                process_btn.disabled = True
                process_btn.label = "Processing..."
            else:
                process_btn.disabled = not self.can_submit
                process_btn.label = f"Process {self.media_type.title()}"
                
        except Exception as e:
            logger.error(f"[{self.media_type}] Error updating status display: {e}")
    
    def watch_simple_mode(self, simple: bool):
        """Handle mode changes - to be extended by subclasses."""
        if simple:
            self.add_class("simple-mode")
            self.remove_class("advanced-mode")
        else:
            self.add_class("advanced-mode")
            self.remove_class("simple-mode")
        
        logger.debug(f"[{self.media_type}] Switched to {'simple' if simple else 'advanced'} mode")
    
    # Utility Methods
    
    def get_file_filters(self) -> List[tuple]:
        """Get file filters for the file browser. Override in subclasses."""
        return [("All Files", "*")]
    
    def get_field_value(self, field_id: str, default: Any = "") -> Any:
        """Get a field value from form data."""
        return self.form_data.get(field_id, default)
    
    def set_field_value(self, field_id: str, value: Any):
        """Set a field value in form data."""
        self.form_data = {**self.form_data, field_id: value}
    
    # Default CSS for the base window
    DEFAULT_CSS = """
    /* Base media ingestion window styling */
    .ingest-main-scroll {
        height: 100%;
        width: 100%;
        padding: 1;
    }
    
    .status-dashboard {
        dock: top;
        height: auto;
        min-height: 3;
        background: $surface;
        border: round $primary;
        padding: 1;
        margin-bottom: 1;
    }
    
    .essential-section, .media-options-section {
        margin-bottom: 2;
        padding: 1;
        border: round $surface;
        background: $surface;
    }
    
    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        border-bottom: solid $primary;
        padding-bottom: 1;
    }
    
    .form-label-primary {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .form-label {
        margin-bottom: 1;
    }
    
    .form-input {
        height: 3;
        width: 100%;
        margin-bottom: 1;
        border: solid $primary;
        padding: 0 1;
    }
    
    .form-input:focus {
        border: solid $accent;
        background: $accent 10%;
    }
    
    .form-input.error {
        border: solid $error;
        background: $error 10%;
    }
    
    .metadata-row {
        layout: horizontal;
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    
    .metadata-col {
        width: 1fr;
        height: auto;
    }
    
    .file-selection-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .file-button {
        margin-right: 1;
    }
    
    .file-list-display {
        min-height: 5;
        max-height: 10;
        border: round $primary;
        background: $surface;
        padding: 1;
        margin-bottom: 1;
    }
    
    .empty-message {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }
    
    .file-summary {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .file-item, .url-item {
        margin-bottom: 1;
        color: $text-muted;
    }
    
    .process-button-section {
        dock: bottom;
        height: 5;
        padding: 1;
        border-top: solid $primary;
    }
    
    .process-button {
        width: 100%;
        height: 3;
        text-style: bold;
    }
    
    .mode-toggle-container {
        margin-bottom: 2;
        padding: 1;
        border: round $surface;
    }
    
    .mode-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .url-input-section {
        margin-top: 1;
        padding: 1;
        border: round $primary;
        background: $surface;
    }
    
    .url-textarea {
        min-height: 5;
        max-height: 10;
        margin-bottom: 1;
    }
    
    .url-actions {
        /* Spacing handled by margin on buttons */
    }
    
    .hidden {
        display: none;
    }
    """