"""Base screen class for media ingestion screens."""

from typing import TYPE_CHECKING, Optional, List, Dict, Any, AsyncIterator
from pathlib import Path
import asyncio
import time
from loguru import logger

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import (
    Static, Button, Input, Label, Select, Checkbox, 
    TextArea, RadioSet, RadioButton, ProgressBar, Collapsible
)
from textual.reactive import reactive
from textual import on, work
from textual.message import Message

from ..models import ProcessingStatus, BaseMediaFormData
from ...ScreenNavigation.navigation_system import ScreenState

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ProcessingStatusUpdate(Message):
    """Message for processing status updates."""
    def __init__(self, status: ProcessingStatus):
        super().__init__()
        self.status = status


class MediaIngestNavigation(Container):
    """Navigation bar for media ingestion screens."""
    
    DEFAULT_CSS = """
    MediaIngestNavigation {
        height: 3;
        width: 100%;
        dock: top;
        background: $panel;
        border-bottom: solid $primary;
        padding: 0 2;
    }
    
    .media-nav-links {
        height: 100%;
        width: 100%;
        layout: horizontal;
        align: left middle;
    }
    
    .media-nav-link {
        margin: 0 1;
        min-width: 8;
        background: transparent;
        border: none;
    }
    
    .media-nav-link:hover {
        text-style: underline;
    }
    
    .media-nav-link.active {
        text-style: bold;
        color: $warning;
    }
    
    .nav-separator {
        margin: 0 1;
        color: $text-muted;
    }
    """
    
    def __init__(self, active_media_type: str = "video", **kwargs):
        super().__init__(**kwargs)
        self.active_media_type = active_media_type
    
    def compose(self) -> ComposeResult:
        """Compose the navigation bar."""
        with Horizontal(classes="media-nav-links"):
            yield Button("Video", id="media-nav-video", 
                        classes=f"media-nav-link {'active' if self.active_media_type == 'video' else ''}")
            yield Static("|", classes="nav-separator")
            yield Button("Audio", id="media-nav-audio",
                        classes=f"media-nav-link {'active' if self.active_media_type == 'audio' else ''}")
            yield Static("|", classes="nav-separator")
            yield Button("PDF", id="media-nav-pdf",
                        classes=f"media-nav-link {'active' if self.active_media_type == 'pdf' else ''}")
            yield Static("|", classes="nav-separator")
            yield Button("Documents", id="media-nav-document",
                        classes=f"media-nav-link {'active' if self.active_media_type == 'document' else ''}")
            yield Static("|", classes="nav-separator")
            yield Button("Ebooks", id="media-nav-ebook",
                        classes=f"media-nav-link {'active' if self.active_media_type == 'ebook' else ''}")
            yield Static("|", classes="nav-separator")
            yield Button("Web", id="media-nav-web",
                        classes=f"media-nav-link {'active' if self.active_media_type == 'web' else ''}")
    
    @on(Button.Pressed, ".media-nav-link")
    def handle_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation link clicks."""
        button_id = event.button.id
        if not button_id:
            return
        
        # Extract media type from button ID
        media_type = button_id.replace("media-nav-", "")
        
        # Update active state
        for button in self.query(".media-nav-link"):
            button.remove_class("active")
        event.button.add_class("active")
        
        # Post navigation message
        self.post_message(NavigateToMediaType(media_type))


class NavigateToMediaType(Message):
    """Message to request media type navigation."""
    
    def __init__(self, media_type: str):
        super().__init__()
        self.media_type = media_type


class BaseMediaIngestScreen(Screen):
    """
    Base screen class for media ingestion.
    
    Provides common functionality:
    - Form state management
    - Validation
    - Status updates
    - File selection
    - Processing coordination
    - State save/restore
    """
    
    # Reactive properties
    processing_status = reactive(ProcessingStatus())
    form_data = reactive({})
    
    # CSS
    DEFAULT_CSS = """
    BaseMediaIngestScreen {
        background: $background;
    }
    
    .ingest-content {
        height: 100%;
        width: 100%;
        padding-top: 3;
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
        height: 10;
        width: 100%;
        margin-bottom: 1;
    }
    
    .form-select {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .option-group {
        padding: 1;
        margin-bottom: 1;
        border: dashed $primary-lighten-3;
    }
    
    .option-title {
        text-style: italic;
        margin-bottom: 1;
    }
    
    .action-buttons {
        width: 100%;
        height: auto;
        margin-top: 2;
        layout: horizontal;
        align: center middle;
    }
    
    .action-button {
        margin: 0 1;
        min-width: 16;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "base", **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_type = media_type
        self.processing_worker = None
        self.screen_state = ScreenState(f"media_ingest_{media_type}")
        
        logger.info(f"Initializing {self.__class__.__name__} for media type: {media_type}")
    
    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        # Navigation bar
        yield MediaIngestNavigation(active_media_type=self.media_type)
        
        # Main content area
        with Container(classes="ingest-content"):
            with VerticalScroll(classes="ingest-main-scroll"):
                # Status dashboard
                with Container(classes="status-dashboard", id="status-dashboard"):
                    with Horizontal():
                        yield Static("Ready to process media", 
                                   classes="status-message", 
                                   id="status-message")
                        yield Static("0/0 files", 
                                   classes="file-counter", 
                                   id="file-counter")
                    yield ProgressBar(total=100, 
                                    show_eta=False,
                                    classes="progress-bar hidden",
                                    id="progress-bar")
                
                # Input section
                yield from self.create_input_section()
                
                # Media-specific options
                yield from self.create_media_options()
                
                # Advanced options section
                yield Label("Advanced Options", classes="section-title")
                with Container(classes="advanced-options-container", id="advanced-options"):
                    yield from self.create_advanced_options()
                
                # Action buttons
                with Horizontal(classes="action-buttons"):
                    yield Button("Process", 
                               variant="primary", 
                               classes="action-button",
                               id="process-button")
                    yield Button("Clear", 
                               variant="default", 
                               classes="action-button",
                               id="clear-button")
                    yield Button("Cancel", 
                               variant="error", 
                               classes="action-button hidden",
                               id="cancel-button")
    
    def create_input_section(self) -> ComposeResult:
        """Create the input section (files and URLs)."""
        with Container(classes="form-section"):
            yield Label("Input Sources", classes="section-title")
            
            yield Label("Files (comma-separated paths):")
            yield Input(placeholder="/path/to/file1.mp4, /path/to/file2.mp4",
                       id="files-input",
                       classes="form-input")
            
            yield Label("URLs (one per line):")
            yield TextArea("",
                         id="urls-input",
                         classes="form-textarea")
            
            yield Button("Browse Files...", 
                       variant="default",
                       id="browse-button")
    
    def create_media_options(self) -> ComposeResult:
        """Override in subclasses to create media-specific options."""
        yield Container()  # Empty container by default
    
    def create_advanced_options(self) -> ComposeResult:
        """Override in subclasses to create advanced options."""
        yield Container()  # Empty container by default
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Restore state if available
        self.restore_state()
        
        # Focus on first input
        try:
            self.query_one("#files-input").focus()
        except Exception:
            pass
    
    def on_unmount(self) -> None:
        """Called when the screen is unmounted."""
        # Save state before leaving
        self.save_state()
    
    def save_state(self) -> None:
        """Save the current screen state."""
        self.screen_state.save_from_screen(self)
        
        # Save additional custom data
        self.screen_state.custom_data["processing_status"] = self.processing_status
        self.screen_state.custom_data["media_type"] = self.media_type
        
        logger.debug(f"Saved state for {self.media_type} ingestion screen")
    
    def restore_state(self) -> None:
        """Restore the previous screen state."""
        self.screen_state.restore_to_screen(self)
        
        # Restore custom data
        if "processing_status" in self.screen_state.custom_data:
            self.processing_status = self.screen_state.custom_data["processing_status"]
        
        logger.debug(f"Restored state for {self.media_type} ingestion screen")
    
    @on(Button.Pressed, "#process-button")
    async def handle_process(self) -> None:
        """Handle the process button click."""
        logger.info(f"Starting {self.media_type} processing")
        
        # Validate form
        try:
            form_data = self.get_validated_form_data()
        except Exception as e:
            self.update_status(ProcessingStatus(
                state="error",
                error=str(e),
                message=f"Validation error: {str(e)}"
            ))
            return
        
        # Start processing
        self.processing_worker = self.run_worker(
            self.process_media(form_data),
            exclusive=True
        )
        
        # Update UI state
        self.query_one("#process-button").disabled = True
        self.query_one("#clear-button").disabled = True
        self.query_one("#cancel-button").remove_class("hidden")
    
    @on(Button.Pressed, "#clear-button")
    def handle_clear(self) -> None:
        """Handle the clear button click."""
        # Clear all form fields
        self.query_one("#files-input", Input).value = ""
        self.query_one("#urls-input", TextArea).text = ""
        
        # Clear status
        self.update_status(ProcessingStatus(
            state="ready",
            message="Ready to process media"
        ))
    
    @on(Button.Pressed, "#cancel-button")
    async def handle_cancel(self) -> None:
        """Handle the cancel button click."""
        if self.processing_worker:
            await self.processing_worker.cancel()
            self.processing_worker = None
        
        # Reset UI state
        self.query_one("#process-button").disabled = False
        self.query_one("#clear-button").disabled = False
        self.query_one("#cancel-button").add_class("hidden")
        
        self.update_status(ProcessingStatus(
            state="cancelled",
            message="Processing cancelled"
        ))
    
    @on(NavigateToMediaType)
    async def handle_media_navigation(self, message: NavigateToMediaType) -> None:
        """Handle navigation to a different media type."""
        media_type = message.media_type
        
        # Save current state before navigating
        self.save_state()
        
        # Navigate to the appropriate screen
        screen_class = self.get_media_screen_class(media_type)
        if screen_class:
            new_screen = screen_class(self.app_instance, media_type)
            self.app.switch_screen(new_screen)
            logger.info(f"Navigated to {media_type} ingestion screen")
    
    def get_media_screen_class(self, media_type: str):
        """Get the screen class for a media type."""
        # Import here to avoid circular imports
        from .video_screen import VideoIngestScreen
        from .audio_screen import AudioIngestScreen
        from .pdf_screen import PDFIngestScreen
        from .document_screen import DocumentIngestScreen
        from .ebook_screen import EbookIngestScreen
        from .web_screen import WebIngestScreen
        
        screen_map = {
            "video": VideoIngestScreen,
            "audio": AudioIngestScreen,
            "pdf": PDFIngestScreen,
            "document": DocumentIngestScreen,
            "ebook": EbookIngestScreen,
            "web": WebIngestScreen,
        }
        
        return screen_map.get(media_type)
    
    def get_validated_form_data(self) -> Dict[str, Any]:
        """Get and validate form data."""
        # Get basic inputs
        files_input = self.query_one("#files-input").value
        urls_input = self.query_one("#urls-input").text
        
        # Parse files
        files = []
        if files_input:
            for file_path in files_input.split(","):
                file_path = file_path.strip()
                if file_path:
                    files.append(Path(file_path))
        
        # Parse URLs
        urls = []
        if urls_input:
            for line in urls_input.split("\n"):
                url = line.strip()
                if url:
                    urls.append(url)
        
        # Build form data
        form_data = {
            "files": files,
            "urls": urls,
        }
        
        return form_data
    
    @work
    async def process_media(self, form_data: Dict[str, Any]) -> None:
        """Process media files (override in subclasses)."""
        try:
            async for status in self.process_media_impl(form_data):
                self.call_from_thread(self.update_status, status)
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.call_from_thread(
                self.update_status,
                ProcessingStatus(
                    state="error",
                    error=str(e),
                    message=f"Processing failed: {str(e)}"
                )
            )
        finally:
            # Reset UI state
            self.call_from_thread(self.reset_ui_after_processing)
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Implementation of media processing (override in subclasses)."""
        # Default implementation - just simulate processing
        yield ProcessingStatus(
            state="processing",
            progress=0.0,
            message="Starting processing..."
        )
        
        await asyncio.sleep(2)
        
        yield ProcessingStatus(
            state="complete",
            progress=1.0,
            message="Processing complete"
        )
    
    def update_status(self, status: ProcessingStatus) -> None:
        """Update the status display."""
        self.processing_status = status
        
        # Update status message
        status_widget = self.query_one("#status-message", Static)
        status_widget.update(status.message or "Processing...")
        
        # Update file counter
        if status.total_files:
            counter_widget = self.query_one("#file-counter", Static)
            counter_widget.update(f"{status.files_processed}/{status.total_files} files")
        
        # Update progress bar
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        if status.state == "processing":
            progress_bar.remove_class("hidden")
            if status.progress is not None:
                progress_bar.update(progress=status.progress * 100)
        else:
            progress_bar.add_class("hidden")
    
    def reset_ui_after_processing(self) -> None:
        """Reset UI elements after processing completes."""
        self.query_one("#process-button").disabled = False
        self.query_one("#clear-button").disabled = False
        self.query_one("#cancel-button").add_class("hidden")