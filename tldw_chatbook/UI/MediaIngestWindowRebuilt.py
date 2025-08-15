"""
Media Ingestion Window - Rebuilt with Textual best practices.

Features:
- Proper screen-based navigation for media types
- Remote/Local toggle with TLDW API integration
- Enhanced file picker integration
- Full support for all media types
- Reactive UI with status updates
- Following Textual's documentation best practices
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List, AsyncIterator
from pathlib import Path
import asyncio
from datetime import datetime
from loguru import logger

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical, Grid
from textual.widgets import (
    Static, Button, Input, Label, Select, Checkbox, 
    TextArea, RadioSet, RadioButton, ProgressBar, 
    TabbedContent, TabPane, Switch, DataTable, RichLog
)
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from textual.css.query import QueryError

# Import enhanced file picker
from ..Widgets.enhanced_file_picker import EnhancedFilePicker, RecentLocations, BookmarksManager

# Import ingestion backends
from ..Local_Ingestion.video_processing import process_video_file
from ..Local_Ingestion.audio_processing import process_audio_file
from ..Local_Ingestion.PDF_Processing_Lib import process_pdf
from ..Local_Ingestion.Document_Processing_Lib import process_document
from ..Local_Ingestion.Book_Ingestion_Lib import process_ebook
from ..config import get_cli_setting

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MediaProcessingStatus(Message):
    """Message for media processing status updates."""
    def __init__(self, status: str, progress: float = 0.0, message: str = "", error: str = ""):
        super().__init__()
        self.status = status
        self.progress = progress
        self.message = message
        self.error = error


class MediaIngestWindowRebuilt(Container):
    """
    Rebuilt Media Ingestion Window following Textual best practices.
    
    Features:
    - Tabbed interface for different media types
    - Remote/Local mode toggle
    - Enhanced file picker integration
    - Real-time processing status
    - Reactive UI updates
    """
    
    # CSS following Textual best practices
    DEFAULT_CSS = """
    MediaIngestWindowRebuilt {
        layout: vertical;
        height: 100%;
        width: 100%;
        background: $background;
    }
    
    /* Header Section */
    #ingest-header {
        height: 4;
        width: 100%;
        padding: 0 2;
        background: $panel;
        border-bottom: solid $primary;
    }
    
    .header-content {
        height: 100%;
        width: 100%;
        layout: horizontal;
        align: left middle;
    }
    
    .header-title {
        width: auto;
        text-style: bold;
        margin-right: 2;
    }
    
    .mode-toggle-container {
        width: auto;
        layout: horizontal;
        align: center middle;
        margin-left: auto;
    }
    
    .mode-label {
        margin-right: 1;
    }
    
    #mode-switch {
        width: auto;
    }
    
    .remote-config {
        width: auto;
        margin-left: 2;
    }
    
    /* Main Content */
    #ingest-content {
        height: 1fr;
        width: 100%;
        padding: 1 2;
    }
    
    /* Status Dashboard */
    .status-dashboard {
        height: auto;
        width: 100%;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary;
        background: $surface;
    }
    
    .status-header {
        layout: horizontal;
        width: 100%;
        margin-bottom: 1;
    }
    
    .status-message {
        width: 1fr;
    }
    
    .status-counter {
        width: auto;
        text-align: right;
    }
    
    #progress-bar {
        height: 1;
        width: 100%;
    }
    
    .hidden {
        display: none;
    }
    
    /* Tabs */
    TabbedContent {
        height: 100%;
        width: 100%;
    }
    
    TabPane {
        padding: 1;
    }
    
    /* Form Elements */
    .form-section {
        width: 100%;
        margin-bottom: 2;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .form-group {
        margin-bottom: 1;
    }
    
    .form-label {
        margin-bottom: 0;
    }
    
    .form-input {
        height: 3;
        width: 100%;
    }
    
    .form-textarea {
        height: 8;
        width: 100%;
    }
    
    .form-select {
        height: 3;
        width: 100%;
    }
    
    /* File List */
    .file-list-container {
        height: 20;
        width: 100%;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }
    
    #file-list {
        height: 100%;
        width: 100%;
    }
    
    /* Options Grid */
    .options-grid {
        grid-size: 2 4;
        grid-gutter: 1 1;
        width: 100%;
        margin-bottom: 1;
    }
    
    .option-item {
        height: 3;
        padding: 0 1;
    }
    
    /* Action Buttons */
    .action-buttons {
        width: 100%;
        height: 3;
        layout: horizontal;
        align: center middle;
        margin-top: 2;
    }
    
    .action-button {
        margin: 0 1;
        min-width: 12;
    }
    
    .process-button {
        background: $success;
    }
    
    .cancel-button {
        background: $error;
    }
    
    /* Processing Log */
    .processing-log {
        height: 15;
        width: 100%;
        border: solid $secondary;
        margin-top: 1;
    }
    """
    
    # Reactive properties
    is_remote_mode = reactive(False)
    processing_active = reactive(False)
    selected_files = reactive([])
    current_media_type = reactive("video")
    
    # Bindings for keyboard shortcuts
    BINDINGS = [
        Binding("ctrl+o", "open_file_picker", "Open Files"),
        Binding("ctrl+p", "toggle_processing", "Process/Stop"),
        Binding("ctrl+r", "toggle_remote_mode", "Toggle Remote/Local"),
        Binding("ctrl+c", "clear_all", "Clear All"),
    ]
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.processing_worker: Optional[asyncio.Task] = None
        self.recent_locations = RecentLocations(context="media_ingest")
        self.bookmarks = BookmarksManager(context="media_ingest")
        
        # Load TLDW API configuration
        self.tldw_api_config = {
            "base_url": get_cli_setting("tldw_api", "base_url", "http://127.0.0.1:8000"),
            "auth_token": get_cli_setting("tldw_api", "auth_token", "")
        }
        
        logger.info("MediaIngestWindowRebuilt initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the UI following Textual best practices."""
        # Header with mode toggle
        with Container(id="ingest-header"):
            with Horizontal(classes="header-content"):
                yield Static("📁 Media Ingestion", classes="header-title")
                
                with Container(classes="mode-toggle-container"):
                    yield Static("Mode:", classes="mode-label")
                    yield Switch(id="mode-switch", value=False)
                    yield Static("Local", id="mode-indicator", classes="mode-label")
                    yield Button("⚙️ Configure", 
                               id="configure-remote",
                               classes="remote-config hidden")
        
        # Main content area
        with Container(id="ingest-content"):
            # Status dashboard
            with Container(classes="status-dashboard"):
                with Horizontal(classes="status-header"):
                    yield Static("Ready to process media", 
                               id="status-message",
                               classes="status-message")
                    yield Static("0 files selected", 
                               id="status-counter",
                               classes="status-counter")
                yield ProgressBar(id="progress-bar", 
                                total=100,
                                show_eta=False,
                                classes="hidden")
            
            # Tabbed content for media types
            with TabbedContent():
                with TabPane("🎬 Video", id="tab-video"):
                    yield from self.compose_video_tab()
                
                with TabPane("🎵 Audio", id="tab-audio"):
                    yield from self.compose_audio_tab()
                
                with TabPane("📄 PDF", id="tab-pdf"):
                    yield from self.compose_pdf_tab()
                
                with TabPane("📝 Documents", id="tab-documents"):
                    yield from self.compose_documents_tab()
                
                with TabPane("📚 Ebooks", id="tab-ebooks"):
                    yield from self.compose_ebooks_tab()
                
                with TabPane("🌐 Web", id="tab-web"):
                    yield from self.compose_web_tab()
    
    def compose_video_tab(self) -> ComposeResult:
        """Compose the video ingestion tab."""
        with VerticalScroll():
            # File selection
            with Container(classes="form-section"):
                yield Label("Input Files", classes="section-title")
                
                with Container(classes="file-list-container"):
                    yield DataTable(id="video-file-list", zebra_stripes=True)
                
                with Horizontal():
                    yield Button("Browse Files...", 
                               id="video-browse-files",
                               variant="primary")
                    yield Button("Add URL", 
                               id="video-add-url",
                               variant="default")
                    yield Button("Clear", 
                               id="video-clear-files",
                               variant="warning")
            
            # Processing options
            with Container(classes="form-section"):
                yield Label("Processing Options", classes="section-title")
                
                with Grid(classes="options-grid"):
                    # Transcription options
                    yield Checkbox("Enable Transcription", 
                                 id="video-transcribe",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Diarization", 
                                 id="video-diarization",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Generate Subtitles", 
                                 id="video-subtitles",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Keyframes", 
                                 id="video-keyframes",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Generate Thumbnails", 
                                 id="video-thumbnails",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Compress Video", 
                                 id="video-compress",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("AI Summarization", 
                                 id="video-summarize",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Chapter Detection", 
                                 id="video-chapters",
                                 value=False,
                                 classes="option-item")
            
            # Advanced settings
            with Container(classes="form-section"):
                yield Label("Advanced Settings", classes="section-title")
                
                with Horizontal():
                    with Container(classes="form-group"):
                        yield Label("Transcription Model:", classes="form-label")
                        yield Select(
                            [("base", "Base"), ("small", "Small"), 
                             ("medium", "Medium"), ("large", "Large")],
                            id="video-model",
                            value="base",
                            classes="form-select"
                        )
                    
                    with Container(classes="form-group"):
                        yield Label("Language:", classes="form-label")
                        yield Select(
                            [("auto", "Auto-detect"), ("en", "English"),
                             ("es", "Spanish"), ("fr", "French"),
                             ("de", "German"), ("zh", "Chinese")],
                            id="video-language",
                            value="auto",
                            classes="form-select"
                        )
                
                with Container(classes="form-group"):
                    yield Label("Custom Prompt (for AI analysis):", classes="form-label")
                    yield TextArea(
                        "Summarize the key points and main topics discussed in this video.",
                        id="video-prompt",
                        classes="form-textarea"
                    )
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("🚀 Process Videos", 
                           id="video-process",
                           classes="action-button process-button")
                yield Button("⏹️ Cancel", 
                           id="video-cancel",
                           classes="action-button cancel-button hidden")
            
            # Processing log
            yield Label("Processing Log", classes="section-title")
            yield RichLog(id="video-log", 
                        classes="processing-log",
                        highlight=True,
                        markup=True)
    
    def compose_audio_tab(self) -> ComposeResult:
        """Compose the audio ingestion tab."""
        with VerticalScroll():
            # File selection
            with Container(classes="form-section"):
                yield Label("Input Files", classes="section-title")
                
                with Container(classes="file-list-container"):
                    yield DataTable(id="audio-file-list", zebra_stripes=True)
                
                with Horizontal():
                    yield Button("Browse Files...", 
                               id="audio-browse-files",
                               variant="primary")
                    yield Button("Add URL", 
                               id="audio-add-url",
                               variant="default")
                    yield Button("Clear", 
                               id="audio-clear-files",
                               variant="warning")
            
            # Processing options
            with Container(classes="form-section"):
                yield Label("Processing Options", classes="section-title")
                
                with Grid(classes="options-grid"):
                    yield Checkbox("Enable Transcription", 
                                 id="audio-transcribe",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Diarization", 
                                 id="audio-diarization",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Noise Reduction", 
                                 id="audio-denoise",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Generate Waveform", 
                                 id="audio-waveform",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Metadata", 
                                 id="audio-metadata",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("AI Summarization", 
                                 id="audio-summarize",
                                 value=True,
                                 classes="option-item")
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("🚀 Process Audio", 
                           id="audio-process",
                           classes="action-button process-button")
                yield Button("⏹️ Cancel", 
                           id="audio-cancel",
                           classes="action-button cancel-button hidden")
            
            # Processing log
            yield Label("Processing Log", classes="section-title")
            yield RichLog(id="audio-log", 
                        classes="processing-log",
                        highlight=True,
                        markup=True)
    
    def compose_pdf_tab(self) -> ComposeResult:
        """Compose the PDF ingestion tab."""
        with VerticalScroll():
            # File selection
            with Container(classes="form-section"):
                yield Label("Input Files", classes="section-title")
                
                with Container(classes="file-list-container"):
                    yield DataTable(id="pdf-file-list", zebra_stripes=True)
                
                with Horizontal():
                    yield Button("Browse Files...", 
                               id="pdf-browse-files",
                               variant="primary")
                    yield Button("Clear", 
                               id="pdf-clear-files",
                               variant="warning")
            
            # Processing options
            with Container(classes="form-section"):
                yield Label("Processing Options", classes="section-title")
                
                with Grid(classes="options-grid"):
                    yield Checkbox("Extract Text", 
                                 id="pdf-extract-text",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("OCR for Images", 
                                 id="pdf-ocr",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Images", 
                                 id="pdf-extract-images",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Tables", 
                                 id="pdf-extract-tables",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Generate Outline", 
                                 id="pdf-outline",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("AI Summarization", 
                                 id="pdf-summarize",
                                 value=True,
                                 classes="option-item")
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("🚀 Process PDFs", 
                           id="pdf-process",
                           classes="action-button process-button")
                yield Button("⏹️ Cancel", 
                           id="pdf-cancel",
                           classes="action-button cancel-button hidden")
            
            # Processing log
            yield Label("Processing Log", classes="section-title")
            yield RichLog(id="pdf-log", 
                        classes="processing-log",
                        highlight=True,
                        markup=True)
    
    def compose_documents_tab(self) -> ComposeResult:
        """Compose the documents ingestion tab."""
        with VerticalScroll():
            # File selection
            with Container(classes="form-section"):
                yield Label("Input Files", classes="section-title")
                
                with Container(classes="file-list-container"):
                    yield DataTable(id="doc-file-list", zebra_stripes=True)
                
                with Horizontal():
                    yield Button("Browse Files...", 
                               id="doc-browse-files",
                               variant="primary")
                    yield Button("Clear", 
                               id="doc-clear-files",
                               variant="warning")
            
            # Supported formats info
            yield Static(
                "📝 Supported: DOCX, DOC, ODT, RTF, TXT, MD, CSV, XML",
                classes="form-label"
            )
            
            # Processing options
            with Container(classes="form-section"):
                yield Label("Processing Options", classes="section-title")
                
                with Grid(classes="options-grid"):
                    yield Checkbox("Extract Text", 
                                 id="doc-extract-text",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Preserve Formatting", 
                                 id="doc-preserve-format",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Images", 
                                 id="doc-extract-images",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Tables", 
                                 id="doc-extract-tables",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Convert to Markdown", 
                                 id="doc-to-markdown",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("AI Summarization", 
                                 id="doc-summarize",
                                 value=True,
                                 classes="option-item")
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("🚀 Process Documents", 
                           id="doc-process",
                           classes="action-button process-button")
                yield Button("⏹️ Cancel", 
                           id="doc-cancel",
                           classes="action-button cancel-button hidden")
            
            # Processing log
            yield Label("Processing Log", classes="section-title")
            yield RichLog(id="doc-log", 
                        classes="processing-log",
                        highlight=True,
                        markup=True)
    
    def compose_ebooks_tab(self) -> ComposeResult:
        """Compose the ebooks ingestion tab."""
        with VerticalScroll():
            # File selection
            with Container(classes="form-section"):
                yield Label("Input Files", classes="section-title")
                
                with Container(classes="file-list-container"):
                    yield DataTable(id="ebook-file-list", zebra_stripes=True)
                
                with Horizontal():
                    yield Button("Browse Files...", 
                               id="ebook-browse-files",
                               variant="primary")
                    yield Button("Clear", 
                               id="ebook-clear-files",
                               variant="warning")
            
            # Supported formats info
            yield Static(
                "📚 Supported: EPUB, MOBI, AZW3, FB2, CBZ, CBR",
                classes="form-label"
            )
            
            # Processing options
            with Container(classes="form-section"):
                yield Label("Processing Options", classes="section-title")
                
                with Grid(classes="options-grid"):
                    yield Checkbox("Extract Text", 
                                 id="ebook-extract-text",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Metadata", 
                                 id="ebook-metadata",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Cover", 
                                 id="ebook-extract-cover",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Generate TOC", 
                                 id="ebook-toc",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Chapter Splitting", 
                                 id="ebook-split-chapters",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("AI Summarization", 
                                 id="ebook-summarize",
                                 value=True,
                                 classes="option-item")
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("🚀 Process Ebooks", 
                           id="ebook-process",
                           classes="action-button process-button")
                yield Button("⏹️ Cancel", 
                           id="ebook-cancel",
                           classes="action-button cancel-button hidden")
            
            # Processing log
            yield Label("Processing Log", classes="section-title")
            yield RichLog(id="ebook-log", 
                        classes="processing-log",
                        highlight=True,
                        markup=True)
    
    def compose_web_tab(self) -> ComposeResult:
        """Compose the web ingestion tab."""
        with VerticalScroll():
            # URL input
            with Container(classes="form-section"):
                yield Label("Web URLs", classes="section-title")
                
                yield TextArea(
                    placeholder="Enter URLs, one per line:\nhttps://example.com/article\nhttps://youtube.com/watch?v=...",
                    id="web-urls",
                    classes="form-textarea"
                )
                
                with Horizontal():
                    yield Button("Add URL", 
                               id="web-add-url",
                               variant="primary")
                    yield Button("Clear", 
                               id="web-clear-urls",
                               variant="warning")
            
            # Processing options
            with Container(classes="form-section"):
                yield Label("Processing Options", classes="section-title")
                
                with Grid(classes="options-grid"):
                    yield Checkbox("Extract Article Text", 
                                 id="web-extract-article",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Download Media", 
                                 id="web-download-media",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("Convert to Markdown", 
                                 id="web-to-markdown",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Extract Metadata", 
                                 id="web-metadata",
                                 value=True,
                                 classes="option-item")
                    
                    yield Checkbox("Screenshot Page", 
                                 id="web-screenshot",
                                 value=False,
                                 classes="option-item")
                    
                    yield Checkbox("AI Summarization", 
                                 id="web-summarize",
                                 value=True,
                                 classes="option-item")
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("🚀 Process URLs", 
                           id="web-process",
                           classes="action-button process-button")
                yield Button("⏹️ Cancel", 
                           id="web-cancel",
                           classes="action-button cancel-button hidden")
            
            # Processing log
            yield Label("Processing Log", classes="section-title")
            yield RichLog(id="web-log", 
                        classes="processing-log",
                        highlight=True,
                        markup=True)
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        # Set up file lists
        self.setup_file_lists()
        
        # Update mode indicator
        self.update_mode_indicator()
        
        logger.info("MediaIngestWindowRebuilt mounted")
    
    def setup_file_lists(self) -> None:
        """Set up the DataTable widgets for file lists."""
        for media_type in ["video", "audio", "pdf", "doc", "ebook"]:
            try:
                table = self.query_one(f"#{media_type}-file-list", DataTable)
                table.add_columns("File Name", "Size", "Type", "Status")
                table.cursor_type = "row"
            except QueryError:
                pass
    
    @on(Switch.Changed, "#mode-switch")
    def handle_mode_toggle(self, event: Switch.Changed) -> None:
        """Handle remote/local mode toggle."""
        self.is_remote_mode = event.value
        self.update_mode_indicator()
        
        # Show/hide remote configuration button
        config_button = self.query_one("#configure-remote")
        if self.is_remote_mode:
            config_button.remove_class("hidden")
        else:
            config_button.add_class("hidden")
        
        logger.info(f"Mode switched to: {'Remote' if self.is_remote_mode else 'Local'}")
    
    def update_mode_indicator(self) -> None:
        """Update the mode indicator text."""
        indicator = self.query_one("#mode-indicator", Static)
        indicator.update("Remote 🌐" if self.is_remote_mode else "Local 💻")
    
    @on(Button.Pressed)
    def handle_button_press(self, event: Button.Pressed) -> None:
        """Handle all button presses."""
        button_id = event.button.id
        if not button_id:
            return
        
        # Route to appropriate handler based on button ID
        if "browse-files" in button_id:
            media_type = button_id.split("-")[0]
            self.open_file_picker(media_type)
        elif "process" in button_id:
            media_type = button_id.split("-")[0]
            self.start_processing(media_type)
        elif "cancel" in button_id:
            media_type = button_id.split("-")[0]
            self.cancel_processing(media_type)
        elif "clear" in button_id:
            media_type = button_id.split("-")[0]
            self.clear_files(media_type)
        elif button_id == "configure-remote":
            self.show_remote_config()
    
    def open_file_picker(self, media_type: str) -> None:
        """Open the enhanced file picker for media type."""
        # Define file filters based on media type
        filters = {
            "video": ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm", "*.flv"],
            "audio": ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg", "*.wma"],
            "pdf": ["*.pdf"],
            "doc": ["*.docx", "*.doc", "*.odt", "*.rtf", "*.txt", "*.md"],
            "ebook": ["*.epub", "*.mobi", "*.azw3", "*.fb2", "*.cbz", "*.cbr"]
        }
        
        # Create and show file picker
        picker = EnhancedFilePicker(
            title=f"Select {media_type.title()} Files",
            filters=filters.get(media_type, ["*"]),
            multiple=True,
            show_hidden=False,
            on_file_selected=lambda files: self.add_files(media_type, files)
        )
        
        self.app.push_screen(picker)
    
    def add_files(self, media_type: str, files: List[Path]) -> None:
        """Add selected files to the appropriate list."""
        try:
            table = self.query_one(f"#{media_type}-file-list", DataTable)
            
            for file_path in files:
                # Get file info
                stat = file_path.stat()
                size = self.format_file_size(stat.st_size)
                file_type = file_path.suffix[1:].upper()
                
                # Add to table
                table.add_row(
                    file_path.name,
                    size,
                    file_type,
                    "Ready"
                )
                
                # Add to recent locations
                self.recent_locations.add(file_path, media_type)
            
            # Update counter
            self.update_file_counter(media_type)
            
            logger.info(f"Added {len(files)} files to {media_type} list")
            
        except Exception as e:
            logger.error(f"Error adding files: {e}")
    
    def format_file_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def update_file_counter(self, media_type: str = None) -> None:
        """Update the file counter display."""
        total_files = 0
        
        # Count files across all media types if not specified
        media_types = [media_type] if media_type else ["video", "audio", "pdf", "doc", "ebook"]
        
        for mt in media_types:
            try:
                table = self.query_one(f"#{mt}-file-list", DataTable)
                total_files += len(table.rows)
            except QueryError:
                pass
        
        counter = self.query_one("#status-counter", Static)
        counter.update(f"{total_files} files selected")
    
    def clear_files(self, media_type: str) -> None:
        """Clear files from the specified media type list."""
        try:
            table = self.query_one(f"#{media_type}-file-list", DataTable)
            table.clear()
            self.update_file_counter(media_type)
            logger.info(f"Cleared {media_type} file list")
        except QueryError:
            pass
    
    @work(exclusive=True)
    async def start_processing(self, media_type: str) -> None:
        """Start processing files for the specified media type."""
        logger.info(f"Starting {media_type} processing")
        
        # Update UI state
        self.processing_active = True
        self.update_ui_for_processing(media_type, True)
        
        try:
            # Get files from table
            table = self.query_one(f"#{media_type}-file-list", DataTable)
            files = [row[0] for row in table.rows]
            
            if not files:
                self.show_status("No files to process", "warning")
                return
            
            # Get processing options
            options = self.get_processing_options(media_type)
            
            # Process files
            if self.is_remote_mode:
                await self.process_files_remote(media_type, files, options)
            else:
                await self.process_files_local(media_type, files, options)
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.show_status(f"Error: {str(e)}", "error")
        finally:
            self.processing_active = False
            self.update_ui_for_processing(media_type, False)
    
    def get_processing_options(self, media_type: str) -> Dict[str, Any]:
        """Get processing options for the media type."""
        options = {}
        
        # Get common options
        if media_type in ["video", "audio"]:
            options["transcribe"] = self.query_one(f"#{media_type}-transcribe", Checkbox).value
            options["diarization"] = self.query_one(f"#{media_type}-diarization", Checkbox).value
            options["summarize"] = self.query_one(f"#{media_type}-summarize", Checkbox).value
        
        # Get media-specific options
        if media_type == "video":
            options["subtitles"] = self.query_one("#video-subtitles", Checkbox).value
            options["keyframes"] = self.query_one("#video-keyframes", Checkbox).value
            options["thumbnails"] = self.query_one("#video-thumbnails", Checkbox).value
            options["model"] = self.query_one("#video-model", Select).value
            options["language"] = self.query_one("#video-language", Select).value
            options["prompt"] = self.query_one("#video-prompt", TextArea).text
        
        # Add more media-specific options as needed
        
        return options
    
    async def process_files_local(self, media_type: str, files: List[str], options: Dict[str, Any]) -> None:
        """Process files locally using the ingestion backends."""
        log_widget = self.query_one(f"#{media_type}-log", RichLog)
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_bar.remove_class("hidden")
        
        total_files = len(files)
        
        for i, file_name in enumerate(files):
            file_path = Path(file_name)
            
            # Update status
            self.show_status(f"Processing {file_name} ({i+1}/{total_files})")
            progress_bar.update(progress=(i / total_files) * 100)
            
            # Log start
            log_widget.write(f"[cyan]Processing:[/cyan] {file_name}")
            
            try:
                # Call appropriate processor based on media type
                if media_type == "video":
                    result = await self.process_video_local(file_path, options)
                elif media_type == "audio":
                    result = await self.process_audio_local(file_path, options)
                elif media_type == "pdf":
                    result = await self.process_pdf_local(file_path, options)
                elif media_type == "doc":
                    result = await self.process_document_local(file_path, options)
                elif media_type == "ebook":
                    result = await self.process_ebook_local(file_path, options)
                else:
                    result = {"status": "error", "message": f"Unsupported media type: {media_type}"}
                
                # Log result
                if result.get("status") == "success":
                    log_widget.write(f"[green]✓ Completed:[/green] {file_name}")
                else:
                    log_widget.write(f"[red]✗ Failed:[/red] {file_name} - {result.get('message', 'Unknown error')}")
                
            except Exception as e:
                log_widget.write(f"[red]✗ Error:[/red] {file_name} - {str(e)}")
                logger.error(f"Error processing {file_name}: {e}")
            
            # Small delay between files
            await asyncio.sleep(0.1)
        
        # Complete
        progress_bar.update(progress=100)
        progress_bar.add_class("hidden")
        self.show_status(f"Completed processing {total_files} files", "success")
        log_widget.write(f"[bold green]Processing complete![/bold green]")
    
    async def process_files_remote(self, media_type: str, files: List[str], options: Dict[str, Any]) -> None:
        """Process files using the remote TLDW API."""
        # TODO: Implement remote API processing
        log_widget = self.query_one(f"#{media_type}-log", RichLog)
        log_widget.write("[yellow]Remote processing not yet implemented[/yellow]")
        self.show_status("Remote processing coming soon!", "info")
    
    async def process_video_local(self, file_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process a video file locally."""
        try:
            from ..Local_Ingestion.video_processing import LocalVideoProcessor
            from ..DB.Client_Media_DB_v2 import MediaDatabase
            
            # Initialize processor
            media_db = MediaDatabase()
            processor = LocalVideoProcessor(media_db)
            
            # Process the video
            result = await asyncio.to_thread(
                processor.process_local_video,
                str(file_path),
                transcribe=options.get("transcribe", True),
                whisper_model=options.get("model", "base"),
                language=options.get("language", "auto"),
                diarize=options.get("diarization", False),
                summarize=options.get("summarize", True),
                custom_prompt=options.get("prompt", "")
            )
            
            return {"status": "success", "message": "Video processed", "result": result}
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {"status": "error", "message": str(e)}
    
    async def process_audio_local(self, file_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process an audio file locally."""
        try:
            from ..Local_Ingestion.audio_processing import LocalAudioProcessor
            from ..DB.Client_Media_DB_v2 import MediaDatabase
            
            # Initialize processor
            media_db = MediaDatabase()
            processor = LocalAudioProcessor(media_db)
            
            # Process the audio
            result = await asyncio.to_thread(
                processor.process_local_audio,
                str(file_path),
                transcribe=options.get("transcribe", True),
                whisper_model=options.get("model", "base"),
                language=options.get("language", "auto"),
                diarize=options.get("diarization", False),
                summarize=options.get("summarize", True)
            )
            
            return {"status": "success", "message": "Audio processed", "result": result}
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {"status": "error", "message": str(e)}
    
    async def process_pdf_local(self, file_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process a PDF file locally."""
        try:
            from ..Local_Ingestion.PDF_Processing_Lib import process_and_store_pdf
            
            # Process the PDF
            result = await asyncio.to_thread(
                process_and_store_pdf,
                str(file_path),
                perform_ocr=options.get("ocr", True),
                extract_images=options.get("extract_images", False),
                extract_tables=options.get("extract_tables", True),
                summarize=options.get("summarize", True)
            )
            
            return {"status": "success", "message": "PDF processed", "result": result}
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"status": "error", "message": str(e)}
    
    async def process_document_local(self, file_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document file locally."""
        try:
            from ..Local_Ingestion.Document_Processing_Lib import process_and_store_document
            
            # Process the document
            result = await asyncio.to_thread(
                process_and_store_document,
                str(file_path),
                preserve_formatting=options.get("preserve_format", True),
                extract_images=options.get("extract_images", False),
                extract_tables=options.get("extract_tables", True),
                to_markdown=options.get("to_markdown", True),
                summarize=options.get("summarize", True)
            )
            
            return {"status": "success", "message": "Document processed", "result": result}
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"status": "error", "message": str(e)}
    
    async def process_ebook_local(self, file_path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process an ebook file locally."""
        try:
            from ..Local_Ingestion.Book_Ingestion_Lib import process_and_store_ebook
            
            # Process the ebook
            result = await asyncio.to_thread(
                process_and_store_ebook,
                str(file_path),
                extract_metadata=options.get("metadata", True),
                extract_cover=options.get("extract_cover", True),
                generate_toc=options.get("toc", True),
                split_chapters=options.get("split_chapters", True),
                summarize=options.get("summarize", True)
            )
            
            return {"status": "success", "message": "Ebook processed", "result": result}
            
        except Exception as e:
            logger.error(f"Error processing ebook: {e}")
            return {"status": "error", "message": str(e)}
    
    def cancel_processing(self, media_type: str) -> None:
        """Cancel ongoing processing."""
        if self.processing_worker:
            self.processing_worker.cancel()
        
        self.processing_active = False
        self.update_ui_for_processing(media_type, False)
        self.show_status("Processing cancelled", "warning")
        
        log_widget = self.query_one(f"#{media_type}-log", RichLog)
        log_widget.write("[yellow]Processing cancelled by user[/yellow]")
    
    def update_ui_for_processing(self, media_type: str, is_processing: bool) -> None:
        """Update UI elements based on processing state."""
        try:
            process_button = self.query_one(f"#{media_type}-process", Button)
            cancel_button = self.query_one(f"#{media_type}-cancel", Button)
            
            if is_processing:
                process_button.disabled = True
                cancel_button.remove_class("hidden")
            else:
                process_button.disabled = False
                cancel_button.add_class("hidden")
        except QueryError:
            pass
    
    def show_status(self, message: str, level: str = "info") -> None:
        """Show a status message."""
        status_widget = self.query_one("#status-message", Static)
        
        # Add color based on level
        if level == "error":
            styled_message = f"[red]{message}[/red]"
        elif level == "warning":
            styled_message = f"[yellow]{message}[/yellow]"
        elif level == "success":
            styled_message = f"[green]{message}[/green]"
        else:
            styled_message = message
        
        status_widget.update(styled_message)
    
    def show_remote_config(self) -> None:
        """Show remote configuration dialog."""
        # TODO: Implement remote configuration dialog
        self.app.notify("Remote configuration dialog coming soon!", severity="information")
    
    # Action handlers
    def action_open_file_picker(self) -> None:
        """Action to open file picker."""
        # Get current tab
        # For now, default to video
        self.open_file_picker("video")
    
    def action_toggle_processing(self) -> None:
        """Action to toggle processing."""
        if self.processing_active:
            self.cancel_processing(self.current_media_type)
        else:
            self.start_processing(self.current_media_type)
    
    def action_toggle_remote_mode(self) -> None:
        """Action to toggle remote/local mode."""
        switch = self.query_one("#mode-switch", Switch)
        switch.value = not switch.value
    
    def action_clear_all(self) -> None:
        """Action to clear all files."""
        for media_type in ["video", "audio", "pdf", "doc", "ebook"]:
            self.clear_files(media_type)