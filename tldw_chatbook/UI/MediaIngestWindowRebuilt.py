"""
Media Ingestion Window - Rebuilt following Textual best practices.

This module provides a clean, modern interface for ingesting media content
both from local files and remote sources via the TLDW API.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Union
from datetime import datetime

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    DirectoryTree,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Checkbox,
    LoadingIndicator,
    RichLog,
    Collapsible,
    RadioSet,
    RadioButton,
)
from textual.message import Message
from textual.validation import Number, URL

# Import ingestion modules
from ..Local_Ingestion import (
    ingest_local_file,
    detect_file_type,
    get_supported_extensions,
    FileIngestionError,
)
from ..tldw_api import TLDWAPIClient
from ..DB.Client_Media_DB_v2 import MediaDatabase

if TYPE_CHECKING:
    from ..app import TldwCli


# Custom Messages
class ProcessingStarted(Message):
    """Message sent when processing starts."""
    
    def __init__(self, file_count: int) -> None:
        self.file_count = file_count
        super().__init__()


class ProcessingComplete(Message):
    """Message sent when processing completes."""
    
    def __init__(self, results: List[Dict[str, Any]]) -> None:
        self.results = results
        super().__init__()


class ProcessingError(Message):
    """Message sent when processing encounters an error."""
    
    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__()


class LocalIngestionPanel(ScrollableContainer):
    """Panel for local file ingestion following Textual best practices."""
    
    DEFAULT_CSS = """
    LocalIngestionPanel {
        layout: vertical;
        padding: 1;
        height: 100%;
        background: $panel;
    }
    
    LocalIngestionPanel .file-selection-container {
        height: 15;
        min-height: 10;
        border: solid $primary;
        margin-bottom: 1;
        padding: 1;
    }
    
    LocalIngestionPanel .options-container {
        height: auto;
        margin-bottom: 1;
        padding: 1;
    }
    
    LocalIngestionPanel .process-button-container {
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    
    LocalIngestionPanel DirectoryTree {
        height: 100%;
        background: $boost;
    }
    
    LocalIngestionPanel Label {
        height: auto;
        margin-bottom: 1;
    }
    
    LocalIngestionPanel Input {
        height: 3;
    }
    """
    
    # Reactive properties
    selected_files: reactive[List[Path]] = reactive([])
    processing: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the local ingestion panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        # Get the media database from the app instance
        self.media_db = getattr(app_instance, 'media_db', None)
        self.supported_extensions = get_supported_extensions()
    
    def compose(self) -> ComposeResult:
        """Compose the local ingestion interface."""
        with Container(classes="file-selection-container"):
            yield Label("Select Files to Ingest:")
            yield DirectoryTree(".", id="file-tree")
        
        with Container(classes="options-container"):
            yield Label("Metadata (Optional):")
            with Horizontal():
                yield Input(placeholder="Title", id="local-title")
                yield Input(placeholder="Author", id="local-author")
            yield Input(
                placeholder="Keywords (comma-separated)", 
                id="local-keywords"
            )
            
            with Collapsible(Label("Advanced Options"), collapsed=True):
                yield Checkbox("Perform analysis/summarization", id="local-analyze")
                yield Checkbox("Enable chunking", value=True, id="local-chunk")
                with Horizontal():
                    yield Label("Chunk size:")
                    yield Input("500", id="local-chunk-size", validators=[Number()])
        
        with Container(classes="process-button-container"):
            yield Button(
                "Process Selected Files",
                variant="primary",
                id="local-process-btn",
                disabled=True
            )
    
    @on(DirectoryTree.FileSelected)
    def handle_file_selection(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection from the directory tree."""
        path = event.path
        
        # Check if file has supported extension
        if path.suffix.lower() in [ext for exts in self.supported_extensions.values() for ext in exts]:
            if path not in self.selected_files:
                self.selected_files.append(path)
                self.notify(f"Selected: {path.name}", severity="information")
                
                # Enable process button if files are selected
                process_btn = self.query_one("#local-process-btn", Button)
                process_btn.disabled = False
        else:
            self.notify(
                f"Unsupported file type: {path.suffix}",
                severity="warning"
            )
    
    @on(Button.Pressed, "#local-process-btn")
    def handle_process_button(self) -> None:
        """Handle the process button click."""
        if not self.selected_files:
            self.notify("No files selected", severity="warning")
            return
        
        if not self.processing:
            self.processing = True
            self.process_files()
    
    @work(exclusive=True, thread=True)
    async def process_files(self) -> None:
        """Process selected files in a background thread."""
        try:
            # Disable button during processing
            process_btn = self.query_one("#local-process-btn", Button)
            process_btn.disabled = True
            process_btn.label = "Processing..."
            
            # Get form values
            title = self.query_one("#local-title", Input).value or None
            author = self.query_one("#local-author", Input).value or None
            keywords_str = self.query_one("#local-keywords", Input).value
            keywords = [k.strip() for k in keywords_str.split(",")] if keywords_str else None
            
            perform_analysis = self.query_one("#local-analyze", Checkbox).value
            perform_chunking = self.query_one("#local-chunk", Checkbox).value
            chunk_size = int(self.query_one("#local-chunk-size", Input).value or "500")
            
            results = []
            errors = []
            
            # Check if media_db is available
            if not self.media_db:
                logger.error("Media database not available")
                self.notify("Database not initialized", severity="error")
                return
            
            # Process each file
            for file_path in self.selected_files:
                try:
                    logger.info(f"Processing file: {file_path}")
                    
                    chunk_options = {
                        "method": "sentences",
                        "size": chunk_size,
                        "overlap": 100,
                    } if perform_chunking else None
                    
                    result = ingest_local_file(
                        file_path=file_path,
                        media_db=self.media_db,
                        title=title or file_path.stem,
                        author=author,
                        keywords=keywords,
                        perform_analysis=perform_analysis,
                        chunk_options=chunk_options
                    )
                    
                    results.append({
                        "file": str(file_path),
                        "status": "success",
                        "media_id": result.get("media_id"),
                        "title": result.get("title")
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    errors.append({
                        "file": str(file_path),
                        "status": "error",
                        "error": str(e)
                    })
            
            # Post completion message
            self.post_message(ProcessingComplete(results + errors))
            
            # Show summary notification
            success_count = len(results)
            error_count = len(errors)
            if error_count == 0:
                self.notify(
                    f"Successfully processed {success_count} file(s)",
                    severity="information"
                )
            else:
                self.notify(
                    f"Processed {success_count} file(s), {error_count} error(s)",
                    severity="warning"
                )
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.post_message(ProcessingError(str(e)))
            self.notify(f"Processing failed: {e}", severity="error")
        
        finally:
            # Reset UI state
            self.processing = False
            self.selected_files = []
            process_btn.disabled = True
            process_btn.label = "Process Selected Files"


class RemoteIngestionPanel(ScrollableContainer):
    """Panel for remote TLDW API ingestion following Textual best practices."""
    
    DEFAULT_CSS = """
    RemoteIngestionPanel {
        layout: vertical;
        padding: 1;
        height: 100%;
        background: $panel;
    }
    
    RemoteIngestionPanel .media-type-container {
        height: auto;
        margin-bottom: 1;
        padding: 1;
    }
    
    RemoteIngestionPanel .url-input-container {
        height: auto;
        margin-bottom: 1;
        padding: 1;
    }
    
    RemoteIngestionPanel .dynamic-options {
        height: 20;
        max-height: 20;
        overflow-y: auto;
        border: solid $secondary;
        padding: 1;
        margin-bottom: 1;
        background: $boost;
    }
    
    RemoteIngestionPanel .api-button-container {
        height: 3;
        align: center middle;
    }
    
    RemoteIngestionPanel Label {
        height: auto;
        margin-bottom: 1;
    }
    
    RemoteIngestionPanel TextArea {
        height: 10;
        background: $boost;
    }
    """
    
    # Reactive properties
    media_type: reactive[str] = reactive("video")
    processing: reactive[bool] = reactive(False)
    
    # Media type options
    MEDIA_TYPES = [
        ("video", "Video"),
        ("audio", "Audio"),
        ("pdf", "PDF Document"),
        ("document", "Document (Word/ODT)"),
        ("ebook", "E-Book"),
        ("plaintext", "Plain Text"),
    ]
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the remote ingestion panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.api_client = None  # Will be initialized when needed
    
    def compose(self) -> ComposeResult:
        """Compose the remote ingestion interface."""
        with Container(classes="media-type-container"):
            yield Label("Select Media Type:")
            yield Select(
                self.MEDIA_TYPES,
                id="media-type-select"
            )
        
        with Container(classes="url-input-container"):
            yield Label("Enter URL(s):")
            yield TextArea(
                "",
                id="url-input",
                tab_behavior="indent"
            )
            yield Label("(One URL per line)", classes="dim")
        
        with ScrollableContainer(classes="dynamic-options", id="dynamic-options-container"):
            # This will be populated based on media type
            yield Container(id="dynamic-options")
        
        with Container(classes="api-button-container"):
            yield Button(
                "Process via TLDW API",
                variant="primary",
                id="api-process-btn"
            )
    
    @on(Select.Changed, "#media-type-select")
    def handle_media_type_change(self, event: Select.Changed) -> None:
        """Handle media type selection change."""
        self.media_type = str(event.value)
        self.update_dynamic_options()
    
    def update_dynamic_options(self) -> None:
        """Update the dynamic options based on selected media type."""
        container = self.query_one("#dynamic-options", Container)
        container.remove_children()
        
        # Build list of widgets to mount based on media type
        widgets_to_mount = []
        
        if self.media_type in ["video", "audio"]:
            widgets_to_mount.extend([
                Label("Transcription Options:"),
                Input(
                    placeholder="Transcription model",
                    value="deepdml/faster-whisper-large-v3-turbo-ct2",
                    id="transcription-model"
                ),
                Select(
                    [("en", "English"), ("auto", "Auto-detect")],
                    id="transcription-language"
                ),
                Checkbox("Include timestamps", value=True, id="include-timestamps"),
                Checkbox("Enable diarization", id="enable-diarization"),
            ])
        
        elif self.media_type == "pdf":
            widgets_to_mount.extend([
                Label("PDF Options:"),
                Select(
                    [
                        ("pymupdf4llm", "PyMuPDF for LLM"),
                        ("pymupdf", "PyMuPDF Standard"),
                        ("docling", "Docling"),
                    ],
                    id="pdf-engine"
                ),
            ])
        
        elif self.media_type == "ebook":
            widgets_to_mount.extend([
                Label("E-Book Options:"),
                Select(
                    [
                        ("filtered", "Filtered extraction"),
                        ("markdown", "Markdown format"),
                        ("basic", "Basic text"),
                    ],
                    id="extraction-method"
                ),
            ])
        
        # Add common chunking options section
        # Create a container for chunking options instead of Collapsible for dynamic content
        widgets_to_mount.extend([
            Label("Chunking Options:", classes="section-label"),
            Checkbox("Enable chunking", value=True, id="enable-chunking"),
            Select(
                [
                    ("sentences", "By sentences"),
                    ("paragraphs", "By paragraphs"),
                    ("tokens", "By tokens"),
                    ("semantic", "Semantic chunking"),
                ],
                id="chunk-method"
            ),
            Label("Chunk size:"),
            Input("500", id="chunk-size", validators=[Number()]),
        ])
        
        # Mount all widgets at once
        if widgets_to_mount:
            container.mount(*widgets_to_mount)
    
    @on(Button.Pressed, "#api-process-btn")
    def handle_process_button(self) -> None:
        """Handle the API process button click."""
        urls_text = self.query_one("#url-input", TextArea).text
        if not urls_text.strip():
            self.notify("Please enter at least one URL", severity="warning")
            return
        
        if not self.processing:
            self.processing = True
            self.process_remote_content(urls_text)
    
    @work(exclusive=True, thread=True)
    async def process_remote_content(self, urls_text: str) -> None:
        """Process remote content via TLDW API."""
        try:
            # Parse URLs
            urls = [url.strip() for url in urls_text.strip().split("\n") if url.strip()]
            
            # Initialize API client if needed
            if not self.api_client:
                # Get API configuration from app config
                api_config = self.app_instance.app_config.get("tldw_api", {})
                api_url = api_config.get("url", "http://localhost:8000")
                api_key = api_config.get("api_key")
                
                self.api_client = TLDWAPIClient(
                    base_url=api_url,
                    api_key=api_key
                )
            
            # Prepare request based on media type
            request_data = {
                "urls": urls,
                "perform_chunking": self.query_one("#enable-chunking", Checkbox).value,
                "chunk_size": int(self.query_one("#chunk-size", Input).value or "500"),
            }
            
            # Add media-specific options
            if self.media_type in ["video", "audio"]:
                request_data.update({
                    "transcription_model": self.query_one("#transcription-model", Input).value,
                    "transcription_language": self.query_one("#transcription-language", Select).value,
                    "timestamp_option": self.query_one("#include-timestamps", Checkbox).value,
                    "diarize": self.query_one("#enable-diarization", Checkbox).value,
                })
            elif self.media_type == "pdf":
                request_data["pdf_parsing_engine"] = self.query_one("#pdf-engine", Select).value
            elif self.media_type == "ebook":
                request_data["extraction_method"] = self.query_one("#extraction-method", Select).value
            
            # Process via API
            logger.info(f"Processing {len(urls)} URL(s) via TLDW API")
            
            # Call appropriate API method based on media type
            if self.media_type == "video":
                response = await self.api_client.process_video(**request_data)
            elif self.media_type == "audio":
                response = await self.api_client.process_audio(**request_data)
            elif self.media_type == "pdf":
                response = await self.api_client.process_pdf(**request_data)
            elif self.media_type == "document":
                response = await self.api_client.process_document(**request_data)
            elif self.media_type == "ebook":
                response = await self.api_client.process_ebook(**request_data)
            elif self.media_type == "plaintext":
                response = await self.api_client.process_plaintext(**request_data)
            else:
                raise ValueError(f"Unsupported media type: {self.media_type}")
            
            # Process results
            results = []
            if hasattr(response, 'results'):
                for result in response.results:
                    results.append({
                        "url": result.input_ref,
                        "status": result.status.lower(),
                        "media_type": result.media_type,
                        "content": result.content[:500] if result.content else None,
                        "error": result.error
                    })
            
            self.post_message(ProcessingComplete(results))
            self.notify(
                f"Processed {len(results)} item(s) via API",
                severity="information"
            )
            
        except Exception as e:
            logger.error(f"API processing error: {e}")
            self.post_message(ProcessingError(str(e)))
            self.notify(f"API processing failed: {e}", severity="error")
        
        finally:
            self.processing = False


class IngestionResultsPanel(Container):
    """Panel for displaying ingestion results."""
    
    DEFAULT_CSS = """
    IngestionResultsPanel {
        layout: vertical;
        height: 100%;
        border: solid $primary;
        padding: 1;
        background: $panel;
    }
    
    IngestionResultsPanel .results-header {
        height: 3;
        margin-bottom: 1;
    }
    
    IngestionResultsPanel RichLog {
        height: 1fr;
        border: solid $secondary;
        background: $boost;
        padding: 1;
    }
    
    IngestionResultsPanel Label {
        height: auto;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the results display panel."""
        with Container(classes="results-header"):
            yield Label("Processing Results:", id="results-label")
        yield RichLog(id="results-log", highlight=True, markup=True)
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a result to the display."""
        log = self.query_one("#results-log", RichLog)
        
        status = result.get("status", "unknown")
        if status == "success":
            icon = "✓"
            style = "green"
        elif status == "error":
            icon = "✗"
            style = "red"
        else:
            icon = "?"
            style = "yellow"
        
        # Format the result message
        file_or_url = result.get("file") or result.get("url", "Unknown")
        message = f"[{style}]{icon}[/{style}] {file_or_url}"
        
        if status == "success":
            if media_id := result.get("media_id"):
                message += f" (ID: {media_id})"
        elif status == "error":
            if error := result.get("error"):
                message += f"\n  Error: {error}"
        
        log.write(message)
    
    def clear_results(self) -> None:
        """Clear all results from the display."""
        log = self.query_one("#results-log", RichLog)
        log.clear()


class MediaIngestWindowRebuilt(Widget):
    """
    Main Media Ingestion Window following Textual best practices.
    
    This widget provides a tabbed interface for ingesting media content
    from both local files and remote sources via the TLDW API.
    """
    
    DEFAULT_CSS = """
    MediaIngestWindowRebuilt {
        layout: vertical;
        height: 100%;
        width: 100%;
    }
    
    MediaIngestWindowRebuilt TabbedContent {
        height: 2fr;
        margin-bottom: 1;
        background: $surface;
    }
    
    MediaIngestWindowRebuilt IngestionResultsPanel {
        height: 1fr;
        min-height: 10;
    }
    
    MediaIngestWindowRebuilt .loading-container {
        align: center middle;
        height: 100%;
    }
    
    MediaIngestWindowRebuilt TabPane {
        padding: 0;
    }
    """
    
    # Reactive properties
    current_tab: reactive[str] = reactive("local")
    is_processing: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the Media Ingestion Window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.info("MediaIngestWindowRebuilt initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the main ingestion interface."""
        with TabbedContent(initial="local-tab"):
            with TabPane("Local Files", id="local-tab"):
                yield LocalIngestionPanel(self.app_instance, id="local-panel")
            
            with TabPane("Remote (TLDW API)", id="remote-tab"):
                yield RemoteIngestionPanel(self.app_instance, id="remote-panel")
        
        yield IngestionResultsPanel(id="results-panel")
    
    @on(TabbedContent.TabActivated)
    def handle_tab_change(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab switching."""
        self.current_tab = "local" if event.tab.id == "local-tab" else "remote"
        logger.debug(f"Switched to {self.current_tab} tab")
    
    @on(ProcessingStarted)
    def handle_processing_started(self, event: ProcessingStarted) -> None:
        """Handle processing started event."""
        self.is_processing = True
        results_panel = self.query_one("#results-panel", IngestionResultsPanel)
        results_panel.clear_results()
        
        log = results_panel.query_one("#results-log", RichLog)
        log.write(f"[cyan]Processing {event.file_count} item(s)...[/cyan]")
    
    @on(ProcessingComplete)
    def handle_processing_complete(self, event: ProcessingComplete) -> None:
        """Handle processing completion."""
        self.is_processing = False
        results_panel = self.query_one("#results-panel", IngestionResultsPanel)
        
        for result in event.results:
            results_panel.add_result(result)
    
    @on(ProcessingError)
    def handle_processing_error(self, event: ProcessingError) -> None:
        """Handle processing errors."""
        self.is_processing = False
        results_panel = self.query_one("#results-panel", IngestionResultsPanel)
        
        log = results_panel.query_one("#results-log", RichLog)
        log.write(f"[red]Error: {event.error}[/red]")
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        logger.info("MediaIngestWindowRebuilt mounted")
        self.notify("Media Ingestion ready", severity="information")