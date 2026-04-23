"""
Media Ingestion Window - Rebuilt following Textual best practices.

This module provides a clean, modern interface for ingesting media content
from local files and from server-backed ingestion sources.
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
    ProgressBar,
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
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Widgets.Media import MediaIngestionSourcePanel

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
    
    .info-label {
        color: $accent;
        text-style: italic;
        margin-bottom: 1;
    }
    
    #ingest-progress-bar {
        margin-top: 1;
        height: 1;
    }
    
    #ingest-progress-bar.hidden {
        display: none;
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
            yield Label("", id="batch-info-label", classes="info-label")
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
            yield ProgressBar(total=100, show_eta=False, id="ingest-progress-bar", classes="hidden")
    
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
                
                # Update batch label
                count = len(self.selected_files)
                label = self.query_one("#batch-info-label", Label)
                label.update(f"Applying metadata to {count} selected file{'s' if count > 1 else ''}")
                label.remove_class("hidden")
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
            
            # Show progress bar
            progress_bar = self.query_one("#ingest-progress-bar", ProgressBar)
            progress_bar.remove_class("hidden")
            progress_bar.update(total=len(self.selected_files), progress=0)
            
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
                
                # Update progress
                progress_bar.advance(1)
            
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
            
            # Reset and hide progress bar
            progress_bar = self.query_one("#ingest-progress-bar", ProgressBar)
            progress_bar.add_class("hidden")
            progress_bar.update(progress=0)
            
            # Reset label
            label = self.query_one("#batch-info-label", Label)
            label.update("")


class RemoteIngestionPanel(ScrollableContainer):
    """Panel for server-backed media ingest jobs."""
    
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

    RemoteIngestionPanel #remote-panel-disabled {
        padding: 2;
        color: $text-muted;
        text-style: italic;
    }

    RemoteIngestionPanel #remote-job-status {
        border: solid $secondary;
        background: $boost;
        min-height: 5;
        padding: 1;
        margin-top: 1;
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
    media_type: reactive[str] = reactive("document")
    processing: reactive[bool] = reactive(False)
    runtime_backend: reactive[str] = reactive("local")
    
    # Media type options
    MEDIA_TYPES = [
        ("Document", "document"),
        ("PDF Document", "pdf"),
        ("E-Book", "ebook"),
        ("Audio", "audio"),
        ("Video", "video"),
        ("Email", "email"),
        ("Code", "code"),
    ]
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the remote ingestion panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.scope_service = getattr(app_instance, "media_reading_scope_service", None)
        self.runtime_state = getattr(app_instance, "media_runtime_state", None)
        self.last_batch_id: Optional[str] = None
        self.current_jobs: list[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the remote ingestion interface."""
        yield Static("Server ingest jobs require server mode.", id="remote-panel-disabled")
        with Container(id="remote-panel-main"):
            with Container(classes="media-type-container"):
                yield Label("Select Media Type:")
                media_select = Select(
                    self.MEDIA_TYPES,
                    id="media-type-select",
                )
                media_select.value = "document"
                yield media_select

            with Container(classes="url-input-container"):
                yield Label("Enter URL(s):")
                yield TextArea(
                    "",
                    id="url-input",
                    tab_behavior="indent",
                )
                yield Label("(One URL per line)", classes="dim")

            with ScrollableContainer(classes="dynamic-options", id="dynamic-options-container"):
                yield Container(id="dynamic-options")

            with Container(classes="web-content-options", id="web-content-options"):
                yield Label("Web Content Ingest:", classes="section-label")
                web_scrape_method = Select(
                    [
                        ("Individual URLs", "individual"),
                        ("Sitemap", "sitemap"),
                        ("URL Level", "url_level"),
                        ("Recursive Scraping", "recursive_scraping"),
                    ],
                    id="web-scrape-method",
                )
                web_scrape_method.value = "individual"
                yield web_scrape_method
                yield Label("Max pages:")
                yield Input("3", id="web-max-pages", validators=[Number()])
                yield Label("Max depth:")
                yield Input("3", id="web-max-depth", validators=[Number()])
                yield Checkbox("Perform analysis", value=True, id="web-perform-analysis")

            with Container(classes="api-button-container"):
                yield Button(
                    "Submit Server Jobs",
                    variant="primary",
                    id="api-process-btn",
                )
                yield Button(
                    "Ingest Web Content",
                    variant="success",
                    id="web-content-ingest-btn",
                )
                yield Button(
                    "Refresh Batch",
                    id="refresh-batch-btn",
                    disabled=True,
                )
                yield Button(
                    "Watch Batch",
                    id="watch-batch-btn",
                    disabled=True,
                )
                yield Button(
                    "Cancel Batch",
                    id="cancel-batch-btn",
                    disabled=True,
                )
            with Container(classes="job-control-container"):
                yield Label("Load known server batch:")
                yield Input(
                    "",
                    placeholder="Batch ID",
                    id="server-ingest-batch-id",
                )
                yield Button(
                    "Load Batch",
                    id="load-batch-btn",
                    disabled=True,
                )
                yield Label("Server Job Controls:")
                yield Select(
                    [],
                    prompt="Select server job",
                    allow_blank=True,
                    id="server-ingest-job-select",
                    disabled=True,
                )
                yield Button(
                    "Cancel Selected Job",
                    id="cancel-job-btn",
                    variant="error",
                    disabled=True,
                )
            yield Static("No server ingest jobs submitted yet.", id="remote-job-status")

    async def _maybe_await(self, value: Any) -> Any:
        import inspect

        if inspect.isawaitable(value):
            return await value
        return value

    def on_mount(self) -> None:
        """Initialize defaults once dynamic controls are mounted."""
        self.media_type = "document"
        self.update_dynamic_options()
        self.run_worker(self.refresh_for_mode(), exclusive=True)

    def _current_runtime_backend(self) -> str:
        runtime_backend = self.runtime_backend
        if self.runtime_state is not None:
            runtime_backend = str(getattr(self.runtime_state, "runtime_backend", runtime_backend) or "local")
        normalized_backend = str(runtime_backend or "local").strip().lower()
        if normalized_backend not in {"local", "server"}:
            return "local"
        return normalized_backend

    def _show_server_ui(self, enabled: bool) -> None:
        self.query_one("#remote-panel-disabled", Static).display = not enabled
        self.query_one("#remote-panel-main", Container).display = enabled

    def _set_process_controls_disabled(self, disabled: bool) -> None:
        self.query_one("#media-type-select", Select).disabled = disabled
        self.query_one("#url-input", TextArea).disabled = disabled
        self.query_one("#api-process-btn", Button).disabled = disabled
        for selector, widget_type in (
            ("#web-content-ingest-btn", Button),
            ("#web-scrape-method", Select),
            ("#web-max-pages", Input),
            ("#web-max-depth", Input),
            ("#web-perform-analysis", Checkbox),
        ):
            try:
                self.query_one(selector, widget_type).disabled = disabled
            except Exception:
                pass

    def _set_batch_controls_disabled(self, disabled: bool) -> None:
        self.query_one("#refresh-batch-btn", Button).disabled = disabled
        self.query_one("#watch-batch-btn", Button).disabled = disabled
        self.query_one("#cancel-batch-btn", Button).disabled = disabled

    def _set_job_controls_disabled(self, disabled: bool) -> None:
        try:
            self.query_one("#server-ingest-job-select", Select).disabled = disabled
            self.query_one("#cancel-job-btn", Button).disabled = disabled
        except Exception:
            pass

    def _set_batch_lookup_controls_disabled(self, disabled: bool) -> None:
        try:
            self.query_one("#server-ingest-batch-id", Input).disabled = disabled
            self.query_one("#load-batch-btn", Button).disabled = disabled
        except Exception:
            pass

    async def refresh_for_mode(self) -> None:
        """Refresh the panel for the current runtime backend."""
        self.runtime_backend = self._current_runtime_backend()
        enabled = self.runtime_backend == "server" and self.scope_service is not None
        self._show_server_ui(enabled)
        self._set_process_controls_disabled(not enabled)
        self._set_batch_lookup_controls_disabled(not enabled)
        self._set_batch_controls_disabled(not enabled or not self.last_batch_id)
        self._set_job_controls_disabled(not enabled or not self.current_jobs)
        if self.runtime_backend != "server":
            self.query_one("#remote-job-status", Static).update("Server ingest jobs require server mode.")
        elif self.scope_service is None:
            self.query_one("#remote-job-status", Static).update("Media ingest job service is unavailable.")
    
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
            transcription_language = Select(
                [("English", "en"), ("Auto-detect", "auto")],
                id="transcription-language",
            )
            transcription_language.value = "en"
            widgets_to_mount.extend([
                Label("Transcription Options:"),
                Input(
                    placeholder="Transcription model",
                    value="deepdml/faster-whisper-large-v3-turbo-ct2",
                    id="transcription-model"
                ),
                transcription_language,
                Checkbox("Include timestamps", value=True, id="include-timestamps"),
                Checkbox("Enable diarization", id="enable-diarization"),
            ])
        
        elif self.media_type == "pdf":
            pdf_engine = Select(
                [
                    ("PyMuPDF for LLM", "pymupdf4llm"),
                    ("PyMuPDF Standard", "pymupdf"),
                    ("Docling", "docling"),
                ],
                id="pdf-engine",
            )
            pdf_engine.value = "pymupdf4llm"
            widgets_to_mount.extend([
                Label("PDF Options:"),
                pdf_engine,
            ])
        
        elif self.media_type == "ebook":
            extraction_method = Select(
                [
                    ("Filtered extraction", "filtered"),
                    ("Markdown format", "markdown"),
                    ("Basic text", "basic"),
                ],
                id="extraction-method",
            )
            extraction_method.value = "filtered"
            widgets_to_mount.extend([
                Label("E-Book Options:"),
                extraction_method,
            ])
        
        # Add common chunking options section
        # Create a container for chunking options instead of Collapsible for dynamic content
        chunk_method = Select(
            [
                ("By sentences", "sentences"),
                ("By paragraphs", "paragraphs"),
                ("By tokens", "tokens"),
                ("Semantic chunking", "semantic"),
            ],
            id="chunk-method",
        )
        chunk_method.value = "sentences"
        widgets_to_mount.extend([
            Label("Chunking Options:", classes="section-label"),
            Checkbox("Enable chunking", value=True, id="enable-chunking"),
            chunk_method,
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
            self.run_worker(self.process_remote_content(urls_text), exclusive=True)

    @on(Button.Pressed, "#web-content-ingest-btn")
    def handle_web_content_ingest_button(self) -> None:
        """Handle server web-content ingest button click."""
        urls_text = self.query_one("#url-input", TextArea).text
        if not urls_text.strip():
            self.notify("Please enter at least one URL", severity="warning")
            return

        if not self.processing:
            self.run_worker(self.process_web_content_ingest(urls_text), exclusive=True)

    @on(Button.Pressed, "#refresh-batch-btn")
    def handle_refresh_batch(self) -> None:
        """Refresh the last submitted server ingest batch."""
        if self.processing:
            return
        self.run_worker(self.refresh_last_batch_jobs(), exclusive=True)

    @on(Button.Pressed, "#load-batch-btn")
    def handle_load_batch(self) -> None:
        """Load a known server ingest batch by its batch id."""
        if self.processing:
            return
        self.run_worker(self.load_batch_by_id(), exclusive=True)

    @on(Button.Pressed, "#watch-batch-btn")
    def handle_watch_batch(self) -> None:
        """Watch server ingest job events for the last submitted batch."""
        if self.processing:
            return
        self.run_worker(self.watch_last_batch_events(), exclusive=True)

    @on(Button.Pressed, "#cancel-batch-btn")
    def handle_cancel_batch(self) -> None:
        """Cancel the last submitted server ingest batch."""
        if self.processing:
            return
        self.run_worker(self.cancel_last_batch_jobs(reason="user-requested"), exclusive=True)

    @on(Button.Pressed, "#cancel-job-btn")
    def handle_cancel_selected_job(self) -> None:
        """Cancel the selected server ingest job."""
        if self.processing:
            return
        self.run_worker(self.cancel_selected_job(reason="user-requested"), exclusive=True)
    
    async def process_remote_content(self, urls_text: str) -> None:
        """Submit server-backed media ingest jobs."""
        try:
            self.runtime_backend = self._current_runtime_backend()
            if self.runtime_backend != "server":
                self.notify("Server ingest jobs require server mode.", severity="warning")
                return
            if self.scope_service is None:
                self.notify("Media ingest job service is unavailable.", severity="error")
                return

            self.processing = True
            self._set_process_controls_disabled(True)

            # Parse URLs
            urls = [url.strip() for url in urls_text.strip().split("\n") if url.strip()]
            if not urls:
                self.notify("Please enter at least one URL", severity="warning")
                return
            
            # Prepare request based on media type
            request_data = {
                "perform_chunking": self.query_one("#enable-chunking", Checkbox).value,
                "chunk_method": self.query_one("#chunk-method", Select).value,
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
            
            # Submit jobs through the runtime-policy-aware media seam.
            logger.info(f"Submitting {len(urls)} URL(s) as server media ingest job(s)")
            response = await self._maybe_await(
                self.scope_service.submit_media_ingest_jobs(
                    mode="server",
                    media_type=self.media_type,
                    urls=urls,
                    **request_data,
                )
            )
            self._render_submission_response(dict(response or {}))
            self.notify("Server ingest jobs submitted", severity="information")
        except Exception as exc:
            logger.error(f"Error submitting server ingest jobs: {exc}", exc_info=True)
            self.query_one("#remote-job-status", Static).update(f"Error: {exc}")
            self.notify(f"Server ingest job submission failed: {exc}", severity="error")
        finally:
            self.processing = False
            self._set_process_controls_disabled(self.runtime_backend != "server" or self.scope_service is None)
            self._set_batch_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.last_batch_id
            )
            self._set_batch_lookup_controls_disabled(self.runtime_backend != "server" or self.scope_service is None)
            self._set_job_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.current_jobs
            )

    async def process_web_content_ingest(self, urls_text: str) -> None:
        """Run the server web-content ingest helper for URL scraping/crawling."""
        try:
            self.runtime_backend = self._current_runtime_backend()
            if self.runtime_backend != "server":
                self.notify("Server web-content ingest requires server mode.", severity="warning")
                return
            if self.scope_service is None or not hasattr(self.scope_service, "ingest_web_content"):
                self.notify("Server web-content ingest service is unavailable.", severity="error")
                return

            self.processing = True
            self._set_process_controls_disabled(True)

            urls = [url.strip() for url in urls_text.strip().split("\n") if url.strip()]
            if not urls:
                self.notify("Please enter at least one URL", severity="warning")
                return

            request_data = {
                "scrape_method": self.query_one("#web-scrape-method", Select).value,
                "max_pages": int(self.query_one("#web-max-pages", Input).value or "3"),
                "max_depth": int(self.query_one("#web-max-depth", Input).value or "3"),
                "perform_analysis": self.query_one("#web-perform-analysis", Checkbox).value,
                "perform_chunking": self.query_one("#enable-chunking", Checkbox).value,
                "chunk_method": self.query_one("#chunk-method", Select).value,
                "chunk_size": int(self.query_one("#chunk-size", Input).value or "500"),
            }

            logger.info(f"Running server web-content ingest for {len(urls)} URL(s)")
            response = await self._maybe_await(
                self.scope_service.ingest_web_content(
                    mode="server",
                    urls=urls,
                    **request_data,
                )
            )
            self._render_web_content_response(dict(response or {}))
            self.notify("Server web-content ingest completed", severity="information")
        except Exception as exc:
            logger.error(f"Error running server web-content ingest: {exc}", exc_info=True)
            self.query_one("#remote-job-status", Static).update(f"Error: {exc}")
            self.notify(f"Server web-content ingest failed: {exc}", severity="error")
        finally:
            self.processing = False
            self._set_process_controls_disabled(self.runtime_backend != "server" or self.scope_service is None)
            self._set_batch_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.last_batch_id
            )
            self._set_batch_lookup_controls_disabled(self.runtime_backend != "server" or self.scope_service is None)
            self._set_job_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.current_jobs
            )

    async def load_batch_by_id(self, batch_id: str | None = None) -> None:
        """Load a server ingest batch when the user already has its batch id."""
        requested_batch_id = str(batch_id or "").strip()
        if not requested_batch_id:
            try:
                requested_batch_id = self.query_one("#server-ingest-batch-id", Input).value.strip()
            except Exception:
                requested_batch_id = ""
        if not requested_batch_id:
            self.notify("Enter a server ingest batch ID to load.", severity="warning")
            return
        self.last_batch_id = requested_batch_id
        await self.refresh_last_batch_jobs()

    async def watch_last_batch_events(self, *, after_id: int = 0) -> None:
        """Consume live server ingest events for the current batch."""
        if not self.last_batch_id:
            self.notify("No server ingest batch to watch.", severity="warning")
            return
        self.runtime_backend = self._current_runtime_backend()
        if self.runtime_backend != "server" or self.scope_service is None:
            self.notify("Server ingest jobs require server mode.", severity="warning")
            return

        stream_events = getattr(self.scope_service, "stream_media_ingest_job_events", None)
        if stream_events is None:
            self.notify("Server ingest job event stream is unavailable.", severity="error")
            return

        self.processing = True
        self._set_batch_controls_disabled(True)
        try:
            async for event in stream_events(mode="server", batch_id=self.last_batch_id, after_id=after_id):
                self._apply_job_stream_event(dict(event or {}))
            self.notify("Server ingest batch event stream ended", severity="information")
        except Exception as exc:
            logger.error(f"Error watching server ingest batch: {exc}", exc_info=True)
            self.query_one("#remote-job-status", Static).update(f"Error: {exc}")
            self.notify(f"Server ingest batch watch failed: {exc}", severity="error")
        finally:
            self.processing = False
            self._set_batch_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.last_batch_id
            )
            self._set_batch_lookup_controls_disabled(self.runtime_backend != "server" or self.scope_service is None)
            self._set_job_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.current_jobs
            )

    async def refresh_last_batch_jobs(self, *, limit: int = 100) -> None:
        """Refresh the current server ingest batch status."""
        if not self.last_batch_id:
            self.notify("No server ingest batch to refresh.", severity="warning")
            return
        self.runtime_backend = self._current_runtime_backend()
        if self.runtime_backend != "server" or self.scope_service is None:
            self.notify("Server ingest jobs require server mode.", severity="warning")
            return

        self.processing = True
        self._set_batch_controls_disabled(True)
        try:
            response = await self._maybe_await(
                self.scope_service.list_media_ingest_jobs(
                    mode="server",
                    batch_id=self.last_batch_id,
                    limit=limit,
                )
            )
            self._render_job_list_response(dict(response or {}))
            self.notify("Server ingest batch refreshed", severity="information")
        except Exception as exc:
            logger.error(f"Error refreshing server ingest batch: {exc}", exc_info=True)
            self.query_one("#remote-job-status", Static).update(f"Error: {exc}")
            self.notify(f"Server ingest batch refresh failed: {exc}", severity="error")
        finally:
            self.processing = False
            self._set_batch_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.last_batch_id
            )
            self._set_batch_lookup_controls_disabled(self.runtime_backend != "server" or self.scope_service is None)
            self._set_job_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.current_jobs
            )

    async def cancel_last_batch_jobs(self, *, reason: str | None = None) -> None:
        """Cancel the current server ingest batch."""
        if not self.last_batch_id:
            self.notify("No server ingest batch to cancel.", severity="warning")
            return
        self.runtime_backend = self._current_runtime_backend()
        if self.runtime_backend != "server" or self.scope_service is None:
            self.notify("Server ingest jobs require server mode.", severity="warning")
            return

        self.processing = True
        self._set_batch_controls_disabled(True)
        try:
            response = await self._maybe_await(
                self.scope_service.cancel_media_ingest_jobs_batch(
                    mode="server",
                    batch_id=self.last_batch_id,
                    reason=reason,
                )
            )
            self._render_batch_cancel_response(dict(response or {}))
            self.notify("Server ingest batch cancellation requested", severity="information")
        except Exception as exc:
            logger.error(f"Error cancelling server ingest batch: {exc}", exc_info=True)
            self.query_one("#remote-job-status", Static).update(f"Error: {exc}")
            self.notify(f"Server ingest batch cancellation failed: {exc}", severity="error")
        finally:
            self.processing = False
            self._set_batch_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.last_batch_id
            )
            self._set_batch_lookup_controls_disabled(self.runtime_backend != "server" or self.scope_service is None)
            self._set_job_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.current_jobs
            )

    def _selected_job_id(self) -> Optional[str]:
        try:
            value = self.query_one("#server-ingest-job-select", Select).value
        except Exception:
            return None
        if value in (None, "", Select.BLANK):
            return None
        return str(value)

    async def cancel_selected_job(self, *, reason: str | None = None) -> None:
        """Cancel the selected server ingest job."""
        job_id = self._selected_job_id()
        if not job_id:
            self.notify("Select a server ingest job to cancel.", severity="warning")
            return
        self.runtime_backend = self._current_runtime_backend()
        if self.runtime_backend != "server" or self.scope_service is None:
            self.notify("Server ingest jobs require server mode.", severity="warning")
            return

        self.processing = True
        self._set_batch_controls_disabled(True)
        self._set_job_controls_disabled(True)
        try:
            response = await self._maybe_await(
                self.scope_service.cancel_media_ingest_job(
                    mode="server",
                    job_id=job_id,
                    reason=reason,
                )
            )
            self._render_job_cancel_response(dict(response or {}), fallback_job_id=job_id)
            self.notify("Server ingest job cancellation requested", severity="information")
        except Exception as exc:
            logger.error(f"Error cancelling server ingest job {job_id}: {exc}", exc_info=True)
            self.query_one("#remote-job-status", Static).update(f"Error: {exc}")
            self.notify(f"Server ingest job cancellation failed: {exc}", severity="error")
        finally:
            self.processing = False
            self._set_batch_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.last_batch_id
            )
            self._set_batch_lookup_controls_disabled(self.runtime_backend != "server" or self.scope_service is None)
            self._set_job_controls_disabled(
                self.runtime_backend != "server" or self.scope_service is None or not self.current_jobs
            )

    @staticmethod
    def _job_source_id(job: Dict[str, Any]) -> str:
        source_id = job.get("source_id") or job.get("job_id")
        if source_id not in (None, ""):
            return str(source_id)
        job_id = job.get("id")
        if isinstance(job_id, str) and ":" in job_id:
            return job_id.rsplit(":", 1)[-1]
        return str(job_id or "")

    def _update_job_selection(self, jobs: list[Dict[str, Any]]) -> None:
        self.current_jobs = [dict(job) for job in jobs if isinstance(job, dict)]
        try:
            job_select = self.query_one("#server-ingest-job-select", Select)
            cancel_button = self.query_one("#cancel-job-btn", Button)
        except Exception:
            return

        if not self.current_jobs:
            job_select.set_options([("No jobs", Select.BLANK)])
            job_select.value = Select.BLANK
            job_select.disabled = True
            cancel_button.disabled = True
            return

        options = []
        for job in self.current_jobs:
            source_id = self._job_source_id(job)
            status = str(job.get("status") or "unknown")
            source = str(job.get("source") or "").strip()
            if len(source) > 42:
                source = f"{source[:39]}..."
            label = f"{source_id}: {status}"
            if source:
                label = f"{label} {source}"
            options.append((label, source_id))
        job_select.set_options(options)
        valid_values = {value for _, value in options}
        if job_select.value not in valid_values:
            job_select.value = options[0][1]
        job_select.disabled = False
        cancel_button.disabled = False

    def _apply_job_stream_event(self, event: Dict[str, Any]) -> None:
        event_name = str(event.get("event") or "")
        if event_name == "snapshot":
            self._render_job_list_response(
                {
                    "batch_id": event.get("batch_id") or self.last_batch_id,
                    "jobs": list(event.get("jobs") or []),
                }
            )
            return

        if event_name != "job":
            return

        attrs = dict(event.get("attrs") or {})
        raw_job_id = event.get("job_id")
        canonical_id = event.get("id")
        candidates = {str(value) for value in (raw_job_id, canonical_id) if value not in (None, "")}
        if isinstance(canonical_id, str) and ":" in canonical_id:
            candidates.add(canonical_id.rsplit(":", 1)[-1])

        updated_jobs: list[Dict[str, Any]] = []
        matched = False
        for job in self.current_jobs:
            job_data = dict(job)
            job_candidates = {
                str(value)
                for value in (
                    self._job_source_id(job_data),
                    job_data.get("id"),
                    job_data.get("job_id"),
                    job_data.get("source_id"),
                )
                if value not in (None, "")
            }
            if candidates and candidates.intersection(job_candidates):
                job_data.update(attrs)
                if raw_job_id not in (None, ""):
                    job_data.setdefault("job_id", raw_job_id)
                    job_data.setdefault("source_id", str(raw_job_id))
                if canonical_id not in (None, ""):
                    job_data.setdefault("id", canonical_id)
                matched = True
            updated_jobs.append(job_data)

        if not matched:
            job_data = dict(attrs)
            if raw_job_id not in (None, ""):
                job_data["job_id"] = raw_job_id
                job_data["source_id"] = str(raw_job_id)
            if canonical_id not in (None, ""):
                job_data["id"] = canonical_id
            updated_jobs.append(job_data)

        self._render_job_list_response(
            {
                "batch_id": self.last_batch_id,
                "jobs": updated_jobs,
            }
        )

    @staticmethod
    def _format_job_status_line(job: Dict[str, Any]) -> str:
        job_label = job.get("id") or job.get("job_id") or "unknown"
        status = job.get("status", "unknown")
        source = str(job.get("source") or "").strip()
        details = []
        progress_percent = job.get("progress_percent")
        if progress_percent not in (None, ""):
            try:
                numeric_percent = float(progress_percent)
                percent_text = (
                    f"{int(numeric_percent)}%"
                    if numeric_percent.is_integer()
                    else f"{numeric_percent:.1f}%"
                )
            except (TypeError, ValueError):
                percent_text = f"{progress_percent}%"
            details.append(percent_text)
        progress_message = str(job.get("progress_message") or "").strip()
        if progress_message:
            details.append(progress_message)

        line = f"- {job_label}: {status}"
        if details:
            line = f"{line} ({' - '.join(details)})"
        if source:
            line = f"{line} {source}"
        return line.rstrip()

    def _render_submission_response(self, response: Dict[str, Any]) -> None:
        batch_id = response.get("batch_id")
        if batch_id:
            self.last_batch_id = str(batch_id)
        lines = [f"Batch: {batch_id or 'unknown'}"]
        jobs = [dict(job) for job in list(response.get("jobs") or []) if isinstance(job, dict)]
        self._update_job_selection(jobs)
        for job in jobs:
            lines.append(self._format_job_status_line(job))
        errors = list(response.get("errors") or [])
        if errors:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in errors)
        self.query_one("#remote-job-status", Static).update("\n".join(lines))

    def _render_job_list_response(self, response: Dict[str, Any]) -> None:
        batch_id = response.get("batch_id") or self.last_batch_id
        if batch_id:
            self.last_batch_id = str(batch_id)
        lines = [f"Batch: {batch_id or 'unknown'}"]
        jobs = [dict(job) for job in list(response.get("jobs") or []) if isinstance(job, dict)]
        self._update_job_selection(jobs)
        for job in jobs:
            lines.append(self._format_job_status_line(job))
        if len(lines) == 1:
            lines.append("- No jobs found")
        self.query_one("#remote-job-status", Static).update("\n".join(lines))

    def _render_batch_cancel_response(self, response: Dict[str, Any]) -> None:
        batch_id = response.get("batch_id") or self.last_batch_id
        if batch_id:
            self.last_batch_id = str(batch_id)
        lines = [
            f"Batch: {batch_id or 'unknown'}",
            f"Cancellation requested: {bool(response.get('success', False))}",
            f"Requested: {response.get('requested', 0)}",
            f"Cancelled: {response.get('cancelled', 0)}",
            f"Already terminal: {response.get('already_terminal', 0)}",
            f"Failed: {response.get('failed', 0)}",
        ]
        if response.get("message"):
            lines.append(str(response["message"]))
        self.query_one("#remote-job-status", Static).update("\n".join(lines))

    def _render_job_cancel_response(self, response: Dict[str, Any], *, fallback_job_id: str) -> None:
        job_id = response.get("job_id") or fallback_job_id
        lines = [
            f"Job {job_id} cancellation requested",
            f"Success: {bool(response.get('success', False))}",
        ]
        if response.get("status"):
            lines.append(f"Status: {response['status']}")
        if response.get("message"):
            lines.append(str(response["message"]))
        self.query_one("#remote-job-status", Static).update("\n".join(lines))

    def _render_web_content_response(self, response: Dict[str, Any]) -> None:
        lines = [
            f"Web content ingest: {response.get('status', 'unknown')}",
            str(response.get("message") or "No message"),
            f"Count: {response.get('count', 0)}",
        ]
        for item in list(response.get("results") or []):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("url") or "Untitled"
            status = "ok" if item.get("extraction_successful", True) else "failed"
            lines.append(f"- {title}: {status}")
        media_ids = list(response.get("media_ids") or [])
        if media_ids:
            lines.append(f"Media IDs: {', '.join(str(media_id) for media_id in media_ids)}")
        self.query_one("#remote-job-status", Static).update("\n".join(lines))


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
    from local files and for managing server-backed ingestion sources.
    """
    
    DEFAULT_CSS = """
    MediaIngestWindowRebuilt {
        layout: vertical;
        height: 100%;
        width: 100%;
        background: $background; /* Ensure background is rendered */
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
        self.runtime_state = getattr(app_instance, "media_runtime_state", None)
        logger.info("MediaIngestWindowRebuilt initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the main ingestion interface."""
        with TabbedContent(initial="local-tab"):
            with TabPane("Local Files", id="local-tab"):
                self.local_panel = LocalIngestionPanel(self.app_instance, id="local-panel")
                yield self.local_panel
            
            with TabPane("Server Sources", id="sources-tab"):
                self.source_panel = MediaIngestionSourcePanel(self.app_instance, id="source-panel")
                yield self.source_panel

            with TabPane("Server Jobs", id="remote-tab"):
                self.remote_panel = RemoteIngestionPanel(self.app_instance, id="remote-panel")
                yield self.remote_panel
        
        yield IngestionResultsPanel(id="results-panel")
    
    @on(TabbedContent.TabActivated)
    def handle_tab_change(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab switching."""
        if event.tab.id == "local-tab":
            self.current_tab = "local"
        elif event.tab.id == "remote-tab":
            self.current_tab = "remote"
        else:
            self.current_tab = "sources"
        logger.debug(f"Switched to {self.current_tab} tab")
        self.run_worker(self.refresh_backend_view(), exclusive=True)
    
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
        self.run_worker(self.refresh_backend_view(), exclusive=True)
        self.notify("Media Ingestion ready", severity="information")

    async def refresh_backend_view(self) -> None:
        """Refresh the backend-aware server source panel."""
        if self.runtime_state is None:
            self.runtime_state = getattr(self.app_instance, "media_runtime_state", None)

        runtime_backend = "local"
        if self.runtime_state is not None:
            runtime_backend = str(getattr(self.runtime_state, "runtime_backend", "local") or "local")

        for panel_name in ("source_panel", "remote_panel"):
            panel = getattr(self, panel_name, None)
            if panel is None:
                continue
            panel.runtime_backend = runtime_backend
            await panel.refresh_for_mode()
