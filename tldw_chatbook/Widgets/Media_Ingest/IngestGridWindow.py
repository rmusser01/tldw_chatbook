# tldw_chatbook/Widgets/Media_Ingest/IngestGridWindow.py
# Grid-based compact layout for media ingestion with improved space efficiency

from typing import TYPE_CHECKING, List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Grid, Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, 
    Label, ProgressBar, LoadingIndicator
)
from textual.reactive import reactive
from tldw_chatbook.config import get_media_ingestion_defaults
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from tldw_chatbook.Widgets.file_list_item_enhanced import FileListEnhanced
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

class IngestGridWindow(Container):
    """Space-efficient grid-based media ingestion interface."""
    
    DEFAULT_CSS = """
    IngestGridWindow {
        height: 100%;
        width: 100%;
    }
    
    /* Main container */
    .grid-ingest-container {
        height: 100%;
        width: 100%;
        padding: 1;
    }
    
    /* Status bar docked at top */
    .grid-status-bar {
        dock: top;
        height: 3;
        background: $surface;
        border: round $accent;
        padding: 0 2;
        margin-bottom: 1;
    }
    
    .grid-status-bar.hidden {
        display: none;
    }
    
    .status-progress {
        width: 1fr;
        height: 1;
    }
    
    .status-text-inline {
        width: 1fr;
        text-align: center;
        margin-top: 1;
    }
    
    /* Main grid - 3 columns */
    .ingest-grid-main {
        grid-size: 3 1;
        grid-columns: 1fr 1fr 1fr;
        grid-gutter: 1;
        height: 1fr;
        width: 100%;
    }
    
    /* Grid cells */
    .grid-cell {
        border: round $surface;
        padding: 1;
        background: $surface-lighten-1;
        height: 100%;
    }
    
    /* Section headers */
    .section-icon-header {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        height: 2;
    }
    
    /* Input row */
    .input-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .flex-input {
        width: 1fr;
    }
    
    .icon-button {
        width: 3;
        min-width: 3;
        margin-left: 1;
    }
    
    /* Compact textarea */
    .compact-textarea {
        height: 5;
        min-height: 5;
        max-height: 8;
        margin-bottom: 1;
    }
    
    /* Settings subgrid */
    .settings-subgrid {
        grid-size: 3 2;
        grid-columns: 5 1fr;
        grid-rows: auto auto auto;
        grid-gutter: 1;
        margin-bottom: 1;
    }
    
    .inline-label {
        width: 5;
        text-align: right;
        align: right middle;
    }
    
    /* Checkbox grid */
    .checkbox-grid {
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 1;
        margin-top: 1;
    }
    
    /* Time inputs */
    .time-range-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .time-input {
        width: 8;
    }
    
    .time-arrow {
        width: 2;
        text-align: center;
        margin: 0 1;
    }
    
    /* Chunking row */
    .chunk-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .mini-input {
        width: 6;
        margin: 0 1;
    }
    
    .separator {
        width: 1;
        text-align: center;
    }
    
    /* Action container */
    .action-container {
        margin-top: 2;
    }
    
    .primary-action {
        width: 100%;
        height: 3;
        text-style: bold;
    }
    
    .settings-toggle {
        width: 3;
        height: 3;
        margin-top: 1;
    }
    
    /* Advanced panel */
    .advanced-panel {
        dock: bottom;
        height: auto;
        max-height: 10;
        background: $surface-darken-1;
        border: thick $primary;
        padding: 1;
        margin-top: 1;
    }
    
    .advanced-panel.hidden {
        display: none;
    }
    
    .advanced-grid {
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 1;
    }
    
    /* Subtle info text */
    .subtle-info {
        color: $text-muted;
        margin-top: 1;
    }
    """
    
    processing = reactive(False)
    show_advanced = reactive(False)
    selected_files_count = reactive(0)
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "video", **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_type = media_type
        self.selected_local_files = []
        self.transcription_service = TranscriptionService()
        self._current_model_list = []
        
        logger.debug(f"[Grid] IngestGridWindow initialized for {media_type}")
    
    def compose(self) -> ComposeResult:
        """Compose the grid-based ingestion form."""
        media_defaults = get_media_ingestion_defaults(self.media_type)
        
        with Container(classes="grid-ingest-container"):
            # Status bar (hidden initially, shows during processing)
            with Container(id="grid-status-bar", classes="grid-status-bar hidden"):
                yield ProgressBar(id="progress", classes="status-progress")
                yield Static("Ready", id="status-text", classes="status-text-inline")
            
            # Main grid layout - 3 columns
            with Grid(classes="ingest-grid-main"):
                # Column 1: Input Sources
                with Container(classes="grid-cell input-sources"):
                    yield Static("📁 Input", classes="section-icon-header")
                    
                    # Compact file picker with inline browse
                    with Horizontal(classes="input-row"):
                        yield Input(
                            placeholder="Drop files or browse →",
                            id="file-input",
                            classes="flex-input",
                            disabled=True  # Visual only, use button
                        )
                        yield Button("📂", id="browse", classes="icon-button")
                    
                    # URL input with smart detection
                    yield Label(f"Paste {self.media_type} URLs (one per line):")
                    yield TextArea(
                        "",
                        id="url-input",
                        classes="compact-textarea"
                    )
                    
                    # Selected files display
                    yield FileListEnhanced(
                        id="selected-files",
                        show_summary=True,
                        max_height=8
                    )
                    
                    # Active files counter
                    yield Static("No files selected", id="file-count", classes="subtle-info")
                
                # Column 2: Quick Settings
                with Container(classes="grid-cell quick-settings"):
                    yield Static("⚡ Quick Setup", classes="section-icon-header")
                    
                    # Inline labeled inputs using Grid
                    with Grid(classes="settings-subgrid"):
                        yield Static("Title:", classes="inline-label")
                        yield Input(id="title", placeholder="Auto-detect")
                        
                        yield Static("Lang:", classes="inline-label")
                        yield Select(
                            [("Auto", "auto"), ("English", "en"), ("Spanish", "es"), 
                             ("French", "fr"), ("German", "de"), ("Chinese", "zh")],
                            id="language",
                            value="auto"
                        )
                        
                        yield Static("Model:", classes="inline-label")
                        yield Select(
                            [("Fast", "base"), ("Accurate", "large"), ("Best", "large-v3")],
                            id="model",
                            value="base"
                        )
                    
                    # Compact checkboxes in grid
                    with Grid(classes="checkbox-grid"):
                        if self.media_type in ["video", "audio"]:
                            yield Checkbox("Extract audio", True, id="audio-only")
                            yield Checkbox("Timestamps", True, id="timestamps")
                            yield Checkbox("Summary", True, id="summary")
                            yield Checkbox("Diarize", False, id="diarize")
                        else:
                            yield Checkbox("Summary", True, id="summary")
                            yield Checkbox("Keywords", True, id="keywords")
                            yield Checkbox("Chunking", True, id="chunking")
                            yield Checkbox("OCR", False, id="ocr")
                
                # Column 3: Processing Options & Actions
                with Container(classes="grid-cell processing-section"):
                    yield Static("🚀 Process", classes="section-icon-header")
                    
                    # Time range for video/audio (hidden by default)
                    with Horizontal(classes="time-range-row hidden", id="time-range"):
                        yield Input(placeholder="00:00:00", id="start-time", classes="time-input")
                        yield Static("→", classes="time-arrow")
                        yield Input(placeholder="00:00:00", id="end-time", classes="time-input")
                    
                    # Chunking settings in one line
                    with Horizontal(classes="chunk-row"):
                        yield Checkbox("Chunk:", value=True, id="chunk-enable")
                        yield Input("500", id="chunk-size", classes="mini-input")
                        yield Static("/", classes="separator")
                        yield Input("200", id="chunk-overlap", classes="mini-input")
                    
                    # Keywords input
                    yield Input(
                        placeholder="Keywords (comma-separated)",
                        id="keywords-input"
                    )
                    
                    # Action buttons
                    with Container(classes="action-container"):
                        yield Button(
                            f"Process {self.media_type.title()}",
                            id="process",
                            variant="success",
                            classes="primary-action"
                        )
                        yield Button(
                            "Cancel",
                            id="cancel",
                            variant="error",
                            classes="primary-action hidden"
                        )
                        
                        # Advanced options toggle
                        yield Button("⚙", id="advanced-toggle", classes="settings-toggle")
            
            # Advanced panel (hidden by default, slides in from bottom)
            with Container(id="advanced-panel", classes="advanced-panel hidden"):
                with Grid(classes="advanced-grid"):
                    # Advanced options in compact grid
                    yield Input(placeholder="Custom prompt", id="custom-prompt")
                    
                    # API provider selection
                    api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
                    api_options = [(name, name) for name in api_providers if name]
                    if not api_options:
                        api_options = [("No Providers", Select.BLANK)]
                    yield Select(
                        api_options,
                        id="api-provider",
                        prompt="Analysis API..."
                    )
                    
                    if self.media_type in ["video", "audio"]:
                        yield Checkbox("VAD Filter", id="vad")
                        yield Checkbox("Download full", id="download-full")
                    else:
                        yield Checkbox("Adaptive chunking", id="adaptive-chunk")
                        yield Checkbox("Multi-level chunks", id="multi-level")
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        # Show time range for video/audio
        if self.media_type in ["video", "audio"]:
            try:
                time_range = self.query_one("#time-range", Horizontal)
                time_range.remove_class("hidden")
            except:
                pass
        
        # Initialize transcription models if needed
        if self.media_type in ["video", "audio"]:
            self.run_worker(self._initialize_models, exclusive=True, thread=True)
    
    def _initialize_models(self) -> None:
        """Initialize transcription models in background."""
        try:
            providers = self.transcription_service.get_available_providers()
            if providers:
                default_provider = providers[0]
                models = self.transcription_service.get_models_for_provider(default_provider)
                self._current_model_list = models
                
                # Update model select on main thread
                self.call_from_thread(self._update_model_select, models)
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _update_model_select(self, models: List[str]) -> None:
        """Update model select widget with available models."""
        try:
            model_select = self.query_one("#model", Select)
            if models:
                model_options = [(m, m) for m in models[:5]]  # Limit to 5 for space
                model_select.set_options(model_options)
                if models:
                    model_select.value = models[0]
        except Exception as e:
            logger.error(f"Error updating model select: {e}")
    
    def watch_selected_files_count(self, count: int) -> None:
        """Update file counter when files change."""
        try:
            counter = self.query_one("#file-count", Static)
            if count == 0:
                counter.update("No files selected")
            elif count == 1:
                counter.update("1 file selected")
            else:
                counter.update(f"{count} files selected")
        except:
            pass
    
    def watch_processing(self, is_processing: bool) -> None:
        """Toggle UI state when processing."""
        try:
            status_bar = self.query_one("#grid-status-bar", Container)
            process_btn = self.query_one("#process", Button)
            cancel_btn = self.query_one("#cancel", Button)
            
            if is_processing:
                status_bar.remove_class("hidden")
                process_btn.add_class("hidden")
                cancel_btn.remove_class("hidden")
            else:
                status_bar.add_class("hidden")
                process_btn.remove_class("hidden")
                cancel_btn.add_class("hidden")
        except:
            pass
    
    @on(Button.Pressed, "#browse")
    async def handle_browse_files(self, event: Button.Pressed) -> None:
        """Handle file browser button."""
        # Define filters based on media type
        if self.media_type == "video":
            filters = Filters(
                ("Video Files", lambda p: p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm")),
                ("All Files", lambda _: True)
            )
        elif self.media_type == "audio":
            filters = Filters(
                ("Audio Files", lambda p: p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma")),
                ("All Files", lambda _: True)
            )
        elif self.media_type == "pdf":
            filters = Filters(
                ("PDF Files", lambda p: p.suffix.lower() == ".pdf"),
                ("All Files", lambda _: True)
            )
        else:
            filters = Filters(("All Files", lambda _: True))
        
        await self.app.push_screen(
            FileOpen(
                title=f"Select {self.media_type.title()} Files",
                filters=filters
            ),
            callback=self.handle_file_selection
        )
    
    async def handle_file_selection(self, path: Path | None) -> None:
        """Handle file selection from dialog."""
        if path:
            file_list = self.query_one("#selected-files", FileListEnhanced)
            file_list.add_file(path)
            self.selected_local_files.append(path)
            self.selected_files_count = len(self.selected_local_files)
            
            # Update app instance selected files
            if not hasattr(self.app_instance, 'selected_local_files'):
                self.app_instance.selected_local_files = {}
            
            media_key = f'local_{self.media_type}'
            if media_key not in self.app_instance.selected_local_files:
                self.app_instance.selected_local_files[media_key] = []
            
            if path not in self.app_instance.selected_local_files[media_key]:
                self.app_instance.selected_local_files[media_key].append(path)
    
    @on(Button.Pressed, "#advanced-toggle")
    def toggle_advanced(self, event: Button.Pressed) -> None:
        """Toggle advanced options panel."""
        panel = self.query_one("#advanced-panel", Container)
        if panel.has_class("hidden"):
            panel.remove_class("hidden")
            event.button.label = "⚙✓"
        else:
            panel.add_class("hidden")
            event.button.label = "⚙"
    
    @on(Button.Pressed, "#process")
    async def handle_process(self, event: Button.Pressed) -> None:
        """Handle process button."""
        # Validate inputs
        if not self.selected_local_files:
            urls_text = self.query_one("#url-input", TextArea).text
            if not urls_text.strip():
                self.app_instance.notify("Please select files or enter URLs", severity="warning")
                return
        
        # Update UI state
        self.processing = True
        
        # Import the actual processing handler based on media type
        if self.media_type == "video":
            from tldw_chatbook.Event_Handlers.ingest_events import handle_local_video_process
            await handle_local_video_process(self.app_instance)
        elif self.media_type == "audio":
            from tldw_chatbook.Event_Handlers.ingest_events import handle_local_audio_process
            await handle_local_audio_process(self.app_instance)
        elif self.media_type == "pdf":
            from tldw_chatbook.Event_Handlers.ingest_events import handle_local_pdf_process
            await handle_local_pdf_process(self.app_instance)
        else:
            # Generic document processing
            from tldw_chatbook.Event_Handlers.ingest_events import handle_local_document_process
            await handle_local_document_process(self.app_instance)
        
        # Reset UI state
        self.processing = False
    
    @on(Button.Pressed, "#cancel")
    async def handle_cancel(self, event: Button.Pressed) -> None:
        """Handle cancel button."""
        # TODO: Implement cancellation logic
        self.processing = False
        self.app_instance.notify("Processing cancelled", severity="warning")
    
    def get_form_data(self) -> Dict[str, Any]:
        """Collect all form data for processing."""
        data = {
            "files": self.selected_local_files,
            "urls": self.query_one("#url-input", TextArea).text.strip().split("\n"),
            "title": self.query_one("#title", Input).value,
            "language": self.query_one("#language", Select).value,
            "model": self.query_one("#model", Select).value,
            "chunk_enable": self.query_one("#chunk-enable", Checkbox).value,
            "chunk_size": int(self.query_one("#chunk-size", Input).value or 500),
            "chunk_overlap": int(self.query_one("#chunk-overlap", Input).value or 200),
            "keywords": self.query_one("#keywords-input", Input).value,
        }
        
        # Add media-specific options
        if self.media_type in ["video", "audio"]:
            data.update({
                "audio_only": self.query_one("#audio-only", Checkbox).value if self.query("#audio-only") else False,
                "timestamps": self.query_one("#timestamps", Checkbox).value if self.query("#timestamps") else False,
                "summary": self.query_one("#summary", Checkbox).value if self.query("#summary") else False,
                "diarize": self.query_one("#diarize", Checkbox).value if self.query("#diarize") else False,
                "start_time": self.query_one("#start-time", Input).value if self.query("#start-time") else "",
                "end_time": self.query_one("#end-time", Input).value if self.query("#end-time") else "",
            })
        
        # Add advanced options if panel is open
        if not self.query_one("#advanced-panel").has_class("hidden"):
            data.update({
                "custom_prompt": self.query_one("#custom-prompt", Input).value,
                "api_provider": self.query_one("#api-provider", Select).value,
            })
            
            if self.media_type in ["video", "audio"]:
                data.update({
                    "vad_filter": self.query_one("#vad", Checkbox).value if self.query("#vad") else False,
                    "download_full": self.query_one("#download-full", Checkbox).value if self.query("#download-full") else False,
                })
            else:
                data.update({
                    "adaptive_chunk": self.query_one("#adaptive-chunk", Checkbox).value if self.query("#adaptive-chunk") else False,
                    "multi_level": self.query_one("#multi-level", Checkbox).value if self.query("#multi-level") else False,
                })
        
        return data

# End of IngestGridWindow.py