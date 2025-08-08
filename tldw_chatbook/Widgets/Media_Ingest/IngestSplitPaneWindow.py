# tldw_chatbook/Widgets/Media_Ingest/IngestSplitPaneWindow.py
# Split-pane interface with live preview for media ingestion

from typing import TYPE_CHECKING, List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label,
    TabbedContent, TabPane, RichLog, ListView, ListItem, Markdown
)
from textual.reactive import reactive
from tldw_chatbook.config import get_media_ingestion_defaults
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="IngestSplitPaneWindow")


class IngestSplitPaneWindow(Container):
    """Split-pane interface with input on left and live preview on right."""
    
    DEFAULT_CSS = """
    IngestSplitPaneWindow {
        height: 100%;
        width: 100%;
    }
    
    /* Split-pane container */
    .split-pane-container {
        height: 100%;
        width: 100%;
    }
    
    /* Left pane - 40% width */
    .left-pane {
        width: 40%;
        min-width: 30;
        border-right: solid $primary;
        padding: 1;
        height: 100%;
    }
    
    /* Right pane - 60% width */
    .right-pane {
        width: 60%;
        padding: 1;
        height: 100%;
    }
    
    /* Pane headers */
    .pane-header {
        dock: top;
        height: 3;
        border-bottom: solid $surface;
        margin-bottom: 1;
        align: center middle;
    }
    
    .pane-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }
    
    .counter-badge {
        background: $accent;
        color: $background;
        padding: 0 1;
        text-align: center;
        min-width: 5;
    }
    
    /* Smart input field */
    .smart-input {
        width: 100%;
        height: 3;
        margin-bottom: 1;
        border: solid $surface;
    }
    
    .smart-input:focus {
        border: solid $primary;
    }
    
    /* Button row */
    .button-row {
        height: 3;
        margin-bottom: 2;
    }
    
    .button-row Button {
        width: 1fr;
        margin-right: 1;
    }
    
    .button-row Button:last-child {
        margin-right: 0;
    }
    
    /* Option grid */
    .option-grid {
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 1;
        padding: 1;
    }
    
    /* Setting groups */
    .setting-group {
        margin-bottom: 2;
        padding: 1;
        border: round $surface;
    }
    
    .group-title {
        text-style: bold;
        color: $secondary;
        margin-bottom: 1;
    }
    
    .setting-row {
        height: 3;
        align: left middle;
        margin-bottom: 1;
    }
    
    .setting-label {
        width: 10;
        text-align: right;
        margin-right: 1;
    }
    
    .setting-input {
        width: 1fr;
    }
    
    .setting-input-sm {
        width: 8;
    }
    
    /* Action bar */
    .action-bar {
        dock: bottom;
        height: 4;
        border-top: solid $primary;
        padding-top: 1;
        align: center middle;
    }
    
    .process-button {
        width: 1fr;
        height: 3;
        text-style: bold;
    }
    
    .icon-btn {
        width: 3;
        margin-left: 1;
    }
    
    /* Preview header */
    .preview-header {
        dock: top;
        height: 3;
        border-bottom: solid $surface;
        margin-bottom: 1;
    }
    
    .preview-tab {
        width: 1fr;
        height: 3;
        background: transparent;
        border: none;
        color: $text-muted;
    }
    
    .preview-tab.active {
        background: $surface;
        color: $text;
        text-style: bold;
        border-bottom: thick $accent;
    }
    
    /* Preview content */
    .preview-content {
        height: 1fr;
        overflow-y: auto;
    }
    
    .preview-panel {
        width: 100%;
        height: 100%;
    }
    
    .preview-panel.hidden {
        display: none;
    }
    
    /* Metadata list */
    .metadata-list {
        width: 100%;
        border: round $surface;
        padding: 1;
        background: $surface;
    }
    
    .metadata-item {
        padding: 1;
        border-bottom: solid $surface-lighten-1;
    }
    
    .metadata-item:last-child {
        border-bottom: none;
    }
    
    .metadata-key {
        text-style: bold;
        color: $primary;
    }
    
    .metadata-value {
        color: $text;
        margin-left: 2;
    }
    
    /* Transcript viewer */
    .transcript-viewer {
        padding: 2;
        background: $surface;
        border: round $primary;
        height: 100%;
        overflow-y: auto;
    }
    
    /* Status log */
    .status-log {
        height: 100%;
        background: $surface-darken-1;
        border: round $primary;
        padding: 1;
    }
    
    /* Tab content areas */
    .tab-scroll {
        height: 100%;
        padding: 1;
    }
    
    /* Batch table */
    .batch-list {
        height: 100%;
        border: round $surface;
        background: $surface;
    }
    
    /* File counter update */
    .file-count-display {
        text-align: center;
        margin: 1;
        color: $text-muted;
    }
    """
    
    preview_mode = reactive("metadata")  # metadata, transcript, status
    selected_files_count = reactive(0)
    processing = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "video", **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_type = media_type
        self.selected_files = []
        self.selected_urls = []
        self.transcription_service = TranscriptionService()
        self.metadata_cache = {}
        
        logger.debug(f"[SplitPane] IngestSplitPaneWindow initialized for {media_type}")
    
    def compose(self) -> ComposeResult:
        """Compose the split-pane UI."""
        media_defaults = get_media_ingestion_defaults(self.media_type)
        
        with Horizontal(classes="split-pane-container"):
            # Left Pane: Input and Configuration
            with Container(classes="left-pane"):
                # Compact header with file counter
                with Horizontal(classes="pane-header"):
                    yield Static("Media Input", classes="pane-title")
                    yield Static("0 files", id="file-counter", classes="counter-badge")
                
                # Tabbed configuration (replaces mode toggle)
                with TabbedContent(id="config-tabs"):
                    with TabPane("Essential", id="essential-tab"):
                        with VerticalScroll(classes="tab-scroll"):
                            # Smart input field
                            yield Label("Files or URLs:")
                            yield Input(
                                placeholder="Paste URLs or drag files here",
                                id="smart-input",
                                classes="smart-input"
                            )
                            
                            # File browser buttons
                            with Horizontal(classes="button-row"):
                                yield Button("Browse", id="browse")
                                yield Button("YouTube", id="youtube")
                                yield Button("Clear", id="clear")
                            
                            # Essential options grid
                            with Container(classes="option-grid"):
                                if self.media_type in ["video", "audio"]:
                                    yield Checkbox("Audio only", True, id="audio")
                                    yield Checkbox("Summary", True, id="summary")
                                    yield Checkbox("Timestamps", True, id="stamps")
                                    yield Checkbox("Quick mode", True, id="quick")
                                else:
                                    yield Checkbox("Summary", True, id="summary")
                                    yield Checkbox("Keywords", True, id="keywords")
                                    yield Checkbox("Chunking", True, id="chunking")
                                    yield Checkbox("OCR", False, id="ocr")
                    
                    with TabPane("Advanced", id="advanced-tab"):
                        with VerticalScroll(classes="tab-scroll"):
                            # Transcription settings for media
                            if self.media_type in ["video", "audio"]:
                                with Container(classes="setting-group"):
                                    yield Static("Transcription", classes="group-title")
                                    with Horizontal(classes="setting-row"):
                                        yield Static("Provider:", classes="setting-label")
                                        providers = self.transcription_service.get_available_providers()
                                        provider_options = [(p, p) for p in providers]
                                        yield Select(
                                            provider_options,
                                            id="provider",
                                            classes="setting-input",
                                            value=providers[0] if providers else None
                                        )
                                    with Horizontal(classes="setting-row"):
                                        yield Static("Model:", classes="setting-label")
                                        yield Select(
                                            [],
                                            id="model",
                                            classes="setting-input"
                                        )
                            
                            # Processing settings
                            with Container(classes="setting-group"):
                                yield Static("Processing", classes="group-title")
                                with Horizontal(classes="setting-row"):
                                    yield Static("Chunk:", classes="setting-label")
                                    yield Input("500", id="chunk", classes="setting-input-sm")
                                    yield Static("/", classes="separator")
                                    yield Input("200", id="overlap", classes="setting-input-sm")
                    
                    with TabPane("Batch", id="batch-tab"):
                        with VerticalScroll(classes="tab-scroll"):
                            # Batch processing list
                            yield ListView(id="batch-list", classes="batch-list")
                            yield Static("Drop multiple files to batch process", 
                                       classes="file-count-display")
                
                # Action bar (always visible)
                with Horizontal(classes="action-bar"):
                    yield Button(
                        "▶ Process",
                        id="process",
                        variant="success",
                        classes="process-button"
                    )
                    yield Button("⏸", id="pause", classes="icon-btn hidden")
                    yield Button("⏹", id="stop", classes="icon-btn hidden")
            
            # Right Pane: Preview and Status
            with Container(classes="right-pane"):
                # Preview mode selector
                with Horizontal(classes="preview-header"):
                    yield Button("Metadata", id="preview-meta", classes="preview-tab active")
                    yield Button("Transcript", id="preview-trans", classes="preview-tab")
                    yield Button("Status", id="preview-status", classes="preview-tab")
                
                # Preview content area
                with Container(id="preview-content", classes="preview-content"):
                    # Metadata preview
                    with Container(id="metadata-preview", classes="preview-panel"):
                        with VerticalScroll():
                            yield Container(id="metadata-display", classes="metadata-list")
                    
                    # Transcript preview  
                    with Container(id="transcript-preview", classes="preview-panel hidden"):
                        yield Markdown(
                            "Transcript will appear here after processing...",
                            id="transcript-md",
                            classes="transcript-viewer"
                        )
                    
                    # Status/Log preview
                    with Container(id="status-preview", classes="preview-panel hidden"):
                        yield RichLog(
                            id="status-log",
                            classes="status-log",
                            highlight=True,
                            markup=True
                        )
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        # Initialize transcription models if needed
        if self.media_type in ["video", "audio"]:
            self.run_worker(self._initialize_models, exclusive=True, thread=True)
        
        # Set initial metadata display
        self.update_metadata_display({
            "Type": self.media_type.title(),
            "Status": "Ready",
            "Files": "0 selected",
            "Configuration": "Default"
        })
    
    def _initialize_models(self) -> None:
        """Initialize transcription models in background."""
        try:
            provider_select = self.query_one("#provider", Select)
            if provider_select and provider_select.value:
                models = self.transcription_service.get_models_for_provider(provider_select.value)
                self.call_from_thread(self._update_model_select, models)
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _update_model_select(self, models: List[str]) -> None:
        """Update model select widget."""
        try:
            model_select = self.query_one("#model", Select)
            if models:
                model_options = [(m, m) for m in models]
                model_select.set_options(model_options)
                if models:
                    model_select.value = models[0]
        except Exception as e:
            logger.error(f"Error updating model select: {e}")
    
    @on(Input.Changed, "#smart-input")
    def handle_smart_input(self, event: Input.Changed) -> None:
        """Handle smart input changes with auto-detection."""
        value = event.value.strip()
        if not value:
            return
        
        # Detect URLs
        if value.startswith(("http://", "https://", "www.")):
            self.add_url(value)
            event.input.value = ""  # Clear after adding
            self.update_preview_for_url(value)
        # Detect file paths
        elif "/" in value or "\\" in value:
            path = Path(value)
            if path.exists():
                self.add_file(path)
                event.input.value = ""  # Clear after adding
                self.update_preview_for_file(path)
    
    @on(Button.Pressed, "#browse")
    async def handle_browse(self, event: Button.Pressed) -> None:
        """Handle file browse button."""
        # Define filters based on media type
        if self.media_type == "video":
            filters = Filters(
                ("Video Files", lambda p: p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov")),
                ("All Files", lambda _: True)
            )
        elif self.media_type == "audio":
            filters = Filters(
                ("Audio Files", lambda p: p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")),
                ("All Files", lambda _: True)
            )
        else:
            filters = Filters(("All Files", lambda _: True))
        
        await self.app.push_screen(
            FileOpen(
                title=f"Select {self.media_type.title()} Files",
                filters=filters
            ),
            callback=lambda p: self.add_file(p) if p else None
        )
    
    @on(Button.Pressed, "#clear")
    def handle_clear(self, event: Button.Pressed) -> None:
        """Clear all selections."""
        self.selected_files.clear()
        self.selected_urls.clear()
        self.selected_files_count = 0
        
        # Clear batch list
        batch_list = self.query_one("#batch-list", ListView)
        batch_list.clear()
        
        # Update displays
        self.update_file_counter()
        self.update_metadata_display({
            "Type": self.media_type.title(),
            "Status": "Ready",
            "Files": "0 selected",
            "Configuration": "Default"
        })
    
    def add_file(self, path: Path) -> None:
        """Add a file to the selection."""
        if path not in self.selected_files:
            self.selected_files.append(path)
            self.selected_files_count = len(self.selected_files)
            
            # Add to batch list
            batch_list = self.query_one("#batch-list", ListView)
            batch_list.append(ListItem(Static(f"📁 {path.name}")))
            
            # Update counter
            self.update_file_counter()
            
            # Update preview
            self.update_preview_for_file(path)
    
    def add_url(self, url: str) -> None:
        """Add a URL to the selection."""
        if url not in self.selected_urls:
            self.selected_urls.append(url)
            
            # Add to batch list
            batch_list = self.query_one("#batch-list", ListView)
            batch_list.append(ListItem(Static(f"🔗 {url[:50]}...")))
            
            # Update counter
            self.update_file_counter()
            
            # Update preview
            self.update_preview_for_url(url)
    
    def update_file_counter(self) -> None:
        """Update the file counter badge."""
        total = len(self.selected_files) + len(self.selected_urls)
        counter = self.query_one("#file-counter", Static)
        if total == 0:
            counter.update("0 files")
        elif total == 1:
            counter.update("1 file")
        else:
            counter.update(f"{total} files")
    
    def update_preview_for_file(self, path: Path) -> None:
        """Update preview for a selected file."""
        metadata = {
            "Name": path.name,
            "Path": str(path.parent),
            "Size": f"{path.stat().st_size / 1024:.1f} KB",
            "Type": path.suffix[1:].upper() if path.suffix else "Unknown",
            "Modified": Path(path).stat().st_mtime
        }
        self.update_metadata_display(metadata)
    
    def update_preview_for_url(self, url: str) -> None:
        """Update preview for a URL."""
        metadata = {
            "URL": url,
            "Type": "Web Resource",
            "Status": "Ready to download",
            "Protocol": url.split("://")[0].upper() if "://" in url else "UNKNOWN"
        }
        self.update_metadata_display(metadata)
    
    def update_metadata_display(self, metadata: Dict[str, Any]) -> None:
        """Update the metadata display panel."""
        container = self.query_one("#metadata-display", Container)
        container.remove_children()
        
        for key, value in metadata.items():
            item = Container(classes="metadata-item")
            item.mount(Static(f"{key}:", classes="metadata-key"))
            item.mount(Static(str(value), classes="metadata-value"))
            container.mount(item)
    
    @on(Button.Pressed, ".preview-tab")
    def handle_preview_tab(self, event: Button.Pressed) -> None:
        """Handle preview tab switching."""
        # Remove active from all tabs
        for tab in self.query(".preview-tab"):
            tab.remove_class("active")
        
        # Add active to clicked tab
        event.button.add_class("active")
        
        # Hide all panels
        for panel in self.query(".preview-panel"):
            panel.add_class("hidden")
        
        # Show selected panel
        if event.button.id == "preview-meta":
            self.query_one("#metadata-preview").remove_class("hidden")
            self.preview_mode = "metadata"
        elif event.button.id == "preview-trans":
            self.query_one("#transcript-preview").remove_class("hidden")
            self.preview_mode = "transcript"
        elif event.button.id == "preview-status":
            self.query_one("#status-preview").remove_class("hidden")
            self.preview_mode = "status"
    
    @on(Button.Pressed, "#process")
    async def handle_process(self, event: Button.Pressed) -> None:
        """Handle process button."""
        if not self.selected_files and not self.selected_urls:
            self.app_instance.notify("Please select files or enter URLs", severity="warning")
            return
        
        # Update UI state
        self.processing = True
        event.button.add_class("hidden")
        self.query_one("#pause").remove_class("hidden")
        self.query_one("#stop").remove_class("hidden")
        
        # Switch to status view
        self.query_one("#preview-status").press()
        
        # Log start
        status_log = self.query_one("#status-log", RichLog)
        status_log.write("[bold green]Starting processing...[/]")
        
        # Start processing (would connect to actual processing)
        self.simulate_processing()
    
    @work(thread=True)
    def simulate_processing(self) -> None:
        """Simulate processing (replace with actual processing)."""
        import time
        
        status_log = self.query_one("#status-log", RichLog)
        
        for i, file in enumerate(self.selected_files + self.selected_urls):
            name = file.name if isinstance(file, Path) else file[:30]
            self.call_from_thread(status_log.write, f"Processing: {name}")
            time.sleep(1)  # Simulate work
            self.call_from_thread(status_log.write, f"[green]✓[/] Completed: {name}")
        
        self.call_from_thread(self.processing_complete)
    
    def processing_complete(self) -> None:
        """Mark processing as complete."""
        self.processing = False
        self.query_one("#process").remove_class("hidden")
        self.query_one("#pause").add_class("hidden")
        self.query_one("#stop").add_class("hidden")
        
        status_log = self.query_one("#status-log", RichLog)
        status_log.write("[bold green]✓ All processing complete![/]")
        
        self.app_instance.notify("Processing complete!", severity="information")

# End of IngestSplitPaneWindow.py