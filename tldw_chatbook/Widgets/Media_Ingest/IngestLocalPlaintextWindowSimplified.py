# tldw_chatbook/Widgets/IngestLocalPlaintextWindowSimplified.py
# Simplified version of plaintext ingestion with progressive disclosure

from typing import TYPE_CHECKING, List
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible, RadioSet, RadioButton
)
from textual import on
from textual.reactive import reactive
from tldw_chatbook.config import get_media_ingestion_defaults
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from tldw_chatbook.Widgets.file_list_item_enhanced import FileListEnhanced
from tldw_chatbook.Widgets.status_dashboard import StatusDashboard

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

class IngestLocalPlaintextWindowSimplified(Vertical):
    """Simplified window for ingesting plaintext files with progressive disclosure."""
    
    # Reactive property for simple/advanced mode
    simple_mode = reactive(True)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        
        # Load saved preference
        from ..Utils.ingestion_preferences import get_ingestion_mode_preference
        self.simple_mode = get_ingestion_mode_preference("plaintext")
        
        logger.debug("[Plaintext] IngestLocalPlaintextWindowSimplified initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the simplified plaintext ingestion form."""
        # Get plaintext-specific default settings from config
        plaintext_defaults = get_media_ingestion_defaults("plaintext")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            # Status dashboard at top
            yield StatusDashboard(id="plaintext-status-dashboard")
            
            # Mode toggle
            with Container(classes="mode-toggle-container"):
                yield Static("Text File Processing", classes="sidebar-title")
                with RadioSet(id="plaintext-mode-toggle", classes="mode-toggle"):
                    yield RadioButton("Simple Mode", value=True, id="plaintext-simple-radio")
                    yield RadioButton("Advanced Mode", id="plaintext-advanced-radio")
            
            # Essential fields container (always visible)
            with Container(classes="essential-fields"):
                yield Label("Select Text Files", classes="form-label-primary")
                
                # File selection
                with Horizontal(classes="file-selection-row"):
                    yield Button("Browse Files", id="ingest-local-plaintext-select-files", variant="primary")
                    yield Button("Clear All", id="ingest-local-plaintext-clear-files", variant="default")
                
                # Selected files display with metadata
                yield Label("Selected Files:", classes="form-label")
                yield FileListEnhanced(
                    id="ingest-local-plaintext-files-list",
                    show_summary=True,
                    max_height=10
                )
                
                # Basic metadata
                with Horizontal(classes="metadata-row"):
                    with Vertical(classes="metadata-col"):
                        yield Label("Title (Optional):")
                        yield Input(
                            id="ingest-local-plaintext-title", 
                            placeholder="Use filename"
                        )
                    with Vertical(classes="metadata-col"):
                        yield Label("Author (Optional):")
                        yield Input(
                            id="ingest-local-plaintext-author",
                            placeholder="Optional"
                        )
                
                # Process button
                yield Button(
                    "Process Text Files", 
                    id="ingest-local-plaintext-process", 
                    variant="success",
                    classes="process-button"
                )
            
            # Basic options (visible in simple mode)
            with Container(id="plaintext-basic-options", classes="basic-options-container"):
                # Text encoding
                yield Label("Text Encoding:")
                yield Select(
                    [
                        ("UTF-8 (Default)", "utf-8"),
                        ("ASCII", "ascii"),
                        ("Latin-1", "latin-1"),
                        ("Auto-detect", "auto")
                    ],
                    id="ingest-local-plaintext-encoding",
                    value="utf-8"
                )
                
                yield Checkbox(
                    "Remove extra whitespace", 
                    value=True,
                    id="ingest-local-plaintext-remove-whitespace"
                )
                yield Checkbox(
                    "Convert to paragraphs", 
                    value=False,
                    id="ingest-local-plaintext-paragraphs"
                )
            
            # Advanced options (hidden in simple mode)
            with Container(id="plaintext-advanced-options", classes="advanced-options-container hidden"):
                # Keywords
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-plaintext-keywords", classes="ingest-textarea-small")
                
                # Text processing options
                with Collapsible(title="📝 Text Processing Options", collapsed=True):
                    yield Label("Line Ending:")
                    yield Select(
                        [
                            ("Auto", "auto"),
                            ("Unix (LF)", "lf"),
                            ("Windows (CRLF)", "crlf")
                        ],
                        id="ingest-local-plaintext-line-ending",
                        value="auto"
                    )
                    
                    yield Label("Split Pattern (Regex, optional):")
                    yield Input(
                        id="ingest-local-plaintext-split-pattern",
                        placeholder="e.g., \\n\\n+ for double newlines",
                        tooltip="Regular expression pattern for custom text splitting"
                    )
                
                # Chunking options
                with Collapsible(title="📄 Chunking Options", collapsed=True):
                    yield Checkbox(
                        "Enable chunking", 
                        True,
                        id="ingest-local-plaintext-perform-chunking"
                    )
                    
                    yield Label("Chunking Method:")
                    chunk_method_options = [
                        ("Paragraphs", "paragraphs"),
                        ("Sentences", "sentences"),
                        ("Tokens", "tokens"),
                        ("Words", "words"),
                        ("Sliding Window", "sliding_window")
                    ]
                    yield Select(
                        chunk_method_options,
                        id="ingest-local-plaintext-chunk-method",
                        value=plaintext_defaults.get("chunk_method", "paragraphs"),
                        prompt="Select chunking method..."
                    )
                    
                    with Horizontal(classes="chunk-settings-row"):
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Size:")
                            yield Input(
                                str(plaintext_defaults.get("chunk_size", 500)),
                                id="ingest-local-plaintext-chunk-size",
                                type="integer"
                            )
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Overlap:")
                            yield Input(
                                str(plaintext_defaults.get("chunk_overlap", 200)),
                                id="ingest-local-plaintext-chunk-overlap",
                                type="integer"
                            )
                    
                    yield Label("Chunk Language (e.g., 'en', optional):")
                    yield Input(
                        plaintext_defaults.get("chunk_language", ""),
                        id="ingest-local-plaintext-chunk-lang",
                        placeholder="Defaults to media language"
                    )
                    
                    yield Checkbox(
                        "Use Adaptive Chunking",
                        plaintext_defaults.get("use_adaptive_chunking", False),
                        id="ingest-local-plaintext-adaptive-chunking"
                    )
                    yield Checkbox(
                        "Use Multi-level Chunking",
                        plaintext_defaults.get("use_multi_level_chunking", False),
                        id="ingest-local-plaintext-multi-level-chunking"
                    )
                
                # Database options
                with Collapsible(title="💾 Database Options", collapsed=True):
                    yield Checkbox(
                        "Overwrite if exists in database",
                        False,
                        id="ingest-local-plaintext-overwrite-existing"
                    )
            
            # Status area for processing feedback
            yield LoadingIndicator(id="ingest-local-plaintext-loading", classes="hidden")
            yield TextArea(
                "",
                id="ingest-local-plaintext-status",
                read_only=True,
                classes="ingest-status-area hidden"
            )
    
    def watch_simple_mode(self, simple_mode: bool) -> None:
        """React to mode toggle changes."""
        # Only try to update UI if the widget is mounted
        if not self.is_mounted:
            return
            
        try:
            basic_options = self.query_one("#plaintext-basic-options")
            advanced_options = self.query_one("#plaintext-advanced-options")
            
            if simple_mode:
                basic_options.remove_class("hidden")
                advanced_options.add_class("hidden")
            else:
                basic_options.add_class("hidden")
                advanced_options.remove_class("hidden")
                
            logger.debug(f"Plaintext ingestion mode changed to: {'simple' if simple_mode else 'advanced'}")
        except Exception as e:
            logger.error(f"Error toggling plaintext mode: {e}")
    
    @on(RadioSet.Changed, "#plaintext-mode-toggle")
    def handle_mode_toggle(self, event: RadioSet.Changed) -> None:
        """Handle mode toggle changes."""
        self.simple_mode = event.radio_set.pressed_index == 0
        
        # Save preference
        from ..Utils.ingestion_preferences import save_ingestion_mode_preference
        save_ingestion_mode_preference("plaintext", self.simple_mode)
    
    @on(Button.Pressed, "#ingest-local-plaintext-select-files")
    async def handle_browse_files(self, event: Button.Pressed) -> None:
        """Handle file browser button."""
        filters = Filters(
            ("Text Files", lambda p: p.suffix.lower() in (".txt", ".text", ".md", ".markdown", ".rst", ".log")),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select Text Files",
                filters=filters
            ),
            callback=self.handle_file_selection
        )
    
    async def handle_file_selection(self, path: Path | None) -> None:
        """Handle file selection from dialog."""
        if path:
            file_list = self.query_one("#ingest-local-plaintext-files-list", FileListEnhanced)
            file_list.add_file(path)
            self.selected_local_files.append(path)
            
            # Update app instance selected files
            if not hasattr(self.app_instance, 'selected_local_files'):
                self.app_instance.selected_local_files = {}
            
            if 'local_plaintext' not in self.app_instance.selected_local_files:
                self.app_instance.selected_local_files['local_plaintext'] = []
            
            if path not in self.app_instance.selected_local_files['local_plaintext']:
                self.app_instance.selected_local_files['local_plaintext'].append(path)
    
    @on(Button.Pressed, "#ingest-local-plaintext-clear-files")
    async def handle_clear_files(self, event: Button.Pressed) -> None:
        """Handle clear files button."""
        file_list = self.query_one("#ingest-local-plaintext-files-list", FileListEnhanced)
        file_list.clear()
        self.selected_local_files.clear()
        
        # Clear app instance files
        if hasattr(self.app_instance, 'selected_local_files') and 'local_plaintext' in self.app_instance.selected_local_files:
            self.app_instance.selected_local_files['local_plaintext'].clear()
    
    @on(Button.Pressed, "#ingest-local-plaintext-process")
    async def handle_submit(self, event: Button.Pressed) -> None:
        """Handle submit button."""
        # Update status dashboard
        status_dashboard = self.query_one("#plaintext-status-dashboard", StatusDashboard)
        status_dashboard.start_processing(
            total_files=len(self.selected_local_files),
            message="Processing text files..."
        )
        
        # Import the actual plaintext processing handler
        from ..Event_Handlers.ingest_events import handle_local_plaintext_process
        
        # Call the real processing function
        await handle_local_plaintext_process(self.app_instance)

# End of IngestLocalPlaintextWindowSimplified.py