# tldw_chatbook/Widgets/IngestLocalEbookWindowSimplified.py
# Simplified version of ebook ingestion with progressive disclosure

from typing import TYPE_CHECKING, List
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible, RadioSet, RadioButton
)
from textual import on, work
from textual.reactive import reactive
from ..config import get_media_ingestion_defaults
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..Widgets.prompt_selector import PromptSelector
from ..Widgets.file_list_item_enhanced import FileListEnhanced
from ..Widgets.status_dashboard import StatusDashboard

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalEbookWindowSimplified(Vertical):
    """Simplified window for ingesting ebook content locally with progressive disclosure."""
    
    # Reactive property for simple/advanced mode
    simple_mode = reactive(True)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        
        # Load saved preference
        from ..Utils.ingestion_preferences import get_ingestion_mode_preference
        self.simple_mode = get_ingestion_mode_preference("ebook")
        
        logger.debug("[Ebook] IngestLocalEbookWindowSimplified initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the simplified ebook ingestion form."""
        # Get ebook-specific default settings from config
        ebook_defaults = get_media_ingestion_defaults("ebook")
        
        # Check if ebook processing is available
        ebook_processing_available = DEPENDENCIES_AVAILABLE.get('ebook_processing', False)
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            # Status dashboard at top
            yield StatusDashboard(id="ebook-status-dashboard")
            
            # Mode toggle
            with Container(classes="mode-toggle-container"):
                yield Static("Ebook Processing", classes="sidebar-title")
                with RadioSet(id="ebook-mode-toggle", classes="mode-toggle"):
                    yield RadioButton("Simple Mode", value=True, id="ebook-simple-radio")
                    yield RadioButton("Advanced Mode", id="ebook-advanced-radio")
            
            # Warning if ebook processing not available
            if not ebook_processing_available:
                yield Static(
                    "⚠️ Ebook processing not available. Install with: pip install tldw_chatbook[ebook]",
                    classes="warning-message"
                )
            
            # Essential fields container (always visible)
            with Container(classes="essential-fields"):
                yield Label("Select Ebook Files or Enter URLs", classes="form-label-primary")
                
                # File selection
                with Horizontal(classes="file-selection-row"):
                    yield Button(
                        "Browse Files", 
                        id="local-browse-local-files-button-ebook", 
                        variant="primary" if ebook_processing_available else "default",
                        disabled=not ebook_processing_available
                    )
                    yield Button("Clear All", id="local-clear-files-ebook", variant="default")
                
                # URL input
                yield Label("Ebook URLs (one per line):")
                yield TextArea(
                    id="local-urls-ebook", 
                    classes="ingest-textarea-small"
                )
                
                # Selected files display with metadata
                yield Label("Selected Files:", classes="form-label")
                yield FileListEnhanced(
                    id="local-selected-files-ebook",
                    show_summary=True,
                    max_height=10
                )
                
                # Basic metadata
                with Horizontal(classes="metadata-row"):
                    with Vertical(classes="metadata-col"):
                        yield Label("Title (Optional):")
                        yield Input(
                            id="local-title-ebook", 
                            placeholder="Auto-detected from file"
                        )
                    with Vertical(classes="metadata-col"):
                        yield Label("Author (Optional):")
                        yield Input(
                            id="local-author-ebook",
                            placeholder="Auto-detected from file"
                        )
                
                # Process button
                yield Button(
                    "Process Ebooks", 
                    id="local-submit-ebook", 
                    variant="success" if ebook_processing_available else "default",
                    classes="process-button",
                    disabled=not ebook_processing_available
                )
            
            # Basic options (visible in simple mode)
            with Container(id="ebook-basic-options", classes="basic-options-container"):
                # Ebook extraction method
                yield Label("Extraction Method:")
                if ebook_processing_available:
                    extraction_options = [
                        ("Filtered (Recommended)", "filtered"),
                        ("Markdown", "markdown"),
                        ("Basic", "basic")
                    ]
                    yield Select(
                        extraction_options,
                        id="local-ebook-extraction-method-ebook",
                        value="filtered"
                    )
                else:
                    yield Select(
                        [("No processing available", Select.BLANK)],
                        id="local-ebook-extraction-method-ebook",
                        disabled=True
                    )
                
                yield Checkbox(
                    "Generate summary", 
                    value=True,
                    id="local-perform-analysis-ebook"
                )
                yield Checkbox(
                    "Overwrite if exists in database", 
                    value=False,
                    id="local-overwrite-db-ebook"
                )
            
            # Advanced options (hidden in simple mode)
            with Container(id="ebook-advanced-options", classes="advanced-options-container hidden"):
                # Keywords
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="local-keywords-ebook", classes="ingest-textarea-small")
                
                # Analysis options
                with Collapsible(title="📊 Analysis Options", collapsed=True):
                    # Prompt selector widget
                    yield PromptSelector(
                        self.app_instance,
                        system_prompt_id="local-system-prompt-ebook",
                        user_prompt_id="local-custom-prompt-ebook",
                        media_type="document",  # Using 'document' as ebooks are similar
                        id="local-prompt-selector-ebook"
                    )
                    
                    yield Label("Analysis API Provider:")
                    analysis_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
                    analysis_options = [(name, name) for name in analysis_providers if name]
                    if not analysis_options:
                        analysis_options = [("No Providers Configured", Select.BLANK)]
                    
                    yield Select(
                        analysis_options,
                        id="local-analysis-api-name-ebook",
                        prompt="Select API for Analysis..."
                    )
                    
                    yield Label("Analysis API Key (if needed):")
                    yield Input(
                        "",
                        id="local-analysis-api-key-ebook",
                        placeholder="API key for analysis provider",
                        password=True
                    )
                
                # Chunking options
                with Collapsible(title="📄 Chunking Options", collapsed=True):
                    yield Checkbox(
                        "Enable chunking", 
                        True,
                        id="local-perform-chunking-ebook"
                    )
                    
                    yield Label("Chunking Method:")
                    chunk_method_options = [
                        ("Ebook Chapters", "ebook_chapters"),
                        ("Semantic", "semantic"),
                        ("Tokens", "tokens"),
                        ("Paragraphs", "paragraphs"),
                        ("Sentences", "sentences"),
                        ("Words", "words"),
                        ("JSON", "json")
                    ]
                    yield Select(
                        chunk_method_options,
                        id="local-chunk-method-ebook",
                        value=ebook_defaults.get("chunk_method", "ebook_chapters"),
                        prompt="Select chunking method..."
                    )
                    
                    with Horizontal(classes="chunk-settings-row"):
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Size:")
                            yield Input(
                                str(ebook_defaults.get("chunk_size", 1000)),
                                id="local-chunk-size-ebook",
                                type="integer"
                            )
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Overlap:")
                            yield Input(
                                str(ebook_defaults.get("chunk_overlap", 200)),
                                id="local-chunk-overlap-ebook",
                                type="integer"
                            )
                    
                    yield Label("Chunk Language (e.g., 'en', optional):")
                    yield Input(
                        ebook_defaults.get("chunk_language", ""),
                        id="local-chunk-lang-ebook",
                        placeholder="Defaults to media language"
                    )
                    
                    yield Checkbox(
                        "Use Adaptive Chunking",
                        ebook_defaults.get("use_adaptive_chunking", False),
                        id="local-adaptive-chunking-ebook"
                    )
                    yield Checkbox(
                        "Use Multi-level Chunking",
                        ebook_defaults.get("use_multi_level_chunking", False),
                        id="local-multi-level-chunking-ebook"
                    )
                    
                    yield Label("Custom Chapter Pattern (Regex, optional):")
                    yield Input(
                        id="local-custom-chapter-pattern-ebook",
                        placeholder="e.g., ^Chapter\\s+\\d+"
                    )
                
                # Advanced analysis options
                with Collapsible(title="🔬 Advanced Analysis Options", collapsed=True):
                    yield Checkbox(
                        "Summarize Recursively (if chunked)",
                        False,
                        id="local-summarize-recursively-ebook"
                    )
                    yield Checkbox(
                        "Perform Rolling Summarization",
                        False,
                        id="local-perform-rolling-summarization-ebook"
                    )
            
            # Status area for processing feedback
            yield LoadingIndicator(id="local-loading-indicator-ebook", classes="hidden")
            yield TextArea(
                "",
                id="local-status-area-ebook",
                read_only=True,
                classes="ingest-status-area hidden"
            )
    
    def watch_simple_mode(self, simple_mode: bool) -> None:
        """React to mode toggle changes."""
        # Only try to update UI if the widget is mounted
        if not self.is_mounted:
            return
            
        try:
            basic_options = self.query_one("#ebook-basic-options")
            advanced_options = self.query_one("#ebook-advanced-options")
            
            if simple_mode:
                basic_options.remove_class("hidden")
                advanced_options.add_class("hidden")
            else:
                basic_options.add_class("hidden")
                advanced_options.remove_class("hidden")
                
            logger.debug(f"Ebook ingestion mode changed to: {'simple' if simple_mode else 'advanced'}")
        except Exception as e:
            logger.error(f"Error toggling ebook mode: {e}")
    
    @on(RadioSet.Changed, "#ebook-mode-toggle")
    def handle_mode_toggle(self, event: RadioSet.Changed) -> None:
        """Handle mode toggle changes."""
        self.simple_mode = event.radio_set.pressed_index == 0
        
        # Save preference
        from ..Utils.ingestion_preferences import save_ingestion_mode_preference
        save_ingestion_mode_preference("ebook", self.simple_mode)
    
    @on(Button.Pressed, "#local-browse-local-files-button-ebook")
    async def handle_browse_files(self, event: Button.Pressed) -> None:
        """Handle file browser button."""
        filters = Filters(
            ("Ebook Files", lambda p: p.suffix.lower() in (".epub", ".mobi", ".azw3", ".fb2", ".lit", ".pdb")),
            ("PDF Files", lambda p: p.suffix.lower() == ".pdf"),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select Ebook Files",
                filters=filters
            ),
            callback=self.handle_file_selection
        )
    
    async def handle_file_selection(self, path: Path | None) -> None:
        """Handle file selection from dialog."""
        if path:
            file_list = self.query_one("#local-selected-files-ebook", FileListEnhanced)
            file_list.add_file(path)
            self.selected_local_files.append(path)
            
            # Update app instance selected files
            if not hasattr(self.app_instance, 'selected_local_files'):
                self.app_instance.selected_local_files = {}
            
            if 'local_ebook' not in self.app_instance.selected_local_files:
                self.app_instance.selected_local_files['local_ebook'] = []
            
            if path not in self.app_instance.selected_local_files['local_ebook']:
                self.app_instance.selected_local_files['local_ebook'].append(path)
    
    @on(Button.Pressed, "#local-clear-files-ebook")
    async def handle_clear_files(self, event: Button.Pressed) -> None:
        """Handle clear files button."""
        file_list = self.query_one("#local-selected-files-ebook", FileListEnhanced)
        file_list.clear()
        self.selected_local_files.clear()
        
        # Clear app instance files
        if hasattr(self.app_instance, 'selected_local_files') and 'local_ebook' in self.app_instance.selected_local_files:
            self.app_instance.selected_local_files['local_ebook'].clear()
    
    @on(Button.Pressed, "#local-submit-ebook")
    async def handle_submit(self, event: Button.Pressed) -> None:
        """Handle submit button."""
        # Update status dashboard
        status_dashboard = self.query_one("#ebook-status-dashboard", StatusDashboard)
        status_dashboard.start_processing(
            total_files=len(self.selected_local_files),
            message="Processing ebook files..."
        )
        
        # Import the actual ebook processing handler
        from ..Event_Handlers.ingest_events import handle_local_ebook_process
        
        # Call the real processing function
        await handle_local_ebook_process(self.app_instance)

# End of IngestLocalEbookWindowSimplified.py