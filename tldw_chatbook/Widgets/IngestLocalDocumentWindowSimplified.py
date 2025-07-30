# tldw_chatbook/Widgets/IngestLocalDocumentWindowSimplified.py
# Simplified version of document ingestion with progressive disclosure

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
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..Widgets.status_dashboard import StatusDashboard
from ..Widgets.file_list_item_enhanced import FileListEnhanced

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalDocumentWindowSimplified(Vertical):
    """Simplified window for ingesting document content locally with progressive disclosure."""
    
    # Reactive property for simple/advanced mode
    simple_mode = reactive(True)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        
        # Load saved preference
        from ..Utils.ingestion_preferences import get_ingestion_mode_preference
        self.simple_mode = get_ingestion_mode_preference("document")
        
        logger.debug("[Document] IngestLocalDocumentWindowSimplified initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the simplified document ingestion form."""
        # Get document-specific default settings from config
        document_defaults = get_media_ingestion_defaults("document")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            # Status dashboard at top
            yield StatusDashboard(
                id="document-status-dashboard",
                show_file_counter=True,
                show_time=True,
                show_actions=True
            )
            
            # Mode toggle
            with Container(classes="mode-toggle-container"):
                yield Static("Document Processing", classes="sidebar-title")
                with RadioSet(id="document-mode-toggle", classes="mode-toggle"):
                    yield RadioButton("Simple Mode", value=True, id="document-simple-radio")
                    yield RadioButton("Advanced Mode", id="document-advanced-radio")
            
            # Essential fields container (always visible)
            with Container(classes="essential-fields"):
                yield Label("Select Document Files", classes="form-label-primary")
                
                # File selection
                with Horizontal(classes="file-selection-row"):
                    yield Button("Browse Files", id="ingest-local-document-select-files", variant="primary")
                    yield Button("Clear All", id="ingest-local-document-clear-files", variant="default")
                
                # Selected files display with metadata
                yield Label("Selected Files:", classes="form-label")
                yield FileListEnhanced(
                    id="ingest-local-document-files-list",
                    show_summary=True,
                    max_height=10
                )
                
                # Basic metadata
                with Horizontal(classes="metadata-row"):
                    with Vertical(classes="metadata-col"):
                        yield Label("Title (Optional):")
                        yield Input(
                            id="ingest-local-document-title", 
                            placeholder="Auto-detected from file"
                        )
                    with Vertical(classes="metadata-col"):
                        yield Label("Keywords (Optional):")
                        yield Input(
                            id="ingest-local-document-keywords",
                            placeholder="Comma-separated tags"
                        )
                
                # Process button
                yield Button(
                    "Process Documents", 
                    id="ingest-local-document-process", 
                    variant="success",
                    classes="process-button"
                )
            
            # Basic options (visible in simple mode)
            with Container(id="document-basic-options", classes="basic-options-container"):
                yield Checkbox(
                    "Extract text only", 
                    value=False,
                    id="ingest-local-document-text-only"
                )
                yield Checkbox(
                    "Generate summary", 
                    value=True,
                    id="ingest-local-document-perform-analysis"
                )
                yield Checkbox(
                    "Preserve formatting", 
                    value=True,
                    id="ingest-local-document-preserve-formatting"
                )
            
            # Advanced options (hidden in simple mode)
            with Container(id="document-advanced-options", classes="advanced-options-container hidden"):
                # Analysis options
                with Collapsible(title="📊 Analysis Options", collapsed=True):
                    yield Label("Custom Analysis Prompt:")
                    yield TextArea(
                        id="ingest-local-document-custom-prompt",
                        classes="ingest-textarea-medium",
                        placeholder="Provide specific instructions for analysis..."
                    )
                    
                    yield Label("System Prompt:")
                    yield TextArea(
                        id="ingest-local-document-system-prompt",
                        classes="ingest-textarea-medium",
                        placeholder="Optional system prompt for analysis..."
                    )
                    
                    yield Label("Analysis Provider:")
                    analysis_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
                    analysis_options = [(name, name) for name in analysis_providers if name]
                    if not analysis_options:
                        analysis_options = [("No Providers Configured", Select.BLANK)]
                    
                    yield Select(
                        analysis_options,
                        id="ingest-local-document-api-name",
                        prompt="Select API for Analysis..."
                    )
                    
                    yield Checkbox(
                        "Summarize recursively", 
                        False,
                        id="ingest-local-document-summarize-recursively"
                    )
                    yield Checkbox(
                        "Perform rolling summarization", 
                        False,
                        id="ingest-local-document-perform-rolling-summarization"
                    )
                
                # Chunking options
                with Collapsible(title="📄 Chunking Options", collapsed=True):
                    yield Checkbox(
                        "Enable chunking", 
                        True,
                        id="ingest-local-document-perform-chunking"
                    )
                    
                    yield Label("Chunk Method:")
                    yield Select(
                        [
                            ("Sentences", "sentences"),
                            ("Semantic", "semantic"),
                            ("Tokens", "tokens"),
                            ("Words", "words"),
                            ("Paragraphs", "paragraphs")
                        ],
                        id="ingest-local-document-chunk-method",
                        value="sentences",
                        prompt="Select chunking method..."
                    )
                    
                    with Horizontal(classes="chunk-settings-row"):
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Size:")
                            yield Input("1000", id="ingest-local-document-chunk-size", type="integer")
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Overlap:")
                            yield Input("200", id="ingest-local-document-chunk-overlap", type="integer")
                    
                    yield Checkbox(
                        "Use adaptive chunking", 
                        False,
                        id="ingest-local-document-use-adaptive-chunking"
                    )
                    yield Checkbox(
                        "Use multi-level chunking", 
                        False,
                        id="ingest-local-document-use-multi-level-chunking"
                    )
            
            # Status area
            yield create_status_area("ingest-local-document")
    
    def watch_simple_mode(self, simple_mode: bool) -> None:
        """React to mode toggle changes."""
        try:
            basic_options = self.query_one("#document-basic-options")
            advanced_options = self.query_one("#document-advanced-options")
            
            if simple_mode:
                basic_options.remove_class("hidden")
                advanced_options.add_class("hidden")
            else:
                basic_options.add_class("hidden")
                advanced_options.remove_class("hidden")
                
            logger.debug(f"Document ingestion mode changed to: {'simple' if simple_mode else 'advanced'}")
        except Exception as e:
            logger.error(f"Error toggling document mode: {e}")
    
    @on(RadioSet.Changed, "#document-mode-toggle")
    def handle_mode_toggle(self, event: RadioSet.Changed) -> None:
        """Handle mode toggle changes."""
        self.simple_mode = event.radio_set.pressed_index == 0
        
        # Save preference
        from ..Utils.ingestion_preferences import save_ingestion_mode_preference
        save_ingestion_mode_preference("document", self.simple_mode)
    
    @on(Button.Pressed, "#ingest-local-document-select-files")
    async def handle_browse_files(self, event: Button.Pressed) -> None:
        """Handle file browser button."""
        filters = Filters(
            ("Document Files", lambda p: p.suffix.lower() in (".docx", ".doc", ".odt", ".rtf", ".pptx", ".ppt", ".xlsx", ".xls", ".ods", ".odp")),
            ("Microsoft Word", lambda p: p.suffix.lower() in (".docx", ".doc")),
            ("OpenDocument", lambda p: p.suffix.lower() in (".odt", ".ods", ".odp")),
            ("Microsoft Office", lambda p: p.suffix.lower() in (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls")),
            ("Rich Text", lambda p: p.suffix.lower() == ".rtf"),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select Document Files",
                filters=filters
            ),
            callback=self.handle_file_selection
        )
    
    async def handle_file_selection(self, path: Path | None) -> None:
        """Handle file selection from dialog."""
        if path:
            file_list = self.query_one("#ingest-local-document-files-list", FileListEnhanced)
            file_list.add_file(path)
            self.selected_local_files.append(path)
            
            # Update app instance selected files
            if not hasattr(self.app_instance, 'selected_local_files'):
                self.app_instance.selected_local_files = {}
            
            if 'local_document' not in self.app_instance.selected_local_files:
                self.app_instance.selected_local_files['local_document'] = []
            
            if path not in self.app_instance.selected_local_files['local_document']:
                self.app_instance.selected_local_files['local_document'].append(path)
    
    @on(Button.Pressed, "#ingest-local-document-clear-files")
    async def handle_clear_files(self, event: Button.Pressed) -> None:
        """Handle clear files button."""
        file_list = self.query_one("#ingest-local-document-files-list", FileListEnhanced)
        file_list.clear()
        self.selected_local_files.clear()
        
        # Clear app instance files
        if hasattr(self.app_instance, 'selected_local_files') and 'local_document' in self.app_instance.selected_local_files:
            self.app_instance.selected_local_files['local_document'].clear()
    
    @on(Button.Pressed, "#ingest-local-document-process")
    async def handle_submit(self, event: Button.Pressed) -> None:
        """Handle submit button."""
        # Import the actual document processing handler
        from ..Event_Handlers.ingest_events import handle_local_document_process
        
        # Call the real processing function
        await handle_local_document_process(self.app_instance)

def create_status_area(id_prefix: str) -> ComposeResult:
    """Create a standard status area for ingestion forms."""
    from ..Widgets.status_widget import EnhancedStatusWidget
    
    yield EnhancedStatusWidget(
        title="Processing Status",
        id=f"{id_prefix}-status-widget",
        max_messages=50
    )

# End of IngestLocalDocumentWindowSimplified.py