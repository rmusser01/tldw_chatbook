# tldw_chatbook/Widgets/IngestLocalPdfWindowSimplified.py
# Simplified version of PDF ingestion with progressive disclosure

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
from tldw_chatbook.config import get_media_ingestion_defaults
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from tldw_chatbook.Widgets.status_dashboard import StatusDashboard
from tldw_chatbook.Widgets.file_list_item_enhanced import FileListEnhanced

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

class IngestLocalPdfWindowSimplified(Vertical):
    """Simplified window for ingesting PDF content locally with progressive disclosure."""
    
    # Reactive property for simple/advanced mode
    simple_mode = reactive(True)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        
        # Load saved preference
        from ..Utils.ingestion_preferences import get_ingestion_mode_preference
        self.simple_mode = get_ingestion_mode_preference("pdf")
        
        logger.debug("[PDF] IngestLocalPdfWindowSimplified initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the simplified PDF ingestion form."""
        # Get PDF-specific default settings from config
        pdf_defaults = get_media_ingestion_defaults("pdf")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            # Status dashboard at top
            yield StatusDashboard(
                id="pdf-status-dashboard",
                show_file_counter=True,
                show_time=True,
                show_actions=True
            )
            
            # Mode toggle
            with Container(classes="mode-toggle-container"):
                yield Static("PDF Processing", classes="sidebar-title")
                with RadioSet(id="pdf-mode-toggle", classes="mode-toggle"):
                    yield RadioButton("Simple Mode", value=True, id="pdf-simple-radio")
                    yield RadioButton("Advanced Mode", id="pdf-advanced-radio")
            
            # Essential fields container (always visible)
            with Container(classes="essential-fields"):
                yield Label("Select PDF Files", classes="form-label-primary")
                
                # File selection
                with Horizontal(classes="file-selection-row"):
                    yield Button("Browse Files", id="local-browse-local-files-button-pdf", variant="primary")
                    yield Button("Clear All", id="local-clear-files-pdf", variant="default")
                
                # Selected files display with metadata
                yield Label("Selected Files:", classes="form-label")
                yield FileListEnhanced(
                    id="local-selected-local-files-list-pdf",
                    show_summary=True,
                    max_height=10
                )
                
                # Basic metadata
                with Horizontal(classes="metadata-row"):
                    with Vertical(classes="metadata-col"):
                        yield Label("Title (Optional):")
                        yield Input(
                            id="local-title-pdf", 
                            placeholder="Auto-detected from file"
                        )
                    with Vertical(classes="metadata-col"):
                        yield Label("Keywords (Optional):")
                        yield Input(
                            id="local-keywords-pdf",
                            placeholder="Comma-separated tags"
                        )
                
                # Process button
                yield Button(
                    "Process PDFs", 
                    id="local-submit-pdf", 
                    variant="success",
                    classes="process-button"
                )
            
            # Basic options (visible in simple mode)
            with Container(id="pdf-basic-options", classes="basic-options-container"):
                yield Label("PDF Engine:")
                yield Select(
                    [
                        ("PyMuPDF4LLM (Recommended)", "pymupdf4llm"),
                        ("PyMuPDF", "pymupdf"),
                        ("Docling", "docling")
                    ],
                    id="local-pdf-engine-pdf",
                    value="pymupdf4llm"
                )
                yield Checkbox(
                    "Generate summary", 
                    value=True,
                    id="local-perform-analysis-pdf"
                )
                yield Checkbox(
                    "Extract images", 
                    value=False,
                    id="local-extract-images-pdf"
                )
            
            # Advanced options (hidden in simple mode)
            with Container(id="pdf-advanced-options", classes="advanced-options-container hidden"):
                # Analysis options
                with Collapsible(title="📊 Analysis Options", collapsed=True):
                    yield Label("Custom Analysis Prompt:")
                    yield TextArea(
                        id="local-custom-prompt-pdf",
                        classes="ingest-textarea-medium"
                    )
                    
                    yield Label("System Prompt (Optional):")
                    yield TextArea(
                        id="local-system-prompt-pdf",
                        classes="ingest-textarea-medium"
                    )
                    
                    yield Label("Analysis Provider:")
                    analysis_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
                    analysis_options = [(name, name) for name in analysis_providers if name]
                    if not analysis_options:
                        analysis_options = [("No Providers Configured", Select.BLANK)]
                    
                    yield Select(
                        analysis_options,
                        id="local-api-name-pdf",
                        prompt="Select API for Analysis..."
                    )
                    
                    yield Checkbox(
                        "Summarize recursively", 
                        False,
                        id="local-summarize-recursively-pdf"
                    )
                    yield Checkbox(
                        "Perform rolling summarization", 
                        False,
                        id="local-perform-rolling-summarization-pdf"
                    )
                
                # Chunking options
                with Collapsible(title="📄 Chunking Options", collapsed=True):
                    yield Checkbox(
                        "Enable chunking", 
                        True,
                        id="local-perform-chunking-pdf"
                    )
                    
                    yield Label("Chunk Method:")
                    yield Select(
                        [
                            ("Semantic", "semantic"),
                            ("Tokens", "tokens"),
                            ("Sentences", "sentences"),
                            ("Words", "words"),
                            ("Paragraphs", "paragraphs")
                        ],
                        id="local-chunk-method-pdf",
                        value="semantic",
                        prompt="Select chunking method..."
                    )
                    
                    with Horizontal(classes="chunk-settings-row"):
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Size:")
                            yield Input("500", id="local-chunk-size-pdf", type="integer")
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Overlap:")
                            yield Input("200", id="local-chunk-overlap-pdf", type="integer")
                    
                    yield Checkbox(
                        "Use adaptive chunking", 
                        False,
                        id="local-adaptive-chunking-pdf"
                    )
                    yield Checkbox(
                        "Use multi-level chunking", 
                        False,
                        id="local-multi-level-chunking-pdf"
                    )
            
            # Status area for processing feedback
            yield LoadingIndicator(id="local-loading-indicator-pdf", classes="hidden")
    
    def watch_simple_mode(self, simple_mode: bool) -> None:
        """React to mode toggle changes."""
        # Only try to update UI if the widget is mounted
        if not self.is_mounted:
            return
            
        try:
            basic_options = self.query_one("#pdf-basic-options")
            advanced_options = self.query_one("#pdf-advanced-options")
            
            if simple_mode:
                basic_options.remove_class("hidden")
                advanced_options.add_class("hidden")
            else:
                basic_options.add_class("hidden")
                advanced_options.remove_class("hidden")
                
            logger.debug(f"PDF ingestion mode changed to: {'simple' if simple_mode else 'advanced'}")
        except Exception as e:
            logger.error(f"Error toggling PDF mode: {e}")
    
    @on(RadioSet.Changed, "#pdf-mode-toggle")
    def handle_mode_toggle(self, event: RadioSet.Changed) -> None:
        """Handle mode toggle changes."""
        self.simple_mode = event.radio_set.pressed_index == 0
        
        # Save preference
        from ..Utils.ingestion_preferences import save_ingestion_mode_preference
        save_ingestion_mode_preference("pdf", self.simple_mode)
    
    @on(Button.Pressed, "#local-browse-local-files-button-pdf")
    async def handle_browse_files(self, event: Button.Pressed) -> None:
        """Handle file browser button."""
        filters = Filters(
            ("PDF Files", lambda p: p.suffix.lower() == ".pdf"),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select PDF Files",
                filters=filters
            ),
            callback=self.handle_file_selection
        )
    
    async def handle_file_selection(self, path: Path | None) -> None:
        """Handle file selection from dialog."""
        if path:
            file_list = self.query_one("#local-selected-local-files-list-pdf", FileListEnhanced)
            file_list.add_file(path)
            self.selected_local_files.append(path)
            
            # Update app instance selected files
            if not hasattr(self.app_instance, 'selected_local_files'):
                self.app_instance.selected_local_files = {}
            
            if 'local_pdf' not in self.app_instance.selected_local_files:
                self.app_instance.selected_local_files['local_pdf'] = []
            
            if path not in self.app_instance.selected_local_files['local_pdf']:
                self.app_instance.selected_local_files['local_pdf'].append(path)
    
    @on(Button.Pressed, "#local-clear-files-pdf")
    async def handle_clear_files(self, event: Button.Pressed) -> None:
        """Handle clear files button."""
        file_list = self.query_one("#local-selected-local-files-list-pdf", FileListEnhanced)
        file_list.clear()
        self.selected_local_files.clear()
        
        # Clear app instance files
        if hasattr(self.app_instance, 'selected_local_files') and 'local_pdf' in self.app_instance.selected_local_files:
            self.app_instance.selected_local_files['local_pdf'].clear()
    
    @on(Button.Pressed, "#local-submit-pdf")
    async def handle_submit(self, event: Button.Pressed) -> None:
        """Handle submit button."""
        # Import the actual PDF processing handler
        from ..Event_Handlers.ingest_events import handle_local_pdf_process
        
        # Call the real processing function
        await handle_local_pdf_process(self.app_instance)

# End of IngestLocalPdfWindowSimplified.py