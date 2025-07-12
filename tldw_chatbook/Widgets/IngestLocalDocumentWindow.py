# tldw_chatbook/Widgets/IngestLocalDocumentWindow.py

from typing import TYPE_CHECKING
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible
)
from ..config import get_media_ingestion_defaults

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalDocumentWindow(Vertical):
    """Window for ingesting document content locally."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        logger.debug("IngestLocalDocumentWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the document ingestion form."""
        # Get available API providers for analysis from app config
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]
        
        # Get document-specific default chunking settings from config
        document_defaults = get_media_ingestion_defaults("document")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("Local Document Processing", classes="sidebar-title")
            
            yield Static("Supported Formats: DOCX, ODT, RTF, PPTX, XLSX, ODS, ODP", classes="ingest-info-text")
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- File Selection ---
            yield Button("Browse Local Files...", id="local-browse-local-files-button-document")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="local-selected-local-files-list-document", classes="ingest-selected-files-list")
            yield Button("Clear Selection", id="local-clear-files-document", variant="warning")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="local-title-document", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="local-author-document", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="local-keywords-document", classes="ingest-textarea-small")
            
            # --- Processing Method ---
            yield Label("Processing Method:")
            processing_options = [
                ("Auto (Best Available)", "auto"),
                ("Docling (Advanced)", "docling"),
                ("Native Libraries", "native")
            ]
            yield Select(processing_options, id="local-processing-method-document", 
                        value="auto", prompt="Select processing method...")
            
            # --- Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="local-custom-prompt-document", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="local-system-prompt-document", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="local-perform-analysis-document")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="local-analysis-api-name-document",
                        prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="local-analysis-api-key-document",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="local-chunking-collapsible-document"):
                yield Checkbox("Perform Chunking", True, id="local-perform-chunking-document")
                yield Label("Chunking Method:")
                chunk_method_options = [
                    ("semantic", "semantic"),
                    ("tokens", "tokens"),
                    ("paragraphs", "paragraphs"),
                    ("sentences", "sentences"),
                    ("words", "words"),
                    ("sliding_window", "sliding_window")
                ]
                yield Select(chunk_method_options, id="local-chunk-method-document", 
                            value=document_defaults.get("chunk_method", "sentences"),
                            prompt="Select chunking method...")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input(str(document_defaults.get("chunk_size", 1500)), 
                                   id="local-chunk-size-document", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input(str(document_defaults.get("chunk_overlap", 100)), 
                                   id="local-chunk-overlap-document", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(document_defaults.get("chunk_language", ""), id="local-chunk-lang-document", 
                           placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", 
                              document_defaults.get("use_adaptive_chunking", False), 
                              id="local-adaptive-chunking-document")
                yield Checkbox("Use Multi-level Chunking", 
                              document_defaults.get("use_multi_level_chunking", False), 
                              id="local-multi-level-chunking-document")
            
            # --- Document-Specific Options ---
            with Collapsible(title="Document-Specific Options", collapsed=True):
                yield Checkbox("Extract Tables", True, id="local-extract-tables-document")
                yield Checkbox("Extract Images (if supported)", False, id="local-extract-images-document")
                yield Checkbox("Preserve Formatting", True, id="local-preserve-formatting-document")
                yield Checkbox("Include Metadata", True, id="local-include-metadata-document")
                yield Label("Max Pages (0 = all):")
                yield Input("0", id="local-max-pages-document", type="integer")
            
            # --- Process Button ---
            yield Button("Process Documents", id="local-process-button-document", variant="primary")
            
            # --- Status Display ---
            yield Static("Processing Status", classes="sidebar-title")
            status_area = TextArea(
                "",
                id="local-status-area-document",
                read_only=True,
                classes="ingest-status-area"
            )
            status_area.display = False
            yield status_area
            
            # Loading indicator
            loading_indicator = LoadingIndicator(id="local-loading-indicator-document")
            loading_indicator.display = False
            yield loading_indicator