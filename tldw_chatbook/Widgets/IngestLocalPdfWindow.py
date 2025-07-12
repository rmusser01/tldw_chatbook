# tldw_chatbook/Widgets/IngestLocalPdfWindow.py

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
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalPdfWindow(Vertical):
    """Window for ingesting PDF content locally."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        logger.debug("IngestLocalPdfWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the PDF ingestion form."""
        # Get available API providers for analysis from app config
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]
        
        # Get PDF-specific default chunking settings from config
        pdf_defaults = get_media_ingestion_defaults("pdf")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("Local PDF Processing", classes="sidebar-title")
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- Common Input Fields ---
            yield Label("Media URLs (one per line):")
            yield TextArea(id="local-urls-pdf", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id="local-browse-local-files-button-pdf")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="local-selected-local-files-list-pdf", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="local-title-pdf", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="local-author-pdf", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="local-keywords-pdf", classes="ingest-textarea-small")
            
            # --- Common Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="local-custom-prompt-pdf", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="local-system-prompt-pdf", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="local-perform-analysis-pdf")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="local-analysis-api-name-pdf",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="local-analysis-api-key-pdf",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="local-chunking-collapsible-pdf"):
                yield Checkbox("Perform Chunking", True, id="local-perform-chunking-pdf")
                yield Label("Chunking Method:")
                chunk_method_options = [
                    ("semantic", "semantic"),
                    ("tokens", "tokens"),
                    ("paragraphs", "paragraphs"),
                    ("sentences", "sentences"),
                    ("words", "words"),
                    ("ebook_chapters", "ebook_chapters"),
                    ("json", "json")
                ]
                yield Select(chunk_method_options, id="local-chunk-method-pdf", 
                            value=pdf_defaults.get("chunk_method", "semantic"),
                            prompt="Default (per type)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input(str(pdf_defaults.get("chunk_size", 500)), 
                                   id="local-chunk-size-pdf", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input(str(pdf_defaults.get("chunk_overlap", 200)), 
                                   id="local-chunk-overlap-pdf", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(pdf_defaults.get("chunk_language", ""), id="local-chunk-lang-pdf", 
                           placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", 
                              pdf_defaults.get("use_adaptive_chunking", False), 
                              id="local-adaptive-chunking-pdf")
                yield Checkbox("Use Multi-level Chunking", 
                              pdf_defaults.get("use_multi_level_chunking", False), 
                              id="local-multi-level-chunking-pdf")
                yield Label("Custom Chapter Pattern (Regex, optional):")
                yield Input(id="local-custom-chapter-pattern-pdf", placeholder="e.g., ^Chapter\\s+\\d+")
            
            # --- Common Analysis Options ---
            with Collapsible(title="Advanced Analysis Options", collapsed=True,
                             id="local-analysis-opts-collapsible-pdf"):
                yield Checkbox("Summarize Recursively (if chunked)", False, id="local-summarize-recursively-pdf")
                yield Checkbox("Perform Rolling Summarization", False, id="local-perform-rolling-summarization-pdf")
            
            # --- PDF Specific Options ---
            yield Static("PDF Specific Options", classes="sidebar-title")
            
            # Check available PDF processing engines
            pdf_engine_options = []
            default_engine = None
            
            if DEPENDENCIES_AVAILABLE.get('pymupdf4llm', False):
                pdf_engine_options.append(("pymupdf4llm", "pymupdf4llm"))
                default_engine = "pymupdf4llm"
            if DEPENDENCIES_AVAILABLE.get('pymupdf', False):
                pdf_engine_options.append(("pymupdf", "pymupdf"))
                if not default_engine:
                    default_engine = "pymupdf"
            if DEPENDENCIES_AVAILABLE.get('docling', False):
                pdf_engine_options.append(("docling", "docling"))
                if not default_engine:
                    default_engine = "docling"
                    
            if pdf_engine_options:
                yield Label("PDF Parsing Engine:")
                yield Select(pdf_engine_options, id="local-pdf-engine-pdf", value=default_engine)
            else:
                yield Static("⚠️ No PDF processing engines available. Install with: pip install tldw_chatbook[pdf]", 
                           classes="warning-message")
                yield Select([("No engines available", Select.BLANK)], id="local-pdf-engine-pdf", disabled=True)
            
            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id="local-overwrite-db-pdf")
            
            # Only enable submit button if PDF processing is available
            pdf_processing_available = DEPENDENCIES_AVAILABLE.get('pdf_processing', False)
            yield Button(
                "Process PDF Locally", 
                id="local-submit-pdf", 
                variant="primary" if pdf_processing_available else "default",
                classes="ingest-submit-button",
                disabled=not pdf_processing_available
            )
            yield LoadingIndicator(id="local-loading-indicator-pdf", classes="hidden")
            yield TextArea(
                "",
                id="local-status-area-pdf",
                read_only=True,
                classes="ingest-status-area hidden"
            )