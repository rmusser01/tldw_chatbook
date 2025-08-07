# tldw_chatbook/Widgets/IngestLocalEbookWindow.py

from typing import TYPE_CHECKING
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible
)
from tldw_chatbook.config import get_media_ingestion_defaults
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
from tldw_chatbook.Widgets.prompt_selector import PromptSelector

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

class IngestLocalEbookWindow(Vertical):
    """Window for ingesting ebook content locally."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        logger.debug("IngestLocalEbookWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the ebook ingestion form."""
        # Get available API providers for analysis from app config
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]
        
        # Get ebook-specific default chunking settings from config
        ebook_defaults = get_media_ingestion_defaults("ebook")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("Local Ebook Processing", classes="sidebar-title")
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- Common Input Fields ---
            yield Label("Media URLs (one per line):")
            yield TextArea(id="local-urls-ebook", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id="local-browse-local-files-button-ebook")
            yield Button("Clear Selection", id="local-clear-files-ebook", variant="warning")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="local-selected-local-files-list-ebook", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="local-title-ebook", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="local-author-ebook", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="local-keywords-ebook", classes="ingest-textarea-small")
            
            # --- Common Processing Options ---
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="local-perform-analysis-ebook")
            
            # Prompt selector widget
            yield PromptSelector(
                self.app_instance,
                system_prompt_id="local-system-prompt-ebook",
                user_prompt_id="local-custom-prompt-ebook",
                media_type="document",  # Using 'document' as ebooks are similar
                id="local-prompt-selector-ebook"
            )
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="local-analysis-api-name-ebook",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="local-analysis-api-key-ebook",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="local-chunking-collapsible-ebook"):
                yield Checkbox("Perform Chunking", True, id="local-perform-chunking-ebook")
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
                yield Select(chunk_method_options, id="local-chunk-method-ebook", 
                            value=ebook_defaults.get("chunk_method", "ebook_chapters"),
                            prompt="Default (per type)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input(str(ebook_defaults.get("chunk_size", 1000)), 
                                   id="local-chunk-size-ebook", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input(str(ebook_defaults.get("chunk_overlap", 200)), 
                                   id="local-chunk-overlap-ebook", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(ebook_defaults.get("chunk_language", ""), id="local-chunk-lang-ebook", 
                           placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", 
                              ebook_defaults.get("use_adaptive_chunking", False), 
                              id="local-adaptive-chunking-ebook")
                yield Checkbox("Use Multi-level Chunking", 
                              ebook_defaults.get("use_multi_level_chunking", False), 
                              id="local-multi-level-chunking-ebook")
                yield Label("Custom Chapter Pattern (Regex, optional):")
                yield Input(id="local-custom-chapter-pattern-ebook", placeholder="e.g., ^Chapter\\s+\\d+")
            
            # --- Common Analysis Options ---
            with Collapsible(title="Advanced Analysis Options", collapsed=True,
                             id="local-analysis-opts-collapsible-ebook"):
                yield Checkbox("Summarize Recursively (if chunked)", False, id="local-summarize-recursively-ebook")
                yield Checkbox("Perform Rolling Summarization", False, id="local-perform-rolling-summarization-ebook")
            
            # --- Ebook Specific Options ---
            yield Static("Ebook Specific Options", classes="sidebar-title")
            
            # Check if ebook processing is available
            ebook_processing_available = DEPENDENCIES_AVAILABLE.get('ebook_processing', False)
            
            if ebook_processing_available:
                yield Label("Ebook Extraction Method:")
                ebook_extraction_options = [("filtered", "filtered"), ("markdown", "markdown"), ("basic", "basic")]
                yield Select(ebook_extraction_options, id="local-ebook-extraction-method-ebook", value="filtered")
            else:
                yield Static("⚠️ Ebook processing not available. Install with: pip install tldw_chatbook[ebook]", 
                           classes="warning-message")
                yield Select([("No processing available", Select.BLANK)], id="local-ebook-extraction-method-ebook", disabled=True)
            
            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id="local-overwrite-db-ebook")
            
            # Only enable submit button if ebook processing is available
            yield Button(
                "Process Ebook Locally", 
                id="local-submit-ebook", 
                variant="primary" if ebook_processing_available else "default",
                classes="ingest-submit-button",
                disabled=not ebook_processing_available
            )
            yield LoadingIndicator(id="local-loading-indicator-ebook", classes="hidden")
            yield TextArea(
                "",
                id="local-status-area-ebook",
                read_only=True,
                classes="ingest-status-area hidden"
            )