# tldw_chatbook/Widgets/IngestTldwApiDocumentWindow.py

from typing import TYPE_CHECKING
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible
)

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

class IngestTldwApiDocumentWindow(Vertical):
    """Window for ingesting document content via tldw API."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        logger.debug("IngestTldwApiDocumentWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the document ingestion form."""
        # Get default API URL from app config
        default_api_url = self.app_instance.app_config.get("tldw_api", {}).get("base_url", "http://127.0.0.1:8000")
        
        # Get available API providers for analysis from app config
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("TLDW API Configuration", classes="sidebar-title")
            yield Label("API Endpoint URL:")
            yield Input(default_api_url, id="tldw-api-endpoint-url-document", placeholder="http://localhost:8000")
            
            yield Label("Authentication Method:")
            yield Select(
                [
                    ("Token from Config", "config_token"),
                    ("Custom Token", "custom_token"),
                ],
                prompt="Select Auth Method...",
                id="tldw-api-auth-method-document",
                value="config_token"
            )
            yield Label("Custom Auth Token:", id="tldw-api-custom-token-label-document", classes="hidden")
            yield Input(
                "",
                id="tldw-api-custom-token-document",
                placeholder="Enter custom Bearer token",
                password=True,
                classes="hidden",
                tooltip="Enter your Bearer token for the TLDW API. This is used if 'Custom Token' is selected as the authentication method."
            )
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- Common Input Fields ---
            yield Label("Media URLs (one per line):")
            yield TextArea(id="tldw-api-urls-document", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id="tldw-api-browse-local-files-button-document")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="tldw-api-selected-local-files-list-document", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="tldw-api-title-document", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="tldw-api-author-document", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="tldw-api-keywords-document", classes="ingest-textarea-small")
            
            # --- Web Scraping Options (for URLs) ---
            with Collapsible(title="Web Scraping Options", collapsed=True, id="tldw-api-webscraping-collapsible-document"):
                yield Checkbox("Use Cookies for Web Scraping", False, id="tldw-api-use-cookies-document")
                yield Label("Cookies (JSON format):")
                yield TextArea(
                    id="tldw-api-cookies-document", 
                    classes="ingest-textarea-small",
                    tooltip="Paste cookies in JSON format for authenticated web scraping"
                )
            
            # --- Common Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="tldw-api-custom-prompt-document", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="tldw-api-system-prompt-document", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="tldw-api-perform-analysis-document")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="tldw-api-analysis-api-name-document",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="tldw-api-analysis-api-key-document",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="tldw-api-chunking-collapsible-document"):
                yield Checkbox("Perform Chunking", True, id="tldw-api-perform-chunking-document")
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
                yield Select(chunk_method_options, id="tldw-api-chunk-method-document", prompt="Default (per type)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="tldw-api-chunk-size-document", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="tldw-api-chunk-overlap-document", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(id="tldw-api-chunk-lang-document", placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", False, id="tldw-api-adaptive-chunking-document")
                yield Checkbox("Use Multi-level Chunking", False, id="tldw-api-multi-level-chunking-document")
                yield Label("Custom Chapter Pattern (Regex, optional):")
                yield Input(id="tldw-api-custom-chapter-pattern-document", placeholder="e.g., ^Chapter\\s+\\d+")
            
            # --- Common Analysis Options ---
            with Collapsible(title="Advanced Analysis Options", collapsed=True,
                             id="tldw-api-analysis-opts-collapsible-document"):
                yield Checkbox("Summarize Recursively (if chunked)", False, id="tldw-api-summarize-recursively-document")
                yield Checkbox("Perform Rolling Summarization", False, id="tldw-api-perform-rolling-summarization-document")
            
            # --- Document Specific Options ---
            yield Static("Document Specific Options", classes="sidebar-title")
            
            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id="tldw-api-overwrite-db-document")
            
            yield Button("Submit to TLDW API", id="tldw-api-submit-document", variant="primary", classes="ingest-submit-button")
            yield LoadingIndicator(id="tldw-api-loading-indicator-document", classes="hidden")
            yield TextArea(
                "",
                id="tldw-api-status-area-document",
                read_only=True,
                classes="ingest-status-area hidden"
            )