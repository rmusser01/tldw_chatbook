# tldw_chatbook/Widgets/IngestTldwApiMediaWikiWindow.py

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
    from ..app import TldwCli

class IngestTldwApiMediaWikiWindow(Vertical):
    """Window for ingesting MediaWiki dump content via tldw API."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        logger.debug("IngestTldwApiMediaWikiWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the MediaWiki dump ingestion form."""
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
            yield Input(default_api_url, id="tldw-api-endpoint-url-mediawiki_dump", placeholder="http://localhost:8000")
            
            yield Label("Authentication Method:")
            yield Select(
                [
                    ("Token from Config", "config_token"),
                    ("Custom Token", "custom_token"),
                ],
                prompt="Select Auth Method...",
                id="tldw-api-auth-method-mediawiki_dump",
                value="config_token"
            )
            yield Label("Custom Auth Token:", id="tldw-api-custom-token-label-mediawiki_dump", classes="hidden")
            yield Input(
                "",
                id="tldw-api-custom-token-mediawiki_dump",
                placeholder="Enter custom Bearer token",
                password=True,
                classes="hidden",
                tooltip="Enter your Bearer token for the TLDW API. This is used if 'Custom Token' is selected as the authentication method."
            )
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- Common Input Fields ---
            yield Label("Media URLs (one per line):")
            yield TextArea(id="tldw-api-urls-mediawiki_dump", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id="tldw-api-browse-local-files-button-mediawiki_dump")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="tldw-api-selected-local-files-list-mediawiki_dump", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="tldw-api-title-mediawiki_dump", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="tldw-api-author-mediawiki_dump", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="tldw-api-keywords-mediawiki_dump", classes="ingest-textarea-small")
            
            # --- Web Scraping Options (for URLs) ---
            with Collapsible(title="Web Scraping Options", collapsed=True, id="tldw-api-webscraping-collapsible-mediawiki_dump"):
                yield Checkbox("Use Cookies for Web Scraping", False, id="tldw-api-use-cookies-mediawiki_dump")
                yield Label("Cookies (JSON format):")
                yield TextArea(
                    id="tldw-api-cookies-mediawiki_dump", 
                    classes="ingest-textarea-small",
                    tooltip="Paste cookies in JSON format for authenticated web scraping"
                )
            
            # --- Common Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="tldw-api-custom-prompt-mediawiki_dump", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="tldw-api-system-prompt-mediawiki_dump", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="tldw-api-perform-analysis-mediawiki_dump")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="tldw-api-analysis-api-name-mediawiki_dump",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="tldw-api-analysis-api-key-mediawiki_dump",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="tldw-api-chunking-collapsible-mediawiki_dump"):
                yield Checkbox("Perform Chunking", True, id="tldw-api-perform-chunking-mediawiki_dump")
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
                yield Select(chunk_method_options, id="tldw-api-chunk-method-mediawiki_dump", prompt="Default (per type)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="tldw-api-chunk-size-mediawiki_dump", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="tldw-api-chunk-overlap-mediawiki_dump", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(id="tldw-api-chunk-lang-mediawiki_dump", placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", False, id="tldw-api-adaptive-chunking-mediawiki_dump")
                yield Checkbox("Use Multi-level Chunking", False, id="tldw-api-multi-level-chunking-mediawiki_dump")
                yield Label("Custom Chapter Pattern (Regex, optional):")
                yield Input(id="tldw-api-custom-chapter-pattern-mediawiki_dump", placeholder="e.g., ^Chapter\\s+\\d+")
            
            # --- Common Analysis Options ---
            with Collapsible(title="Advanced Analysis Options", collapsed=True,
                             id="tldw-api-analysis-opts-collapsible-mediawiki_dump"):
                yield Checkbox("Summarize Recursively (if chunked)", False, id="tldw-api-summarize-recursively-mediawiki_dump")
                yield Checkbox("Perform Rolling Summarization", False, id="tldw-api-perform-rolling-summarization-mediawiki_dump")
            
            # --- MediaWiki Specific Options ---
            yield Static("MediaWiki Dump Specific Options (Note: Only one local file at a time)", classes="sidebar-title")
            yield Label("Wiki Name (for identification):")
            yield Input(id="tldw-api-mediawiki-wiki-name-mediawiki_dump", placeholder="e.g., my_wiki_backup")
            yield Label("Namespaces (comma-sep IDs, optional):")
            yield Input(id="tldw-api-mediawiki-namespaces-mediawiki_dump", placeholder="e.g., 0,14")
            yield Checkbox("Skip Redirect Pages (recommended)", True, id="tldw-api-mediawiki-skip-redirects-mediawiki_dump")
            yield Label("Chunk Max Size:")
            yield Input("1000", id="tldw-api-mediawiki-chunk-max-size-mediawiki_dump", type="integer")
            yield Label("Vector DB API (optional):")
            yield Input(id="tldw-api-mediawiki-api-name-vector-db-mediawiki_dump", placeholder="For embeddings")
            yield Label("Vector DB API Key (optional):")
            yield Input(id="tldw-api-mediawiki-api-key-vector-db-mediawiki_dump", password=True, placeholder="API key for vector DB")
            
            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id="tldw-api-overwrite-db-mediawiki_dump")
            
            yield Button("Submit to TLDW API", id="tldw-api-submit-mediawiki_dump", variant="primary", classes="ingest-submit-button")
            yield LoadingIndicator(id="tldw-api-loading-indicator-mediawiki_dump", classes="hidden")
            yield TextArea(
                "",
                id="tldw-api-status-area-mediawiki_dump",
                read_only=True,
                classes="ingest-status-area hidden"
            )