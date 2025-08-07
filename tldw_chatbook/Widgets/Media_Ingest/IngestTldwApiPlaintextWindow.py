# tldw_chatbook/Widgets/IngestTldwApiPlaintextWindow.py

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

class IngestTldwApiPlaintextWindow(Vertical):
    """Window for ingesting plaintext content via tldw API."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        logger.debug("IngestTldwApiPlaintextWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the plaintext ingestion form."""
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
            yield Input(default_api_url, id="tldw-api-endpoint-url-plaintext", placeholder="http://localhost:8000")
            
            yield Label("Authentication Method:")
            yield Select(
                [
                    ("Token from Config", "config_token"),
                    ("Custom Token", "custom_token"),
                ],
                prompt="Select Auth Method...",
                id="tldw-api-auth-method-plaintext",
                value="config_token"
            )
            yield Label("Custom Auth Token:", id="tldw-api-custom-token-label-plaintext", classes="hidden")
            yield Input(
                "",
                id="tldw-api-custom-token-plaintext",
                placeholder="Enter custom Bearer token",
                password=True,
                classes="hidden",
                tooltip="Enter your Bearer token for the TLDW API. This is used if 'Custom Token' is selected as the authentication method."
            )
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- Common Input Fields ---
            yield Label("Media URLs (one per line):")
            yield TextArea(id="tldw-api-urls-plaintext", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id="tldw-api-browse-local-files-button-plaintext")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="tldw-api-selected-local-files-list-plaintext", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="tldw-api-title-plaintext", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="tldw-api-author-plaintext", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="tldw-api-keywords-plaintext", classes="ingest-textarea-small")
            
            # --- Plaintext Specific Options ---
            yield Static("Plaintext Processing Options", classes="sidebar-title")
            
            yield Label("Text Encoding:")
            yield Select(
                [
                    ("UTF-8", "utf-8"), 
                    ("ASCII", "ascii"), 
                    ("Latin-1", "latin-1"), 
                    ("Auto-detect", "auto")
                ],
                id="tldw-api-encoding-plaintext",
                value="utf-8",
                prompt="Select encoding..."
            )
            
            yield Label("Line Ending:")
            yield Select(
                [
                    ("Auto", "auto"), 
                    ("Unix (LF)", "lf"), 
                    ("Windows (CRLF)", "crlf")
                ],
                id="tldw-api-line-ending-plaintext",
                value="auto",
                prompt="Select line ending..."
            )
            
            yield Checkbox("Remove Extra Whitespace", True, id="tldw-api-remove-whitespace-plaintext")
            yield Checkbox("Convert to Paragraphs", False, id="tldw-api-convert-paragraphs-plaintext")
            
            yield Label("Split Pattern (Regex, optional):")
            yield Input(
                id="tldw-api-split-pattern-plaintext", 
                placeholder="e.g., \\n\\n+ for double newlines",
                tooltip="Regular expression pattern for custom text splitting"
            )
            
            # --- Common Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="tldw-api-custom-prompt-plaintext", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="tldw-api-system-prompt-plaintext", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="tldw-api-perform-analysis-plaintext")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="tldw-api-analysis-api-name-plaintext",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="tldw-api-analysis-api-key-plaintext",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="tldw-api-chunking-collapsible-plaintext"):
                yield Checkbox("Perform Chunking", True, id="tldw-api-perform-chunking-plaintext")
                yield Label("Chunking Method:")
                chunk_method_options = [
                    ("paragraphs", "paragraphs"),
                    ("sentences", "sentences"),
                    ("semantic", "semantic"),
                    ("tokens", "tokens"),
                    ("words", "words"),
                    ("json", "json")
                ]
                yield Select(chunk_method_options, id="tldw-api-chunk-method-plaintext", 
                           value="paragraphs", prompt="Select chunking method...")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="tldw-api-chunk-size-plaintext", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="tldw-api-chunk-overlap-plaintext", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(id="tldw-api-chunk-lang-plaintext", placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", False, id="tldw-api-adaptive-chunking-plaintext")
                yield Checkbox("Use Multi-level Chunking", False, id="tldw-api-multi-level-chunking-plaintext")
                yield Label("Custom Chapter Pattern (Regex, optional):")
                yield Input(id="tldw-api-custom-chapter-pattern-plaintext", placeholder="e.g., ^Chapter\\s+\\d+")
            
            # --- Common Analysis Options ---
            with Collapsible(title="Advanced Analysis Options", collapsed=True,
                             id="tldw-api-analysis-opts-collapsible-plaintext"):
                yield Checkbox("Summarize Recursively (if chunked)", False, id="tldw-api-summarize-recursively-plaintext")
                yield Checkbox("Perform Rolling Summarization", False, id="tldw-api-perform-rolling-summarization-plaintext")
            
            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id="tldw-api-overwrite-db-plaintext")
            
            yield Button("Submit to TLDW API", id="tldw-api-submit-plaintext", variant="primary", classes="ingest-submit-button")
            yield LoadingIndicator(id="tldw-api-loading-indicator-plaintext", classes="hidden")
            yield TextArea(
                "",
                id="tldw-api-status-area-plaintext",
                read_only=True,
                classes="ingest-status-area hidden"
            )