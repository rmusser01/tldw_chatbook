# tldw_chatbook/Widgets/IngestTldwApiVideoWindow.py

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

class IngestTldwApiVideoWindow(Vertical):
    """Window for ingesting video content via tldw API."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        logger.debug("IngestTldwApiVideoWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the video ingestion form."""
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
            yield Input(default_api_url, id="tldw-api-endpoint-url-video", placeholder="http://localhost:8000")
            
            yield Label("Authentication Method:")
            yield Select(
                [
                    ("Token from Config", "config_token"),
                    ("Custom Token", "custom_token"),
                ],
                prompt="Select Auth Method...",
                id="tldw-api-auth-method-video",
                value="config_token"
            )
            yield Label("Custom Auth Token:", id="tldw-api-custom-token-label-video", classes="hidden")
            yield Input(
                "",
                id="tldw-api-custom-token-video",
                placeholder="Enter custom Bearer token",
                password=True,
                classes="hidden",
                tooltip="Enter your Bearer token for the TLDW API. This is used if 'Custom Token' is selected as the authentication method."
            )
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- Common Input Fields ---
            yield Label("Media URLs (one per line):")
            yield TextArea(id="tldw-api-urls-video", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id="tldw-api-browse-local-files-button-video")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="tldw-api-selected-local-files-list-video", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="tldw-api-title-video", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="tldw-api-author-video", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="tldw-api-keywords-video", classes="ingest-textarea-small")
            
            # --- Web Scraping Options (for URLs) ---
            with Collapsible(title="Web Scraping Options", collapsed=True, id="tldw-api-webscraping-collapsible-video"):
                yield Checkbox("Use Cookies for Web Scraping", False, id="tldw-api-use-cookies-video")
                yield Label("Cookies (JSON format):")
                yield TextArea(
                    id="tldw-api-cookies-video", 
                    classes="ingest-textarea-small",
                    tooltip="Paste cookies in JSON format for authenticated web scraping"
                )
            
            # --- Common Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="tldw-api-custom-prompt-video", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="tldw-api-system-prompt-video", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="tldw-api-perform-analysis-video")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="tldw-api-analysis-api-name-video",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="tldw-api-analysis-api-key-video",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="tldw-api-chunking-collapsible-video"):
                yield Checkbox("Perform Chunking", True, id="tldw-api-perform-chunking-video")
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
                yield Select(chunk_method_options, id="tldw-api-chunk-method-video", prompt="Default (per type)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="tldw-api-chunk-size-video", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="tldw-api-chunk-overlap-video", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(id="tldw-api-chunk-lang-video", placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", False, id="tldw-api-adaptive-chunking-video")
                yield Checkbox("Use Multi-level Chunking", False, id="tldw-api-multi-level-chunking-video")
                yield Label("Custom Chapter Pattern (Regex, optional):")
                yield Input(id="tldw-api-custom-chapter-pattern-video", placeholder="e.g., ^Chapter\\s+\\d+")
            
            # --- Common Analysis Options ---
            with Collapsible(title="Advanced Analysis Options", collapsed=True,
                             id="tldw-api-analysis-opts-collapsible-video"):
                yield Checkbox("Summarize Recursively (if chunked)", False, id="tldw-api-summarize-recursively-video")
                yield Checkbox("Perform Rolling Summarization", False, id="tldw-api-perform-rolling-summarization-video")
            
            # --- Video Specific Options ---
            yield Static("Video Specific Options", classes="sidebar-title")
            yield Label("Transcription Model:")
            yield Input("deepdml/faster-whisper-large-v3-turbo-ct2", id="tldw-api-video-transcription-model-video")
            yield Label("Transcription Language (e.g., 'en'):")
            yield Input("en", id="tldw-api-video-transcription-language-video")
            yield Checkbox("Enable Speaker Diarization", False, id="tldw-api-video-diarize-video")
            yield Checkbox("Include Timestamps in Transcription", True, id="tldw-api-video-timestamp-video")
            yield Checkbox("Enable VAD (Voice Activity Detection)", False, id="tldw-api-video-vad-video")
            yield Checkbox("Perform Confabulation Check of Analysis", False, id="tldw-api-video-confab-check-video")
            with Horizontal(classes="ingest-form-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Start Time (HH:MM:SS or secs):")
                    yield Input(id="tldw-api-video-start-time-video", placeholder="Optional")
                with Vertical(classes="ingest-form-col"):
                    yield Label("End Time (HH:MM:SS or secs):")
                    yield Input(id="tldw-api-video-end-time-video", placeholder="Optional")
            
            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id="tldw-api-overwrite-db-video")
            
            yield Button("Submit to TLDW API", id="tldw-api-submit-video", variant="primary", classes="ingest-submit-button")
            
            # --- Cancel Button (hidden by default) ---
            yield Button("Cancel", id="tldw-api-cancel-video", variant="error", classes="ingest-submit-button hidden")
            
            yield LoadingIndicator(id="tldw-api-loading-indicator-video", classes="hidden")
            yield TextArea(
                "",
                id="tldw-api-status-area-video",
                read_only=True,
                classes="ingest-status-area hidden"
            )