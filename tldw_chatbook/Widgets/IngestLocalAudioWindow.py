# tldw_chatbook/Widgets/IngestLocalAudioWindow.py

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
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalAudioWindow(Vertical):
    """Window for ingesting audio content locally."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        logger.debug("IngestLocalAudioWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the audio ingestion form."""
        # Get available API providers for analysis from app config
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]
        
        # Get audio-specific default settings from config
        audio_defaults = get_media_ingestion_defaults("audio")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("Local Audio Processing", classes="sidebar-title")
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- File Selection ---
            yield Label("Media URLs (one per line, e.g., YouTube):")
            yield TextArea(id="local-urls-audio", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id="local-browse-local-files-button-audio")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="local-selected-local-files-list-audio", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="local-title-audio", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="local-author-audio", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="local-keywords-audio", classes="ingest-textarea-small")
            
            # --- Transcription Options ---
            yield Static("Transcription Options", classes="sidebar-title")
            yield Label("Transcription Model:")
            yield Input(
                audio_defaults.get("transcription_model", "base"),
                id="local-transcription-model-audio",
                placeholder="e.g., base, small, medium, large"
            )
            yield Label("Transcription Language (ISO code):")
            yield Input(
                audio_defaults.get("transcription_language", "en"),
                id="local-transcription-language-audio",
                placeholder="e.g., en, es, fr, de, zh"
            )
            yield Checkbox(
                "Enable Voice Activity Detection (VAD)", 
                audio_defaults.get("vad_filter", False), 
                id="local-vad-filter-audio"
            )
            yield Checkbox(
                "Enable Speaker Diarization", 
                audio_defaults.get("diarize", False), 
                id="local-diarize-audio"
            )
            yield Checkbox(
                "Include Timestamps", 
                True, 
                id="local-timestamps-audio"
            )
            
            # --- Analysis Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="local-custom-prompt-audio", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="local-system-prompt-audio", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="local-perform-analysis-audio")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="local-analysis-api-name-audio",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="local-analysis-api-key-audio",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="local-chunking-collapsible-audio"):
                yield Checkbox("Perform Chunking", True, id="local-perform-chunking-audio")
                yield Label("Chunking Method:")
                chunk_method_options = [
                    ("sentences", "sentences"),
                    ("semantic", "semantic"),
                    ("tokens", "tokens"),
                    ("paragraphs", "paragraphs"),
                    ("words", "words")
                ]
                yield Select(chunk_method_options, id="local-chunk-method-audio", 
                            value=audio_defaults.get("chunk_method", "sentences"),
                            prompt="Default (sentences)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input(str(audio_defaults.get("chunk_size", 500)), 
                                   id="local-chunk-size-audio", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input(str(audio_defaults.get("chunk_overlap", 200)), 
                                   id="local-chunk-overlap-audio", type="integer")
                yield Checkbox("Use Adaptive Chunking", False, id="local-use-adaptive-chunking-audio")
                yield Checkbox("Use Multi-level Chunking", False, id="local-use-multi-level-chunking-audio")
                yield Label("Chunk Language (for semantic chunking, e.g., 'en'):")
                yield Input(id="local-chunk-language-audio", placeholder="Auto-detect if empty")
                yield Checkbox("Summarize Recursively", False, id="local-summarize-recursively-audio")
            
            # --- Cookie Options ---
            with Collapsible(title="Cookie Options (for URL downloads)", collapsed=True):
                yield Checkbox("Use Cookies", False, id="local-use-cookies-audio")
                yield Label("Cookies (JSON format):")
                yield TextArea(id="local-cookies-audio", classes="ingest-textarea-small")
            
            # --- Other Options ---
            yield Checkbox("Keep Original Audio Files", True, id="local-keep-original-audio")
            yield Checkbox("Overwrite if exists in database", False, id="local-overwrite-if-exists-audio")
            
            # --- Submit Button ---
            yield Button("Submit", id="local-submit-audio", variant="primary")
            yield LoadingIndicator(id="local-loading-indicator-audio", classes="hidden")
            yield TextArea("", id="local-status-audio", read_only=True, classes="ingest-status-area")