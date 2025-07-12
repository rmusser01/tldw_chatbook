# tldw_chatbook/Widgets/IngestLocalVideoWindow.py

from typing import TYPE_CHECKING
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible
)
from textual import on
from ..config import get_media_ingestion_defaults
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..Local_Ingestion.transcription_service import TranscriptionService

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalVideoWindow(Vertical):
    """Window for ingesting video content locally."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        self.transcription_service = TranscriptionService()
        logger.debug("IngestLocalVideoWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the video ingestion form."""
        # Get available API providers for analysis from app config
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]
        
        # Get video-specific default settings from config
        video_defaults = get_media_ingestion_defaults("video")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("Local Video Processing", classes="sidebar-title")
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- File Selection ---
            yield Label("Media URLs (one per line, e.g., YouTube):")
            yield TextArea(id="local-urls-video", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id="local-browse-local-files-button-video")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id="local-selected-local-files-list-video", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="local-title-video", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="local-author-video", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id="local-keywords-video", classes="ingest-textarea-small")
            
            # --- Video Processing Options ---
            yield Static("Video Processing Options", classes="sidebar-title")
            yield Checkbox(
                "Extract Audio Only (faster, no video file kept)", 
                video_defaults.get("extract_audio_only", True), 
                id="local-extract-audio-only-video"
            )
            yield Checkbox(
                "Download Full Video (if URL)", 
                False, 
                id="local-download-video-video"
            )
            
            # Time range options
            with Horizontal(classes="ingest-form-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Start Time (HH:MM:SS or seconds):")
                    yield Input(id="local-start-time-video", placeholder="Optional")
                with Vertical(classes="ingest-form-col"):
                    yield Label("End Time (HH:MM:SS or seconds):")
                    yield Input(id="local-end-time-video", placeholder="Optional")
            
            # --- Transcription Options ---
            yield Static("Transcription Options", classes="sidebar-title")
            
            # Get available providers
            available_providers = self.transcription_service.get_available_providers()
            if not available_providers:
                yield Label("No transcription providers available. Please install dependencies.")
            else:
                # Provider selection
                yield Label("Transcription Provider:")
                default_provider = video_defaults.get("transcription_provider", "faster-whisper")
                if default_provider not in available_providers:
                    default_provider = available_providers[0]
                provider_options = [(p, p) for p in available_providers]
                yield Select(
                    provider_options,
                    id="local-transcription-provider-video",
                    value=default_provider,
                    prompt="Select transcription provider..."
                )
                
                # Model selection (will be populated based on provider)
                yield Label("Transcription Model:")
                models = self.transcription_service.list_available_models(default_provider)
                model_list = models.get(default_provider, [])
                default_model = video_defaults.get("transcription_model", "base")
                if default_model not in model_list and model_list:
                    default_model = model_list[0]
                model_options = [(m, m) for m in model_list] if model_list else [("No models available", "")]
                yield Select(
                    model_options,
                    id="local-transcription-model-video",
                    value=default_model if model_list else "",
                    prompt="Select model..."
                )
            
            yield Label("Source Language (ISO code):")
            yield Input(
                video_defaults.get("transcription_language", "en"),
                id="local-transcription-language-video",
                placeholder="e.g., en, es, fr, de, zh, or 'auto' for detection"
            )
            
            # Translation options (shown for compatible providers)
            with Container(id="local-translation-container-video", classes="hidden"):
                yield Label("Target Language for Translation (optional):")
                yield Input(
                    "",
                    id="local-translation-target-video",
                    placeholder="e.g., en (leave empty for no translation)"
                )
            yield Checkbox(
                "Enable Voice Activity Detection (VAD)", 
                video_defaults.get("vad_filter", False), 
                id="local-vad-filter-video"
            )
            yield Checkbox(
                "Enable Speaker Diarization", 
                video_defaults.get("diarize", False), 
                id="local-diarize-video"
            )
            yield Checkbox(
                "Include Timestamps", 
                True, 
                id="local-timestamps-video"
            )
            
            # --- Analysis Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="local-custom-prompt-video", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="local-system-prompt-video", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="local-perform-analysis-video")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="local-analysis-api-name-video",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id="local-analysis-api-key-video",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="local-chunking-collapsible-video"):
                yield Checkbox("Perform Chunking", True, id="local-perform-chunking-video")
                yield Label("Chunking Method:")
                chunk_method_options = [
                    ("sentences", "sentences"),
                    ("semantic", "semantic"),
                    ("tokens", "tokens"),
                    ("paragraphs", "paragraphs"),
                    ("words", "words")
                ]
                yield Select(chunk_method_options, id="local-chunk-method-video", 
                            value=video_defaults.get("chunk_method", "sentences"),
                            prompt="Default (sentences)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input(str(video_defaults.get("chunk_size", 500)), 
                                   id="local-chunk-size-video", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input(str(video_defaults.get("chunk_overlap", 200)), 
                                   id="local-chunk-overlap-video", type="integer")
                yield Checkbox("Use Adaptive Chunking", False, id="local-use-adaptive-chunking-video")
                yield Checkbox("Use Multi-level Chunking", False, id="local-use-multi-level-chunking-video")
                yield Label("Chunk Language (for semantic chunking, e.g., 'en'):")
                yield Input(id="local-chunk-language-video", placeholder="Auto-detect if empty")
                yield Checkbox("Summarize Recursively", False, id="local-summarize-recursively-video")
            
            # --- Cookie Options ---
            with Collapsible(title="Cookie Options (for URL downloads)", collapsed=True):
                yield Checkbox("Use Cookies", False, id="local-use-cookies-video")
                yield Label("Cookies (JSON format):")
                yield TextArea(id="local-cookies-video", classes="ingest-textarea-small")
            
            # --- Other Options ---
            yield Checkbox("Keep Original Video Files", False, id="local-keep-original-video")
            yield Checkbox("Overwrite if exists in database", False, id="local-overwrite-if-exists-video")
            
            # --- Submit Button ---
            yield Button("Submit", id="local-submit-video", variant="primary")
            yield LoadingIndicator(id="local-loading-indicator-video", classes="hidden")
            yield TextArea("", id="local-status-video", read_only=True, classes="ingest-status-area")
    
    @on(Select.Changed, "#local-transcription-provider-video")
    def on_provider_changed(self, event: Select.Changed) -> None:
        """Update available models when provider changes."""
        provider = str(event.value)
        logger.debug(f"Transcription provider changed to: {provider}")
        
        # Get model select widget
        model_select = self.query_one("#local-transcription-model-video", Select)
        
        # Get available models for the selected provider
        models = self.transcription_service.list_available_models(provider)
        model_list = models.get(provider, [])
        
        # Update model options
        if model_list:
            model_options = [(m, m) for m in model_list]
            model_select.set_options(model_options)
            # Set first model as default
            model_select.value = model_list[0]
        else:
            model_select.set_options([("No models available", "")])
            model_select.value = ""
        
        # Show/hide translation options based on provider
        translation_container = self.query_one("#local-translation-container-video", Container)
        if provider in ["faster-whisper", "canary"]:
            translation_container.remove_class("hidden")
        else:
            translation_container.add_class("hidden")