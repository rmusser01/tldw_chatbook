# tldw_chatbook/Widgets/IngestLocalAudioWindow.py

from typing import TYPE_CHECKING, List, Tuple
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

class IngestLocalAudioWindow(Vertical):
    """Window for ingesting audio content locally."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        self.transcription_service = TranscriptionService()
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
            
            # Get available providers
            available_providers = self.transcription_service.get_available_providers()
            if not available_providers:
                yield Label("No transcription providers available. Please install dependencies.")
            else:
                # Provider selection
                yield Label("Transcription Provider:")
                default_provider = audio_defaults.get("transcription_provider", "faster-whisper")
                if default_provider not in available_providers:
                    default_provider = available_providers[0]
                provider_options = [(p, p) for p in available_providers]
                yield Select(
                    provider_options,
                    id="local-transcription-provider-audio",
                    value=default_provider,
                    prompt="Select transcription provider..."
                )
                
                # Model selection (will be populated based on provider)
                yield Label("Transcription Model:")
                
                # Start with an empty Select widget that will be populated when provider is selected
                yield Select(
                    [],
                    id="local-transcription-model-audio",
                    prompt="Select a provider first...",
                    allow_blank=True
                )
            
            yield Label("Source Language (ISO code):")
            yield Input(
                audio_defaults.get("transcription_language", "en"),
                id="local-transcription-language-audio",
                placeholder="e.g., en, es, fr, de, zh, or 'auto' for detection"
            )
            
            # Translation options (shown for compatible providers)
            with Container(id="local-translation-container-audio", classes="hidden"):
                yield Label("Target Language for Translation (optional):")
                yield Input(
                    "",
                    id="local-translation-target-audio",
                    placeholder="e.g., en (leave empty for no translation)"
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
    
    def on_mount(self) -> None:
        """Called when widget is mounted. Initialize model options for default provider."""
        try:
            # Get the provider select widget
            provider_select = self.query_one("#local-transcription-provider-audio", Select)
            if provider_select.value and provider_select.value != Select.BLANK:
                # Trigger the provider change handler to populate models
                self._update_models_for_provider(str(provider_select.value))
        except Exception as e:
            logger.error(f"Error initializing model options: {e}")
    
    def _update_models_for_provider(self, provider: str) -> None:
        """Update model options for the given provider."""
        logger.debug(f"Updating models for provider: {provider}")
        
        # Get model select widget
        model_select = self.query_one("#local-transcription-model-audio", Select)
        
        # Get available models for the selected provider
        models = self.transcription_service.list_available_models(provider)
        model_list = models.get(provider, [])
        
        logger.debug(f"Available models for {provider}: {model_list}")
        
        # Update model options
        if model_list:
            # Create user-friendly display names for models
            model_options = self._get_model_display_options(provider, model_list)
            model_select.set_options(model_options)
            model_select.prompt = "Select model..."
            # The Select widget will automatically select the first option
            logger.debug(f"Updated model options for {provider}, count: {len(model_options)}")
        else:
            logger.warning(f"No models available for provider {provider}")
            # Clear options when no models available
            model_select.set_options([])
            model_select.prompt = "No models available"
    
    def _get_model_display_options(self, provider: str, model_list: List[str]) -> List[Tuple[str, str]]:
        """Generate user-friendly display names for models based on provider."""
        if provider == 'parakeet-mlx':
            return [(m, "Parakeet TDT 0.6B v2 (Real-time ASR)") for m in model_list]
        elif provider == 'lightning-whisper-mlx':
            # Map Whisper model names to friendly names
            whisper_names = {
                'tiny': 'Tiny (39M params, fastest)',
                'tiny.en': 'Tiny English (39M params)',
                'base': 'Base (74M params)',
                'base.en': 'Base English (74M params)',
                'small': 'Small (244M params)',
                'small.en': 'Small English (244M params)',
                'medium': 'Medium (769M params)',
                'medium.en': 'Medium English (769M params)',
                'large-v1': 'Large v1 (1.5B params)',
                'large-v2': 'Large v2 (1.5B params)',
                'large-v3': 'Large v3 (1.5B params, latest)',
                'large': 'Large (1.5B params)',
                'distil-large-v2': 'Distil Large v2 (faster)',
                'distil-large-v3': 'Distil Large v3 (faster)',
                'distil-medium.en': 'Distil Medium English',
                'distil-small.en': 'Distil Small English'
            }
            return [(m, whisper_names.get(m, m)) for m in model_list]
        elif provider == 'faster-whisper':
            # Similar mapping for faster-whisper
            whisper_names = {
                'tiny': 'Tiny (39M params, fastest)',
                'tiny.en': 'Tiny English (39M params)',
                'base': 'Base (74M params)',
                'base.en': 'Base English (74M params)',
                'small': 'Small (244M params)',
                'small.en': 'Small English (244M params)',
                'medium': 'Medium (769M params)',
                'medium.en': 'Medium English (769M params)',
                'large-v1': 'Large v1 (1.5B params)',
                'large-v2': 'Large v2 (1.5B params)',
                'large-v3': 'Large v3 (1.5B params, latest)',
                'large': 'Large (1.5B params)',
                'distil-large-v2': 'Distil Large v2 (faster)',
                'distil-large-v3': 'Distil Large v3 (faster)',
                'distil-medium.en': 'Distil Medium English',
                'distil-small.en': 'Distil Small English',
                'deepdml/faster-distil-whisper-large-v3.5': 'Distil Large v3.5 (DeepDML)',
                'deepdml/faster-whisper-large-v3-turbo-ct2': 'Large v3 Turbo (DeepDML)',
                'nyrahealth/faster_CrisperWhisper': 'CrisperWhisper (NyraHealth)'
            }
            return [(m, whisper_names.get(m, m)) for m in model_list]
        elif provider == 'qwen2audio':
            return [(m, "Qwen2 Audio 7B Instruct") for m in model_list]
        elif provider == 'parakeet':
            # NVIDIA Parakeet models
            parakeet_names = {
                'nvidia/parakeet-tdt-1.1b': 'Parakeet TDT 1.1B',
                'nvidia/parakeet-rnnt-1.1b': 'Parakeet RNN-T 1.1B',
                'nvidia/parakeet-ctc-1.1b': 'Parakeet CTC 1.1B',
                'nvidia/parakeet-tdt-0.6b': 'Parakeet TDT 0.6B',
                'nvidia/parakeet-rnnt-0.6b': 'Parakeet RNN-T 0.6B',
                'nvidia/parakeet-ctc-0.6b': 'Parakeet CTC 0.6B',
                'nvidia/parakeet-tdt-0.6b-v2': 'Parakeet TDT 0.6B v2'
            }
            return [(m, parakeet_names.get(m, m)) for m in model_list]
        elif provider == 'canary':
            # NVIDIA Canary models
            canary_names = {
                'nvidia/canary-1b-flash': 'Canary 1B Flash (fastest)',
                'nvidia/canary-1b': 'Canary 1B'
            }
            return [(m, canary_names.get(m, m)) for m in model_list]
        else:
            # Default: use model name as-is
            return [(m, m) for m in model_list]
    
    @on(Select.Changed, "#local-transcription-provider-audio")
    def on_provider_changed(self, event: Select.Changed) -> None:
        """Update available models when provider changes."""
        if event.value and event.value != Select.BLANK:
            provider = str(event.value)
            logger.debug(f"Transcription provider changed to: {provider}")
            self._update_models_for_provider(provider)
            
            # Show/hide translation options based on provider
            translation_container = self.query_one("#local-translation-container-audio", Container)
            if provider in ["faster-whisper", "canary"]:
                translation_container.remove_class("hidden")
            else:
                translation_container.add_class("hidden")