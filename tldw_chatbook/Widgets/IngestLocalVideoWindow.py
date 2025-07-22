# tldw_chatbook/Widgets/IngestLocalVideoWindow.py

from typing import TYPE_CHECKING, List, Tuple
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible
)
from textual import on, work
from ..config import get_media_ingestion_defaults
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..Widgets.prompt_selector import PromptSelector
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
        self._current_model_list = []  # Store the actual model IDs
        logger.debug("[Video] IngestLocalVideoWindow initialized.")
    
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
                
                # Start with an empty Select widget that will be populated when provider is selected
                yield Select(
                    [],
                    id="local-transcription-model-video",
                    prompt="Select a provider first...",
                    allow_blank=True
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
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="local-perform-analysis-video")
            
            # Prompt selector widget
            yield PromptSelector(
                self.app_instance,
                system_prompt_id="local-system-prompt-video",
                user_prompt_id="local-custom-prompt-video",
                media_type="video",
                id="local-prompt-selector-video"
            )
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
    
    
    def _update_models_for_provider(self, provider: str, model_select: Select) -> None:
        """Update model options for the given provider."""
        logger.debug(f"[Video] Updating models for provider: {provider}")
        
        try:
            # Clear existing options first
            model_select.clear()
            
            # Get available models for the selected provider
            models = self.transcription_service.list_available_models(provider)
            logger.debug(f"[Video] Returned models dict: {models}")
            model_list = models.get(provider, [])
            
            logger.debug(f"[Video] Available models for {provider}: {model_list}")
        
            # Update model options
            if model_list:
                # Store the actual model IDs
                self._current_model_list = model_list
                # Create user-friendly display names for models
                model_options = self._get_model_display_options(provider, model_list)
                # Swap tuple order for Select widget: (value, label) where label is displayed
                select_options = [(model_id, display_name) for model_id, display_name in model_options]
                logger.debug(f"[Video] Setting {len(select_options)} model options for {provider}")
                logger.debug(f"[Video] First option example: value='{select_options[0][0]}', label='{select_options[0][1]}'")
                model_select.set_options(select_options)
                model_select.prompt = "Select model..."
                logger.info(f"[Video] Successfully updated model dropdown with {len(select_options)} models for {provider}")
                if select_options:
                    logger.debug(f"[Video] First few models: {select_options[:3]}")
            else:
                logger.warning(f"[Video] No models available for provider {provider}")
                # Clear options when no models available
                self._current_model_list = []
                model_select.set_options([])
                model_select.prompt = "No models available"
            
        except Exception as e:
            logger.error(f"[Video] Error updating models for provider {provider}: {e}", exc_info=True)
    
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
    
    def get_selected_model_id(self) -> str:
        """Get the actual model ID for the selected model.
        
        Since we now store model IDs as the value in the Select widget,
        we can simply return the selected value.
        """
        model_select = self.query_one("#local-transcription-model-video", Select)
        selected_value = str(model_select.value) if model_select.value else ""
        logger.debug(f"[Video] get_selected_model_id: returning '{selected_value}'")
        return selected_value
    
