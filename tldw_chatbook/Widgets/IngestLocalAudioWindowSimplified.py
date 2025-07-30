# tldw_chatbook/Widgets/IngestLocalAudioWindowSimplified.py
# Simplified version of audio ingestion with progressive disclosure

from typing import TYPE_CHECKING, List
from pathlib import Path
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    ListView, ListItem, LoadingIndicator, Collapsible, RadioSet, RadioButton
)
from textual import on, work
from textual.reactive import reactive
from ..config import get_media_ingestion_defaults
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..Widgets.prompt_selector import PromptSelector
from ..Local_Ingestion.transcription_service import TranscriptionService
from ..Widgets.status_dashboard import StatusDashboard
from ..Widgets.file_list_item_enhanced import FileListEnhanced

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalAudioWindowSimplified(Vertical):
    """Simplified window for ingesting audio content locally with progressive disclosure."""
    
    # Reactive property for simple/advanced mode
    simple_mode = reactive(True)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = []
        self.transcription_service = TranscriptionService()
        self._current_model_list = []
        
        # Load saved preference
        from ..Utils.ingestion_preferences import get_ingestion_mode_preference
        self.simple_mode = get_ingestion_mode_preference("audio")
        
        logger.debug("[Audio] IngestLocalAudioWindowSimplified initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the simplified audio ingestion form."""
        # Get audio-specific default settings from config
        audio_defaults = get_media_ingestion_defaults("audio")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            # Status dashboard at top
            yield StatusDashboard(
                id="audio-status-dashboard",
                show_file_counter=True,
                show_time=True,
                show_actions=True
            )
            
            # Mode toggle
            with Container(classes="mode-toggle-container"):
                yield Static("Audio Processing", classes="sidebar-title")
                with RadioSet(id="audio-mode-toggle", classes="mode-toggle"):
                    yield RadioButton("Simple Mode", value=True, id="audio-simple-radio")
                    yield RadioButton("Advanced Mode", id="audio-advanced-radio")
            
            # Essential fields container (always visible)
            with Container(classes="essential-fields"):
                yield Label("Select Audio Files or Enter URLs", classes="form-label-primary")
                
                # File selection
                with Horizontal(classes="file-selection-row"):
                    yield Button("Browse Files", id="local-browse-local-files-button-audio", variant="primary")
                    yield Button("Clear All", id="local-clear-files-audio", variant="default")
                
                # URL input
                yield TextArea(
                    id="local-urls-audio", 
                    classes="ingest-textarea-small",
                    placeholder="Enter audio URLs (one per line)"
                )
                
                # Selected files display with metadata
                yield Label("Selected Files:", classes="form-label")
                yield FileListEnhanced(
                    id="local-selected-files-audio",
                    show_summary=True,
                    max_height=10
                )
                
                # Basic metadata
                with Horizontal(classes="metadata-row"):
                    with Vertical(classes="metadata-col"):
                        yield Label("Title (Optional):")
                        yield Input(
                            id="local-title-audio", 
                            placeholder="Auto-detected from file"
                        )
                    with Vertical(classes="metadata-col"):
                        yield Label("Keywords (Optional):")
                        yield Input(
                            id="local-keywords-audio",
                            placeholder="Comma-separated tags"
                        )
                
                # Process button
                yield Button(
                    "Process Audio Files", 
                    id="local-submit-audio", 
                    variant="success",
                    classes="process-button"
                )
            
            # Basic options (visible in simple mode)
            with Container(id="audio-basic-options", classes="basic-options-container"):
                yield Checkbox(
                    "Generate summary", 
                    value=True,
                    id="local-generate-summary-audio"
                )
                yield Checkbox(
                    "Include timestamps in transcript", 
                    value=True,
                    id="local-timestamps-audio"
                )
                yield Checkbox(
                    "Auto-detect language", 
                    value=True,
                    id="local-auto-detect-language-audio"
                )
            
            # Advanced options (hidden in simple mode)
            with Container(id="audio-advanced-options", classes="advanced-options-container hidden"):
                # Transcription settings
                with Collapsible(title="🎙️ Transcription Settings", collapsed=True):
                    # Provider selection
                    yield Label("Transcription Provider:")
                    available_providers = self.transcription_service.get_available_providers()
                    default_provider = audio_defaults.get("transcription_provider", "faster-whisper")
                    if default_provider not in available_providers and available_providers:
                        default_provider = available_providers[0]
                    provider_options = [(p, p) for p in available_providers] if available_providers else []
                    
                    yield Select(
                        provider_options,
                        id="local-transcription-provider-audio",
                        value=default_provider if provider_options else None,
                        prompt="Select transcription provider..." if provider_options else "No providers available"
                    )
                    
                    # Model selection
                    yield Label("Transcription Model:")
                    yield Select(
                        [],
                        id="local-transcription-model-audio",
                        prompt="Select a provider first...",
                        allow_blank=True
                    )
                    
                    yield Label("Source Language:")
                    yield Input(
                        audio_defaults.get("transcription_language", "en"),
                        id="local-transcription-language-audio",
                        placeholder="e.g., en, es, fr, or 'auto'"
                    )
                    
                    yield Checkbox(
                        "Enable Voice Activity Detection", 
                        audio_defaults.get("vad_filter", False),
                        id="local-vad-filter-audio"
                    )
                    yield Checkbox(
                        "Enable Speaker Diarization", 
                        audio_defaults.get("diarize", False),
                        id="local-diarize-audio"
                    )
                
                # Analysis options
                with Collapsible(title="📊 Analysis Options", collapsed=True):
                    yield Label("Custom Analysis Prompt:")
                    yield TextArea(
                        id="local-custom-prompt-audio",
                        classes="ingest-textarea-medium",
                        placeholder="Provide specific instructions for analysis..."
                    )
                    
                    yield Label("Analysis Provider:")
                    analysis_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
                    analysis_options = [(name, name) for name in analysis_providers if name]
                    if not analysis_options:
                        analysis_options = [("No Providers Configured", Select.BLANK)]
                    
                    yield Select(
                        analysis_options,
                        id="local-api-name-audio",
                        prompt="Select API for Analysis..."
                    )
                
                # Chunking options
                with Collapsible(title="📄 Chunking Options", collapsed=True):
                    yield Checkbox(
                        "Enable chunking", 
                        True,
                        id="local-perform-chunking-audio"
                    )
                    
                    with Horizontal(classes="chunk-settings-row"):
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Size:")
                            yield Input("500", id="local-chunk-size-audio", type="integer")
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Overlap:")
                            yield Input("200", id="local-chunk-overlap-audio", type="integer")
            
            # Status area for processing feedback
            yield LoadingIndicator(id="local-loading-indicator-audio", classes="hidden")
    
    def watch_simple_mode(self, simple_mode: bool) -> None:
        """React to mode toggle changes."""
        try:
            basic_options = self.query_one("#audio-basic-options")
            advanced_options = self.query_one("#audio-advanced-options")
            
            if simple_mode:
                basic_options.remove_class("hidden")
                advanced_options.add_class("hidden")
            else:
                basic_options.add_class("hidden")
                advanced_options.remove_class("hidden")
                
            logger.debug(f"Audio ingestion mode changed to: {'simple' if simple_mode else 'advanced'}")
        except Exception as e:
            logger.error(f"Error toggling audio mode: {e}")
    
    @on(RadioSet.Changed, "#audio-mode-toggle")
    def handle_mode_toggle(self, event: RadioSet.Changed) -> None:
        """Handle mode toggle changes."""
        self.simple_mode = event.radio_set.pressed_index == 0
        
        # Save preference
        from ..Utils.ingestion_preferences import save_ingestion_mode_preference
        save_ingestion_mode_preference("audio", self.simple_mode)
    
    @work(thread=True)
    async def _initialize_models(self) -> None:
        """Initialize transcription models in background."""
        try:
            # Get selected provider
            provider_select = self.query_one("#local-transcription-provider-audio", Select)
            if provider_select.value:
                models = self.transcription_service.get_models_for_provider(provider_select.value)
                self._current_model_list = models
                
                # Update model select on main thread
                self.call_from_thread(self._update_model_select, models)
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _update_model_select(self, models: List[str]) -> None:
        """Update model select widget with available models."""
        try:
            model_select = self.query_one("#local-transcription-model-audio", Select)
            model_options = [(m, m) for m in models]
            model_select.set_options(model_options)
            
            # Set default model
            default_model = self.get_default_model_for_provider(
                self.query_one("#local-transcription-provider-audio", Select).value
            )
            if default_model in models:
                model_select.value = default_model
        except Exception as e:
            logger.error(f"Error updating model select: {e}")
    
    def get_default_model_for_provider(self, provider: str) -> str:
        """Get default model for a transcription provider."""
        provider_default_models = {
            'parakeet-mlx': 'mlx-community/parakeet-tdt-0.6b-v2',
            'lightning-whisper-mlx': 'base',
            'faster-whisper': 'base',
            'qwen2audio': 'Qwen2-Audio-7B-Instruct',
            'parakeet': 'nvidia/parakeet-tdt-1.1b',
            'canary': 'nvidia/canary-1b-flash'
        }
        return provider_default_models.get(provider, 'base')
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        # Initialize models in background
        self.run_worker(self._initialize_models(), exclusive=True)
    
    @on(Button.Pressed, "#local-browse-local-files-button-audio")
    async def handle_browse_files(self, event: Button.Pressed) -> None:
        """Handle file browser button."""
        filters = Filters(
            ("Audio Files", lambda p: p.suffix.lower() in (".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus", ".aiff")),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select Audio Files",
                filters=filters
            ),
            callback=self.handle_file_selection
        )
    
    async def handle_file_selection(self, path: Path | None) -> None:
        """Handle file selection from dialog."""
        if path:
            file_list = self.query_one("#local-selected-files-audio", FileListEnhanced)
            file_list.add_file(path)
            self.selected_local_files.append(path)
    
    @on(Button.Pressed, "#local-clear-files-audio")
    async def handle_clear_files(self, event: Button.Pressed) -> None:
        """Handle clear files button."""
        file_list = self.query_one("#local-selected-files-audio", FileListEnhanced)
        file_list.clear()
        self.selected_local_files.clear()
    
    @on(Button.Pressed, "#local-submit-audio")
    async def handle_submit(self, event: Button.Pressed) -> None:
        """Handle submit button."""
        # Import the actual audio processing handler
        from ..Event_Handlers.ingest_events import handle_local_audio_process
        
        # Call the real processing function
        await handle_local_audio_process(self.app_instance)

# End of IngestLocalAudioWindowSimplified.py