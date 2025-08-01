# tldw_chatbook/Widgets/IngestLocalVideoWindowSimplified.py
# Simplified version of video ingestion with progressive disclosure

from typing import TYPE_CHECKING, List, Tuple
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
from ..Widgets.file_list_item_enhanced import FileListEnhanced

if TYPE_CHECKING:
    from ..app import TldwCli

class IngestLocalVideoWindowSimplified(Vertical):
    """Simplified window for ingesting video content locally with progressive disclosure."""
    
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
        self.simple_mode = get_ingestion_mode_preference("video")
        
        logger.debug("[Video] IngestLocalVideoWindowSimplified initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the simplified video ingestion form."""
        # Get video-specific default settings from config
        video_defaults = get_media_ingestion_defaults("video")
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            # Status dashboard at top
            with Container(id="video-status-dashboard", classes="status-dashboard"):
                yield Label("Ready to process video files", id="video-status-text")
                yield Container(id="video-progress-container", classes="hidden")
            
            # Mode toggle
            with Container(classes="mode-toggle-container"):
                yield Static("Video Processing", classes="sidebar-title")
                with RadioSet(id="video-mode-toggle", classes="mode-toggle"):
                    yield RadioButton("Simple Mode", value=True, id="video-simple-radio")
                    yield RadioButton("Advanced Mode", id="video-advanced-radio")
            
            # Essential fields container (always visible)
            with Container(classes="essential-fields"):
                yield Label("Select Video Files or Enter URLs", classes="form-label-primary")
                
                # File selection
                with Horizontal(classes="file-selection-row"):
                    yield Button("Browse Files", id="local-browse-local-files-button-video", variant="primary")
                    yield Button("Clear All", id="local-clear-files-video", variant="default")
                
                # URL input
                yield Label("Video URLs (one per line):")
                yield TextArea(
                    id="local-urls-video", 
                    classes="ingest-textarea-small"
                )
                
                # Selected files display with metadata
                yield Label("Selected Files:", classes="form-label")
                yield FileListEnhanced(
                    id="local-selected-files-video",
                    show_summary=True,
                    max_height=10
                )
                
                # Basic metadata
                with Horizontal(classes="metadata-row"):
                    with Vertical(classes="metadata-col"):
                        yield Label("Title (Optional):")
                        yield Input(
                            id="local-title-video", 
                            placeholder="Auto-detected from file"
                        )
                    with Vertical(classes="metadata-col"):
                        yield Label("Keywords (Optional):")
                        yield Input(
                            id="local-keywords-video",
                            placeholder="Comma-separated tags"
                        )
                
                # Process button
                yield Button(
                    "Process Videos", 
                    id="local-submit-video", 
                    variant="success",
                    classes="process-button"
                )
            
            # Basic options (visible in simple mode)
            with Container(id="video-basic-options", classes="basic-options-container"):
                yield Checkbox(
                    "Extract audio only (faster processing)", 
                    value=True,
                    id="local-extract-audio-only-video"
                )
                yield Checkbox(
                    "Generate summary", 
                    value=True,
                    id="local-generate-summary-video"
                )
                yield Checkbox(
                    "Include timestamps in transcript", 
                    value=True,
                    id="local-timestamps-video"
                )
            
            # Advanced options (hidden in simple mode)
            with Container(id="video-advanced-options", classes="advanced-options-container hidden"):
                # Transcription settings
                with Collapsible(title="🎙️ Transcription Settings", collapsed=True):
                    # Provider selection
                    yield Label("Transcription Provider:")
                    available_providers = self.transcription_service.get_available_providers()
                    default_provider = video_defaults.get("transcription_provider", "faster-whisper")
                    if default_provider not in available_providers and available_providers:
                        default_provider = available_providers[0]
                    provider_options = [(p, p) for p in available_providers] if available_providers else []
                    
                    yield Select(
                        provider_options,
                        id="local-transcription-provider-video",
                        value=default_provider if provider_options else None,
                        prompt="Select transcription provider..." if provider_options else "No providers available"
                    )
                    
                    # Model selection
                    yield Label("Transcription Model:")
                    yield Select(
                        [],
                        id="local-transcription-model-video",
                        prompt="Select a provider first...",
                        allow_blank=True
                    )
                    
                    yield Label("Source Language:")
                    yield Input(
                        video_defaults.get("transcription_language", "en"),
                        id="local-transcription-language-video",
                        placeholder="e.g., en, es, fr, or 'auto'"
                    )
                    
                    yield Checkbox(
                        "Enable Voice Activity Detection", 
                        video_defaults.get("vad_filter", False),
                        id="local-vad-filter-video"
                    )
                    yield Checkbox(
                        "Enable Speaker Diarization", 
                        video_defaults.get("diarize", False),
                        id="local-diarize-video"
                    )
                
                # Processing options
                with Collapsible(title="⚙️ Processing Options", collapsed=True):
                    # Time range
                    with Horizontal(classes="time-range-row"):
                        with Vertical(classes="time-col"):
                            yield Label("Start Time:")
                            yield Input(id="local-start-time-video", placeholder="HH:MM:SS")
                        with Vertical(classes="time-col"):
                            yield Label("End Time:")
                            yield Input(id="local-end-time-video", placeholder="HH:MM:SS")
                    
                    yield Checkbox(
                        "Download full video (if URL)", 
                        False,
                        id="local-download-video-video"
                    )
                
                # Analysis options
                with Collapsible(title="📊 Analysis Options", collapsed=True):
                    yield Label("Custom Analysis Prompt:")
                    yield TextArea(
                        id="local-custom-prompt-video",
                        classes="ingest-textarea-medium"
                    )
                    
                    yield Label("Analysis Provider:")
                    analysis_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
                    analysis_options = [(name, name) for name in analysis_providers if name]
                    if not analysis_options:
                        analysis_options = [("No Providers Configured", Select.BLANK)]
                    
                    yield Select(
                        analysis_options,
                        id="local-api-name-video",
                        prompt="Select API for Analysis..."
                    )
                
                # Chunking options
                with Collapsible(title="📄 Chunking Options", collapsed=True):
                    yield Checkbox(
                        "Enable chunking", 
                        True,
                        id="local-perform-chunking-video"
                    )
                    
                    with Horizontal(classes="chunk-settings-row"):
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Size:")
                            yield Input("500", id="local-chunk-size-video", type="integer")
                        with Vertical(classes="chunk-col"):
                            yield Label("Chunk Overlap:")
                            yield Input("200", id="local-chunk-overlap-video", type="integer")
            
            # Status area for processing feedback
            yield LoadingIndicator(id="local-loading-indicator-video", classes="hidden")
            yield TextArea(
                "",
                id="local-status-area-video",
                read_only=True,
                classes="ingest-status-area hidden"
            )
    
    def watch_simple_mode(self, simple_mode: bool) -> None:
        """React to mode toggle changes."""
        # Only try to update UI if the widget is mounted
        if not self.is_mounted:
            return
            
        try:
            basic_options = self.query_one("#video-basic-options")
            advanced_options = self.query_one("#video-advanced-options")
            
            if simple_mode:
                basic_options.remove_class("hidden")
                advanced_options.add_class("hidden")
            else:
                basic_options.add_class("hidden")
                advanced_options.remove_class("hidden")
                
            logger.debug(f"Video ingestion mode changed to: {'simple' if simple_mode else 'advanced'}")
        except Exception as e:
            logger.error(f"Error toggling video mode: {e}")
    
    @on(RadioSet.Changed, "#video-mode-toggle")
    def handle_mode_toggle(self, event: RadioSet.Changed) -> None:
        """Handle mode toggle changes."""
        self.simple_mode = event.radio_set.pressed_index == 0
        
        # Save preference
        from ..Utils.ingestion_preferences import save_ingestion_mode_preference
        save_ingestion_mode_preference("video", self.simple_mode)
    
    def _initialize_models(self) -> None:
        """Initialize transcription models in background."""
        try:
            # Check if the element exists before querying
            provider_selects = self.query("#local-transcription-provider-video")
            if not provider_selects:
                logger.debug("Transcription provider select not found - likely in simple mode")
                return
                
            # Get selected provider
            provider_select = provider_selects.first(Select)
            if provider_select and provider_select.value:
                models = self.transcription_service.get_models_for_provider(provider_select.value)
                self._current_model_list = models
                
                # Update model select on main thread
                self.call_from_thread(self._update_model_select, models)
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _update_model_select(self, models: List[str]) -> None:
        """Update model select widget with available models."""
        try:
            model_select = self.query_one("#local-transcription-model-video", Select)
            model_options = [(m, m) for m in models]
            model_select.set_options(model_options)
            
            # Set default model
            default_model = self.get_default_model_for_provider(
                self.query_one("#local-transcription-provider-video", Select).value
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
        self.run_worker(self._initialize_models, exclusive=True, thread=True)
    
    @on(Select.Changed, "#local-transcription-provider-video")
    async def handle_provider_change(self, event: Select.Changed) -> None:
        """Handle transcription provider change."""
        if event.value:
            self.run_worker(self._initialize_models, exclusive=True, thread=True)
    
    @on(Button.Pressed, "#local-browse-local-files-button-video")
    async def handle_browse_files(self, event: Button.Pressed) -> None:
        """Handle file browser button."""
        from ..Widgets.enhanced_file_picker import Filters
        
        filters = Filters(
            ("Video Files", lambda p: p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg")),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select Video Files",
                filters=filters
            ),
            callback=self.handle_file_selection
        )
    
    async def handle_file_selection(self, path: Path | None) -> None:
        """Handle file selection from dialog."""
        if path:
            file_list = self.query_one("#local-selected-files-video", FileListEnhanced)
            file_list.add_file(path)
            self.selected_local_files.append(path)
            
            # Update app instance selected files
            if not hasattr(self.app_instance, 'selected_local_files'):
                self.app_instance.selected_local_files = {}
            
            if 'local_video' not in self.app_instance.selected_local_files:
                self.app_instance.selected_local_files['local_video'] = []
            
            if path not in self.app_instance.selected_local_files['local_video']:
                self.app_instance.selected_local_files['local_video'].append(path)
    
    @on(Button.Pressed, "#local-clear-files-video")
    async def handle_clear_files(self, event: Button.Pressed) -> None:
        """Handle clear files button."""
        file_list = self.query_one("#local-selected-files-video", FileListEnhanced)
        file_list.clear()
        self.selected_local_files.clear()
        
        # Clear app instance files
        if hasattr(self.app_instance, 'selected_local_files') and 'local_video' in self.app_instance.selected_local_files:
            self.app_instance.selected_local_files['local_video'].clear()
    
    @on(Button.Pressed, "#local-submit-video")
    async def handle_submit(self, event: Button.Pressed) -> None:
        """Handle submit button."""
        # Import the actual video processing handler
        from ..Event_Handlers.ingest_events import handle_local_video_process
        
        # Call the real processing function
        await handle_local_video_process(self.app_instance)

# End of IngestLocalVideoWindowSimplified.py