# STTS_Window.py
# Description: S/TT/S (Speech/Text-to-Speech) tab with TTS Playground, Settings, and AudioBook/Podcast Generation
#
# Imports
from typing import Optional, Dict, Any, List
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import Label, Button, TextArea, Select, Input, Static, RichLog, Switch, Collapsible, Rule
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding
from loguru import logger

# Local imports
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent, STTSSettingsSaveEvent, STTSAudioBookGenerateEvent
)
from tldw_chatbook.Widgets.voice_blend_dialog import VoiceBlendDialog
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, EnhancedFileSave as FileSave
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.UI.Dictation_Window_Improved import ImprovedDictationWindow as DictationWindow
# Note: Not using form_components due to generator/widget incompatibility

import json

#######################################################################################################################
#
# Classes:

class TTSPlaygroundWidget(Widget):
    """TTS Playground for testing different providers and settings"""
    
    BINDINGS = [
        Binding("ctrl+g", "generate_tts", "Generate Speech"),
        Binding("ctrl+r", "random_text", "Random Text"),
        Binding("ctrl+l", "clear_text", "Clear Text"),
        Binding("ctrl+p", "play_audio", "Play Audio"),
        Binding("ctrl+s", "stop_audio", "Stop Audio"),
    ]
    
    DEFAULT_CSS = """
    TTSPlaygroundWidget {
        height: 100%;
        width: 100%;
    }
    
    .tts-playground-container {
        padding: 1;
        height: 100%;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
    }
    
    #kokoro-language-row {
        display: none;
    }
    
    #kokoro-language-row.visible {
        display: block;
    }
    
    .provider-settings {
        display: none;
    }
    
    #kokoro-settings.visible {
        display: block;
    }
    
    #elevenlabs-settings.visible {
        display: block;
    }
    
    #chatterbox-settings.visible {
        display: block;
    }
    
    #higgs-settings.visible {
        display: block;
    }
    
    .watermark-notice {
        color: $warning;
        margin-top: 1;
    }
    
    .status-text {
        margin: 1 0;
        text-style: italic;
    }
    
    .audio-player {
        height: 10;
        border: solid $primary;
        padding: 1;
        margin-top: 1;
    }
    
    .generation-log {
        height: 20;
        border: solid $secondary;
        margin-top: 1;
    }
    
    .audio-progress {
        width: 100%;
        margin: 0 1;
    }
    
    .audio-time {
        width: auto;
        margin: 0 1;
    }
    
    .hidden {
        display: none;
    }
    
    .generation-status {
        height: 4;
        margin: 1 0;
        border: solid $primary;
        padding: 0 1;
    }
    
    #generation-status-text {
        margin-bottom: 0;
    }
    
    #generation-progress {
        margin-top: 0;
    }
    
    .tts-text-input, #tts-text-input {
        height: 10;
        min-height: 5;
        max-height: 20;
        border: solid $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    .text-input-container {
        height: auto;
        min-height: 12;
        margin-bottom: 1;
    }
    
    .example-text {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .quick-tips {
        border: solid $secondary;
        padding: 1;
        margin: 1 0;
        background: $boost;
    }
    
    .tip-text {
        color: $text-muted;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.current_audio_file = None
        self.reference_audio_path = None
        self.higgs_reference_audio_path = None
        self._progress_timer_task = None
        self._play_worker_task = None
        self.example_texts = [
            "Welcome to the Text-to-Speech playground! This is where you can experiment with different voices, providers, and settings to create natural-sounding speech.",
            "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
            "In a world of artificial intelligence, the ability to convert text into natural speech opens countless possibilities.",
            "Testing, one, two, three. Can you hear the difference between various voice models?",
            "Good morning! Today's weather is sunny with a high of 75 degrees. Perfect for a walk in the park."
        ]
        
    def compose(self) -> ComposeResult:
        """Compose the TTS Playground UI"""
        with ScrollableContainer(classes="tts-playground-container"):
            yield Label("🎤 TTS Playground", classes="section-title")
            
            # Text input area
            with Vertical(classes="text-input-container"):
                yield Label("Text to Synthesize:")
                yield Static(
                    "Example: Hello! Welcome to the TTS Playground. Try different voices and settings.",
                    classes="example-text"
                )
                yield TextArea(
                    "Welcome to the Text-to-Speech playground! This is where you can experiment with different voices, providers, and settings to create natural-sounding speech.",
                    id="tts-text-input",
                    classes="tts-text-input"
                )
            
            # Provider selection
            with Horizontal(classes="form-row"):
                yield Label("Provider:", classes="form-label")
                yield Select(
                    options=[
                        ("openai", "OpenAI"),
                        ("elevenlabs", "ElevenLabs"),
                        ("kokoro", "Kokoro (Local)"),
                        ("chatterbox", "Chatterbox (Local)"),
                        ("higgs", "Higgs Audio (Local)"),
                        ("alltalk", "AllTalk (Local)"),
                    ],
                    id="tts-provider-select"
                )
            
            # Voice selection (will be populated based on provider)
            with Horizontal(classes="form-row"):
                yield Label("Voice:", classes="form-label")
                yield Select(
                    options=[],  # Will be populated on mount
                    id="tts-voice-select"
                )
            
            # Model selection
            with Horizontal(classes="form-row"):
                yield Label("Model:", classes="form-label")
                yield Select(
                    options=[],  # Will be populated on mount
                    id="tts-model-select"
                )
            
            # Language selection (for Kokoro)
            with Horizontal(classes="form-row", id="kokoro-language-row"):
                yield Label("Language:", classes="form-label")
                yield Select(
                    options=[
                        ("en-us", "American English"),
                        ("en-gb", "British English"),
                        ("ja", "Japanese"),
                        ("zh", "Mandarin Chinese"),
                        ("es", "Spanish"),
                        ("fr", "French"),
                        ("hi", "Hindi"),
                        ("it", "Italian"),
                        ("pt-br", "Brazilian Portuguese"),
                    ],
                    id="tts-language-select"
                )
            
            # Kokoro-specific settings
            with Vertical(id="kokoro-settings", classes="provider-settings"):
                # ONNX/PyTorch toggle
                with Horizontal(classes="form-row"):
                    yield Label("Use ONNX:", classes="form-label")
                    yield Switch(
                        id="tts-kokoro-use-onnx",
                        value=get_cli_setting("app_tts", "KOKORO_USE_ONNX", True)
                    )
            
            # Speed control
            with Horizontal(classes="form-row"):
                yield Label("Speed:", classes="form-label")
                yield Input(
                    id="tts-speed-input",
                    value="1.0",
                    placeholder="0.25-4.0",
                    type="number"
                )
            
            # ElevenLabs-specific settings
            with Vertical(id="elevenlabs-settings", classes="provider-settings"):
                with Horizontal(classes="form-row"):
                    yield Label("Voice Stability:", classes="form-label")
                    yield Input(
                        id="tts-stability-input",
                        value="0.5",
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Similarity Boost:", classes="form-label")
                    yield Input(
                        id="tts-similarity-input",
                        value="0.8",
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Style:", classes="form-label")
                    yield Input(
                        id="tts-style-input",
                        value="0.0",
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Speaker Boost:", classes="form-label")
                    yield Switch(id="tts-speaker-boost-switch", value=True)
            
            # Chatterbox-specific settings
            with Vertical(id="chatterbox-settings", classes="provider-settings"):
                with Horizontal(classes="form-row"):
                    yield Label("Exaggeration:", classes="form-label")
                    yield Input(
                        id="tts-exaggeration-input",
                        value="0.5",
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("CFG Weight:", classes="form-label")
                    yield Input(
                        id="tts-cfg-weight-input",
                        value="0.5",
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        id="tts-temperature-input",
                        value="0.5",
                        placeholder="0.0-2.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Candidates:", classes="form-label")
                    yield Input(
                        id="tts-num-candidates-input",
                        value="1",
                        placeholder="1-5",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Whisper Validation:", classes="form-label")
                    yield Switch(id="tts-validate-whisper-switch", value=False)
                
                with Horizontal(classes="form-row"):
                    yield Label("Text Preprocessing:", classes="form-label")
                    yield Switch(id="tts-preprocess-text-switch", value=True)
                
                with Horizontal(classes="form-row"):
                    yield Label("Audio Normalization:", classes="form-label")
                    yield Switch(id="tts-normalize-audio-switch", value=True)
                
                with Horizontal(classes="form-row"):
                    yield Label("Target dB:", classes="form-label")
                    yield Input(
                        id="tts-target-db-input",
                        value="-20.0",
                        placeholder="-30 to -10",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Random Seed:", classes="form-label")
                    yield Input(
                        id="tts-random-seed-input",
                        value="",
                        placeholder="Optional (e.g., 42)",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Reference Audio:", classes="form-label")
                    yield Button("📁 Upload Audio", id="reference-audio-btn", variant="default")
                    yield Button("❌ Clear", id="clear-reference-audio-btn", variant="default", disabled=True)
                
                yield Static("No reference audio selected", id="reference-audio-status", classes="status-text")
                
                # Watermark notice
                yield Static(
                    "⚠️ Note: Generated audio includes watermarking for responsible AI use",
                    classes="watermark-notice"
                )
            
            # Higgs-specific settings
            with Vertical(id="higgs-settings", classes="provider-settings"):
                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        id="tts-higgs-temperature-input",
                        value="0.7",
                        placeholder="0.0-2.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Top P:", classes="form-label")
                    yield Input(
                        id="tts-higgs-top-p-input",
                        value="0.9",
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Repetition Penalty:", classes="form-label")
                    yield Input(
                        id="tts-higgs-repetition-penalty-input",
                        value="1.1",
                        placeholder="1.0+",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Voice Cloning:", classes="form-label")
                    yield Switch(id="tts-higgs-voice-cloning-switch", value=True)
                
                with Horizontal(classes="form-row"):
                    yield Label("Multi-speaker Mode:", classes="form-label")
                    yield Switch(id="tts-higgs-multi-speaker-switch", value=True)
                
                with Horizontal(classes="form-row", id="higgs-voice-upload-row"):
                    yield Label("Voice Reference:", classes="form-label")
                    yield Button("📁 Upload Voice", id="higgs-voice-upload-btn", variant="default")
                    yield Button("❌ Clear", id="higgs-clear-voice-btn", variant="default", disabled=True)
                
                yield Static("No voice reference selected", id="higgs-voice-status", classes="status-text")
                
                with Horizontal(classes="form-row"):
                    yield Label("Speaker Delimiter:", classes="form-label")
                    yield Input(
                        id="tts-higgs-delimiter-input",
                        value="|||",
                        placeholder="Default: |||"
                    )
                
                yield Static(
                    "💡 For multi-speaker: Use format 'Speaker|||Text' in your input",
                    classes="help-text"
                )
            
            # Format selection
            with Horizontal(classes="form-row"):
                yield Label("Format:", classes="form-label")
                yield Select(
                    options=[
                        ("mp3", "MP3"),
                        ("opus", "Opus"),
                        ("aac", "AAC"),
                        ("flac", "FLAC"),
                        ("wav", "WAV"),
                        ("pcm", "PCM"),
                    ],
                    id="tts-format-select"
                )
            
            # Generate button and quick actions
            with Horizontal(classes="form-row"):
                yield Button("🔊 Generate Speech", id="tts-generate-btn", variant="primary")
                yield Button("🎲 Random Text", id="tts-random-text-btn", variant="default")
                yield Button("🗑️ Clear", id="tts-clear-text-btn", variant="default")
            
            # Audio player placeholder
            with Container(id="audio-player-container", classes="audio-player"):
                yield Static("Audio player will appear here after generation", id="audio-player-status")
                
                # Progress bar for playback
                from textual.widgets import ProgressBar
                yield ProgressBar(
                    total=100,
                    show_eta=False,
                    show_percentage=False,
                    id="audio-progress-bar",
                    classes="audio-progress hidden"
                )
                yield Static("0:00 / 0:00", id="audio-time-display", classes="audio-time hidden")
                
                with Horizontal():
                    yield Button("▶️ Play", id="audio-play-btn", disabled=True)
                    yield Button("⏸️ Pause", id="pause-audio-btn", disabled=True)
                    yield Button("⏹️ Stop", id="stop-audio-btn", disabled=True)
                    yield Button("💾 Export", id="audio-export-btn", disabled=True)
            
            # Generation status and progress
            with Container(id="generation-status-container", classes="generation-status hidden"):
                yield Static("Ready to generate", id="generation-status-text")
                yield ProgressBar(id="generation-progress", show_eta=True, show_percentage=True)
            
            # Generation log
            yield Label("Generation Log:")
            yield RichLog(id="tts-generation-log", classes="generation-log", highlight=True, markup=True)
            
            # Keyboard shortcuts info
            yield Rule()
            yield Static(
                "Shortcuts: Ctrl+G=Generate | Ctrl+R=Random | Ctrl+L=Clear | Ctrl+P=Play | Ctrl+S=Stop",
                classes="tip-text"
            )
    
    def on_mount(self) -> None:
        """Initialize default values on mount"""
        # Delay initialization to ensure widgets are ready
        self.set_timer(0.1, self._initialize_defaults)
    
    async def on_unmount(self) -> None:
        """Clean up resources when widget is unmounted"""
        try:
            # Cancel any active progress timer
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                await asyncio.sleep(0.05)
            
            # Cancel any active play worker
            if hasattr(self, '_play_worker_task') and self._play_worker_task and not self._play_worker_task.done():
                self._play_worker_task.cancel()
                await asyncio.sleep(0.05)
            
            # Stop audio playback if active
            if hasattr(self.app, 'audio_player'):
                await self.app.audio_player.stop()
                
            logger.debug("TTSPlaygroundWidget cleanup completed")
        except Exception as e:
            logger.error(f"Error during TTSPlaygroundWidget cleanup: {e}")
    
    def _initialize_defaults(self) -> None:
        """Initialize default values after mount"""
        try:
            # Set default provider
            provider_select = self.query_one("#tts-provider-select", Select)
            default_provider = get_cli_setting("app_tts", "default_provider", "openai")
            
            # Try to set the provider value
            if default_provider in ["openai", "elevenlabs", "kokoro", "chatterbox", "higgs", "alltalk"]:
                try:
                    provider_select.value = default_provider
                    current_provider = default_provider
                except Exception as e:
                    logger.debug(f"Could not set provider immediately: {e}")
                    current_provider = provider_select.value or "openai"
            else:
                current_provider = provider_select.value or "openai"
            
            # If no value selected (shouldn't happen with options), default to openai
            if current_provider is None or current_provider == Select.BLANK:
                current_provider = "openai"
            
            logger.debug(f"Initializing with provider: {current_provider}")
            
            # Always update voice and model options based on current provider
            if current_provider != Select.BLANK:
                self._update_voice_options(current_provider)
                self._update_model_options(current_provider)
            
            # Set default format
            format_select = self.query_one("#tts-format-select", Select)
            default_format = get_cli_setting("app_tts", "default_format", "mp3")
            if default_format in ["mp3", "opus", "aac", "flac", "wav", "pcm"]:
                try:
                    format_select.value = default_format
                except Exception as e:
                    logger.debug(f"Could not set format immediately: {e}")
                    
        except Exception as e:
            logger.warning(f"Error initializing defaults: {e}")
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle provider/model selection changes"""
        if event.select.id == "tts-provider-select":
            # Get the provider select widget
            provider_select = self.query_one("#tts-provider-select", Select)
            
            # The event.value might be the display text, so we need to find the key
            # by matching against the options
            provider_value = None
            for option_value, option_label in [
                ("openai", "OpenAI"),
                ("elevenlabs", "ElevenLabs"),
                ("kokoro", "Kokoro (Local)"),
                ("chatterbox", "Chatterbox (Local)"),
                ("higgs", "Higgs Audio (Local)"),
                ("alltalk", "AllTalk (Local)"),
            ]:
                if event.value == option_label or event.value == option_value:
                    provider_value = option_value
                    break
            
            logger.info(f"Provider select changed - event.value: {event.value!r}")
            logger.info(f"Resolved provider key: {provider_value!r}")
            logger.debug(f"Provider widget value is: {provider_select.value}")
            
            # The widget value should be the key (e.g., "kokoro")
            if provider_value and provider_value != Select.BLANK:
                self._update_voice_options(provider_value)
                self._update_model_options(provider_value)
            
            # Show/hide provider-specific settings
            if provider_value and provider_value != Select.BLANK:
                # Show/hide Kokoro language row
                language_row = self.query_one("#kokoro-language-row", Horizontal)
                if provider_value == "kokoro":
                    logger.debug("Showing Kokoro language selection row")
                    language_row.add_class("visible")
                else:
                    logger.debug("Hiding Kokoro language selection row")
                    language_row.remove_class("visible")
                
                # Show/hide Kokoro settings
                kokoro_settings = self.query_one("#kokoro-settings", Vertical)
                if provider_value == "kokoro":
                    logger.debug("Showing Kokoro settings")
                    kokoro_settings.add_class("visible")
                else:
                    logger.debug("Hiding Kokoro settings")
                    kokoro_settings.remove_class("visible")
                
                # Show/hide ElevenLabs settings
                elevenlabs_settings = self.query_one("#elevenlabs-settings", Vertical)
                if provider_value == "elevenlabs":
                    elevenlabs_settings.add_class("visible")
                else:
                    elevenlabs_settings.remove_class("visible")
                
                # Show/hide Chatterbox settings
                chatterbox_settings = self.query_one("#chatterbox-settings", Vertical)
                if provider_value == "chatterbox":
                    chatterbox_settings.add_class("visible")
                else:
                    chatterbox_settings.remove_class("visible")
                
                # Show/hide Higgs settings
                higgs_settings = self.query_one("#higgs-settings", Vertical)
                if provider_value == "higgs":
                    higgs_settings.add_class("visible")
                    # Check if Higgs is installed
                    self._check_higgs_installation()
                else:
                    higgs_settings.remove_class("visible")
        
        elif event.select.id == "tts-voice-select":
            # Validate voice selection (prevent selecting separators)
            if not self._is_valid_voice(event.value):
                # Find and select the first valid voice
                voice_select = event.select
                for value, _ in voice_select._options:
                    if self._is_valid_voice(value):
                        voice_select.value = value
                        break
    
    def _is_valid_voice(self, voice: str) -> bool:
        """Check if a voice value is valid (not a separator)"""
        return bool(voice) and not str(voice).startswith("_separator")
    
    def _set_valid_voice(self, voice_select, voice_options, fallback="af_bella"):
        """Set a valid voice from options, skipping separators"""
        # Find first valid voice option (skip separators)
        valid_voice = None
        for value, label in voice_options:
            if self._is_valid_voice(value):
                valid_voice = value
                logger.debug(f"Found valid voice: {value} ({label})")
                break
        
        # Try to set the value with error handling
        try:
            if valid_voice:
                logger.info(f"Setting voice to: {valid_voice}")
                voice_select.value = valid_voice
            else:
                logger.warning(f"No valid voice found, using fallback: {fallback}")
                voice_select.value = fallback
            
            # Log final state only if successfully set
            logger.info(f"Voice select final value: {voice_select.value}")
        except Exception as e:
            logger.debug(f"Could not set voice value immediately: {e}")
            # The value will be set later when the widget is ready
    
    def _safe_set_select_value(self, select_widget, value: str, widget_name: str = "widget") -> None:
        """Safely set a Select widget value with error handling"""
        try:
            select_widget.value = value
            logger.debug(f"Successfully set {widget_name} to: {value}")
        except Exception as e:
            logger.debug(f"Could not set {widget_name} value immediately: {e}")
    
    def _get_select_key(self, select_widget) -> Optional[str]:
        """Get the actual key from a Select widget, not the display text"""
        if not hasattr(select_widget, '_options') or not select_widget._options:
            return None
        
        current_value = select_widget.value
        if current_value == Select.BLANK:
            return None
            
        # Find the key that matches the current value
        for key, label in select_widget._options:
            if label == current_value or key == current_value:
                return key
        
        return None
    
    def _update_voice_options(self, provider: str) -> None:
        """Update voice options based on provider"""
        logger.info(f"_update_voice_options called with provider: '{provider}' (type: {type(provider)})")
        
        # Handle Select.BLANK
        if provider == Select.BLANK or str(provider) == "Select.BLANK":
            logger.debug("Provider is Select.BLANK, skipping update")
            return
            
        try:
            voice_select = self.query_one("#tts-voice-select", Select)
            logger.debug(f"Found voice select widget, current value: {voice_select.value}, options: {len(voice_select._options) if hasattr(voice_select, '_options') else 'unknown'}")
        except Exception as e:
            logger.error(f"Failed to find voice select widget: {e}")
            return
        
        if provider == "openai":
            voice_select.set_options([
                ("alloy", "Alloy"),
                ("ash", "Ash"),
                ("ballad", "Ballad"),
                ("coral", "Coral"),
                ("echo", "Echo"),
                ("fable", "Fable"),
                ("onyx", "Onyx"),
                ("nova", "Nova"),
                ("sage", "Sage"),
                ("shimmer", "Shimmer"),
                ("verse", "Verse"),
            ])
            # Don't set value directly - let Select handle it
            try:
                voice_select.value = "alloy"
            except Exception as e:
                logger.debug(f"Could not set default voice value: {e}")
        elif provider == "elevenlabs":
            voice_select.set_options([
                ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
                ("AZnzlk1XvdvUeBnXmlld", "Domi"),
                ("EXAVITQu4vr4xnSDxMaL", "Bella"),
                ("ErXwobaYiN019PkySvjV", "Antoni"),
                ("MF3mGyEYCl7XYWbV9V6O", "Elli"),
                ("TxGEqnHWrfWFTfGW9XjX", "Josh"),
                ("VR6AewLTigWG4xSOukaG", "Arnold"),
                ("pNInz6obpgDQGcFmaJgB", "Adam"),
                ("yoZ06aMxZJJ28mfd3POQ", "Sam"),
            ])
            try:
                voice_select.value = "21m00Tcm4TlvDq8ikWAM"
            except Exception as e:
                logger.debug(f"Could not set default ElevenLabs voice value: {e}")
        elif provider == "kokoro":
            logger.info(f"Setting up Kokoro voices for provider: {provider}")
            logger.debug(f"Voice select widget state - value: {voice_select.value}, enabled: {not voice_select.disabled}")
            voice_options = [
                # American Female voices
                ("af_alloy", "Alloy (US Female)"),
                ("af_aoede", "Aoede (US Female)"),
                ("af_bella", "Bella (US Female)"),
                ("af_heart", "Heart (US Female)"),
                ("af_jessica", "Jessica (US Female)"),
                ("af_kore", "Kore (US Female)"),
                ("af_nicole", "Nicole (US Female)"),
                ("af_nova", "Nova (US Female)"),
                ("af_river", "River (US Female)"),
                ("af_sarah", "Sarah (US Female)"),
                ("af_sky", "Sky (US Female)"),
                # American Male voices
                ("am_adam", "Adam (US Male)"),
                ("am_michael", "Michael (US Male)"),
                # British Female voices
                ("bf_emma", "Emma (UK Female)"),
                ("bf_isabella", "Isabella (UK Female)"),
                # British Male voices
                ("bm_george", "George (UK Male)"),
                ("bm_lewis", "Lewis (UK Male)"),
                # Note: Additional voices for other languages (Japanese, Chinese, Spanish, etc.) 
                # can be added here with proper voice codes
            ]
            
            # Add saved voice blends
            blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            if blend_file.exists():
                try:
                    import json
                    with open(blend_file, 'r') as f:
                        blends = json.load(f)
                        if blends:
                            # Add separator
                            voice_options.append(("_separator", "──── Voice Blends ────"))
                            # Add each blend
                            for blend_name, blend_data in blends.items():
                                display_name = f"🎭 {blend_name}"
                                if blend_data.get('description'):
                                    display_name += f" - {blend_data['description'][:30]}"
                                voice_options.append((f"blend:{blend_name}", display_name))
                except Exception as e:
                    logger.error(f"Failed to load voice blends: {e}")
            
            voice_select.set_options(voice_options)
            logger.debug(f"Set {len(voice_options)} voice options for Kokoro")
            
            # Use helper method to set valid voice
            self._set_valid_voice(voice_select, voice_options, fallback="af_bella")
        elif provider == "chatterbox":
            logger.info(f"Setting up Chatterbox voices for provider: {provider}")
            voice_options = [
                ("default", "Default Voice"),
                ("_separator", "──── Custom Voices ────"),
                ("custom", "Upload Reference Audio"),
            ]
            
            # Add saved voice profiles
            try:
                from tldw_chatbook.TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
                voice_dir = Path.home() / ".config" / "tldw_cli" / "chatterbox_voices"
                if voice_dir.exists():
                    manager = ChatterboxVoiceManager(voice_dir)
                    profiles = manager.list_profiles()
                    if profiles:
                        voice_options.append(("_separator2", "──── Saved Profiles ────"))
                        for profile in profiles:
                            voice_options.append((profile["name"], profile["display_name"]))
                    logger.info(f"Loaded {len(profiles)} Chatterbox voice profiles")
            except Exception as e:
                logger.warning(f"Could not load Chatterbox voice profiles: {e}")
            
            voice_select.set_options(voice_options)
            self._safe_set_select_value(voice_select, "default", "Chatterbox voice")
        elif provider == "higgs":
            logger.info(f"Setting up Higgs voices for provider: {provider}")
            voice_options = [
                # Default Higgs voices
                ("professional_female", "Professional Female"),
                ("warm_female", "Warm Female"),
                ("storyteller_male", "Storyteller Male"),
                ("deep_male", "Deep Male"),
                ("energetic_female", "Energetic Female"),
                ("soft_female", "Soft Female"),
                # Separator for custom voices
                ("_separator", "──── Custom Voices ────"),
                ("custom", "Upload Reference Audio"),
            ]
            
            # Add saved voice profiles
            try:
                from tldw_chatbook.TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager
                voice_dir = Path.home() / ".config" / "tldw_cli" / "higgs_voices"
                if voice_dir.exists():
                    manager = HiggsVoiceProfileManager(voice_dir)
                    profiles = manager.list_profiles()
                    if profiles:
                        # Add separator
                        voice_options.append(("_separator2", "──── Saved Profiles ────"))
                        # Add each profile
                        for profile in profiles:
                            display_name = f"📎 {profile['display_name']}"
                            if profile.get('description'):
                                display_name += f" - {profile['description'][:30]}"
                            voice_options.append((f"profile:{profile['name']}", display_name))
            except Exception as e:
                logger.debug(f"Could not load Higgs voice profiles: {e}")
            
            voice_select.set_options(voice_options)
            self._set_valid_voice(voice_select, voice_options, fallback="professional_female")
        elif provider == "alltalk":
            voice_select.set_options([
                ("female_01.wav", "Female 01"),
                ("female_02.wav", "Female 02"),
                ("female_03.wav", "Female 03"),
                ("female_04.wav", "Female 04"),
                ("male_01.wav", "Male 01"),
                ("male_02.wav", "Male 02"),
                ("male_03.wav", "Male 03"),
                ("male_04.wav", "Male 04"),
                # AllTalk typically supports more voices, these are common defaults
            ])
            self._safe_set_select_value(voice_select, "female_01.wav", "AllTalk voice")
        else:
            logger.warning(f"Unknown TTS provider: {provider}")
            voice_select.set_options([("default", "Default")])
    
    def _update_model_options(self, provider: str) -> None:
        """Update model options based on provider"""
        logger.debug(f"Updating model options for provider: {provider}")
        
        # Handle Select.BLANK
        if provider == Select.BLANK or str(provider) == "Select.BLANK":
            logger.debug("Provider is Select.BLANK, skipping model update")
            return
            
        model_select = self.query_one("#tts-model-select", Select)
        
        if provider == "openai":
            model_select.set_options([
                ("tts-1", "TTS-1 (Standard)"),
                ("tts-1-hd", "TTS-1-HD (High Quality)"),
            ])
            self._safe_set_select_value(model_select, "tts-1", "OpenAI model")
        elif provider == "elevenlabs":
            model_select.set_options([
                ("eleven_monolingual_v1", "Eleven Monolingual v1"),
                ("eleven_multilingual_v1", "Eleven Multilingual v1"),
                ("eleven_multilingual_v2", "Eleven Multilingual v2 (Default)"),
                ("eleven_turbo_v2", "Eleven Turbo v2"),
                ("eleven_turbo_v2_5", "Eleven Turbo v2.5"),
                ("eleven_flash_v2", "Eleven Flash v2 (Low Latency)"),
                ("eleven_flash_v2_5", "Eleven Flash v2.5 (Ultra Low Latency)"),
            ])
            self._safe_set_select_value(model_select, "eleven_multilingual_v2", "ElevenLabs model")
        elif provider == "kokoro":
            logger.info("Setting Kokoro model options")
            model_select.set_options([
                ("kokoro", "Kokoro 82M"),
            ])
            self._safe_set_select_value(model_select, "kokoro", "Kokoro model")
            logger.info("Kokoro model options configured")
        elif provider == "chatterbox":
            model_select.set_options([
                ("chatterbox", "Chatterbox 0.5B"),
            ])
            self._safe_set_select_value(model_select, "chatterbox", "Chatterbox model")
        elif provider == "higgs":
            logger.info("Setting Higgs model options")
            model_select.set_options([
                ("higgs-audio-v2", "Higgs Audio V2 3B"),
            ])
            self._safe_set_select_value(model_select, "higgs-audio-v2", "Higgs model")
            logger.info("Higgs model options configured")
        elif provider == "alltalk":
            model_select.set_options([
                ("alltalk", "AllTalk TTS"),
            ])
            self._safe_set_select_value(model_select, "alltalk", "AllTalk model")
        else:
            logger.warning(f"Unknown TTS provider for model: {provider}")
            model_select.set_options([("default", "Default Model")])
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        logger.debug(f"TTSPlaygroundWidget received button press: {event.button.id}")
        if event.button.id == "tts-generate-btn":
            self._generate_tts()
            event.stop()  # Prevent event from bubbling up
        elif event.button.id == "tts-random-text-btn":
            self._insert_random_text()
            event.stop()
        elif event.button.id == "tts-clear-text-btn":
            self._clear_text()
            event.stop()
        elif event.button.id == "audio-play-btn":
            self._play_audio()
            event.stop()
        elif event.button.id == "pause-audio-btn":
            logger.debug("Pause button clicked")
            self._pause_audio()
            event.stop()
        elif event.button.id == "stop-audio-btn":
            logger.debug("Stop button clicked")
            self._stop_audio()
            event.stop()
        elif event.button.id == "audio-export-btn":
            self._export_audio()
            event.stop()
        elif event.button.id == "reference-audio-btn":
            self._select_reference_audio()
            event.stop()
        elif event.button.id == "clear-reference-audio-btn":
            self._clear_reference_audio()
            event.stop()
        elif event.button.id == "higgs-voice-upload-btn":
            self._upload_higgs_voice()
            event.stop()
        elif event.button.id == "higgs-clear-voice-btn":
            self._clear_higgs_voice()
            event.stop()
    
    def _generate_tts(self) -> None:
        """Generate TTS audio"""
        # Get form values
        text_area = self.query_one("#tts-text-input", TextArea)
        text = text_area.text.strip()
        
        if not text:
            self.app.notify("Please enter text to synthesize", severity="warning")
            return
        
        provider_select = self.query_one("#tts-provider-select", Select)
        voice_select = self.query_one("#tts-voice-select", Select)
        model_select = self.query_one("#tts-model-select", Select)
        
        # Get the actual keys, not display text
        provider = self._get_select_key(provider_select) or provider_select.value
        voice = self._get_select_key(voice_select) or voice_select.value
        model = self._get_select_key(model_select) or model_select.value
        
        # Validate voice selection
        if not self._is_valid_voice(voice):
            self.app.notify("Please select a valid voice", severity="warning")
            return
        speed = float(self.query_one("#tts-speed-input", Input).value or "1.0")
        format_select = self.query_one("#tts-format-select", Select)
        format = format_select.value
        
        # Debug logging
        logger.debug(f"Format select value: {format!r}, type: {type(format)}")
        logger.debug(f"Format select options: {format_select._options}")
        
        # Ensure format has a valid value
        if not format or format == Select.BLANK or str(format) == "Select.BLANK":
            format = "mp3"
            logger.warning("No format selected, defaulting to mp3")
        elif isinstance(format, tuple):
            # If it's a tuple, take the first element
            format = format[0]
            logger.debug(f"Format was tuple, extracted: {format}")
        
        # Additional validation - also handle uppercase
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        format_lower = format.lower() if isinstance(format, str) else format
        if format_lower in valid_formats:
            format = format_lower
        else:
            logger.warning(f"Invalid format '{format}', defaulting to mp3")
            format = "mp3"
        
        # Collect provider-specific settings
        extra_params = {}
        if provider == "kokoro":
            language_select = self.query_one("#tts-language-select", Select)
            language = self._get_select_key(language_select) or language_select.value
            extra_params["language"] = language
            # Add ONNX setting
            use_onnx = self.query_one("#tts-kokoro-use-onnx", Switch).value
            extra_params["use_onnx"] = use_onnx
        elif provider == "elevenlabs":
            stability = float(self.query_one("#tts-stability-input", Input).value or "0.5")
            similarity = float(self.query_one("#tts-similarity-input", Input).value or "0.8")
            style = float(self.query_one("#tts-style-input", Input).value or "0.0")
            speaker_boost = self.query_one("#tts-speaker-boost-switch", Switch).value
            extra_params["stability"] = stability
            extra_params["similarity_boost"] = similarity
            extra_params["style"] = style
            extra_params["use_speaker_boost"] = speaker_boost
        elif provider == "chatterbox":
            exaggeration = float(self.query_one("#tts-exaggeration-input", Input).value or "0.5")
            cfg_weight = float(self.query_one("#tts-cfg-weight-input", Input).value or "0.5")
            temperature = float(self.query_one("#tts-temperature-input", Input).value or "0.5")
            num_candidates = int(self.query_one("#tts-num-candidates-input", Input).value or "1")
            validate_whisper = self.query_one("#tts-validate-whisper-switch", Switch).value
            preprocess_text = self.query_one("#tts-preprocess-text-switch", Switch).value
            normalize_audio = self.query_one("#tts-normalize-audio-switch", Switch).value
            target_db = float(self.query_one("#tts-target-db-input", Input).value or "-20.0")
            random_seed_input = self.query_one("#tts-random-seed-input", Input).value.strip()
            
            extra_params["exaggeration"] = exaggeration
            extra_params["cfg_weight"] = cfg_weight
            extra_params["temperature"] = temperature
            extra_params["num_candidates"] = num_candidates
            extra_params["validate_with_whisper"] = validate_whisper
            extra_params["preprocess_text"] = preprocess_text
            extra_params["normalize_audio"] = normalize_audio
            extra_params["target_db"] = target_db
            if random_seed_input:
                extra_params["random_seed"] = int(random_seed_input)
            
            # Handle voice selection
            if voice == "custom" and self.reference_audio_path:
                # Use custom voice with reference audio
                voice = f"custom:{self.reference_audio_path}"
            elif voice == "custom":
                self.app.notify("Please select reference audio for custom voice", severity="warning")
                self.query_one("#tts-generate-btn", Button).disabled = False
                return
            elif voice not in ["default", "custom", "_separator", "_separator2"] and not voice.startswith("custom:"):
                # This is a saved profile - format it as profile:name
                voice = f"profile:{voice}"
        elif provider == "higgs":
            # Collect Higgs-specific parameters
            temperature = float(self.query_one("#tts-higgs-temperature-input", Input).value)
            top_p = float(self.query_one("#tts-higgs-top-p-input", Input).value)
            repetition_penalty = float(self.query_one("#tts-higgs-repetition-penalty-input", Input).value)
            enable_voice_cloning = self.query_one("#tts-higgs-voice-cloning-switch", Switch).value
            enable_multi_speaker = self.query_one("#tts-higgs-multi-speaker-switch", Switch).value
            speaker_delimiter = self.query_one("#tts-higgs-delimiter-input", Input).value
            
            extra_params["temperature"] = temperature
            extra_params["top_p"] = top_p
            extra_params["repetition_penalty"] = repetition_penalty
            extra_params["enable_voice_cloning"] = enable_voice_cloning
            extra_params["enable_multi_speaker"] = enable_multi_speaker
            extra_params["speaker_delimiter"] = speaker_delimiter
            
            # Handle voice selection for custom upload
            if voice == "custom" and hasattr(self, 'higgs_reference_audio_path') and self.higgs_reference_audio_path:
                # Use custom voice with reference audio
                voice = f"custom:{self.higgs_reference_audio_path}"
            elif voice == "custom":
                self.app.notify("Please upload reference audio for custom voice", severity="warning")
                self.query_one("#tts-generate-btn", Button).disabled = False
                return
            elif voice not in ["professional_female", "warm_female", "storyteller_male", "deep_male", 
                              "energetic_female", "soft_female", "custom", "_separator", "_separator2"] and not voice.startswith("custom:"):
                # This is a saved profile - format it as profile:name
                voice = f"profile:{voice}"
        
        # Log the request
        log = self.query_one("#tts-generation-log", RichLog)
        log.write(f"[bold blue]Generating TTS...[/bold blue]")
        log.write(f"Provider: {provider}")
        log.write(f"Voice: {voice}")
        log.write(f"Model: {model}")
        log.write(f"Speed: {speed}")
        log.write(f"Format: {format}")
        log.write(f"Text length: {len(text)} characters")
        
        # Debug log the actual values
        logger.debug(f"TTS generation - provider: {provider!r}, voice: {voice!r}, model: {model!r}")
        
        # Log provider-specific settings
        if extra_params:
            log.write(f"Extra settings: {extra_params}")
        
        # Disable generate button
        self.query_one("#tts-generate-btn", Button).disabled = True
        
        # Post event to generate TTS
        self.app.post_message(STTSPlaygroundGenerateEvent(
            text=text,
            provider=provider,
            voice=voice,
            model=model,
            speed=speed,
            format=format,
            extra_params=extra_params
        ))
    
    def _generation_complete(self, success: bool, audio_file: Optional[Path] = None) -> None:
        """Handle TTS generation completion"""
        self.query_one("#tts-generate-btn", Button).disabled = False
        
        if success and audio_file:
            # Store the audio file path
            self.current_audio_file = audio_file
            
            log = self.query_one("#tts-generation-log", RichLog)
            log.write("[bold green]✓ TTS generation complete![/bold green]")
            
            # Enable audio controls
            self.query_one("#audio-play-btn", Button).disabled = False
            self.query_one("#pause-audio-btn", Button).disabled = True  # Disabled until playing
            self.query_one("#stop-audio-btn", Button).disabled = True   # Disabled until playing
            self.query_one("#audio-export-btn", Button).disabled = False
            self.query_one("#audio-player-status", Static).update("Audio ready to play")
            
            self.app.notify("TTS generation complete!", severity="information")
        else:
            log = self.query_one("#tts-generation-log", RichLog)
            log.write("[bold red]✗ TTS generation failed![/bold red]")
            self.app.notify("TTS generation failed", severity="error")
    
    def _play_audio(self) -> None:
        """Play the generated audio"""
        logger.debug(f"_play_audio called, current_audio_file: {self.current_audio_file}")
        
        # Check if we're already playing
        if hasattr(self, '_play_worker_task') and self._play_worker_task and not self._play_worker_task.done():
            logger.debug("Play already in progress, ignoring request")
            return
        
        if not self.current_audio_file:
            self.app.notify("No audio file to play", severity="warning")
            return
        
        # Convert to Path if it's a string
        if isinstance(self.current_audio_file, str):
            audio_path = Path(self.current_audio_file)
        else:
            audio_path = self.current_audio_file
            
        logger.debug(f"Audio path: {audio_path}, exists: {audio_path.exists() if audio_path else False}")
        
        if not audio_path.exists():
            self.app.notify(f"Audio file not found: {audio_path.name}", severity="warning")
            return
            
        if self._ensure_audio_player():
            # Cancel any existing progress timer first
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                self._progress_timer_task = None
                logger.debug("Cancelled existing progress timer")
            
            # Enable pause and stop buttons
            self.query_one("#pause-audio-btn", Button).disabled = False
            self.query_one("#stop-audio-btn", Button).disabled = False
            self.query_one("#audio-player-status", Static).update("Playing...")
            
            # Use the new audio player method
            # Store the worker task so we can check if it's running
            self._play_worker_task = self.run_worker(self._play_audio_async, exclusive=False)
        else:
            self.app.notify("Audio playback not available", severity="warning")
    
    async def _play_audio_async(self) -> None:
        """Play audio asynchronously using the audio player"""
        try:
            # Use the stored audio file path - ensure it's a Path object
            if self.current_audio_file:
                # Convert to Path if it's a string
                if isinstance(self.current_audio_file, str):
                    audio_path = Path(self.current_audio_file)
                else:
                    audio_path = self.current_audio_file
                    
                if audio_path.exists():
                    # Get current player state before stopping
                    from tldw_chatbook.TTS.audio_player import PlaybackState
                    current_state = await self.app.audio_player.get_state()
                    logger.debug(f"Current player state before play: {current_state}")
                    
                    # Always force stop any existing playback first
                    stop_result = await self.app.audio_player.stop()
                    logger.debug(f"Stop result: {stop_result}")
                    
                    # Small delay to ensure clean state
                    import asyncio
                    await asyncio.sleep(0.2)
                    
                    # Check state after stop
                    state_after_stop = await self.app.audio_player.get_state()
                    logger.debug(f"Player state after stop: {state_after_stop}")
                    
                    # Attempt to play the audio file
                    logger.info(f"Attempting to play audio file: {audio_path}")
                    success = await self.app.audio_player.play(audio_path)
                    logger.debug(f"Play result: {success}")
                    
                    if success:
                        # Cancel any existing progress timer
                        if self._progress_timer_task and not self._progress_timer_task.done():
                            self._progress_timer_task.cancel()
                            await asyncio.sleep(0.05)  # Small delay to ensure cancellation
                        
                        # Start new progress timer
                        self._progress_timer_task = asyncio.create_task(self._update_progress_timer())
                        logger.debug("Started new progress timer")
                        
                        # Wait a tiny bit to ensure the player has started
                        await asyncio.sleep(0.1)
                        
                        # Double-check the player is actually playing
                        is_playing = await self.app.audio_player.is_playing()
                        logger.debug(f"Player is_playing check after start: {is_playing}")
                        
                        # Clear the worker task reference as it's now running
                        self._play_worker_task = None
                    else:
                        logger.error("Failed to start playback - play() returned False")
                        self.app.notify("Failed to start playback", severity="error")
                        # Reset button states on failure
                        self.query_one("#audio-play-btn", Button).disabled = False
                        self.query_one("#pause-audio-btn", Button).disabled = True
                        self.query_one("#stop-audio-btn", Button).disabled = True
                        self.query_one("#audio-player-status", Static).update("Playback failed")
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
                    self.app.notify(f"Audio file not found: {audio_path.name}", severity="warning")
            else:
                logger.warning("No audio file to play")
                self.app.notify("No audio file to play", severity="warning")
        except Exception as e:
            logger.error(f"Error playing audio: {e}", exc_info=True)
            self.app.notify(f"Playback error: {str(e)}", severity="error")
            # Reset button states on error
            self.query_one("#audio-play-btn", Button).disabled = False
            self.query_one("#pause-audio-btn", Button).disabled = True
            self.query_one("#stop-audio-btn", Button).disabled = True
            self.query_one("#audio-player-status", Static).update("Playback error")
    
    def _ensure_audio_player(self) -> bool:
        """Ensure audio player is initialized (lazy loading)"""
        if not hasattr(self.app, 'audio_player'):
            try:
                from tldw_chatbook.TTS.audio_player import AsyncAudioPlayer
                self.app.audio_player = AsyncAudioPlayer()
                logger.info("Audio player initialized on first use")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize audio player: {e}")
                self.app.notify("Failed to initialize audio player", severity="error")
                return False
        return True
    
    def _pause_audio(self) -> None:
        """Pause audio playback"""
        logger.debug("_pause_audio called")
        if self._ensure_audio_player():
            logger.debug("Audio player available, running pause worker")
            # Don't use exclusive worker for pause - it needs to interrupt play
            self.run_worker(self._pause_audio_async, exclusive=False)
        else:
            logger.debug("Audio player not available")
            self.app.notify("Audio player not available", severity="warning")
    
    async def _pause_audio_async(self) -> None:
        """Pause audio playback asynchronously"""
        try:
            from tldw_chatbook.TTS.audio_player import PlaybackState
            import asyncio
            
            logger.debug("_pause_audio_async called")
            # Small delay to ensure UI is ready
            await asyncio.sleep(0.1)
            
            state = await self.app.audio_player.get_state()
            logger.debug(f"Current playback state: {state}")
            if state == PlaybackState.PLAYING:
                success = await self.app.audio_player.pause()
                if success:
                    # Update button states
                    self.query_one("#pause-audio-btn", Button).label = "▶️ Resume"
                    self.app.notify("Playback paused", severity="information")
                else:
                    self.app.notify("Failed to pause playback", severity="warning")
            elif state == PlaybackState.PAUSED:
                success = await self.app.audio_player.resume()
                if success:
                    # Update button states
                    self.query_one("#pause-audio-btn", Button).label = "⏸️ Pause"
                    self.app.notify("Playback resumed", severity="information")
                    # Cancel any existing timer and restart
                    if self._progress_timer_task and not self._progress_timer_task.done():
                        self._progress_timer_task.cancel()
                    import asyncio
                    self._progress_timer_task = asyncio.create_task(self._update_progress_timer())
                else:
                    self.app.notify("Failed to resume playback", severity="warning")
        except Exception as e:
            logger.error(f"Error toggling pause: {e}")
            from rich.markup import escape
            self.app.notify(f"Error: {escape(str(e))}", severity="error")
    
    def _stop_audio(self) -> None:
        """Stop audio playback"""
        logger.debug("_stop_audio called")
        if self._ensure_audio_player():
            logger.debug("Audio player available, running stop worker")
            # Don't use exclusive worker for stop - it needs to interrupt play
            self.run_worker(self._stop_audio_async, exclusive=False)
        else:
            logger.debug("Audio player not available")
            self.app.notify("Audio player not available", severity="warning")
    
    async def _stop_audio_async(self) -> None:
        """Stop audio playback asynchronously"""
        try:
            logger.debug("_stop_audio_async called")
            # Cancel progress timer if running
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                self._progress_timer_task = None
            
            # Force stop any playback
            success = await self.app.audio_player.stop()
            logger.debug(f"Stop result: {success}")
            
            # Also ensure progress timer is cancelled
            if self._progress_timer_task and not self._progress_timer_task.done():
                self._progress_timer_task.cancel()
                await asyncio.sleep(0.1)  # Give it time to cancel
            
            # Always reset button states regardless of success
            # (audio may have already finished playing)
            self.query_one("#audio-play-btn", Button).disabled = False  # Re-enable play button
            self.query_one("#pause-audio-btn", Button).label = "⏸️ Pause"
            self.query_one("#pause-audio-btn", Button).disabled = True
            self.query_one("#stop-audio-btn", Button).disabled = True
            
            if success:
                self.query_one("#audio-player-status", Static).update("Playback stopped")
                self.app.notify("Playback stopped", severity="information")
            else:
                # Audio already finished or wasn't playing
                self.query_one("#audio-player-status", Static).update("Audio ready to play")
                logger.debug("Audio may have already finished playing")
        except Exception as e:
            logger.error(f"Error stopping playback: {e}")
            from rich.markup import escape
            self.app.notify(f"Error: {escape(str(e))}", severity="error")
    
    def _export_audio(self) -> None:
        """Export the generated audio"""
        if not self.current_audio_file:
            self.app.notify("No audio file to export", severity="warning")
            return
        
        # Create file save dialog
        filters = Filters(
            ("Audio Files", lambda p: p.suffix.lower() in [".mp3", ".wav", ".aac", ".flac", ".opus"]),
            ("All Files", lambda p: True)
        )
        
        # Get original filename and extension
        original_path = Path(self.current_audio_file)
        default_name = f"tts_export_{original_path.stem}{original_path.suffix}"
        
        file_picker = FileSave(
            title="Export Audio File",
            filters=filters,
            default_filename=default_name,
            context="audio_export"
        )
        
        self.app.push_screen(file_picker, self._handle_audio_export)
    
    def _handle_audio_export(self, path: str | None) -> None:
        """Handle audio file export"""
        if not path or not self.current_audio_file:
            return
        
        try:
            import shutil
            source_path = Path(self.current_audio_file)
            dest_path = Path(path)
            
            # If different format requested, we need conversion
            if source_path.suffix.lower() != dest_path.suffix.lower():
                # For now, just copy - format conversion would require audio service
                self.app.notify(
                    f"Format conversion not yet implemented. Exporting as {source_path.suffix}",
                    severity="warning"
                )
                dest_path = dest_path.with_suffix(source_path.suffix)
            
            # Copy the file
            shutil.copy2(source_path, dest_path)
            self.app.notify(f"Audio exported to: {dest_path.name}", severity="success")
            
        except Exception as e:
            logger.error(f"Failed to export audio: {e}")
            self.app.notify(f"Export failed: {str(e)}", severity="error")
    
    def _select_reference_audio(self) -> None:
        """Select reference audio file for voice cloning"""
        # Create file picker for audio files using pre-imported FileOpen
        filters = Filters(
            ("Audio Files", lambda p: p.suffix.lower() in [".wav", ".mp3", ".m4a", ".flac", ".aac"]),
            ("All Files", lambda p: True)
        )
        
        file_picker = FileOpen(
            title="Select Reference Audio",
            filters=filters,
            context="reference_audio"
        )
        
        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_reference_audio_selection)
    
    def _handle_reference_audio_selection(self, path: str | None) -> None:
        """Handle reference audio file selection"""
        if path:
            self.reference_audio_path = path
            # Update status
            status = self.query_one("#reference-audio-status", Static)
            filename = Path(path).name
            status.update(f"Selected: {filename}")
            # Enable clear button
            self.query_one("#clear-reference-audio-btn", Button).disabled = False
            logger.info(f"Reference audio selected: {path}")
        else:
            logger.info("Reference audio selection cancelled")
    
    def _clear_reference_audio(self) -> None:
        """Clear the selected reference audio"""
        self.reference_audio_path = None
        # Update status
        status = self.query_one("#reference-audio-status", Static)
        status.update("No reference audio selected")
        # Disable clear button
        self.query_one("#clear-reference-audio-btn", Button).disabled = True
        logger.info("Reference audio cleared")
    
    def _upload_higgs_voice(self) -> None:
        """Open file dialog to select reference audio for Higgs voice cloning"""
        def handle_selection(path: Optional[Path]) -> None:
            if path:
                self.higgs_reference_audio_path = str(path)
                # Update status
                status = self.query_one("#higgs-voice-status", Static)
                status.update(f"Selected: {path.name}")
                # Enable clear button
                self.query_one("#higgs-clear-voice-btn", Button).disabled = False
                logger.info(f"Higgs reference audio selected: {path}")
            else:
                logger.info("Higgs reference audio selection cancelled")
        
        file_open = FileOpen(
            title="Select Reference Audio for Voice Cloning",
            filters=Filters(
                (
                    "Audio Files",
                    lambda p: p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
                ),
                ("All Files", lambda p: True),
            ),
            must_exist=True
        )
        self.app.push_screen(file_open, handle_selection)
    
    def _clear_higgs_voice(self) -> None:
        """Clear the selected Higgs reference audio"""
        self.higgs_reference_audio_path = None
        # Update status
        status = self.query_one("#higgs-voice-status", Static)
        status.update("No voice reference selected")
        # Disable clear button
        self.query_one("#higgs-clear-voice-btn", Button).disabled = True
        logger.info("Higgs reference audio cleared")
    
    def _check_higgs_installation(self) -> None:
        """Check if Higgs Audio is properly installed"""
        try:
            import boson_multimodal
            logger.info("Higgs Audio is installed and available")
        except ImportError:
            self.app.notify(
                "⚠️ Higgs Audio not installed! Run: ./scripts/install_higgs.sh",
                severity="warning",
                timeout=10
            )
            logger.warning("Higgs Audio (boson_multimodal) is not installed")
    
    def _insert_random_text(self) -> None:
        """Insert a random example text"""
        import random
        text_area = self.query_one("#tts-text-input", TextArea)
        text_area.text = random.choice(self.example_texts)
        text_area.focus()
        self.app.notify("Random example text inserted", severity="information")
    
    def _clear_text(self) -> None:
        """Clear the text input"""
        text_area = self.query_one("#tts-text-input", TextArea)
        text_area.clear()
        text_area.focus()
        self.app.notify("Text cleared", severity="information")
    
    def action_generate_tts(self) -> None:
        """Keyboard shortcut action for generate"""
        self._generate_tts()
    
    def action_random_text(self) -> None:
        """Keyboard shortcut action for random text"""
        self._insert_random_text()
    
    def action_clear_text(self) -> None:
        """Keyboard shortcut action for clear text"""
        self._clear_text()
    
    def action_play_audio(self) -> None:
        """Keyboard shortcut action for play audio"""
        if not self.query_one("#audio-play-btn", Button).disabled:
            self._play_audio()
    
    def action_stop_audio(self) -> None:
        """Keyboard shortcut action for stop audio"""
        if not self.query_one("#stop-audio-btn", Button).disabled:
            self._stop_audio()
    
    async def _update_progress_timer(self) -> None:
        """Update progress bar during playback"""
        import asyncio
        from tldw_chatbook.TTS.audio_player import PlaybackState
        from textual.widgets import ProgressBar
        
        # Ensure audio player exists
        if not hasattr(self.app, 'audio_player'):
            return
            
        while True:
            try:
                state = await self.app.audio_player.get_state()
                if state == PlaybackState.PLAYING:
                    position = await self.app.audio_player.get_position()
                    duration = await self.app.audio_player.get_duration()
                    
                    if duration and duration > 0:
                        # Update progress bar
                        progress_bar = self.query_one("#audio-progress-bar", ProgressBar)
                        progress_bar.update(progress=position, total=duration)
                        
                        # Update time display
                        time_display = self.query_one("#audio-time-display")
                        current_time = self._format_time(position)
                        total_time = self._format_time(duration)
                        time_display.update(f"{current_time} / {total_time}")
                        
                        # Show progress elements
                        progress_bar.remove_class("hidden")
                        time_display.remove_class("hidden")
                elif state in [PlaybackState.IDLE, PlaybackState.FINISHED]:
                    # Hide progress elements
                    self.query_one("#audio-progress-bar").add_class("hidden")
                    self.query_one("#audio-time-display").add_class("hidden")
                    
                    # Reset button states when playback finishes
                    self.query_one("#audio-play-btn", Button).disabled = False
                    self.query_one("#pause-audio-btn", Button).disabled = True
                    self.query_one("#pause-audio-btn", Button).label = "⏸️ Pause"
                    self.query_one("#stop-audio-btn", Button).disabled = True
                    self.query_one("#audio-player-status", Static).update("Playback complete")
                    
                    # Notify that playback is complete
                    if state == PlaybackState.FINISHED:
                        self.app.notify("Playback complete", severity="information")
                    
                    break
                
                await asyncio.sleep(0.1)  # Update every 100ms
            except asyncio.CancelledError:
                logger.debug("Progress timer cancelled")
                break
            except Exception as e:
                logger.error(f"Error updating progress: {e}")
                break
        
        # Ensure UI is reset on exit
        try:
            self.query_one("#audio-play-btn", Button).disabled = False
            self.query_one("#pause-audio-btn", Button).disabled = True
            self.query_one("#stop-audio-btn", Button).disabled = True
            self.query_one("#audio-player-status", Static).update("Ready to play")
        except Exception as e:
            logger.debug(f"Could not reset UI on progress timer exit: {e}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


class TTSSettingsWidget(Widget):
    """TTS Settings for global configuration"""
    
    # Store file paths
    kokoro_model_path = reactive("")
    kokoro_voices_path = reactive("")
    chatterbox_voice_dir = reactive("")
    
    DEFAULT_CSS = """
    TTSSettingsWidget {
        height: 100%;
        width: 100%;
    }
    
    .tts-settings-container {
        padding: 1;
        height: 100%;
    }
    
    .settings-section {
        margin-bottom: 2;
    }
    
    .voice-blends-container {
        height: 5;
        background: $surface;
        border: solid $primary;
        padding: 0 1;
    }
    
    .voice-blends-list {
        padding: 1;
    }
    
    .subsection-label {
        text-style: bold;
        margin: 1 0;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .form-label {
        width: 20;
        height: 1;
        margin-top: 1;
    }
    
    .path-browse-button {
        min-width: 3;
        width: 3;
        height: 3;
        margin-left: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the TTS Settings UI"""
        with ScrollableContainer(classes="tts-settings-container"):
            yield Label("⚙️ TTS Settings", classes="section-title")
            
            # Default provider settings
            with Collapsible(title="Default Provider Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Default Provider:", classes="form-label")
                    yield Select(
                        options=[
                            ("openai", "OpenAI"),
                            ("elevenlabs", "ElevenLabs"),
                            ("kokoro", "Kokoro (Local)"),
                            ("chatterbox", "Chatterbox (Local)"),
                            ("higgs", "Higgs Audio (Local)"),
                            ("alltalk", "AllTalk (Local Server)"),
                        ],
                        id="default-provider-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Default Voice:", classes="form-label")
                    yield Select(
                        options=[("alloy", "Alloy")],  # Will be updated based on provider
                        id="default-voice-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Default Model:", classes="form-label")
                    yield Select(
                        options=[("tts-1", "TTS-1")],  # Will be updated based on provider
                        id="default-model-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Default Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("mp3", "MP3"),
                            ("opus", "Opus"),
                            ("aac", "AAC"),
                            ("flac", "FLAC"),
                            ("wav", "WAV"),
                        ],
                        id="default-format-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Default Speed:", classes="form-label")
                    yield Input(
                        id="default-speed-input",
                        value=str(get_cli_setting("app_tts", "default_speed", 1.0)),
                        placeholder="0.25-4.0",
                        type="number"
                    )
            
            # OpenAI settings
            with Collapsible(title="OpenAI Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("API Key:", classes="form-label")
                    yield Input(
                        id="openai-api-key-input",
                        password=True,
                        placeholder="sk-..."
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Base URL:", classes="form-label")
                    yield Input(
                        id="openai-base-url-input",
                        value=get_cli_setting("app_tts", "OPENAI_BASE_URL", "https://api.openai.com/v1/audio/speech"),
                        placeholder="Custom API endpoint (optional)"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Organization ID:", classes="form-label")
                    yield Input(
                        id="openai-org-id-input",
                        placeholder="org-... (optional)"
                    )
            
            # ElevenLabs settings
            with Collapsible(title="ElevenLabs Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("API Key:", classes="form-label")
                    yield Input(
                        id="elevenlabs-api-key-input",
                        password=True,
                        placeholder="Your ElevenLabs API key"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Model:", classes="form-label")
                    yield Select(
                        options=[
                            ("eleven_multilingual_v2", "Multilingual v2"),
                            ("eleven_turbo_v2", "Turbo v2"),
                            ("eleven_multilingual_v1", "Multilingual v1"),
                            ("eleven_monolingual_v1", "Monolingual v1"),
                        ],
                        id="elevenlabs-model-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Output Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("mp3_44100_192", "MP3 192kbps"),
                            ("mp3_44100_128", "MP3 128kbps"),
                            ("mp3_44100_96", "MP3 96kbps"),
                            ("mp3_44100_64", "MP3 64kbps"),
                            ("mp3_44100_32", "MP3 32kbps"),
                            ("pcm_44100", "PCM 44.1kHz"),
                            ("pcm_24000", "PCM 24kHz"),
                            ("pcm_16000", "PCM 16kHz"),
                            ("ulaw_8000", "μ-law 8kHz"),
                        ],
                        id="elevenlabs-format-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Voice Stability:", classes="form-label")
                    yield Input(
                        id="elevenlabs-stability-input",
                        value=str(get_cli_setting("app_tts", "ELEVENLABS_VOICE_STABILITY", "0.5")),
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Similarity Boost:", classes="form-label")
                    yield Input(
                        id="elevenlabs-similarity-input",
                        value=str(get_cli_setting("app_tts", "ELEVENLABS_SIMILARITY_BOOST", "0.8")),
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Style:", classes="form-label")
                    yield Input(
                        id="elevenlabs-style-input",
                        value=str(get_cli_setting("app_tts", "ELEVENLABS_STYLE", "0.0")),
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Speaker Boost:", classes="form-label")
                    yield Switch(
                        id="elevenlabs-speaker-boost-switch",
                        value=get_cli_setting("app_tts", "ELEVENLABS_USE_SPEAKER_BOOST", True)
                    )
            
            # Kokoro settings
            with Collapsible(title="Kokoro Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Device:", classes="form-label")
                    yield Select(
                        options=[
                            ("cpu", "CPU"),
                            ("cuda", "CUDA (GPU)"),
                        ],
                        id="kokoro-device-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Use ONNX:", classes="form-label")
                    yield Switch(
                        id="kokoro-use-onnx-switch",
                        value=get_cli_setting("app_tts", "KOKORO_USE_ONNX", True)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Model Path:", classes="form-label")
                    yield Button(
                        "📁 Select model file",
                        id="kokoro-browse-model-btn",
                        variant="default"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Voices JSON:", classes="form-label")
                    yield Button(
                        "📁 Select voices.json",
                        id="kokoro-browse-voices-btn",
                        variant="default"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Max Tokens:", classes="form-label")
                    yield Input(
                        id="kokoro-max-tokens-input",
                        value=str(get_cli_setting("app_tts", "KOKORO_MAX_TOKENS", "500")),
                        placeholder="Max tokens per chunk",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Voice Mixing:", classes="form-label")
                    yield Switch(
                        id="kokoro-voice-mixing-switch",
                        value=get_cli_setting("app_tts", "KOKORO_ENABLE_VOICE_MIXING", False)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Performance Tracking:", classes="form-label")
                    yield Switch(
                        id="kokoro-performance-switch",
                        value=get_cli_setting("app_tts", "KOKORO_TRACK_PERFORMANCE", True)
                    )
                
                # Voice blends section
                yield Label("Voice Blends:", classes="form-label")
                with ScrollableContainer(classes="voice-blends-container"):
                    yield Static(id="kokoro-voice-blends-list", classes="voice-blends-list")
                with Horizontal(classes="form-row"):
                    yield Button("➕ Add Blend", id="add-voice-blend-btn", variant="default")
                    yield Button("📥 Import", id="import-blends-btn", variant="default")
                    yield Button("📤 Export", id="export-blends-btn", variant="default")
            
            # Chatterbox settings
            with Collapsible(title="Chatterbox Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Device:", classes="form-label")
                    yield Select(
                        options=[
                            ("cpu", "CPU"),
                            ("cuda", "CUDA (GPU)"),
                        ],
                        id="chatterbox-device-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Voice Directory:", classes="form-label")
                    yield Button(
                        "📁 Select voice directory",
                        id="chatterbox-browse-voice-dir-btn",
                        variant="default"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Emotion Exaggeration:", classes="form-label")
                    yield Input(
                        id="chatterbox-exaggeration-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_EXAGGERATION", "0.5")),
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("CFG Weight:", classes="form-label")
                    yield Input(
                        id="chatterbox-cfg-weight-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_CFG_WEIGHT", "0.5")),
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        id="chatterbox-temperature-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_TEMPERATURE", "0.5")),
                        placeholder="0.0-2.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Chunk Size:", classes="form-label")
                    yield Input(
                        id="chatterbox-chunk-size-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_CHUNK_SIZE", "1024")),
                        placeholder="Audio chunk size",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Random Seed:", classes="form-label")
                    yield Input(
                        id="chatterbox-seed-input",
                        value=get_cli_setting("app_tts", "CHATTERBOX_RANDOM_SEED", ""),
                        placeholder="Random seed (optional)"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Number of Candidates:", classes="form-label")
                    yield Input(
                        id="chatterbox-candidates-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_NUM_CANDIDATES", "1")),
                        placeholder="1-5",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Whisper Validation:", classes="form-label")
                    yield Switch(
                        id="chatterbox-whisper-switch",
                        value=get_cli_setting("app_tts", "CHATTERBOX_VALIDATE_WHISPER", False)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Text Preprocessing:", classes="form-label")
                    yield Switch(
                        id="chatterbox-preprocess-switch",
                        value=get_cli_setting("app_tts", "CHATTERBOX_PREPROCESS_TEXT", True)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Audio Normalization:", classes="form-label")
                    yield Switch(
                        id="chatterbox-normalize-switch",
                        value=get_cli_setting("app_tts", "CHATTERBOX_NORMALIZE_AUDIO", True)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Target dB:", classes="form-label")
                    yield Input(
                        id="chatterbox-target-db-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_TARGET_DB", "-20.0")),
                        placeholder="-40 to 0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Max Text Chunk:", classes="form-label")
                    yield Input(
                        id="chatterbox-max-chunk-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_MAX_CHUNK_SIZE", "500")),
                        placeholder="Max characters per chunk",
                        type="number"
                    )
                
                # Streaming settings subsection
                yield Label("Streaming Settings:", classes="subsection-label")
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Streaming:", classes="form-label")
                    yield Switch(
                        id="chatterbox-streaming-switch",
                        value=get_cli_setting("app_tts", "CHATTERBOX_STREAMING", True)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Stream Chunk Size:", classes="form-label")
                    yield Input(
                        id="chatterbox-stream-chunk-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_STREAM_CHUNK_SIZE", "4096")),
                        placeholder="Stream chunk size",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Crossfade:", classes="form-label")
                    yield Switch(
                        id="chatterbox-crossfade-switch",
                        value=get_cli_setting("app_tts", "CHATTERBOX_ENABLE_CROSSFADE", True)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Crossfade Duration:", classes="form-label")
                    yield Input(
                        id="chatterbox-crossfade-ms-input",
                        value=str(get_cli_setting("app_tts", "CHATTERBOX_CROSSFADE_MS", "50")),
                        placeholder="Duration in ms",
                        type="number"
                    )
            
            # Higgs Audio settings
            with Collapsible(title="Higgs Audio Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Model Path:", classes="form-label")
                    yield Input(
                        id="higgs-model-path-input",
                        value=get_cli_setting("HiggsSettings", "model_path", "bosonai/higgs-audio-v2-generation-3B-base"),
                        placeholder="Model path or HuggingFace ID"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Voice Samples Dir:", classes="form-label")
                    yield Input(
                        id="higgs-voices-dir-input",
                        value=str(get_cli_setting("HiggsSettings", "voice_samples_dir", "~/.config/tldw_cli/higgs_voices")),
                        placeholder="Path to voice samples"
                    )
                    yield Button("📁", id="higgs-voices-browse-btn", classes="path-browse-button")
                
                with Horizontal(classes="form-row"):
                    yield Label("Device:", classes="form-label")
                    yield Select(
                        options=[
                            ("auto", "Auto-detect"),
                            ("cpu", "CPU"),
                            ("cuda", "CUDA (GPU)"),
                            ("cuda:0", "CUDA Device 0"),
                            ("cuda:1", "CUDA Device 1"),
                        ],
                        id="higgs-device-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Flash Attention:", classes="form-label")
                    yield Switch(
                        id="higgs-flash-attn-switch",
                        value=get_cli_setting("HiggsSettings", "enable_flash_attn", True)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Data Type:", classes="form-label")
                    yield Select(
                        options=[
                            ("float32", "Float32 (Full precision)"),
                            ("float16", "Float16 (Half precision)"),
                            ("bfloat16", "BFloat16 (Better range)"),
                        ],
                        id="higgs-dtype-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Max Reference Duration:", classes="form-label")
                    yield Input(
                        id="higgs-max-ref-duration-input",
                        value=str(get_cli_setting("HiggsSettings", "max_reference_duration", "30")),
                        placeholder="Seconds (e.g., 30)",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Default Language:", classes="form-label")
                    yield Select(
                        options=[
                            ("en", "English"),
                            ("es", "Spanish"),
                            ("fr", "French"),
                            ("de", "German"),
                            ("it", "Italian"),
                            ("pt", "Portuguese"),
                            ("ru", "Russian"),
                            ("zh", "Chinese"),
                            ("ja", "Japanese"),
                            ("ko", "Korean"),
                        ],
                        id="higgs-language-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Voice Cloning:", classes="form-label")
                    yield Switch(
                        id="higgs-voice-cloning-switch",
                        value=get_cli_setting("HiggsSettings", "enable_voice_cloning", True)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Multi-speaker:", classes="form-label")
                    yield Switch(
                        id="higgs-multi-speaker-switch",
                        value=get_cli_setting("HiggsSettings", "enable_multi_speaker", True)
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Speaker Delimiter:", classes="form-label")
                    yield Input(
                        id="higgs-delimiter-input",
                        value=get_cli_setting("HiggsSettings", "speaker_delimiter", "|||"),
                        placeholder="Default: |||"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Performance Tracking:", classes="form-label")
                    yield Switch(
                        id="higgs-track-performance-switch",
                        value=get_cli_setting("HiggsSettings", "track_performance", True)
                    )
                
                # Generation parameters
                yield Label("Generation Parameters:", classes="subsection-label")
                
                with Horizontal(classes="form-row"):
                    yield Label("Max New Tokens:", classes="form-label")
                    yield Input(
                        id="higgs-max-tokens-input",
                        value=str(get_cli_setting("HiggsSettings", "max_new_tokens", "4096")),
                        placeholder="Max tokens to generate",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        id="higgs-temperature-input",
                        value=str(get_cli_setting("HiggsSettings", "temperature", "0.7")),
                        placeholder="0.0-2.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Top P:", classes="form-label")
                    yield Input(
                        id="higgs-top-p-input",
                        value=str(get_cli_setting("HiggsSettings", "top_p", "0.9")),
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Repetition Penalty:", classes="form-label")
                    yield Input(
                        id="higgs-repetition-penalty-input",
                        value=str(get_cli_setting("HiggsSettings", "repetition_penalty", "1.1")),
                        placeholder="1.0 = no penalty",
                        type="number"
                    )
            
            # AllTalk settings
            with Collapsible(title="AllTalk Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Server URL:", classes="form-label")
                    yield Input(
                        id="alltalk-url-input",
                        value=get_cli_setting("app_tts", "ALLTALK_TTS_URL_DEFAULT", "http://127.0.0.1:7851"),
                        placeholder="AllTalk server URL"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Voice:", classes="form-label")
                    yield Input(
                        id="alltalk-voice-input",
                        value=get_cli_setting("app_tts", "ALLTALK_TTS_VOICE_DEFAULT", "female_01.wav"),
                        placeholder="Voice file name"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Language:", classes="form-label")
                    yield Select(
                        options=[
                            ("en", "English"),
                            ("es", "Spanish"),
                            ("fr", "French"),
                            ("de", "German"),
                            ("it", "Italian"),
                            ("pt", "Portuguese"),
                            ("ru", "Russian"),
                            ("zh", "Chinese"),
                            ("ja", "Japanese"),
                            ("ko", "Korean"),
                        ],
                        id="alltalk-language-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Output Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("wav", "WAV"),
                            ("mp3", "MP3"),
                            ("opus", "Opus"),
                            ("flac", "FLAC"),
                        ],
                        id="alltalk-format-select"
                    )
            
            # Save button
            yield Button("💾 Save Settings", id="save-settings-btn", variant="primary")
    
    def on_mount(self) -> None:
        """Set initial values from config after mount"""
        try:
            # Set default provider
            provider_select = self.query_one("#default-provider-select", Select)
            default_provider = get_cli_setting("app_tts", "default_provider", "openai")
            if default_provider in ["openai", "elevenlabs", "kokoro", "chatterbox", "higgs", "alltalk"]:
                provider_select.value = default_provider
            
            # Load voice blends
            self._load_kokoro_voice_blends()
            
            # Load and display file paths
            self.kokoro_model_path = get_cli_setting("app_tts", "KOKORO_ONNX_MODEL_PATH_DEFAULT", "")
            self.kokoro_voices_path = get_cli_setting("app_tts", "KOKORO_ONNX_VOICES_JSON_DEFAULT", "")
            self.chatterbox_voice_dir = get_cli_setting("app_tts", "CHATTERBOX_VOICE_DIR", "~/.config/tldw_cli/chatterbox_voices")
            
            # Update button labels
            self._update_file_button_labels()
            
            # Update voice and model options based on default provider
            self._update_default_voice_options(default_provider)
            self._update_default_model_options(default_provider)
            
            # Set default voice and model
            default_voice = get_cli_setting("app_tts", "default_voice", "alloy")
            default_model = get_cli_setting("app_tts", "default_model", "tts-1")
            
            voice_select = self.query_one("#default-voice-select", Select)
            model_select = self.query_one("#default-model-select", Select)
            
            # Try to set the values if they exist in options
            try:
                if any(opt[0] == default_voice for opt in voice_select._options):
                    voice_select.value = default_voice
            except:
                pass
                
            try:
                if any(opt[0] == default_model for opt in model_select._options):
                    model_select.value = default_model
            except:
                pass
            
            # Set default format
            format_select = self.query_one("#default-format-select", Select)
            default_format = get_cli_setting("app_tts", "default_format", "mp3")
            if default_format in ["mp3", "opus", "aac", "flac", "wav"]:
                format_select.value = default_format
            
            # Set Kokoro device
            try:
                device_select = self.query_one("#kokoro-device-select", Select)
                kokoro_device = get_cli_setting("app_tts", "KOKORO_DEVICE_DEFAULT", "cpu")
                if kokoro_device in ["cpu", "cuda"]:
                    device_select.value = kokoro_device
            except Exception as e:
                logger.debug(f"Could not set Kokoro device: {e}")
            
            # Set Chatterbox device
            try:
                chatterbox_device_select = self.query_one("#chatterbox-device-select", Select)
                chatterbox_device = get_cli_setting("app_tts", "CHATTERBOX_DEVICE", "cpu")
                if chatterbox_device in ["cpu", "cuda"]:
                    chatterbox_device_select.value = chatterbox_device
            except Exception as e:
                logger.debug(f"Could not set Chatterbox device: {e}")
            
            # Set ElevenLabs model
            elevenlabs_model_select = self.query_one("#elevenlabs-model-select", Select)
            elevenlabs_model = get_cli_setting("app_tts", "ELEVENLABS_DEFAULT_MODEL", "eleven_multilingual_v2")
            if elevenlabs_model in ["eleven_multilingual_v2", "eleven_turbo_v2", "eleven_multilingual_v1", "eleven_monolingual_v1"]:
                elevenlabs_model_select.value = elevenlabs_model
            
            # Set ElevenLabs format
            elevenlabs_format_select = self.query_one("#elevenlabs-format-select", Select)
            elevenlabs_format = get_cli_setting("app_tts", "ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_192")
            if elevenlabs_format in ["mp3_44100_192", "mp3_44100_128", "mp3_44100_96", "mp3_44100_64", "mp3_44100_32", "pcm_44100", "pcm_24000", "pcm_16000", "ulaw_8000"]:
                elevenlabs_format_select.value = elevenlabs_format
            
            # Set AllTalk language
            alltalk_language_select = self.query_one("#alltalk-language-select", Select)
            alltalk_language = get_cli_setting("app_tts", "ALLTALK_TTS_LANGUAGE_DEFAULT", "en")
            if alltalk_language in ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]:
                alltalk_language_select.value = alltalk_language
            
            # Set AllTalk format
            alltalk_format_select = self.query_one("#alltalk-format-select", Select)
            alltalk_format = get_cli_setting("app_tts", "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT", "wav")
            if alltalk_format in ["wav", "mp3", "opus", "flac"]:
                alltalk_format_select.value = alltalk_format
            
            # Set Higgs settings - Select widgets are already initialized in compose(),
            # but we ensure they have the correct values here
            try:
                # Device
                higgs_device_select = self.query_one("#higgs-device-select", Select)
                higgs_device = get_cli_setting("HiggsSettings", "device", "auto")
                if higgs_device in ["auto", "cpu", "cuda", "cuda:0", "cuda:1"]:
                    higgs_device_select.value = higgs_device
                
                # Data type
                higgs_dtype_select = self.query_one("#higgs-dtype-select", Select)
                higgs_dtype = get_cli_setting("HiggsSettings", "dtype", "bfloat16")
                if higgs_dtype in ["float32", "float16", "bfloat16"]:
                    higgs_dtype_select.value = higgs_dtype
                
                # Language
                higgs_language_select = self.query_one("#higgs-language-select", Select)
                higgs_language = get_cli_setting("HiggsSettings", "default_language", "en")
                if higgs_language in ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]:
                    higgs_language_select.value = higgs_language
            except Exception as e:
                logger.debug(f"Could not set Higgs initial values: {e}")
            
            # Load and display Kokoro voice blends
            self._load_kokoro_voice_blends()
            
        except Exception as e:
            logger.warning(f"Failed to set initial values: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "save-settings-btn":
            self._save_settings()
            event.stop()  # Prevent event from bubbling up
        elif event.button.id == "add-voice-blend-btn":
            self.run_worker(self._show_add_voice_blend_dialog)
            event.stop()
        elif event.button.id == "import-blends-btn":
            self._import_voice_blends()
            event.stop()
        elif event.button.id == "export-blends-btn":
            self._export_voice_blends()
            event.stop()
        elif event.button.id == "kokoro-browse-model-btn":
            self._browse_kokoro_model()
            event.stop()
        elif event.button.id == "kokoro-browse-voices-btn":
            self._browse_kokoro_voices()
            event.stop()
        elif event.button.id == "chatterbox-browse-voice-dir-btn":
            self._browse_chatterbox_voice_dir()
            event.stop()
        elif event.button.id == "higgs-voices-browse-btn":
            self._browse_higgs_voices_dir()
            event.stop()
    
    def _is_valid_voice(self, voice: str) -> bool:
        """Check if a voice value is valid (not a separator)"""
        return bool(voice) and not str(voice).startswith("_separator")
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes"""
        if event.select.id == "default-provider-select":
            # Update voice and model options when provider changes
            self._update_default_voice_options(event.value)
            self._update_default_model_options(event.value)
        elif event.select.id == "default-voice-select":
            # Validate voice selection (prevent selecting separators)
            if not self._is_valid_voice(event.value):
                # Find and select the first valid voice
                voice_select = event.select
                for value, _ in voice_select._options:
                    if self._is_valid_voice(value):
                        voice_select.value = value
                        break
    
    def _update_default_voice_options(self, provider: str) -> None:
        """Update default voice options based on provider"""
        voice_select = self.query_one("#default-voice-select", Select)
        
        if provider == "openai":
            voice_select.set_options([
                ("alloy", "Alloy"),
                ("ash", "Ash"),
                ("ballad", "Ballad"),
                ("coral", "Coral"),
                ("echo", "Echo"),
                ("fable", "Fable"),
                ("nova", "Nova"),
                ("onyx", "Onyx"),
                ("sage", "Sage"),
                ("shimmer", "Shimmer"),
                ("verse", "Verse"),
            ])
            # Set the saved default or fallback to alloy
            default_voice = get_cli_setting("app_tts", "default_voice", "alloy")
            if default_voice in [v[0] for v in voice_select._options if v[0] != Select.BLANK]:
                voice_select.value = default_voice
            else:
                voice_select.value = "alloy"
        elif provider == "elevenlabs":
            voice_select.set_options([
                ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
                ("AZnzlk1XvdvUeBnXmlld", "Domi"),
                ("EXAVITQu4vr4xnSDxMaL", "Bella"),
                ("ErXwobaYiN019PkySvjV", "Antoni"),
                ("MF3mGyEYCl7XYWbV9V6O", "Elli"),
                ("TxGEqnHWrfWFTfGW9XjX", "Josh"),
                ("VR6AewLTigWG4xSOukaG", "Arnold"),
                ("pNInz6obpgDQGcFmaJgB", "Adam"),
                ("yoZ06aMxZJJ28mfd3POQ", "Sam"),
            ])
            try:
                voice_select.value = "21m00Tcm4TlvDq8ikWAM"
            except Exception as e:
                logger.debug(f"Could not set default ElevenLabs voice value: {e}")
        elif provider == "kokoro":
            logger.info(f"Setting up Kokoro voices for provider: {provider}")
            voice_options = [
                # American Female voices
                ("af_alloy", "Alloy (US Female)"),
                ("af_aoede", "Aoede (US Female)"),
                ("af_bella", "Bella (US Female)"),
                ("af_heart", "Heart (US Female)"),
                ("af_jessica", "Jessica (US Female)"),
                ("af_kore", "Kore (US Female)"),
                ("af_nicole", "Nicole (US Female)"),
                ("af_nova", "Nova (US Female)"),
                ("af_river", "River (US Female)"),
                ("af_sarah", "Sarah (US Female)"),
                ("af_sky", "Sky (US Female)"),
                # American Male voices
                ("am_adam", "Adam (US Male)"),
                ("am_michael", "Michael (US Male)"),
                # British Female voices
                ("bf_emma", "Emma (UK Female)"),
                ("bf_isabella", "Isabella (UK Female)"),
                # British Male voices
                ("bm_george", "George (UK Male)"),
                ("bm_lewis", "Lewis (UK Male)"),
            ]
            
            # Add saved voice blends
            blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            if blend_file.exists():
                try:
                    import json
                    with open(blend_file, 'r') as f:
                        blends = json.load(f)
                        if blends:
                            # Add separator
                            voice_options.append(("_separator", "──── Voice Blends ────"))
                            # Add each blend
                            for blend_name, blend_data in blends.items():
                                display_name = f"🎭 {blend_name}"
                                if blend_data.get('description'):
                                    display_name += f" - {blend_data['description'][:30]}"
                                voice_options.append((f"blend:{blend_name}", display_name))
                except Exception as e:
                    logger.error(f"Failed to load voice blends: {e}")
            
            voice_select.set_options(voice_options)
            
            # Find first valid voice option (skip separators)
            valid_voice = None
            for value, _ in voice_options:
                if self._is_valid_voice(value):
                    valid_voice = value
                    break
            
            if valid_voice:
                voice_select.value = valid_voice
            else:
                voice_select.value = "af_bella"  # Fallback
        elif provider == "chatterbox":
            voice_select.set_options([
                ("default", "Default Voice"),
                ("custom", "Custom (Upload Reference)"),
            ])
            voice_select.value = "default"
        elif provider == "higgs":
            voice_select.set_options([
                ("professional_female", "Professional Female"),
                ("warm_female", "Warm Female"),
                ("storyteller_male", "Storyteller Male"),
                ("deep_male", "Deep Male"),
                ("energetic_female", "Energetic Female"),
                ("soft_female", "Soft Female"),
            ])
            voice_select.value = "professional_female"
        elif provider == "alltalk":
            voice_select.set_options([
                ("female_01.wav", "Female 01"),
                ("female_02.wav", "Female 02"),
                ("female_03.wav", "Female 03"),
                ("female_04.wav", "Female 04"),
                ("male_01.wav", "Male 01"),
                ("male_02.wav", "Male 02"),
                ("male_03.wav", "Male 03"),
                ("male_04.wav", "Male 04"),
            ])
            voice_select.value = "female_01.wav"
    
    def _update_default_model_options(self, provider: str) -> None:
        """Update default model options based on provider"""
        model_select = self.query_one("#default-model-select", Select)
        
        if provider == "openai":
            model_select.set_options([
                ("tts-1", "TTS-1 (Standard)"),
                ("tts-1-hd", "TTS-1-HD (High Quality)"),
            ])
            # Set the saved default or fallback
            default_model = get_cli_setting("app_tts", "default_model", "tts-1")
            if default_model in ["tts-1", "tts-1-hd"]:
                model_select.value = default_model
            else:
                model_select.value = "tts-1"
        elif provider == "elevenlabs":
            model_select.set_options([
                ("eleven_monolingual_v1", "Eleven Monolingual v1"),
                ("eleven_multilingual_v1", "Eleven Multilingual v1"),
                ("eleven_multilingual_v2", "Eleven Multilingual v2 (Default)"),
                ("eleven_turbo_v2", "Eleven Turbo v2"),
                ("eleven_turbo_v2_5", "Eleven Turbo v2.5"),
                ("eleven_flash_v2", "Eleven Flash v2 (Low Latency)"),
                ("eleven_flash_v2_5", "Eleven Flash v2.5 (Ultra Low Latency)"),
            ])
            model_select.value = "eleven_multilingual_v2"
        elif provider == "kokoro":
            logger.info("Setting Kokoro model options")
            model_select.set_options([
                ("kokoro", "Kokoro 82M"),
            ])
            model_select.value = "kokoro"
            logger.info("Kokoro model set successfully")
        elif provider == "chatterbox":
            model_select.set_options([
                ("chatterbox", "Chatterbox 0.5B"),
            ])
            model_select.value = "chatterbox"
        elif provider == "higgs":
            logger.info("Setting Higgs model options")
            model_select.set_options([
                ("higgs-audio-v2", "Higgs Audio V2 3B"),
            ])
            model_select.value = "higgs-audio-v2"
            logger.info("Higgs model set successfully")
        elif provider == "alltalk":
            model_select.set_options([
                ("alltalk", "AllTalk TTS"),
            ])
            model_select.value = "alltalk"
    
    def _save_settings(self) -> None:
        """Save TTS settings"""
        try:
            # Collect all settings
            settings = {}
            
            # Default settings
            settings["default_provider"] = self.query_one("#default-provider-select", Select).value
            settings["default_voice"] = self.query_one("#default-voice-select", Select).value
            settings["default_model"] = self.query_one("#default-model-select", Select).value
            settings["default_format"] = self.query_one("#default-format-select", Select).value
            settings["default_speed"] = self._validate_numeric_input(
                self.query_one("#default-speed-input", Input).value, 0.25, 4.0, 1.0
            )
            
            # OpenAI settings
            openai_key = self.query_one("#openai-api-key-input", Input).value
            if openai_key:
                settings["openai_api_key"] = openai_key
            
            base_url = self.query_one("#openai-base-url-input", Input).value
            if base_url and base_url != "https://api.openai.com/v1/audio/speech":
                settings["OPENAI_BASE_URL"] = base_url
            
            org_id = self.query_one("#openai-org-id-input", Input).value
            if org_id:
                settings["OPENAI_ORG_ID"] = org_id
            
            # ElevenLabs settings
            elevenlabs_key = self.query_one("#elevenlabs-api-key-input", Input).value
            if elevenlabs_key:
                settings["elevenlabs_api_key"] = elevenlabs_key
            
            settings["ELEVENLABS_DEFAULT_MODEL"] = self.query_one("#elevenlabs-model-select", Select).value
            settings["ELEVENLABS_OUTPUT_FORMAT"] = self.query_one("#elevenlabs-format-select", Select).value
            settings["ELEVENLABS_VOICE_STABILITY"] = self._validate_numeric_input(
                self.query_one("#elevenlabs-stability-input", Input).value, 0.0, 1.0, 0.5
            )
            settings["ELEVENLABS_SIMILARITY_BOOST"] = self._validate_numeric_input(
                self.query_one("#elevenlabs-similarity-input", Input).value, 0.0, 1.0, 0.8
            )
            settings["ELEVENLABS_STYLE"] = self._validate_numeric_input(
                self.query_one("#elevenlabs-style-input", Input).value, 0.0, 1.0, 0.0
            )
            settings["ELEVENLABS_USE_SPEAKER_BOOST"] = self.query_one("#elevenlabs-speaker-boost-switch", Switch).value
            
            # Kokoro settings
            settings["KOKORO_DEVICE_DEFAULT"] = self.query_one("#kokoro-device-select", Select).value
            settings["KOKORO_USE_ONNX"] = self.query_one("#kokoro-use-onnx-switch", Switch).value
            
            if self.kokoro_model_path:
                settings["KOKORO_ONNX_MODEL_PATH_DEFAULT"] = self.kokoro_model_path
            
            if self.kokoro_voices_path:
                settings["KOKORO_ONNX_VOICES_JSON_DEFAULT"] = self.kokoro_voices_path
            
            settings["KOKORO_MAX_TOKENS"] = int(self._validate_numeric_input(
                self.query_one("#kokoro-max-tokens-input", Input).value, 1, 10000, 500
            ))
            settings["KOKORO_ENABLE_VOICE_MIXING"] = self.query_one("#kokoro-voice-mixing-switch", Switch).value
            settings["KOKORO_TRACK_PERFORMANCE"] = self.query_one("#kokoro-performance-switch", Switch).value
            
            # Chatterbox settings
            settings["CHATTERBOX_DEVICE"] = self.query_one("#chatterbox-device-select", Select).value
            
            if self.chatterbox_voice_dir:
                settings["CHATTERBOX_VOICE_DIR"] = self.chatterbox_voice_dir
                
            settings["CHATTERBOX_EXAGGERATION"] = self._validate_numeric_input(
                self.query_one("#chatterbox-exaggeration-input", Input).value, 0.0, 1.0, 0.5
            )
            settings["CHATTERBOX_CFG_WEIGHT"] = self._validate_numeric_input(
                self.query_one("#chatterbox-cfg-weight-input", Input).value, 0.0, 1.0, 0.5
            )
            settings["CHATTERBOX_TEMPERATURE"] = self._validate_numeric_input(
                self.query_one("#chatterbox-temperature-input", Input).value, 0.0, 2.0, 0.5
            )
            settings["CHATTERBOX_CHUNK_SIZE"] = int(self._validate_numeric_input(
                self.query_one("#chatterbox-chunk-size-input", Input).value, 256, 8192, 1024
            ))
            
            seed = self.query_one("#chatterbox-seed-input", Input).value
            if seed:
                try:
                    settings["CHATTERBOX_RANDOM_SEED"] = int(seed)
                except ValueError:
                    pass
                    
            settings["CHATTERBOX_NUM_CANDIDATES"] = int(self._validate_numeric_input(
                self.query_one("#chatterbox-candidates-input", Input).value, 1, 5, 1
            ))
            settings["CHATTERBOX_VALIDATE_WHISPER"] = self.query_one("#chatterbox-whisper-switch", Switch).value
            settings["CHATTERBOX_PREPROCESS_TEXT"] = self.query_one("#chatterbox-preprocess-switch", Switch).value
            settings["CHATTERBOX_NORMALIZE_AUDIO"] = self.query_one("#chatterbox-normalize-switch", Switch).value
            settings["CHATTERBOX_TARGET_DB"] = self._validate_numeric_input(
                self.query_one("#chatterbox-target-db-input", Input).value, -40, 0, -20
            )
            settings["CHATTERBOX_MAX_CHUNK_SIZE"] = int(self._validate_numeric_input(
                self.query_one("#chatterbox-max-chunk-input", Input).value, 50, 5000, 500
            ))
            settings["CHATTERBOX_STREAMING"] = self.query_one("#chatterbox-streaming-switch", Switch).value
            settings["CHATTERBOX_STREAM_CHUNK_SIZE"] = int(self._validate_numeric_input(
                self.query_one("#chatterbox-stream-chunk-input", Input).value, 512, 16384, 4096
            ))
            settings["CHATTERBOX_ENABLE_CROSSFADE"] = self.query_one("#chatterbox-crossfade-switch", Switch).value
            settings["CHATTERBOX_CROSSFADE_MS"] = int(self._validate_numeric_input(
                self.query_one("#chatterbox-crossfade-ms-input", Input).value, 10, 500, 50
            ))
            
            # Higgs settings
            higgs_model_path = self.query_one("#higgs-model-path-input", Input).value
            if higgs_model_path:
                settings["HIGGS_MODEL_PATH"] = higgs_model_path
            
            higgs_voices_dir = self.query_one("#higgs-voices-dir-input", Input).value
            if higgs_voices_dir:
                settings["HIGGS_VOICE_SAMPLES_DIR"] = higgs_voices_dir
            
            # Use _get_select_key to get actual key value from Select widgets
            higgs_device_select = self.query_one("#higgs-device-select", Select)
            higgs_device_value = self._get_select_key(higgs_device_select)
            if higgs_device_value:
                settings["HIGGS_DEVICE"] = higgs_device_value
            
            settings["HIGGS_ENABLE_FLASH_ATTN"] = self.query_one("#higgs-flash-attn-switch", Switch).value
            
            higgs_dtype_select = self.query_one("#higgs-dtype-select", Select)
            higgs_dtype_value = self._get_select_key(higgs_dtype_select)
            if higgs_dtype_value:
                settings["HIGGS_DTYPE"] = higgs_dtype_value
            
            settings["HIGGS_MAX_REFERENCE_DURATION"] = int(self._validate_numeric_input(
                self.query_one("#higgs-max-ref-duration-input", Input).value, 1, 60, 30
            ))
            
            higgs_language_select = self.query_one("#higgs-language-select", Select)
            higgs_language_value = self._get_select_key(higgs_language_select)
            if higgs_language_value:
                settings["HIGGS_DEFAULT_LANGUAGE"] = higgs_language_value
            settings["HIGGS_ENABLE_VOICE_CLONING"] = self.query_one("#higgs-voice-cloning-switch", Switch).value
            settings["HIGGS_ENABLE_MULTI_SPEAKER"] = self.query_one("#higgs-multi-speaker-switch", Switch).value
            settings["HIGGS_SPEAKER_DELIMITER"] = self.query_one("#higgs-delimiter-input", Input).value
            settings["HIGGS_TRACK_PERFORMANCE"] = self.query_one("#higgs-track-performance-switch", Switch).value
            
            settings["HIGGS_MAX_NEW_TOKENS"] = int(self._validate_numeric_input(
                self.query_one("#higgs-max-tokens-input", Input).value, 512, 8192, 4096
            ))
            settings["HIGGS_TEMPERATURE"] = self._validate_numeric_input(
                self.query_one("#higgs-temperature-input", Input).value, 0.0, 2.0, 0.7
            )
            settings["HIGGS_TOP_P"] = self._validate_numeric_input(
                self.query_one("#higgs-top-p-input", Input).value, 0.0, 1.0, 0.9
            )
            settings["HIGGS_REPETITION_PENALTY"] = self._validate_numeric_input(
                self.query_one("#higgs-repetition-penalty-input", Input).value, 1.0, 2.0, 1.1
            )
            
            # AllTalk settings
            url = self.query_one("#alltalk-url-input", Input).value
            if url:
                settings["ALLTALK_TTS_URL_DEFAULT"] = url
            
            voice = self.query_one("#alltalk-voice-input", Input).value
            if voice:
                settings["ALLTALK_TTS_VOICE_DEFAULT"] = voice
                
            settings["ALLTALK_TTS_LANGUAGE_DEFAULT"] = self.query_one("#alltalk-language-select", Select).value
            settings["ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT"] = self.query_one("#alltalk-format-select", Select).value
            
            # Post save event
            self.app.post_message(STTSSettingsSaveEvent(settings))
            self.app.notify("TTS settings saved successfully", severity="information")
            
        except Exception as e:
            logger.error(f"Failed to collect settings: {e}")
            self.app.notify(f"Failed to save settings: {e}", severity="error")
    
    def _validate_numeric_input(self, value: str, min_val: float, max_val: float, default: float) -> float:
        """Validate and convert numeric input"""
        try:
            if not value:
                return default
            num_val = float(value)
            return max(min_val, min(max_val, num_val))
        except ValueError:
            return default
    
    def _get_select_key(self, select_widget) -> Optional[str]:
        """Get the actual key from a Select widget, not the display text"""
        if not hasattr(select_widget, '_options') or not select_widget._options:
            return None
        
        current_value = select_widget.value
        if current_value == Select.BLANK:
            return None
            
        # Find the key that matches the current value
        for key, label in select_widget._options:
            if label == current_value or key == current_value:
                return key
        
        return None
    
    def _load_kokoro_voice_blends(self) -> None:
        """Load and display Kokoro voice blends"""
        try:
            # Get voice blends from stored config
            blend_list = self.query_one("#kokoro-voice-blends-list", Static)
            
            # Load blends from config file
            blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            if blend_file.exists():
                with open(blend_file, 'r') as f:
                    blends = json.load(f)
                
                if blends:
                    # Format blends for display
                    blend_text = ""
                    for blend_name, blend_data in blends.items():
                        voices_str = ", ".join([f"{v[0]} ({v[1]:.2f})" for v in blend_data.get("voices", [])])
                        blend_text += f"[bold]{blend_name}[/bold]: {voices_str}\n"
                        if blend_data.get("description"):
                            blend_text += f"  [dim]{blend_data['description']}[/dim]\n"
                    blend_list.update(blend_text.strip())
                else:
                    blend_list.update("[dim]No voice blends configured[/dim]")
            else:
                blend_list.update("[dim]No voice blends configured[/dim]")
            
        except Exception as e:
            logger.error(f"Failed to load voice blends: {e}")
            blend_list.update("[red]Error loading voice blends[/red]")
    
    async def _show_add_voice_blend_dialog(self) -> None:
        """Show dialog to add a new voice blend"""
        try:
            # Show the voice blend dialog
            result = await self.app.push_screen_wait(VoiceBlendDialog())
            
            if result:
                # Save the blend
                blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
                blend_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing blends
                if blend_file.exists():
                    with open(blend_file, 'r') as f:
                        blends = json.load(f)
                else:
                    blends = {}
                
                # Add new blend
                blends[result['name']] = result
                
                # Save back
                with open(blend_file, 'w') as f:
                    json.dump(blends, f, indent=2)
                
                # Refresh display
                self._load_kokoro_voice_blends()
                self.app.notify(f"Voice blend '{result['name']}' created successfully", severity="success")
                
        except Exception as e:
            logger.error(f"Failed to create voice blend: {e}")
            self.app.notify(f"Error creating voice blend: {e}", severity="error")
    
    def _import_voice_blends(self) -> None:
        """Import voice blends from file"""
        try:
            filters = Filters(
                ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                ("All Files", lambda p: True)
            )
            
            file_picker = FileOpen(
                title="Import Voice Blends",
                filters=filters,
                context="voice_blends_import"
            )
            
            self.app.push_screen(file_picker, self._handle_import_file)
            
        except Exception as e:
            logger.error(f"Failed to show import dialog: {e}")
            self.app.notify(f"Error showing import dialog: {e}", severity="error")
    
    def _handle_import_file(self, path: str | None) -> None:
        """Handle the imported file"""
        if not path:
            return
        
        try:
            import_path = Path(path)
            
            # Load the import file
            with open(import_path, 'r') as f:
                imported_blends = json.load(f)
            
            # Load existing blends
            blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            blend_file.parent.mkdir(parents=True, exist_ok=True)
            
            if blend_file.exists():
                with open(blend_file, 'r') as f:
                    existing_blends = json.load(f)
            else:
                existing_blends = {}
            
            # Merge blends (imported overwrites existing with same name)
            existing_blends.update(imported_blends)
            
            # Save merged blends
            with open(blend_file, 'w') as f:
                json.dump(existing_blends, f, indent=2)
            
            # Refresh display
            self._load_kokoro_voice_blends()
            self.app.notify(
                f"Imported {len(imported_blends)} voice blend(s) successfully",
                severity="success"
            )
            
        except Exception as e:
            logger.error(f"Failed to import voice blends: {e}")
            self.app.notify(f"Error importing voice blends: {e}", severity="error")
    
    def _export_voice_blends(self) -> None:
        """Export voice blends to file"""
        try:
            # Load existing blends
            blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            
            if not blend_file.exists():
                self.app.notify("No voice blends to export", severity="warning")
                return
            
            with open(blend_file, 'r') as f:
                blends = json.load(f)
            
            if not blends:
                self.app.notify("No voice blends to export", severity="warning")
                return
            
            # Store blends temporarily for export
            self._export_blends = blends
            
            filters = Filters(
                ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                ("All Files", lambda p: True)
            )
            
            file_picker = FileSave(
                title="Export Voice Blends",
                filters=filters,
                default_filename="kokoro_voice_blends_export.json",
                context="voice_blends_export"
            )
            
            self.app.push_screen(file_picker, self._handle_export_file)
            
        except Exception as e:
            logger.error(f"Failed to export voice blends: {e}")
            self.app.notify(f"Error exporting voice blends: {e}", severity="error")
    
    def _handle_export_file(self, path: str | None) -> None:
        """Handle the export file location"""
        if not path or not hasattr(self, '_export_blends'):
            return
        
        try:
            export_path = Path(path)
            
            # Write the blends to the selected file
            with open(export_path, 'w') as f:
                json.dump(self._export_blends, f, indent=2)
            
            self.app.notify(
                f"Exported {len(self._export_blends)} voice blend(s) to: {export_path.name}",
                severity="success"
            )
            
            # Clean up temporary storage
            del self._export_blends
            
        except Exception as e:
            logger.error(f"Failed to export voice blends: {e}")
            self.app.notify(f"Error exporting voice blends: {e}", severity="error")
    
    def _browse_kokoro_model(self) -> None:
        """Browse for Kokoro model file"""
        # Create file picker for model files
        filters = Filters(
            ("ONNX Models", lambda p: p.suffix.lower() in [".onnx"]),
            ("All Files", lambda p: True)
        )
        
        # Get current value as starting path
        current_value = self.kokoro_model_path
        location = Path(current_value).parent if current_value and Path(current_value).parent.exists() else Path.home()
        
        file_picker = FileOpen(
            location=str(location),
            title="Select Kokoro Model File",
            filters=filters,
            context="kokoro_model"
        )
        
        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_kokoro_model_selection)
    
    def _handle_kokoro_model_selection(self, path: Path | None) -> None:
        """Handle Kokoro model file selection"""
        if path:
            # Update the stored path
            self.kokoro_model_path = str(path)
            # Update button label
            self._update_file_button_labels()
            logger.info(f"Kokoro model selected: {path}")
    
    def _browse_kokoro_voices(self) -> None:
        """Browse for Kokoro voices JSON file"""
        # Create file picker for JSON files
        filters = Filters(
            ("JSON Files", lambda p: p.suffix.lower() in [".json"]),
            ("All Files", lambda p: True)
        )
        
        # Get current value as starting path
        current_value = self.kokoro_voices_path
        location = Path(current_value).parent if current_value and Path(current_value).parent.exists() else Path.home()
        
        file_picker = FileOpen(
            location=str(location),
            title="Select Voices Configuration File",
            filters=filters,
            context="kokoro_voices"
        )
        
        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_kokoro_voices_selection)
    
    def _handle_kokoro_voices_selection(self, path: Path | None) -> None:
        """Handle Kokoro voices file selection"""
        if path:
            # Update the stored path
            self.kokoro_voices_path = str(path)
            # Update button label
            self._update_file_button_labels()
            logger.info(f"Kokoro voices config selected: {path}")
    
    def _browse_chatterbox_voice_dir(self) -> None:
        """Browse for Chatterbox voice directory"""
        # For directory selection, we'll use the file picker and guide user to select a file in the target directory
        # then extract the directory path
        
        # Get current value as starting path
        current_value = self.chatterbox_voice_dir
        if current_value.startswith("~"):
            current_value = str(Path(current_value).expanduser())
        location = Path(current_value) if current_value and Path(current_value).exists() else Path.home()
        
        # Create a filter that shows directories prominently
        filters = Filters(
            ("Directories", lambda p: p.is_dir() if p.exists() else False),
            ("All Files", lambda p: True)
        )
        
        file_picker = FileOpen(
            location=str(location),
            title="Select Voice Directory (choose any file in target directory)",
            filters=filters,
            context="chatterbox_voices_dir"
        )
        
        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_chatterbox_voice_dir_selection)
    
    def _handle_chatterbox_voice_dir_selection(self, path: Path | None) -> None:
        """Handle Chatterbox voice directory selection"""
        if path:
            # Get the directory from the selected path
            directory = path if path.is_dir() else path.parent
            # Update the stored path
            self.chatterbox_voice_dir = str(directory)
            # Update button label
            self._update_file_button_labels()
            logger.info(f"Chatterbox voice directory selected: {directory}")
    
    def _update_file_button_labels(self) -> None:
        """Update file picker button labels based on selected paths"""
        # Update Kokoro model button
        model_btn = self.query_one("#kokoro-browse-model-btn", Button)
        if self.kokoro_model_path:
            model_btn.label = f"📁 {Path(self.kokoro_model_path).name}"
        else:
            model_btn.label = "📁 Select model file"
        
        # Update Kokoro voices button
        voices_btn = self.query_one("#kokoro-browse-voices-btn", Button)
        if self.kokoro_voices_path:
            voices_btn.label = f"📁 {Path(self.kokoro_voices_path).name}"
        else:
            voices_btn.label = "📁 Select voices.json"
        
        # Update Chatterbox voice directory button
        voice_dir_btn = self.query_one("#chatterbox-browse-voice-dir-btn", Button)
        if self.chatterbox_voice_dir:
            voice_dir_btn.label = f"📁 {Path(self.chatterbox_voice_dir).name}"
        else:
            voice_dir_btn.label = "📁 Select voice directory"
    
    def _browse_higgs_voices_dir(self) -> None:
        """Browse for Higgs voices directory"""
        # Get current value as starting path
        voices_input = self.query_one("#higgs-voices-dir-input", Input)
        current_value = voices_input.value
        if current_value.startswith("~"):
            current_value = str(Path(current_value).expanduser())
        location = Path(current_value) if current_value and Path(current_value).exists() else Path.home()
        
        # Create filter for directory selection
        filters = Filters(
            ("Directories", lambda p: p.is_dir() if p.exists() else False),
            ("All Files", lambda p: True)
        )
        
        file_picker = FileOpen(
            location=str(location),
            title="Select Higgs Voice Samples Directory (choose any file in target directory)",
            filters=filters,
            context="higgs_voices_dir"
        )
        
        # Push the file picker screen
        self.app.push_screen(file_picker, self._handle_higgs_voices_dir_selection)
    
    def _handle_higgs_voices_dir_selection(self, path: Optional[Path]) -> None:
        """Handle the selection of Higgs voices directory"""
        if path:
            # Extract directory from selected file
            if path.is_file():
                dir_path = path.parent
            else:
                dir_path = path
            
            # Update input
            voices_input = self.query_one("#higgs-voices-dir-input", Input)
            voices_input.value = str(dir_path)
            logger.info(f"Higgs voices directory selected: {dir_path}")
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes"""
        if event.select.id == "default-provider-select":
            # Update voice and model options when provider changes
            self._update_default_voice_model_options(event.value)
    
    def _update_default_voice_model_options(self, provider: str) -> None:
        """Update default voice and model options based on selected provider"""
        voice_select = self.query_one("#default-voice-select", Select)
        model_select = self.query_one("#default-model-select", Select)
        
        if provider == "openai":
            voice_select.set_options([
                ("alloy", "Alloy"),
                ("ash", "Ash"),
                ("ballad", "Ballad"),
                ("coral", "Coral"),
                ("echo", "Echo"),
                ("fable", "Fable"),
                ("onyx", "Onyx"),
                ("nova", "Nova"),
                ("sage", "Sage"),
                ("shimmer", "Shimmer"),
                ("verse", "Verse"),
            ])
            model_select.set_options([
                ("tts-1", "TTS-1 (Standard)"),
                ("tts-1-hd", "TTS-1-HD (High Quality)"),
            ])
        elif provider == "elevenlabs":
            voice_select.set_options([
                ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
                ("AZnzlk1XvdvUeBnXmlld", "Domi"),
                ("EXAVITQu4vr4xnSDxMaL", "Bella"),
                ("ErXwobaYiN019PkySvjV", "Antoni"),
                ("MF3mGyEYCl7XYWbV9V6O", "Elli"),
                ("TxGEqnHWrfWFTfGW9XjX", "Josh"),
                ("VR6AewLTigWG4xSOukaG", "Arnold"),
                ("pNInz6obpgDQGcFmaJgB", "Adam"),
                ("yoZ06aMxZJJ28mfd3POQ", "Sam"),
            ])
            model_select.set_options([
                ("eleven_monolingual_v1", "Eleven Monolingual v1"),
                ("eleven_multilingual_v1", "Eleven Multilingual v1"),
                ("eleven_multilingual_v2", "Eleven Multilingual v2 (Default)"),
                ("eleven_turbo_v2", "Eleven Turbo v2"),
                ("eleven_turbo_v2_5", "Eleven Turbo v2.5"),
                ("eleven_flash_v2", "Eleven Flash v2 (Low Latency)"),
                ("eleven_flash_v2_5", "Eleven Flash v2.5 (Ultra Low Latency)"),
            ])
        elif provider == "kokoro":
            logger.info(f"Setting up Kokoro voices for provider: {provider}")
            voice_options = [
                ("af_alloy", "Alloy (US Female)"),
                ("af_aoede", "Aoede (US Female)"),
                ("af_bella", "Bella (US Female)"),
                ("af_heart", "Heart (US Female)"),
                ("af_jessica", "Jessica (US Female)"),
                ("af_kore", "Kore (US Female)"),
                ("af_nicole", "Nicole (US Female)"),
                ("af_nova", "Nova (US Female)"),
                ("af_river", "River (US Female)"),
                ("af_sarah", "Sarah (US Female)"),
                ("af_sky", "Sky (US Female)"),
                ("am_adam", "Adam (US Male)"),
                ("am_michael", "Michael (US Male)"),
                ("bf_emma", "Emma (UK Female)"),
                ("bf_isabella", "Isabella (UK Female)"),
                ("bm_george", "George (UK Male)"),
                ("bm_lewis", "Lewis (UK Male)"),
            ]
            
            # Add saved voice blends
            blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            if blend_file.exists():
                try:
                    import json
                    with open(blend_file, 'r') as f:
                        blends = json.load(f)
                        if blends:
                            # Add separator
                            voice_options.append(("_separator", "──── Voice Blends ────"))
                            # Add each blend
                            for blend_name, blend_data in blends.items():
                                display_name = f"🎭 {blend_name}"
                                if blend_data.get('description'):
                                    display_name += f" - {blend_data['description'][:30]}"
                                voice_options.append((f"blend:{blend_name}", display_name))
                except Exception as e:
                    logger.error(f"Failed to load voice blends: {e}")
            
            voice_select.set_options(voice_options)
            
            # Find first valid voice option (skip separators)
            valid_voice = None
            for value, _ in voice_options:
                if self._is_valid_voice(value):
                    valid_voice = value
                    break
            
            if valid_voice:
                voice_select.value = valid_voice
            
            model_select.set_options([
                ("kokoro", "Kokoro 82M"),
            ])
        elif provider == "chatterbox":
            voice_select.set_options([
                ("default", "Default Voice"),
                ("custom", "Custom (Upload Reference)"),
            ])
            model_select.set_options([
                ("chatterbox", "Chatterbox 0.5B"),
            ])
        elif provider == "alltalk":
            voice_select.set_options([
                ("female_01.wav", "Female 01"),
                ("female_02.wav", "Female 02"),
                ("female_03.wav", "Female 03"),
                ("female_04.wav", "Female 04"),
                ("male_01.wav", "Male 01"),
                ("male_02.wav", "Male 02"),
                ("male_03.wav", "Male 03"),
                ("male_04.wav", "Male 04"),
            ])
            model_select.set_options([
                ("alltalk", "AllTalk TTS"),
            ])


class AudioBookGenerationWidget(Widget):
    """AudioBook/Podcast Generation widget"""
    
    DEFAULT_CSS = """
    AudioBookGenerationWidget {
        height: 100%;
        width: 100%;
    }
    
    .audiobook-container {
        padding: 1;
        height: 100%;
    }
    
    .chapter-list {
        height: 20;
        border: solid $primary;
        margin: 1 0;
        overflow-y: auto;
    }
    
    #audiobook-generation-log {
        height: 15;
        border: solid $secondary;
    }
    
    .cost-estimate {
        color: $warning;
        margin: 1 0;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.content_text = ""
        self.detected_chapters = []
        self.generated_audiobook_path = None
    
    def compose(self) -> ComposeResult:
        """Compose the AudioBook/Podcast UI"""
        with ScrollableContainer(classes="audiobook-container"):
            yield Label("📚 AudioBook/Podcast Generation", classes="section-title")
            
            # Import section
            with Collapsible(title="Import Content", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Import From:", classes="form-label")
                    yield Select(
                        options=[
                            ("file", "Text File"),
                            ("notes", "Notes"),
                            ("conversation", "Conversation"),
                            ("paste", "Paste Text"),
                        ],
                        id="import-source-select"
                )
                
                yield Button("📁 Import Content", id="import-content-btn", variant="default")
            
            # Content preview
            yield Label("Content Preview:")
            yield TextArea(
                id="content-preview",
                disabled=True
            )
            
            # Chapter Editor - Enhanced visual chapter editing
            with Collapsible(title="📖 Chapter Editor", classes="settings-section", collapsed=False):
                from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget
                yield ChapterEditorWidget(id="chapter-editor-widget")
            
            # Voice assignment - Enhanced character voice management
            with Collapsible(title="🎭 Voice Assignment", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Narrator Voice:", classes="form-label")
                    yield Select(
                        options=[
                            ("alloy", "Alloy"),
                            ("echo", "Echo"),
                            ("fable", "Fable"),
                            ("onyx", "Onyx"),
                            ("nova", "Nova"),
                            ("shimmer", "Shimmer"),
                        ],
                        id="narrator-voice-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Multi-voice:", classes="form-label")
                    yield Switch(id="multi-voice-switch", value=False)
                
                # Character voice widget
                from tldw_chatbook.Widgets.TTS.character_voice_widget import CharacterVoiceWidget
                yield CharacterVoiceWidget(id="character-voice-widget")
            
            # Generation settings
            with Collapsible(title="Generation Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Provider:", classes="form-label")
                    yield Select(
                        options=[
                            ("openai", "OpenAI"),
                            ("elevenlabs", "ElevenLabs"),
                            ("kokoro", "Kokoro (Local)"),
                            ("chatterbox", "Chatterbox (Local)"),
                        ],
                        id="audiobook-provider-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Audio Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("mp3", "MP3"),
                            ("m4b", "M4B (AudioBook)"),
                            ("opus", "Opus"),
                            ("aac", "AAC"),
                            ("wav", "WAV"),
                        ],
                        id="audiobook-format-select"
                )
                
                with Horizontal(classes="form-row"):
                    yield Label("Include Chapter Markers:", classes="form-label")
                    yield Switch(id="chapter-markers-switch", value=True)
                
                with Horizontal(classes="form-row"):
                    yield Label("Background Music:", classes="form-label")
                    yield Switch(id="background-music-switch", value=False)
            
            # Cost estimate
            yield Static("", id="cost-estimate", classes="cost-estimate")
            
            # Generate button
            yield Button("🎙️ Generate AudioBook", id="generate-audiobook-btn", variant="primary")
            
            # Export button (initially disabled)
            yield Button("💾 Export AudioBook", id="audiobook-export-btn", variant="success", disabled=True)
            
            # Progress section
            yield Rule()
            yield Label("Generation Progress:")
            yield RichLog(id="audiobook-generation-log", highlight=True, markup=True)
    
    def on_mount(self) -> None:
        """Set initial values from config after mount"""
        # Delay initialization to ensure widgets are ready
        self.set_timer(0.1, self._initialize_audiobook_defaults)
    
    def _initialize_audiobook_defaults(self) -> None:
        """Initialize default values after widgets are ready"""
        try:
            # Set audiobook provider
            provider_select = self.query_one("#audiobook-provider-select", Select)
            default_provider = get_cli_setting("app_tts", "default_provider", "openai")
            if default_provider in ["openai", "elevenlabs", "kokoro", "chatterbox"]:
                try:
                    provider_select.value = default_provider
                except Exception as e:
                    logger.debug(f"Could not set audiobook provider: {e}")
            
            # Set default format to m4b
            format_select = self.query_one("#audiobook-format-select", Select)
            try:
                format_select.value = "m4b"
            except Exception as e:
                logger.debug(f"Could not set audiobook format: {e}")
        except Exception as e:
            logger.warning(f"Failed to set audiobook defaults: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "import-content-btn":
            self._import_content()
            event.stop()  # Prevent event from bubbling up
        elif event.button.id == "generate-audiobook-btn":
            self._generate_audiobook()
            event.stop()
        elif event.button.id == "audiobook-export-btn":
            self._export_audiobook()
            event.stop()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes"""
        if event.select.id == "audiobook-provider-select":
            # Update character voice widget provider
            try:
                from tldw_chatbook.Widgets.TTS.character_voice_widget import CharacterVoiceWidget
                voice_widget = self.query_one("#character-voice-widget", CharacterVoiceWidget)
                voice_widget.provider = event.value
                logger.info(f"Updated voice widget provider to: {event.value}")
            except Exception as e:
                logger.debug(f"Could not update voice widget provider: {e}")
    
    def _import_content(self) -> None:
        """Import content for audiobook generation"""
        import_source = self.query_one("#import-source-select", Select).value
        
        if import_source == "file":
            self._import_from_file()
        elif import_source == "notes":
            self._import_from_notes()
        elif import_source == "conversation":
            self._import_from_conversation()
        elif import_source == "paste":
            self._import_from_paste()
    
    def _import_from_file(self) -> None:
        """Import content from a text file"""
        try:
            # Create file picker for text files using pre-imported FileOpen
            filters = Filters(
                ("Text Files", lambda p: p.suffix.lower() in [".txt", ".md", ".rst"]),
                ("eBook Files", lambda p: p.suffix.lower() in [".epub", ".mobi"]),
                ("All Files", lambda p: True)
            )
            
            file_picker = FileOpen(
                title="Select Text File for AudioBook",
                filters=filters,
                context="audiobook_text"
            )
            
            # Mount the file picker
            self.app.push_screen(file_picker, self._handle_file_selection)
        except ImportError:
            # Fallback to simple file input
            self.app.notify("File picker not available. Please paste your text instead.", severity="warning")
    
    def _handle_file_selection(self, path: str | None) -> None:
        """Handle file selection for audiobook content"""
        if not path:
            return
        
        try:
            # Read the file content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update content preview
            self.content_text = content
            content_preview = self.query_one("#content-preview", TextArea)
            content_preview.load_text(content[:1000] + "..." if len(content) > 1000 else content)
            content_preview.disabled = False
            
            # Detect chapters if enabled
            if self.query_one("#auto-chapters-switch", Switch).value:
                self._detect_chapters()
            
            self.app.notify(f"Imported {len(content)} characters from {Path(path).name}", severity="information")
            
        except Exception as e:
            logger.error(f"Failed to import file: {e}")
            self.app.notify(f"Failed to import file: {e}", severity="error")
    
    def _import_from_notes(self) -> None:
        """Import content from notes"""
        from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import NoteSelectionDialog
        from tldw_chatbook.DB.ChaChaNotes_DB import fetch_all_notes
        
        try:
            # Fetch all notes from database
            notes = fetch_all_notes()
            if not notes:
                self.app.notify("No notes found in database", severity="warning")
                return
            
            # Show note selection dialog
            def handle_note_selection(selected_ids: Optional[List[int]]) -> None:
                if selected_ids:
                    # Fetch full content for selected notes
                    from tldw_chatbook.DB.ChaChaNotes_DB import fetch_note_by_id
                    combined_content = []
                    
                    for note_id in selected_ids:
                        note = fetch_note_by_id(note_id)
                        if note:
                            # Add note title as chapter if it exists
                            if note.get('title'):
                                combined_content.append(f"# {note['title']}\n")
                            combined_content.append(note.get('content', ''))
                            combined_content.append("\n\n")  # Separator between notes
                    
                    # Load combined content
                    self.content_text = "\n".join(combined_content)
                    content_preview = self.query_one("#content-preview", TextArea)
                    preview_text = self.content_text[:1000] + "..." if len(self.content_text) > 1000 else self.content_text
                    content_preview.load_text(preview_text)
                    content_preview.disabled = False
                    
                    # Detect chapters if enabled
                    if self.query_one("#auto-chapters-switch", Switch).value:
                        self._detect_chapters()
                    
                    self.app.notify(f"Imported {len(selected_ids)} note(s)", severity="information")
            
            self.app.push_screen(NoteSelectionDialog(notes), handle_note_selection)
            
        except Exception as e:
            logger.error(f"Failed to import from notes: {e}")
            self.app.notify(f"Failed to import notes: {e}", severity="error")
    
    def _import_from_conversation(self) -> None:
        """Import content from conversation"""
        from tldw_chatbook.Widgets.conversation_selection_dialog import ConversationSelectionDialog
        from tldw_chatbook.DB.ChaChaNotes_DB import fetch_all_conversations
        
        try:
            # Fetch all conversations from database
            conversations = fetch_all_conversations()
            if not conversations:
                self.app.notify("No conversations found in database", severity="warning")
                return
            
            # Show conversation selection dialog
            def handle_conversation_selection(selection: Optional[Dict[str, Any]]) -> None:
                if selection:
                    # Fetch messages for selected conversation
                    from tldw_chatbook.DB.ChaChaNotes_DB import fetch_messages_by_conversation_id
                    messages = fetch_messages_by_conversation_id(selection['conversation_id'])
                    
                    if not messages:
                        self.app.notify("No messages found in conversation", severity="warning")
                        return
                    
                    # Build content based on options
                    content_parts = []
                    for msg in messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        
                        # Filter based on inclusion options
                        if selection.get('include_all'):
                            pass  # Include all messages
                        elif selection.get('include_user') and role != 'user':
                            continue
                        elif selection.get('include_assistant') and role != 'assistant':
                            continue
                        
                        # Format based on speaker option
                        if selection.get('include_speakers'):
                            speaker_name = "User" if role == "user" else "Assistant"
                            content_parts.append(f"{speaker_name}: {content}")
                        else:
                            content_parts.append(content)
                        
                        content_parts.append("")  # Empty line between messages
                    
                    # Load combined content
                    self.content_text = "\n".join(content_parts)
                    content_preview = self.query_one("#content-preview", TextArea)
                    preview_text = self.content_text[:1000] + "..." if len(self.content_text) > 1000 else self.content_text
                    content_preview.load_text(preview_text)
                    content_preview.disabled = False
                    
                    # Auto-detect chapters might not be suitable for conversations
                    # but run it if enabled
                    if self.query_one("#auto-chapters-switch", Switch).value:
                        self._detect_chapters()
                    
                    self.app.notify(f"Imported conversation with {len(messages)} messages", severity="information")
            
            self.app.push_screen(ConversationSelectionDialog(conversations), handle_conversation_selection)
            
        except Exception as e:
            logger.error(f"Failed to import from conversation: {e}")
            self.app.notify(f"Failed to import conversation: {e}", severity="error")
    
    def _import_from_paste(self) -> None:
        """Import content from clipboard paste"""
        # Enable the content preview for editing
        content_preview = self.query_one("#content-preview", TextArea)
        content_preview.disabled = False
        content_preview.focus()
        self.app.notify("Paste your text into the content preview area", severity="information")
    
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area content changes"""
        if event.text_area.id == "content-preview":
            self.content_text = event.text_area.text
            # Detect chapters if auto-detect is enabled
            if self.query_one("#auto-chapters-switch", Switch).value and self.content_text:
                self._detect_chapters()
    
    def on_chapter_edit_event(self, event) -> None:
        """Handle chapter edit events from the chapter editor"""
        from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditEvent
        if isinstance(event, ChapterEditEvent):
            # Update our internal chapter list
            try:
                from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget
                chapter_editor = self.query_one("#chapter-editor-widget", ChapterEditorWidget)
                self.detected_chapters = chapter_editor.get_chapters()
                logger.info(f"Chapter {event.action}: {event.chapter.title}")
            except Exception as e:
                logger.error(f"Failed to handle chapter edit: {e}")
    
    def on_chapter_preview_event(self, event) -> None:
        """Handle chapter preview requests"""
        from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterPreviewEvent
        if isinstance(event, ChapterPreviewEvent):
            if event.preview_type == "audio":
                self._preview_chapter_audio(event.chapter)
    
    def on_character_detection_event(self, event) -> None:
        """Handle character detection requests"""
        from tldw_chatbook.Widgets.TTS.character_voice_widget import CharacterDetectionEvent, CharacterVoiceWidget
        if isinstance(event, CharacterDetectionEvent):
            # Detect characters from current content
            if self.content_text:
                try:
                    voice_widget = self.query_one("#character-voice-widget", CharacterVoiceWidget)
                    characters = voice_widget.detect_characters_from_text(
                        self.content_text, 
                        event.auto_assign
                    )
                    self.app.notify(f"Detected {len(characters)} characters", severity="information")
                except Exception as e:
                    logger.error(f"Failed to detect characters: {e}")
                    self.app.notify(f"Failed to detect characters: {e}", severity="error")
            else:
                self.app.notify("Please import content first", severity="warning")
    
    def on_character_voice_assign_event(self, event) -> None:
        """Handle character voice assignments"""
        from tldw_chatbook.Widgets.TTS.character_voice_widget import CharacterVoiceAssignEvent
        if isinstance(event, CharacterVoiceAssignEvent):
            logger.info(f"Voice assigned: {event.character_name} → {event.voice_id}")
    
    def _detect_chapters(self) -> None:
        """Detect chapters in the content"""
        if not self.content_text:
            return
        
        try:
            from tldw_chatbook.TTS.audiobook_generator import ChapterDetector
            from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget
            
            # Detect chapters
            self.detected_chapters = ChapterDetector.detect_chapters(self.content_text)
            
            # Update the chapter editor widget
            try:
                chapter_editor = self.query_one("#chapter-editor-widget", ChapterEditorWidget)
                chapter_editor.set_chapters(self.detected_chapters)
                self.app.notify(f"Detected {len(self.detected_chapters)} chapters", severity="information")
            except Exception as e:
                logger.warning(f"Could not update chapter editor: {e}")
                # Fall back to old display method if chapter editor not found
                chapter_list = self.query_one("#chapter-list", Static)
                if self.detected_chapters:
                    chapter_display = []
                    for i, chapter in enumerate(self.detected_chapters):
                        chapter_display.append(f"{i+1}. {chapter.title} ({len(chapter.content.split())} words)")
                    
                    chapter_list.update("\n".join(chapter_display))
                    self.app.notify(f"Detected {len(self.detected_chapters)} chapters", severity="information")
                else:
                    chapter_list.update("No chapters detected")
                
        except Exception as e:
            logger.error(f"Failed to detect chapters: {e}")
            self.app.notify(f"Failed to detect chapters: {e}", severity="error")
    
    def _generate_audiobook(self) -> None:
        """Generate the audiobook"""
        # Validate content
        if not self.content_text:
            self.app.notify("Please import content first", severity="warning")
            return
        
        # Get settings from UI
        provider = self.query_one("#audiobook-provider-select", Select).value
        audio_format = self.query_one("#audiobook-format-select", Select).value
        narrator_voice = self.query_one("#narrator-voice-select", Select).value
        
        # Validate voice selection
        if not narrator_voice or narrator_voice == Select.BLANK:
            self.app.notify("Please select a valid narrator voice", severity="warning")
            return
        
        multi_voice = self.query_one("#multi-voice-switch", Switch).value
        include_chapters = self.query_one("#chapter-markers-switch", Switch).value
        background_music = self.query_one("#background-music-switch", Switch).value
        
        # Get chapters from the chapter editor widget
        try:
            from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget
            chapter_editor = self.query_one("#chapter-editor-widget", ChapterEditorWidget)
            chapters = chapter_editor.get_chapters()
        except Exception as e:
            logger.warning(f"Could not get chapters from editor: {e}")
            chapters = self.detected_chapters
        
        # Get title from first chapter or use default
        title = "Untitled AudioBook"
        if chapters:
            # Use book title if detected, otherwise use first chapter
            for chapter in chapters:
                if "title" in chapter.title.lower() or chapter.number == 1:
                    title = chapter.title
                    break
        
        # Get character voice assignments if multi-voice is enabled
        character_voices = {}
        if multi_voice:
            try:
                from tldw_chatbook.Widgets.TTS.character_voice_widget import CharacterVoiceWidget
                voice_widget = self.query_one("#character-voice-widget", CharacterVoiceWidget)
                character_voices = voice_widget.get_voice_assignments()
                logger.info(f"Using character voices: {character_voices}")
            except Exception as e:
                logger.warning(f"Could not get character voices: {e}")
        
        # Prepare options
        options = {
            "title": title,
            "author": "Unknown",
            "provider": provider,
            "model": self._get_model_for_provider(provider),
            "chapter_detection": include_chapters,
            "multi_voice": multi_voice,
            "character_voices": character_voices,
            "background_music": None if not background_music else True,
            "enable_ssml": provider in ["elevenlabs"],
            "normalize_audio": True,
        }
        
        # Log start
        log = self.query_one("#audiobook-generation-log", RichLog)
        log.clear()
        log.write("[bold yellow]Starting audiobook generation...[/bold yellow]")
        log.write(f"Provider: {provider}")
        log.write(f"Format: {audio_format}")
        log.write(f"Content length: {len(self.content_text)} characters")
        
        # Estimate cost
        self._estimate_cost(provider, len(self.content_text))
        
        # Disable generate button
        self.query_one("#generate-audiobook-btn", Button).disabled = True
        
        # Post event to generate audiobook
        self.app.post_message(STTSAudioBookGenerateEvent(
            content=self.content_text,
            chapters=self.detected_chapters if include_chapters else [],
            narrator_voice=narrator_voice,
            output_format=audio_format,
            options=options
        ))
    
    def _get_model_for_provider(self, provider: str) -> str:
        """Get default model for provider"""
        models = {
            "openai": "tts-1",
            "elevenlabs": "eleven_multilingual_v2",
            "kokoro": "kokoro-v0_19",
            "chatterbox": "chatterbox-v1"
        }
        return models.get(provider, "tts-1")
    
    def _estimate_cost(self, provider: str, char_count: int) -> None:
        """Estimate and display cost"""
        # Simple cost estimation (prices per 1K characters)
        costs_per_1k = {
            "openai": 0.015,  # TTS-1 pricing
            "elevenlabs": 0.13,  # Starter pricing
            "kokoro": 0.0,  # Local
            "chatterbox": 0.0,  # Local
        }
        
        cost_per_1k = costs_per_1k.get(provider, 0.0)
        estimated_cost = (char_count / 1000) * cost_per_1k
        
        cost_display = self.query_one("#cost-estimate", Static)
        if estimated_cost > 0:
            cost_display.update(f"Estimated cost: ${estimated_cost:.2f}")
        else:
            cost_display.update("Free (using local model)")
    
    def _is_valid_voice(self, voice: str) -> bool:
        """Check if a voice value is valid (not a separator)"""
        return bool(voice) and not str(voice).startswith("_separator")
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes"""
        if event.select.id == "audiobook-provider-select":
            # Update narrator voice options based on provider
            self._update_voice_options(event.value)
            # Update cost estimate
            if self.content_text:
                self._estimate_cost(event.value, len(self.content_text))
        elif event.select.id == "narrator-voice-select":
            # Validate voice selection (prevent selecting separators)
            if not self._is_valid_voice(event.value):
                # Find and select the first valid voice
                voice_select = event.select
                for value, _ in voice_select._options:
                    if self._is_valid_voice(value):
                        voice_select.value = value
                        break
    
    def _update_voice_options(self, provider: str) -> None:
        """Update voice options based on provider"""
        voice_select = self.query_one("#narrator-voice-select", Select)
        
        if provider == "openai":
            voice_select.set_options([
                ("alloy", "Alloy"),
                ("echo", "Echo"),
                ("fable", "Fable"),
                ("onyx", "Onyx"),
                ("nova", "Nova"),
                ("shimmer", "Shimmer"),
            ])
        elif provider == "elevenlabs":
            voice_select.set_options([
                ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
                ("AZnzlk1XvdvUeBnXmlld", "Domi"),
                ("EXAVITQu4vr4xnSDxMaL", "Bella"),
                ("ErXwobaYiN019PkySvjV", "Antoni"),
                ("MF3mGyEYCl7XYWbV9V6O", "Elli"),
            ])
        elif provider == "kokoro":
            logger.info(f"Setting up Kokoro voices for provider: {provider}")
            voice_options = [
                ("af_bella", "Bella (US Female)"),
                ("af_nicole", "Nicole (US Female)"),
                ("af_sarah", "Sarah (US Female)"),
                ("am_adam", "Adam (US Male)"),
                ("am_michael", "Michael (US Male)"),
                ("bf_emma", "Emma (UK Female)"),
                ("bm_george", "George (UK Male)"),
            ]
            
            # Add saved voice blends
            blend_file = Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            if blend_file.exists():
                try:
                    import json
                    with open(blend_file, 'r') as f:
                        blends = json.load(f)
                        if blends:
                            # Add separator
                            voice_options.append(("_separator", "──── Voice Blends ────"))
                            # Add each blend
                            for blend_name, blend_data in blends.items():
                                display_name = f"🎭 {blend_name}"
                                if blend_data.get('description'):
                                    display_name += f" - {blend_data['description'][:30]}"
                                voice_options.append((f"blend:{blend_name}", display_name))
                except Exception as e:
                    logger.error(f"Failed to load voice blends: {e}")
            
            voice_select.set_options(voice_options)
            
            # Find first valid voice option (skip separators)
            valid_voice = None
            for value, _ in voice_options:
                if self._is_valid_voice(value):
                    valid_voice = value
                    break
            
            if valid_voice:
                voice_select.value = valid_voice
                
        elif provider == "chatterbox":
            voice_select.set_options([
                ("default", "Default"),
                ("custom", "Custom Voice"),
            ])
    
    def _export_audiobook(self) -> None:
        """Export the generated audiobook"""
        if not self.generated_audiobook_path:
            self.app.notify("No audiobook to export", severity="warning")
            return
        
        try:
            # Create file picker for save location using pre-imported FileSave
            filters = Filters(
                ("AudioBook Files", lambda p: p.suffix.lower() in [".m4b", ".mp3"]),
                ("All Files", lambda p: True)
            )
            
            file_picker = FileSave(
                title="Save AudioBook As",
                filters=filters,
                default_filename=self.generated_audiobook_path.name,
                context="audiobook_save"
            )
            
            # Mount the file picker
            self.app.push_screen(file_picker, self._handle_export_location)
        except ImportError:
            # Fallback
            self.app.notify(f"AudioBook saved to: {self.generated_audiobook_path}", severity="information")
    
    def _handle_export_location(self, path: str | None) -> None:
        """Handle export location selection"""
        if not path or not self.generated_audiobook_path:
            return
        
        try:
            import shutil
            shutil.copy2(self.generated_audiobook_path, path)
            self.app.notify(f"AudioBook exported to: {Path(path).name}", severity="information")
        except Exception as e:
            logger.error(f"Failed to export audiobook: {e}")
            self.app.notify(f"Failed to export audiobook: {e}", severity="error")
    
    def audiobook_generation_complete(self, success: bool, path: Optional[Path] = None) -> None:
        """Handle audiobook generation completion"""
        # Re-enable generate button
        self.query_one("#generate-audiobook-btn", Button).disabled = False
        
        if success and path:
            self.generated_audiobook_path = path
            # Enable export button
            self.query_one("#audiobook-export-btn", Button).disabled = False
            
            # Update log
            log = self.query_one("#audiobook-generation-log", RichLog)
            log.write("[bold green]✓ AudioBook generation complete![/bold green]")
            log.write(f"Output file: {path.name}")
        else:
            # Update log
            log = self.query_one("#audiobook-generation-log", RichLog)
            log.write("[bold red]✗ AudioBook generation failed![/bold red]")
    
    def _preview_chapter_audio(self, chapter) -> None:
        """Generate audio preview for a single chapter"""
        try:
            # Get current settings
            provider = self.query_one("#audiobook-provider-select", Select).value
            narrator_voice = self.query_one("#narrator-voice-select", Select).value
            
            if not narrator_voice or narrator_voice == Select.BLANK:
                self.app.notify("Please select a valid narrator voice", severity="warning")
                return
            
            # Limit preview to first 500 characters
            preview_text = chapter.content[:500] + "..." if len(chapter.content) > 500 else chapter.content
            
            # Log preview generation
            log = self.query_one("#audiobook-generation-log", RichLog)
            log.write(f"[yellow]Generating preview for: {chapter.title}[/yellow]")
            
            # Create TTS request event
            from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSPlaygroundGenerateEvent
            
            # Post event to generate preview
            self.post_message(STTSPlaygroundGenerateEvent(
                text=preview_text,
                provider=provider,
                voice=narrator_voice,
                model=self._get_model_for_provider(provider),
                speed=1.0,
                format="mp3",
                extra_params={"preview_chapter": chapter.title}
            ))
            
            log.write("[green]Preview generation started...[/green]")
            
        except Exception as e:
            logger.error(f"Failed to preview chapter audio: {e}")
            self.app.notify(f"Failed to generate preview: {e}", severity="error")
    
    def _get_model_for_provider(self, provider: str) -> str:
        """Get the default model for a given provider"""
        model_map = {
            "openai": "tts-1",
            "elevenlabs": "eleven_multilingual_v2",
            "kokoro": "kokoro",
            "chatterbox": "chatterbox",
            "alltalk": "alltalk"
        }
        return model_map.get(provider, "default")


class STTSWindow(Container):
    """Main S/TT/S window containing all sub-windows"""
    
    DEFAULT_CSS = """
    STTSWindow {
        layout: horizontal;
        height: 100%;
    }
    
    .stts-sidebar {
        width: 30;
        border-right: solid $primary;
        padding: 1;
    }
    
    .stts-content {
        width: 1fr;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .sidebar-button {
        width: 100%;
        margin-bottom: 1;
    }
    """
    
    current_view = reactive("playground")
    
    def __init__(self, app_instance, **kwargs):
        """Initialize the S/TT/S window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
    
    def compose(self) -> ComposeResult:
        """Compose the S/TT/S window"""
        # Sidebar
        with Vertical(classes="stts-sidebar"):
            yield Label("S/TT/S Menu", classes="section-title")
            yield Button("🎤 TTS Playground", id="view-playground-btn", classes="sidebar-button", variant="primary")
            yield Button("⚙️ TTS Settings", id="view-settings-btn", classes="sidebar-button")
            yield Button("📚 AudioBook/Podcast", id="view-audiobook-btn", classes="sidebar-button")
            
            # Additional features
            yield Rule()
            yield Label("Additional Features:", classes="section-title")
            yield Button("🎙️ Voice Cloning", id="view-voice-cloning-btn", classes="sidebar-button")
            yield Button("🔤 Speech Recognition", id="view-stt-btn", classes="sidebar-button")
            yield Button("🎵 Audio Effects", id="view-effects-btn", classes="sidebar-button", disabled=True)
        
        # Content area
        with Container(classes="stts-content"):
            # Show playground by default
            yield TTSPlaygroundWidget()
    
    def watch_current_view(self, old_view: str, new_view: str) -> None:
        """Handle view changes"""
        # Remove old content
        content_container = self.query_one(".stts-content", Container)
        
        # Give widgets a chance to clean up before removal
        for child in content_container.children:
            if hasattr(child, 'cleanup') and callable(child.cleanup):
                try:
                    child.cleanup()
                except Exception as e:
                    logger.debug(f"Error during widget cleanup: {e}")
        
        # Remove all children from the container
        content_container.remove_children()
        
        # Add new content based on view
        if new_view == "playground":
            content_container.mount(TTSPlaygroundWidget())
        elif new_view == "settings":
            content_container.mount(TTSSettingsWidget())
        elif new_view == "audiobook":
            content_container.mount(AudioBookGenerationWidget())
        elif new_view == "dictation":
            content_container.mount(DictationWindow())
        
        # Update button variants
        for btn in self.query(".sidebar-button").results(Button):
            btn.variant = "default"
        
        if new_view == "playground":
            self.query_one("#view-playground-btn", Button).variant = "primary"
        elif new_view == "settings":
            self.query_one("#view-settings-btn", Button).variant = "primary"
        elif new_view == "audiobook":
            self.query_one("#view-audiobook-btn", Button).variant = "primary"
        elif new_view == "dictation":
            self.query_one("#view-stt-btn", Button).variant = "primary"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses and delegate to content widgets"""
        # Handle sidebar buttons
        if event.button.id == "view-playground-btn":
            self.current_view = "playground"
        elif event.button.id == "view-settings-btn":
            self.current_view = "settings"
        elif event.button.id == "view-audiobook-btn":
            self.current_view = "audiobook"
        elif event.button.id == "view-voice-cloning-btn":
            # Import and push the Voice Cloning window
            from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow
            self.app.push_screen(VoiceCloningWindow())
        elif event.button.id == "view-stt-btn":
            self.current_view = "dictation"
        elif event.button.id == "view-effects-btn":
            self.app.notify("Audio Effects coming soon!", severity="information")
        else:
            # Try to delegate to the active content widget
            try:
                content_container = self.query_one(".stts-content", Container)
                if content_container.children:
                    # Get the active widget (should be only one)
                    active_widget = content_container.children[0]
                    if hasattr(active_widget, 'on_button_pressed'):
                        active_widget.on_button_pressed(event)
            except Exception as e:
                logger.debug(f"Could not delegate button event: {e}")

#
# End of STTS_Window.py
#######################################################################################################################