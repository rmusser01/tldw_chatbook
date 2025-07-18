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
from textual.message import Message
from textual import work
from loguru import logger

# Local imports
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSRequestEvent, TTSCompleteEvent
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent, STTSSettingsSaveEvent, STTSAudioBookGenerateEvent
)
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.voice_blend_dialog import VoiceBlendDialog
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, EnhancedFileSave as FileSave
from tldw_chatbook.Third_Party.textual_fspicker import Filters
# Note: Not using form_components due to generator/widget incompatibility

import json

#######################################################################################################################
#
# Classes:

class TTSPlaygroundWidget(Widget):
    """TTS Playground for testing different providers and settings"""
    
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
    
    #elevenlabs-settings.visible {
        display: block;
    }
    
    #chatterbox-settings.visible {
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
    """
    
    def __init__(self):
        super().__init__()
        self.current_audio_file = None
        self.reference_audio_path = None
        
    def compose(self) -> ComposeResult:
        """Compose the TTS Playground UI"""
        with ScrollableContainer(classes="tts-playground-container"):
            yield Label("🎤 TTS Playground", classes="section-title")
            
            # Text input area
            yield Label("Text to Synthesize:")
            yield TextArea(
                id="tts-text-input"
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
                        ("alltalk", "AllTalk (Local)"),
                    ],
                    id="tts-provider-select"
                )
            
            # Voice selection (will be populated based on provider)
            with Horizontal(classes="form-row"):
                yield Label("Voice:", classes="form-label")
                yield Select(
                    options=[("alloy", "Alloy")],
                    id="tts-voice-select"
                )
            
            # Model selection
            with Horizontal(classes="form-row"):
                yield Label("Model:", classes="form-label")
                yield Select(
                    options=[("tts-1", "TTS-1"), ("tts-1-hd", "TTS-1-HD")],
                    id="tts-model-select"
                )
            
            # Language selection (for Kokoro)
            with Horizontal(classes="form-row", id="kokoro-language-row"):
                yield Label("Language:", classes="form-label")
                yield Select(
                    options=[
                        ("a", "American English"),
                        ("b", "British English"),
                        ("j", "Japanese"),
                        ("z", "Mandarin Chinese"),
                        ("e", "Spanish"),
                        ("f", "French"),
                        ("h", "Hindi"),
                        ("i", "Italian"),
                        ("p", "Brazilian Portuguese"),
                    ],
                    id="tts-language-select"
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
            
            # Generate button
            yield Button("🔊 Generate Speech", id="tts-generate-btn", variant="primary")
            
            # Audio player placeholder
            with Container(id="audio-player-container", classes="audio-player"):
                yield Static("Audio player will appear here after generation", id="audio-player-status")
                with Horizontal():
                    yield Button("▶️ Play", id="audio-play-btn", disabled=True)
                    yield Button("⏸️ Pause", id="audio-pause-btn", disabled=True)
                    yield Button("⏹️ Stop", id="audio-stop-btn", disabled=True)
                    yield Button("💾 Export", id="audio-export-btn", disabled=True)
            
            # Generation log
            yield Label("Generation Log:")
            yield RichLog(id="tts-generation-log", classes="generation-log", highlight=True, markup=True)
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle provider/model selection changes"""
        if event.select.id == "tts-provider-select":
            self._update_voice_options(event.value)
            self._update_model_options(event.value)
            
            # Show/hide language selection based on provider
            language_row = self.query_one("#kokoro-language-row", Horizontal)
            if event.value == "kokoro":
                language_row.add_class("visible")
            else:
                language_row.remove_class("visible")
            
            # Show/hide ElevenLabs settings
            elevenlabs_settings = self.query_one("#elevenlabs-settings", Vertical)
            if event.value == "elevenlabs":
                elevenlabs_settings.add_class("visible")
            else:
                elevenlabs_settings.remove_class("visible")
            
            # Show/hide Chatterbox settings
            chatterbox_settings = self.query_one("#chatterbox-settings", Vertical)
            if event.value == "chatterbox":
                chatterbox_settings.add_class("visible")
            else:
                chatterbox_settings.remove_class("visible")
    
    def _update_voice_options(self, provider: str) -> None:
        """Update voice options based on provider"""
        voice_select = self.query_one("#tts-voice-select", Select)
        
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
            voice_select.value = "21m00Tcm4TlvDq8ikWAM"
        elif provider == "kokoro":
            voice_select.set_options([
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
            ])
            voice_select.value = "af_bella"
        elif provider == "chatterbox":
            voice_select.set_options([
                ("default", "Default Voice"),
                ("custom", "Custom (Upload Reference)"),
                # Predefined voices will be loaded dynamically if available
            ])
            voice_select.value = "default"
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
            voice_select.value = "female_01.wav"
    
    def _update_model_options(self, provider: str) -> None:
        """Update model options based on provider"""
        model_select = self.query_one("#tts-model-select", Select)
        
        if provider == "openai":
            model_select.set_options([
                ("tts-1", "TTS-1 (Standard)"),
                ("tts-1-hd", "TTS-1-HD (High Quality)"),
            ])
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
            model_select.set_options([
                ("kokoro", "Kokoro 82M"),
            ])
            model_select.value = "kokoro"
        elif provider == "chatterbox":
            model_select.set_options([
                ("chatterbox", "Chatterbox 0.5B"),
            ])
            model_select.value = "chatterbox"
        elif provider == "alltalk":
            model_select.set_options([
                ("alltalk", "AllTalk TTS"),
            ])
            model_select.value = "alltalk"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "tts-generate-btn":
            self._generate_tts()
        elif event.button.id == "audio-play-btn":
            self._play_audio()
        elif event.button.id == "audio-pause-btn":
            self._pause_audio()
        elif event.button.id == "audio-stop-btn":
            self._stop_audio()
        elif event.button.id == "audio-export-btn":
            self._export_audio()
        elif event.button.id == "reference-audio-btn":
            self._select_reference_audio()
        elif event.button.id == "clear-reference-audio-btn":
            self._clear_reference_audio()
    
    def _generate_tts(self) -> None:
        """Generate TTS audio"""
        # Get form values
        text_area = self.query_one("#tts-text-input", TextArea)
        text = text_area.text.strip()
        
        if not text:
            self.app.notify("Please enter text to synthesize", severity="warning")
            return
        
        provider = self.query_one("#tts-provider-select", Select).value
        voice = self.query_one("#tts-voice-select", Select).value
        model = self.query_one("#tts-model-select", Select).value
        speed = float(self.query_one("#tts-speed-input", Input).value or "1.0")
        format = self.query_one("#tts-format-select", Select).value
        
        # Collect provider-specific settings
        extra_params = {}
        if provider == "kokoro":
            language = self.query_one("#tts-language-select", Select).value
            extra_params["language"] = language
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
        
        # Log the request
        log = self.query_one("#tts-generation-log", RichLog)
        log.write(f"[bold blue]Generating TTS...[/bold blue]")
        log.write(f"Provider: {provider}")
        log.write(f"Voice: {voice}")
        log.write(f"Model: {model}")
        log.write(f"Speed: {speed}")
        log.write(f"Format: {format}")
        log.write(f"Text length: {len(text)} characters")
        
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
    
    def _generation_complete(self, success: bool) -> None:
        """Handle TTS generation completion"""
        self.query_one("#tts-generate-btn", Button).disabled = False
        
        if success:
            log = self.query_one("#tts-generation-log", RichLog)
            log.write("[bold green]✓ TTS generation complete![/bold green]")
            
            # Enable audio controls
            self.query_one("#audio-play-btn", Button).disabled = False
            self.query_one("#audio-export-btn", Button).disabled = False
            self.query_one("#audio-player-status", Static).update("Audio ready to play")
            
            self.app.notify("TTS generation complete!", severity="information")
        else:
            log = self.query_one("#tts-generation-log", RichLog)
            log.write("[bold red]✗ TTS generation failed![/bold red]")
            self.app.notify("TTS generation failed", severity="error")
    
    def _play_audio(self) -> None:
        """Play the generated audio"""
        # Call the app's play method if it has the event handler
        if hasattr(self.app, 'play_current_audio'):
            self.app.run_worker(self.app.play_current_audio())
        else:
            self.app.notify("Audio playback not available", severity="warning")
    
    def _pause_audio(self) -> None:
        """Pause audio playback"""
        # TODO: Implement audio pause
        pass
    
    def _stop_audio(self) -> None:
        """Stop audio playback"""
        # TODO: Implement audio stop
        pass
    
    def _export_audio(self) -> None:
        """Export the generated audio"""
        # TODO: Implement audio export
        self.app.notify("Export functionality coming soon!", severity="information")
    
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
            if default_provider in ["openai", "elevenlabs", "kokoro", "chatterbox", "alltalk"]:
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
            self._update_default_voice_model_options(default_provider)
            
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
            device_select = self.query_one("#kokoro-device-select", Select)
            kokoro_device = get_cli_setting("app_tts", "KOKORO_DEVICE_DEFAULT", "cpu")
            if kokoro_device in ["cpu", "cuda"]:
                device_select.value = kokoro_device
            
            # Set Chatterbox device
            chatterbox_device_select = self.query_one("#chatterbox-device-select", Select)
            chatterbox_device = get_cli_setting("app_tts", "CHATTERBOX_DEVICE", "cpu")
            if chatterbox_device in ["cpu", "cuda"]:
                chatterbox_device_select.value = chatterbox_device
            
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
            
            # Load and display Kokoro voice blends
            self._load_kokoro_voice_blends()
            
        except Exception as e:
            logger.warning(f"Failed to set initial values: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "save-settings-btn":
            self._save_settings()
        elif event.button.id == "add-voice-blend-btn":
            self.run_worker(self._show_add_voice_blend_dialog)
        elif event.button.id == "import-blends-btn":
            self._import_voice_blends()
        elif event.button.id == "export-blends-btn":
            self._export_voice_blends()
        elif event.button.id == "kokoro-browse-model-btn":
            self._browse_kokoro_model()
        elif event.button.id == "kokoro-browse-voices-btn":
            self._browse_kokoro_voices()
        elif event.button.id == "chatterbox-browse-voice-dir-btn":
            self._browse_chatterbox_voice_dir()
    
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
            # For now, use a simple file dialog workaround
            import_path = Path.home() / "Downloads" / "kokoro_voice_blends_export.json"
            
            if not import_path.exists():
                self.app.notify(
                    f"Please place your import file at: {import_path}",
                    severity="information"
                )
                return
            
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
            
            # Export to Downloads folder
            export_path = Path.home() / "Downloads" / "kokoro_voice_blends_export.json"
            
            with open(export_path, 'w') as f:
                json.dump(blends, f, indent=2)
            
            self.app.notify(
                f"Exported {len(blends)} voice blend(s) to: {export_path}",
                severity="success"
            )
            
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
            voice_select.set_options([
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
            ])
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
            
            # Chapter detection
            with Collapsible(title="Chapter Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Auto-detect Chapters:", classes="form-label")
                    yield Switch(id="auto-chapters-switch", value=True)
                
                with Horizontal(classes="form-row"):
                    yield Label("Chapter Pattern:", classes="form-label")
                    yield Input(
                        id="chapter-pattern-input",
                        value="Chapter \\d+",
                        placeholder="Regex pattern for chapters"
                    )
            
            # Chapter list
            yield Label("Detected Chapters:")
            yield Static("No chapters detected yet", id="chapter-list", classes="chapter-list")
            
            # Voice assignment
            with Collapsible(title="Voice Assignment", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Narrator Voice:", classes="form-label")
                    yield Select(
                        options=[("alloy", "Alloy")],
                        id="narrator-voice-select"
                )
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Multi-voice:", classes="form-label")
                    yield Switch(id="multi-voice-switch", value=False)
                
                yield Label("Character Voices:")
                yield Static("Configure character voices here", id="character-voices")
            
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
                        value="m4b",
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
        try:
            # Set audiobook provider
            provider_select = self.query_one("#audiobook-provider-select", Select)
            default_provider = get_cli_setting("app_tts", "default_provider", "openai")
            if default_provider in ["openai", "elevenlabs", "kokoro", "chatterbox"]:
                provider_select.value = default_provider
        except Exception as e:
            logger.warning(f"Failed to set audiobook provider: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "import-content-btn":
            self._import_content()
        elif event.button.id == "generate-audiobook-btn":
            self._generate_audiobook()
        elif event.button.id == "audiobook-export-btn":
            self._export_audiobook()
    
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
        # TODO: Implement note selection dialog
        self.app.notify("Note import will be implemented when note selection dialog is ready", severity="information")
    
    def _import_from_conversation(self) -> None:
        """Import content from conversation"""
        # TODO: Implement conversation selection dialog
        self.app.notify("Conversation import will be implemented when conversation selection dialog is ready", severity="information")
    
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
    
    def _detect_chapters(self) -> None:
        """Detect chapters in the content"""
        if not self.content_text:
            return
        
        try:
            from tldw_chatbook.TTS.audiobook_generator import ChapterDetector
            
            # Detect chapters
            self.detected_chapters = ChapterDetector.detect_chapters(self.content_text)
            
            # Update chapter list display
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
        multi_voice = self.query_one("#multi-voice-switch", Switch).value
        include_chapters = self.query_one("#chapter-markers-switch", Switch).value
        background_music = self.query_one("#background-music-switch", Switch).value
        
        # Get title from first chapter or use default
        title = "Untitled AudioBook"
        if self.detected_chapters:
            # Use book title if detected, otherwise use first chapter
            for chapter in self.detected_chapters:
                if "title" in chapter.title.lower() or chapter.number == 1:
                    title = chapter.title
                    break
        
        # Prepare options
        options = {
            "title": title,
            "author": "Unknown",
            "provider": provider,
            "model": self._get_model_for_provider(provider),
            "chapter_detection": include_chapters,
            "multi_voice": multi_voice,
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
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes"""
        if event.select.id == "audiobook-provider-select":
            # Update narrator voice options based on provider
            self._update_voice_options(event.value)
            # Update cost estimate
            if self.content_text:
                self._estimate_cost(event.value, len(self.content_text))
    
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
            voice_select.set_options([
                ("af", "Afrikaans"),
                ("en", "English"), 
                ("es", "Spanish"),
                ("fr", "French"),
                ("de", "German"),
            ])
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
            
            # Future features
            yield Rule()
            yield Label("Coming Soon:", classes="section-title")
            yield Button("🎙️ Voice Cloning", id="view-voice-cloning-btn", classes="sidebar-button", disabled=True)
            yield Button("🔤 Speech Recognition", id="view-stt-btn", classes="sidebar-button", disabled=True)
            yield Button("🎵 Audio Effects", id="view-effects-btn", classes="sidebar-button", disabled=True)
        
        # Content area
        with Container(classes="stts-content"):
            # Show playground by default
            yield TTSPlaygroundWidget()
    
    def watch_current_view(self, old_view: str, new_view: str) -> None:
        """Handle view changes"""
        # Remove old content
        content_container = self.query_one(".stts-content", Container)
        content_container.remove_children()
        
        # Add new content based on view
        if new_view == "playground":
            content_container.mount(TTSPlaygroundWidget())
        elif new_view == "settings":
            content_container.mount(TTSSettingsWidget())
        elif new_view == "audiobook":
            content_container.mount(AudioBookGenerationWidget())
        
        # Update button variants
        for btn in self.query(".sidebar-button").results(Button):
            btn.variant = "default"
        
        if new_view == "playground":
            self.query_one("#view-playground-btn", Button).variant = "primary"
        elif new_view == "settings":
            self.query_one("#view-settings-btn", Button).variant = "primary"
        elif new_view == "audiobook":
            self.query_one("#view-audiobook-btn", Button).variant = "primary"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses"""
        if event.button.id == "view-playground-btn":
            self.current_view = "playground"
        elif event.button.id == "view-settings-btn":
            self.current_view = "settings"
        elif event.button.id == "view-audiobook-btn":
            self.current_view = "audiobook"
        elif event.button.id == "view-voice-cloning-btn":
            self.app.notify("Voice Cloning coming soon!", severity="information")
        elif event.button.id == "view-stt-btn":
            self.app.notify("Speech Recognition coming soon!", severity="information")
        elif event.button.id == "view-effects-btn":
            self.app.notify("Audio Effects coming soon!", severity="information")

#
# End of STTS_Window.py
#######################################################################################################################