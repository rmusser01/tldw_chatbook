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
# Note: Not using form_components due to generator/widget incompatibility

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
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFilePicker
        
        # Create file picker for audio files
        file_picker = EnhancedFilePicker(
            title="Select Reference Audio",
            filters=[
                ("Audio Files", "*.wav,*.mp3,*.m4a,*.flac,*.aac"),
                ("All Files", "*"),
            ],
            select_type="file"
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
                        ],
                        id="default-provider-select"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Default Voice:", classes="form-label")
                    yield Input(
                        id="default-voice-input",
                        value=get_cli_setting("app_tts", "default_voice", "alloy"),
                        placeholder="Default voice ID"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Default Model:", classes="form-label")
                    yield Input(
                        id="default-model-input",
                        value=get_cli_setting("app_tts", "default_model", "tts-1"),
                        placeholder="Default model"
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
                    yield Label("Voice Stability:", classes="form-label")
                    yield Input(
                        id="elevenlabs-stability-input",
                        value="0.5",
                        placeholder="0.0-1.0",
                        type="number"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Similarity Boost:", classes="form-label")
                    yield Input(
                        id="elevenlabs-similarity-input",
                        value="0.8",
                        placeholder="0.0-1.0",
                        type="number"
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
                        value=get_cli_setting("app_tts", "kokoro_use_onnx", True)
                )
                
                with Horizontal(classes="form-row"):
                    yield Label("Model Path:", classes="form-label")
                    yield Input(
                        id="kokoro-model-path-input",
                        placeholder="Path to Kokoro model"
                    )
            
            # Save button
            yield Button("💾 Save Settings", id="save-settings-btn", variant="primary")
    
    def on_mount(self) -> None:
        """Set initial values from config after mount"""
        try:
            # Set default provider
            provider_select = self.query_one("#default-provider-select", Select)
            default_provider = get_cli_setting("app_tts", "default_provider", "openai")
            if default_provider in ["openai", "elevenlabs", "kokoro"]:
                provider_select.value = default_provider
            
            # Set default format
            format_select = self.query_one("#default-format-select", Select)
            default_format = get_cli_setting("app_tts", "default_format", "mp3")
            if default_format in ["mp3", "opus", "aac", "flac", "wav"]:
                format_select.value = default_format
            
            # Set Kokoro device
            device_select = self.query_one("#kokoro-device-select", Select)
            kokoro_device = get_cli_setting("app_tts", "kokoro_device", "cpu")
            if kokoro_device in ["cpu", "cuda"]:
                device_select.value = kokoro_device
        except Exception as e:
            logger.warning(f"Failed to set initial values: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle save button press"""
        if event.button.id == "save-settings-btn":
            self._save_settings()
    
    def _save_settings(self) -> None:
        """Save TTS settings"""
        try:
            # Collect all settings
            settings = {}
            
            # Default settings
            settings["default_provider"] = self.query_one("#default-provider-select", Select).value
            settings["default_voice"] = self.query_one("#default-voice-input", Input).value
            settings["default_model"] = self.query_one("#default-model-input", Input).value
            settings["default_format"] = self.query_one("#default-format-select", Select).value
            settings["default_speed"] = float(self.query_one("#default-speed-input", Input).value or "1.0")
            
            # OpenAI settings
            openai_key = self.query_one("#openai-api-key-input", Input).value
            if openai_key:
                settings["openai_api_key"] = openai_key
            
            # ElevenLabs settings
            elevenlabs_key = self.query_one("#elevenlabs-api-key-input", Input).value
            if elevenlabs_key:
                settings["elevenlabs_api_key"] = elevenlabs_key
            
            stability = self.query_one("#elevenlabs-stability-input", Input).value
            if stability:
                settings["elevenlabs_voice_stability"] = float(stability)
            
            similarity = self.query_one("#elevenlabs-similarity-input", Input).value
            if similarity:
                settings["elevenlabs_similarity_boost"] = float(similarity)
            
            # Kokoro settings
            settings["kokoro_device"] = self.query_one("#kokoro-device-select", Select).value
            settings["kokoro_use_onnx"] = self.query_one("#kokoro-use-onnx-switch", Switch).value
            
            model_path = self.query_one("#kokoro-model-path-input", Input).value
            if model_path:
                settings["kokoro_model_path"] = model_path
            
            # Post save event
            self.app.post_message(STTSSettingsSaveEvent(settings))
            
        except Exception as e:
            logger.error(f"Failed to collect settings: {e}")
            self.app.notify(f"Failed to save settings: {e}", severity="error")


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
                        value=get_cli_setting("app_tts", "default_provider", "openai"),
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
            from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFilePicker
            
            # Create file picker for text files
            file_picker = EnhancedFilePicker(
                title="Select Text File for AudioBook",
                filters=[
                    ("Text Files", "*.txt,*.md,*.rst"),
                    ("eBook Files", "*.epub,*.mobi"),
                    ("All Files", "*"),
                ],
                select_type="file"
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
            from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFilePicker
            
            # Create file picker for save location
            file_picker = EnhancedFilePicker(
                title="Save AudioBook As",
                filters=[
                    ("AudioBook Files", "*.m4b,*.mp3"),
                    ("All Files", "*"),
                ],
                select_type="save",
                default_filename=self.generated_audiobook_path.name
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