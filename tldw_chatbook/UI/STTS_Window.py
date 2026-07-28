# STTS_Window.py
# Description: S/TT/S (Speech/Text-to-Speech) tab with TTS Playground, Settings, and AudioBook/Podcast Generation
#
# Imports
import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import (
    Label,
    Button,
    TextArea,
    Select,
    Input,
    Static,
    RichLog,
    Switch,
    Collapsible,
    Rule,
)
from textual.css.query import QueryError
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding
from textual import on, work
from loguru import logger
from rich.text import Text

# Local imports
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent,
    STTSSettingsSaveEvent,
    STTSAudioBookGenerateEvent,
)
from tldw_chatbook.TTS import (
    STTSGeneratedAudio,
    STTSPlaygroundRequest,
    TTSPreferencesSnapshot,
    get_tts_service,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSOperationError,
    TTSProviderCatalog,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.legacy_catalogs import (
    LEGACY_DEFAULT_MODELS,
    LEGACY_DEFAULT_VOICES,
    LEGACY_VOICE_OPTIONS,
)
from tldw_chatbook.UI.stts_playground_catalog import (
    AUDIO_CPP_PROVIDER_ID,
    CatalogRequestToken,
    FIRST_AVAILABLE_MODEL_ID,
    LOADING_SELECT_VALUE,
    PlaygroundControls,
    SERVER_DEFAULT_VOICE_ID,
    UNAVAILABLE_SELECT_VALUE,
    SelectSentinel,
    SelectValue,
    controls_from_catalog,
    provider_options,
    voice_id_for_request,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.destination_recovery import optional_dependency_recovery_state
from tldw_chatbook.Widgets.voice_blend_dialog import VoiceBlendDialog
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen as FileOpen,
    EnhancedFileSave as FileSave,
)
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.UI.Dictation_Window_Improved import (
    ImprovedDictationWindow as DictationWindow,
)
from tldw_chatbook.Utils.optional_deps import (
    DEPENDENCIES_AVAILABLE,
    check_stt_deps,
    check_tts_deps,
)
# Note: Not using form_components due to generator/widget incompatibility

import json

#######################################################################################################################
#
# Classes:





































































class TTSSettingsWidget(Widget):
    """TTS Settings for global configuration"""

    OPENAI_TTS_DEFAULT_URL = "https://api.openai.com/v1/audio/speech"

    # Store file paths
    kokoro_model_path = reactive("")
    kokoro_voices_path = reactive("")
    chatterbox_voice_dir = reactive("")

    def __init__(self) -> None:
        super().__init__()
        self._audio_cpp_discovery_generation = 0
        self._preferences_snapshot: TTSPreferencesSnapshot | None = None

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

    @staticmethod
    def _load_audio_cpp_config() -> AudioCppConfig:
        """Load a safe external audio.cpp form snapshot."""
        candidate = get_cli_setting("app_tts", "audio_cpp", {})
        try:
            values = candidate if isinstance(candidate, Mapping) else {}
            return AudioCppConfig.from_mapping(values)
        except (TypeError, ValueError):
            logger.warning("Stored audio.cpp settings are invalid; using defaults")
            return AudioCppConfig()

    def compose(self) -> ComposeResult:
        """Compose the TTS Settings UI"""
        audio_cpp_config = self._load_audio_cpp_config()
        with ScrollableContainer(classes="tts-settings-container"):
            yield Label("⚙️ TTS Settings", classes="section-title")

            # Default provider settings
            with Collapsible(
                title="Default Provider Settings", classes="settings-section"
            ):
                with Horizontal(classes="form-row"):
                    yield Label("Default Provider:", classes="form-label")
                    yield Select(
                        options=[
                            ("OpenAI", "openai"),
                            ("audio.cpp (External Server)", "audio_cpp"),
                            ("ElevenLabs", "elevenlabs"),
                            ("Kokoro (Local)", "kokoro"),
                            ("Chatterbox (Local)", "chatterbox"),
                            ("Higgs Audio (Local)", "higgs"),
                            ("AllTalk (Local Server)", "alltalk"),
                        ],
                        id="default-provider-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Default Voice:", classes="form-label")
                    yield Select(
                        options=[
                            ("Alloy", "alloy")
                        ],  # Will be updated based on provider
                        id="default-voice-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Default Model:", classes="form-label")
                    yield Select(
                        options=[
                            ("TTS-1", "tts-1")
                        ],  # Will be updated based on provider
                        id="default-model-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Default Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("MP3", "mp3"),
                            ("Opus", "opus"),
                            ("AAC", "aac"),
                            ("FLAC", "flac"),
                            ("WAV", "wav"),
                        ],
                        id="default-format-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Default Speed:", classes="form-label")
                    yield Input(
                        id="default-speed-input",
                        value=str(get_cli_setting("app_tts", "default_speed", 1.0)),
                        placeholder="0.25-4.0",
                        type="number",
                    )

            with Collapsible(
                title="audio.cpp External Server",
                id="audio-cpp-settings",
                classes="settings-section",
                collapsed=False,
            ):
                with Horizontal(classes="form-row"):
                    yield Label("Mode:", classes="form-label")
                    yield Static("External", id="audio-cpp-mode-value")

                with Horizontal(classes="form-row"):
                    yield Label("Base URL:", classes="form-label")
                    yield Input(
                        id="audio-cpp-base-url-input",
                        value=audio_cpp_config.base_url,
                        placeholder="http://127.0.0.1:8080",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Connect timeout:", classes="form-label")
                    yield Input(
                        id="audio-cpp-connect-timeout-input",
                        value=str(audio_cpp_config.connect_timeout_seconds),
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Synthesis timeout:", classes="form-label")
                    yield Input(
                        id="audio-cpp-synthesis-timeout-input",
                        value=str(audio_cpp_config.synthesis_timeout_seconds),
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max input characters:", classes="form-label")
                    yield Input(
                        id="audio-cpp-max-input-characters-input",
                        value=str(audio_cpp_config.max_input_characters),
                        type="integer",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max response bytes:", classes="form-label")
                    yield Input(
                        id="audio-cpp-max-response-bytes-input",
                        value=str(audio_cpp_config.max_response_bytes),
                        type="integer",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max metadata bytes:", classes="form-label")
                    yield Input(
                        id="audio-cpp-max-metadata-bytes-input",
                        value=str(audio_cpp_config.max_metadata_bytes),
                        type="integer",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max catalog models:", classes="form-label")
                    yield Input(
                        id="audio-cpp-max-catalog-models-input",
                        value=str(audio_cpp_config.max_catalog_models),
                        type="integer",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max voices per model:", classes="form-label")
                    yield Input(
                        id="audio-cpp-max-voices-per-model-input",
                        value=str(audio_cpp_config.max_voices_per_model),
                        type="integer",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max identifier chars:", classes="form-label")
                    yield Input(
                        id="audio-cpp-max-identifier-characters-input",
                        value=str(audio_cpp_config.max_identifier_characters),
                        type="integer",
                    )

                yield Static(
                    "External synthesis sends submitted text to the configured "
                    "server. Save changes before testing.",
                    id="audio-cpp-privacy-notice",
                )
                with Horizontal(classes="form-row"):
                    yield Button(
                        "Test Connection",
                        id="audio-cpp-test-connection-btn",
                    )
                    yield Button(
                        "Refresh Models",
                        id="audio-cpp-refresh-models-btn",
                    )
                yield Static(
                    "Not checked",
                    id="audio-cpp-discovery-status",
                )

            # OpenAI settings
            with Collapsible(title="OpenAI Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("API Key:", classes="form-label")
                    yield Input(
                        id="openai-api-key-input", password=True, placeholder="sk-..."
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Base URL:", classes="form-label")
                    yield Input(
                        id="openai-base-url-input",
                        value=get_cli_setting(
                            "app_tts",
                            "OPENAI_BASE_URL",
                            "https://api.openai.com/v1/audio/speech",
                        ),
                        placeholder="Custom API endpoint (optional)",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Organization ID:", classes="form-label")
                    yield Input(
                        id="openai-org-id-input",
                        value=get_cli_setting("app_tts", "OPENAI_ORG_ID", ""),
                        placeholder="org-... (optional)",
                    )

            # ElevenLabs settings
            with Collapsible(title="ElevenLabs Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("API Key:", classes="form-label")
                    yield Input(
                        id="elevenlabs-api-key-input",
                        password=True,
                        placeholder="Your ElevenLabs API key",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Model:", classes="form-label")
                    yield Select(
                        options=[
                            ("Multilingual v2", "eleven_multilingual_v2"),
                            ("Turbo v2", "eleven_turbo_v2"),
                            ("Multilingual v1", "eleven_multilingual_v1"),
                            ("Monolingual v1", "eleven_monolingual_v1"),
                        ],
                        id="elevenlabs-model-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Output Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("MP3 192kbps", "mp3_44100_192"),
                            ("MP3 128kbps", "mp3_44100_128"),
                            ("MP3 96kbps", "mp3_44100_96"),
                            ("MP3 64kbps", "mp3_44100_64"),
                            ("MP3 32kbps", "mp3_44100_32"),
                            ("PCM 44.1kHz", "pcm_44100"),
                            ("PCM 24kHz", "pcm_24000"),
                            ("PCM 16kHz", "pcm_16000"),
                            ("μ-law 8kHz", "ulaw_8000"),
                        ],
                        id="elevenlabs-format-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Voice Stability:", classes="form-label")
                    yield Input(
                        id="elevenlabs-stability-input",
                        value=str(
                            get_cli_setting(
                                "app_tts", "ELEVENLABS_VOICE_STABILITY", "0.5"
                            )
                        ),
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Similarity Boost:", classes="form-label")
                    yield Input(
                        id="elevenlabs-similarity-input",
                        value=str(
                            get_cli_setting(
                                "app_tts", "ELEVENLABS_SIMILARITY_BOOST", "0.8"
                            )
                        ),
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Style:", classes="form-label")
                    yield Input(
                        id="elevenlabs-style-input",
                        value=str(
                            get_cli_setting("app_tts", "ELEVENLABS_STYLE", "0.0")
                        ),
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Speaker Boost:", classes="form-label")
                    yield Switch(
                        id="elevenlabs-speaker-boost-switch",
                        value=get_cli_setting(
                            "app_tts", "ELEVENLABS_USE_SPEAKER_BOOST", True
                        ),
                    )

            # Kokoro settings
            with Collapsible(title="Kokoro Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Device:", classes="form-label")
                    yield Select(
                        options=[
                            ("CPU", "cpu"),
                            ("CUDA (GPU)", "cuda"),
                        ],
                        id="kokoro-device-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Use ONNX:", classes="form-label")
                    yield Switch(
                        id="kokoro-use-onnx-switch",
                        value=get_cli_setting("app_tts", "KOKORO_USE_ONNX", True),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Model Path:", classes="form-label")
                    yield Button(
                        "📁 Select model file",
                        id="kokoro-browse-model-btn",
                        variant="default",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Voices JSON:", classes="form-label")
                    yield Button(
                        "📁 Select voices.json",
                        id="kokoro-browse-voices-btn",
                        variant="default",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max Tokens:", classes="form-label")
                    yield Input(
                        id="kokoro-max-tokens-input",
                        value=str(
                            get_cli_setting("app_tts", "KOKORO_MAX_TOKENS", "500")
                        ),
                        placeholder="Max tokens per chunk",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable Voice Mixing:", classes="form-label")
                    yield Switch(
                        id="kokoro-voice-mixing-switch",
                        value=get_cli_setting(
                            "app_tts", "KOKORO_ENABLE_VOICE_MIXING", False
                        ),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Performance Tracking:", classes="form-label")
                    yield Switch(
                        id="kokoro-performance-switch",
                        value=get_cli_setting(
                            "app_tts", "KOKORO_TRACK_PERFORMANCE", True
                        ),
                    )

                # Voice blends section
                yield Label("Voice Blends:", classes="form-label")
                with ScrollableContainer(classes="voice-blends-container"):
                    yield Static(
                        id="kokoro-voice-blends-list", classes="voice-blends-list"
                    )
                with Horizontal(classes="form-row"):
                    yield Button(
                        "➕ Add Blend", id="add-voice-blend-btn", variant="default"
                    )
                    yield Button("📥 Import", id="import-blends-btn", variant="default")
                    yield Button("📤 Export", id="export-blends-btn", variant="default")

            # Chatterbox settings
            with Collapsible(title="Chatterbox Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Device:", classes="form-label")
                    yield Select(
                        options=[
                            ("CPU", "cpu"),
                            ("CUDA (GPU)", "cuda"),
                        ],
                        id="chatterbox-device-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Voice Directory:", classes="form-label")
                    yield Button(
                        "📁 Select voice directory",
                        id="chatterbox-browse-voice-dir-btn",
                        variant="default",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Emotion Exaggeration:", classes="form-label")
                    yield Input(
                        id="chatterbox-exaggeration-input",
                        value=str(
                            get_cli_setting("app_tts", "CHATTERBOX_EXAGGERATION", "0.5")
                        ),
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("CFG Weight:", classes="form-label")
                    yield Input(
                        id="chatterbox-cfg-weight-input",
                        value=str(
                            get_cli_setting("app_tts", "CHATTERBOX_CFG_WEIGHT", "0.5")
                        ),
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        id="chatterbox-temperature-input",
                        value=str(
                            get_cli_setting("app_tts", "CHATTERBOX_TEMPERATURE", "0.5")
                        ),
                        placeholder="0.0-2.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Chunk Size:", classes="form-label")
                    yield Input(
                        id="chatterbox-chunk-size-input",
                        value=str(
                            get_cli_setting("app_tts", "CHATTERBOX_CHUNK_SIZE", "1024")
                        ),
                        placeholder="Audio chunk size",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Random Seed:", classes="form-label")
                    yield Input(
                        id="chatterbox-seed-input",
                        value=get_cli_setting("app_tts", "CHATTERBOX_RANDOM_SEED", ""),
                        placeholder="Random seed (optional)",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Number of Candidates:", classes="form-label")
                    yield Input(
                        id="chatterbox-candidates-input",
                        value=str(
                            get_cli_setting("app_tts", "CHATTERBOX_NUM_CANDIDATES", "1")
                        ),
                        placeholder="1-5",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Whisper Validation:", classes="form-label")
                    yield Switch(
                        id="chatterbox-whisper-switch",
                        value=get_cli_setting(
                            "app_tts", "CHATTERBOX_VALIDATE_WHISPER", False
                        ),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Text Preprocessing:", classes="form-label")
                    yield Switch(
                        id="chatterbox-preprocess-switch",
                        value=get_cli_setting(
                            "app_tts", "CHATTERBOX_PREPROCESS_TEXT", True
                        ),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Audio Normalization:", classes="form-label")
                    yield Switch(
                        id="chatterbox-normalize-switch",
                        value=get_cli_setting(
                            "app_tts", "CHATTERBOX_NORMALIZE_AUDIO", True
                        ),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Target dB:", classes="form-label")
                    yield Input(
                        id="chatterbox-target-db-input",
                        value=str(
                            get_cli_setting("app_tts", "CHATTERBOX_TARGET_DB", "-20.0")
                        ),
                        placeholder="-40 to 0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max Text Chunk:", classes="form-label")
                    yield Input(
                        id="chatterbox-max-chunk-input",
                        value=str(
                            get_cli_setting(
                                "app_tts", "CHATTERBOX_MAX_CHUNK_SIZE", "500"
                            )
                        ),
                        placeholder="Max characters per chunk",
                        type="number",
                    )

                # Streaming settings subsection
                yield Label("Streaming Settings:", classes="subsection-label")

                with Horizontal(classes="form-row"):
                    yield Label("Enable Streaming:", classes="form-label")
                    yield Switch(
                        id="chatterbox-streaming-switch",
                        value=get_cli_setting("app_tts", "CHATTERBOX_STREAMING", True),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Stream Chunk Size:", classes="form-label")
                    yield Input(
                        id="chatterbox-stream-chunk-input",
                        value=str(
                            get_cli_setting(
                                "app_tts", "CHATTERBOX_STREAM_CHUNK_SIZE", "4096"
                            )
                        ),
                        placeholder="Stream chunk size",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable Crossfade:", classes="form-label")
                    yield Switch(
                        id="chatterbox-crossfade-switch",
                        value=get_cli_setting(
                            "app_tts", "CHATTERBOX_ENABLE_CROSSFADE", True
                        ),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Crossfade Duration:", classes="form-label")
                    yield Input(
                        id="chatterbox-crossfade-ms-input",
                        value=str(
                            get_cli_setting("app_tts", "CHATTERBOX_CROSSFADE_MS", "50")
                        ),
                        placeholder="Duration in ms",
                        type="number",
                    )

            # Higgs Audio settings
            with Collapsible(title="Higgs Audio Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Model Path:", classes="form-label")
                    yield Input(
                        id="higgs-model-path-input",
                        value=get_cli_setting(
                            "HiggsSettings",
                            "model_path",
                            "bosonai/higgs-audio-v2-generation-3B-base",
                        ),
                        placeholder="Model path or HuggingFace ID",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Voice Samples Dir:", classes="form-label")
                    yield Input(
                        id="higgs-voices-dir-input",
                        value=str(
                            get_cli_setting(
                                "HiggsSettings",
                                "voice_samples_dir",
                                "~/.config/tldw_cli/higgs_voices",
                            )
                        ),
                        placeholder="Path to voice samples",
                    )
                    yield Button(
                        "📁", id="higgs-voices-browse-btn", classes="path-browse-button"
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Device:", classes="form-label")
                    yield Select(
                        options=[
                            ("Auto-detect", "auto"),
                            ("CPU", "cpu"),
                            ("CUDA (GPU)", "cuda"),
                            ("CUDA Device 0", "cuda:0"),
                            ("CUDA Device 1", "cuda:1"),
                        ],
                        id="higgs-device-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable Flash Attention:", classes="form-label")
                    yield Switch(
                        id="higgs-flash-attn-switch",
                        value=get_cli_setting(
                            "HiggsSettings", "enable_flash_attn", True
                        ),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Data Type:", classes="form-label")
                    yield Select(
                        options=[
                            ("Float32 (Full precision)", "float32"),
                            ("Float16 (Half precision)", "float16"),
                            ("BFloat16 (Better range)", "bfloat16"),
                        ],
                        id="higgs-dtype-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Max Reference Duration:", classes="form-label")
                    yield Input(
                        id="higgs-max-ref-duration-input",
                        value=str(
                            get_cli_setting(
                                "HiggsSettings", "max_reference_duration", "30"
                            )
                        ),
                        placeholder="Seconds (e.g., 30)",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Default Language:", classes="form-label")
                    yield Select(
                        options=[
                            ("English", "en"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("German", "de"),
                            ("Italian", "it"),
                            ("Portuguese", "pt"),
                            ("Russian", "ru"),
                            ("Chinese", "zh"),
                            ("Japanese", "ja"),
                            ("Korean", "ko"),
                        ],
                        id="higgs-language-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable Voice Cloning:", classes="form-label")
                    yield Switch(
                        id="higgs-voice-cloning-switch",
                        value=get_cli_setting(
                            "HiggsSettings", "enable_voice_cloning", True
                        ),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable Multi-speaker:", classes="form-label")
                    yield Switch(
                        id="higgs-multi-speaker-switch",
                        value=get_cli_setting(
                            "HiggsSettings", "enable_multi_speaker", True
                        ),
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Speaker Delimiter:", classes="form-label")
                    yield Input(
                        id="higgs-delimiter-input",
                        value=get_cli_setting(
                            "HiggsSettings", "speaker_delimiter", "|||"
                        ),
                        placeholder="Default: |||",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Performance Tracking:", classes="form-label")
                    yield Switch(
                        id="higgs-track-performance-switch",
                        value=get_cli_setting(
                            "HiggsSettings", "track_performance", True
                        ),
                    )

                # Generation parameters
                yield Label("Generation Parameters:", classes="subsection-label")

                with Horizontal(classes="form-row"):
                    yield Label("Max New Tokens:", classes="form-label")
                    yield Input(
                        id="higgs-max-tokens-input",
                        value=str(
                            get_cli_setting("HiggsSettings", "max_new_tokens", "4096")
                        ),
                        placeholder="Max tokens to generate",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        id="higgs-temperature-input",
                        value=str(
                            get_cli_setting("HiggsSettings", "temperature", "0.7")
                        ),
                        placeholder="0.0-2.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Top P:", classes="form-label")
                    yield Input(
                        id="higgs-top-p-input",
                        value=str(get_cli_setting("HiggsSettings", "top_p", "0.9")),
                        placeholder="0.0-1.0",
                        type="number",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Repetition Penalty:", classes="form-label")
                    yield Input(
                        id="higgs-repetition-penalty-input",
                        value=str(
                            get_cli_setting(
                                "HiggsSettings", "repetition_penalty", "1.1"
                            )
                        ),
                        placeholder="1.0 = no penalty",
                        type="number",
                    )

            # AllTalk settings
            with Collapsible(title="AllTalk Settings", classes="settings-section"):
                with Horizontal(classes="form-row"):
                    yield Label("Server URL:", classes="form-label")
                    yield Input(
                        id="alltalk-url-input",
                        value=get_cli_setting(
                            "app_tts",
                            "ALLTALK_TTS_URL_DEFAULT",
                            "http://127.0.0.1:7851",
                        ),
                        placeholder="AllTalk server URL",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Voice:", classes="form-label")
                    yield Input(
                        id="alltalk-voice-input",
                        value=get_cli_setting(
                            "app_tts", "ALLTALK_TTS_VOICE_DEFAULT", "female_01.wav"
                        ),
                        placeholder="Voice file name",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Language:", classes="form-label")
                    yield Select(
                        options=[
                            ("English", "en"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("German", "de"),
                            ("Italian", "it"),
                            ("Portuguese", "pt"),
                            ("Russian", "ru"),
                            ("Chinese", "zh"),
                            ("Japanese", "ja"),
                            ("Korean", "ko"),
                        ],
                        id="alltalk-language-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Output Format:", classes="form-label")
                    yield Select(
                        options=[
                            ("WAV", "wav"),
                            ("MP3", "mp3"),
                            ("Opus", "opus"),
                            ("FLAC", "flac"),
                        ],
                        id="alltalk-format-select",
                    )

            # Save button
            yield Button("💾 Save Settings", id="save-settings-btn", variant="primary")

    def on_mount(self) -> None:
        """Set initial values from config after mount"""
        self.call_after_refresh(self._set_initial_values)

    def _set_initial_values(self) -> None:
        """Apply config values after all child Select widgets have mounted."""
        try:
            default_provider = get_cli_setting(
                "app_tts",
                "default_provider",
                "openai",
            )
            is_audio_cpp = default_provider == AUDIO_CPP_PROVIDER_ID
            preference_values: dict[str, object] = {
                "default_provider": default_provider,
                "default_model": get_cli_setting(
                    "app_tts",
                    "default_model",
                    "" if is_audio_cpp else "tts-1",
                ),
                "default_voice": get_cli_setting(
                    "app_tts",
                    "default_voice",
                    "" if is_audio_cpp else "alloy",
                ),
                "default_format": get_cli_setting(
                    "app_tts",
                    "default_format",
                    "wav" if is_audio_cpp else "mp3",
                ),
                "default_speed": get_cli_setting(
                    "app_tts",
                    "default_speed",
                    1.0,
                ),
            }
            missing_mode = object()
            for mode_key in ("default_model_mode", "default_voice_mode"):
                mode = get_cli_setting("app_tts", mode_key, missing_mode)
                if mode is not missing_mode:
                    preference_values[mode_key] = mode
            preferences = TTSPreferencesSnapshot.from_settings(
                {"app_tts": preference_values}
            )
            self._preferences_snapshot = preferences

            # Set default provider
            provider_select = self.query_one("#default-provider-select", Select)
            if preferences.provider_id in [
                "openai",
                "audio_cpp",
                "elevenlabs",
                "kokoro",
                "chatterbox",
                "higgs",
                "alltalk",
            ]:
                provider_select.value = preferences.provider_id

            # Load voice blends
            self._load_kokoro_voice_blends()

            # Load and display file paths
            self.kokoro_model_path = get_cli_setting(
                "app_tts", "KOKORO_ONNX_MODEL_PATH_DEFAULT", ""
            )
            self.kokoro_voices_path = get_cli_setting(
                "app_tts", "KOKORO_ONNX_VOICES_JSON_DEFAULT", ""
            )
            self.chatterbox_voice_dir = get_cli_setting(
                "app_tts",
                "CHATTERBOX_VOICE_DIR",
                "~/.config/tldw_cli/chatterbox_voices",
            )

            # Update button labels
            self._update_file_button_labels()

            # Update voice and model options based on default provider
            self._update_default_voice_options(preferences.provider_id)
            self._update_default_model_options(preferences.provider_id)

            # Set default voice and model
            default_voice: object = (
                SERVER_DEFAULT_VOICE_ID
                if preferences.voice_mode == "server_default"
                else preferences.voice_id
            )
            default_model: object = (
                FIRST_AVAILABLE_MODEL_ID
                if preferences.model_mode == "first_available"
                else preferences.model_id
            )

            voice_select = self.query_one("#default-voice-select", Select)
            model_select = self.query_one("#default-model-select", Select)

            # Try to set the values if they exist in options
            try:
                if any(opt[1] == default_voice for opt in voice_select._options):
                    voice_select.value = default_voice
            except Exception:
                pass

            try:
                if any(opt[1] == default_model for opt in model_select._options):
                    model_select.value = default_model
            except Exception:
                pass

            # Set default format
            format_select = self.query_one("#default-format-select", Select)
            if preferences.response_format in ["mp3", "opus", "aac", "flac", "wav"]:
                format_select.value = preferences.response_format
            self.query_one("#default-speed-input", Input).value = str(preferences.speed)
            self._update_audio_cpp_default_constraints(preferences.provider_id)

            # Set Kokoro device
            try:
                device_select = self.query_one("#kokoro-device-select", Select)
                kokoro_device = get_cli_setting(
                    "app_tts", "KOKORO_DEVICE_DEFAULT", "cpu"
                )
                if kokoro_device in ["cpu", "cuda"]:
                    device_select.value = kokoro_device
            except Exception as e:
                logger.debug(f"Could not set Kokoro device: {e}")

            # Set Chatterbox device
            try:
                chatterbox_device_select = self.query_one(
                    "#chatterbox-device-select", Select
                )
                chatterbox_device = get_cli_setting(
                    "app_tts", "CHATTERBOX_DEVICE", "cpu"
                )
                if chatterbox_device in ["cpu", "cuda"]:
                    chatterbox_device_select.value = chatterbox_device
            except Exception as e:
                logger.debug(f"Could not set Chatterbox device: {e}")

            # Set ElevenLabs model
            elevenlabs_model_select = self.query_one("#elevenlabs-model-select", Select)
            elevenlabs_model = get_cli_setting(
                "app_tts", "ELEVENLABS_DEFAULT_MODEL", "eleven_multilingual_v2"
            )
            if elevenlabs_model in [
                "eleven_multilingual_v2",
                "eleven_turbo_v2",
                "eleven_multilingual_v1",
                "eleven_monolingual_v1",
            ]:
                elevenlabs_model_select.value = elevenlabs_model

            # Set ElevenLabs format
            elevenlabs_format_select = self.query_one(
                "#elevenlabs-format-select", Select
            )
            elevenlabs_format = get_cli_setting(
                "app_tts", "ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_192"
            )
            if elevenlabs_format in [
                "mp3_44100_192",
                "mp3_44100_128",
                "mp3_44100_96",
                "mp3_44100_64",
                "mp3_44100_32",
                "pcm_44100",
                "pcm_24000",
                "pcm_16000",
                "ulaw_8000",
            ]:
                elevenlabs_format_select.value = elevenlabs_format

            # Set AllTalk language
            alltalk_language_select = self.query_one("#alltalk-language-select", Select)
            alltalk_language = get_cli_setting(
                "app_tts", "ALLTALK_TTS_LANGUAGE_DEFAULT", "en"
            )
            if alltalk_language in [
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "ru",
                "zh",
                "ja",
                "ko",
            ]:
                alltalk_language_select.value = alltalk_language

            # Set AllTalk format
            alltalk_format_select = self.query_one("#alltalk-format-select", Select)
            alltalk_format = get_cli_setting(
                "app_tts", "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT", "wav"
            )
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
                higgs_language = get_cli_setting(
                    "HiggsSettings", "default_language", "en"
                )
                if higgs_language in [
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "ru",
                    "zh",
                    "ja",
                    "ko",
                ]:
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
        elif event.button.id == "audio-cpp-test-connection-btn":
            self._discover_audio_cpp("test")
            event.stop()
        elif event.button.id == "audio-cpp-refresh-models-btn":
            self._discover_audio_cpp("refresh")
            event.stop()
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

    def _is_valid_voice(self, voice: object) -> bool:
        """Check if a voice value is valid (not a separator)"""
        return bool(voice) and not str(voice).startswith("_separator")

    @on(Select.Changed)
    def on_default_selects_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes"""
        if event.select.id == "default-provider-select":
            # Update voice and model options when provider changes
            self._update_default_voice_options(event.value)
            self._update_default_model_options(event.value)
            self._update_audio_cpp_default_constraints(event.value)
        elif event.select.id == "default-voice-select":
            # Validate voice selection (prevent selecting separators)
            if not self._is_valid_voice(event.value):
                # Find and select the first valid voice
                voice_select = event.select
                for _, value in voice_select._options:
                    if self._is_valid_voice(value):
                        voice_select.value = value
                        break

    def _update_default_voice_options(self, provider: str) -> None:
        """Update default voice options based on provider"""
        voice_select = self.query_one("#default-voice-select", Select)

        if provider == AUDIO_CPP_PROVIDER_ID:
            options = [("Server default", SERVER_DEFAULT_VOICE_ID)]
            selected = SERVER_DEFAULT_VOICE_ID
            preferences = self._preferences_snapshot
            if (
                preferences is not None
                and preferences.provider_id == AUDIO_CPP_PROVIDER_ID
                and preferences.voice_mode == "exact"
                and preferences.voice_id is not None
            ):
                options.append(("Saved server voice", preferences.voice_id))
                selected = preferences.voice_id
            voice_select.set_options(options)
            voice_select.value = selected
        elif provider == "openai":
            voice_select.set_options(
                [
                    ("Alloy", "alloy"),
                    ("Ash", "ash"),
                    ("Ballad", "ballad"),
                    ("Coral", "coral"),
                    ("Echo", "echo"),
                    ("Fable", "fable"),
                    ("Nova", "nova"),
                    ("Onyx", "onyx"),
                    ("Sage", "sage"),
                    ("Shimmer", "shimmer"),
                    ("Verse", "verse"),
                ]
            )
            # Set the saved default or fallback to alloy
            default_voice = get_cli_setting("app_tts", "default_voice", "alloy")
            if default_voice in [
                option[1]
                for option in voice_select._options
                if option[1] != Select.BLANK
            ]:
                voice_select.value = default_voice
            else:
                voice_select.value = "alloy"
        elif provider == "elevenlabs":
            voice_select.set_options(
                [
                    ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
                    ("Domi", "AZnzlk1XvdvUeBnXmlld"),
                    ("Bella", "EXAVITQu4vr4xnSDxMaL"),
                    ("Antoni", "ErXwobaYiN019PkySvjV"),
                    ("Elli", "MF3mGyEYCl7XYWbV9V6O"),
                    ("Josh", "TxGEqnHWrfWFTfGW9XjX"),
                    ("Arnold", "VR6AewLTigWG4xSOukaG"),
                    ("Adam", "pNInz6obpgDQGcFmaJgB"),
                    ("Sam", "yoZ06aMxZJJ28mfd3POQ"),
                ]
            )
            try:
                voice_select.value = "21m00Tcm4TlvDq8ikWAM"
            except Exception as e:
                logger.debug(f"Could not set default ElevenLabs voice value: {e}")
        elif provider == "kokoro":
            logger.info(f"Setting up Kokoro voices for provider: {provider}")
            voice_options = [
                # American Female voices
                ("Alloy (US Female)", "af_alloy"),
                ("Aoede (US Female)", "af_aoede"),
                ("Bella (US Female)", "af_bella"),
                ("Heart (US Female)", "af_heart"),
                ("Jessica (US Female)", "af_jessica"),
                ("Kore (US Female)", "af_kore"),
                ("Nicole (US Female)", "af_nicole"),
                ("Nova (US Female)", "af_nova"),
                ("River (US Female)", "af_river"),
                ("Sarah (US Female)", "af_sarah"),
                ("Sky (US Female)", "af_sky"),
                # American Male voices
                ("Adam (US Male)", "am_adam"),
                ("Michael (US Male)", "am_michael"),
                # British Female voices
                ("Emma (UK Female)", "bf_emma"),
                ("Isabella (UK Female)", "bf_isabella"),
                # British Male voices
                ("George (UK Male)", "bm_george"),
                ("Lewis (UK Male)", "bm_lewis"),
            ]

            # Add saved voice blends
            blend_file = (
                Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            )
            if blend_file.exists():
                try:
                    import json

                    with open(blend_file, "r") as f:
                        blends = json.load(f)
                        if blends:
                            # Add separator
                            voice_options.append(
                                ("──── Voice Blends ────", "_separator")
                            )
                            # Add each blend
                            for blend_name, blend_data in blends.items():
                                display_name = f"🎭 {blend_name}"
                                if blend_data.get("description"):
                                    display_name += (
                                        f" - {blend_data['description'][:30]}"
                                    )
                                voice_options.append(
                                    (display_name, f"blend:{blend_name}")
                                )
                except Exception as e:
                    logger.error(f"Failed to load voice blends: {e}")

            voice_select.set_options(voice_options)

            # Find first valid voice option (skip separators)
            valid_voice = None
            for _, value in voice_options:
                if self._is_valid_voice(value):
                    valid_voice = value
                    break

            if valid_voice:
                voice_select.value = valid_voice
            else:
                voice_select.value = "af_bella"  # Fallback
        elif provider == "chatterbox":
            voice_select.set_options(
                [
                    ("Default Voice", "default"),
                    ("Custom (Upload Reference)", "custom"),
                ]
            )
            voice_select.value = "default"
        elif provider == "higgs":
            voice_select.set_options(
                [
                    ("Professional Female", "professional_female"),
                    ("Warm Female", "warm_female"),
                    ("Storyteller Male", "storyteller_male"),
                    ("Deep Male", "deep_male"),
                    ("Energetic Female", "energetic_female"),
                    ("Soft Female", "soft_female"),
                ]
            )
            voice_select.value = "professional_female"
        elif provider == "alltalk":
            voice_select.set_options(
                [
                    ("Female 01", "female_01.wav"),
                    ("Female 02", "female_02.wav"),
                    ("Female 03", "female_03.wav"),
                    ("Female 04", "female_04.wav"),
                    ("Male 01", "male_01.wav"),
                    ("Male 02", "male_02.wav"),
                    ("Male 03", "male_03.wav"),
                    ("Male 04", "male_04.wav"),
                ]
            )
            voice_select.value = "female_01.wav"

    def _update_default_model_options(self, provider: str) -> None:
        """Update default model options based on provider"""
        model_select = self.query_one("#default-model-select", Select)

        if provider == AUDIO_CPP_PROVIDER_ID:
            options = [
                ("First available server model", FIRST_AVAILABLE_MODEL_ID),
            ]
            selected = FIRST_AVAILABLE_MODEL_ID
            preferences = self._preferences_snapshot
            if (
                preferences is not None
                and preferences.provider_id == AUDIO_CPP_PROVIDER_ID
                and preferences.model_mode == "exact"
                and preferences.model_id is not None
            ):
                options.append(("Saved server model", preferences.model_id))
                selected = preferences.model_id
            model_select.set_options(options)
            model_select.value = selected
        elif provider == "openai":
            model_select.set_options(
                [
                    ("TTS-1 (Standard)", "tts-1"),
                    ("TTS-1-HD (High Quality)", "tts-1-hd"),
                ]
            )
            # Set the saved default or fallback
            default_model = get_cli_setting("app_tts", "default_model", "tts-1")
            if default_model in ["tts-1", "tts-1-hd"]:
                model_select.value = default_model
            else:
                model_select.value = "tts-1"
        elif provider == "elevenlabs":
            model_select.set_options(
                [
                    ("Eleven Monolingual v1", "eleven_monolingual_v1"),
                    ("Eleven Multilingual v1", "eleven_multilingual_v1"),
                    ("Eleven Multilingual v2 (Default)", "eleven_multilingual_v2"),
                    ("Eleven Turbo v2", "eleven_turbo_v2"),
                    ("Eleven Turbo v2.5", "eleven_turbo_v2_5"),
                    ("Eleven Flash v2 (Low Latency)", "eleven_flash_v2"),
                    ("Eleven Flash v2.5 (Ultra Low Latency)", "eleven_flash_v2_5"),
                ]
            )
            model_select.value = "eleven_multilingual_v2"
        elif provider == "kokoro":
            logger.info("Setting Kokoro model options")
            model_select.set_options(
                [
                    ("Kokoro 82M", "kokoro"),
                ]
            )
            model_select.value = "kokoro"
            logger.info("Kokoro model set successfully")
        elif provider == "chatterbox":
            model_select.set_options(
                [
                    ("Chatterbox 0.5B", "chatterbox"),
                ]
            )
            model_select.value = "chatterbox"
        elif provider == "higgs":
            logger.info("Setting Higgs model options")
            model_select.set_options(
                [
                    ("Higgs Audio V2 3B", "higgs-audio-v2"),
                ]
            )
            model_select.value = "higgs-audio-v2"
            logger.info("Higgs model set successfully")
        elif provider == "alltalk":
            model_select.set_options(
                [
                    ("AllTalk TTS", "alltalk"),
                ]
            )
            model_select.value = "alltalk"

    def _update_audio_cpp_default_constraints(self, provider: object) -> None:
        """Mirror native audio.cpp's fixed format and speed without discovery."""
        format_select = self.query_one("#default-format-select", Select)
        speed_input = self.query_one("#default-speed-input", Input)
        is_audio_cpp = provider == AUDIO_CPP_PROVIDER_ID
        format_select.disabled = is_audio_cpp
        speed_input.disabled = is_audio_cpp
        if is_audio_cpp:
            format_select.value = "wav"
            speed_input.value = "1.0"

    def _save_settings(self) -> None:
        """Save TTS settings"""
        try:
            provider = self.query_one("#default-provider-select", Select).value
            if not isinstance(provider, str):
                raise ValueError("A default TTS provider must be selected")

            selected_model = self.query_one("#default-model-select", Select).value
            model_mode: Literal["exact", "first_available"]
            model_id: str | None
            if selected_model is FIRST_AVAILABLE_MODEL_ID:
                model_mode = "first_available"
                model_id = None
            elif isinstance(selected_model, str):
                model_mode = "exact"
                model_id = selected_model
            else:
                raise ValueError("A default TTS model must be selected")

            selected_voice = self.query_one("#default-voice-select", Select).value
            voice_mode: Literal["exact", "server_default"]
            voice_id: str | None
            if selected_voice is SERVER_DEFAULT_VOICE_ID:
                voice_mode = "server_default"
                voice_id = None
            elif isinstance(selected_voice, str):
                voice_mode = "exact"
                voice_id = selected_voice
            else:
                raise ValueError("A default TTS voice must be selected")

            response_format = self.query_one(
                "#default-format-select",
                Select,
            ).value
            if not isinstance(response_format, str):
                raise ValueError("A default TTS format must be selected")
            speed = self._validate_numeric_input(
                self.query_one("#default-speed-input", Input).value,
                0.25,
                4.0,
                1.0,
            )
            preferences = TTSPreferencesSnapshot(
                provider_id=provider,
                model_mode=model_mode,
                model_id=model_id,
                voice_mode=voice_mode,
                voice_id=voice_id,
                response_format=response_format,
                speed=speed,
            )

            # Collect all settings
            settings = {}

            settings["audio_cpp"] = self._collect_audio_cpp_config().to_mapping()

            # OpenAI settings
            openai_key = self.query_one("#openai-api-key-input", Input).value
            if openai_key:
                settings["openai_api_key"] = openai_key

            settings["OPENAI_BASE_URL"] = self._normalize_openai_base_url(
                self.query_one("#openai-base-url-input", Input).value
            )
            org_id = self.query_one("#openai-org-id-input", Input).value.strip()
            if "\r" in org_id or "\n" in org_id:
                raise ValueError("OpenAI organization ID cannot contain line breaks")
            settings["OPENAI_ORG_ID"] = org_id

            # ElevenLabs settings
            elevenlabs_key = self.query_one("#elevenlabs-api-key-input", Input).value
            if elevenlabs_key:
                settings["elevenlabs_api_key"] = elevenlabs_key

            settings["ELEVENLABS_DEFAULT_MODEL"] = self.query_one(
                "#elevenlabs-model-select", Select
            ).value
            settings["ELEVENLABS_OUTPUT_FORMAT"] = self.query_one(
                "#elevenlabs-format-select", Select
            ).value
            settings["ELEVENLABS_VOICE_STABILITY"] = self._validate_numeric_input(
                self.query_one("#elevenlabs-stability-input", Input).value,
                0.0,
                1.0,
                0.5,
            )
            settings["ELEVENLABS_SIMILARITY_BOOST"] = self._validate_numeric_input(
                self.query_one("#elevenlabs-similarity-input", Input).value,
                0.0,
                1.0,
                0.8,
            )
            settings["ELEVENLABS_STYLE"] = self._validate_numeric_input(
                self.query_one("#elevenlabs-style-input", Input).value, 0.0, 1.0, 0.0
            )
            settings["ELEVENLABS_USE_SPEAKER_BOOST"] = self.query_one(
                "#elevenlabs-speaker-boost-switch", Switch
            ).value

            # Kokoro settings
            settings["KOKORO_DEVICE_DEFAULT"] = self.query_one(
                "#kokoro-device-select", Select
            ).value
            settings["KOKORO_USE_ONNX"] = self.query_one(
                "#kokoro-use-onnx-switch", Switch
            ).value

            if self.kokoro_model_path:
                settings["KOKORO_ONNX_MODEL_PATH_DEFAULT"] = self.kokoro_model_path

            if self.kokoro_voices_path:
                settings["KOKORO_ONNX_VOICES_JSON_DEFAULT"] = self.kokoro_voices_path

            settings["KOKORO_MAX_TOKENS"] = int(
                self._validate_numeric_input(
                    self.query_one("#kokoro-max-tokens-input", Input).value,
                    1,
                    10000,
                    500,
                )
            )
            settings["KOKORO_ENABLE_VOICE_MIXING"] = self.query_one(
                "#kokoro-voice-mixing-switch", Switch
            ).value
            settings["KOKORO_TRACK_PERFORMANCE"] = self.query_one(
                "#kokoro-performance-switch", Switch
            ).value

            # Chatterbox settings
            settings["CHATTERBOX_DEVICE"] = self.query_one(
                "#chatterbox-device-select", Select
            ).value

            if self.chatterbox_voice_dir:
                settings["CHATTERBOX_VOICE_DIR"] = self.chatterbox_voice_dir

            settings["CHATTERBOX_EXAGGERATION"] = self._validate_numeric_input(
                self.query_one("#chatterbox-exaggeration-input", Input).value,
                0.0,
                1.0,
                0.5,
            )
            settings["CHATTERBOX_CFG_WEIGHT"] = self._validate_numeric_input(
                self.query_one("#chatterbox-cfg-weight-input", Input).value,
                0.0,
                1.0,
                0.5,
            )
            settings["CHATTERBOX_TEMPERATURE"] = self._validate_numeric_input(
                self.query_one("#chatterbox-temperature-input", Input).value,
                0.0,
                2.0,
                0.5,
            )
            settings["CHATTERBOX_CHUNK_SIZE"] = int(
                self._validate_numeric_input(
                    self.query_one("#chatterbox-chunk-size-input", Input).value,
                    256,
                    8192,
                    1024,
                )
            )

            seed = self.query_one("#chatterbox-seed-input", Input).value
            if seed:
                try:
                    settings["CHATTERBOX_RANDOM_SEED"] = int(seed)
                except ValueError:
                    pass

            settings["CHATTERBOX_NUM_CANDIDATES"] = int(
                self._validate_numeric_input(
                    self.query_one("#chatterbox-candidates-input", Input).value, 1, 5, 1
                )
            )
            settings["CHATTERBOX_VALIDATE_WHISPER"] = self.query_one(
                "#chatterbox-whisper-switch", Switch
            ).value
            settings["CHATTERBOX_PREPROCESS_TEXT"] = self.query_one(
                "#chatterbox-preprocess-switch", Switch
            ).value
            settings["CHATTERBOX_NORMALIZE_AUDIO"] = self.query_one(
                "#chatterbox-normalize-switch", Switch
            ).value
            settings["CHATTERBOX_TARGET_DB"] = self._validate_numeric_input(
                self.query_one("#chatterbox-target-db-input", Input).value, -40, 0, -20
            )
            settings["CHATTERBOX_MAX_CHUNK_SIZE"] = int(
                self._validate_numeric_input(
                    self.query_one("#chatterbox-max-chunk-input", Input).value,
                    50,
                    5000,
                    500,
                )
            )
            settings["CHATTERBOX_STREAMING"] = self.query_one(
                "#chatterbox-streaming-switch", Switch
            ).value
            settings["CHATTERBOX_STREAM_CHUNK_SIZE"] = int(
                self._validate_numeric_input(
                    self.query_one("#chatterbox-stream-chunk-input", Input).value,
                    512,
                    16384,
                    4096,
                )
            )
            settings["CHATTERBOX_ENABLE_CROSSFADE"] = self.query_one(
                "#chatterbox-crossfade-switch", Switch
            ).value
            settings["CHATTERBOX_CROSSFADE_MS"] = int(
                self._validate_numeric_input(
                    self.query_one("#chatterbox-crossfade-ms-input", Input).value,
                    10,
                    500,
                    50,
                )
            )

            # Higgs settings
            higgs_model_path = self.query_one("#higgs-model-path-input", Input).value
            if higgs_model_path:
                settings["HIGGS_MODEL_PATH"] = higgs_model_path

            higgs_voices_dir = self.query_one("#higgs-voices-dir-input", Input).value
            if higgs_voices_dir:
                settings["HIGGS_VOICE_SAMPLES_DIR"] = higgs_voices_dir

            settings["HIGGS_DEVICE"] = self.query_one(
                "#higgs-device-select", Select
            ).value

            settings["HIGGS_ENABLE_FLASH_ATTN"] = self.query_one(
                "#higgs-flash-attn-switch", Switch
            ).value

            settings["HIGGS_DTYPE"] = self.query_one(
                "#higgs-dtype-select", Select
            ).value

            settings["HIGGS_MAX_REFERENCE_DURATION"] = int(
                self._validate_numeric_input(
                    self.query_one("#higgs-max-ref-duration-input", Input).value,
                    1,
                    60,
                    30,
                )
            )

            settings["HIGGS_DEFAULT_LANGUAGE"] = self.query_one(
                "#higgs-language-select", Select
            ).value
            settings["HIGGS_ENABLE_VOICE_CLONING"] = self.query_one(
                "#higgs-voice-cloning-switch", Switch
            ).value
            settings["HIGGS_ENABLE_MULTI_SPEAKER"] = self.query_one(
                "#higgs-multi-speaker-switch", Switch
            ).value
            settings["HIGGS_SPEAKER_DELIMITER"] = self.query_one(
                "#higgs-delimiter-input", Input
            ).value
            settings["HIGGS_TRACK_PERFORMANCE"] = self.query_one(
                "#higgs-track-performance-switch", Switch
            ).value

            settings["HIGGS_MAX_NEW_TOKENS"] = int(
                self._validate_numeric_input(
                    self.query_one("#higgs-max-tokens-input", Input).value,
                    512,
                    8192,
                    4096,
                )
            )
            settings["HIGGS_TEMPERATURE"] = self._validate_numeric_input(
                self.query_one("#higgs-temperature-input", Input).value, 0.0, 2.0, 0.7
            )
            settings["HIGGS_TOP_P"] = self._validate_numeric_input(
                self.query_one("#higgs-top-p-input", Input).value, 0.0, 1.0, 0.9
            )
            settings["HIGGS_REPETITION_PENALTY"] = self._validate_numeric_input(
                self.query_one("#higgs-repetition-penalty-input", Input).value,
                1.0,
                2.0,
                1.1,
            )

            # AllTalk settings
            url = self.query_one("#alltalk-url-input", Input).value
            if url:
                settings["ALLTALK_TTS_URL_DEFAULT"] = url

            voice = self.query_one("#alltalk-voice-input", Input).value
            if voice:
                settings["ALLTALK_TTS_VOICE_DEFAULT"] = voice

            settings["ALLTALK_TTS_LANGUAGE_DEFAULT"] = self.query_one(
                "#alltalk-language-select", Select
            ).value
            settings["ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT"] = self.query_one(
                "#alltalk-format-select", Select
            ).value

            # Post save event
            self.app.post_message(
                STTSSettingsSaveEvent(settings, preferences=preferences)
            )

        except Exception:
            logger.error("Failed to collect TTS settings")
            self.app.notify("Failed to save settings", severity="error")

    def _collect_audio_cpp_config(self) -> AudioCppConfig:
        """Validate the complete external audio.cpp settings form."""
        return AudioCppConfig.from_mapping(
            {
                "mode": "external",
                "base_url": self.query_one("#audio-cpp-base-url-input", Input).value,
                "connect_timeout_seconds": float(
                    self.query_one("#audio-cpp-connect-timeout-input", Input).value
                ),
                "synthesis_timeout_seconds": float(
                    self.query_one("#audio-cpp-synthesis-timeout-input", Input).value
                ),
                "max_input_characters": self._audio_cpp_integer(
                    "#audio-cpp-max-input-characters-input"
                ),
                "max_response_bytes": self._audio_cpp_integer(
                    "#audio-cpp-max-response-bytes-input"
                ),
                "max_metadata_bytes": self._audio_cpp_integer(
                    "#audio-cpp-max-metadata-bytes-input"
                ),
                "max_catalog_models": self._audio_cpp_integer(
                    "#audio-cpp-max-catalog-models-input"
                ),
                "max_voices_per_model": self._audio_cpp_integer(
                    "#audio-cpp-max-voices-per-model-input"
                ),
                "max_identifier_characters": self._audio_cpp_integer(
                    "#audio-cpp-max-identifier-characters-input"
                ),
            }
        )

    def _audio_cpp_integer(self, selector: str) -> int:
        """Parse one audio.cpp integer field without coercing fractions."""
        return int(self.query_one(selector, Input).value)

    @work(
        exclusive=True,
        group="stts-audio-cpp-settings-discovery",
        exit_on_error=False,
    )
    async def _discover_audio_cpp(self, action: str) -> None:
        """Test or refresh the currently saved external audio.cpp service."""
        self._audio_cpp_discovery_generation += 1
        request_generation = self._audio_cpp_discovery_generation
        status = self.query_one("#audio-cpp-discovery-status", Static)
        status.update("Checking saved settings…")
        service = None
        before_revision: int | None = None
        try:
            service = await get_tts_service()
            before_revision = service.configuration_revision("audio_cpp")
            catalog = await service.get_catalog("audio_cpp", refresh=True)
            after_revision = service.configuration_revision("audio_cpp")
            if request_generation != self._audio_cpp_discovery_generation:
                return
            if before_revision != after_revision:
                self._report_audio_cpp_settings_changed(status)
                return
            if (
                catalog.provider_id != "audio_cpp"
                or catalog.health.state != "available"
                or not catalog.health.fresh
            ):
                status.update("Unavailable")
                self.app.notify(
                    "audio.cpp is not ready; check the saved settings",
                    severity="error",
                )
                return

            model_count = len(catalog.models)
            noun = "model" if model_count == 1 else "models"
            if action == "test":
                message = f"audio.cpp connection is ready ({model_count} {noun})"
            else:
                message = f"audio.cpp models refreshed ({model_count} {noun})"
            status.update(message)
            self.app.notify(message, severity="information")
        except asyncio.CancelledError:
            raise
        except Exception:
            if request_generation != self._audio_cpp_discovery_generation:
                return
            if service is not None and before_revision is not None:
                try:
                    after_revision = service.configuration_revision("audio_cpp")
                except Exception:
                    after_revision = None
                if request_generation != self._audio_cpp_discovery_generation:
                    return
                if after_revision is not None and before_revision != after_revision:
                    self._report_audio_cpp_settings_changed(status)
                    return
            logger.warning("audio.cpp settings discovery failed")
            status.update("Unavailable")
            self.app.notify(
                "audio.cpp is not ready; check the saved settings",
                severity="error",
            )

    def _report_audio_cpp_settings_changed(self, status: Static) -> None:
        """Report that an explicit discovery result is no longer authoritative."""
        status.update("Settings changed; retry")
        self.app.notify(
            "audio.cpp settings changed; retry the check",
            severity="warning",
        )

    def _validate_numeric_input(
        self, value: str, min_val: float, max_val: float, default: float
    ) -> float:
        """Validate and convert numeric input"""
        try:
            if not value:
                return default
            num_val = float(value)
            return max(min_val, min(max_val, num_val))
        except ValueError:
            return default

    @classmethod
    def _normalize_openai_base_url(cls, value: str) -> str:
        """Return a safe absolute OpenAI-compatible speech endpoint."""
        normalized = value.strip() or cls.OPENAI_TTS_DEFAULT_URL
        parsed = urlsplit(normalized)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.netloc
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or "\r" in normalized
            or "\n" in normalized
        ):
            raise ValueError(
                "OpenAI base URL must be an absolute HTTP(S) URL without "
                "credentials or a fragment"
            )
        return normalized

    def _load_kokoro_voice_blends(self) -> None:
        """Load and display Kokoro voice blends"""
        try:
            # Get voice blends from stored config
            blend_list = self.query_one("#kokoro-voice-blends-list", Static)

            # Load blends from config file
            blend_file = (
                Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            )
            if blend_file.exists():
                with open(blend_file, "r") as f:
                    blends = json.load(f)

                if blends:
                    # Format blends for display
                    blend_text = ""
                    for blend_name, blend_data in blends.items():
                        voices_str = ", ".join(
                            [
                                f"{v[0]} ({v[1]:.2f})"
                                for v in blend_data.get("voices", [])
                            ]
                        )
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
                blend_file = (
                    Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
                )
                blend_file.parent.mkdir(parents=True, exist_ok=True)

                # Load existing blends
                if blend_file.exists():
                    with open(blend_file, "r") as f:
                        blends = json.load(f)
                else:
                    blends = {}

                # Add new blend
                blends[result["name"]] = result

                # Save back
                with open(blend_file, "w") as f:
                    json.dump(blends, f, indent=2)

                # Refresh display
                self._load_kokoro_voice_blends()
                self.app.notify(
                    f"Voice blend '{result['name']}' created successfully",
                    severity="success",
                )

        except Exception as e:
            logger.error(f"Failed to create voice blend: {e}")
            self.app.notify(f"Error creating voice blend: {e}", severity="error")

    def _import_voice_blends(self) -> None:
        """Import voice blends from file"""
        try:
            filters = Filters(
                ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                ("All Files", lambda p: True),
            )

            file_picker = FileOpen(
                title="Import Voice Blends",
                filters=filters,
                context="voice_blends_import",
            )

            self.app.push_screen(file_picker, self._handle_import_file)

        except Exception as e:
            logger.error(f"Failed to show import dialog: {e}")
            self.app.notify(f"Error showing import dialog: {e}", severity="error")

    def _handle_import_file(self, path: Optional[str]) -> None:
        """Handle the imported file"""
        if not path:
            return

        try:
            import_path = Path(path)

            # Load the import file
            with open(import_path, "r") as f:
                imported_blends = json.load(f)

            # Load existing blends
            blend_file = (
                Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            )
            blend_file.parent.mkdir(parents=True, exist_ok=True)

            if blend_file.exists():
                with open(blend_file, "r") as f:
                    existing_blends = json.load(f)
            else:
                existing_blends = {}

            # Merge blends (imported overwrites existing with same name)
            existing_blends.update(imported_blends)

            # Save merged blends
            with open(blend_file, "w") as f:
                json.dump(existing_blends, f, indent=2)

            # Refresh display
            self._load_kokoro_voice_blends()
            self.app.notify(
                f"Imported {len(imported_blends)} voice blend(s) successfully",
                severity="success",
            )

        except Exception as e:
            logger.error(f"Failed to import voice blends: {e}")
            self.app.notify(f"Error importing voice blends: {e}", severity="error")

    def _export_voice_blends(self) -> None:
        """Export voice blends to file"""
        try:
            # Load existing blends
            blend_file = (
                Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            )

            if not blend_file.exists():
                self.app.notify("No voice blends to export", severity="warning")
                return

            with open(blend_file, "r") as f:
                blends = json.load(f)

            if not blends:
                self.app.notify("No voice blends to export", severity="warning")
                return

            # Store blends temporarily for export
            self._export_blends = blends

            filters = Filters(
                ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                ("All Files", lambda p: True),
            )

            file_picker = FileSave(
                title="Export Voice Blends",
                filters=filters,
                default_filename="kokoro_voice_blends_export.json",
                context="voice_blends_export",
            )

            self.app.push_screen(file_picker, self._handle_export_file)

        except Exception as e:
            logger.error(f"Failed to export voice blends: {e}")
            self.app.notify(f"Error exporting voice blends: {e}", severity="error")

    def _handle_export_file(self, path: Optional[str]) -> None:
        """Handle the export file location"""
        if not path or not hasattr(self, "_export_blends"):
            return

        try:
            export_path = Path(path)

            # Write the blends to the selected file
            with open(export_path, "w") as f:
                json.dump(self._export_blends, f, indent=2)

            self.app.notify(
                f"Exported {len(self._export_blends)} voice blend(s) to: {export_path.name}",
                severity="success",
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
            ("All Files", lambda p: True),
        )

        # Get current value as starting path
        current_value = self.kokoro_model_path
        location = (
            Path(current_value).parent
            if current_value and Path(current_value).parent.exists()
            else Path.home()
        )

        file_picker = FileOpen(
            location=str(location),
            title="Select Kokoro Model File",
            filters=filters,
            context="kokoro_model",
        )

        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_kokoro_model_selection)

    def _handle_kokoro_model_selection(self, path: Optional[Path]) -> None:
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
            ("All Files", lambda p: True),
        )

        # Get current value as starting path
        current_value = self.kokoro_voices_path
        location = (
            Path(current_value).parent
            if current_value and Path(current_value).parent.exists()
            else Path.home()
        )

        file_picker = FileOpen(
            location=str(location),
            title="Select Voices Configuration File",
            filters=filters,
            context="kokoro_voices",
        )

        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_kokoro_voices_selection)

    def _handle_kokoro_voices_selection(self, path: Optional[Path]) -> None:
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
        location = (
            Path(current_value)
            if current_value and Path(current_value).exists()
            else Path.home()
        )

        # Create a filter that shows directories prominently
        filters = Filters(
            ("Directories", lambda p: p.is_dir() if p.exists() else False),
            ("All Files", lambda p: True),
        )

        file_picker = FileOpen(
            location=str(location),
            title="Select Voice Directory (choose any file in target directory)",
            filters=filters,
            context="chatterbox_voices_dir",
        )

        # Mount the file picker
        self.app.push_screen(file_picker, self._handle_chatterbox_voice_dir_selection)

    def _handle_chatterbox_voice_dir_selection(self, path: Optional[Path]) -> None:
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
        location = (
            Path(current_value)
            if current_value and Path(current_value).exists()
            else Path.home()
        )

        # Create filter for directory selection
        filters = Filters(
            ("Directories", lambda p: p.is_dir() if p.exists() else False),
            ("All Files", lambda p: True),
        )

        file_picker = FileOpen(
            location=str(location),
            title="Select Higgs Voice Samples Directory (choose any file in target directory)",
            filters=filters,
            context="higgs_voices_dir",
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
                        id="import-source-select",
                    )

                yield Button(
                    "📁 Import Content", id="import-content-btn", variant="default"
                )

            # Content preview
            yield Label("Content Preview:")
            yield TextArea(id="content-preview", disabled=True)

            # Chapter Editor - Enhanced visual chapter editing
            with Collapsible(
                title="📖 Chapter Editor", classes="settings-section", collapsed=False
            ):
                from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                    ChapterEditorWidget,
                )

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
                        id="narrator-voice-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable Multi-voice:", classes="form-label")
                    yield Switch(id="multi-voice-switch", value=False)

                # Character voice widget
                from tldw_chatbook.Widgets.TTS.character_voice_widget import (
                    CharacterVoiceWidget,
                )

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
                        id="audiobook-provider-select",
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
                        id="audiobook-format-select",
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
            yield Button(
                "🎙️ Generate AudioBook", id="generate-audiobook-btn", variant="primary"
            )

            # Export button (initially disabled)
            yield Button(
                "💾 Export AudioBook",
                id="audiobook-export-btn",
                variant="success",
                disabled=True,
            )

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

    @on(Select.Changed)
    def on_audiobook_provider_select_for_voice_widget_changed(
        self, event: Select.Changed
    ) -> None:
        """Handle select widget changes"""
        if event.select.id == "audiobook-provider-select":
            # Update character voice widget provider
            try:
                from tldw_chatbook.Widgets.TTS.character_voice_widget import (
                    CharacterVoiceWidget,
                )

                voice_widget = self.query_one(
                    "#character-voice-widget", CharacterVoiceWidget
                )
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
                ("All Files", lambda p: True),
            )

            file_picker = FileOpen(
                title="Select Text File for AudioBook",
                filters=filters,
                context="audiobook_text",
            )

            # Mount the file picker
            self.app.push_screen(file_picker, self._handle_file_selection)
        except ImportError:
            # Fallback to simple file input
            self.app.notify(
                "File picker not available. Please paste your text instead.",
                severity="warning",
            )

    def _handle_file_selection(self, path: Optional[str]) -> None:
        """Handle file selection for audiobook content"""
        if not path:
            return

        try:
            # Read the file content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Update content preview
            self.content_text = content
            content_preview = self.query_one("#content-preview", TextArea)
            content_preview.load_text(
                content[:1000] + "..." if len(content) > 1000 else content
            )
            content_preview.disabled = False

            # Detect chapters if enabled
            if self.query_one("#auto-chapters-switch", Switch).value:
                self._detect_chapters()

            self.app.notify(
                f"Imported {len(content)} characters from {Path(path).name}",
                severity="information",
            )

        except Exception as e:
            logger.error(f"Failed to import file: {e}")
            self.app.notify(f"Failed to import file: {e}", severity="error")

    def _import_from_notes(self) -> None:
        """Import content from notes"""
        from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import (
            NoteSelectionDialog,
        )
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
                            if note.get("title"):
                                combined_content.append(f"# {note['title']}\n")
                            combined_content.append(note.get("content", ""))
                            combined_content.append("\n\n")  # Separator between notes

                    # Load combined content
                    self.content_text = "\n".join(combined_content)
                    content_preview = self.query_one("#content-preview", TextArea)
                    preview_text = (
                        self.content_text[:1000] + "..."
                        if len(self.content_text) > 1000
                        else self.content_text
                    )
                    content_preview.load_text(preview_text)
                    content_preview.disabled = False

                    # Detect chapters if enabled
                    if self.query_one("#auto-chapters-switch", Switch).value:
                        self._detect_chapters()

                    self.app.notify(
                        f"Imported {len(selected_ids)} note(s)", severity="information"
                    )

            self.app.push_screen(NoteSelectionDialog(notes), handle_note_selection)

        except Exception as e:
            logger.error(f"Failed to import from notes: {e}")
            self.app.notify(f"Failed to import notes: {e}", severity="error")

    def _import_from_conversation(self) -> None:
        """Import content from conversation"""
        from tldw_chatbook.Widgets.conversation_selection_dialog import (
            ConversationSelectionDialog,
        )
        from tldw_chatbook.DB.ChaChaNotes_DB import fetch_all_conversations

        try:
            # Fetch all conversations from database
            conversations = fetch_all_conversations()
            if not conversations:
                self.app.notify(
                    "No conversations found in database", severity="warning"
                )
                return

            # Show conversation selection dialog
            def handle_conversation_selection(
                selection: Optional[Dict[str, Any]],
            ) -> None:
                if selection:
                    # Fetch messages for selected conversation
                    from tldw_chatbook.DB.ChaChaNotes_DB import (
                        fetch_messages_by_conversation_id,
                    )

                    messages = fetch_messages_by_conversation_id(
                        selection["conversation_id"]
                    )

                    if not messages:
                        self.app.notify(
                            "No messages found in conversation", severity="warning"
                        )
                        return

                    # Build content based on options
                    content_parts = []
                    for msg in messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")

                        # Filter based on inclusion options
                        if selection.get("include_all"):
                            pass  # Include all messages
                        elif selection.get("include_user") and role != "user":
                            continue
                        elif selection.get("include_assistant") and role != "assistant":
                            continue

                        # Format based on speaker option
                        if selection.get("include_speakers"):
                            speaker_name = "User" if role == "user" else "Assistant"
                            content_parts.append(f"{speaker_name}: {content}")
                        else:
                            content_parts.append(content)

                        content_parts.append("")  # Empty line between messages

                    # Load combined content
                    self.content_text = "\n".join(content_parts)
                    content_preview = self.query_one("#content-preview", TextArea)
                    preview_text = (
                        self.content_text[:1000] + "..."
                        if len(self.content_text) > 1000
                        else self.content_text
                    )
                    content_preview.load_text(preview_text)
                    content_preview.disabled = False

                    # Auto-detect chapters might not be suitable for conversations
                    # but run it if enabled
                    if self.query_one("#auto-chapters-switch", Switch).value:
                        self._detect_chapters()

                    self.app.notify(
                        f"Imported conversation with {len(messages)} messages",
                        severity="information",
                    )

            self.app.push_screen(
                ConversationSelectionDialog(conversations),
                handle_conversation_selection,
            )

        except Exception as e:
            logger.error(f"Failed to import from conversation: {e}")
            self.app.notify(f"Failed to import conversation: {e}", severity="error")

    def _import_from_paste(self) -> None:
        """Import content from clipboard paste"""
        # Enable the content preview for editing
        content_preview = self.query_one("#content-preview", TextArea)
        content_preview.disabled = False
        content_preview.focus()
        self.app.notify(
            "Paste your text into the content preview area", severity="information"
        )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area content changes"""
        if event.text_area.id == "content-preview":
            self.content_text = event.text_area.text
            # Detect chapters if auto-detect is enabled
            if (
                self.query_one("#auto-chapters-switch", Switch).value
                and self.content_text
            ):
                self._detect_chapters()

    def on_chapter_edit_event(self, event) -> None:
        """Handle chapter edit events from the chapter editor"""
        from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditEvent

        if isinstance(event, ChapterEditEvent):
            # Update our internal chapter list
            try:
                from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                    ChapterEditorWidget,
                )

                chapter_editor = self.query_one(
                    "#chapter-editor-widget", ChapterEditorWidget
                )
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
        from tldw_chatbook.Widgets.TTS.character_voice_widget import (
            CharacterDetectionEvent,
            CharacterVoiceWidget,
        )

        if isinstance(event, CharacterDetectionEvent):
            # Detect characters from current content
            if self.content_text:
                try:
                    voice_widget = self.query_one(
                        "#character-voice-widget", CharacterVoiceWidget
                    )
                    characters = voice_widget.detect_characters_from_text(
                        self.content_text, event.auto_assign
                    )
                    self.app.notify(
                        f"Detected {len(characters)} characters", severity="information"
                    )
                except Exception as e:
                    logger.error(f"Failed to detect characters: {e}")
                    self.app.notify(
                        f"Failed to detect characters: {e}", severity="error"
                    )
            else:
                self.app.notify("Please import content first", severity="warning")

    def on_character_voice_assign_event(self, event) -> None:
        """Handle character voice assignments"""
        from tldw_chatbook.Widgets.TTS.character_voice_widget import (
            CharacterVoiceAssignEvent,
        )

        if isinstance(event, CharacterVoiceAssignEvent):
            logger.info(f"Voice assigned: {event.character_name} → {event.voice_id}")

    def _detect_chapters(self) -> None:
        """Detect chapters in the content"""
        if not self.content_text:
            return

        try:
            from tldw_chatbook.TTS.audiobook_generator import ChapterDetector
            from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                ChapterEditorWidget,
            )

            # Detect chapters
            self.detected_chapters = ChapterDetector.detect_chapters(self.content_text)

            # Update the chapter editor widget
            try:
                chapter_editor = self.query_one(
                    "#chapter-editor-widget", ChapterEditorWidget
                )
                chapter_editor.set_chapters(self.detected_chapters)
                self.app.notify(
                    f"Detected {len(self.detected_chapters)} chapters",
                    severity="information",
                )
            except Exception as e:
                logger.warning(f"Could not update chapter editor: {e}")
                # Fall back to old display method if chapter editor not found
                chapter_list = self.query_one("#chapter-list", Static)
                if self.detected_chapters:
                    chapter_display = []
                    for i, chapter in enumerate(self.detected_chapters):
                        chapter_display.append(
                            f"{i + 1}. {chapter.title} ({len(chapter.content.split())} words)"
                        )

                    chapter_list.update("\n".join(chapter_display))
                    self.app.notify(
                        f"Detected {len(self.detected_chapters)} chapters",
                        severity="information",
                    )
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
            from tldw_chatbook.Widgets.TTS.chapter_editor_widget import (
                ChapterEditorWidget,
            )

            chapter_editor = self.query_one(
                "#chapter-editor-widget", ChapterEditorWidget
            )
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
                from tldw_chatbook.Widgets.TTS.character_voice_widget import (
                    CharacterVoiceWidget,
                )

                voice_widget = self.query_one(
                    "#character-voice-widget", CharacterVoiceWidget
                )
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
        self.app.post_message(
            STTSAudioBookGenerateEvent(
                content=self.content_text,
                chapters=self.detected_chapters if include_chapters else [],
                narrator_voice=narrator_voice,
                output_format=audio_format,
                options=options,
            )
        )

    def _get_model_for_provider(self, provider: str) -> str:
        """Get default model for provider"""
        models = {
            "openai": "tts-1",
            "elevenlabs": "eleven_multilingual_v2",
            "kokoro": "kokoro-v0_19",
            "chatterbox": "chatterbox-v1",
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

    @on(Select.Changed)
    def on_audiobook_selects_changed(self, event: Select.Changed) -> None:
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
            voice_select.set_options(
                [
                    ("alloy", "Alloy"),
                    ("echo", "Echo"),
                    ("fable", "Fable"),
                    ("onyx", "Onyx"),
                    ("nova", "Nova"),
                    ("shimmer", "Shimmer"),
                ]
            )
        elif provider == "elevenlabs":
            voice_select.set_options(
                [
                    ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
                    ("AZnzlk1XvdvUeBnXmlld", "Domi"),
                    ("EXAVITQu4vr4xnSDxMaL", "Bella"),
                    ("ErXwobaYiN019PkySvjV", "Antoni"),
                    ("MF3mGyEYCl7XYWbV9V6O", "Elli"),
                ]
            )
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
            blend_file = (
                Path.home() / ".config" / "tldw_cli" / "kokoro_voice_blends.json"
            )
            if blend_file.exists():
                try:
                    import json

                    with open(blend_file, "r") as f:
                        blends = json.load(f)
                        if blends:
                            # Add separator
                            voice_options.append(
                                ("_separator", "──── Voice Blends ────")
                            )
                            # Add each blend
                            for blend_name, blend_data in blends.items():
                                display_name = f"🎭 {blend_name}"
                                if blend_data.get("description"):
                                    display_name += (
                                        f" - {blend_data['description'][:30]}"
                                    )
                                voice_options.append(
                                    (f"blend:{blend_name}", display_name)
                                )
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
            voice_select.set_options(
                [
                    ("default", "Default"),
                    ("custom", "Custom Voice"),
                ]
            )

    def _export_audiobook(self) -> None:
        """Export the generated audiobook"""
        if not self.generated_audiobook_path:
            self.app.notify("No audiobook to export", severity="warning")
            return

        try:
            # Create file picker for save location using pre-imported FileSave
            filters = Filters(
                ("AudioBook Files", lambda p: p.suffix.lower() in [".m4b", ".mp3"]),
                ("All Files", lambda p: True),
            )

            file_picker = FileSave(
                title="Save AudioBook As",
                filters=filters,
                default_filename=self.generated_audiobook_path.name,
                context="audiobook_save",
            )

            # Mount the file picker
            self.app.push_screen(file_picker, self._handle_export_location)
        except ImportError:
            # Fallback
            self.app.notify(
                f"AudioBook saved to: {self.generated_audiobook_path}",
                severity="information",
            )

    def _handle_export_location(self, path: Optional[str]) -> None:
        """Handle export location selection"""
        if not path or not self.generated_audiobook_path:
            return

        try:
            import shutil

            shutil.copy2(self.generated_audiobook_path, path)
            self.app.notify(
                f"AudioBook exported to: {Path(path).name}", severity="information"
            )
        except Exception as e:
            logger.error(f"Failed to export audiobook: {e}")
            self.app.notify(f"Failed to export audiobook: {e}", severity="error")

    def audiobook_generation_complete(
        self, success: bool, path: Optional[Path] = None
    ) -> None:
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
                self.app.notify(
                    "Please select a valid narrator voice", severity="warning"
                )
                return

            # Limit preview to first 500 characters
            preview_text = (
                chapter.content[:500] + "..."
                if len(chapter.content) > 500
                else chapter.content
            )

            # Log preview generation
            log = self.query_one("#audiobook-generation-log", RichLog)
            log.write(f"[yellow]Generating preview for: {chapter.title}[/yellow]")

            # Create TTS request event
            from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
                STTSPlaygroundGenerateEvent,
            )

            # Post event to generate preview
            self.post_message(
                STTSPlaygroundGenerateEvent(
                    STTSPlaygroundRequest(
                        operation_id=str(uuid4()),
                        provider_id=provider,
                        model_id=self._get_model_for_provider(provider),
                        text=preview_text,
                        voice_id=narrator_voice,
                        response_format="mp3",
                        speed=1.0,
                        options={"preview_chapter": chapter.title},
                    )
                )
            )

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
            "alltalk": "alltalk",
        }
        return model_map.get(provider, "default")


class STTSWindow(Container):
    """Main S/TT/S window containing all sub-windows"""

    DEFAULT_CSS = """
    STTSWindow {
        layout: horizontal;
        height: 100%;
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

    .speech-capability-status {
        margin-top: 1;
        padding: 1;
        border: round $surface;
        color: $text-muted;
    }
    """

    current_view = reactive("playground")

    def __init__(self, app_instance, **kwargs):
        """Initialize the S/TT/S window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        """Compose the S/TT/S window: content only.

        The sidebar that used to lead this method -- six view buttons and the
        capability status line -- moved into the Lab frame's rail and status
        chip (``UI/Screens/stts_screen.py``), so that Speech has the same
        chrome as Models and Evals instead of a second, differently-styled
        navigation column inside the body.

        The window keeps ownership of ``current_view`` and of mounting the
        matching content widget; the screen only points it at a view.
        """
        with Container(classes="stts-content"):
            # Show playground by default
            yield SpeechPlaygroundPane(id="speech-playground-pane")

    def _speech_capability_status_text(self) -> str:
        """Return a concise local speech dependency status for the sidebar."""
        check_tts_deps()
        check_stt_deps()

        if self._speech_dependencies_available():
            return "Local speech: ready"

        return self._speech_dependency_recovery_state().visible_copy

    def _speech_capability_status_tooltip(self) -> str:
        """Return install guidance for local speech dependencies."""
        if self._speech_dependencies_available():
            return "Local TTS and STT dependencies are available."
        return self._speech_dependency_recovery_state().disabled_tooltip

    def _speech_dependencies_available(self) -> bool:
        return bool(DEPENDENCIES_AVAILABLE.get("tts_processing", False)) and bool(
            DEPENDENCIES_AVAILABLE.get("stt_processing", False)
        )

    def _speech_dependency_recovery_state(self):
        missing_dependencies = []
        if not DEPENDENCIES_AVAILABLE.get("tts_processing", False):
            missing_dependencies.append("local_tts")
        if not DEPENDENCIES_AVAILABLE.get("stt_processing", False):
            missing_dependencies.extend(
                ("transcription_faster_whisper", "speech_recording")
            )

        return optional_dependency_recovery_state(
            unavailable_what="Local speech providers",
            missing_dependencies=tuple(missing_dependencies),
            install_target='pip install "tldw_chatbook[local_tts,transcription_faster_whisper,speech_recording]"',
            stable_selector="speech-capability-status",
            recovery_action="Settings > Speech",
        )

    def watch_current_view(self, old_view: str, new_view: str) -> None:
        """Handle view changes.

        Returns early when the content container is not mounted yet. The
        window is now the Lab frame's deferred body, so it is mounted after
        first paint rather than composed inline -- and a reactive watcher can
        fire against a window whose own children have not been composed. The
        unguarded `query_one` raised NoMatches out of the frame's body mount,
        which took down the whole screen. Mirrors the same QueryError
        tolerance `LLMManagementWindow.watch_active_view` carries.
        """
        try:
            content_container = self.query_one(".stts-content", Container)
        except QueryError:
            logger.debug(
                "STTS content container not mounted yet; deferring view "
                f"change to '{new_view}' until compose completes."
            )
            return

        # Give widgets a chance to clean up before removal
        for child in content_container.children:
            if hasattr(child, "cleanup") and callable(child.cleanup):
                try:
                    child.cleanup()
                except Exception as e:
                    logger.debug(f"Error during widget cleanup: {e}")

        # Remove all children from the container
        content_container.remove_children()

        # Add new content based on view
        if new_view == "playground":
            content_container.mount(
                SpeechPlaygroundPane(id="speech-playground-pane")
            )
        elif new_view == "settings":
            content_container.mount(TTSSettingsWidget())
        elif new_view == "audiobook":
            content_container.mount(AudioBookGenerationWidget())
        elif new_view == "dictation":
            content_container.mount(DictationWindow())

        # Selection styling is the rail's job now. These lines used to
        # `query_one("#view-*-btn")` for the four view buttons; those live on
        # STTSScreen since the sidebar moved, so every one of them would raise
        # NoMatches on the first view change. The screen watches
        # `current_view` and applies `is-active` itself.

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
                    if hasattr(active_widget, "on_button_pressed"):
                        active_widget.on_button_pressed(event)
            except Exception as e:
                logger.debug(f"Could not delegate button event: {e}")


#
# End of STTS_Window.py
#######################################################################################################################
