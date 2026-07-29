"""Persisted TTS settings: loading, collecting, saving, and the pickers.

Moved whole from `TTSSettingsWidget` rather than reimplemented. The riskiest
piece is `_save_settings`, and measuring it first changed how it had to be
handled: it writes no config. It collects 47 values off the controls and
posts one `STTSSettingsSaveEvent`; persistence lives in the event handler,
which this phase does not touch. `Tests/UI/test_speech_settings_save_equivalence.py`
holds that posted dict fixed against a snapshot taken before any of this
moved.

Like the Playground's mixins, these methods query their controls by id, so
any host mounting the settings ids inherits them unchanged. That is why the
rebuilt pane kept the legacy ids.

Two hooks exist for one reason: the tests patch `get_cli_setting` and
`get_tts_service` on the `STTS_Window` module. Code that moved here would
resolve them from this module instead and the patches would silently detach
-- exactly what happened to the catalog move in phase 1, where twelve patch
sites came adrift and every select sat on LOADING. `TTSSettingsWidget`
overrides both to resolve from its own module, so the existing seam still
works without editing a single test.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Literal

from loguru import logger
from rich.text import Text
from textual import on, work
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.widgets import Button, Input, Label, Select, Static, Switch

from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS import TTSPreferencesSnapshot, get_tts_service
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen as FileOpen,
    EnhancedFileSave as FileSave,
)
from tldw_chatbook.UI.stts_playground_catalog import (
    AUDIO_CPP_PROVIDER_ID,
    FIRST_AVAILABLE_MODEL_ID,
    SERVER_DEFAULT_VOICE_ID,
    SelectSentinel,
)


class SpeechSettingsMixin:
    """Settings load/save behaviour, independent of the layout."""

    #: Class constants travel with the methods that read them. This one did
    #: not at first, and `_normalize_openai_base_url` -- a @classmethod --
    #: failed with "type object 'SpeechSettingsPane' has no attribute
    #: 'OPENAI_TTS_DEFAULT_URL'" on the rebuilt host only, since the legacy
    #: widget still had its own copy.
    OPENAI_TTS_DEFAULT_URL = "https://api.openai.com/v1/audio/speech"

    def _tts_service_factory(self):
        """Return the TTS service, awaitable.

        A hook so the module-level name stays patchable where the tests
        already patch it.

        Returns:
            An awaitable yielding the TTS service.
        """
        return get_tts_service()

    def _cli_setting(self, *args: Any, **kwargs: Any) -> Any:
        """Read a config setting.

        Overridable for the same reason as `_tts_service_factory`.

        Args:
            args: Forwarded to ``get_cli_setting``.
            kwargs: Forwarded to ``get_cli_setting``.

        Returns:
            The configured value, or the supplied default.
        """
        return get_cli_setting(*args, **kwargs)


    def _load_audio_cpp_config(self) -> AudioCppConfig:
        """Load a safe external audio.cpp form snapshot."""
        candidate = self._cli_setting("app_tts", "audio_cpp", {})
        try:
            values = candidate if isinstance(candidate, Mapping) else {}
            return AudioCppConfig.from_mapping(values)
        except (TypeError, ValueError):
            logger.warning("Stored audio.cpp settings are invalid; using defaults")
            return AudioCppConfig()

    def on_mount(self) -> None:
        """Set initial values from config after mount"""
        self.call_after_refresh(self._set_initial_values)

    def _set_initial_values(self) -> None:
        """Apply config values after all child Select widgets have mounted."""
        try:
            default_provider = self._cli_setting(
                "app_tts",
                "default_provider",
                "openai",
            )
            is_audio_cpp = default_provider == AUDIO_CPP_PROVIDER_ID
            preference_values: dict[str, object] = {
                "default_provider": default_provider,
                "default_model": self._cli_setting(
                    "app_tts",
                    "default_model",
                    "" if is_audio_cpp else "tts-1",
                ),
                "default_voice": self._cli_setting(
                    "app_tts",
                    "default_voice",
                    "" if is_audio_cpp else "alloy",
                ),
                "default_format": self._cli_setting(
                    "app_tts",
                    "default_format",
                    "wav" if is_audio_cpp else "mp3",
                ),
                "default_speed": self._cli_setting(
                    "app_tts",
                    "default_speed",
                    1.0,
                ),
            }
            missing_mode = object()
            for mode_key in ("default_model_mode", "default_voice_mode"):
                mode = self._cli_setting("app_tts", mode_key, missing_mode)
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
            self.kokoro_model_path = self._cli_setting(
                "app_tts", "KOKORO_ONNX_MODEL_PATH_DEFAULT", ""
            )
            self.kokoro_voices_path = self._cli_setting(
                "app_tts", "KOKORO_ONNX_VOICES_JSON_DEFAULT", ""
            )
            self.chatterbox_voice_dir = self._cli_setting(
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
                kokoro_device = self._cli_setting(
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
                chatterbox_device = self._cli_setting(
                    "app_tts", "CHATTERBOX_DEVICE", "cpu"
                )
                if chatterbox_device in ["cpu", "cuda"]:
                    chatterbox_device_select.value = chatterbox_device
            except Exception as e:
                logger.debug(f"Could not set Chatterbox device: {e}")

            # Set ElevenLabs model
            elevenlabs_model_select = self.query_one("#elevenlabs-model-select", Select)
            elevenlabs_model = self._cli_setting(
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
            elevenlabs_format = self._cli_setting(
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
            alltalk_language = self._cli_setting(
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
            alltalk_format = self._cli_setting(
                "app_tts", "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT", "wav"
            )
            if alltalk_format in ["wav", "mp3", "opus", "flac"]:
                alltalk_format_select.value = alltalk_format

            # Set Higgs settings - Select widgets are already initialized in compose(),
            # but we ensure they have the correct values here
            try:
                # Device
                higgs_device_select = self.query_one("#higgs-device-select", Select)
                higgs_device = self._cli_setting("HiggsSettings", "device", "auto")
                if higgs_device in ["auto", "cpu", "cuda", "cuda:0", "cuda:1"]:
                    higgs_device_select.value = higgs_device

                # Data type
                higgs_dtype_select = self.query_one("#higgs-dtype-select", Select)
                higgs_dtype = self._cli_setting("HiggsSettings", "dtype", "bfloat16")
                if higgs_dtype in ["float32", "float16", "bfloat16"]:
                    higgs_dtype_select.value = higgs_dtype

                # Language
                higgs_language_select = self.query_one("#higgs-language-select", Select)
                higgs_language = self._cli_setting(
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

    def handle_default_selects_changed(self, event: Select.Changed) -> None:
        """Respond to a change on one of the default selects.

        NOT decorated here. Textual registers `@on` handlers in its
        metaclass, scanning each class's own namespace, so a mixin that is
        not itself a MessagePump contributes nothing -- no error, no
        warning, the handler simply never runs and changing the default
        provider stops repopulating the model and voice lists. Each host
        declares the decorated handler and delegates to this.
        """
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
            default_voice = self._cli_setting("app_tts", "default_voice", "alloy")
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
            default_model = self._cli_setting("app_tts", "default_model", "tts-1")
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
            # Deliberately terse, and NOT `opt(exception=True)`. These values
            # include API keys and endpoints, and
            # `test_settings_widget_does_not_echo_collection_error_details`
            # exists to keep them out of the log. I widened this while
            # debugging and that test caught it.
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
            service = await self._tts_service_factory()
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
