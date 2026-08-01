"""Pure global Speech & TTS Settings state and validation.

The mapping in this module is intentionally bounded to Chatbook's seven
built-in providers.  It is an ownership contract for one Settings category,
not a provider plug-in or schema-driven form system.
"""

from __future__ import annotations

import math
import os
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.Utils.input_validation import (
    provider_api_key_validation_error,
    validate_url,
)

BUILT_IN_TTS_PROVIDER_ORDER = (
    "audio_cpp",
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)

TTS_PROVIDER_LABELS = MappingProxyType(
    {
        "audio_cpp": "audio.cpp",
        "openai": "OpenAI",
        "elevenlabs": "ElevenLabs",
        "kokoro": "Kokoro",
        "chatterbox": "Chatterbox",
        "higgs": "Higgs",
        "alltalk": "AllTalk",
    }
)

# Explicit global ownership destination.  The names are stable model keys;
# rendering remains deliberately provider-specific in the panel.
GLOBAL_TTS_PROVIDER_FIELD_IDS = MappingProxyType(
    {
        "audio_cpp": (
            "base_url",
            "connect_timeout_seconds",
            "synthesis_timeout_seconds",
            "max_input_characters",
            "max_response_bytes",
            "max_metadata_bytes",
            "max_catalog_models",
            "max_voices_per_model",
            "max_identifier_characters",
        ),
        "openai": ("credential", "base_url", "organization_id"),
        "elevenlabs": (
            "credential",
            "output_format",
            "stability",
            "similarity_boost",
            "style",
            "speaker_boost",
        ),
        "kokoro": (
            "device",
            "use_onnx",
            "onnx_model_path",
            "voices_json_path",
            "max_tokens",
            "voice_mixing",
            "track_performance",
        ),
        "chatterbox": (
            "device",
            "voice_resource_directory",
            "temperature",
            "chunk_size",
            "random_seed",
            "candidates",
            "validate_whisper",
            "preprocess_text",
            "normalize_audio",
            "target_db",
            "max_chunk_size",
            "streaming",
            "stream_chunk_size",
            "crossfade",
            "crossfade_ms",
        ),
        "higgs": (
            "model_path",
            "voice_resource_directory",
            "device",
            "enable_flash_attention",
            "dtype",
            "max_reference_duration",
            "language",
            "voice_cloning",
            "multi_speaker",
            "speaker_delimiter",
            "track_performance",
            "max_new_tokens",
            "temperature",
            "top_p",
            "repetition_penalty",
        ),
        "alltalk": ("server_url", "language"),
    }
)


_PROVIDER_NON_SECRET_DEFAULTS: dict[str, dict[str, object]] = {
    "audio_cpp": AudioCppConfig().to_mapping(),
    "openai": {
        "base_url": "https://api.openai.com/v1/audio/speech",
        "organization_id": "",
    },
    "elevenlabs": {
        "output_format": "mp3_44100_192",
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,
        "speaker_boost": True,
    },
    "kokoro": {
        "device": "cpu",
        "use_onnx": True,
        "onnx_model_path": "",
        "voices_json_path": "",
        "max_tokens": 500,
        "voice_mixing": False,
        "track_performance": True,
    },
    "chatterbox": {
        "device": "cpu",
        "voice_resource_directory": "~/.config/tldw_cli/chatterbox_voices",
        "temperature": 0.5,
        "chunk_size": 1024,
        "random_seed": "",
        "candidates": 1,
        "validate_whisper": False,
        "preprocess_text": True,
        "normalize_audio": True,
        "target_db": -20.0,
        "max_chunk_size": 500,
        "streaming": True,
        "stream_chunk_size": 4096,
        "crossfade": True,
        "crossfade_ms": 50,
    },
    "higgs": {
        "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
        "voice_resource_directory": "~/.config/tldw_cli/higgs_voices",
        "device": "auto",
        "enable_flash_attention": True,
        "dtype": "bfloat16",
        "max_reference_duration": 30,
        "language": "en",
        "voice_cloning": True,
        "multi_speaker": True,
        "speaker_delimiter": "|||",
        "track_performance": True,
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
    "alltalk": {
        "server_url": "http://127.0.0.1:7851",
        "language": "en",
    },
}

_CREDENTIAL_ENVIRONMENT_VARIABLES = MappingProxyType(
    {
        "openai": "OPENAI_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
    }
)
_CREDENTIAL_SETTING_KEYS = MappingProxyType(
    {
        "openai": "openai_api_key",
        "elevenlabs": "elevenlabs_api_key",
    }
)
_CREDENTIAL_LOCAL_LOCATIONS = MappingProxyType(
    {
        "openai": (
            ("api_settings.openai", "api_key"),
            ("openai_api", "api_key"),
            ("API", "openai_api_key"),
        ),
        "elevenlabs": (
            ("api_settings.elevenlabs", "api_key"),
            ("elevenlabs_api", "api_key"),
            ("API", "elevenlabs_api_key"),
        ),
    }
)

_MAX_GLOBAL_IDENTIFIER_CHARACTERS = 512
_UNSAFE_IDENTIFIER_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})


class CredentialSource(StrEnum):
    """Effective credential source without carrying a secret value."""

    ENVIRONMENT = "Environment"
    SAVED_LOCAL = "Saved local config"
    MISSING = "Missing"


class CredentialIntent(StrEnum):
    """Explicit local credential mutations accepted by ADR-012."""

    SET = "set"
    REPLACE = "replace"
    CLEAR = "clear"


@dataclass
class GlobalSpeechTTSDefaults:
    """Mutable draft of application-wide request defaults."""

    provider_id: str
    model_mode: str
    model_id: str | None
    voice_mode: str
    voice_id: str | None
    response_format: str
    speed: float

    def snapshot(
        self,
        *,
        max_identifier_characters: int = _MAX_GLOBAL_IDENTIFIER_CHARACTERS,
    ) -> TTSPreferencesSnapshot:
        """Return the validated immutable TTS admission snapshot."""
        provider_id = _choice(
            "defaults",
            "provider_id",
            self.provider_id,
            frozenset(BUILT_IN_TTS_PROVIDER_ORDER),
        )
        response_format = _choice(
            "defaults",
            "response_format",
            self.response_format,
            frozenset({"mp3", "opus", "aac", "flac", "wav"}),
        )
        speed = _number(
            "defaults",
            "default_speed",
            self.speed,
            0.25,
            4.0,
        )
        model_mode = _choice(
            "defaults",
            "model_mode",
            self.model_mode,
            frozenset({"exact", "first_available"}),
        )
        voice_mode = _choice(
            "defaults",
            "voice_mode",
            self.voice_mode,
            frozenset({"exact", "server_default"}),
        )
        model_id = (
            _identifier(
                "defaults",
                "default_model",
                self.model_id,
                max_characters=max_identifier_characters,
            )
            if model_mode == "exact"
            else self.model_id
        )
        voice_id = (
            _identifier(
                "defaults",
                "default_voice",
                self.voice_id,
                max_characters=max_identifier_characters,
            )
            if voice_mode == "exact"
            else self.voice_id
        )
        if provider_id == "audio_cpp":
            if response_format != "wav":
                _validation_error(
                    "defaults",
                    "response_format",
                    "audio.cpp global output must be WAV.",
                )
            if speed != 1.0:
                _validation_error(
                    "defaults",
                    "default_speed",
                    "audio.cpp global speed must be 1.0.",
                )
        try:
            return TTSPreferencesSnapshot(
                provider_id=provider_id,
                model_mode=model_mode,  # type: ignore[arg-type]
                model_id=model_id,
                voice_mode=voice_mode,  # type: ignore[arg-type]
                voice_id=voice_id,
                response_format=response_format,
                speed=speed,
            )
        except (TypeError, ValueError) as error:
            message = str(error)
            field_id = "provider_id"
            for candidate in (
                "model_mode",
                "default_model",
                "voice_mode",
                "default_voice",
                "response_format",
                "default_speed",
            ):
                if candidate in message:
                    field_id = candidate
                    break
            raise GlobalSpeechTTSValidationError(
                "defaults",
                field_id,
                "The global TTS default is incomplete or invalid.",
            ) from None


@dataclass(frozen=True, slots=True)
class GlobalSpeechTTSCredentialState:
    """Safe source metadata for one credential-capable provider."""

    provider_id: str
    setting_key: str
    environment_variable: str
    source: CredentialSource
    local_saved: bool
    local_shadowed: bool


@dataclass
class GlobalSpeechTTSState:
    """One editable global state; credentials contain metadata only."""

    defaults: GlobalSpeechTTSDefaults
    providers: dict[str, dict[str, object]]
    credentials: dict[str, GlobalSpeechTTSCredentialState]


@dataclass(frozen=True, slots=True)
class GlobalSpeechTTSSaveProposal:
    """Validated ordinary-Save proposal with no credential mutation."""

    settings: Mapping[str, object]
    delete_setting_keys: tuple[str, ...]
    preferences: TTSPreferencesSnapshot
    changed_provider_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GlobalSpeechTTSCredentialMutation:
    """One explicit Set/Replace/Clear local-config mutation."""

    provider_id: str
    setting_key: str
    value: str | None
    delete: bool


class GlobalSpeechTTSValidationError(ValueError):
    """Bounded field-specific validation failure without rejected input."""

    def __init__(self, provider_id: str, field_id: str, message: str) -> None:
        self.provider_id = provider_id
        self.field_id = field_id
        super().__init__(message)


def _raw_settings(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = settings.get("COMPREHENSIVE_CONFIG_RAW")
    return raw if isinstance(raw, Mapping) else settings


def _section(settings: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    raw = _raw_settings(settings)
    section = raw.get(name)
    if isinstance(section, Mapping):
        return section
    current: object = raw
    for part in name.split("."):
        if not isinstance(current, Mapping):
            current = None
            break
        current = current.get(part)
    if isinstance(current, Mapping):
        return current
    if name == "app_tts":
        normalized = settings.get("APP_TTS_CONFIG")
        if isinstance(normalized, Mapping):
            return normalized
    return {}


def _first_local_credential(
    settings: Mapping[str, Any],
    locations: tuple[tuple[str, str], ...],
) -> object | None:
    """Return the first configured local credential across canonical/legacy paths."""
    raw = _raw_settings(settings)
    for source in (raw, settings):
        for section_name, key in locations:
            section = source.get(section_name)
            if not isinstance(section, Mapping):
                current: object = source
                for part in section_name.split("."):
                    if not isinstance(current, Mapping):
                        current = None
                        break
                    current = current.get(part)
                section = current
            if isinstance(section, Mapping):
                value = section.get(key)
                if isinstance(value, str) and value:
                    return value
        if source is settings:
            break
    return None


def _value(
    section: Mapping[str, Any],
    key: str,
    default: object,
) -> object:
    return deepcopy(section.get(key, default))


def _credential_states(
    settings: Mapping[str, Any],
    environment: Mapping[str, str],
) -> dict[str, GlobalSpeechTTSCredentialState]:
    result: dict[str, GlobalSpeechTTSCredentialState] = {}
    for provider_id, variable in _CREDENTIAL_ENVIRONMENT_VARIABLES.items():
        setting_key = _CREDENTIAL_SETTING_KEYS[provider_id]
        local_value = _first_local_credential(
            settings,
            _CREDENTIAL_LOCAL_LOCATIONS[provider_id],
        )
        local_saved = isinstance(local_value, str) and bool(local_value)
        environment_present = bool(environment.get(variable))
        source = (
            CredentialSource.ENVIRONMENT
            if environment_present
            else CredentialSource.SAVED_LOCAL
            if local_saved
            else CredentialSource.MISSING
        )
        result[provider_id] = GlobalSpeechTTSCredentialState(
            provider_id=provider_id,
            setting_key=setting_key,
            environment_variable=variable,
            source=source,
            local_saved=local_saved,
            local_shadowed=environment_present and local_saved,
        )
    return result


def load_global_speech_tts_state(
    settings: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> GlobalSpeechTTSState:
    """Load global TTS state without provider I/O or secret projection."""
    if not isinstance(settings, Mapping):
        raise TypeError("Global Speech & TTS settings must be a mapping")
    environment = os.environ if environment is None else environment
    preferences = TTSPreferencesSnapshot.from_settings(settings)
    app_tts = _section(settings, "app_tts")
    higgs = _section(settings, "HiggsSettings")

    try:
        audio_cpp = AudioCppConfig.from_mapping(
            app_tts.get("audio_cpp", {})
            if isinstance(app_tts.get("audio_cpp", {}), Mapping)
            else {}
        ).to_mapping()
    except ValueError:
        audio_cpp = AudioCppConfig().to_mapping()

    providers = deepcopy(_PROVIDER_NON_SECRET_DEFAULTS)
    providers["audio_cpp"] = audio_cpp
    providers["openai"].update(
        {
            "base_url": _value(
                app_tts,
                "OPENAI_BASE_URL",
                providers["openai"]["base_url"],
            ),
            "organization_id": _value(app_tts, "OPENAI_ORG_ID", ""),
        }
    )
    providers["elevenlabs"].update(
        {
            "output_format": _value(
                app_tts,
                "ELEVENLABS_OUTPUT_FORMAT",
                providers["elevenlabs"]["output_format"],
            ),
            "stability": _value(
                app_tts,
                "ELEVENLABS_VOICE_STABILITY",
                providers["elevenlabs"]["stability"],
            ),
            "similarity_boost": _value(
                app_tts,
                "ELEVENLABS_SIMILARITY_BOOST",
                providers["elevenlabs"]["similarity_boost"],
            ),
            "style": _value(
                app_tts,
                "ELEVENLABS_STYLE",
                providers["elevenlabs"]["style"],
            ),
            "speaker_boost": _value(
                app_tts,
                "ELEVENLABS_USE_SPEAKER_BOOST",
                providers["elevenlabs"]["speaker_boost"],
            ),
        }
    )
    providers["kokoro"].update(
        {
            "device": _value(app_tts, "KOKORO_DEVICE_DEFAULT", "cpu"),
            "use_onnx": _value(app_tts, "KOKORO_USE_ONNX", True),
            "onnx_model_path": _value(app_tts, "KOKORO_ONNX_MODEL_PATH_DEFAULT", ""),
            "voices_json_path": _value(app_tts, "KOKORO_ONNX_VOICES_JSON_DEFAULT", ""),
            "max_tokens": _value(app_tts, "KOKORO_MAX_TOKENS", 500),
            "voice_mixing": _value(app_tts, "KOKORO_ENABLE_VOICE_MIXING", False),
            "track_performance": _value(app_tts, "KOKORO_TRACK_PERFORMANCE", True),
        }
    )
    providers["chatterbox"].update(
        {
            "device": _value(app_tts, "CHATTERBOX_DEVICE", "cpu"),
            "voice_resource_directory": _value(
                app_tts,
                "CHATTERBOX_VOICE_DIR",
                providers["chatterbox"]["voice_resource_directory"],
            ),
            "temperature": _value(app_tts, "CHATTERBOX_TEMPERATURE", 0.5),
            "chunk_size": _value(app_tts, "CHATTERBOX_CHUNK_SIZE", 1024),
            "random_seed": _value(app_tts, "CHATTERBOX_RANDOM_SEED", ""),
            "candidates": _value(app_tts, "CHATTERBOX_NUM_CANDIDATES", 1),
            "validate_whisper": _value(app_tts, "CHATTERBOX_VALIDATE_WHISPER", False),
            "preprocess_text": _value(app_tts, "CHATTERBOX_PREPROCESS_TEXT", True),
            "normalize_audio": _value(app_tts, "CHATTERBOX_NORMALIZE_AUDIO", True),
            "target_db": _value(app_tts, "CHATTERBOX_TARGET_DB", -20.0),
            "max_chunk_size": _value(app_tts, "CHATTERBOX_MAX_CHUNK_SIZE", 500),
            "streaming": _value(app_tts, "CHATTERBOX_STREAMING", True),
            "stream_chunk_size": _value(app_tts, "CHATTERBOX_STREAM_CHUNK_SIZE", 4096),
            "crossfade": _value(app_tts, "CHATTERBOX_ENABLE_CROSSFADE", True),
            "crossfade_ms": _value(app_tts, "CHATTERBOX_CROSSFADE_MS", 50),
        }
    )
    providers["higgs"].update(
        {
            "model_path": _value(higgs, "model_path", providers["higgs"]["model_path"]),
            "voice_resource_directory": _value(
                higgs,
                "voice_samples_dir",
                providers["higgs"]["voice_resource_directory"],
            ),
            "device": _value(higgs, "device", "auto"),
            "enable_flash_attention": _value(higgs, "enable_flash_attn", True),
            "dtype": _value(higgs, "dtype", "bfloat16"),
            "max_reference_duration": _value(higgs, "max_reference_duration", 30),
            "language": _value(higgs, "default_language", "en"),
            "voice_cloning": _value(higgs, "enable_voice_cloning", True),
            "multi_speaker": _value(higgs, "enable_multi_speaker", True),
            "speaker_delimiter": _value(higgs, "speaker_delimiter", "|||"),
            "track_performance": _value(higgs, "track_performance", True),
            "max_new_tokens": _value(higgs, "max_new_tokens", 4096),
            "temperature": _value(higgs, "temperature", 0.7),
            "top_p": _value(higgs, "top_p", 0.9),
            "repetition_penalty": _value(higgs, "repetition_penalty", 1.1),
        }
    )
    providers["alltalk"].update(
        {
            "server_url": _value(
                app_tts,
                "ALLTALK_TTS_URL_DEFAULT",
                providers["alltalk"]["server_url"],
            ),
            "language": _value(app_tts, "ALLTALK_TTS_LANGUAGE_DEFAULT", "en"),
        }
    )
    return GlobalSpeechTTSState(
        defaults=GlobalSpeechTTSDefaults(
            provider_id=preferences.provider_id,
            model_mode=preferences.model_mode,
            model_id=preferences.model_id,
            voice_mode=preferences.voice_mode,
            voice_id=preferences.voice_id,
            response_format=preferences.response_format,
            speed=preferences.speed,
        ),
        providers=providers,
        credentials=_credential_states(settings, environment),
    )


def _validation_error(provider_id: str, field_id: str, message: str) -> None:
    raise GlobalSpeechTTSValidationError(provider_id, field_id, message)


def _string(
    provider_id: str,
    field_id: str,
    value: object,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        _validation_error(provider_id, field_id, "Enter a valid text value.")
    if value != value.strip() or "\x00" in value or "\r" in value or "\n" in value:
        _validation_error(provider_id, field_id, "Enter a valid text value.")
    if not allow_empty and not value:
        _validation_error(provider_id, field_id, "This field is required.")
    if len(value) > 4096:
        _validation_error(provider_id, field_id, "This value is too long.")
    return value


def _identifier(
    provider_id: str,
    field_id: str,
    value: object,
    *,
    max_characters: int,
) -> str:
    """Validate an opaque model or voice identifier without echoing it."""
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > max_characters
        or any(
            unicodedata.category(character) in _UNSAFE_IDENTIFIER_CATEGORIES
            for character in value
        )
    ):
        _validation_error(
            provider_id,
            field_id,
            "Choose a valid saved identifier.",
        )
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        _validation_error(
            provider_id,
            field_id,
            "Choose a valid saved identifier.",
        )
    return value


def _choice(
    provider_id: str,
    field_id: str,
    value: object,
    allowed: frozenset[str],
) -> str:
    text = _string(provider_id, field_id, value)
    if text not in allowed:
        _validation_error(provider_id, field_id, "Choose a supported value.")
    return text


def _boolean(provider_id: str, field_id: str, value: object) -> bool:
    if type(value) is not bool:
        _validation_error(provider_id, field_id, "Choose enabled or disabled.")
    return value


def _number(
    provider_id: str,
    field_id: str,
    value: object,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        _validation_error(provider_id, field_id, "Enter a number in range.")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _validation_error(provider_id, field_id, "Enter a number in range.")
    if not math.isfinite(number) or not minimum <= number <= maximum:
        _validation_error(provider_id, field_id, "Enter a number in range.")
    return number


def _integer(
    provider_id: str,
    field_id: str,
    value: object,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        _validation_error(provider_id, field_id, "Enter a whole number in range.")
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _validation_error(provider_id, field_id, "Enter a whole number in range.")
    if str(value).strip() not in {str(integer), f"+{integer}"}:
        _validation_error(provider_id, field_id, "Enter a whole number in range.")
    if not minimum <= integer <= maximum:
        _validation_error(provider_id, field_id, "Enter a whole number in range.")
    return integer


def _url(provider_id: str, field_id: str, value: object) -> str:
    text = _string(provider_id, field_id, value)
    try:
        has_fragment = bool(urlsplit(text).fragment)
    except (UnicodeError, ValueError):
        has_fragment = True
    if not validate_url(text) or has_fragment:
        _validation_error(provider_id, field_id, "Enter a valid HTTP or HTTPS URL.")
    return text


def _server_url_without_query(
    provider_id: str,
    field_id: str,
    value: object,
) -> str:
    """Validate a server base URL that will have endpoint paths appended."""
    text = _url(provider_id, field_id, value)
    try:
        has_query = bool(urlsplit(text).query)
    except (UnicodeError, ValueError):
        has_query = True
    if has_query:
        _validation_error(
            provider_id,
            field_id,
            "Enter a server URL without a query string.",
        )
    return text


def _path_syntax(
    provider_id: str,
    field_id: str,
    value: object,
    *,
    allow_empty: bool = False,
) -> str:
    text = _string(provider_id, field_id, value, allow_empty=allow_empty)
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        _validation_error(provider_id, field_id, "Enter a valid path.")
    return text


def _validated_provider_values(
    provider_id: str,
    values: Mapping[str, object],
) -> dict[str, object]:
    if provider_id == "audio_cpp":
        candidate: dict[str, object] = {
            "mode": "external",
            "base_url": _string(provider_id, "base_url", values.get("base_url")),
            "connect_timeout_seconds": _number(
                provider_id,
                "connect_timeout_seconds",
                values.get("connect_timeout_seconds"),
                0.001,
                86_400,
            ),
            "synthesis_timeout_seconds": _number(
                provider_id,
                "synthesis_timeout_seconds",
                values.get("synthesis_timeout_seconds"),
                0.001,
                86_400,
            ),
        }
        for field_id in (
            "max_input_characters",
            "max_response_bytes",
            "max_metadata_bytes",
            "max_catalog_models",
            "max_voices_per_model",
            "max_identifier_characters",
        ):
            candidate[field_id] = _integer(
                provider_id,
                field_id,
                values.get(field_id),
                1,
                2**63 - 1,
            )
        try:
            return AudioCppConfig.from_mapping(candidate).to_mapping()
        except ValueError as error:
            message = str(error)
            field_id = next(
                (
                    field
                    for field in GLOBAL_TTS_PROVIDER_FIELD_IDS[provider_id]
                    if field in message
                ),
                "base_url",
            )
            _validation_error(
                provider_id,
                field_id,
                "The external audio.cpp setting is invalid.",
            )

    if provider_id == "openai":
        base_url = _url(provider_id, "base_url", values.get("base_url"))
        organization_id = _string(
            provider_id,
            "organization_id",
            values.get("organization_id", ""),
            allow_empty=True,
        )
        return {"base_url": base_url, "organization_id": organization_id}

    if provider_id == "elevenlabs":
        return {
            "output_format": _choice(
                provider_id,
                "output_format",
                values.get("output_format"),
                frozenset(
                    {
                        "mp3_44100_192",
                        "mp3_44100_128",
                        "mp3_44100_96",
                        "mp3_44100_64",
                        "mp3_44100_32",
                        "pcm_44100",
                        "pcm_24000",
                        "pcm_16000",
                        "ulaw_8000",
                    }
                ),
            ),
            "stability": _number(
                provider_id, "stability", values.get("stability"), 0.0, 1.0
            ),
            "similarity_boost": _number(
                provider_id,
                "similarity_boost",
                values.get("similarity_boost"),
                0.0,
                1.0,
            ),
            "style": _number(provider_id, "style", values.get("style"), 0.0, 1.0),
            "speaker_boost": _boolean(
                provider_id, "speaker_boost", values.get("speaker_boost")
            ),
        }

    if provider_id == "kokoro":
        return {
            "device": _choice(
                provider_id,
                "device",
                values.get("device"),
                frozenset({"cpu", "cuda", "mps"}),
            ),
            "use_onnx": _boolean(provider_id, "use_onnx", values.get("use_onnx")),
            "onnx_model_path": _path_syntax(
                provider_id,
                "onnx_model_path",
                values.get("onnx_model_path", ""),
                allow_empty=True,
            ),
            "voices_json_path": _path_syntax(
                provider_id,
                "voices_json_path",
                values.get("voices_json_path", ""),
                allow_empty=True,
            ),
            "max_tokens": _integer(
                provider_id, "max_tokens", values.get("max_tokens"), 1, 10_000
            ),
            "voice_mixing": _boolean(
                provider_id, "voice_mixing", values.get("voice_mixing")
            ),
            "track_performance": _boolean(
                provider_id,
                "track_performance",
                values.get("track_performance"),
            ),
        }

    if provider_id == "chatterbox":
        random_seed = values.get("random_seed", "")
        if random_seed == "":
            normalized_seed: int | str = ""
        else:
            normalized_seed = _integer(
                provider_id, "random_seed", random_seed, -(2**63), 2**63 - 1
            )
        return {
            "device": _choice(
                provider_id,
                "device",
                values.get("device"),
                frozenset({"cpu", "cuda"}),
            ),
            "voice_resource_directory": _path_syntax(
                provider_id,
                "voice_resource_directory",
                values.get("voice_resource_directory"),
            ),
            "temperature": _number(
                provider_id,
                "temperature",
                values.get("temperature"),
                0.0,
                2.0,
            ),
            "chunk_size": _integer(
                provider_id, "chunk_size", values.get("chunk_size"), 256, 8192
            ),
            "random_seed": normalized_seed,
            "candidates": _integer(
                provider_id, "candidates", values.get("candidates"), 1, 5
            ),
            "validate_whisper": _boolean(
                provider_id,
                "validate_whisper",
                values.get("validate_whisper"),
            ),
            "preprocess_text": _boolean(
                provider_id, "preprocess_text", values.get("preprocess_text")
            ),
            "normalize_audio": _boolean(
                provider_id, "normalize_audio", values.get("normalize_audio")
            ),
            "target_db": _number(
                provider_id, "target_db", values.get("target_db"), -40.0, 0.0
            ),
            "max_chunk_size": _integer(
                provider_id,
                "max_chunk_size",
                values.get("max_chunk_size"),
                50,
                5000,
            ),
            "streaming": _boolean(provider_id, "streaming", values.get("streaming")),
            "stream_chunk_size": _integer(
                provider_id,
                "stream_chunk_size",
                values.get("stream_chunk_size"),
                512,
                16_384,
            ),
            "crossfade": _boolean(provider_id, "crossfade", values.get("crossfade")),
            "crossfade_ms": _integer(
                provider_id,
                "crossfade_ms",
                values.get("crossfade_ms"),
                10,
                500,
            ),
        }

    if provider_id == "higgs":
        return {
            "model_path": _path_syntax(
                provider_id, "model_path", values.get("model_path")
            ),
            "voice_resource_directory": _path_syntax(
                provider_id,
                "voice_resource_directory",
                values.get("voice_resource_directory"),
            ),
            "device": _choice(
                provider_id,
                "device",
                values.get("device"),
                frozenset({"auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"}),
            ),
            "enable_flash_attention": _boolean(
                provider_id,
                "enable_flash_attention",
                values.get("enable_flash_attention"),
            ),
            "dtype": _choice(
                provider_id,
                "dtype",
                values.get("dtype"),
                frozenset({"float32", "float16", "bfloat16"}),
            ),
            "max_reference_duration": _integer(
                provider_id,
                "max_reference_duration",
                values.get("max_reference_duration"),
                1,
                60,
            ),
            "language": _choice(
                provider_id,
                "language",
                values.get("language"),
                frozenset({"en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"}),
            ),
            "voice_cloning": _boolean(
                provider_id, "voice_cloning", values.get("voice_cloning")
            ),
            "multi_speaker": _boolean(
                provider_id, "multi_speaker", values.get("multi_speaker")
            ),
            "speaker_delimiter": _string(
                provider_id,
                "speaker_delimiter",
                values.get("speaker_delimiter"),
            ),
            "track_performance": _boolean(
                provider_id,
                "track_performance",
                values.get("track_performance"),
            ),
            "max_new_tokens": _integer(
                provider_id,
                "max_new_tokens",
                values.get("max_new_tokens"),
                512,
                8192,
            ),
            "temperature": _number(
                provider_id,
                "temperature",
                values.get("temperature"),
                0.0,
                2.0,
            ),
            "top_p": _number(provider_id, "top_p", values.get("top_p"), 0.0, 1.0),
            "repetition_penalty": _number(
                provider_id,
                "repetition_penalty",
                values.get("repetition_penalty"),
                1.0,
                2.0,
            ),
        }

    if provider_id == "alltalk":
        return {
            "server_url": _server_url_without_query(
                provider_id, "server_url", values.get("server_url")
            ),
            "language": _choice(
                provider_id,
                "language",
                values.get("language"),
                frozenset({"en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"}),
            ),
        }

    raise ValueError("Unknown built-in TTS provider")


def _provider_event_settings(
    provider_id: str,
    values: Mapping[str, object],
) -> dict[str, object]:
    if provider_id == "audio_cpp":
        return {"audio_cpp": deepcopy(dict(values))}
    if provider_id == "openai":
        return {
            "OPENAI_BASE_URL": values["base_url"],
            "OPENAI_ORG_ID": values["organization_id"],
        }
    if provider_id == "elevenlabs":
        return {
            "ELEVENLABS_OUTPUT_FORMAT": values["output_format"],
            "ELEVENLABS_VOICE_STABILITY": values["stability"],
            "ELEVENLABS_SIMILARITY_BOOST": values["similarity_boost"],
            "ELEVENLABS_STYLE": values["style"],
            "ELEVENLABS_USE_SPEAKER_BOOST": values["speaker_boost"],
        }
    if provider_id == "kokoro":
        return {
            "KOKORO_DEVICE_DEFAULT": values["device"],
            "KOKORO_USE_ONNX": values["use_onnx"],
            "KOKORO_ONNX_MODEL_PATH_DEFAULT": values["onnx_model_path"],
            "KOKORO_ONNX_VOICES_JSON_DEFAULT": values["voices_json_path"],
            "KOKORO_MAX_TOKENS": values["max_tokens"],
            "KOKORO_ENABLE_VOICE_MIXING": values["voice_mixing"],
            "KOKORO_TRACK_PERFORMANCE": values["track_performance"],
        }
    if provider_id == "chatterbox":
        result = {
            "CHATTERBOX_DEVICE": values["device"],
            "CHATTERBOX_VOICE_DIR": values["voice_resource_directory"],
            "CHATTERBOX_TEMPERATURE": values["temperature"],
            "CHATTERBOX_CHUNK_SIZE": values["chunk_size"],
            "CHATTERBOX_NUM_CANDIDATES": values["candidates"],
            "CHATTERBOX_VALIDATE_WHISPER": values["validate_whisper"],
            "CHATTERBOX_PREPROCESS_TEXT": values["preprocess_text"],
            "CHATTERBOX_NORMALIZE_AUDIO": values["normalize_audio"],
            "CHATTERBOX_TARGET_DB": values["target_db"],
            "CHATTERBOX_MAX_CHUNK_SIZE": values["max_chunk_size"],
            "CHATTERBOX_STREAMING": values["streaming"],
            "CHATTERBOX_STREAM_CHUNK_SIZE": values["stream_chunk_size"],
            "CHATTERBOX_ENABLE_CROSSFADE": values["crossfade"],
            "CHATTERBOX_CROSSFADE_MS": values["crossfade_ms"],
        }
        if values["random_seed"] != "":
            result["CHATTERBOX_RANDOM_SEED"] = values["random_seed"]
        return result
    if provider_id == "higgs":
        return {
            "HIGGS_MODEL_PATH": values["model_path"],
            "HIGGS_VOICE_SAMPLES_DIR": values["voice_resource_directory"],
            "HIGGS_DEVICE": values["device"],
            "HIGGS_ENABLE_FLASH_ATTN": values["enable_flash_attention"],
            "HIGGS_DTYPE": values["dtype"],
            "HIGGS_MAX_REFERENCE_DURATION": values["max_reference_duration"],
            "HIGGS_DEFAULT_LANGUAGE": values["language"],
            "HIGGS_ENABLE_VOICE_CLONING": values["voice_cloning"],
            "HIGGS_ENABLE_MULTI_SPEAKER": values["multi_speaker"],
            "HIGGS_SPEAKER_DELIMITER": values["speaker_delimiter"],
            "HIGGS_TRACK_PERFORMANCE": values["track_performance"],
            "HIGGS_MAX_NEW_TOKENS": values["max_new_tokens"],
            "HIGGS_TEMPERATURE": values["temperature"],
            "HIGGS_TOP_P": values["top_p"],
            "HIGGS_REPETITION_PENALTY": values["repetition_penalty"],
        }
    if provider_id == "alltalk":
        return {
            "ALLTALK_TTS_URL_DEFAULT": values["server_url"],
            "ALLTALK_TTS_LANGUAGE_DEFAULT": values["language"],
        }
    raise ValueError("Unknown built-in TTS provider")


def build_global_speech_tts_save_proposal(
    original: GlobalSpeechTTSState,
    draft: GlobalSpeechTTSState,
    *,
    configure_provider: str,
) -> GlobalSpeechTTSSaveProposal:
    """Validate one ordinary Save and include only adapter-affecting edits."""
    if configure_provider not in BUILT_IN_TTS_PROVIDER_ORDER:
        raise ValueError("Unknown built-in TTS provider")
    identifier_limit = _MAX_GLOBAL_IDENTIFIER_CHARACTERS
    if draft.defaults.provider_id == "audio_cpp":
        configured_limit = _integer(
            "audio_cpp",
            "max_identifier_characters",
            draft.providers["audio_cpp"].get("max_identifier_characters"),
            1,
            2**63 - 1,
        )
        identifier_limit = min(identifier_limit, configured_limit)
    preferences = draft.defaults.snapshot(
        max_identifier_characters=identifier_limit,
    )
    validated = _validated_provider_values(
        configure_provider,
        draft.providers[configure_provider],
    )
    original_validated = _validated_provider_values(
        configure_provider,
        original.providers[configure_provider],
    )
    if validated == original_validated:
        settings: dict[str, object] = {}
        delete_setting_keys: tuple[str, ...] = ()
        changed_provider_ids: tuple[str, ...] = ()
    else:
        settings = _provider_event_settings(configure_provider, validated)
        delete_setting_keys = (
            ("CHATTERBOX_RANDOM_SEED",)
            if configure_provider == "chatterbox"
            and original_validated.get("random_seed") != ""
            and validated.get("random_seed") == ""
            else ()
        )
        changed_provider_ids = (configure_provider,)
    return GlobalSpeechTTSSaveProposal(
        settings=MappingProxyType(settings),
        delete_setting_keys=delete_setting_keys,
        preferences=preferences,
        changed_provider_ids=changed_provider_ids,
    )


def restore_non_secret_defaults(
    state: GlobalSpeechTTSState,
    *,
    configure_provider: str,
) -> GlobalSpeechTTSState:
    """Reset shared and selected-provider draft values, never credentials."""
    if configure_provider not in BUILT_IN_TTS_PROVIDER_ORDER:
        raise ValueError("Unknown built-in TTS provider")
    restored = deepcopy(state)
    default_preferences = TTSPreferencesSnapshot.from_settings({})
    restored.defaults = GlobalSpeechTTSDefaults(
        provider_id=default_preferences.provider_id,
        model_mode=default_preferences.model_mode,
        model_id=default_preferences.model_id,
        voice_mode=default_preferences.voice_mode,
        voice_id=default_preferences.voice_id,
        response_format=default_preferences.response_format,
        speed=default_preferences.speed,
    )
    restored.providers[configure_provider] = deepcopy(
        _PROVIDER_NON_SECRET_DEFAULTS[configure_provider]
    )
    return restored


def build_credential_mutation(
    state: GlobalSpeechTTSCredentialState,
    intent: CredentialIntent,
    value: str | None,
) -> GlobalSpeechTTSCredentialMutation:
    """Validate one explicit local Set/Replace/Clear operation."""
    if type(intent) is not CredentialIntent:
        raise TypeError("Credential intent is invalid")
    if intent is CredentialIntent.CLEAR:
        if not state.local_saved:
            raise ValueError("No saved local credential exists")
        if value is not None:
            raise ValueError("Clear credential does not accept a value")
        return GlobalSpeechTTSCredentialMutation(
            provider_id=state.provider_id,
            setting_key=state.setting_key,
            value=None,
            delete=True,
        )

    expected_intent = (
        CredentialIntent.REPLACE if state.local_saved else CredentialIntent.SET
    )
    if intent is not expected_intent:
        raise ValueError("Credential intent does not match saved local state")
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\r" in value
        or "\n" in value
        or "\x00" in value
        or value.lower() in {"<saved>", "<masked>", "saved"}
        or set(value) <= {"*", "•", "·"}
        or provider_api_key_validation_error(value) is not None
    ):
        raise ValueError("Enter a new credential value")
    return GlobalSpeechTTSCredentialMutation(
        provider_id=state.provider_id,
        setting_key=state.setting_key,
        value=value,
        delete=False,
    )
