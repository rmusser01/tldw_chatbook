"""Pure global Speech & TTS Settings state and validation.

The mapping in this module is intentionally bounded to Chatbook's seven
built-in providers.  It is an ownership contract for one Settings category,
not a provider plug-in or schema-driven form system.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from ipaddress import ip_address
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from tldw_chatbook.TTS.adapter_types import TTSNativeCapabilityObservation
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppManagedSetupSource,
    AudioCppSettingsConfig,
)
from tldw_chatbook.TTS.audio_cpp_guided_launch import (
    select_audio_cpp_guided_backend,
)
from tldw_chatbook.TTS.audio_cpp_managed_config import (
    validate_audio_cpp_managed_launch,
)
from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY
from tldw_chatbook.TTS.openai_compatible_config import (
    OpenAIAuthenticationMode,
    OpenAICompatibleEndpoint,
    normalize_openai_authentication_mode,
    normalize_openai_compatible_endpoint,
    openai_destination_fingerprint,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.sample_audio_validation import (
    CONTENT_TYPES_BY_FORMAT,
    MAX_PLAYABLE_AUDIO_BYTES,
    audio_body_matches_format,
    compressed_audio_has_decodable_frame,
    wav_has_complete_frames,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    ProviderTestFingerprint,
    SpeechTTSConfigurationState,
    SpeechTTSConnectionState,
    SpeechTTSTestOperation,
)
from tldw_chatbook.Utils.input_validation import (
    provider_api_key_validation_error,
    validate_url,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple

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

_AUDIO_CPP_MANAGED_VALIDATION_ERRORS = MappingProxyType(
    {
        "audio.cpp managed_binary_path must be an absolute executable file": (
            "managed_binary_path",
            "Choose an existing audiocpp_server file that is executable.",
        ),
        "audio.cpp managed_server_json_path must be an absolute readable file": (
            "managed_server_json_path",
            "Choose an existing server.json file that is readable.",
        ),
        "audio.cpp server.json must be at most 1048576 bytes": (
            "managed_server_json_path",
            "server.json must be 1 MiB or smaller.",
        ),
        "audio.cpp server.json must be UTF-8 JSON": (
            "managed_server_json_path",
            "server.json must use UTF-8 encoding.",
        ),
        "audio.cpp server.json must be strict JSON": (
            "managed_server_json_path",
            "server.json must contain strict JSON with no duplicate keys or "
            "non-JSON values.",
        ),
        "audio.cpp server.json must contain one JSON object": (
            "managed_server_json_path",
            "server.json must contain one JSON object.",
        ),
        "audio.cpp server.json host must be exactly 127.0.0.1": (
            "managed_server_json_path",
            "server.json must set host exactly to 127.0.0.1.",
        ),
        "audio.cpp server.json port must be an integer from 1 through 65535": (
            "managed_server_json_path",
            "server.json must set port to a whole number from 1 through 65535.",
        ),
    }
)

# Explicit global ownership destination.  The names are stable model keys;
# rendering remains deliberately provider-specific in the panel.
GLOBAL_TTS_PROVIDER_FIELD_IDS = MappingProxyType(
    {
        "audio_cpp": (
            "mode",
            "base_url",
            "managed_setup_source",
            "managed_binary_path",
            "managed_server_json_path",
            "guided_binary_path",
            "guided_binary_source",
            "guided_packages",
            "guided_default_model_id",
            "guided_backend_preference",
            "guided_device",
            "guided_threads",
            "guided_max_request_body_bytes",
            "guided_busy_timeout_ms",
            "managed_startup_timeout_seconds",
            "managed_health_check_interval_seconds",
            "managed_termination_grace_seconds",
            "connect_timeout_seconds",
            "synthesis_timeout_seconds",
            "max_input_characters",
            "max_response_bytes",
            "max_metadata_bytes",
            "max_catalog_models",
            "max_voices_per_model",
            "max_identifier_characters",
        ),
        "openai": (
            "credential",
            "authentication_mode",
            "base_url",
            "organization_id",
        ),
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
    "audio_cpp": AudioCppSettingsConfig().to_mapping(),
    "openai": {
        "authentication_mode": OpenAIAuthenticationMode.API_KEY.value,
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


class GlobalSpeechTTSEffectiveSource(StrEnum):
    """Bounded provenance labels used by the global configuration inspector."""

    ENVIRONMENT = "Environment"
    SAVED_LOCAL = "Saved local config"
    DEFAULT = "Default"
    INHERITED = "Inherited"


GLOBAL_TTS_PROVIDER_ENVIRONMENT_FIELDS = MappingProxyType(
    {
        "kokoro": MappingProxyType(
            {
                "onnx_model_path": "KOKORO_MODEL_PATH",
                "voices_json_path": "KOKORO_VOICES_PATH",
            }
        ),
        "higgs": MappingProxyType({"model_path": "HIGGS_MODEL_PATH"}),
    }
)
"""Legacy initialization fields whose process environment wins at runtime."""


class CredentialIntent(StrEnum):
    """Explicit local credential mutations accepted by ADR-012."""

    SET = "set"
    REPLACE = "replace"
    CLEAR = "clear"


class AudioCppExactChoiceState(StrEnum):
    """Truthful status for one exact audio.cpp selector axis."""

    NOT_OBSERVED = "Not observed"
    FRESH = "Fresh"
    STALE = "Stale"
    UNVERIFIED = "Unverified"
    MISSING = "Missing"


@dataclass(frozen=True, slots=True)
class AudioCppAxisChoices:
    """Safe exact choices and observation state for one selector axis."""

    options: tuple[tuple[str, str], ...]
    state: AudioCppExactChoiceState

    @property
    def exact_allowed(self) -> bool:
        """Return whether the Exact policy can be represented safely."""
        return bool(self.options)


@dataclass(frozen=True, slots=True)
class AudioCppGlobalChoices:
    """Read-only projection of the latest accepted audio.cpp observation."""

    model: AudioCppAxisChoices
    voice: AudioCppAxisChoices
    configuration_revision: int | None
    catalog_revision: int | None
    observed_at: datetime | None


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
    default_profile_id: str | None = None

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


@dataclass(frozen=True, slots=True)
class OpenAIPlaintextConfirmation:
    """Non-secret consent bound only to one normalized endpoint origin."""

    origin_fingerprint: str

    def __post_init__(self) -> None:
        value = self.origin_fingerprint
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("OpenAI plaintext confirmation is invalid")


@dataclass
class GlobalSpeechTTSState:
    """One editable global state; credentials contain metadata only."""

    defaults: GlobalSpeechTTSDefaults
    providers: dict[str, dict[str, object]]
    credentials: dict[str, GlobalSpeechTTSCredentialState]
    defaults_source: GlobalSpeechTTSEffectiveSource
    provider_sources: dict[str, GlobalSpeechTTSEffectiveSource]
    provider_field_sources: dict[str, dict[str, GlobalSpeechTTSEffectiveSource]]
    openai_plaintext_confirmation: OpenAIPlaintextConfirmation | None = None
    openai_plaintext_confirmation_cleanup_needed: bool = False


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


class ProcessProviderTestEvidenceStore:
    """Process-only successful sample evidence keyed by provider fingerprint."""

    _DEFAULT_MAX_SAMPLE_BYTES = MAX_PLAYABLE_AUDIO_BYTES
    _CONTENT_TYPES_BY_FORMAT = CONTENT_TYPES_BY_FORMAT

    def __init__(self) -> None:
        self._successful_samples: dict[str, ProviderTestFingerprint] = {}
        self._catalog_states: dict[
            str, tuple[ProviderTestFingerprint, SpeechTTSConnectionState]
        ] = {}

    @staticmethod
    def _wav_has_complete_frames(body: bytes) -> bool:
        return wav_has_complete_frames(body)

    @staticmethod
    def _compressed_audio_has_decodable_frame(
        body: bytes,
        response_format: str,
    ) -> bool:
        """Decode at most one bounded audio frame, failing closed without PyAV."""

        return compressed_audio_has_decodable_frame(body, response_format)

    @classmethod
    def _audio_body_matches_format(
        cls,
        body: bytes,
        response_format: str,
        *,
        sample_rate_hz: int | None,
        channels: int | None,
        sample_width_bytes: int | None,
    ) -> bool:
        return audio_body_matches_format(
            body,
            response_format,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
        )

    def record_successful_sample(
        self,
        fingerprint: ProviderTestFingerprint,
        *,
        status_code: int,
        response_format: str,
        body: bytes,
        content_type: str | None = None,
        max_bytes: int = _DEFAULT_MAX_SAMPLE_BYTES,
        sample_rate_hz: int | None = None,
        channels: int | None = None,
        sample_width_bytes: int | None = None,
    ) -> bool:
        """Record only a bounded, format-valid successful speech response."""

        if type(fingerprint) is not ProviderTestFingerprint:
            raise TypeError("Provider test fingerprint is invalid")
        if (
            type(status_code) is not int
            or not 200 <= status_code < 300
            or type(response_format) is not str
            or type(body) is not bytes
            or (
                content_type is not None
                and (
                    type(content_type) is not str
                    or content_type.split(";", 1)[0].strip().lower()
                    not in self._CONTENT_TYPES_BY_FORMAT.get(
                        response_format.lower(),
                        frozenset(),
                    )
                )
            )
            or type(max_bytes) is not int
            or max_bytes <= 0
            or not 0 < len(body) <= max_bytes
            or not self._audio_body_matches_format(
                body,
                response_format.lower(),
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                sample_width_bytes=sample_width_bytes,
            )
        ):
            return False
        self._successful_samples[fingerprint.provider_id] = fingerprint
        return True

    def record_catalog(
        self,
        fingerprint: ProviderTestFingerprint,
        state: SpeechTTSConnectionState,
    ) -> None:
        """Record one bounded catalog observation for the exact fingerprint."""

        if type(fingerprint) is not ProviderTestFingerprint:
            raise TypeError("Provider test fingerprint is invalid")
        if type(state) is not SpeechTTSConnectionState:
            raise TypeError("Speech TTS catalog state is invalid")
        self._catalog_states[fingerprint.provider_id] = (fingerprint, state)

    def sample_state(
        self, fingerprint: ProviderTestFingerprint
    ) -> SpeechTTSConnectionState:
        return (
            SpeechTTSConnectionState.REACHABLE
            if self._successful_samples.get(fingerprint.provider_id) == fingerprint
            else SpeechTTSConnectionState.NOT_TESTED
        )

    def catalog_state(
        self, fingerprint: ProviderTestFingerprint
    ) -> SpeechTTSConnectionState:
        evidence = self._catalog_states.get(fingerprint.provider_id)
        if evidence is None or evidence[0] != fingerprint:
            return SpeechTTSConnectionState.NOT_TESTED
        return evidence[1]

    def sample_operation(
        self, fingerprint: ProviderTestFingerprint
    ) -> SpeechTTSTestOperation:
        return SpeechTTSTestOperation.SAMPLE


def process_provider_test_evidence_store(
    owner: object,
) -> ProcessProviderTestEvidenceStore:
    """Return one non-persisted evidence store for an app-process owner."""

    attribute = "_tts_provider_test_evidence"
    existing = getattr(owner, attribute, None)
    if type(existing) is ProcessProviderTestEvidenceStore:
        return existing
    store = ProcessProviderTestEvidenceStore()
    setattr(owner, attribute, store)
    return store


def build_provider_test_fingerprint(
    state: GlobalSpeechTTSState,
    *,
    provider_id: str,
    saved_revision: int,
) -> ProviderTestFingerprint:
    """Build the process-local test identity for one provider configuration."""

    if provider_id not in BUILT_IN_TTS_PROVIDER_ORDER:
        raise ValueError("Unknown built-in TTS provider")
    if type(saved_revision) is not int or saved_revision < 0:
        raise ValueError("Saved provider revision must be a non-negative integer")

    raw_values = state.providers.get(provider_id)
    if not isinstance(raw_values, Mapping):
        raise TypeError("Provider configuration is unavailable")
    allowed_fields = GLOBAL_TTS_PROVIDER_FIELD_IDS[provider_id]
    validated = _validated_provider_values(
        provider_id,
        {
            field_id: raw_values[field_id]
            for field_id in allowed_fields
            if field_id != "credential" and field_id in raw_values
        },
    )

    def render(value: object) -> str:
        if type(value) is bool:
            return "true" if value else "false"
        if value is None:
            return ""
        if isinstance(value, (tuple, list)):
            return "[" + ",".join(render(item) for item in value) + "]"
        if isinstance(value, Mapping):
            return (
                "{"
                + ",".join(
                    f"{key!s}:{render(value[key])}" for key in sorted(value, key=str)
                )
                + "}"
            )
        return str(value)

    normalized = {key: render(value) for key, value in validated.items()}
    credential = state.credentials.get(provider_id)
    credential_required = not (
        provider_id == "openai"
        and validated.get("authentication_mode") == OpenAIAuthenticationMode.NONE.value
    )
    if credential is not None:
        normalized["credential_present"] = (
            "true"
            if credential_required and credential.source is not CredentialSource.MISSING
            else "false"
        )
        normalized["credential_source"] = (
            credential.source.value if credential_required else "not_used"
        )
    return ProviderTestFingerprint(
        provider_id=provider_id,
        normalized_fields=tuple(sorted(normalized.items())),
        saved_revision=saved_revision,
    )


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
    persisted = settings.get("COMPREHENSIVE_CONFIG_RAW")
    # When the raw persisted mapping is available it is the only authority for
    # whether a local secret exists. Normalized compatibility projections may
    # contain an environment-resolved credential and must not turn that value
    # back into a fictitious saved fallback. Plain mappings without the raw
    # envelope remain supported for focused callers and tests.
    sources = (persisted,) if isinstance(persisted, Mapping) else (settings,)
    for source in sources:
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
    return None


def _value(
    section: Mapping[str, Any],
    key: str,
    default: object,
) -> object:
    return deepcopy(section.get(key, default))


def _normalize_default_profile_id(value: object) -> str | None:
    """Normalize an ``app_tts.default_profile_id`` load value.

    Absent, non-string, empty, and whitespace-only values load as ``None``.
    A non-empty string that is not a well-formed UUID still loads as-is: it
    is a defined dangling state that later surfaces honestly and refuses at
    speak time rather than being silently discarded here.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _load_openai_authentication_mode(
    raw_mode: object,
    base_url: object,
) -> str:
    """Load authentication fail-closed when mode or destination is untrusted."""

    if raw_mode not in {
        OpenAIAuthenticationMode.API_KEY.value,
        OpenAIAuthenticationMode.NONE.value,
    }:
        return OpenAIAuthenticationMode.API_KEY.value
    try:
        endpoint = normalize_openai_compatible_endpoint(str(base_url))
        return normalize_openai_authentication_mode(
            raw_mode,
            endpoint=endpoint,
        ).value
    except (TypeError, ValueError):
        return OpenAIAuthenticationMode.API_KEY.value


def _load_openai_plaintext_confirmation(
    value: object,
    *,
    authentication_mode: object,
    base_url: object,
) -> OpenAIPlaintextConfirmation | None:
    if authentication_mode != OpenAIAuthenticationMode.NONE.value:
        return None
    try:
        confirmation = OpenAIPlaintextConfirmation(value)  # type: ignore[arg-type]
        endpoint = normalize_openai_compatible_endpoint(str(base_url))
    except (TypeError, ValueError):
        return None
    if (
        urlsplit(endpoint.origin).scheme != "http"
        or _is_loopback_openai_endpoint(endpoint)
        or confirmation.origin_fingerprint
        != openai_destination_fingerprint("openai", endpoint)
    ):
        return None
    return confirmation


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

    raw_audio_cpp = app_tts.get("audio_cpp", {})
    try:
        if not isinstance(raw_audio_cpp, Mapping):
            raise ValueError("Invalid audio.cpp settings")
        load_values = AudioCppSettingsConfig().to_mapping()
        load_values.update(
            {
                field_id: deepcopy(raw_audio_cpp[field_id])
                for field_id in GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"]
                if field_id in raw_audio_cpp
            }
        )
        audio_cpp = _validated_provider_values("audio_cpp", load_values)
    except (GlobalSpeechTTSValidationError, ValueError):
        audio_cpp = AudioCppSettingsConfig().to_mapping()

    providers = deepcopy(_PROVIDER_NON_SECRET_DEFAULTS)
    providers["audio_cpp"] = audio_cpp
    openai_base_url = _value(
        app_tts,
        "OPENAI_BASE_URL",
        providers["openai"]["base_url"],
    )
    providers["openai"].update(
        {
            "authentication_mode": _load_openai_authentication_mode(
                app_tts.get("OPENAI_AUTH_MODE"),
                openai_base_url,
            ),
            "base_url": openai_base_url,
            "organization_id": _value(app_tts, "OPENAI_ORG_ID", ""),
        }
    )
    raw_app_tts = _raw_settings(settings).get("app_tts")
    openai_confirmation_persisted = (
        isinstance(raw_app_tts, Mapping)
        and "OPENAI_NONE_HTTP_CONFIRMATION" in raw_app_tts
    )
    openai_plaintext_confirmation = _load_openai_plaintext_confirmation(
        app_tts.get("OPENAI_NONE_HTTP_CONFIRMATION"),
        authentication_mode=providers["openai"]["authentication_mode"],
        base_url=providers["openai"]["base_url"],
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
    credentials = _credential_states(settings, environment)
    raw = _raw_settings(settings)
    raw_app_tts = raw.get("app_tts")
    raw_app_tts = raw_app_tts if isinstance(raw_app_tts, Mapping) else {}
    raw_tts_settings = raw.get("tts_settings")
    raw_tts_settings = raw_tts_settings if isinstance(raw_tts_settings, Mapping) else {}
    raw_higgs = raw.get("HiggsSettings")
    raw_higgs = raw_higgs if isinstance(raw_higgs, Mapping) else {}
    default_keys = {
        "default_provider",
        "default_model_mode",
        "default_model",
        "default_voice_mode",
        "default_voice",
        "default_format",
        "default_speed",
    }
    defaults_saved = bool(default_keys.intersection(raw_app_tts)) or bool(
        {
            "default_tts_provider",
            "default_openai_tts_model",
            "default_tts_voice",
            "default_openai_tts_output_format",
            "default_openai_tts_speed",
        }.intersection(raw_tts_settings)
    )
    provider_prefixes = {
        "openai": ("OPENAI_",),
        "elevenlabs": ("ELEVENLABS_",),
        "kokoro": ("KOKORO_",),
        "chatterbox": ("CHATTERBOX_",),
        "alltalk": ("ALLTALK_",),
    }
    provider_sources = {
        provider_id: GlobalSpeechTTSEffectiveSource.DEFAULT
        for provider_id in BUILT_IN_TTS_PROVIDER_ORDER
    }
    if isinstance(raw_app_tts.get("audio_cpp"), Mapping):
        provider_sources["audio_cpp"] = GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
    if raw_higgs:
        provider_sources["higgs"] = GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
    for provider_id, prefixes in provider_prefixes.items():
        if any(
            any(key.startswith(prefix) for prefix in prefixes)
            for key in raw_app_tts
            if isinstance(key, str)
        ):
            provider_sources[provider_id] = GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
    for provider_id, credential in credentials.items():
        if credential.source is CredentialSource.ENVIRONMENT:
            provider_sources[provider_id] = GlobalSpeechTTSEffectiveSource.ENVIRONMENT
        elif (
            credential.source is CredentialSource.SAVED_LOCAL
            and provider_sources[provider_id] is GlobalSpeechTTSEffectiveSource.DEFAULT
        ):
            provider_sources[provider_id] = GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
    provider_field_sources: dict[str, dict[str, GlobalSpeechTTSEffectiveSource]] = {
        provider_id: {} for provider_id in BUILT_IN_TTS_PROVIDER_ORDER
    }
    for provider_id, field_variables in GLOBAL_TTS_PROVIDER_ENVIRONMENT_FIELDS.items():
        for field_id, variable in field_variables.items():
            if variable in environment:
                provider_field_sources[provider_id][field_id] = (
                    GlobalSpeechTTSEffectiveSource.ENVIRONMENT
                )
                provider_sources[provider_id] = (
                    GlobalSpeechTTSEffectiveSource.ENVIRONMENT
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
            default_profile_id=_normalize_default_profile_id(
                _value(app_tts, "default_profile_id", None)
            ),
        ),
        providers=providers,
        credentials=credentials,
        defaults_source=(
            GlobalSpeechTTSEffectiveSource.SAVED_LOCAL
            if defaults_saved
            else GlobalSpeechTTSEffectiveSource.DEFAULT
        ),
        provider_sources=provider_sources,
        provider_field_sources=provider_field_sources,
        openai_plaintext_confirmation=openai_plaintext_confirmation,
        openai_plaintext_confirmation_cleanup_needed=(
            openai_confirmation_persisted and openai_plaintext_confirmation is None
        ),
    )


def audio_cpp_transport_warning(value: object) -> str | None:
    """Return a fixed warning for a valid non-loopback plain-HTTP origin.

    Invalid drafts are left to field validation, and the submitted origin is
    never included in the warning text.
    """
    try:
        canonical = AudioCppConfig.from_mapping({"base_url": value}).base_url
        parsed = urlsplit(canonical)
    except (TypeError, UnicodeError, ValueError):
        return None
    if parsed.scheme != "http" or parsed.hostname is None:
        return None
    hostname = parsed.hostname.rstrip(".").lower()
    local = hostname == "localhost" or hostname.endswith(".localhost")
    if not local:
        try:
            local = ip_address(hostname).is_loopback
        except ValueError:
            pass
    if local:
        return None
    return (
        "Warning: this non-loopback HTTP server is not transport-encrypted; "
        "submitted text and returned audio may be visible in transit. Use HTTPS "
        "when available."
    )


def _pinned_option(identifier: str, state: AudioCppExactChoiceState) -> tuple[str, str]:
    suffix = "Missing" if state is AudioCppExactChoiceState.MISSING else "Unverified"
    return (f"{identifier} ({suffix})", identifier)


def _append_pinned_choice(
    options: tuple[tuple[str, str], ...],
    identifier: str | None,
    state: AudioCppExactChoiceState,
) -> tuple[tuple[str, str], ...]:
    if not identifier or any(value == identifier for _label, value in options):
        return options
    return (*options, _pinned_option(identifier, state))


def project_audio_cpp_global_choices(
    defaults: GlobalSpeechTTSDefaults,
    *,
    observation: TTSNativeCapabilityObservation | None,
    current_configuration_revision: int | None,
    saved_configuration_revision: int | None = None,
    applied_configuration_revision: int | None = None,
) -> AudioCppGlobalChoices:
    """Project cached audio.cpp choices without performing provider work."""
    pinned_model = (
        defaults.model_id
        if defaults.model_mode == "exact" and isinstance(defaults.model_id, str)
        else None
    )
    pinned_voice = (
        defaults.voice_id
        if defaults.voice_mode == "exact" and isinstance(defaults.voice_id, str)
        else None
    )
    if observation is None or observation.snapshot.catalog is None:
        model_state = (
            AudioCppExactChoiceState.UNVERIFIED
            if pinned_model
            else AudioCppExactChoiceState.NOT_OBSERVED
        )
        voice_state = (
            AudioCppExactChoiceState.UNVERIFIED
            if pinned_voice
            else AudioCppExactChoiceState.NOT_OBSERVED
        )
        model_options = _append_pinned_choice((), pinned_model, model_state)
        voice_options = _append_pinned_choice((), pinned_voice, voice_state)
        return AudioCppGlobalChoices(
            model=AudioCppAxisChoices(model_options, model_state),
            voice=AudioCppAxisChoices(voice_options, voice_state),
            configuration_revision=(
                observation.snapshot.configuration_revision if observation else None
            ),
            catalog_revision=None,
            observed_at=observation.observed_at if observation else None,
        )

    snapshot = observation.snapshot
    catalog = snapshot.catalog
    applied_matches_saved = (
        saved_configuration_revision is None
        or applied_configuration_revision is None
        or applied_configuration_revision == saved_configuration_revision
    )
    same_configuration = (
        applied_matches_saved
        and current_configuration_revision is not None
        and snapshot.configuration_revision == current_configuration_revision
    )
    fresh = same_configuration and catalog.health.fresh
    model_options = tuple(
        (model.display_name or model.model_id, model.model_id)
        for model in catalog.models
    )

    if not fresh:
        state = AudioCppExactChoiceState.STALE
        model_options = _append_pinned_choice(model_options, pinned_model, state)
        voice_options: tuple[tuple[str, str], ...] = ()
        result = snapshot.voice_results.get(pinned_model) if pinned_model else None
        if result is not None and result.state == "complete":
            voice_options = tuple((voice, voice) for voice in result.voices)
        voice_options = _append_pinned_choice(voice_options, pinned_voice, state)
        return AudioCppGlobalChoices(
            model=AudioCppAxisChoices(model_options, state),
            voice=AudioCppAxisChoices(voice_options, state),
            configuration_revision=snapshot.configuration_revision,
            catalog_revision=catalog.revision,
            observed_at=observation.observed_at,
        )

    model_ids = {value for _label, value in model_options}
    if pinned_model and pinned_model not in model_ids:
        model_state = (
            AudioCppExactChoiceState.UNVERIFIED
            if catalog.approximate
            else AudioCppExactChoiceState.MISSING
        )
        model_options = _append_pinned_choice(
            model_options,
            pinned_model,
            model_state,
        )
    else:
        model_state = AudioCppExactChoiceState.FRESH

    voice_options: tuple[tuple[str, str], ...] = ()
    voice_state = AudioCppExactChoiceState.UNVERIFIED
    selected_model_is_known = pinned_model is not None and pinned_model in model_ids
    result = (
        snapshot.voice_results.get(pinned_model) if selected_model_is_known else None
    )
    if (
        result is not None
        and result.state == "complete"
        and result.catalog_revision == catalog.revision
    ):
        voice_options = tuple((voice, voice) for voice in result.voices)
        voice_ids = {value for _label, value in voice_options}
        if pinned_voice and pinned_voice not in voice_ids:
            voice_state = AudioCppExactChoiceState.MISSING
            voice_options = _append_pinned_choice(
                voice_options,
                pinned_voice,
                voice_state,
            )
        else:
            voice_state = AudioCppExactChoiceState.FRESH
    else:
        voice_options = _append_pinned_choice(
            voice_options,
            pinned_voice,
            AudioCppExactChoiceState.UNVERIFIED,
        )

    return AudioCppGlobalChoices(
        model=AudioCppAxisChoices(model_options, model_state),
        voice=AudioCppAxisChoices(voice_options, voice_state),
        configuration_revision=snapshot.configuration_revision,
        catalog_revision=catalog.revision,
        observed_at=observation.observed_at,
    )


def _validation_error(provider_id: str, field_id: str, message: str) -> None:
    raise GlobalSpeechTTSValidationError(provider_id, field_id, message)


def detect_audio_cpp_server_binary() -> str | None:
    """Return the platform-resolved ``audiocpp_server`` path, if present.

    Detection is deliberately an explicit draft helper. It does not validate,
    persist, execute, or contact the discovered program.

    Returns:
        The exact detected executable path, or ``None`` when it is unavailable.
    """

    detected = shutil.which("audiocpp_server")
    return detected if isinstance(detected, str) and detected else None


def validate_audio_cpp_managed_settings(values: Mapping[str, object]) -> None:
    """Validate selected Managed artifacts with bounded field diagnostics.

    External mode returns without touching dormant managed paths. Managed mode
    reuses the launch validator, which reads but never modifies the selected
    executable or ``server.json`` and performs no process or network work.

    Args:
        values: Full two-mode audio.cpp Settings draft.

    Raises:
        GlobalSpeechTTSValidationError: If the selected Managed artifacts are
            invalid or unsafe.
    """

    validated = _validated_provider_values("audio_cpp", values)
    if validated.get("mode") != "managed":
        return

    setup_source = validated.get(
        "managed_setup_source",
        AudioCppManagedSetupSource.USER_JSON.value,
    )
    if setup_source == AudioCppManagedSetupSource.GUIDED.value:
        try:
            config = AudioCppSettingsConfig.from_mapping(validated)
        except ValueError:
            raise GlobalSpeechTTSValidationError(
                "audio_cpp",
                "managed_setup_source",
                "Review the Guided audio.cpp settings.",
            ) from None
        try:
            binary = validate_path_simple(
                config.guided_binary_path,
                require_exists=True,
            )
            info = binary.stat()
            executable = (
                binary.is_absolute()
                and stat.S_ISREG(info.st_mode)
                and os.access(binary, os.X_OK)
            )
        except OSError:
            executable = False
        if not executable:
            raise GlobalSpeechTTSValidationError(
                "audio_cpp",
                "guided_binary_path",
                "Choose an existing audiocpp_server file that is executable.",
            ) from None
        if not config.guided_packages:
            raise GlobalSpeechTTSValidationError(
                "audio_cpp",
                "guided_packages",
                "Add and review at least one compatible model package.",
            ) from None
        if config.guided_default_model_id is None:
            raise GlobalSpeechTTSValidationError(
                "audio_cpp",
                "guided_default_model_id",
                "Choose the default model for Guided audio.cpp speech.",
            ) from None
        try:
            recipes = tuple(
                AUDIO_CPP_RECIPE_REGISTRY.validate_accepted(package)
                for package in config.guided_packages
            )
        except ValueError:
            raise GlobalSpeechTTSValidationError(
                "audio_cpp",
                "guided_packages",
                "One or more model packages require review.",
            ) from None
        if (
            select_audio_cpp_guided_backend(
                config.guided_backend_preference,
                recipes,
            )
            is None
        ):
            raise GlobalSpeechTTSValidationError(
                "audio_cpp",
                "guided_backend_preference",
                "Choose Auto or a backend supported by every reviewed package "
                "on this host.",
            ) from None
        return

    config = AudioCppConfig.from_mapping(validated)
    failure: tuple[str, str] | None = None
    try:
        validate_audio_cpp_managed_launch(config)
    except ValueError as error:
        failure = _AUDIO_CPP_MANAGED_VALIDATION_ERRORS.get(
            str(error),
            (
                "managed_server_json_path",
                "Review server.json and choose a valid managed server configuration.",
            ),
        )

    if failure is not None:
        raise GlobalSpeechTTSValidationError("audio_cpp", *failure)


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
        allowed_fields = frozenset(GLOBAL_TTS_PROVIDER_FIELD_IDS["audio_cpp"])
        if set(values) - allowed_fields:
            _validation_error(
                provider_id,
                "mode",
                "The audio.cpp settings contain an unsupported field.",
            )
        mode = _choice(
            provider_id,
            "mode",
            values.get("mode", "external"),
            frozenset({"external", "managed"}),
        )
        candidate: dict[str, object] = {
            "mode": mode,
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
        if mode == "external":
            candidate["base_url"] = _string(
                provider_id,
                "base_url",
                values.get("base_url"),
            )
        else:
            setup_source = _choice(
                provider_id,
                "managed_setup_source",
                values.get("managed_setup_source", "user_json"),
                frozenset({"user_json", "guided"}),
            )
            candidate["managed_setup_source"] = setup_source
            if setup_source == "user_json":
                candidate.update(
                    {
                        "managed_binary_path": _path_syntax(
                            provider_id,
                            "managed_binary_path",
                            values.get("managed_binary_path"),
                        ),
                        "managed_server_json_path": _path_syntax(
                            provider_id,
                            "managed_server_json_path",
                            values.get("managed_server_json_path"),
                        ),
                    }
                )
            for field_id, minimum, maximum in (
                ("managed_startup_timeout_seconds", 1.0, 300.0),
                ("managed_health_check_interval_seconds", 2.0, 300.0),
                ("managed_termination_grace_seconds", 0.1, 60.0),
            ):
                candidate[field_id] = _number(
                    provider_id,
                    field_id,
                    values.get(field_id),
                    minimum,
                    maximum,
                )
            if setup_source == "guided":
                try:
                    guided_values = dict(values)
                    guided_values.update(candidate)
                    return AudioCppSettingsConfig.from_mapping(
                        guided_values
                    ).to_mapping()
                except ValueError as error:
                    field_id = "managed_setup_source"
                    errors = getattr(error, "errors", None)
                    if callable(errors):
                        details = errors()
                        if details:
                            location = details[0].get("loc", ())
                            if location and location[0] in allowed_fields:
                                field_id = str(location[0])
                    _validation_error(
                        provider_id,
                        field_id,
                        "Review the Guided audio.cpp setting.",
                    )
        try:
            projected = AudioCppConfig.from_mapping(candidate).to_mapping()
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
        durable = AudioCppSettingsConfig().to_mapping()
        durable.update(
            {
                field_id: deepcopy(values[field_id])
                for field_id in GLOBAL_TTS_PROVIDER_FIELD_IDS[provider_id]
                if field_id in values
            }
        )
        durable.update(projected)
        if mode == "managed" and "base_url" not in durable:
            durable["base_url"] = AudioCppConfig().base_url
        return durable

    if provider_id == "openai":
        try:
            endpoint = normalize_openai_compatible_endpoint(
                _string(provider_id, "base_url", values.get("base_url"))
            )
        except ValueError:
            _validation_error(
                provider_id,
                "base_url",
                "Enter a valid OpenAI-compatible speech endpoint.",
            )
        try:
            authentication_mode = normalize_openai_authentication_mode(
                values.get("authentication_mode"),
                endpoint=endpoint,
            ).value
        except ValueError:
            _validation_error(
                provider_id,
                "authentication_mode",
                "Official OpenAI requires API key authentication.",
            )
        organization_id = _string(
            provider_id,
            "organization_id",
            values.get("organization_id", ""),
            allow_empty=True,
        )
        return {
            "authentication_mode": authentication_mode,
            "base_url": endpoint.speech_url,
            "organization_id": organization_id,
        }

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
            "OPENAI_AUTH_MODE": values["authentication_mode"],
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


def _is_loopback_openai_endpoint(endpoint: OpenAICompatibleEndpoint) -> bool:
    hostname = urlsplit(endpoint.origin).hostname
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def required_openai_plaintext_confirmation_fingerprint(
    state: GlobalSpeechTTSState,
) -> str | None:
    """Return the consent fingerprint required by the current OpenAI draft."""

    try:
        values = _validated_provider_values("openai", state.providers["openai"])
        endpoint = normalize_openai_compatible_endpoint(str(values["base_url"]))
    except (KeyError, GlobalSpeechTTSValidationError, TypeError, ValueError):
        return None
    if (
        values["authentication_mode"] != OpenAIAuthenticationMode.NONE.value
        or urlsplit(endpoint.origin).scheme != "http"
        or _is_loopback_openai_endpoint(endpoint)
    ):
        return None
    return openai_destination_fingerprint("openai", endpoint)


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
    try:
        original_validated = _validated_provider_values(
            configure_provider,
            original.providers[configure_provider],
        )
    except GlobalSpeechTTSValidationError:
        if configure_provider != "openai":
            raise
        # A valid OpenAI draft must be able to replace malformed persisted
        # values; an empty comparison sentinel forces the bounded full update.
        original_validated = {}
    required_confirmation: str | None = None
    if configure_provider == "openai":
        required_confirmation = required_openai_plaintext_confirmation_fingerprint(
            draft
        )
        confirmation = draft.openai_plaintext_confirmation
        if required_confirmation is not None and (
            confirmation is None
            or confirmation.origin_fingerprint != required_confirmation
        ):
            raise GlobalSpeechTTSValidationError(
                "openai",
                "authentication_mode",
                "Confirm unauthenticated plaintext HTTP before saving.",
            )
    if validated == original_validated:
        settings: dict[str, object] = {}
        delete_setting_keys: list[str] = []
        changed_provider_ids: tuple[str, ...] = ()
    else:
        settings = _provider_event_settings(configure_provider, validated)
        delete_setting_keys = (
            ["CHATTERBOX_RANDOM_SEED"]
            if configure_provider == "chatterbox"
            and original_validated.get("random_seed") != ""
            and validated.get("random_seed") == ""
            else []
        )
        changed_provider_ids = (configure_provider,)
    if configure_provider == "openai":
        original_confirmation = original.openai_plaintext_confirmation
        draft_confirmation = draft.openai_plaintext_confirmation
        if required_confirmation is None:
            if (
                original_confirmation is not None
                or original.openai_plaintext_confirmation_cleanup_needed
                or draft.openai_plaintext_confirmation_cleanup_needed
            ):
                delete_setting_keys.append("OPENAI_NONE_HTTP_CONFIRMATION")
        elif (
            draft_confirmation is not None
            and draft_confirmation != original_confirmation
        ):
            settings["OPENAI_NONE_HTTP_CONFIRMATION"] = (
                draft_confirmation.origin_fingerprint
            )
    if (
        original.openai_plaintext_confirmation_cleanup_needed
        or draft.openai_plaintext_confirmation_cleanup_needed
    ) and "OPENAI_NONE_HTTP_CONFIRMATION" not in settings:
        if "OPENAI_NONE_HTTP_CONFIRMATION" not in delete_setting_keys:
            delete_setting_keys.append("OPENAI_NONE_HTTP_CONFIRMATION")
    # The saved default-profile pick is a distinct precedence rung above the
    # raw defaults axes (`preferences`, above) and is never part of that
    # snapshot, so it is diffed here independent of `configure_provider`.
    if draft.defaults.default_profile_id != original.defaults.default_profile_id:
        if draft.defaults.default_profile_id is None:
            delete_setting_keys.append("default_profile_id")
        else:
            settings["default_profile_id"] = draft.defaults.default_profile_id
    return GlobalSpeechTTSSaveProposal(
        settings=MappingProxyType(settings),
        delete_setting_keys=tuple(delete_setting_keys),
        preferences=preferences,
        changed_provider_ids=changed_provider_ids,
    )


def global_speech_tts_provider_configuration_changed(
    original: GlobalSpeechTTSState,
    draft: GlobalSpeechTTSState,
    *,
    provider_id: str,
) -> bool:
    """Validate and compare only one provider's adapter configuration."""

    if provider_id not in BUILT_IN_TTS_PROVIDER_ORDER:
        raise ValueError("Unknown built-in TTS provider")
    return _validated_provider_values(
        provider_id,
        draft.providers[provider_id],
    ) != _validated_provider_values(
        provider_id,
        original.providers[provider_id],
    )


def global_speech_tts_provider_configuration_state(
    state: GlobalSpeechTTSState,
    *,
    provider_id: str,
) -> SpeechTTSConfigurationState:
    """Project one provider adapter's saved/default validity and provenance."""

    if provider_id not in BUILT_IN_TTS_PROVIDER_ORDER:
        raise ValueError("Unknown built-in TTS provider")
    try:
        _validated_provider_values(provider_id, state.providers[provider_id])
    except GlobalSpeechTTSValidationError as error:
        if "required" in str(error).lower():
            return SpeechTTSConfigurationState.INCOMPLETE
        return SpeechTTSConfigurationState.INVALID
    credential = state.credentials.get(provider_id)
    credential_required = not (
        provider_id == "openai"
        and state.providers["openai"].get("authentication_mode")
        == OpenAIAuthenticationMode.NONE.value
    )
    if (
        credential_required
        and credential is not None
        and credential.source is CredentialSource.MISSING
    ):
        return SpeechTTSConfigurationState.INCOMPLETE
    source = state.provider_sources.get(
        provider_id,
        GlobalSpeechTTSEffectiveSource.DEFAULT,
    )
    if source is GlobalSpeechTTSEffectiveSource.DEFAULT:
        return SpeechTTSConfigurationState.DEFAULT
    if source is GlobalSpeechTTSEffectiveSource.INHERITED:
        return SpeechTTSConfigurationState.INHERITED
    return SpeechTTSConfigurationState.SAVED


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
        # default_profile_id is a distinct precedence rung above these raw
        # axes (see build_global_speech_tts_save_proposal) and is never part
        # of TTSPreferencesSnapshot, so "restore defaults" must not clear it.
        default_profile_id=state.defaults.default_profile_id,
    )
    environment_owned_values = {
        field_id: deepcopy(state.providers[configure_provider][field_id])
        for field_id, source in state.provider_field_sources.get(
            configure_provider, {}
        ).items()
        if source is GlobalSpeechTTSEffectiveSource.ENVIRONMENT
    }
    restored.providers[configure_provider] = deepcopy(
        _PROVIDER_NON_SECRET_DEFAULTS[configure_provider]
    )
    restored.providers[configure_provider].update(environment_owned_values)
    if configure_provider == "openai":
        restored.openai_plaintext_confirmation = None
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
