# stts_events.py
# Description: Event handlers for S/TT/S (Speech/Text-to-Speech) functionality
#
# Imports
import asyncio
from collections.abc import Callable, Coroutine, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, NamedTuple, Optional
from uuid import UUID

from loguru import logger
from rich.markup import escape

#
# Third-party imports
from textual.message import Message
from textual.widgets import Button, ProgressBar, RichLog, Static

#
# Local imports
from tldw_chatbook.config import get_runtime_config_snapshot
from tldw_chatbook.TTS import (
    OpenAISpeechRequest,
    STTSGeneratedAudio,
    STTSPlaygroundRequest,
    STTSPlaygroundResultProjection,
    TTSCloneReference,
    TTSPreferencesSnapshot,
    TTSRequest,
    TTSRequestedSelectionSnapshot,
    get_tts_service,
)
from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    TTSOperationError,
    TTSProgress,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_cpp_contract import validate_pcm16_wav
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    project_audio_cpp_settings_config,
)
from tldw_chatbook.TTS.effective_settings import TTSSelectionSource
from tldw_chatbook.TTS.legacy_bridge import (
    legacy_provider_config,
    openai_internal_model_id,
)
from tldw_chatbook.TTS.playground_types import (
    PROFILE_SAVE_BLOCK_PROVIDER_OPTIONS,
    ProfileSaveBlockCode,
)
from tldw_chatbook.TTS.TTS_Generation import (
    TTSDefaultActivationOutcome,
    TTSService,
    TTSSettingsPersistenceOutcome,
    TTSSettingsPublication,
    TTSSettingsPublicationLease,
    TTSSettingsPublicationTicket,
    _join_retained_task,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    ProcessProviderTestEvidenceStore,
    build_provider_test_fingerprint,
    load_global_speech_tts_state,
    process_provider_test_evidence_store,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import ProviderTestFingerprint
from tldw_chatbook.Utils.secure_temp_files import (
    create_secure_temp_file,
    secure_delete_file,
)

#
#######################################################################################################################
#
# Event Messages


class _SettingBinding(NamedTuple):
    destinations: tuple[tuple[str, str], ...]
    provider_id: str | None = None
    delete_destinations: tuple[tuple[str, str], ...] | None = None


def _app_tts_binding(
    key: str,
    provider_id: str | None = None,
) -> _SettingBinding:
    return _SettingBinding((("app_tts", key),), provider_id)


_TTS_SETTING_BINDINGS = {
    "audio_cpp": _SettingBinding(
        (("app_tts", "audio_cpp"),),
        "audio_cpp",
    ),
    "default_provider": _SettingBinding(
        (
            ("app_tts", "default_provider"),
            ("tts_settings", "default_tts_provider"),
        )
    ),
    "default_voice": _SettingBinding(
        (
            ("app_tts", "default_voice"),
            ("tts_settings", "default_tts_voice"),
        )
    ),
    "default_model": _SettingBinding(
        (
            ("app_tts", "default_model"),
            ("tts_settings", "default_openai_tts_model"),
        )
    ),
    "default_format": _SettingBinding(
        (
            ("app_tts", "default_format"),
            ("tts_settings", "default_openai_tts_output_format"),
        )
    ),
    "default_speed": _SettingBinding(
        (
            ("app_tts", "default_speed"),
            ("tts_settings", "default_openai_tts_speed"),
        )
    ),
    "default_profile_id": _app_tts_binding("default_profile_id"),
    "openai_api_key": _SettingBinding(
        (("api_settings.openai", "api_key"),),
        "openai",
        (
            ("api_settings.openai", "api_key"),
            ("openai_api", "api_key"),
            ("API", "openai_api_key"),
        ),
    ),
    "OPENAI_BASE_URL": _app_tts_binding("OPENAI_BASE_URL", "openai"),
    "OPENAI_AUTH_MODE": _app_tts_binding("OPENAI_AUTH_MODE", "openai"),
    "OPENAI_NONE_HTTP_CONFIRMATION": _app_tts_binding("OPENAI_NONE_HTTP_CONFIRMATION"),
    "OPENAI_ORG_ID": _app_tts_binding("OPENAI_ORG_ID", "openai"),
    "elevenlabs_api_key": _SettingBinding(
        (("api_settings.elevenlabs", "api_key"),),
        "elevenlabs",
        (
            ("api_settings.elevenlabs", "api_key"),
            ("elevenlabs_api", "api_key"),
            ("API", "elevenlabs_api_key"),
        ),
    ),
    "ELEVENLABS_DEFAULT_MODEL": _app_tts_binding("ELEVENLABS_DEFAULT_MODEL"),
    "ELEVENLABS_OUTPUT_FORMAT": _app_tts_binding(
        "ELEVENLABS_OUTPUT_FORMAT", "elevenlabs"
    ),
    "ELEVENLABS_VOICE_STABILITY": _app_tts_binding(
        "ELEVENLABS_VOICE_STABILITY", "elevenlabs"
    ),
    "ELEVENLABS_SIMILARITY_BOOST": _app_tts_binding(
        "ELEVENLABS_SIMILARITY_BOOST", "elevenlabs"
    ),
    "ELEVENLABS_STYLE": _app_tts_binding("ELEVENLABS_STYLE", "elevenlabs"),
    "ELEVENLABS_USE_SPEAKER_BOOST": _app_tts_binding(
        "ELEVENLABS_USE_SPEAKER_BOOST", "elevenlabs"
    ),
    "KOKORO_DEVICE_DEFAULT": _app_tts_binding("KOKORO_DEVICE_DEFAULT", "kokoro"),
    "KOKORO_USE_ONNX": _app_tts_binding("KOKORO_USE_ONNX", "kokoro"),
    "KOKORO_ONNX_MODEL_PATH_DEFAULT": _app_tts_binding(
        "KOKORO_ONNX_MODEL_PATH_DEFAULT", "kokoro"
    ),
    "KOKORO_ONNX_VOICES_JSON_DEFAULT": _app_tts_binding(
        "KOKORO_ONNX_VOICES_JSON_DEFAULT", "kokoro"
    ),
    "KOKORO_MAX_TOKENS": _app_tts_binding("KOKORO_MAX_TOKENS", "kokoro"),
    "KOKORO_ENABLE_VOICE_MIXING": _app_tts_binding(
        "KOKORO_ENABLE_VOICE_MIXING", "kokoro"
    ),
    "KOKORO_TRACK_PERFORMANCE": _app_tts_binding("KOKORO_TRACK_PERFORMANCE", "kokoro"),
    "CHATTERBOX_DEVICE": _app_tts_binding("CHATTERBOX_DEVICE", "chatterbox"),
    "CHATTERBOX_VOICE_DIR": _app_tts_binding("CHATTERBOX_VOICE_DIR", "chatterbox"),
    "CHATTERBOX_EXAGGERATION": _app_tts_binding("CHATTERBOX_EXAGGERATION"),
    "CHATTERBOX_CFG_WEIGHT": _app_tts_binding("CHATTERBOX_CFG_WEIGHT"),
    "CHATTERBOX_TEMPERATURE": _app_tts_binding("CHATTERBOX_TEMPERATURE", "chatterbox"),
    "CHATTERBOX_CHUNK_SIZE": _app_tts_binding("CHATTERBOX_CHUNK_SIZE", "chatterbox"),
    "CHATTERBOX_RANDOM_SEED": _app_tts_binding("CHATTERBOX_RANDOM_SEED", "chatterbox"),
    "CHATTERBOX_NUM_CANDIDATES": _app_tts_binding(
        "CHATTERBOX_NUM_CANDIDATES", "chatterbox"
    ),
    "CHATTERBOX_VALIDATE_WHISPER": _app_tts_binding(
        "CHATTERBOX_VALIDATE_WHISPER", "chatterbox"
    ),
    "CHATTERBOX_PREPROCESS_TEXT": _app_tts_binding(
        "CHATTERBOX_PREPROCESS_TEXT", "chatterbox"
    ),
    "CHATTERBOX_NORMALIZE_AUDIO": _app_tts_binding(
        "CHATTERBOX_NORMALIZE_AUDIO", "chatterbox"
    ),
    "CHATTERBOX_TARGET_DB": _app_tts_binding("CHATTERBOX_TARGET_DB", "chatterbox"),
    "CHATTERBOX_MAX_CHUNK_SIZE": _app_tts_binding(
        "CHATTERBOX_MAX_CHUNK_SIZE", "chatterbox"
    ),
    "CHATTERBOX_STREAMING": _app_tts_binding("CHATTERBOX_STREAMING", "chatterbox"),
    "CHATTERBOX_STREAM_CHUNK_SIZE": _app_tts_binding(
        "CHATTERBOX_STREAM_CHUNK_SIZE", "chatterbox"
    ),
    "CHATTERBOX_ENABLE_CROSSFADE": _app_tts_binding(
        "CHATTERBOX_ENABLE_CROSSFADE", "chatterbox"
    ),
    "CHATTERBOX_CROSSFADE_MS": _app_tts_binding(
        "CHATTERBOX_CROSSFADE_MS", "chatterbox"
    ),
    "HIGGS_MODEL_PATH": _SettingBinding(
        (("HiggsSettings", "model_path"),),
        "higgs",
    ),
    "HIGGS_VOICE_SAMPLES_DIR": _SettingBinding(
        (("HiggsSettings", "voice_samples_dir"),),
        "higgs",
    ),
    "HIGGS_DEVICE": _SettingBinding(
        (("HiggsSettings", "device"),),
        "higgs",
    ),
    "HIGGS_ENABLE_FLASH_ATTN": _SettingBinding(
        (("HiggsSettings", "enable_flash_attn"),),
        "higgs",
    ),
    "HIGGS_DTYPE": _SettingBinding(
        (("HiggsSettings", "dtype"),),
        "higgs",
    ),
    "HIGGS_MAX_REFERENCE_DURATION": _SettingBinding(
        (("HiggsSettings", "max_reference_duration"),),
        "higgs",
    ),
    "HIGGS_DEFAULT_LANGUAGE": _SettingBinding(
        (("HiggsSettings", "default_language"),),
        "higgs",
    ),
    "HIGGS_ENABLE_VOICE_CLONING": _SettingBinding(
        (("HiggsSettings", "enable_voice_cloning"),),
        "higgs",
    ),
    "HIGGS_ENABLE_MULTI_SPEAKER": _SettingBinding(
        (("HiggsSettings", "enable_multi_speaker"),),
        "higgs",
    ),
    "HIGGS_SPEAKER_DELIMITER": _SettingBinding(
        (("HiggsSettings", "speaker_delimiter"),),
        "higgs",
    ),
    "HIGGS_TRACK_PERFORMANCE": _SettingBinding(
        (("HiggsSettings", "track_performance"),),
        "higgs",
    ),
    "HIGGS_MAX_NEW_TOKENS": _SettingBinding(
        (("HiggsSettings", "max_new_tokens"),),
        "higgs",
    ),
    "HIGGS_TEMPERATURE": _SettingBinding(
        (("HiggsSettings", "temperature"),),
        "higgs",
    ),
    "HIGGS_TOP_P": _SettingBinding(
        (("HiggsSettings", "top_p"),),
        "higgs",
    ),
    "HIGGS_REPETITION_PENALTY": _SettingBinding(
        (("HiggsSettings", "repetition_penalty"),),
        "higgs",
    ),
    "ALLTALK_TTS_URL_DEFAULT": _app_tts_binding("ALLTALK_TTS_URL_DEFAULT", "alltalk"),
    "ALLTALK_TTS_VOICE_DEFAULT": _app_tts_binding("ALLTALK_TTS_VOICE_DEFAULT"),
    "ALLTALK_TTS_LANGUAGE_DEFAULT": _app_tts_binding(
        "ALLTALK_TTS_LANGUAGE_DEFAULT", "alltalk"
    ),
    "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT": _app_tts_binding(
        "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT"
    ),
}
_TTS_PROVIDER_ORDER = (
    "audio_cpp",
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)
_CREDENTIAL_CONFIG_TARGETS = {
    "openai": frozenset(
        {
            ("api_settings.openai", "api_key"),
            ("openai_api", "api_key"),
            ("API", "openai_api_key"),
        }
    ),
    "elevenlabs": frozenset(
        {
            ("api_settings.elevenlabs", "api_key"),
            ("elevenlabs_api", "api_key"),
            ("API", "elevenlabs_api_key"),
        }
    ),
}

_RECOVERY_ACTION_COPY = {
    "check_server": "Check the configured audio.cpp server and retry.",
    "configure_server": "Open STTS Settings and configure the audio.cpp server.",
    "edit_request": "Adjust the text or selected options and retry.",
    "refresh_models": "Refresh models in the STTS Playground and retry.",
    "retry": "Retry from the STTS Playground.",
}


def _effective_provider_config(
    provider_id: str,
    effective_settings: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project one provider's effective registry configuration."""
    if provider_id == "audio_cpp":
        return project_audio_cpp_settings_config(effective_settings).to_mapping()
    return legacy_provider_config(provider_id, effective_settings)


def _merge_section_mutations(
    destination: dict[str, dict[str, Any]],
    additions: Mapping[str, Mapping[str, object]],
) -> None:
    """Merge copied section values, with later additions authoritative."""
    for section, values in additions.items():
        destination.setdefault(section, {}).update(deepcopy(dict(values)))


def _prospective_effective_settings(
    current_settings: Mapping[str, Any],
    section_values: Mapping[str, Mapping[str, object]],
    delete_keys: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Project the in-memory effective settings after one proposed mutation."""
    prospective = deepcopy(dict(current_settings))
    current_raw = current_settings.get("COMPREHENSIVE_CONFIG_RAW", {})
    raw = deepcopy(dict(current_raw)) if isinstance(current_raw, Mapping) else {}

    for section, keys in delete_keys.items():
        current: Any = raw
        for part in section.split("."):
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if not isinstance(current, dict):
            continue
        for key in keys:
            current.pop(key, None)

    for section, values in section_values.items():
        current = raw
        parts = section.split(".")
        for part in parts:
            nested = current.get(part)
            if not isinstance(nested, dict):
                nested = {}
                current[part] = nested
            current = nested
        current.update(deepcopy(dict(values)))

    prospective["COMPREHENSIVE_CONFIG_RAW"] = raw
    touched_targets = {
        (section, key) for section, values in section_values.items() for key in values
    } | {(section, key) for section, keys in delete_keys.items() for key in keys}
    if any(section.startswith("api_settings.") for section, _key in touched_targets):
        raw_api_settings = raw.get("api_settings")
        prospective["api_settings"] = (
            deepcopy(dict(raw_api_settings))
            if isinstance(raw_api_settings, Mapping)
            else {}
        )
    for provider_id, credential_targets in _CREDENTIAL_CONFIG_TARGETS.items():
        if touched_targets & credential_targets:
            prospective.pop(f"{provider_id}_api", None)
    raw_app_tts = raw.get("app_tts")
    if isinstance(raw_app_tts, Mapping):
        normalized_app_tts = prospective.get("APP_TTS_CONFIG", {})
        merged_app_tts = (
            deepcopy(dict(normalized_app_tts))
            if isinstance(normalized_app_tts, Mapping)
            else {}
        )
        merged_app_tts.update(deepcopy(dict(raw_app_tts)))
        prospective["APP_TTS_CONFIG"] = merged_app_tts
    return prospective


class STTSPlaygroundGenerateEvent(Message):
    """Event carrying one immutable Playground generation snapshot."""

    def __init__(self, request: STTSPlaygroundRequest) -> None:
        super().__init__()
        if not isinstance(request, STTSPlaygroundRequest):
            raise TypeError("request must be an STTSPlaygroundRequest")
        self.request = request


class STTSSettingsSaveEvent(Message):
    """Event when TTS settings are saved"""

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        preferences: TTSPreferencesSnapshot | None = None,
        delete_setting_keys: tuple[str, ...] | list[str] = (),
        request_id: int | None = None,
        reply_to: object | None = None,
        commit_defaults_after_handoff: bool = False,
        publication_lease: TTSSettingsPublicationLease | None = None,
    ) -> None:
        super().__init__()
        if request_id is not None:
            if type(request_id) is not int:
                raise TypeError("TTS settings request ID must be an integer")
            if request_id < 0:
                raise ValueError("TTS settings request ID must be nonnegative")
        if type(commit_defaults_after_handoff) is not bool:
            raise TypeError("TTS default activation intent must be boolean")
        if commit_defaults_after_handoff and preferences is None:
            raise ValueError("TTS default activation requires preferences")
        copied_deletes = tuple(delete_setting_keys)
        if not all(isinstance(key, str) and key for key in copied_deletes):
            raise ValueError("TTS setting delete keys must be non-empty strings")
        if len(set(copied_deletes)) != len(copied_deletes):
            raise ValueError("TTS setting delete keys must be unique")
        self.settings = deepcopy(dict(settings))
        self.preferences = preferences
        self.delete_setting_keys = copied_deletes
        self.request_id = request_id
        self.reply_to = reply_to
        self.commit_defaults_after_handoff = commit_defaults_after_handoff
        self.publication_lease = publication_lease

    def _publication_started(self) -> None:
        """Drop event ownership after the service returns a retained ticket."""

        self.publication_lease = None

    def _abandon_publication_lease(self) -> None:
        """Release the transfer only when no service adopted it."""

        publication_lease = self.publication_lease
        if publication_lease is None:
            return
        publication_lease.abandon()
        self.publication_lease = None


@dataclass(frozen=True, slots=True)
class STTSSettingsSaveResult:
    """Safe requester result separating persistence from runtime handoff."""

    request_id: int
    persisted: bool
    provider_statuses: Mapping[str, str]
    failure_phase: str | None = None
    provider_configuration_revisions: Mapping[str, int] = field(default_factory=dict)
    provider_runtime_revisions: Mapping[str, int] = field(default_factory=dict)
    staged_provider_ids: frozenset[str] = frozenset()
    defaults_activated: bool | None = None
    defaults_activation_status: str | None = None

    def __post_init__(self) -> None:
        if type(self.request_id) is not int or self.request_id < 0:
            raise ValueError("TTS settings result request ID is invalid")
        if type(self.persisted) is not bool:
            raise TypeError("TTS settings persistence result must be boolean")
        if (
            self.defaults_activated is not None
            and type(self.defaults_activated) is not bool
        ):
            raise TypeError("TTS default activation result must be boolean")
        if self.defaults_activation_status not in {
            None,
            "activation_not_ready",
            "committed",
            "rolled_back",
            "rollback_failed",
        }:
            raise ValueError("TTS default activation status is invalid")
        if self.defaults_activation_status is not None and (
            self.defaults_activated
            is not (self.defaults_activation_status == "committed")
        ):
            raise ValueError("TTS default activation result is inconsistent")
        allowed_statuses = frozenset(
            {"applied", "unchanged", "pending", "superseded", "unavailable"}
        )
        copied_statuses: dict[str, str] = {}
        for provider_id, status in self.provider_statuses.items():
            if provider_id not in _TTS_PROVIDER_ORDER or status not in allowed_statuses:
                raise ValueError("TTS settings provider result is invalid")
            copied_statuses[provider_id] = status
        copied_configuration_revisions: dict[str, int] = {}
        for provider_id, revision in self.provider_configuration_revisions.items():
            if provider_id not in _TTS_PROVIDER_ORDER:
                raise ValueError("TTS saved provider revision is invalid")
            if type(revision) is not int or revision < 0:
                raise ValueError("TTS saved provider revision is invalid")
            copied_configuration_revisions[provider_id] = revision
        copied_runtime_revisions: dict[str, int] = {}
        for provider_id, revision in self.provider_runtime_revisions.items():
            if provider_id not in _TTS_PROVIDER_ORDER:
                raise ValueError("TTS runtime provider revision is invalid")
            if type(revision) is not int or revision < 0:
                raise ValueError("TTS runtime provider revision is invalid")
            copied_runtime_revisions[provider_id] = revision
        if self.persisted and not set(copied_statuses).issubset(
            copied_configuration_revisions
        ):
            raise ValueError("Persisted TTS provider results require saved revisions")
        if self.failure_phase not in {None, "before_replace", "cache_reload"}:
            raise ValueError("TTS settings failure phase is invalid")
        staged_provider_ids = frozenset(self.staged_provider_ids)
        if any(
            copied_statuses.get(provider_id) != "pending"
            for provider_id in staged_provider_ids
        ):
            raise ValueError("Staged TTS providers require pending results")
        object.__setattr__(
            self,
            "provider_statuses",
            MappingProxyType(copied_statuses),
        )
        object.__setattr__(
            self,
            "provider_configuration_revisions",
            MappingProxyType(copied_configuration_revisions),
        )
        object.__setattr__(
            self,
            "provider_runtime_revisions",
            MappingProxyType(copied_runtime_revisions),
        )
        object.__setattr__(self, "staged_provider_ids", staged_provider_ids)


class STTSProviderConfigurationChanged(Message):
    """Signal that one provider's effective configuration revision changed."""

    def __init__(
        self,
        provider_id: str,
        configuration_revision: int,
        global_preferences_revision: int | None = None,
    ) -> None:
        super().__init__()
        self.provider_id = provider_id
        self.configuration_revision = configuration_revision
        self.global_preferences_revision = global_preferences_revision


@dataclass(frozen=True, slots=True)
class _STTSPlaygroundState:
    """Read-only handler-owned Playground lifecycle snapshot."""

    active_operation_id: str | None
    artifact: STTSPlaygroundResultProjection | None
    generation_active: bool


@dataclass(frozen=True, slots=True)
class _SampleEvidenceCandidate:
    """Saved provider identity captured before one synthesis operation."""

    fingerprint: ProviderTestFingerprint
    runtime_revision: int
    saved_selection: TTSPreferencesSnapshot | None


@dataclass(frozen=True, slots=True)
class _SampleGenerationFacts:
    """Publication/runtime identities and effective-source eligibility."""

    runtime_revision: int
    certifies_saved_configuration: bool
    saved_publication_revision: int | None = None
    saved_selection: TTSPreferencesSnapshot | None = None


@dataclass(frozen=True, slots=True)
class _DefaultActivationIntent:
    """One immutable, generation-fenced pending default activation."""

    preferences: TTSPreferencesSnapshot
    expected_saved_revision: int
    provider_id: str
    token: int


class STTSAudioBookGenerateEvent(Message):
    """Event when audiobook generation is requested"""

    def __init__(
        self,
        content: str,
        chapters: list,
        narrator_voice: str,
        output_format: str,
        options: Dict[str, Any],
    ):
        super().__init__()
        self.content = content
        self.chapters = chapters
        self.narrator_voice = narrator_voice
        self.output_format = output_format
        self.options = options


#######################################################################################################################
#
# Event Handler Mixin


class STTSEventHandler:
    """Event handler for S/TT/S functionality"""

    def __init__(self, app=None):
        self.app = app  # Reference to the main app
        self.provider_test_evidence = process_provider_test_evidence_store(
            app if app is not None else self
        )
        self._stts_service = None
        self._current_audio_file = None
        self._current_playground_artifact: STTSGeneratedAudio | None = None
        self._is_generating = False
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._generation_task: asyncio.Task[None] | None = None
        self._active_playground_operation_id: str | None = None
        self._retired_playground_operation_id: str | None = None
        self._playground_audio_files: set[Path] = set()
        self._playground_operation_files: dict[str, set[Path]] = {}
        self._playground_file_leases: dict[Path, int] = {}
        self._cleanup_task: asyncio.Task[None] | None = None
        self._settings_save_lock = asyncio.Lock()
        self._sample_generation_facts: dict[str, _SampleGenerationFacts] = {}
        self._next_default_activation_token = 1
        self._default_activation_intents: dict[str, _DefaultActivationIntent] = {}

    def _capture_sample_evidence_candidate(
        self,
        request: STTSPlaygroundRequest,
    ) -> _SampleEvidenceCandidate | None:
        """Freeze the saved provider identity before synthesis can begin."""

        service = self._stts_service
        saved_revision = getattr(service, "saved_configuration_revision", None)
        applied_revision = getattr(service, "applied_configuration_revision", None)
        configuration_revision = getattr(service, "configuration_revision", None)
        if (
            not callable(saved_revision)
            or not callable(applied_revision)
            or not callable(configuration_revision)
        ):
            return None
        try:
            saved = saved_revision(request.provider_id)
            applied = applied_revision(request.provider_id)
            runtime = configuration_revision(request.provider_id)
            if (
                type(saved) is not int
                or type(applied) is not int
                or type(runtime) is not int
                or saved < 0
                or applied != saved
                or runtime < 0
            ):
                return None
            values = get_runtime_config_snapshot().values
            state = load_global_speech_tts_state(
                values if isinstance(values, Mapping) else {}
            )
            preferences_snapshot = getattr(service, "preferences_snapshot", None)
            saved_selection = (
                preferences_snapshot() if callable(preferences_snapshot) else None
            )
            if type(saved_selection) is not TTSPreferencesSnapshot:
                saved_selection = None
            if (
                saved_selection is not None
                and saved_selection.provider_id != request.provider_id
            ):
                return None
            return _SampleEvidenceCandidate(
                fingerprint=build_provider_test_fingerprint(
                    state,
                    provider_id=request.provider_id,
                    saved_revision=saved,
                ),
                runtime_revision=runtime,
                saved_selection=saved_selection,
            )
        except Exception:  # noqa: BLE001 - evidence cannot block synthesis
            return None

    @staticmethod
    def _legacy_sample_generation_facts(
        request: STTSPlaygroundRequest,
        candidate: _SampleEvidenceCandidate | None,
    ) -> _SampleGenerationFacts | None:
        """Bind legacy evidence to the exact saved selection before synthesis."""

        if candidate is None:
            return None
        saved = candidate.saved_selection
        certifies = bool(
            saved is not None
            and saved.provider_id == request.provider_id
            and saved.model_mode == "exact"
            and saved.model_id == request.model_id
            and saved.voice_mode == "exact"
            and saved.voice_id == request.voice_id
            and saved.response_format == request.response_format.lower()
            and saved.speed == request.speed
            and not request.options
            and request.clone_audition is None
            and request.profile_preview is None
            and request.studio_draft is None
            and request.studio_preferences is None
        )
        return _SampleGenerationFacts(
            runtime_revision=candidate.runtime_revision,
            certifies_saved_configuration=certifies,
            saved_publication_revision=candidate.fingerprint.saved_revision,
            saved_selection=saved,
        )

    @staticmethod
    def _requested_selection_matches_sample_request(
        selection: TTSRequestedSelectionSnapshot | None,
        request: STTSPlaygroundRequest,
    ) -> bool:
        """Match admitted/effective provenance to one immutable sample request."""

        return bool(
            type(selection) is TTSRequestedSelectionSnapshot
            and selection.provider_id == request.provider_id
            and selection.model_id == request.model_id
            and selection.voice_id == request.voice_id
            and selection.response_format == request.response_format.lower()
            and selection.speed == request.speed
            and dict(selection.options) == dict(request.options)
        )

    def _record_successful_sample_evidence(
        self,
        artifact: STTSGeneratedAudio,
        candidate: _SampleEvidenceCandidate | None,
        facts: _SampleGenerationFacts | None,
    ) -> bool:
        """Record only a bounded, validated artifact for the saved provider."""

        if candidate is None or facts is None:
            return False
        fingerprint = candidate.fingerprint
        if (
            not facts.certifies_saved_configuration
            or artifact.provider_id != fingerprint.provider_id
            or facts.runtime_revision != candidate.runtime_revision
            or (
                facts.saved_publication_revision is not None
                and facts.saved_publication_revision != fingerprint.saved_revision
            )
        ):
            return False
        service = self._stts_service
        saved_revision = getattr(service, "saved_configuration_revision", None)
        applied_revision = getattr(service, "applied_configuration_revision", None)
        configuration_revision = getattr(service, "configuration_revision", None)
        if (
            not callable(saved_revision)
            or not callable(applied_revision)
            or not callable(configuration_revision)
        ):
            return False
        try:
            if (
                saved_revision(artifact.provider_id) != fingerprint.saved_revision
                or applied_revision(artifact.provider_id) != fingerprint.saved_revision
                or configuration_revision(artifact.provider_id)
                != candidate.runtime_revision
            ):
                return False
            if facts.saved_selection is not None:
                preferences_snapshot = getattr(service, "preferences_snapshot", None)
                if not callable(preferences_snapshot):
                    return False
                current_saved_selection = preferences_snapshot()
                if (
                    type(current_saved_selection) is not TTSPreferencesSnapshot
                    or current_saved_selection != facts.saved_selection
                ):
                    return False
            max_bytes = ProcessProviderTestEvidenceStore._DEFAULT_MAX_SAMPLE_BYTES
            with artifact.path.open("rb") as source:
                body = source.read(max_bytes + 1)
            sample_rate_hz = artifact.metadata.get("sample_rate")
            channels = artifact.metadata.get("channels")
            sample_width_bytes = artifact.metadata.get("sample_width_bytes")
            if sample_width_bytes is None:
                bits_per_sample = artifact.metadata.get("bits_per_sample")
                if type(bits_per_sample) is int and bits_per_sample % 8 == 0:
                    sample_width_bytes = bits_per_sample // 8
            return self.provider_test_evidence.record_successful_sample(
                fingerprint,
                status_code=200,
                response_format=artifact.audio_format,
                content_type=artifact.content_type,
                body=body,
                max_bytes=max_bytes,
                sample_rate_hz=(
                    sample_rate_hz if type(sample_rate_hz) is int else None
                ),
                channels=channels if type(channels) is int else None,
                sample_width_bytes=(
                    sample_width_bytes if type(sample_width_bytes) is int else None
                ),
            )
        except Exception:  # noqa: BLE001 - evidence must not fail delivered audio
            logger.warning(
                "TTS sample evidence was not accepted (provider={}).",
                artifact.provider_id,
            )
            return False

    async def initialize_stts(self) -> None:
        """Initialize S/TT/S service"""
        try:
            self._stts_service = await get_tts_service()
            logger.info("S/TT/S service initialized successfully")
        except Exception:
            logger.error("Failed to initialize S/TT/S service")
            self._stts_service = None

    def playground_state(self) -> _STTSPlaygroundState:
        """Return immutable handler-owned generation and artifact state."""
        artifact = self._current_playground_artifact
        return _STTSPlaygroundState(
            active_operation_id=self._active_playground_operation_id,
            artifact=(
                STTSPlaygroundResultProjection.from_artifact(artifact)
                if artifact is not None
                else None
            ),
            generation_active=self._is_generating,
        )

    def retire_playground_context(self) -> None:
        """Discard audio and fence completion from the current Playground context."""

        artifact = self._current_playground_artifact
        if artifact is not None:
            self._current_playground_artifact = None
            if self._current_audio_file == artifact.path:
                self._current_audio_file = None
            self._delete_operation_files(artifact.operation_id)

        operation_id = self._active_playground_operation_id
        if operation_id is not None:
            self._retired_playground_operation_id = operation_id
        task = self._generation_task
        if task is not None and not task.done():
            task.cancel()

    def retire_playground_generation(
        self,
        expected_operation_id: str | None = None,
    ) -> None:
        """Fence only in-flight generation while preserving completed audio."""

        operation_id = expected_operation_id or self._active_playground_operation_id
        if operation_id is None:
            return
        self._retired_playground_operation_id = operation_id
        if self._active_playground_operation_id != operation_id:
            return
        task = self._generation_task
        if task is not None and not task.done():
            task.cancel()

    def start_playground_generation(
        self,
        event: STTSPlaygroundGenerateEvent,
    ) -> None:
        """Start and retain exactly one handler-owned Playground task."""
        if self._cleanup_task is not None:
            logger.debug("Ignoring TTS generation after STTS cleanup started")
            return
        if self._retired_playground_operation_id == event.request.operation_id:
            self._retired_playground_operation_id = None
            return
        if self._generation_task is not None and not self._generation_task.done():
            self.app.notify("TTS generation already in progress", severity="warning")
            return
        if self._is_generating:
            self.app.notify("TTS generation already in progress", severity="warning")
            return

        task = asyncio.create_task(
            self.handle_playground_generate(event),
            name=f"stts_playground_{event.request.operation_id}",
        )
        self._generation_task = task
        self._active_tasks.add(task)
        task.add_done_callback(self._playground_generation_done)

    def _playground_generation_done(self, task: asyncio.Task[None]) -> None:
        self._active_tasks.discard(task)
        if self._generation_task is task:
            self._generation_task = None
        try:
            task.exception()
        except BaseException:
            pass

    def _track_operation_file(self, operation_id: str, path: Path) -> None:
        path = Path(path)
        self._playground_audio_files.add(path)
        self._playground_operation_files.setdefault(operation_id, set()).add(path)

    def _forget_operation_file(self, operation_id: str, path: Path) -> None:
        path = Path(path)
        self._playground_audio_files.discard(path)
        operation_files = self._playground_operation_files.get(operation_id)
        if operation_files is None:
            return
        operation_files.discard(path)
        if not operation_files:
            self._playground_operation_files.pop(operation_id, None)

    def _delete_operation_files(
        self,
        operation_id: str,
        *,
        keep: frozenset[Path] = frozenset(),
    ) -> None:
        for path in tuple(self._playground_operation_files.get(operation_id, ())):
            if path in keep:
                continue
            if self._playground_file_leases.get(path, 0) > 0:
                continue
            if secure_delete_file(path) or not path.exists():
                self._forget_operation_file(operation_id, path)

    def lease_playground_artifact(self, artifact: STTSGeneratedAudio) -> bool:
        """Pin a handler-owned artifact across a deferred UI action."""
        path = Path(artifact.path)
        operation_files = self._playground_operation_files.get(
            artifact.operation_id,
            set(),
        )
        if (
            path not in self._playground_audio_files
            or path not in operation_files
            or not path.exists()
        ):
            return False
        self._playground_file_leases[path] = (
            self._playground_file_leases.get(path, 0) + 1
        )
        return True

    def release_playground_artifact(self, artifact: STTSGeneratedAudio) -> None:
        """Release one lease and retire the artifact when it is no longer current."""
        path = Path(artifact.path)
        count = self._playground_file_leases.get(path, 0)
        if count <= 0:
            return
        if count == 1:
            self._playground_file_leases.pop(path, None)
        else:
            self._playground_file_leases[path] = count - 1
            return

        current_path = (
            self._current_playground_artifact.path
            if self._current_playground_artifact is not None
            else None
        )
        if current_path == path:
            return
        for operation_id, operation_files in tuple(
            self._playground_operation_files.items()
        ):
            if path in operation_files:
                self._delete_operation_files(operation_id)

    def lease_playground_result(self, operation_id: str, path: Path) -> bool:
        """Lease the exact current handler artifact by sanitized identity."""

        artifact = self._current_playground_artifact
        if (
            artifact is None
            or artifact.operation_id != operation_id
            or artifact.path != Path(path)
        ):
            return False
        return self.lease_playground_artifact(artifact)

    def release_playground_result(self, operation_id: str, path: Path) -> None:
        """Release a result lease without publishing the private artifact."""

        artifact = self._current_playground_artifact
        if (
            artifact is not None
            and artifact.operation_id == operation_id
            and artifact.path == Path(path)
        ):
            self.release_playground_artifact(artifact)
            return
        count = self._playground_file_leases.get(Path(path), 0)
        if count <= 0:
            return
        if count == 1:
            self._playground_file_leases.pop(Path(path), None)
        else:
            self._playground_file_leases[Path(path)] = count - 1
            return
        self._delete_operation_files(operation_id)

    async def save_current_playground_profile(
        self,
        operation_id: str,
        display_name: str,
        profile_service: object,
    ) -> object:
        """Save the exact handler-owned result without exposing its artifact."""

        artifact = self._current_playground_artifact
        if artifact is None or artifact.operation_id != operation_id:
            raise RuntimeError("The current speech result changed")
        if artifact.clone_evidence is not None:
            create = getattr(profile_service, "create_clone_from_artifact", None)
        else:
            create = getattr(profile_service, "create_from_artifact", None)
        if not callable(create):
            raise RuntimeError("The voice profile store is unavailable")
        return await create(display_name, artifact)

    def _accept_playground_artifact(self, artifact: STTSGeneratedAudio) -> None:
        """Store the new artifact before securely retiring older files."""
        self._current_playground_artifact = artifact
        self._current_audio_file = artifact.path
        self._track_operation_file(artifact.operation_id, artifact.path)
        for operation_id in tuple(self._playground_operation_files):
            self._delete_operation_files(
                operation_id,
                keep=(
                    frozenset({artifact.path})
                    if operation_id == artifact.operation_id
                    else frozenset()
                ),
            )

    async def _generate_audio_cpp(
        self,
        snapshot: STTSPlaygroundRequest,
        progress_sink: ProgressSink | None,
    ) -> STTSGeneratedAudio:
        """Generate one complete native audio.cpp WAV response."""
        if self._stts_service is None:
            raise RuntimeError("TTS service is not initialized")

        request = TTSRequest(
            provider_id="audio_cpp",
            model_id=snapshot.model_id,
            text=snapshot.text,
            voice=snapshot.voice_id,
            response_format="wav",
            speed=1.0,
            options={},
        )
        response = None
        requested_selection = None
        primary_error: BaseException | None = None
        try:
            response, requested_selection = await self._stts_service.synthesize_exact(
                request,
                progress_sink,
            )
            chunks = [chunk async for chunk in response.byte_stream]
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if response is not None:
                try:
                    await response.aclose()
                except BaseException:
                    if primary_error is None:
                        raise
                    logger.warning(
                        "Failed to close audio.cpp response after {}",
                        type(primary_error).__name__,
                    )

        path = Path(
            create_secure_temp_file(
                b"".join(chunks),
                suffix=f".{response.audio_format.removeprefix('.')}",
                prefix="stts_playground_",
            )
        )
        self._track_operation_file(snapshot.operation_id, path)
        artifact_metadata = dict(response.metadata)
        if type(response.sample_rate) is int and response.sample_rate > 0:
            artifact_metadata["sample_rate"] = response.sample_rate
        try:
            artifact = STTSGeneratedAudio(
                path=path,
                provider_id=response.provider_id,
                model_id=response.model_id,
                voice_id=snapshot.voice_id,
                source_text=snapshot.text,
                operation_id=snapshot.operation_id,
                audio_format=response.audio_format,
                content_type=response.content_type,
                metadata=artifact_metadata,
                requested_selection=requested_selection,
            )
            return artifact
        except BaseException:
            if secure_delete_file(path) or not path.exists():
                self._forget_operation_file(snapshot.operation_id, path)
            raise

    def _build_requested_selection(
        self,
        *,
        provider_id: str,
        model_id: str,
        voice_id: str | None,
        response_format: str,
        speed: float,
        options: Mapping[str, Any],
        configuration_revision: Callable[[], int],
    ) -> tuple[TTSRequestedSelectionSnapshot | None, ProfileSaveBlockCode | None]:
        """Build one save-eligible provenance snapshot, degrading to `None`.

        Shared by every Playground generation path -- native audio_cpp (via
        Studio-effective) and every legacy-bridge provider (via both
        Studio-effective and the standalone legacy bridge). Reading the
        configuration revision and constructing the snapshot both happen
        inside the same guard: either can fail on hostile or momentarily
        unreadable state (a registry read error, an inconsistent effective
        selection), and neither may fail the generation itself -- the caller
        already has real audio by the time this runs. A failure here only
        costs "Save result as profile" eligibility.

        The real `options` are passed rather than hardcoded `{}`:
        `TTSRequestedSelectionSnapshot` requires empty options precisely so a
        generation that used options cannot masquerade as exact provenance,
        and hardcoding here meant that guard could never fire -- so
        Higgs/ElevenLabs/Chatterbox/Kokoro results (whose Inputs always
        populate provider options) would have saved a profile that does not
        reproduce what the user heard. Returns the reason alongside the
        snapshot so the surface can say why Save is unavailable instead of
        quietly dropping it.
        """
        try:
            return (
                TTSRequestedSelectionSnapshot(
                    provider_id=provider_id,
                    model_id=model_id,
                    voice_id=voice_id,
                    response_format=response_format,
                    speed=speed,
                    options=options,
                    configuration_revision=configuration_revision(),
                ),
                None,
            )
        except Exception:  # noqa: BLE001 - best-effort provenance only
            logger.warning(
                "Playground result is not profile-save eligible (provider={}).",
                provider_id,
            )
            return None, self._profile_save_block_code(options)

    @staticmethod
    def _profile_save_block_code(
        options: Mapping[str, Any],
    ) -> ProfileSaveBlockCode | None:
        """Name the one refusal reason a user can act on, or stay silent."""

        try:
            used_options = bool(options)
        except Exception:  # noqa: BLE001 - hostile options explain nothing
            return None
        return PROFILE_SAVE_BLOCK_PROVIDER_OPTIONS if used_options else None

    async def _generate_legacy(
        self,
        snapshot: STTSPlaygroundRequest,
        progress_sink: ProgressSink | None,
    ) -> STTSGeneratedAudio:
        """Retain the existing stream-and-convert path for legacy providers."""
        if self._stts_service is None:
            raise RuntimeError("TTS service is not initialized")

        requested_format = snapshot.response_format.lower()
        if requested_format not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
            requested_format = "mp3"
        request = OpenAISpeechRequest(
            model=snapshot.model_id,
            input=snapshot.text,
            voice=snapshot.voice_id or "default",
            response_format="wav",
            speed=snapshot.speed,
        )
        options = dict(snapshot.options)
        if snapshot.provider_id in {"chatterbox", "higgs"} and options:
            request.extra_params = options

        internal_model_id = self._legacy_internal_model_id(snapshot, options)
        created_paths: set[Path] = set()
        try:
            chunks = [
                chunk
                async for chunk in self._stts_service.generate_audio_stream(
                    request,
                    internal_model_id,
                    progress_sink=progress_sink,
                )
            ]
            wav_file = Path(
                create_secure_temp_file(
                    b"".join(chunks),
                    suffix=".wav",
                    prefix="stts_playground_",
                )
            )
            created_paths.add(wav_file)
            self._track_operation_file(snapshot.operation_id, wav_file)
            output_file = wav_file
            audio_format = "wav"

            if requested_format != "wav":
                conversion_destination = wav_file.with_suffix(f".{requested_format}")
                created_paths.add(conversion_destination)
                self._track_operation_file(
                    snapshot.operation_id,
                    conversion_destination,
                )
                converted_file = await self._convert_audio_format(
                    wav_file,
                    requested_format,
                )
                if converted_file is not None:
                    output_file = Path(converted_file)
                    created_paths.add(output_file)
                    self._track_operation_file(snapshot.operation_id, output_file)
                    audio_format = requested_format
                    if secure_delete_file(wav_file) or not wav_file.exists():
                        self._forget_operation_file(
                            snapshot.operation_id,
                            wav_file,
                        )
                        created_paths.discard(wav_file)
                elif (
                    secure_delete_file(conversion_destination)
                    or not conversion_destination.exists()
                ):
                    self._forget_operation_file(
                        snapshot.operation_id,
                        conversion_destination,
                    )
                    created_paths.discard(conversion_destination)

            requested_selection, profile_save_block_code = (
                self._build_requested_selection(
                    provider_id=snapshot.provider_id,
                    model_id=snapshot.model_id,
                    voice_id=snapshot.voice_id or None,
                    response_format=audio_format,
                    speed=snapshot.speed,
                    options=options,
                    configuration_revision=lambda: (
                        self._stts_service.configuration_revision(snapshot.provider_id)
                    ),
                )
            )
            artifact = STTSGeneratedAudio(
                path=output_file,
                provider_id=snapshot.provider_id,
                model_id=snapshot.model_id,
                voice_id=snapshot.voice_id,
                source_text=snapshot.text,
                operation_id=snapshot.operation_id,
                audio_format=audio_format,
                content_type=self._audio_content_type(audio_format),
                metadata={},
                requested_selection=requested_selection,
                profile_save_block_code=profile_save_block_code,
            )
            return artifact
        except BaseException:
            for path in created_paths:
                if secure_delete_file(path) or not path.exists():
                    self._forget_operation_file(snapshot.operation_id, path)
            raise

    async def _generate_studio_effective(
        self,
        snapshot: STTSPlaygroundRequest,
        progress_sink: ProgressSink | None,
    ) -> STTSGeneratedAudio:
        """Generate from one revision-coherent Studio draft and saved snapshot."""

        if self._stts_service is None:
            raise RuntimeError("TTS service is not initialized")
        if snapshot.studio_draft is None or snapshot.studio_preferences is None:
            raise ValueError("Studio generation requires a complete Studio snapshot")

        response = None
        effective = None
        clone_evidence = None
        primary_error: BaseException | None = None
        profile_reference_resolver = None
        if snapshot.profile_preview is not None:

            async def resolve_profile_reference(
                profile_id: UUID,
                repository_generation: int,
                profile_revision: int,
            ) -> TTSCloneReference:
                loader = getattr(self.app, "_ensure_tts_profile_service", None)
                if not callable(loader):
                    raise RuntimeError("TTS profile service is unavailable")
                profile_service = await loader()
                if profile_service is None:
                    raise RuntimeError("TTS profile service is unavailable")
                loaded = await profile_service.get_profile(profile_id)
                profile = loaded.profile
                if (
                    loaded.repository_generation != repository_generation
                    or profile.revision != profile_revision
                    or profile.provider_id != snapshot.provider_id
                    or profile.model_id != snapshot.model_id
                    or profile.reference is None
                ):
                    raise RuntimeError("TTS profile preview is stale")
                return await profile_service.get_reference(
                    profile_id,
                    expected_generation=repository_generation,
                    expected_revision=profile_revision,
                )

            profile_reference_resolver = resolve_profile_reference
        try:
            synthesize_with_evidence = getattr(
                self._stts_service,
                "synthesize_effective_with_evidence",
                None,
            )
            synthesis_kwargs = dict(
                text=snapshot.text,
                studio_draft=snapshot.studio_draft,
                studio_preferences=snapshot.studio_preferences,
                clone_audition=snapshot.clone_audition,
                profile_preview=snapshot.profile_preview,
                profile_reference_resolver=profile_reference_resolver,
                progress_sink=progress_sink,
            )
            if callable(synthesize_with_evidence):
                response, effective, clone_evidence = await synthesize_with_evidence(
                    **synthesis_kwargs
                )
            else:
                response, effective = await self._stts_service.synthesize_effective(
                    **synthesis_kwargs
                )
            chunks = [chunk async for chunk in response.byte_stream]
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if response is not None:
                try:
                    await response.aclose()
                except BaseException:
                    if primary_error is None:
                        raise
                    logger.warning(
                        "Failed to close Studio TTS response after {}",
                        type(primary_error).__name__,
                    )

        assert response is not None
        assert effective is not None
        complete_audio = b"".join(chunks)
        if clone_evidence is not None:
            validate_pcm16_wav(complete_audio)
        path = Path(
            create_secure_temp_file(
                complete_audio,
                suffix=f".{response.audio_format.removeprefix('.')}",
                prefix="stts_playground_",
            )
        )
        self._track_operation_file(snapshot.operation_id, path)
        requested_selection, profile_save_block_code = self._build_requested_selection(
            provider_id=effective.provider_id,
            model_id=effective.model_id,
            voice_id=effective.voice_id,
            response_format=effective.response_format,
            speed=effective.speed,
            options=effective.provider_options,
            configuration_revision=(lambda: effective.revisions.provider_configuration),
        )
        try:
            artifact = STTSGeneratedAudio(
                path=path,
                provider_id=effective.provider_id,
                model_id=effective.model_id,
                voice_id=effective.voice_id,
                source_text=snapshot.text,
                operation_id=snapshot.operation_id,
                audio_format=response.audio_format,
                content_type=response.content_type,
                metadata=response.metadata,
                requested_selection=requested_selection,
                profile_save_block_code=profile_save_block_code,
                clone_evidence=clone_evidence,
            )
            sources = tuple(getattr(effective, "sources", {}).values()) + tuple(
                getattr(effective, "provider_option_sources", {}).values()
            )
            self._sample_generation_facts[snapshot.operation_id] = (
                _SampleGenerationFacts(
                    runtime_revision=(effective.revisions.provider_configuration),
                    certifies_saved_configuration=bool(
                        not getattr(effective, "studio_preview", False)
                        and snapshot.profile_preview is None
                        and snapshot.clone_audition is None
                        and TTSSelectionSource.STUDIO_DRAFT not in sources
                        and self._requested_selection_matches_sample_request(
                            requested_selection,
                            snapshot,
                        )
                    ),
                )
            )
            return artifact
        except BaseException:
            if secure_delete_file(path) or not path.exists():
                self._forget_operation_file(snapshot.operation_id, path)
            raise

    @staticmethod
    def _legacy_internal_model_id(
        snapshot: STTSPlaygroundRequest,
        options: Mapping[str, Any],
    ) -> str:
        # Cross-reference (TASK-1393 pact convention, greppable both ways):
        # `TTS/legacy_request_builder.build_legacy_speech_request` is a
        # separate, deliberately different copy of this id-derivation logic.
        # This copy serves the live playground and derives kokoro's suffix
        # from `options["use_onnx"]` and alltalk's suffix from
        # `snapshot.model_id`; the builder derives both from fixed constants
        # instead, matching `request_admission._legacy_request`'s table. The
        # two are NOT interchangeable — do not converge them without
        # updating both call sites' expectations.
        provider_id = snapshot.provider_id
        if provider_id == "openai":
            model_id = snapshot.model_id.lower().replace("-", "")
            return openai_internal_model_id(model_id)
        if provider_id == "elevenlabs":
            return f"elevenlabs_{snapshot.model_id}"
        if provider_id == "kokoro":
            engine = "onnx" if options.get("use_onnx", True) else "pytorch"
            return f"local_kokoro_default_{engine}"
        if provider_id == "chatterbox":
            return "local_chatterbox_default"
        if provider_id == "higgs":
            return "local_higgs_v2"
        if provider_id == "alltalk":
            return f"alltalk_{snapshot.model_id}"
        return snapshot.model_id

    @staticmethod
    def _audio_content_type(audio_format: str) -> str:
        return {
            "aac": "audio/aac",
            "flac": "audio/flac",
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "pcm": "audio/L16",
            "wav": "audio/wav",
        }.get(audio_format, "application/octet-stream")

    async def handle_playground_generate(
        self, event: STTSPlaygroundGenerateEvent
    ) -> None:
        """Run playground TTS inside the handler's retained event task."""
        if self._cleanup_task is not None:
            logger.debug("Ignoring TTS generation after STTS cleanup started")
            return
        if self._retired_playground_operation_id == event.request.operation_id:
            self._retired_playground_operation_id = None
            return
        if self._is_generating:
            self.app.notify("TTS generation already in progress", severity="warning")
            return

        if not self._stts_service:
            operation_id = event.request.operation_id
            self._is_generating = True
            self._active_playground_operation_id = operation_id
            self._deliver_generation_failure(
                operation_id,
                "The TTS service is unavailable",
            )
            self.app.notify("TTS service not initialized", severity="error")
            self._is_generating = False
            self._finish_generation_ui(operation_id)
            self._active_playground_operation_id = None
            return

        self._is_generating = True
        self._active_playground_operation_id = event.request.operation_id
        await self._generate_tts_worker(event)

    async def _generate_tts_worker(
        self,
        event: STTSPlaygroundGenerateEvent,
    ) -> None:
        """Generate from one immutable request and deliver one artifact."""
        snapshot = event.request
        sample_candidate = self._capture_sample_evidence_candidate(snapshot)
        self._show_generation_progress(snapshot.operation_id)

        async def progress_callback(info: TTSProgress) -> None:
            self._update_generation_progress(snapshot.operation_id, info)

        try:
            if snapshot.studio_preferences is not None:
                artifact = await self._generate_studio_effective(
                    snapshot,
                    progress_callback,
                )
            elif snapshot.provider_id == "audio_cpp":
                artifact = await self._generate_audio_cpp(
                    snapshot,
                    progress_callback,
                )
            else:
                legacy_facts = self._legacy_sample_generation_facts(
                    snapshot,
                    sample_candidate,
                )
                if legacy_facts is not None:
                    self._sample_generation_facts[snapshot.operation_id] = legacy_facts
                artifact = await self._generate_legacy(
                    snapshot,
                    progress_callback,
                )
            if self._retired_playground_operation_id == snapshot.operation_id:
                self._delete_operation_files(snapshot.operation_id)
                return
            sample_facts = self._sample_generation_facts.pop(
                snapshot.operation_id,
                None,
            )
            if (
                sample_facts is None
                and snapshot.provider_id == "audio_cpp"
                and artifact.requested_selection is not None
            ):
                sample_facts = _SampleGenerationFacts(
                    runtime_revision=(
                        artifact.requested_selection.configuration_revision
                    ),
                    certifies_saved_configuration=(
                        self._requested_selection_matches_sample_request(
                            artifact.requested_selection,
                            snapshot,
                        )
                    ),
                )
            self._record_successful_sample_evidence(
                artifact,
                sample_candidate,
                sample_facts,
            )
            self._accept_playground_artifact(artifact)
            self._deliver_generation_success(
                snapshot.operation_id,
                artifact,
                accepted_clone_draft_revision=(
                    snapshot.clone_audition.draft_revision
                    if snapshot.clone_audition is not None
                    else None
                ),
            )
            self.app.notify("TTS generation complete!", severity="information")
        except asyncio.CancelledError:
            self._delete_operation_files(snapshot.operation_id)
            raise
        except Exception as error:
            self._delete_operation_files(snapshot.operation_id)
            if self._retired_playground_operation_id == snapshot.operation_id:
                return
            message = self._generation_error_copy(error)
            if isinstance(error, TTSOperationError):
                logger.error(
                    "TTS generation failed (code={}, retryable={})",
                    error.code,
                    error.retryable,
                )
            else:
                logger.error(
                    "TTS generation failed ({})",
                    type(error).__name__,
                )
            self._deliver_generation_failure(snapshot.operation_id, message)
            self.app.notify(
                f"TTS generation failed: {escape(message)}",
                severity="error",
            )
        finally:
            self._sample_generation_facts.pop(snapshot.operation_id, None)
            if self._active_playground_operation_id == snapshot.operation_id:
                self._is_generating = False
                if self._retired_playground_operation_id == snapshot.operation_id:
                    self._retired_playground_operation_id = None
                self._finish_generation_ui(snapshot.operation_id)
                self._active_playground_operation_id = None

    def _show_generation_progress(self, operation_id: str) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def show() -> None:
            playground.query_one("#generation-status-container").remove_class("hidden")
            playground.query_one("#generation-progress").update(
                total=100,
                progress=0,
            )

        self._invoke_playground(playground, show)

    def _update_generation_progress(
        self,
        operation_id: str,
        info: TTSProgress,
    ) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def update() -> None:
            playground.query_one(
                "#generation-status-text",
                Static,
            ).update(info.status or "Generating...")
            if info.fraction is not None:
                playground.query_one(
                    "#generation-progress",
                    ProgressBar,
                ).update(progress=info.fraction * 100)
            log = playground.query_one("#tts-generation-log", RichLog)
            audio_duration = info.metrics.get("audio_duration")
            if isinstance(audio_duration, (int, float)):
                log.write(f"[dim]Generated {audio_duration:.1f}s of audio[/dim]")
            elif info.processed is not None:
                if info.total is None:
                    log.write(f"[dim]Processed {info.processed} item(s)[/dim]")
                else:
                    log.write(
                        f"[dim]Processed {info.processed}/{info.total} item(s)[/dim]"
                    )

        self._invoke_playground(playground, update)

    def _deliver_generation_success(
        self,
        operation_id: str,
        artifact: STTSGeneratedAudio,
        *,
        accepted_clone_draft_revision: int | None = None,
    ) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def deliver() -> None:
            playground.query_one("#tts-generation-log", RichLog).write(
                "[bold green]Generation complete[/bold green]"
            )
            callback = getattr(playground, "_generation_complete", None)
            if accepted_clone_draft_revision is not None:
                accept_clone = getattr(
                    playground,
                    "_accept_clone_generation_result",
                    None,
                )
                if callable(accept_clone):
                    accept_clone(operation_id, accepted_clone_draft_revision)
            if callable(callback):
                callback(STTSPlaygroundResultProjection.from_artifact(artifact))
                return
            playground.query_one("#audio-play-btn", Button).disabled = False
            playground.query_one("#audio-export-btn", Button).disabled = False
            playground.query_one("#audio-player-status", Static).update(
                "Audio ready to play"
            )

        self._invoke_playground(playground, deliver)

    def _deliver_generation_failure(
        self,
        operation_id: str,
        message: str,
    ) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def deliver() -> None:
            playground.query_one("#tts-generation-log", RichLog).write(
                f"[bold red]Generation failed: {escape(message)}[/bold red]"
            )
            callback = getattr(playground, "_generation_complete", None)
            if callable(callback):
                callback(None)

        self._invoke_playground(playground, deliver)

    def _finish_generation_ui(self, operation_id: str) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def finish() -> None:
            sync_generate_enabled = getattr(
                playground,
                "_sync_generate_enabled",
                None,
            )
            if callable(sync_generate_enabled):
                sync_generate_enabled()
            else:
                playground.query_one(
                    "#tts-generate-btn",
                    Button,
                ).disabled = False
            playground.query_one("#generation-status-container").add_class("hidden")

        self._invoke_playground(playground, finish)

    def _mounted_playground(self, operation_id: str) -> Any | None:
        if (
            operation_id != self._active_playground_operation_id
            or operation_id == self._retired_playground_operation_id
        ):
            return None
        # `STTSWindow._mount_view` only ever mounts `SpeechPlaygroundPane`
        # for the `playground` view (this used to also try the retired
        # legacy playground widget -- TASK-2951 -- which was already
        # unreachable before that). Tried under both the active screen and
        # the app itself, mirroring the pane's own tests, which mount it
        # under a bare host with no screen.
        from tldw_chatbook.UI.Speech.speech_playground_pane import (
            SpeechPlaygroundPane,
        )

        roots: list[Any] = []
        try:
            active_screen = self.app.screen
        except Exception:
            active_screen = None
        if active_screen is not None:
            roots.append(active_screen)
        roots.append(self.app)

        for root in roots:
            try:
                return root.query_one(SpeechPlaygroundPane)
            except Exception as error:
                logger.debug(
                    "{} is not mounted under {} ({})",
                    SpeechPlaygroundPane.__name__,
                    type(root).__name__,
                    type(error).__name__,
                )
        return None

    @staticmethod
    def _invoke_playground(
        playground: Any,
        callback: object,
        *args: object,
    ) -> None:
        try:
            call_from_thread = getattr(playground, "call_from_thread", None)
            if callable(call_from_thread):
                call_from_thread(callback, *args)
            elif callable(callback):
                callback(*args)
        except Exception as error:
            logger.debug(
                "Playground generation display update failed ({})",
                type(error).__name__,
            )

    @staticmethod
    def _generation_error_copy(error: Exception) -> str:
        if isinstance(error, TTSOperationError):
            parts = [str(error)]
            recovery_copy = _RECOVERY_ACTION_COPY.get(error.recovery_action or "")
            if recovery_copy is not None:
                parts.append(recovery_copy)
            elif error.retryable:
                parts.append(_RECOVERY_ACTION_COPY["retry"])
            return " ".join(parts)
        if isinstance(error, TTSProviderReconfiguringError):
            return "TTS settings are being applied; retry shortly"
        if isinstance(error, TTSRegistryClosedError):
            return "The TTS service is unavailable"
        if isinstance(error, ValueError):
            return "TTS is not configured; open STTS Settings"
        return "Unexpected TTS generation failure; retry"

    async def handle_settings_save(self, event: STTSSettingsSaveEvent) -> None:
        """Handle settings save"""
        try:
            if self._cleanup_task is not None:
                logger.debug("Ignoring STTS settings after cleanup started")
                self._reply_settings_save(
                    event,
                    persisted=False,
                    provider_statuses={},
                    failure_phase="before_replace",
                )
                return
            async with self._settings_save_lock:
                if self._cleanup_task is not None:
                    logger.debug("Ignoring STTS settings after cleanup started")
                    self._reply_settings_save(
                        event,
                        persisted=False,
                        provider_statuses={},
                        failure_phase="before_replace",
                    )
                    return
                await self._persist_settings(event)
        finally:
            event._abandon_publication_lease()

    async def _persist_settings(self, event: STTSSettingsSaveEvent) -> None:
        """Persist and publish one validated, service-owned settings proposal."""
        try:
            from tldw_chatbook import config as config_module

            section_values: dict[str, dict[str, Any]] = {}
            saved_destinations: list[tuple[str, str, str]] = []
            deleted_destinations: list[tuple[str, str, str]] = []
            candidate_provider_ids: set[str] = set()
            for key, value in event.settings.items():
                binding = _TTS_SETTING_BINDINGS.get(key)
                if binding is None:
                    continue
                for section, setting_name in binding.destinations:
                    section_values.setdefault(section, {})[setting_name] = deepcopy(
                        value
                    )
                    saved_destinations.append((key, section, setting_name))
                if binding.provider_id is not None:
                    candidate_provider_ids.add(binding.provider_id)

            logical_sets = set(event.settings)
            logical_deletes = set(event.delete_setting_keys)
            if logical_sets & logical_deletes:
                raise ValueError("A TTS setting cannot be set and deleted together")
            proposed_deletes: dict[str, set[str]] = {}
            for key in event.delete_setting_keys:
                binding = _TTS_SETTING_BINDINGS.get(key)
                if binding is None:
                    raise ValueError("Unknown TTS setting delete key")
                destinations = binding.delete_destinations or binding.destinations
                for section, setting_name in destinations:
                    proposed_deletes.setdefault(section, set()).add(setting_name)
                    deleted_destinations.append((key, section, setting_name))
                if binding.provider_id is not None:
                    candidate_provider_ids.add(binding.provider_id)

            initial_delete_keys = {
                section: tuple(sorted(keys))
                for section, keys in proposed_deletes.items()
            }

            current_settings = getattr(config_module, "settings", {})
            if not isinstance(current_settings, Mapping):
                raise ValueError("Current settings are unavailable")
            provisional_settings = _prospective_effective_settings(
                current_settings,
                section_values,
                initial_delete_keys,
            )
            preferences = (
                event.preferences
                if event.preferences is not None
                else TTSPreferencesSnapshot.from_settings(provisional_settings)
            )
            if not isinstance(preferences, TTSPreferencesSnapshot):
                raise TypeError("Invalid TTS preferences proposal")
            if not event.commit_defaults_after_handoff:
                preference_mutation = preferences.config_mutation()
                _merge_section_mutations(section_values, preference_mutation.sets)
                for section, keys in preference_mutation.deletes.items():
                    proposed_deletes.setdefault(section, set()).update(keys)
            delete_keys = {
                section: tuple(sorted(keys))
                for section, keys in proposed_deletes.items()
            }
            for section, keys in delete_keys.items():
                values = section_values.get(section)
                if values is None:
                    continue
                for key in keys:
                    values.pop(key, None)

            effective_settings = _prospective_effective_settings(
                current_settings,
                section_values,
                delete_keys,
            )
            candidate_providers = [
                provider_id
                for provider_id in _TTS_PROVIDER_ORDER
                if provider_id in candidate_provider_ids
            ]
            provider_configs = {
                provider_id: _effective_provider_config(
                    provider_id,
                    effective_settings,
                )
                for provider_id in candidate_providers
            }

            service = self._stts_service
            if service is None:
                service = await get_tts_service()
                self._stts_service = service
            if not isinstance(service, TTSService) and not callable(
                getattr(service, "begin_preferences_publication", None)
            ):
                raise RuntimeError("The TTS service cannot publish settings")

            frozen_sections = deepcopy(section_values)
            frozen_deletes = deepcopy(delete_keys)

            def persist() -> TTSSettingsPersistenceOutcome:
                result = config_module.apply_settings_mutation_to_cli_config(
                    frozen_sections,
                    delete_keys=frozen_deletes,
                )
                return TTSSettingsPersistenceOutcome(
                    file_replaced=result.file_replaced,
                    caches_reloaded=result.caches_reloaded,
                    failure_phase=result.failure_phase,
                )

            for provider_id in candidate_providers:
                self._default_activation_intents.pop(provider_id, None)
            activation_intent: _DefaultActivationIntent | None = None
            if event.commit_defaults_after_handoff:
                ticket = service.begin_preferences_publication(
                    preferences,
                    provider_configs,
                    persist,
                    publish_preferences=False,
                    publication_lease=event.publication_lease,
                )
                activation_intent = self._new_default_activation_intent(
                    preferences,
                    expected_saved_revision=ticket.generation,
                )
            else:
                ticket = service.begin_preferences_publication(
                    preferences,
                    provider_configs,
                    persist,
                    publication_lease=event.publication_lease,
                )
            event._publication_started()
            publication = await asyncio.shield(ticket.foreground)
            activation_outcome: TTSDefaultActivationOutcome | None = None
            if event.commit_defaults_after_handoff:
                pending_handoff = any(
                    status == "pending"
                    and provider_id not in publication.staged_provider_ids
                    for provider_id, status in publication.provider_statuses.items()
                )
                if pending_handoff:
                    activation_outcome = TTSDefaultActivationOutcome(
                        "activation_not_ready"
                    )
                else:
                    claimed = self._claim_default_activation_intent(activation_intent)
                    activation_outcome = (
                        await self.commit_voice_setup_default(
                            preferences,
                            expected_saved_revision=publication.generation,
                        )
                        if claimed
                        else TTSDefaultActivationOutcome("activation_not_ready")
                    )
            if publication.persistence.caches_reloaded and self.app is not None:
                refreshed_settings = getattr(config_module, "settings", None)
                if isinstance(refreshed_settings, Mapping):
                    self.app.app_config = deepcopy(dict(refreshed_settings))
            if publication.persistence.file_replaced:
                for key, section, setting_name in saved_destinations:
                    logger.info(
                        "Saved {} to [{}].{}",
                        key,
                        section,
                        setting_name,
                    )
                for key, section, setting_name in deleted_destinations:
                    logger.info(
                        "Cleared {} from [{}].{}",
                        key,
                        section,
                        setting_name,
                    )

            self._post_applied_settings_changes(
                service,
                publication,
            )
            if any(
                status == "pending"
                and provider_id not in publication.staged_provider_ids
                for provider_id, status in publication.provider_statuses.items()
            ):
                self._observe_pending_settings_publication(
                    service,
                    ticket,
                    event,
                    activation_intent=activation_intent,
                )
            self._notify_settings_publication(
                publication,
                activation_outcome=activation_outcome,
            )
            self._reply_settings_save(
                event,
                persisted=publication.persistence.file_replaced,
                provider_statuses=publication.provider_statuses,
                failure_phase=publication.persistence.failure_phase,
                provider_configuration_revisions={
                    provider_id: publication.generation
                    for provider_id in publication.provider_statuses
                },
                provider_runtime_revisions=publication.provider_revisions,
                staged_provider_ids=publication.staged_provider_ids,
                defaults_activated=(
                    activation_outcome.activated
                    if activation_outcome is not None
                    else None
                ),
                defaults_activation_status=(
                    activation_outcome.status
                    if activation_outcome is not None
                    else None
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            message = "Failed to save settings"
            logger.error(message)
            self.app.notify(message, severity="error")
            self._reply_settings_save(
                event,
                persisted=False,
                provider_statuses={},
                failure_phase="before_replace",
            )

    @staticmethod
    def _reply_settings_save(
        event: STTSSettingsSaveEvent,
        *,
        persisted: bool,
        provider_statuses: Mapping[str, str],
        failure_phase: str | None,
        provider_configuration_revisions: Mapping[str, int] = MappingProxyType({}),
        provider_runtime_revisions: Mapping[str, int] = MappingProxyType({}),
        staged_provider_ids: frozenset[str] = frozenset(),
        defaults_activated: bool | None = None,
        defaults_activation_status: str | None = None,
    ) -> None:
        """Deliver a bounded result to an optional mounted requester."""
        if event.request_id is None or event.reply_to is None:
            return
        callback = getattr(
            event.reply_to,
            "receive_stts_settings_save_result",
            None,
        )
        if not callable(callback):
            return
        try:
            callback(
                STTSSettingsSaveResult(
                    request_id=event.request_id,
                    persisted=persisted,
                    provider_statuses=provider_statuses,
                    failure_phase=failure_phase,
                    provider_configuration_revisions=(provider_configuration_revisions),
                    provider_runtime_revisions=provider_runtime_revisions,
                    staged_provider_ids=staged_provider_ids,
                    defaults_activated=defaults_activated,
                    defaults_activation_status=defaults_activation_status,
                )
            )
        except Exception:
            logger.debug("TTS settings requester result delivery failed")

    async def commit_voice_setup_default(
        self,
        preferences: TTSPreferencesSnapshot,
        *,
        expected_saved_revision: int,
    ) -> TTSDefaultActivationOutcome:
        """Persist and activate only default axes for one active provider generation."""

        service = self._stts_service
        if service is None or not callable(
            getattr(service, "_commit_voice_setup_default", None)
        ):
            return TTSDefaultActivationOutcome("activation_not_ready")
        mutation = preferences.config_mutation()
        from tldw_chatbook import config as config_module

        current_settings = getattr(config_module, "settings", {})
        try:
            persisted_preferences = TTSPreferencesSnapshot.from_settings(
                current_settings if isinstance(current_settings, Mapping) else {}
            )
        except (TypeError, ValueError):
            persisted_preferences = None
        frozen_sets = {
            section: deepcopy(dict(values)) for section, values in mutation.sets.items()
        }
        frozen_deletes = {
            section: tuple(keys) for section, keys in mutation.deletes.items()
        }

        def persist_defaults() -> TTSSettingsPersistenceOutcome:
            result = config_module.apply_settings_mutation_to_cli_config(
                frozen_sets,
                delete_keys=frozen_deletes,
            )
            return TTSSettingsPersistenceOutcome(
                file_replaced=result.file_replaced,
                caches_reloaded=result.caches_reloaded,
                failure_phase=result.failure_phase,
            )

        def rollback_defaults(
            prior_preferences: TTSPreferencesSnapshot | None,
        ) -> TTSSettingsPersistenceOutcome:
            rollback_preferences = prior_preferences or persisted_preferences
            if rollback_preferences is None:
                return TTSSettingsPersistenceOutcome(
                    file_replaced=False,
                    caches_reloaded=False,
                    failure_phase="before_replace",
                )
            prior_mutation = rollback_preferences.config_mutation()
            prior_sets = {
                section: deepcopy(dict(values))
                for section, values in prior_mutation.sets.items()
            }
            prior_deletes = {
                section: tuple(keys) for section, keys in prior_mutation.deletes.items()
            }
            result = config_module.apply_settings_mutation_to_cli_config(
                prior_sets,
                delete_keys=prior_deletes,
            )
            return TTSSettingsPersistenceOutcome(
                file_replaced=result.file_replaced,
                caches_reloaded=result.caches_reloaded,
                failure_phase=result.failure_phase,
            )

        outcome = await service._commit_voice_setup_default(
            preferences,
            expected_saved_revision=expected_saved_revision,
            persistence=persist_defaults,
            rollback=rollback_defaults,
        )
        if outcome.activated and self.app is not None:
            from tldw_chatbook import config as config_module

            refreshed_settings = getattr(config_module, "settings", None)
            if isinstance(refreshed_settings, Mapping):
                self.app.app_config = deepcopy(dict(refreshed_settings))
        return outcome

    def _new_default_activation_intent(
        self,
        preferences: TTSPreferencesSnapshot,
        *,
        expected_saved_revision: int,
    ) -> _DefaultActivationIntent:
        """Replace any older provider intent with one immutable token."""

        intent = _DefaultActivationIntent(
            preferences=preferences,
            expected_saved_revision=expected_saved_revision,
            provider_id=preferences.provider_id,
            token=self._next_default_activation_token,
        )
        self._next_default_activation_token += 1
        self._default_activation_intents[preferences.provider_id] = intent
        return intent

    def _claim_default_activation_intent(
        self,
        intent: _DefaultActivationIntent | None,
    ) -> bool:
        """Consume an intent exactly once if it remains the newest token."""

        if intent is None:
            return False
        if self._default_activation_intents.get(intent.provider_id) != intent:
            return False
        self._default_activation_intents.pop(intent.provider_id, None)
        return True

    def _discard_default_activation_intent(
        self,
        intent: _DefaultActivationIntent | None,
    ) -> None:
        if (
            intent is not None
            and self._default_activation_intents.get(intent.provider_id) == intent
        ):
            self._default_activation_intents.pop(intent.provider_id, None)

    def _post_applied_settings_changes(
        self,
        service: TTSService,
        publication: TTSSettingsPublication,
    ) -> None:
        """Post every provider-scoped handoff that definitively applied."""
        posted_provider_ids: set[str] = set()
        global_revision = publication.generation if publication.published else None
        for provider_id, status in publication.provider_statuses.items():
            if status != "applied":
                continue
            revision = publication.provider_revisions.get(provider_id)
            if revision is None:
                continue
            self.app.post_message(
                STTSProviderConfigurationChanged(
                    provider_id,
                    revision,
                    global_revision,
                )
            )
            posted_provider_ids.add(provider_id)

        if global_revision is None or posted_provider_ids:
            return
        provider_id = publication.preferences.provider_id
        provider_status = publication.provider_statuses.get(provider_id)
        if provider_status not in {None, "unchanged"}:
            return
        revision = publication.provider_revisions.get(provider_id)
        if revision is None:
            revision_reader = getattr(service, "configuration_revision", None)
            if not callable(revision_reader):
                return
            try:
                revision = revision_reader(provider_id)
            except (KeyError, RuntimeError, TypeError, ValueError):
                return
        if type(revision) is not int or revision < 0:
            return
        self.app.post_message(
            STTSProviderConfigurationChanged(
                provider_id,
                revision,
                global_revision,
            )
        )

    def _observe_pending_settings_publication(
        self,
        service: TTSService,
        ticket: TTSSettingsPublicationTicket,
        event: STTSSettingsSaveEvent,
        *,
        activation_intent: _DefaultActivationIntent | None = None,
    ) -> None:
        """Publish a bounded final handoff for the still-current save."""

        async def observe() -> None:
            try:
                completion = await asyncio.shield(ticket.completion)
            except asyncio.CancelledError:
                self._discard_default_activation_intent(activation_intent)
                raise
            except BaseException:
                self._discard_default_activation_intent(activation_intent)
                return
            activation_outcome: TTSDefaultActivationOutcome | None = None
            if activation_intent is not None:
                provider_status = completion.provider_statuses.get(
                    activation_intent.provider_id
                )
                if (
                    completion.generation == activation_intent.expected_saved_revision
                    and provider_status in {"applied", "unchanged"}
                    and self._claim_default_activation_intent(activation_intent)
                ):
                    activation_outcome = await self.commit_voice_setup_default(
                        activation_intent.preferences,
                        expected_saved_revision=(
                            activation_intent.expected_saved_revision
                        ),
                    )
                else:
                    self._discard_default_activation_intent(activation_intent)
                    activation_outcome = TTSDefaultActivationOutcome(
                        "activation_not_ready"
                    )
            self._post_applied_settings_changes(service, completion)
            self._reply_settings_runtime(
                event,
                completion,
                activation_outcome=activation_outcome,
            )
            if (
                activation_outcome is not None
                and activation_outcome.status == "rollback_failed"
            ):
                self.app.notify(
                    "Defaults were saved, but rollback failed. Runtime still uses the "
                    "previous default; restart may use the new default. Retry to "
                    "reconcile.",
                    severity="error",
                )
            if "unavailable" in completion.provider_statuses.values():
                self.app.notify(
                    "TTS settings are unavailable. Retry/Reconnect.",
                    severity="error",
                )

        self._start_event_task(observe())

    @staticmethod
    def _reply_settings_runtime(
        event: STTSSettingsSaveEvent,
        publication: TTSSettingsPublication,
        *,
        activation_outcome: TTSDefaultActivationOutcome | None = None,
    ) -> None:
        """Deliver one safe final runtime result to the original requester."""

        if event.request_id is None or event.reply_to is None:
            return
        callback = getattr(
            event.reply_to,
            "receive_stts_settings_runtime_result",
            None,
        )
        if not callable(callback):
            return
        provider_statuses = {
            provider_id: status
            for provider_id, status in publication.provider_statuses.items()
            if provider_id not in publication.staged_provider_ids
        }
        if not provider_statuses:
            return
        try:
            callback(
                STTSSettingsSaveResult(
                    request_id=event.request_id,
                    persisted=publication.persistence.file_replaced,
                    provider_statuses=provider_statuses,
                    failure_phase=publication.persistence.failure_phase,
                    provider_configuration_revisions={
                        provider_id: publication.generation
                        for provider_id in provider_statuses
                    },
                    provider_runtime_revisions={
                        provider_id: revision
                        for provider_id, revision in publication.provider_revisions.items()
                        if provider_id in provider_statuses
                    },
                    defaults_activated=(
                        activation_outcome.activated
                        if activation_outcome is not None
                        else None
                    ),
                    defaults_activation_status=(
                        activation_outcome.status
                        if activation_outcome is not None
                        else None
                    ),
                )
            )
        except Exception:
            logger.debug("TTS settings runtime result delivery failed")

    def _notify_settings_publication(
        self,
        publication: TTSSettingsPublication,
        *,
        activation_outcome: TTSDefaultActivationOutcome | None = None,
    ) -> None:
        """Render bounded, value-independent settings publication copy."""
        persistence = publication.persistence
        statuses = publication.provider_statuses
        if not persistence.file_replaced:
            if persistence.failure_phase == "before_replace":
                self.app.notify("Failed to save settings", severity="error")
            else:
                self.app.notify("Settings unchanged", severity="information")
            return
        if (
            activation_outcome is not None
            and activation_outcome.status == "rollback_failed"
        ):
            self.app.notify(
                "Defaults were saved, but rollback failed. Runtime still uses the "
                "previous default; restart may use the new default. Retry to reconcile.",
                severity="error",
            )
            return
        if (
            activation_outcome is not None
            and activation_outcome.status == "activation_not_ready"
            and "pending" in statuses.values()
        ):
            self.app.notify(
                "Settings saved; default activation is waiting for TTS handoff.",
                severity="information",
            )
            return
        if activation_outcome is not None and not activation_outcome.activated:
            self.app.notify(
                "Saved, activation failed. Previous TTS defaults remain active; retry.",
                severity="error",
            )
            return
        if "unavailable" in statuses.values():
            self.app.notify(
                "Settings saved, but TTS is unavailable. Retry/Reconnect.",
                severity="error",
            )
            return
        if not persistence.caches_reloaded:
            self.app.notify(
                "Settings saved and TTS runtime updated; restart recommended.",
                severity="warning",
            )
            return
        if statuses.get("audio_cpp") == "pending":
            if "audio_cpp" in publication.staged_provider_ids:
                self.app.notify(
                    "Saved — open Speech Lab to apply audio.cpp settings",
                    severity="information",
                )
                return
            self.app.notify(
                "Saved — applying after current speech",
                severity="information",
            )
            return
        self.app.notify("Settings saved successfully!", severity="information")

    async def _convert_audio_format(
        self, input_file: Path, output_format: str
    ) -> Optional[Path]:
        """Convert audio file to a different format using ffmpeg"""
        process: asyncio.subprocess.Process | None = None
        try:
            # Create output file with requested format
            output_file = input_file.with_suffix(f".{output_format}")

            # Use ffmpeg for conversion
            cmd = [
                "ffmpeg",
                "-i",
                str(input_file),
                "-y",  # Overwrite output files
                "-loglevel",
                "error",  # Suppress verbose output
            ]

            # Add format-specific options
            if output_format == "mp3":
                # High quality MP3 encoding
                cmd.extend(["-codec:a", "libmp3lame", "-b:a", "192k"])
            elif output_format == "opus":
                cmd.extend(["-codec:a", "libopus", "-b:a", "128k"])
            elif output_format == "aac":
                cmd.extend(["-codec:a", "aac", "-b:a", "192k"])
            elif output_format == "flac":
                cmd.extend(["-codec:a", "flac"])

            cmd.append(str(output_file))

            # Run conversion asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            # Wait for the process to complete
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"Successfully converted audio to {output_format}")
                return output_file
            else:
                stderr_text = stderr.decode("utf-8") if stderr else "Unknown error"
                logger.error(f"ffmpeg conversion failed: {stderr_text}")
                return None

        except asyncio.CancelledError:
            if process is not None:
                terminate_task = asyncio.create_task(
                    self._terminate_conversion_process(process),
                    name="stts_ffmpeg_terminate",
                )
                await _join_retained_task(terminate_task)
            raise
        except FileNotFoundError:
            logger.error(
                "ffmpeg not found. Please install ffmpeg for audio format conversion."
            )
            return None
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None

    @staticmethod
    async def _terminate_conversion_process(
        process: asyncio.subprocess.Process,
    ) -> None:
        """Terminate ffmpeg, escalating to kill if it does not exit promptly."""
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()

    async def handle_audiobook_generate(
        self, event: STTSAudioBookGenerateEvent
    ) -> None:
        """Handle audiobook generation"""
        if self._cleanup_task is not None:
            logger.debug("Ignoring audiobook generation after STTS cleanup started")
            return
        try:
            from tldw_chatbook.TTS.audiobook_generator import (
                AudioBookGenerator,
                AudioBookProgress,
                AudioBookRequest,
            )

            logger.info("AudioBook generation requested")

            # Initialize audiobook generator
            generator = AudioBookGenerator(self._stts_service)
            await generator.initialize()

            # Create audiobook request from event data
            audiobook_request = AudioBookRequest(
                content=event.content,
                title=event.options.get("title", "Untitled Book"),
                author=event.options.get("author", "Unknown"),
                narrator_voice=event.narrator_voice,
                provider=event.options.get("provider", "openai"),
                model=event.options.get("model", "tts-1"),
                output_format=event.output_format,
                chapter_detection=event.options.get("chapter_detection", True),
                multi_voice=event.options.get("multi_voice", False),
                character_voices=event.options.get("character_voices", {}),
                voice_settings=event.options.get("voice_settings", {}),
                background_music=event.options.get("background_music"),
                music_volume=event.options.get("music_volume", 0.1),
                chapter_pause_duration=event.options.get("chapter_pause_duration", 2.0),
                paragraph_pause_duration=event.options.get(
                    "paragraph_pause_duration", 0.5
                ),
                sentence_pause_duration=event.options.get(
                    "sentence_pause_duration", 0.3
                ),
                max_chunk_size=event.options.get("max_chunk_size", 4000),
                enable_ssml=event.options.get("enable_ssml", False),
                normalize_audio=event.options.get("normalize_audio", True),
                target_db=event.options.get("target_db", -20.0),
            )

            # Get cost estimate
            estimated_cost = generator.get_cost_estimate(audiobook_request)

            # Update UI with initial status
            if hasattr(self.app, "query_one"):
                try:
                    from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget

                    audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                    if audiobook_widget:
                        log = audiobook_widget.query_one(
                            "#audiobook-generation-log", RichLog
                        )
                        log.write(
                            "[bold yellow]Starting audiobook generation...[/bold yellow]"
                        )
                        log.write(f"Estimated cost: ${estimated_cost:.2f}")
                except Exception:
                    pass  # UI element not found, continue without UI updates

            # Define progress callback
            async def progress_callback(progress: AudioBookProgress):
                """Update UI with generation progress"""
                if hasattr(self.app, "query_one"):
                    try:
                        from tldw_chatbook.UI.STTS_Window import (
                            AudioBookGenerationWidget,
                        )

                        audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                        if audiobook_widget:
                            log = audiobook_widget.query_one(
                                "#audiobook-generation-log", RichLog
                            )

                            # Update progress message
                            if progress.current_chapter:
                                log.write(
                                    f"[cyan]Processing: {progress.current_chapter}[/cyan]"
                                )

                            # Update progress bar if available
                            if progress.total_chapters > 0:
                                percent_complete = (
                                    progress.completed_chapters
                                    / progress.total_chapters
                                ) * 100
                                log.write(
                                    f"Progress: {progress.completed_chapters}/{progress.total_chapters} chapters ({percent_complete:.1f}%)"
                                )

                            # Show time estimates
                            if progress.estimated_completion:
                                remaining_time = (
                                    progress.estimated_completion - datetime.now()
                                )
                                if remaining_time.total_seconds() > 0:
                                    minutes_remaining = int(
                                        remaining_time.total_seconds() / 60
                                    )
                                    log.write(
                                        f"Estimated time remaining: {minutes_remaining} minutes"
                                    )

                            # Show errors if any
                            for error in progress.errors:
                                log.write(f"[bold red]Error: {error}[/bold red]")
                    except Exception:
                        pass  # UI element not found, continue without UI updates

            # Generate the audiobook
            output_path = await generator.generate_audiobook(
                audiobook_request, progress_callback=progress_callback
            )

            # Update UI with completion
            if hasattr(self.app, "query_one"):
                try:
                    from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget

                    audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                    if audiobook_widget:
                        log = audiobook_widget.query_one(
                            "#audiobook-generation-log", RichLog
                        )
                        log.write(
                            "[bold green]✓ AudioBook generation complete![/bold green]"
                        )
                        log.write(f"Output file: {output_path}")
                        log.write(
                            f"Total duration: {generator.progress.actual_duration / 60:.1f} minutes"
                        )

                        # Enable export button if available
                        export_btn = audiobook_widget.query_one(
                            "#audiobook-export-btn", Button
                        )
                        if export_btn:
                            export_btn.disabled = False
                            # Store the output path for export
                            audiobook_widget.generated_audiobook_path = output_path
                except Exception:
                    pass  # UI element not found

            # Store the generated audiobook path for playback
            self._current_audio_file = output_path

            # Notify the UI widget
            if audiobook_widget and hasattr(
                audiobook_widget, "audiobook_generation_complete"
            ):
                audiobook_widget.audiobook_generation_complete(True, output_path)

            self.app.notify(
                f"AudioBook generated successfully: {output_path.name}",
                severity="information",
            )

        except ImportError as e:
            logger.error(f"Failed to import audiobook generator: {e}")
            self.app.notify(
                "AudioBook generation module not available", severity="error"
            )
        except Exception as e:
            logger.error(f"AudioBook generation failed: {e}")
            self.app.notify(f"AudioBook generation failed: {e}", severity="error")

            # Notify the UI widget of failure
            try:
                from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget

                audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                if audiobook_widget and hasattr(
                    audiobook_widget, "audiobook_generation_complete"
                ):
                    audiobook_widget.audiobook_generation_complete(False)
            except Exception:
                pass

    def _start_event_task(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """Start and retain an event task until it finishes."""
        if self._cleanup_task is not None:
            coroutine.close()
            logger.debug("Ignoring STTS event after cleanup started")
            return
        task = asyncio.create_task(coroutine)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def cleanup_tts_resources(self) -> None:
        """Join retained cleanup before propagating caller cancellation."""
        if self._cleanup_task is None:
            caller = asyncio.current_task()
            self._cleanup_task = asyncio.create_task(
                self._cleanup_owned_resources(caller),
                name="stts_handler_cleanup",
            )
        await _join_retained_task(self._cleanup_task)

    async def _cleanup_owned_resources(
        self,
        caller: asyncio.Task[Any] | None,
    ) -> None:
        """Cancel handler work and delete only playground-owned temporary audio."""
        tasks = tuple(task for task in self._active_tasks if task is not caller)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.difference_update(tasks)

        owned_paths = tuple(self._playground_audio_files)
        self._playground_file_leases.clear()
        for path in owned_paths:
            if secure_delete_file(path) or not path.exists():
                self._playground_audio_files.discard(path)
                for operation_id in tuple(self._playground_operation_files):
                    self._forget_operation_file(operation_id, path)

        if (
            self._current_audio_file in owned_paths
            and self._current_audio_file not in self._playground_audio_files
        ):
            self._current_audio_file = None
        if (
            self._current_playground_artifact is not None
            and self._current_playground_artifact.path
            not in self._playground_audio_files
        ):
            self._current_playground_artifact = None
        if self._generation_task is not None and self._generation_task.done():
            self._generation_task = None
        self._active_playground_operation_id = None
        self._retired_playground_operation_id = None
        self._is_generating = False

    def on_stts_playground_generate_event(
        self, event: STTSPlaygroundGenerateEvent
    ) -> None:
        """Start a retained async task for playground generation."""
        self.start_playground_generation(event)

    def on_stts_settings_save_event(self, event: STTSSettingsSaveEvent) -> None:
        """Handle settings save event"""
        self._start_event_task(self.handle_settings_save(event))

    def on_stts_provider_configuration_changed(
        self,
        event: STTSProviderConfigurationChanged,
    ) -> None:
        """Invalidate any mounted Playground for the changed provider."""
        for widget in self.app.query("STTSWindow"):
            callback = getattr(widget, "receive_provider_configuration_changed", None)
            if callable(callback):
                callback(event)
        for widget in self.app.query("SpeechPlaygroundPane"):
            callback = getattr(widget, "mark_provider_configuration_changed", None)
            if callable(callback):
                callback(event.provider_id, event.configuration_revision)

    def on_stts_audiobook_generate_event(
        self, event: STTSAudioBookGenerateEvent
    ) -> None:
        """Handle audiobook generate event"""
        self._start_event_task(self.handle_audiobook_generate(event))


#
# End of stts_events.py
#######################################################################################################################
