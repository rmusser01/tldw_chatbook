"""Bounded ownership contracts shared by Speech Settings and the Lab.

This module is deliberately pure.  It inventories the current built-in
Speech controls and defines future ownership without mounting a widget,
reading configuration, contacting a provider, or changing a persistence path.
Studio ownership is intentionally conservative: only values the current
request path can honor per operation are classified there. Constructor-time
legacy values remain global until a real request-scoped contract exists.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from tldw_chatbook.TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS
from tldw_chatbook.UI.Speech.speech_settings_model import ALL_SETTINGS_CONTROLS

SHARED_TTS_DEFAULTS_OWNER_ID = "defaults"
"""Inventory owner used for application-wide shared selection controls."""

DEFAULT_TTS_PROVIDER_CONTROL_ID = "default-provider-select"
CONFIGURE_TTS_PROVIDER_CONTROL_ID = "configure-provider-select"

_ALLOWED_OWNER_IDS = frozenset(BUILT_IN_TTS_PROVIDER_IDS) | {
    SHARED_TTS_DEFAULTS_OWNER_ID
}


class SpeechTTSConfigurationState(StrEnum):
    """Locally derived configuration states accepted by ADR-039."""

    INHERITED = "Inherited"
    DEFAULT = "Default"
    SAVED = "Saved"
    UNSAVED = "Unsaved"
    INCOMPLETE = "Incomplete"
    INVALID = "Invalid"


class SpeechTTSConnectionState(StrEnum):
    """Bounded connection states for provider test evidence."""

    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    NOT_TESTED = "not_tested"
    UNSUPPORTED = "unsupported"


class SpeechTTSConfigurationValidity(StrEnum):
    """Local validity kept independent from provider connectivity."""

    VALID = "valid"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


class SpeechTTSTestOperation(StrEnum):
    """Provider operation that produced a connection observation."""

    CATALOG = "catalog"
    SAMPLE = "sample"


@dataclass(frozen=True, slots=True)
class ProviderTestFingerprint:
    """Non-secret identity for one saved provider configuration."""

    provider_id: str
    normalized_fields: tuple[tuple[str, str], ...]
    saved_revision: int

    def __post_init__(self) -> None:
        _validate_provider_id(self.provider_id)
        _validate_revision(
            self.saved_revision, "Saved Speech TTS configuration revision"
        )
        if type(self.normalized_fields) is not tuple:
            raise TypeError("Provider test fields must be a tuple")
        prior_key = ""
        for field in self.normalized_fields:
            if (
                type(field) is not tuple
                or len(field) != 2
                or type(field[0]) is not str
                or type(field[1]) is not str
            ):
                raise TypeError("Provider test fields must contain string pairs")
            key, value = field
            if (
                not key
                or key <= prior_key
                or len(key) > 128
                or len(value) > 4096
                or any(
                    unicodedata.category(character) in {"Cc", "Cf", "Cs"}
                    for character in key + value
                )
            ):
                raise ValueError("Provider test fields are invalid")
            prior_key = key

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.provider_id,
            str(self.saved_revision),
            *(part for field in self.normalized_fields for part in field),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SpeechTTSReadinessProjection:
    """Independent local-configuration and provider-connection projection."""

    configuration: SpeechTTSConfigurationValidity
    connection: SpeechTTSConnectionState
    catalog: SpeechTTSConnectionState
    sample: SpeechTTSConnectionState


def combine_tts_readiness(
    configuration: object,
    catalog: object,
    sample: object,
) -> SpeechTTSReadinessProjection:
    """Combine bounded catalog and sample observations."""

    def connection_state(
        value: object, *, sample_value: bool = False
    ) -> SpeechTTSConnectionState:
        if type(value) is SpeechTTSConnectionState:
            return value
        aliases = {
            "success": SpeechTTSConnectionState.REACHABLE,
            "failure": SpeechTTSConnectionState.UNREACHABLE,
            "not_run": SpeechTTSConnectionState.NOT_TESTED,
            "not tested": SpeechTTSConnectionState.NOT_TESTED,
            "model_listing_unavailable": SpeechTTSConnectionState.UNSUPPORTED,
        }
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in aliases:
                return aliases[normalized]
            try:
                return SpeechTTSConnectionState(normalized)
            except ValueError:
                pass
        label = "sample" if sample_value else "catalog"
        raise ValueError(f"Speech TTS {label} state is invalid")

    if type(configuration) is SpeechTTSConfigurationValidity:
        configuration_state = configuration
    elif type(configuration) is SpeechTTSConfigurationState:
        if configuration is SpeechTTSConfigurationState.INVALID:
            configuration_state = SpeechTTSConfigurationValidity.INVALID
        elif configuration is SpeechTTSConfigurationState.INCOMPLETE:
            configuration_state = SpeechTTSConfigurationValidity.INCOMPLETE
        else:
            configuration_state = SpeechTTSConfigurationValidity.VALID
    elif isinstance(configuration, str):
        try:
            configuration_state = SpeechTTSConfigurationValidity(
                configuration.strip().lower()
            )
        except ValueError:
            raise ValueError("Speech TTS configuration validity is invalid") from None
    else:
        raise TypeError("Speech TTS configuration validity is invalid")

    catalog_state = connection_state(catalog)
    sample_state = connection_state(sample, sample_value=True)
    if sample_state is SpeechTTSConnectionState.REACHABLE:
        combined = SpeechTTSConnectionState.REACHABLE
    elif sample_state is SpeechTTSConnectionState.UNREACHABLE:
        combined = SpeechTTSConnectionState.UNREACHABLE
    elif sample_state is SpeechTTSConnectionState.UNSUPPORTED:
        combined = SpeechTTSConnectionState.UNSUPPORTED
    elif catalog_state is SpeechTTSConnectionState.REACHABLE:
        combined = SpeechTTSConnectionState.REACHABLE
    elif catalog_state is SpeechTTSConnectionState.UNREACHABLE:
        combined = SpeechTTSConnectionState.UNREACHABLE
    elif catalog_state is SpeechTTSConnectionState.UNSUPPORTED:
        combined = SpeechTTSConnectionState.UNSUPPORTED
    else:
        combined = SpeechTTSConnectionState.NOT_TESTED

    return SpeechTTSReadinessProjection(
        configuration=configuration_state,
        connection=combined,
        catalog=catalog_state,
        sample=sample_state,
    )


class SpeechTTSRuntimeState(StrEnum):
    """Provider-observation states kept separate from configuration state."""

    NOT_CHECKED = "Not checked"
    CHECKING = "Checking"
    READY = "Ready"
    STALE = "Stale"
    UNAVAILABLE = "Unavailable"
    RECONFIGURING = "Reconfiguring"


class SpeechTTSNavigationIntent(StrEnum):
    """Operations a Settings/Lab deep link may request the user to reach."""

    CONFIGURE = "configure"
    TEST = "test"
    REFRESH_MODELS = "refresh-models"
    REFRESH_VOICES = "refresh-voices"


class SpeechTTSStatusFreshness(StrEnum):
    """Whether a runtime observation is current for its captured revisions."""

    FRESH = "fresh"
    STALE = "stale"


class SpeechTTSDiagnosticCategory(StrEnum):
    """Bounded, non-sensitive categories suitable for runtime status."""

    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    CONTRACT = "contract"
    CATALOG = "catalog"
    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    PROVIDER = "provider"


def _validate_provider_id(provider_id: object) -> None:
    if type(provider_id) is not str:
        raise TypeError("Speech TTS provider ID must be an exact string")
    if provider_id not in BUILT_IN_TTS_PROVIDER_IDS:
        raise ValueError("Speech TTS provider ID is not built in")


def _validate_revision(
    revision: object,
    label: str,
    *,
    optional: bool = False,
) -> None:
    if optional and revision is None:
        return
    if type(revision) is not int:
        raise TypeError(f"{label} must be an integer")
    if revision < 0:
        raise ValueError(f"{label} must be nonnegative")


def speech_tts_model_scope(model_id: str | None) -> str | None:
    """Return a stable non-secret identity for an optional exact model ID."""

    if model_id is None:
        return None
    if type(model_id) is not str or not model_id:
        raise TypeError("Speech TTS model ID must be a non-empty string")
    if len(model_id) > 512 or any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in model_id
    ):
        raise ValueError("Speech TTS model ID is invalid")
    return f"sha256:{hashlib.sha256(model_id.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SpeechTTSNavigationTarget:
    """A minimal, non-sensitive Settings/Lab navigation value."""

    provider_id: str
    intent: SpeechTTSNavigationIntent | None = None

    def __post_init__(self) -> None:
        _validate_provider_id(self.provider_id)
        if (
            self.intent is not None
            and type(self.intent) is not SpeechTTSNavigationIntent
        ):
            raise TypeError("Speech TTS navigation intent is invalid")


@dataclass(frozen=True, slots=True)
class SpeechTTSRuntimeStatus:
    """A revision-bound runtime observation without free-form payloads."""

    provider_id: str
    saved_configuration_revision: int
    runtime_revision: int | None
    catalog_revision: int | None
    model_scope: str | None
    runtime_state: SpeechTTSRuntimeState
    observed_at: datetime
    freshness: SpeechTTSStatusFreshness
    diagnostic_category: SpeechTTSDiagnosticCategory | None = None
    recovery_action: SpeechTTSNavigationIntent | None = None

    def __post_init__(self) -> None:
        _validate_provider_id(self.provider_id)
        _validate_revision(
            self.saved_configuration_revision,
            "Saved Speech TTS configuration revision",
        )
        _validate_revision(
            self.runtime_revision,
            "Speech TTS runtime revision",
            optional=True,
        )
        _validate_revision(
            self.catalog_revision,
            "Speech TTS catalog revision",
            optional=True,
        )
        if self.model_scope is not None and (
            type(self.model_scope) is not str
            or len(self.model_scope) != 71
            or not self.model_scope.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in self.model_scope.removeprefix("sha256:")
            )
        ):
            raise ValueError("Speech TTS status model scope is invalid")
        if type(self.runtime_state) is not SpeechTTSRuntimeState:
            raise TypeError("Speech TTS runtime state is invalid")
        if type(self.observed_at) is not datetime:
            raise TypeError("Speech TTS observation time must be a datetime")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("Speech TTS observation time must be timezone-aware")
        if type(self.freshness) is not SpeechTTSStatusFreshness:
            raise TypeError("Speech TTS status freshness is invalid")
        if (self.runtime_state is SpeechTTSRuntimeState.STALE) is not (
            self.freshness is SpeechTTSStatusFreshness.STALE
        ):
            raise ValueError(
                "Stale runtime observations must use the Stale runtime state"
            )
        if self.diagnostic_category is not None and (
            type(self.diagnostic_category) is not SpeechTTSDiagnosticCategory
        ):
            raise TypeError("Speech TTS diagnostic category is invalid")
        if self.recovery_action is not None and (
            type(self.recovery_action) is not SpeechTTSNavigationIntent
        ):
            raise TypeError("Speech TTS recovery action is invalid")


class SpeechTTSOwnershipScope(StrEnum):
    """The five non-overlapping destinations accepted by ADR-039."""

    GLOBAL_CONFIGURATION = "global-configuration"
    STUDIO_PREFERENCE = "studio-preference"
    VOICE_PROFILE_OPERATION = "voice-profile-operation"
    RUNTIME_OPERATION_OR_READOUT = "runtime-operation-or-readout"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class SpeechTTSOwnershipRecord:
    """Classify one current Speech control under one canonical owner."""

    control_id: str
    owner_id: str
    scope: SpeechTTSOwnershipScope
    reason: str = ""

    def __post_init__(self) -> None:
        if type(self.control_id) is not str or not self.control_id:
            raise TypeError("Speech TTS ownership control ID must be a string")
        if type(self.owner_id) is not str:
            raise TypeError("Speech TTS ownership owner must be a string")
        if self.owner_id not in _ALLOWED_OWNER_IDS:
            raise ValueError("Speech TTS ownership owner is unknown")
        if type(self.scope) is not SpeechTTSOwnershipScope:
            raise TypeError("Speech TTS ownership scope is invalid")
        if type(self.reason) is not str:
            raise TypeError("Speech TTS ownership reason must be a string")
        if self.scope is SpeechTTSOwnershipScope.RETIRED:
            if not self.reason.strip():
                raise ValueError("Retired Speech TTS controls require a reason")
        elif self.reason:
            raise ValueError("Only retired Speech TTS controls may have a reason")


def _owned(
    owner_id: str,
    scope: SpeechTTSOwnershipScope,
    *control_ids: str,
) -> tuple[SpeechTTSOwnershipRecord, ...]:
    return tuple(
        SpeechTTSOwnershipRecord(
            control_id=control_id,
            owner_id=owner_id,
            scope=scope,
        )
        for control_id in control_ids
    )


SPEECH_TTS_OWNERSHIP_INVENTORY: tuple[SpeechTTSOwnershipRecord, ...] = (
    *_owned(
        SHARED_TTS_DEFAULTS_OWNER_ID,
        SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        "default-format-select",
        "default-model-select",
        DEFAULT_TTS_PROVIDER_CONTROL_ID,
        "default-speed-input",
        "default-voice-select",
    ),
    *_owned(
        "audio_cpp",
        SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        "audio-cpp-base-url-input",
        "audio-cpp-connect-timeout-input",
        "audio-cpp-max-catalog-models-input",
        "audio-cpp-max-identifier-characters-input",
        "audio-cpp-max-input-characters-input",
        "audio-cpp-max-metadata-bytes-input",
        "audio-cpp-max-response-bytes-input",
        "audio-cpp-max-voices-per-model-input",
        "audio-cpp-mode-value",
        "audio-cpp-privacy-notice",
        "audio-cpp-synthesis-timeout-input",
    ),
    *_owned(
        "audio_cpp",
        SpeechTTSOwnershipScope.RUNTIME_OPERATION_OR_READOUT,
        "audio-cpp-discovery-status",
        "audio-cpp-refresh-models-btn",
        "audio-cpp-test-connection-btn",
    ),
    *_owned(
        "openai",
        SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        "openai-api-key-input",
        "openai-base-url-input",
        "openai-org-id-input",
    ),
    *_owned(
        "elevenlabs",
        SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        "elevenlabs-api-key-input",
        "elevenlabs-format-select",
        "elevenlabs-similarity-input",
        "elevenlabs-speaker-boost-switch",
        "elevenlabs-stability-input",
        "elevenlabs-style-input",
    ),
    *_owned(
        "elevenlabs",
        SpeechTTSOwnershipScope.STUDIO_PREFERENCE,
        "elevenlabs-model-select",
    ),
    *_owned(
        "kokoro",
        SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        "kokoro-browse-model-btn",
        "kokoro-browse-voices-btn",
        "kokoro-device-select",
        "kokoro-max-tokens-input",
        "kokoro-performance-switch",
        "kokoro-use-onnx-switch",
        "kokoro-voice-mixing-switch",
    ),
    *_owned(
        "kokoro",
        SpeechTTSOwnershipScope.VOICE_PROFILE_OPERATION,
        "add-voice-blend-btn",
        "export-blends-btn",
        "import-blends-btn",
        "kokoro-voice-blends-list",
    ),
    *_owned(
        "chatterbox",
        SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        "chatterbox-browse-voice-dir-btn",
        "chatterbox-candidates-input",
        "chatterbox-chunk-size-input",
        "chatterbox-crossfade-ms-input",
        "chatterbox-crossfade-switch",
        "chatterbox-device-select",
        "chatterbox-max-chunk-input",
        "chatterbox-normalize-switch",
        "chatterbox-preprocess-switch",
        "chatterbox-seed-input",
        "chatterbox-stream-chunk-input",
        "chatterbox-streaming-switch",
        "chatterbox-target-db-input",
        "chatterbox-temperature-input",
        "chatterbox-whisper-switch",
    ),
    *_owned(
        "chatterbox",
        SpeechTTSOwnershipScope.STUDIO_PREFERENCE,
        "chatterbox-cfg-weight-input",
        "chatterbox-exaggeration-input",
    ),
    *_owned(
        "higgs",
        SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        "higgs-delimiter-input",
        "higgs-device-select",
        "higgs-dtype-select",
        "higgs-flash-attn-switch",
        "higgs-language-select",
        "higgs-max-ref-duration-input",
        "higgs-max-tokens-input",
        "higgs-model-path-input",
        "higgs-multi-speaker-switch",
        "higgs-repetition-penalty-input",
        "higgs-temperature-input",
        "higgs-top-p-input",
        "higgs-track-performance-switch",
        "higgs-voice-cloning-switch",
        "higgs-voices-browse-btn",
        "higgs-voices-dir-input",
    ),
    *_owned(
        "alltalk",
        SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION,
        "alltalk-language-select",
        "alltalk-url-input",
    ),
    *_owned(
        "alltalk",
        SpeechTTSOwnershipScope.STUDIO_PREFERENCE,
        "alltalk-format-select",
        "alltalk-voice-input",
    ),
    SpeechTTSOwnershipRecord(
        control_id="audio-cpp-settings",
        owner_id="audio_cpp",
        scope=SpeechTTSOwnershipScope.RETIRED,
        reason=(
            "Legacy structural container; later owner surfaces replace its layout "
            "only after their controls exist."
        ),
    ),
    SpeechTTSOwnershipRecord(
        control_id="save-settings-btn",
        owner_id=SHARED_TTS_DEFAULTS_OWNER_ID,
        scope=SpeechTTSOwnershipScope.RETIRED,
        reason=(
            "Legacy mixed-scope save; later global and Studio owners replace it "
            "with separate persistence actions."
        ),
    ),
)


def _build_ownership_index(
    records: Iterable[SpeechTTSOwnershipRecord],
) -> Mapping[str, SpeechTTSOwnershipRecord]:
    index: dict[str, SpeechTTSOwnershipRecord] = {}
    duplicates: set[str] = set()
    for record in records:
        if type(record) is not SpeechTTSOwnershipRecord:
            raise TypeError("Speech TTS ownership inventory contains an invalid record")
        if record.control_id in index:
            duplicates.add(record.control_id)
        else:
            index[record.control_id] = record

    if duplicates:
        raise ValueError(
            f"Speech TTS controls have multiple classifications: {sorted(duplicates)}"
        )

    unknown = set(index) - ALL_SETTINGS_CONTROLS
    if unknown:
        raise ValueError(
            f"Speech TTS ownership contains unknown controls: {sorted(unknown)}"
        )

    missing = ALL_SETTINGS_CONTROLS - set(index)
    if missing:
        raise ValueError(f"Speech TTS controls are unclassified: {sorted(missing)}")

    return MappingProxyType(index)


SPEECH_TTS_OWNERSHIP_BY_CONTROL_ID = _build_ownership_index(
    SPEECH_TTS_OWNERSHIP_INVENTORY
)
"""Read-only lookup for the accepted ADR-039 built-in ownership inventory."""


def validate_speech_tts_ownership_inventory(
    records: Iterable[SpeechTTSOwnershipRecord],
) -> Mapping[str, SpeechTTSOwnershipRecord]:
    """Validate completeness and exact ADR-039 ownership for candidate records.

    Args:
        records: Candidate classification records for all current controls.

    Returns:
        A read-only lookup keyed by current control ID.

    Raises:
        TypeError: If a candidate is not an ownership record.
        ValueError: If classifications are missing, duplicated, unknown, or
            contradict the accepted owner/scope contract.
    """
    candidate = _build_ownership_index(records)
    contradictory = sorted(
        control_id
        for control_id, expected in SPEECH_TTS_OWNERSHIP_BY_CONTROL_ID.items()
        if candidate[control_id] != expected
    )
    if contradictory:
        raise ValueError(
            f"Speech TTS ownership contradicts ADR-039 for controls: {contradictory}"
        )
    return candidate
