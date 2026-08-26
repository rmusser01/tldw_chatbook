"""Immutable, explainable TTS selection resolution before adapter admission."""

from __future__ import annotations

import asyncio
import math
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import urlsplit
from uuid import UUID

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.legacy_catalogs import (
    LEGACY_DEFAULT_MODELS,
    LEGACY_DEFAULT_VOICES,
    LEGACY_REQUEST_OPTION_KEYS,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_reference_types import TTSCloneReference
from tldw_chatbook.TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot

ModelMode: TypeAlias = Literal["exact", "first_available"]
VoiceMode: TypeAlias = Literal["exact", "server_default"]
NativeCapabilityReader: TypeAlias = Callable[
    [str, str, str | None], Awaitable[TTSNativeCapabilitySnapshot]
]
TTSResolutionCode: TypeAlias = Literal[
    "catalog_unavailable",
    "invalid_selection",
    "missing_exact",
    "provider_unknown",
    "revision_incoherent",
    "unsupported_selection",
]

_MODEL_MODES = frozenset({"exact", "first_available"})
_VOICE_MODES = frozenset({"exact", "server_default"})
_RESPONSE_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
_MAX_IDENTIFIER_CHARACTERS = 512
_UNSAFE_IDENTIFIER_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_DEFAULT_PROVIDER_ID = "openai"
TTS_REQUEST_OPTION_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "audio_cpp": frozenset(),
        **{
            provider_id: frozenset(option_keys)
            for provider_id, option_keys in LEGACY_REQUEST_OPTION_KEYS.items()
        },
    }
)
"""Validated per-generation options, distinct from the persistable subset."""
_SOURCE_AXES = frozenset(
    {
        "provider_id",
        "model_mode",
        "model_id",
        "voice_mode",
        "voice_id",
        "response_format",
        "speed",
        "provider_options",
    }
)
_ERROR_AXES = _SOURCE_AXES | frozenset(
    {
        "clone_audition",
        "profile_reference",
        "provider_catalog",
        "provider_configuration",
        "studio_preferences",
    }
)
_ERROR_CODES = frozenset(
    {
        "catalog_unavailable",
        "invalid_selection",
        "missing_exact",
        "provider_unknown",
        "revision_incoherent",
        "unsupported_selection",
    }
)


class TTSSelectionSource(StrEnum):
    """The bounded owner that supplied one effective selection axis."""

    EXPLICIT = "explicit"
    CHARACTER_PROFILE = "character_profile"
    DEFAULT_PROFILE = "default_profile"
    STUDIO_DRAFT = "studio_draft"
    STUDIO_SAVED = "studio_saved"
    GLOBAL = "global"
    PROVIDER_FALLBACK = "provider_fallback"


class TTSEffectiveResolutionError(ValueError):
    """One value-free failure that prevents silent selection fallback."""

    def __init__(
        self,
        *,
        code: TTSResolutionCode,
        axis: str,
        source: TTSSelectionSource | None,
    ) -> None:
        if type(code) is not str or code not in _ERROR_CODES:
            raise ValueError("TTS resolution code is invalid")
        if type(axis) is not str or axis not in _ERROR_AXES:
            raise ValueError("TTS resolution axis is invalid")
        if source is not None and type(source) is not TTSSelectionSource:
            raise TypeError("TTS resolution source is invalid")
        self.code = cast(TTSResolutionCode, code)
        self.axis = axis
        self.source = source
        location = "provider" if source is None else source.value
        super().__init__(f"TTS {axis} {code.replace('_', ' ')} ({location})")


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {deepcopy(key): _freeze_value(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return deepcopy(value)


def _freeze_options(
    value: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("TTS provider options must be a mapping")
    return MappingProxyType(
        {deepcopy(key): _freeze_value(option) for key, option in value.items()}
    )


def _validate_nonnegative_revision(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class TTSSelectionOverrides:
    """Sparse request-local values; ``None`` means inherit the next layer."""

    provider_id: str | None = None
    model_mode: str | None = None
    model_id: str | None = None
    voice_mode: str | None = None
    voice_id: str | None = None
    response_format: str | None = None
    speed: float | None = None
    provider_options: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_options",
            _freeze_options(self.provider_options),
        )


@dataclass(frozen=True, slots=True)
class TTSCharacterProfileSelection:
    """A complete selection previously joined to authoritative authorship."""

    selection: TTSSelectionOverrides
    repository_generation: int
    profile_revision: int
    profile_id: UUID
    reference: TTSCloneReference | None = None

    def __post_init__(self) -> None:
        if type(self.selection) is not TTSSelectionOverrides:
            raise TypeError("Character TTS selection is invalid")
        _validate_nonnegative_revision(
            self.repository_generation,
            "Character TTS repository generation",
        )
        if type(self.profile_revision) is not int:
            raise TypeError("Character TTS profile revision must be an integer")
        if self.profile_revision < 1:
            raise ValueError("Character TTS profile revision must be positive")
        if type(self.profile_id) is not UUID:
            raise TypeError("Character TTS profile ID must be a UUID")
        if self.reference is not None and type(self.reference) is not TTSCloneReference:
            raise TypeError("Character TTS clone reference is invalid")


@dataclass(frozen=True, slots=True)
class TTSDefaultProfileSelection:
    """A complete selection naming the app-wide default voice profile."""

    selection: TTSSelectionOverrides
    repository_generation: int
    profile_revision: int
    profile_id: UUID
    reference: TTSCloneReference | None = None

    def __post_init__(self) -> None:
        if type(self.selection) is not TTSSelectionOverrides:
            raise TypeError("Default-profile TTS selection is invalid")
        _validate_nonnegative_revision(
            self.repository_generation,
            "Default-profile TTS repository generation",
        )
        if type(self.profile_revision) is not int:
            raise TypeError("Default-profile TTS profile revision must be an integer")
        if self.profile_revision < 1:
            raise ValueError("Default-profile TTS profile revision must be positive")
        if type(self.profile_id) is not UUID:
            raise TypeError("Default-profile TTS profile ID must be a UUID")
        if self.reference is not None and type(self.reference) is not TTSCloneReference:
            raise TypeError("Default-profile TTS clone reference is invalid")


@dataclass(frozen=True, slots=True)
class TTSStudioDraftSelection:
    """Current validated Studio controls or one non-persistent preview."""

    selection: TTSSelectionOverrides
    base_revision: int
    preview: bool = False

    def __post_init__(self) -> None:
        if type(self.selection) is not TTSSelectionOverrides:
            raise TypeError("Studio TTS draft selection is invalid")
        _validate_nonnegative_revision(
            self.base_revision,
            "Studio TTS draft base revision",
        )
        if type(self.preview) is not bool:
            raise TypeError("Studio TTS preview marker must be a boolean")


@dataclass(frozen=True, slots=True)
class TTSEffectiveSelectionRevisions:
    """Non-secret revisions frozen with one effective selection."""

    global_preferences: int
    studio_preferences: int | None
    character_repository: int | None
    character_profile: int | None
    default_profile_repository: int | None
    default_profile_revision: int | None
    provider_configuration: int
    provider_catalog: int | None
    provider_saved: int | None = None
    provider_applied: int | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative_revision(
            self.global_preferences,
            "Global TTS preference revision",
        )
        _validate_nonnegative_revision(
            self.provider_configuration,
            "TTS provider configuration revision",
        )
        for label, value in (
            ("Studio TTS preference revision", self.studio_preferences),
            ("Character TTS repository generation", self.character_repository),
            (
                "Default-profile TTS repository generation",
                self.default_profile_repository,
            ),
            ("TTS provider catalog revision", self.provider_catalog),
            ("Saved TTS provider publication generation", self.provider_saved),
            ("Applied TTS provider publication generation", self.provider_applied),
        ):
            if value is not None:
                _validate_nonnegative_revision(value, label)
        if self.character_profile is not None:
            if type(self.character_profile) is not int:
                raise TypeError("Character TTS profile revision must be an integer")
            if self.character_profile < 1:
                raise ValueError("Character TTS profile revision must be positive")
        if (self.character_repository is None) is not (self.character_profile is None):
            raise ValueError("Character TTS revisions must be recorded together")
        if self.default_profile_revision is not None:
            if type(self.default_profile_revision) is not int:
                raise TypeError(
                    "Default-profile TTS profile revision must be an integer"
                )
            if self.default_profile_revision < 1:
                raise ValueError(
                    "Default-profile TTS profile revision must be positive"
                )
        if (self.default_profile_repository is None) is not (
            self.default_profile_revision is None
        ):
            raise ValueError("Default-profile TTS revisions must be recorded together")

    @property
    def provider_active(self) -> int:
        """Return the active registry revision used for runtime freshness.

        ``provider_configuration`` remains the compatibility name for this
        registry identity. It is intentionally distinct from saved/applied
        publication generations.
        """

        return self.provider_configuration


def tts_configuration_is_active(
    service: object,
    provider_id: str,
    saved_revision: int,
) -> bool:
    """Return whether one saved provider generation is applied at runtime."""

    if (
        not isinstance(provider_id, str)
        or not provider_id
        or type(saved_revision) is not int
        or saved_revision < 1
    ):
        return False
    saved_reader = getattr(service, "saved_configuration_revision", None)
    applied_reader = getattr(service, "applied_configuration_revision", None)
    active_reader = getattr(service, "configuration_revision", None)
    if not all(
        callable(reader) for reader in (saved_reader, applied_reader, active_reader)
    ):
        return False
    try:
        saved = saved_reader(provider_id)
        applied = applied_reader(provider_id)
        active = active_reader(provider_id)
    except Exception:
        return False
    return bool(
        type(saved) is int
        and type(applied) is int
        and type(active) is int
        and active >= 0
        and saved == saved_revision
        and applied == saved_revision
    )


def _provider_publication_generation(
    provider_revision_reader: Callable[[str], int],
    provider_id: str,
    method_name: str,
) -> int | None:
    """Read publication provenance from a bound service reader when available."""

    owner = getattr(provider_revision_reader, "__self__", None)
    reader = getattr(owner, method_name, None)
    if not callable(reader):
        return None
    try:
        revision = reader(provider_id)
    except Exception:
        return None
    if type(revision) is not int or revision < 0:
        return None
    return revision


@dataclass(frozen=True, slots=True)
class TTSEffectiveSelectionSnapshot:
    """One complete, immutable, text-free TTS selection for admission."""

    provider_id: str
    model_mode: ModelMode
    model_id: str
    voice_mode: VoiceMode
    voice_id: str | None
    response_format: str
    speed: float
    provider_options: Mapping[str, object]
    sources: Mapping[str, TTSSelectionSource]
    revisions: TTSEffectiveSelectionRevisions
    provider_option_sources: Mapping[str, TTSSelectionSource] = field(
        default_factory=dict
    )
    studio_preview: bool = False

    def __post_init__(self) -> None:
        if self.provider_id not in BUILT_IN_TTS_PROVIDER_IDS:
            raise ValueError("Effective TTS provider is invalid")
        if self.model_mode not in _MODEL_MODES:
            raise ValueError("Effective TTS model mode is invalid")
        _validate_identifier(self.model_id, "model_id", None)
        if self.voice_mode not in _VOICE_MODES:
            raise ValueError("Effective TTS voice mode is invalid")
        if self.voice_mode == "exact":
            _validate_identifier(self.voice_id, "voice_id", None)
        elif self.voice_id is not None:
            raise ValueError("Server-default voice must remain omitted")
        if self.response_format not in _RESPONSE_FORMATS:
            raise ValueError("Effective TTS response format is invalid")
        speed = _validate_speed(self.speed, None)
        options = _validated_options(
            self.provider_id,
            self.provider_options,
            None,
        )
        if not isinstance(self.sources, Mapping):
            raise TypeError("Effective TTS sources must be a mapping")
        copied_sources = dict(self.sources)
        if set(copied_sources) != _SOURCE_AXES:
            raise ValueError("Effective TTS sources are incomplete")
        if not all(
            type(source) is TTSSelectionSource for source in copied_sources.values()
        ):
            raise TypeError("Effective TTS source is invalid")
        if type(self.revisions) is not TTSEffectiveSelectionRevisions:
            raise TypeError("Effective TTS revisions are invalid")
        if not isinstance(self.provider_option_sources, Mapping):
            raise TypeError("Effective TTS provider-option sources must be a mapping")
        copied_option_sources = dict(self.provider_option_sources)
        if set(copied_option_sources) != set(options):
            raise ValueError("Effective TTS provider-option sources are incomplete")
        if not all(
            type(source) is TTSSelectionSource
            for source in copied_option_sources.values()
        ):
            raise TypeError("Effective TTS provider-option source is invalid")
        if type(self.studio_preview) is not bool:
            raise TypeError("Effective TTS preview marker must be a boolean")
        if self.provider_id == "audio_cpp":
            if self.response_format != "wav" or speed != 1.0 or options:
                raise ValueError("Effective audio.cpp selection is invalid")
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "provider_options", options)
        object.__setattr__(self, "sources", MappingProxyType(copied_sources))
        object.__setattr__(
            self,
            "provider_option_sources",
            MappingProxyType(copied_option_sources),
        )


@dataclass(frozen=True, slots=True)
class _ProviderFallback:
    selection: TTSSelectionOverrides
    server_default_voice: bool = False


def _fallback_selection(
    provider_id: str,
    *,
    model_mode: str,
    model_id: str | None,
    voice_mode: str,
    voice_id: str | None,
    response_format: str,
) -> _ProviderFallback:
    return _ProviderFallback(
        TTSSelectionOverrides(
            provider_id=provider_id,
            model_mode=model_mode,
            model_id=model_id,
            voice_mode=voice_mode,
            voice_id=voice_id,
            response_format=response_format,
            speed=1.0,
            provider_options={},
        ),
        server_default_voice=voice_mode == "server_default",
    )


_PROVIDER_FALLBACKS: Mapping[str, _ProviderFallback] = MappingProxyType(
    {
        "audio_cpp": _fallback_selection(
            "audio_cpp",
            model_mode="first_available",
            model_id=None,
            voice_mode="server_default",
            voice_id=None,
            response_format="wav",
        ),
        **{
            provider_id: _fallback_selection(
                provider_id,
                model_mode="exact",
                model_id=LEGACY_DEFAULT_MODELS[provider_id],
                voice_mode="exact",
                voice_id=LEGACY_DEFAULT_VOICES[provider_id],
                response_format=(
                    "mp3" if provider_id in {"openai", "elevenlabs"} else "wav"
                ),
            )
            for provider_id in LEGACY_DEFAULT_MODELS
        },
    }
)


@dataclass(frozen=True, slots=True)
class _SelectionLayer:
    source: TTSSelectionSource
    selection: TTSSelectionOverrides
    provider_options_provider_id: str | None = None

    @property
    def owner_provider_id(self) -> str | None:
        return self.selection.provider_id


def _global_layer(preferences: TTSPreferencesSnapshot) -> _SelectionLayer:
    if type(preferences) is not TTSPreferencesSnapshot:
        raise TypeError("Global TTS preferences are invalid")
    return _SelectionLayer(
        TTSSelectionSource.GLOBAL,
        TTSSelectionOverrides(
            provider_id=preferences.provider_id,
            model_mode=preferences.model_mode,
            model_id=preferences.model_id,
            voice_mode=preferences.voice_mode,
            voice_id=preferences.voice_id,
            response_format=preferences.response_format,
            speed=preferences.speed,
        ),
    )


def _studio_saved_layer(
    preferences: StudioTTSPreferencesSnapshot,
    provider_id: str | None = None,
) -> _SelectionLayer:
    if type(preferences) is not StudioTTSPreferencesSnapshot:
        raise TypeError("Saved Studio TTS preferences are invalid")
    selection = preferences.selection
    options: Mapping[str, object] | None = None
    if provider_id is not None:
        candidate = preferences.provider_options.get(provider_id)
        if candidate is not None:
            options = candidate
    return _SelectionLayer(
        TTSSelectionSource.STUDIO_SAVED,
        TTSSelectionOverrides(
            provider_id=selection.provider_id,
            model_mode=selection.model_mode,
            model_id=selection.model_id,
            voice_mode=selection.voice_mode,
            voice_id=selection.voice_id,
            response_format=selection.response_format,
            speed=selection.speed,
            provider_options=options,
        ),
        provider_options_provider_id=(provider_id if options is not None else None),
    )


def _fallback_layer(provider_id: str) -> _SelectionLayer:
    return _SelectionLayer(
        TTSSelectionSource.PROVIDER_FALLBACK,
        _PROVIDER_FALLBACKS[provider_id].selection,
    )


def _require_complete_profile_selection(
    profile: TTSCharacterProfileSelection | TTSDefaultProfileSelection,
    *,
    source: TTSSelectionSource,
) -> None:
    """Reject an incomplete exact profile before any lower layer is consulted."""

    selection = profile.selection
    required_axes = (
        "provider_id",
        "model_mode",
        "response_format",
        "speed",
        "provider_options",
    )
    for axis in required_axes:
        if getattr(selection, axis) is None:
            raise TTSEffectiveResolutionError(
                code="invalid_selection",
                axis=axis,
                source=source,
            )
    if selection.model_mode != "exact":
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis="model_mode",
            source=source,
        )
    if selection.model_id is None:
        raise TTSEffectiveResolutionError(
            code="missing_exact",
            axis="model_id",
            source=source,
        )
    if selection.voice_id is None:
        if selection.voice_mode == "exact":
            raise TTSEffectiveResolutionError(
                code="missing_exact",
                axis="voice_id",
                source=source,
            )
        if selection.voice_mode != "server_default":
            raise TTSEffectiveResolutionError(
                code="invalid_selection",
                axis="voice_mode",
                source=source,
            )
    elif selection.voice_mode != "exact":
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis="voice_mode",
            source=source,
        )


def _require_complete_character_profile(
    profile: TTSCharacterProfileSelection,
) -> None:
    """Reject an incomplete exact character profile (thin named wrapper)."""

    _require_complete_profile_selection(
        profile, source=TTSSelectionSource.CHARACTER_PROFILE
    )


def _provider_for_layers(
    layers: tuple[_SelectionLayer, ...],
) -> tuple[str, TTSSelectionSource]:
    for layer in layers:
        candidate = layer.selection.provider_id
        if candidate is None:
            continue
        if type(candidate) is not str or candidate not in BUILT_IN_TTS_PROVIDER_IDS:
            raise TTSEffectiveResolutionError(
                code="provider_unknown",
                axis="provider_id",
                source=layer.source,
            )
        return candidate, layer.source
    return _DEFAULT_PROVIDER_ID, TTSSelectionSource.PROVIDER_FALLBACK


def _applies_to_provider(layer: _SelectionLayer, provider_id: str) -> bool:
    owner = layer.owner_provider_id
    return owner is None or owner == provider_id


def _pick(
    field_name: str,
    layers: tuple[_SelectionLayer, ...],
    provider_id: str,
) -> tuple[object | None, TTSSelectionSource | None, int | None]:
    for index, layer in enumerate(layers):
        if not _applies_to_provider(layer, provider_id):
            continue
        value = getattr(layer.selection, field_name)
        if value is not None:
            return value, layer.source, index
    return None, None, None


def _identifier_looks_like_endpoint_or_mask(value: str) -> bool:
    stripped = value.strip()
    normalized_mask = stripped.casefold().strip("*•●·_- []<>()")
    if normalized_mask in {"masked", "redacted"} or (
        len(stripped) >= 3 and set(stripped).issubset({"*", "•", "●", "·"})
    ):
        return True
    try:
        parsed = urlsplit(stripped)
    except ValueError:
        return True
    if parsed.netloc or parsed.username is not None or parsed.password is not None:
        return True
    return bool(parsed.scheme and "://" in stripped)


def _validate_identifier(
    value: object,
    axis: str,
    source: TTSSelectionSource | None,
) -> str:
    if type(value) is not str or not value or not value.strip():
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis=axis,
            source=source,
        )
    if (
        value != value.strip()
        or len(value) > _MAX_IDENTIFIER_CHARACTERS
        or any(
            unicodedata.category(character) in _UNSAFE_IDENTIFIER_CATEGORIES
            for character in value
        )
        or _identifier_looks_like_endpoint_or_mask(value)
    ):
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis=axis,
            source=source,
        )
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis=axis,
            source=source,
        ) from None
    return value


def _validate_mode(
    value: object,
    *,
    axis: str,
    allowed: frozenset[str],
    source: TTSSelectionSource | None,
) -> str:
    if type(value) is not str or value not in allowed:
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis=axis,
            source=source,
        )
    return value


def _validate_response_format(
    value: object,
    source: TTSSelectionSource | None,
) -> str:
    if type(value) is not str or value not in _RESPONSE_FORMATS:
        raise TTSEffectiveResolutionError(
            code="unsupported_selection",
            axis="response_format",
            source=source,
        )
    return value


def _validate_speed(
    value: object,
    source: TTSSelectionSource | None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis="speed",
            source=source,
        )
    speed = float(value)
    if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis="speed",
            source=source,
        )
    return speed


def _validated_options(
    provider_id: str,
    value: object,
    source: TTSSelectionSource | None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis="provider_options",
            source=source,
        )
    allowed = TTS_REQUEST_OPTION_KEYS[provider_id]
    normalized: dict[str, object] = {}
    for key, option in value.items():
        if type(key) is not str or key not in allowed:
            raise TTSEffectiveResolutionError(
                code="unsupported_selection",
                axis="provider_options",
                source=source,
            )
        if key in {
            "use_speaker_boost",
            "use_onnx",
            "validate_with_whisper",
        }:
            if type(option) is not bool:
                raise TTSEffectiveResolutionError(
                    code="invalid_selection",
                    axis="provider_options",
                    source=source,
                )
            normalized[key] = option
            continue
        if key == "language":
            normalized[key] = _validate_identifier(
                option,
                "provider_options",
                source,
            )
            continue
        if key == "num_candidates":
            if type(option) is not int or not 1 <= option <= 5:
                raise TTSEffectiveResolutionError(
                    code="invalid_selection",
                    axis="provider_options",
                    source=source,
                )
            normalized[key] = option
            continue
        if isinstance(option, bool) or not isinstance(option, Real):
            raise TTSEffectiveResolutionError(
                code="invalid_selection",
                axis="provider_options",
                source=source,
            )
        number = float(option)
        if not math.isfinite(number):
            raise TTSEffectiveResolutionError(
                code="invalid_selection",
                axis="provider_options",
                source=source,
            )
        minimum, maximum = (
            (0.0, 2.0)
            if key == "temperature"
            else (1.0, 10.0)
            if key == "repetition_penalty"
            else (0.0, 1.0)
        )
        if not minimum <= number <= maximum:
            raise TTSEffectiveResolutionError(
                code="invalid_selection",
                axis="provider_options",
                source=source,
            )
        normalized[key] = number
    return MappingProxyType(normalized)


def _resolve_options(
    *,
    provider_id: str,
    layers: tuple[_SelectionLayer, ...],
) -> tuple[
    Mapping[str, object],
    TTSSelectionSource,
    Mapping[str, TTSSelectionSource],
]:
    """Merge sparse option keys until an explicit empty layer clears inheritance."""

    resolved: dict[str, object] = {}
    option_sources: dict[str, TTSSelectionSource] = {}
    axis_source: TTSSelectionSource | None = None
    allowed = TTS_REQUEST_OPTION_KEYS[provider_id]
    for layer in layers:
        options_owner = layer.provider_options_provider_id
        if options_owner is None:
            applies = _applies_to_provider(layer, provider_id)
        else:
            applies = options_owner == provider_id
        if not applies:
            continue
        raw = layer.selection.provider_options
        if raw is None:
            continue
        axis_source = axis_source or layer.source
        validated = _validated_options(provider_id, raw, layer.source)
        if not validated:
            break
        for key, value in validated.items():
            if key not in resolved:
                resolved[key] = value
                option_sources[key] = layer.source
        if set(resolved) == allowed:
            break

    assert axis_source is not None
    return (
        MappingProxyType(resolved),
        axis_source,
        MappingProxyType(option_sources),
    )


async def _resolve_model(
    *,
    provider_id: str,
    layers: tuple[_SelectionLayer, ...],
    catalog_reader: Callable[[str], Awaitable[TTSProviderCatalog]],
) -> tuple[
    ModelMode,
    str,
    TTSSelectionSource,
    TTSSelectionSource,
    int | None,
    TTSModelInfo | None,
]:
    raw_mode, mode_source, mode_index = _pick("model_mode", layers, provider_id)
    assert mode_source is not None and mode_index is not None
    mode = cast(
        ModelMode,
        _validate_mode(
            raw_mode,
            axis="model_mode",
            allowed=_MODEL_MODES,
            source=mode_source,
        ),
    )
    raw_model, model_source, model_index = _pick("model_id", layers, provider_id)

    if mode == "exact":
        if raw_model is None or model_source is None:
            raise TTSEffectiveResolutionError(
                code="missing_exact",
                axis="model_id",
                source=mode_source,
            )
        return (
            mode,
            _validate_identifier(raw_model, "model_id", model_source),
            mode_source,
            model_source,
            None,
            None,
        )

    if raw_model is not None and model_index is not None and model_index <= mode_index:
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis="model_id",
            source=model_source,
        )
    try:
        catalog = await catalog_reader(provider_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise TTSEffectiveResolutionError(
            code="catalog_unavailable",
            axis="provider_catalog",
            source=mode_source,
        ) from None
    if (
        type(catalog) is not TTSProviderCatalog
        or catalog.provider_id != provider_id
        or type(catalog.revision) is not int
        or catalog.revision < 0
        or type(catalog.health) is not ProviderHealth
        or catalog.health.state != "available"
        or not catalog.health.fresh
        or not catalog.models
    ):
        raise TTSEffectiveResolutionError(
            code=(
                "revision_incoherent"
                if type(catalog) is TTSProviderCatalog
                and catalog.provider_id != provider_id
                else "catalog_unavailable"
            ),
            axis="provider_catalog",
            source=mode_source,
        )
    selected = catalog.models[0]
    if type(selected) is not TTSModelInfo:
        raise TTSEffectiveResolutionError(
            code="catalog_unavailable",
            axis="provider_catalog",
            source=mode_source,
        )
    return (
        mode,
        _validate_identifier(selected.model_id, "model_id", mode_source),
        mode_source,
        mode_source,
        catalog.revision,
        selected,
    )


def _resolve_voice(
    *,
    provider_id: str,
    layers: tuple[_SelectionLayer, ...],
) -> tuple[VoiceMode, str | None, TTSSelectionSource, TTSSelectionSource]:
    raw_mode, mode_source, mode_index = _pick("voice_mode", layers, provider_id)
    assert mode_source is not None and mode_index is not None
    mode = cast(
        VoiceMode,
        _validate_mode(
            raw_mode,
            axis="voice_mode",
            allowed=_VOICE_MODES,
            source=mode_source,
        ),
    )
    raw_voice, voice_source, voice_index = _pick("voice_id", layers, provider_id)
    if mode == "exact":
        if raw_voice is None or voice_source is None:
            raise TTSEffectiveResolutionError(
                code="missing_exact",
                axis="voice_id",
                source=mode_source,
            )
        return (
            mode,
            _validate_identifier(raw_voice, "voice_id", voice_source),
            mode_source,
            voice_source,
        )

    if raw_voice is not None and voice_index is not None and voice_index <= mode_index:
        raise TTSEffectiveResolutionError(
            code="invalid_selection",
            axis="voice_id",
            source=voice_source,
        )
    if not _PROVIDER_FALLBACKS[provider_id].server_default_voice:
        raise TTSEffectiveResolutionError(
            code="unsupported_selection",
            axis="voice_mode",
            source=mode_source,
        )
    return mode, None, mode_source, mode_source


async def _validate_exact_native_capability(
    *,
    provider_id: str,
    model_id: str,
    model_source: TTSSelectionSource,
    voice_mode: VoiceMode,
    voice_id: str | None,
    voice_source: TTSSelectionSource,
    native_capability_reader: NativeCapabilityReader | None,
) -> tuple[int, int, TTSModelInfo]:
    if native_capability_reader is None:
        raise TTSEffectiveResolutionError(
            code="catalog_unavailable",
            axis="provider_catalog",
            source=model_source,
        )
    try:
        snapshot = await native_capability_reader(
            provider_id,
            model_id,
            voice_id if voice_mode == "exact" else None,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        raise TTSEffectiveResolutionError(
            code="catalog_unavailable",
            axis="provider_catalog",
            source=model_source,
        ) from None
    if (
        type(snapshot) is not TTSNativeCapabilitySnapshot
        or snapshot.provider_id != provider_id
        or snapshot.state != "complete"
        or type(snapshot.configuration_revision) is not int
        or snapshot.configuration_revision < 0
        or type(snapshot.catalog) is not TTSProviderCatalog
    ):
        raise TTSEffectiveResolutionError(
            code="revision_incoherent",
            axis="provider_catalog",
            source=model_source,
        )
    catalog = snapshot.catalog
    if (
        catalog.provider_id != provider_id
        or type(catalog.revision) is not int
        or catalog.revision < 0
        or type(catalog.health) is not ProviderHealth
        or catalog.health.state != "available"
        or not catalog.health.fresh
        or catalog.approximate
    ):
        raise TTSEffectiveResolutionError(
            code="catalog_unavailable",
            axis="provider_catalog",
            source=model_source,
        )
    model = next(
        (
            candidate
            for candidate in catalog.models
            if type(candidate) is TTSModelInfo and candidate.model_id == model_id
        ),
        None,
    )
    if model is None:
        raise TTSEffectiveResolutionError(
            code="missing_exact",
            axis="model_id",
            source=model_source,
        )
    if voice_mode == "exact":
        result = snapshot.voice_results.get(model_id)
        if (
            type(result) is not TTSVoiceDiscoveryResult
            or result.provider_id != provider_id
            or result.model_id != model_id
            or result.catalog_revision != catalog.revision
            or result.state != "complete"
        ):
            raise TTSEffectiveResolutionError(
                code="catalog_unavailable",
                axis="provider_catalog",
                source=voice_source,
            )
        assert voice_id is not None
        if voice_id not in result.voices:
            raise TTSEffectiveResolutionError(
                code="missing_exact",
                axis="voice_id",
                source=voice_source,
            )
    return catalog.revision, snapshot.configuration_revision, model


async def _resolve_layers(
    *,
    layers_without_fallback: tuple[_SelectionLayer, ...],
    global_preferences_revision: int,
    studio_preferences_revision: int | None,
    character_profile: TTSCharacterProfileSelection | None,
    default_profile: TTSDefaultProfileSelection | None,
    provider_revision_reader: Callable[[str], int],
    catalog_reader: Callable[[str], Awaitable[TTSProviderCatalog]],
    native_capability_reader: NativeCapabilityReader | None,
    studio_preview: bool,
) -> TTSEffectiveSelectionSnapshot:
    global_revision = _validate_nonnegative_revision(
        global_preferences_revision,
        "Global TTS preference revision",
    )
    provider_id, provider_source = _provider_for_layers(layers_without_fallback)
    layers = (*layers_without_fallback, _fallback_layer(provider_id))

    (
        model_mode,
        model_id,
        model_mode_source,
        model_id_source,
        catalog_revision,
        catalog_model,
    ) = await _resolve_model(
        provider_id=provider_id,
        layers=layers,
        catalog_reader=catalog_reader,
    )
    voice_mode, voice_id, voice_mode_source, voice_id_source = _resolve_voice(
        provider_id=provider_id,
        layers=layers,
    )

    raw_format, format_source, _ = _pick("response_format", layers, provider_id)
    assert format_source is not None
    response_format = _validate_response_format(raw_format, format_source)
    raw_speed, speed_source, _ = _pick("speed", layers, provider_id)
    assert speed_source is not None
    speed = _validate_speed(raw_speed, speed_source)
    options, options_source, option_sources = _resolve_options(
        provider_id=provider_id,
        layers=layers,
    )

    capability_provider_revision: int | None = None
    if provider_id == "audio_cpp":
        if response_format != "wav":
            raise TTSEffectiveResolutionError(
                code="unsupported_selection",
                axis="response_format",
                source=format_source,
            )
        if speed != 1.0:
            raise TTSEffectiveResolutionError(
                code="unsupported_selection",
                axis="speed",
                source=speed_source,
            )
        if options:
            raise TTSEffectiveResolutionError(
                code="unsupported_selection",
                axis="provider_options",
                source=options_source,
            )
        if model_mode == "exact":
            (
                catalog_revision,
                capability_provider_revision,
                catalog_model,
            ) = await _validate_exact_native_capability(
                provider_id=provider_id,
                model_id=model_id,
                model_source=model_id_source,
                voice_mode=voice_mode,
                voice_id=voice_id,
                voice_source=voice_id_source,
                native_capability_reader=native_capability_reader,
            )
    if catalog_model is not None:
        if response_format not in catalog_model.formats:
            raise TTSEffectiveResolutionError(
                code="unsupported_selection",
                axis="response_format",
                source=format_source,
            )
        if not catalog_model.supports_speed and speed != 1.0:
            raise TTSEffectiveResolutionError(
                code="unsupported_selection",
                axis="speed",
                source=speed_source,
            )
        if voice_mode == "server_default" and not (
            catalog_model.omit_voice_uses_server_default
        ):
            raise TTSEffectiveResolutionError(
                code="unsupported_selection",
                axis="voice_mode",
                source=voice_mode_source,
            )

    try:
        provider_revision = provider_revision_reader(provider_id)
    except Exception:
        raise TTSEffectiveResolutionError(
            code="revision_incoherent",
            axis="provider_configuration",
            source=provider_source,
        ) from None
    if type(provider_revision) is not int or provider_revision < 0:
        raise TTSEffectiveResolutionError(
            code="revision_incoherent",
            axis="provider_configuration",
            source=provider_source,
        )
    if (
        capability_provider_revision is not None
        and provider_revision != capability_provider_revision
    ):
        raise TTSEffectiveResolutionError(
            code="revision_incoherent",
            axis="provider_configuration",
            source=provider_source,
        )

    all_axis_sources = (
        provider_source,
        model_mode_source,
        model_id_source,
        voice_mode_source,
        voice_id_source,
        format_source,
        speed_source,
        options_source,
        *option_sources.values(),
    )
    uses_character = any(
        source is TTSSelectionSource.CHARACTER_PROFILE for source in all_axis_sources
    )
    uses_default_profile = any(
        source is TTSSelectionSource.DEFAULT_PROFILE for source in all_axis_sources
    )
    revisions = TTSEffectiveSelectionRevisions(
        global_preferences=global_revision,
        studio_preferences=studio_preferences_revision,
        character_repository=(
            character_profile.repository_generation
            if uses_character and character_profile is not None
            else None
        ),
        character_profile=(
            character_profile.profile_revision
            if uses_character and character_profile is not None
            else None
        ),
        default_profile_repository=(
            default_profile.repository_generation
            if uses_default_profile and default_profile is not None
            else None
        ),
        default_profile_revision=(
            default_profile.profile_revision
            if uses_default_profile and default_profile is not None
            else None
        ),
        provider_configuration=provider_revision,
        provider_catalog=catalog_revision,
        provider_saved=_provider_publication_generation(
            provider_revision_reader,
            provider_id,
            "saved_configuration_revision",
        ),
        provider_applied=_provider_publication_generation(
            provider_revision_reader,
            provider_id,
            "applied_configuration_revision",
        ),
    )
    return TTSEffectiveSelectionSnapshot(
        provider_id=provider_id,
        model_mode=model_mode,
        model_id=model_id,
        voice_mode=voice_mode,
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
        provider_options=options,
        sources={
            "provider_id": provider_source,
            "model_mode": model_mode_source,
            "model_id": model_id_source,
            "voice_mode": voice_mode_source,
            "voice_id": voice_id_source,
            "response_format": format_source,
            "speed": speed_source,
            "provider_options": options_source,
        },
        revisions=revisions,
        provider_option_sources=option_sources,
        studio_preview=studio_preview,
    )


class TTSEffectiveSettingsResolver:
    """Resolve immutable owner snapshots without reading or mutating storage."""

    def project_provider(
        self,
        *,
        global_preferences: TTSPreferencesSnapshot | None,
        explicit: TTSSelectionOverrides | None = None,
        character_profile: TTSCharacterProfileSelection | None = None,
        default_profile: TTSDefaultProfileSelection | None = None,
        studio_preferences: StudioTTSPreferencesSnapshot | None = None,
        studio_draft: TTSStudioDraftSelection | None = None,
    ) -> str:
        """Synchronously project the provider from the canonical layer order."""
        studio_request = studio_preferences is not None or studio_draft is not None
        if studio_request:
            if (
                explicit is not None
                or character_profile is not None
                or default_profile is not None
            ):
                raise TypeError("Studio TTS resolution cannot use non-Studio layers")
            if studio_preferences is None:
                raise TypeError("Studio TTS resolution requires saved preferences")
            layers, _preview = self._studio_layers(
                studio_preferences=studio_preferences,
                studio_draft=studio_draft,
                global_preferences=global_preferences,
            )
        else:
            layers = self._non_studio_layers(
                explicit=explicit,
                character_profile=character_profile,
                default_profile=default_profile,
                global_preferences=global_preferences,
            )
        provider_id, _source = _provider_for_layers(layers)
        return provider_id

    @staticmethod
    def _non_studio_layers(
        *,
        explicit: TTSSelectionOverrides | None,
        character_profile: TTSCharacterProfileSelection | None,
        default_profile: TTSDefaultProfileSelection | None,
        global_preferences: TTSPreferencesSnapshot | None,
    ) -> tuple[_SelectionLayer, ...]:
        if explicit is not None and type(explicit) is not TTSSelectionOverrides:
            raise TypeError("Explicit TTS selection is invalid")
        if character_profile is not None and (
            type(character_profile) is not TTSCharacterProfileSelection
        ):
            raise TypeError("Character TTS profile selection is invalid")
        if character_profile is not None:
            _require_complete_character_profile(character_profile)
        if default_profile is not None and (
            type(default_profile) is not TTSDefaultProfileSelection
        ):
            raise TypeError("Default-profile TTS selection is invalid")
        if default_profile is not None:
            _require_complete_profile_selection(
                default_profile,
                source=TTSSelectionSource.DEFAULT_PROFILE,
            )
        if global_preferences is not None and (
            type(global_preferences) is not TTSPreferencesSnapshot
        ):
            raise TypeError("Global TTS preferences are invalid")
        layers: list[_SelectionLayer] = []
        if explicit is not None:
            layers.append(_SelectionLayer(TTSSelectionSource.EXPLICIT, explicit))
        if character_profile is not None:
            layers.append(
                _SelectionLayer(
                    TTSSelectionSource.CHARACTER_PROFILE,
                    character_profile.selection,
                )
            )
        if default_profile is not None:
            layers.append(
                _SelectionLayer(
                    TTSSelectionSource.DEFAULT_PROFILE,
                    default_profile.selection,
                )
            )
        if global_preferences is not None:
            layers.append(_global_layer(global_preferences))
        return tuple(layers)

    @staticmethod
    def _studio_layers(
        *,
        studio_preferences: StudioTTSPreferencesSnapshot,
        studio_draft: TTSStudioDraftSelection | None,
        global_preferences: TTSPreferencesSnapshot | None,
    ) -> tuple[tuple[_SelectionLayer, ...], bool]:
        if type(studio_preferences) is not StudioTTSPreferencesSnapshot:
            raise TypeError("Saved Studio TTS preferences are invalid")
        if global_preferences is not None and (
            type(global_preferences) is not TTSPreferencesSnapshot
        ):
            raise TypeError("Global TTS preferences are invalid")
        if studio_draft is not None:
            if type(studio_draft) is not TTSStudioDraftSelection:
                raise TypeError("Studio TTS draft is invalid")
            if studio_draft.base_revision != studio_preferences.revision:
                raise TTSEffectiveResolutionError(
                    code="revision_incoherent",
                    axis="studio_preferences",
                    source=TTSSelectionSource.STUDIO_DRAFT,
                )

        preliminary: list[_SelectionLayer] = []
        if studio_draft is not None:
            preliminary.append(
                _SelectionLayer(
                    TTSSelectionSource.STUDIO_DRAFT,
                    studio_draft.selection,
                )
            )
        preliminary.append(_studio_saved_layer(studio_preferences))
        if global_preferences is not None:
            preliminary.append(_global_layer(global_preferences))
        provider_id, _ = _provider_for_layers(tuple(preliminary))

        layers: list[_SelectionLayer] = []
        if studio_draft is not None:
            layers.append(
                _SelectionLayer(
                    TTSSelectionSource.STUDIO_DRAFT,
                    studio_draft.selection,
                )
            )
        layers.append(_studio_saved_layer(studio_preferences, provider_id))
        if global_preferences is not None:
            layers.append(_global_layer(global_preferences))
        return tuple(layers), bool(studio_draft and studio_draft.preview)

    async def resolve_non_studio(
        self,
        *,
        global_preferences: TTSPreferencesSnapshot | None,
        global_preferences_revision: int,
        provider_revision_reader: Callable[[str], int],
        catalog_reader: Callable[[str], Awaitable[TTSProviderCatalog]],
        native_capability_reader: NativeCapabilityReader | None = None,
        explicit: TTSSelectionOverrides | None = None,
        character_profile: TTSCharacterProfileSelection | None = None,
        default_profile: TTSDefaultProfileSelection | None = None,
    ) -> TTSEffectiveSelectionSnapshot:
        """Resolve explicit, character, default-profile, global, and fallback layers.

        Args:
            global_preferences: Persisted global selection, when configured.
            global_preferences_revision: Revision of the global selection snapshot.
            provider_revision_reader: Reader for current provider configuration revisions.
            catalog_reader: Asynchronous reader for provider model and voice catalogs.
            native_capability_reader: Optional reader for observed native capabilities.
            explicit: Optional request-scoped selection overrides.
            character_profile: Optional authoritative character-owned selection.
            default_profile: Optional app-wide default-voice profile selection,
                consulted only when no character profile supplies an axis.

        Returns:
            The immutable effective selection and its provenance.

        Raises:
            TypeError: If a supplied selection object has an invalid type.
            ValueError: If an authoritative character or default-profile selection
                is incomplete.
            TTSEffectiveResolutionError: If the layers cannot produce a valid selection.
        """

        layers = self._non_studio_layers(
            explicit=explicit,
            character_profile=character_profile,
            default_profile=default_profile,
            global_preferences=global_preferences,
        )
        return await _resolve_layers(
            layers_without_fallback=layers,
            global_preferences_revision=global_preferences_revision,
            studio_preferences_revision=None,
            character_profile=character_profile,
            default_profile=default_profile,
            provider_revision_reader=provider_revision_reader,
            catalog_reader=catalog_reader,
            native_capability_reader=native_capability_reader,
            studio_preview=False,
        )

    async def resolve_studio(
        self,
        *,
        studio_preferences: StudioTTSPreferencesSnapshot,
        global_preferences: TTSPreferencesSnapshot | None,
        global_preferences_revision: int,
        provider_revision_reader: Callable[[str], int],
        catalog_reader: Callable[[str], Awaitable[TTSProviderCatalog]],
        native_capability_reader: NativeCapabilityReader | None = None,
        studio_draft: TTSStudioDraftSelection | None = None,
    ) -> TTSEffectiveSelectionSnapshot:
        """Resolve Studio draft/preview, saved, global, and fallback layers."""

        layers, studio_preview = self._studio_layers(
            studio_preferences=studio_preferences,
            studio_draft=studio_draft,
            global_preferences=global_preferences,
        )
        return await _resolve_layers(
            layers_without_fallback=layers,
            global_preferences_revision=global_preferences_revision,
            studio_preferences_revision=studio_preferences.revision,
            character_profile=None,
            default_profile=None,
            provider_revision_reader=provider_revision_reader,
            catalog_reader=catalog_reader,
            native_capability_reader=native_capability_reader,
            studio_preview=studio_preview,
        )


__all__ = [
    "TTSCharacterProfileSelection",
    "TTSDefaultProfileSelection",
    "TTSEffectiveResolutionError",
    "TTSEffectiveSelectionRevisions",
    "TTSEffectiveSelectionSnapshot",
    "TTSEffectiveSettingsResolver",
    "TTSSelectionOverrides",
    "TTSSelectionSource",
    "TTSStudioDraftSelection",
    "TTS_REQUEST_OPTION_KEYS",
    "tts_configuration_is_active",
]
