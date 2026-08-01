"""Versioned, sparse persistence for Speech Studio TTS preferences."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from numbers import Real
from types import MappingProxyType
from typing import Any

from loguru import logger

from tldw_chatbook import config as config_module
from tldw_chatbook.TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS


STUDIO_TTS_SCHEMA_VERSION = 1
_STUDIO_SECTION = "speech_studio"
_MAX_IDENTIFIER_CHARACTERS = 512
_MODEL_MODES = frozenset({"exact", "first_available"})
_VOICE_MODES = frozenset({"exact", "server_default"})
_RESPONSE_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
_SELECTION_FIELDS = (
    "provider_id",
    "model_mode",
    "model_id",
    "voice_mode",
    "voice_id",
    "response_format",
    "speed",
)
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "revision", "selection", "provider_options"}
)

STUDIO_TTS_PROVIDER_OPTION_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "audio_cpp": frozenset(),
        "openai": frozenset(),
        "elevenlabs": frozenset(),
        "kokoro": frozenset(),
        "chatterbox": frozenset({"exaggeration", "cfg_weight"}),
        "higgs": frozenset(),
        "alltalk": frozenset(),
    }
)
"""Request-local option keys admitted by the TASK-1692 ownership contract."""

_LEGACY_DEFAULTS: Mapping[str, object] = MappingProxyType(
    {
        "ELEVENLABS_DEFAULT_MODEL": "eleven_multilingual_v2",
        "ALLTALK_TTS_VOICE_DEFAULT": "female_01.wav",
        "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT": "wav",
        "CHATTERBOX_EXAGGERATION": 0.5,
        "CHATTERBOX_CFG_WEIGHT": 0.5,
    }
)


class StudioTTSLoadState(StrEnum):
    """Bounded outcomes for a Studio preference read."""

    MISSING = "missing"
    LOADED = "loaded"
    MIGRATED = "migrated"
    RECOVERED = "recovered"
    CORRUPT = "corrupt"
    MIGRATION_FAILED = "migration_failed"


class StudioTTSWriteStatus(StrEnum):
    """Bounded outcomes for an atomic Studio preference write."""

    SAVED = "saved"
    UNCHANGED = "unchanged"
    CONFLICT = "conflict"
    FAILED = "failed"
    SAVED_CACHE_RELOAD_FAILED = "saved_cache_reload_failed"


def _require_optional_identifier(name: str, value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if len(value) > _MAX_IDENTIFIER_CHARACTERS:
        raise ValueError(f"{name} exceeds the supported length")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains control characters")
    # Exact selection IDs are opaque (CFG-011). Reject only explicit rendered
    # mask markers here; endpoint and runtime-path *fields* are excluded by the
    # schema rather than inferred from an otherwise valid identifier's shape.
    stripped = value.strip()
    normalized_mask = stripped.casefold().strip("*•●·_- []<>()")
    if normalized_mask in {"masked", "redacted"} or (
        len(stripped) >= 3 and set(stripped).issubset({"*", "•", "●", "·"})
    ):
        raise ValueError(f"{name} cannot be a masked placeholder")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError(f"{name} is invalid") from None
    return value


def _require_optional_mode(
    name: str,
    value: object,
    allowed: frozenset[str],
) -> str | None:
    if value is None:
        return None
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{name} is not a supported {name.replace('_', ' ')}")
    return value


def _require_optional_speed(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("speed must be a number")
    speed = float(value)
    if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        raise ValueError("speed must be between 0.25 and 4.0")
    return speed


def _require_option_value(provider_id: str, key: str, value: object) -> float:
    if provider_id != "chatterbox" or key not in {
        "exaggeration",
        "cfg_weight",
    }:
        raise ValueError("Studio provider option is unsupported")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{key} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{key} must be between 0.0 and 1.0")
    return normalized


@dataclass(frozen=True, slots=True)
class StudioTTSSelectionOverrides:
    """Optional Studio-only overrides; ``None`` means inherit from global."""

    provider_id: str | None = None
    model_mode: str | None = None
    model_id: str | None = None
    voice_mode: str | None = None
    voice_id: str | None = None
    response_format: str | None = None
    speed: float | None = None

    def __post_init__(self) -> None:
        provider_id = self.provider_id
        if provider_id is not None:
            if type(provider_id) is not str:
                raise TypeError("provider must be a string")
            if provider_id not in BUILT_IN_TTS_PROVIDER_IDS:
                raise ValueError("provider is not a built-in TTS provider")
        model_mode = _require_optional_mode(
            "model_mode",
            self.model_mode,
            _MODEL_MODES,
        )
        voice_mode = _require_optional_mode(
            "voice_mode",
            self.voice_mode,
            _VOICE_MODES,
        )
        model_id = _require_optional_identifier("model", self.model_id)
        voice_id = _require_optional_identifier("voice", self.voice_id)
        response_format = self.response_format
        if response_format is not None:
            if (
                type(response_format) is not str
                or response_format not in _RESPONSE_FORMATS
            ):
                raise ValueError("format is not supported")
        speed = _require_optional_speed(self.speed)

        if provider_id == "audio_cpp":
            if response_format not in (None, "wav"):
                raise ValueError("audio.cpp format must be wav")
            if speed not in (None, 1.0):
                raise ValueError("audio.cpp speed must be exactly 1.0")
        if model_mode == "first_available" and model_id is not None:
            raise ValueError("first_available model mode cannot persist a model ID")
        if voice_mode == "server_default" and voice_id is not None:
            raise ValueError("server_default voice mode cannot persist a voice ID")

        object.__setattr__(self, "model_mode", model_mode)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "voice_mode", voice_mode)
        object.__setattr__(self, "voice_id", voice_id)
        object.__setattr__(self, "speed", speed)

    def to_mapping(self) -> dict[str, object]:
        """Return only explicitly overridden selection axes."""

        return {
            name: value
            for name in _SELECTION_FIELDS
            if (value := getattr(self, name)) is not None
        }


def _freeze_provider_options(
    provider_options: Mapping[str, Mapping[str, object]],
) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(provider_options, Mapping):
        raise TypeError("Studio provider options must be a mapping")
    frozen: dict[str, Mapping[str, float]] = {}
    for provider_id, raw_options in provider_options.items():
        if type(provider_id) is not str or provider_id not in BUILT_IN_TTS_PROVIDER_IDS:
            raise ValueError("Studio provider option owner is unknown")
        if not isinstance(raw_options, Mapping):
            raise TypeError("Studio provider options must be mappings")
        allowed = STUDIO_TTS_PROVIDER_OPTION_KEYS[provider_id]
        normalized: dict[str, float] = {}
        for key, value in raw_options.items():
            if type(key) is not str or key not in allowed:
                raise ValueError("Studio provider option is unsupported")
            normalized[key] = _require_option_value(provider_id, key, value)
        if normalized:
            frozen[provider_id] = MappingProxyType(normalized)
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class StudioTTSPreferencesSnapshot:
    """One immutable sparse Studio preference snapshot."""

    schema_version: int = STUDIO_TTS_SCHEMA_VERSION
    revision: int = 0
    selection: StudioTTSSelectionOverrides = field(
        default_factory=StudioTTSSelectionOverrides
    )
    provider_options: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != STUDIO_TTS_SCHEMA_VERSION
        ):
            raise ValueError("Studio TTS schema version is unsupported")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("Studio TTS revision must be nonnegative")
        if type(self.selection) is not StudioTTSSelectionOverrides:
            raise TypeError("Studio TTS selection is invalid")
        frozen = _freeze_provider_options(self.provider_options)
        if self.selection.provider_id == "audio_cpp" and frozen.get("audio_cpp"):
            raise ValueError("audio.cpp does not accept Studio provider options")
        object.__setattr__(self, "provider_options", frozen)

    def section_for_revision(self, revision: int) -> dict[str, object]:
        """Serialize a complete sparse section at an exact next revision."""

        if type(revision) is not int or revision != self.revision + 1:
            raise ValueError("Studio TTS replacement revision must advance by one")
        section: dict[str, object] = {
            "schema_version": STUDIO_TTS_SCHEMA_VERSION,
            "revision": revision,
        }
        selection = self.selection.to_mapping()
        if selection:
            section["selection"] = selection
        options = {
            provider_id: dict(values)
            for provider_id, values in self.provider_options.items()
            if values
        }
        if options:
            section["provider_options"] = options
        return section


@dataclass(frozen=True, slots=True)
class StudioTTSLoadResult:
    """A parsed Studio snapshot plus safe field-name diagnostics."""

    snapshot: StudioTTSPreferencesSnapshot
    state: StudioTTSLoadState
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StudioTTSWriteResult:
    """Outcome of one Studio-only compare-before-publish operation."""

    status: StudioTTSWriteStatus
    snapshot: StudioTTSPreferencesSnapshot | None


def _single_selection_value(name: str, value: object) -> object:
    candidate = StudioTTSSelectionOverrides(**{name: value})
    return getattr(candidate, name)


def _record_issue(issues: list[str], path: str) -> None:
    """Append one bounded diagnostic path at most once."""

    if path not in issues:
        issues.append(path)


def _parse_selection(
    raw: object,
    issues: list[str],
) -> StudioTTSSelectionOverrides:
    if raw is None:
        return StudioTTSSelectionOverrides()
    if not isinstance(raw, Mapping):
        _record_issue(issues, "speech_studio.selection")
        return StudioTTSSelectionOverrides()

    accepted: dict[str, object] = {}
    for key, value in raw.items():
        if type(key) is not str or key not in _SELECTION_FIELDS:
            _record_issue(issues, "speech_studio.selection.unknown_field")
            continue
        issue_path = f"speech_studio.selection.{key}"
        try:
            accepted[key] = _single_selection_value(key, value)
        except (TypeError, ValueError):
            _record_issue(issues, issue_path)

    if accepted.get("provider_id") == "audio_cpp":
        if accepted.get("response_format") not in (None, "wav"):
            _record_issue(issues, "speech_studio.selection.response_format")
            accepted.pop("response_format", None)
        if accepted.get("speed") not in (None, 1.0):
            _record_issue(issues, "speech_studio.selection.speed")
            accepted.pop("speed", None)
    if (
        accepted.get("model_mode") == "first_available"
        and accepted.get("model_id") is not None
    ):
        _record_issue(issues, "speech_studio.selection.model_id")
        accepted.pop("model_id", None)
    if (
        accepted.get("voice_mode") == "server_default"
        and accepted.get("voice_id") is not None
    ):
        _record_issue(issues, "speech_studio.selection.voice_id")
        accepted.pop("voice_id", None)
    return StudioTTSSelectionOverrides(**accepted)


def _parse_provider_options(
    raw: object,
    issues: list[str],
) -> dict[str, dict[str, float]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _record_issue(issues, "speech_studio.provider_options")
        return {}

    accepted: dict[str, dict[str, float]] = {}
    for provider_id, raw_options in raw.items():
        if type(provider_id) is not str or provider_id not in BUILT_IN_TTS_PROVIDER_IDS:
            _record_issue(issues, "speech_studio.provider_options.unknown_provider")
            continue
        provider_path = f"speech_studio.provider_options.{provider_id}"
        if not isinstance(raw_options, Mapping):
            _record_issue(issues, provider_path)
            continue
        allowed = STUDIO_TTS_PROVIDER_OPTION_KEYS[provider_id]
        options: dict[str, float] = {}
        for key, value in raw_options.items():
            if type(key) is not str or key not in allowed:
                _record_issue(issues, f"{provider_path}.unknown_option")
                continue
            option_path = f"{provider_path}.{key}"
            try:
                options[key] = _require_option_value(provider_id, key, value)
            except (TypeError, ValueError):
                _record_issue(issues, option_path)
        if options:
            accepted[provider_id] = options
    return accepted


def _parse_studio_section(raw: object) -> StudioTTSLoadResult:
    if not isinstance(raw, Mapping):
        return StudioTTSLoadResult(
            StudioTTSPreferencesSnapshot(),
            StudioTTSLoadState.CORRUPT,
            ("speech_studio",),
        )

    raw_revision = raw.get("revision", 0)
    revision = raw_revision if type(raw_revision) is int and raw_revision >= 0 else 0
    raw_schema_version = raw.get("schema_version")
    schema_is_valid = (
        type(raw_schema_version) is int
        and raw_schema_version == STUDIO_TTS_SCHEMA_VERSION
    )
    if not schema_is_valid or (type(raw_revision) is not int or raw_revision < 0):
        return StudioTTSLoadResult(
            StudioTTSPreferencesSnapshot(revision=revision),
            StudioTTSLoadState.CORRUPT,
            (
                "speech_studio.schema_version"
                if not schema_is_valid
                else "speech_studio.revision",
            ),
        )

    issues: list[str] = []
    for key in raw:
        if type(key) is not str or key not in _TOP_LEVEL_FIELDS:
            _record_issue(issues, "speech_studio.unknown_field")
    selection = _parse_selection(raw.get("selection"), issues)
    provider_options = _parse_provider_options(raw.get("provider_options"), issues)
    snapshot = StudioTTSPreferencesSnapshot(
        revision=revision,
        selection=selection,
        provider_options=provider_options,
    )
    return StudioTTSLoadResult(
        snapshot,
        StudioTTSLoadState.RECOVERED if issues else StudioTTSLoadState.LOADED,
        tuple(issues),
    )


def _raw_config(runtime_values: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = runtime_values.get("COMPREHENSIVE_CONFIG_RAW", {})
    return raw if isinstance(raw, Mapping) else {}


def _legacy_provider(raw: Mapping[str, Any], app_tts: Mapping[str, Any]) -> str | None:
    provider = app_tts.get("default_provider")
    if provider is None:
        legacy = raw.get("tts_settings", {})
        if isinstance(legacy, Mapping):
            provider = legacy.get("default_tts_provider")
    return provider if provider in BUILT_IN_TTS_PROVIDER_IDS else None


def _legacy_migration(
    raw: Mapping[str, Any],
) -> tuple[StudioTTSPreferencesSnapshot, tuple[str, ...], bool]:
    app_tts = raw.get("app_tts")
    if not isinstance(app_tts, Mapping):
        return StudioTTSPreferencesSnapshot(), (), False

    issues: list[str] = []
    options: dict[str, dict[str, float]] = {}
    selection_values: dict[str, object] = {}
    migration_needed = False

    chatterbox: dict[str, float] = {}
    for legacy_key, option_key in (
        ("CHATTERBOX_EXAGGERATION", "exaggeration"),
        ("CHATTERBOX_CFG_WEIGHT", "cfg_weight"),
    ):
        if legacy_key not in app_tts:
            continue
        value = app_tts[legacy_key]
        if (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and (float(value) == _LEGACY_DEFAULTS[legacy_key])
        ):
            continue
        migration_needed = True
        try:
            chatterbox[option_key] = _require_option_value(
                "chatterbox",
                option_key,
                value,
            )
        except (TypeError, ValueError):
            issues.append(f"app_tts.{legacy_key}")
    if chatterbox:
        options["chatterbox"] = chatterbox

    provider_id = _legacy_provider(raw, app_tts)
    if provider_id == "elevenlabs" and "ELEVENLABS_DEFAULT_MODEL" in app_tts:
        value = app_tts["ELEVENLABS_DEFAULT_MODEL"]
        if value != _LEGACY_DEFAULTS["ELEVENLABS_DEFAULT_MODEL"]:
            migration_needed = True
            try:
                selection_values["model_mode"] = "exact"
                selection_values["model_id"] = _require_optional_identifier(
                    "model",
                    value,
                )
            except (TypeError, ValueError):
                selection_values.pop("model_mode", None)
                selection_values.pop("model_id", None)
                issues.append("app_tts.ELEVENLABS_DEFAULT_MODEL")
    elif provider_id == "alltalk":
        if "ALLTALK_TTS_VOICE_DEFAULT" in app_tts:
            value = app_tts["ALLTALK_TTS_VOICE_DEFAULT"]
            if value != _LEGACY_DEFAULTS["ALLTALK_TTS_VOICE_DEFAULT"]:
                migration_needed = True
                try:
                    selection_values["voice_mode"] = "exact"
                    selection_values["voice_id"] = _require_optional_identifier(
                        "voice",
                        value,
                    )
                except (TypeError, ValueError):
                    selection_values.pop("voice_mode", None)
                    selection_values.pop("voice_id", None)
                    issues.append("app_tts.ALLTALK_TTS_VOICE_DEFAULT")
        if "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT" in app_tts:
            value = app_tts["ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT"]
            if value != _LEGACY_DEFAULTS["ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT"]:
                migration_needed = True
                try:
                    selection_values["response_format"] = _single_selection_value(
                        "response_format",
                        value,
                    )
                except (TypeError, ValueError):
                    selection_values.pop("response_format", None)
                    issues.append("app_tts.ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT")

    return (
        StudioTTSPreferencesSnapshot(
            selection=StudioTTSSelectionOverrides(**selection_values),
            provider_options=options,
        ),
        tuple(issues),
        migration_needed,
    )


class StudioTTSPreferenceStore:
    """Read and atomically replace only the additive Studio TTS section."""

    def load(self, *, migrate: bool = True) -> StudioTTSLoadResult:
        """Load preferences, optionally performing the idempotent v1 migration."""

        runtime = config_module.get_runtime_config_snapshot(force_reload=True)
        raw = _raw_config(runtime.values)
        if _STUDIO_SECTION in raw:
            return _parse_studio_section(raw[_STUDIO_SECTION])

        if not migrate:
            return StudioTTSLoadResult(
                StudioTTSPreferencesSnapshot(),
                StudioTTSLoadState.MISSING,
            )

        candidate, issues, migration_needed = _legacy_migration(raw)
        if not migration_needed:
            return StudioTTSLoadResult(
                candidate,
                StudioTTSLoadState.MISSING,
                issues,
            )

        if issues:
            logger.warning(
                "Studio TTS migration ignored malformed fields: {}",
                ", ".join(issues),
            )
        result = self.save(candidate)
        if result.status is StudioTTSWriteStatus.CONFLICT:
            return self.load(migrate=False)
        if result.snapshot is None:
            return StudioTTSLoadResult(
                candidate,
                StudioTTSLoadState.MIGRATION_FAILED,
                issues,
            )
        return StudioTTSLoadResult(
            result.snapshot,
            StudioTTSLoadState.MIGRATED,
            issues,
        )

    def save(
        self,
        snapshot: StudioTTSPreferencesSnapshot,
    ) -> StudioTTSWriteResult:
        """Save one complete Studio snapshot if its revision is still current."""

        if type(snapshot) is not StudioTTSPreferencesSnapshot:
            raise TypeError("Studio TTS save requires an exact snapshot")
        next_revision = snapshot.revision + 1
        section = snapshot.section_for_revision(next_revision)
        result = config_module.replace_revisioned_settings_section_to_cli_config(
            _STUDIO_SECTION,
            section,
            expected_revision=snapshot.revision,
        )
        if result.conflict:
            return StudioTTSWriteResult(StudioTTSWriteStatus.CONFLICT, None)
        if not result.file_replaced:
            if result.failure_phase is not None:
                return StudioTTSWriteResult(StudioTTSWriteStatus.FAILED, None)
            return StudioTTSWriteResult(StudioTTSWriteStatus.UNCHANGED, snapshot)

        saved = replace(snapshot, revision=next_revision)
        if not result.caches_reloaded:
            return StudioTTSWriteResult(
                StudioTTSWriteStatus.SAVED_CACHE_RELOAD_FAILED,
                saved,
            )
        return StudioTTSWriteResult(StudioTTSWriteStatus.SAVED, saved)

    def reset_to_global(
        self,
        snapshot: StudioTTSPreferencesSnapshot,
    ) -> StudioTTSWriteResult:
        """Delete every override while retaining schema and revision metadata."""

        if type(snapshot) is not StudioTTSPreferencesSnapshot:
            raise TypeError("Studio TTS reset requires an exact snapshot")
        empty = StudioTTSPreferencesSnapshot(revision=snapshot.revision)
        return self.save(empty)


__all__ = [
    "STUDIO_TTS_PROVIDER_OPTION_KEYS",
    "STUDIO_TTS_SCHEMA_VERSION",
    "StudioTTSLoadResult",
    "StudioTTSLoadState",
    "StudioTTSPreferenceStore",
    "StudioTTSPreferencesSnapshot",
    "StudioTTSSelectionOverrides",
    "StudioTTSWriteResult",
    "StudioTTSWriteStatus",
]
