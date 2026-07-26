"""Immutable domain values and validation for TTS generation profiles."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Generic, TypeAlias, TypeVar
from uuid import UUID

from tldw_chatbook.TTS.profile_errors import ProfileValidationError


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]

_MAX_DISPLAY_NAME_CHARACTERS = 128
_MAX_PROVIDER_ID_CHARACTERS = 64
_MAX_OPAQUE_ID_CHARACTERS = 256
_MAX_RESPONSE_FORMAT_CHARACTERS = 32
_MAX_OPTIONS_BYTES = 16 * 1024
_MAX_OPTIONS_CONTAINER_LEVELS = 4
_PROVIDER_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_RESPONSE_FORMAT_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
_UNSAFE_NAME_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_T = TypeVar("_T")


class ProfileRepositoryState(StrEnum):
    """The public lifecycle states of a generation-profile repository."""

    OPEN = "open"
    RESTORING = "restoring"
    UNAVAILABLE = "unavailable"
    CLOSED = "closed"


def _is_unsafe_name_character(character: str) -> bool:
    code_point = ord(character)
    return (
        unicodedata.category(character) in _UNSAFE_NAME_CATEGORIES
        or 0xFDD0 <= code_point <= 0xFDEF
        or code_point & 0xFFFF in (0xFFFE, 0xFFFF)
    )


def _validate_display_name(value: object) -> str:
    if not isinstance(value, str):
        raise ProfileValidationError("display_name")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_DISPLAY_NAME_CHARACTERS
        or any(_is_unsafe_name_character(character) for character in normalized)
    ):
        raise ProfileValidationError("display_name")
    return normalized


def _validate_provider_id(value: object) -> str:
    if not isinstance(value, str) or not _PROVIDER_ID_PATTERN.fullmatch(value):
        raise ProfileValidationError("provider_id")
    return value


def _validate_opaque_id(
    value: object, field_name: str, *, nullable: bool = False
) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_OPAQUE_ID_CHARACTERS
    ):
        raise ProfileValidationError(field_name)
    return value


def _validate_canonical_opaque_id(value: object, field_name: str) -> str:
    identifier = _validate_opaque_id(value, field_name)
    assert identifier is not None
    if identifier != identifier.strip() or any(
        _is_unsafe_name_character(character) for character in identifier
    ):
        raise ProfileValidationError(field_name)
    return identifier


def _validate_speed(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError("speed")
    try:
        speed = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ProfileValidationError("speed") from None
    if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        raise ProfileValidationError("speed")
    return speed


def _validate_response_format(value: object) -> str:
    if not isinstance(value, str):
        raise ProfileValidationError("response_format")
    response_format = value.strip().lower()
    if not _RESPONSE_FORMAT_PATTERN.fullmatch(response_format):
        raise ProfileValidationError("response_format")
    return response_format


def _freeze_json_value(
    value: object,
    depth: int,
    active: set[int],
) -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProfileValidationError("options")
        return value
    if not isinstance(value, (Mapping, list)):
        raise ProfileValidationError("options")
    if depth > _MAX_OPTIONS_CONTAINER_LEVELS or id(value) in active:
        raise ProfileValidationError("options")

    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            frozen_mapping: dict[str, JsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ProfileValidationError("options")
                frozen_mapping[key] = _freeze_json_value(
                    item,
                    depth + 1,
                    active,
                )
            return MappingProxyType(frozen_mapping)
        return tuple(
            _freeze_json_value(
                item,
                depth + 1,
                active,
            )
            for item in value
        )
    except Exception:
        raise ProfileValidationError("options") from None
    finally:
        active.discard(id(value))


def _json_ready(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json_from_frozen(options: Mapping[str, JsonValue]) -> str:
    return json.dumps(
        _json_ready(options),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _freeze_options(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ProfileValidationError("options")
    frozen = _freeze_json_value(
        value,
        1,
        set(),
    )
    if not isinstance(frozen, Mapping):
        raise ProfileValidationError("options")
    try:
        encoded = _canonical_json_from_frozen(frozen).encode("utf-8")
    except Exception:
        raise ProfileValidationError("options") from None
    if len(encoded) > _MAX_OPTIONS_BYTES:
        raise ProfileValidationError("options")
    return frozen


def canonical_json_options(options: Mapping[str, JsonValue]) -> str:
    """Return validated options as a stable compact UTF-8 JSON document."""

    frozen = _freeze_options(options)
    return _canonical_json_from_frozen(frozen)


def _validate_audio_cpp(
    provider_id: str,
    response_format: str,
    speed: float,
    options: Mapping[str, JsonValue],
) -> None:
    if provider_id == "audio_cpp" and (
        response_format != "wav" or speed != 1.0 or bool(options)
    ):
        raise ProfileValidationError("audio_cpp")


def _validate_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileValidationError("revision")
    return value


def _validate_uuid(value: object, field_name: str) -> UUID:
    if not isinstance(value, UUID):
        raise ProfileValidationError(field_name)
    return value


def _validate_utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ProfileValidationError(field_name)
    try:
        offset = value.utcoffset()
    except (TypeError, ValueError):
        raise ProfileValidationError(field_name) from None
    if offset != timedelta(0):
        raise ProfileValidationError(field_name)
    return value.astimezone(UTC)


def _validate_nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileValidationError(field_name)
    return value


@dataclass(frozen=True, slots=True)
class TTSProfileDraft:
    """A complete, validated TTS selection ready for profile persistence."""

    display_name: str
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        display_name = _validate_display_name(self.display_name)
        provider_id = _validate_provider_id(self.provider_id)
        model_id = _validate_opaque_id(self.model_id, "model_id")
        voice_id = _validate_opaque_id(self.voice_id, "voice_id", nullable=True)
        response_format = _validate_response_format(self.response_format)
        speed = _validate_speed(self.speed)
        options = _freeze_options(self.options)
        _validate_audio_cpp(provider_id, response_format, speed, options)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "voice_id", voice_id)
        object.__setattr__(self, "response_format", response_format)
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "options", options)

    @property
    def normalized_name(self) -> str:
        """Return the persisted uniqueness key for this display name."""

        return unicodedata.normalize("NFKC", self.display_name).casefold()


@dataclass(frozen=True, slots=True)
class TTSGenerationProfile:
    """An immutable persisted TTS generation profile."""

    profile_id: UUID
    display_name: str
    normalized_name: str
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, JsonValue]
    revision: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        profile_id = _validate_uuid(self.profile_id, "profile_id")
        display_name = _validate_display_name(self.display_name)
        expected_normalized_name = unicodedata.normalize(
            "NFKC", display_name
        ).casefold()
        if self.normalized_name != expected_normalized_name:
            raise ProfileValidationError("normalized_name")
        provider_id = _validate_provider_id(self.provider_id)
        model_id = _validate_opaque_id(self.model_id, "model_id")
        voice_id = _validate_opaque_id(self.voice_id, "voice_id", nullable=True)
        response_format = _validate_response_format(self.response_format)
        speed = _validate_speed(self.speed)
        options = _freeze_options(self.options)
        _validate_audio_cpp(provider_id, response_format, speed, options)
        revision = _validate_revision(self.revision)
        created_at = _validate_utc_timestamp(self.created_at, "created_at")
        updated_at = _validate_utc_timestamp(self.updated_at, "updated_at")
        if created_at > updated_at:
            raise ProfileValidationError("timestamps")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "voice_id", voice_id)
        object.__setattr__(self, "response_format", response_format)
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)


@dataclass(frozen=True, slots=True)
class CharacterRef:
    """A full authority-scoped character identity for a profile assignment."""

    source: str
    authority_id: str
    character_id: str

    def __post_init__(self) -> None:
        if type(self.source) is not str or self.source not in ("local", "server"):
            raise ProfileValidationError("source")
        object.__setattr__(
            self,
            "authority_id",
            _validate_canonical_opaque_id(self.authority_id, "authority_id"),
        )
        object.__setattr__(
            self,
            "character_id",
            _validate_canonical_opaque_id(self.character_id, "character_id"),
        )


@dataclass(frozen=True, slots=True)
class CharacterTTSAssignment:
    """The exact profile selected for one authority-scoped character."""

    character_ref: CharacterRef
    profile_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.character_ref, CharacterRef):
            raise ProfileValidationError("assignment")
        object.__setattr__(
            self, "profile_id", _validate_uuid(self.profile_id, "profile_id")
        )


@dataclass(frozen=True, slots=True)
class AssignedTTSProfileSnapshot:
    """An immutable joined assignment and exact profile revision."""

    assignment: CharacterTTSAssignment
    profile: TTSGenerationProfile

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, CharacterTTSAssignment) or not isinstance(
            self.profile, TTSGenerationProfile
        ):
            raise ProfileValidationError("assignment")
        if self.assignment.profile_id != self.profile.profile_id:
            raise ProfileValidationError("assignment")


@dataclass(frozen=True, slots=True)
class TTSProfilePage:
    """A bounded page of immutable profile values and its total size."""

    profiles: Sequence[TTSGenerationProfile]
    total: int

    def __post_init__(self) -> None:
        try:
            profiles = tuple(self.profiles)
        except Exception:
            raise ProfileValidationError("profiles") from None
        if not all(isinstance(profile, TTSGenerationProfile) for profile in profiles):
            raise ProfileValidationError("profiles")
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(
            self, "total", _validate_nonnegative_integer(self.total, "total")
        )


@dataclass(frozen=True, slots=True)
class ProfileStoreResult(Generic[_T]):
    """A repository result paired with the lifecycle generation that produced it."""

    generation: int
    value: _T

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation",
            _validate_nonnegative_integer(self.generation, "generation"),
        )


@dataclass(frozen=True, slots=True)
class ProfileBackupReceipt:
    """Safe metadata about a completed profile-store backup."""

    created_at: datetime
    byte_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "created_at", _validate_utc_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self,
            "byte_count",
            _validate_nonnegative_integer(self.byte_count, "byte_count"),
        )


@dataclass(frozen=True, slots=True)
class ProfileRestoreReceipt:
    """Safe metadata about a completed profile-store restore."""

    restored_at: datetime
    profile_count: int
    assignment_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "restored_at",
            _validate_utc_timestamp(self.restored_at, "restored_at"),
        )
        object.__setattr__(
            self,
            "profile_count",
            _validate_nonnegative_integer(self.profile_count, "profile_count"),
        )
        object.__setattr__(
            self,
            "assignment_count",
            _validate_nonnegative_integer(self.assignment_count, "assignment_count"),
        )
