"""Safe versioned Console generation settings in conversation metadata."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings


CONSOLE_GENERATION_SETTINGS_METADATA_KEY = "console_generation_settings"
CONSOLE_GENERATION_SETTINGS_VERSION = 1

_MAX_PROVIDER_CHARS = 128
_MAX_MODEL_CHARS = 256
_REASONING_EFFORT_VALUES = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)
_REASONING_SUMMARY_VALUES = frozenset({"auto", "concise", "detailed", "none"})
_VERBOSITY_VALUES = frozenset({"low", "medium", "high"})
_THINKING_EFFORT_VALUES = frozenset({"off", "low", "medium", "high", "xhigh", "max"})
_SAFE_FIELDS = frozenset(
    {
        "provider",
        "model",
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "max_tokens",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
        "streaming",
    }
)
_OWNED_FIELDS = frozenset({"version", *_SAFE_FIELDS})


class _PersistedConsoleGenerationSettings(BaseModel):
    """Strict validation boundary for conversation-owned persisted settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: int = Field(ge=1, le=1)
    provider: str = Field(min_length=1, max_length=_MAX_PROVIDER_CHARS)
    model: str | None = Field(default=None, max_length=_MAX_MODEL_CHARS)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, allow_inf_nan=False
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, allow_inf_nan=False
    )
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    thinking_effort: str | None = None
    thinking_budget_tokens: int | None = Field(default=None, ge=1024)
    streaming: bool

    @field_validator("provider", "model")
    @classmethod
    def _nonblank_string(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("String values must not be blank.")
        return value

    @field_validator(
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        mode="before",
    )
    @classmethod
    def _strict_json_number(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Value must be a JSON number or null.")
        try:
            return float(value)
        except OverflowError as exc:
            raise ValueError("Value is outside its supported range.") from exc

    @field_validator("reasoning_effort")
    @classmethod
    def _valid_reasoning_effort(cls, value: str | None) -> str | None:
        if value is not None and value not in _REASONING_EFFORT_VALUES:
            raise ValueError("Invalid reasoning effort.")
        return value

    @field_validator("reasoning_summary")
    @classmethod
    def _valid_reasoning_summary(cls, value: str | None) -> str | None:
        if value is not None and value not in _REASONING_SUMMARY_VALUES:
            raise ValueError("Invalid reasoning summary.")
        return value

    @field_validator("verbosity")
    @classmethod
    def _valid_verbosity(cls, value: str | None) -> str | None:
        if value is not None and value not in _VERBOSITY_VALUES:
            raise ValueError("Invalid verbosity.")
        return value

    @field_validator("thinking_effort")
    @classmethod
    def _valid_thinking_effort(cls, value: str | None) -> str | None:
        if value is not None and value not in _THINKING_EFFORT_VALUES:
            raise ValueError("Invalid thinking effort.")
        return value


@dataclass(frozen=True, slots=True)
class ConsoleGenerationSettingsSnapshot:
    """Complete safe generation snapshot owned by one conversation."""

    provider: str
    model: str | None
    temperature: float | None
    top_p: float | None
    min_p: float | None
    top_k: int | None
    max_tokens: int | None
    seed: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    reasoning_effort: str | None
    reasoning_summary: str | None
    verbosity: str | None
    thinking_effort: str | None
    thinking_budget_tokens: int | None
    streaming: bool


class ConsoleGenerationSettingsReadStatus(str, Enum):
    """Classification of the conversation-owned metadata value."""

    ABSENT = "absent"
    VALID = "valid"
    INVALID = "invalid"
    UNSUPPORTED_VERSION = "unsupported_version"


@dataclass(frozen=True, slots=True)
class ConsoleGenerationSettingsReadResult:
    """Safe metadata read result; only ``VALID`` carries a snapshot."""

    status: ConsoleGenerationSettingsReadStatus
    snapshot: ConsoleGenerationSettingsSnapshot | None = None

    def __post_init__(self) -> None:
        carries_snapshot = self.snapshot is not None
        if carries_snapshot != (
            self.status is ConsoleGenerationSettingsReadStatus.VALID
        ):
            raise ValueError("Only a valid metadata read may carry a snapshot.")


class ConsoleGenerationSettingsWriteStatus(str, Enum):
    """Outcome of a complete owned-snapshot compare-and-set write."""

    WRITTEN = "written"
    SUPERSEDED = "superseded"
    INVALID = "invalid"
    UNSUPPORTED_VERSION = "unsupported_version"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ConsoleGenerationSettingsWriteResult:
    """Result of persisting a safe generation settings snapshot."""

    status: ConsoleGenerationSettingsWriteStatus
    snapshot: ConsoleGenerationSettingsSnapshot | None = None

    def __post_init__(self) -> None:
        carries_snapshot = self.snapshot is not None
        if self.status is ConsoleGenerationSettingsWriteStatus.WRITTEN:
            if not carries_snapshot:
                raise ValueError("A written result must carry its snapshot.")
            return
        if (
            carries_snapshot
            and self.status is not ConsoleGenerationSettingsWriteStatus.SUPERSEDED
        ):
            raise ValueError("Only written or superseded results may carry a snapshot.")


class ConsoleGenerationSettingsVersionError(ValueError):
    """The owned metadata belongs to a newer application version."""


def snapshot_from_session_settings(
    settings: ConsoleSessionSettings,
) -> ConsoleGenerationSettingsSnapshot:
    """Project a session onto the exact durable safe-field allowlist."""
    if not isinstance(settings, ConsoleSessionSettings):
        raise TypeError("settings must be ConsoleSessionSettings.")
    return _validated_snapshot(
        ConsoleGenerationSettingsSnapshot(
            provider=settings.provider,
            model=settings.model,
            temperature=settings.temperature,
            top_p=settings.top_p,
            min_p=settings.min_p,
            top_k=settings.top_k,
            max_tokens=settings.max_tokens,
            seed=settings.seed,
            presence_penalty=settings.presence_penalty,
            frequency_penalty=settings.frequency_penalty,
            reasoning_effort=settings.reasoning_effort,
            reasoning_summary=settings.reasoning_summary,
            verbosity=settings.verbosity,
            thinking_effort=settings.thinking_effort,
            thinking_budget_tokens=settings.thinking_budget_tokens,
            streaming=settings.streaming,
        )
    )


def parse_console_generation_settings(
    metadata: object,
) -> ConsoleGenerationSettingsReadResult:
    """Parse one complete version-one snapshot, failing closed as one unit."""
    try:
        outer = strict_json_metadata_object(metadata, none_as_empty=True)
    except ValueError:
        return ConsoleGenerationSettingsReadResult(
            ConsoleGenerationSettingsReadStatus.INVALID
        )

    if CONSOLE_GENERATION_SETTINGS_METADATA_KEY not in outer:
        return ConsoleGenerationSettingsReadResult(
            ConsoleGenerationSettingsReadStatus.ABSENT
        )
    owned = outer[CONSOLE_GENERATION_SETTINGS_METADATA_KEY]
    if not isinstance(owned, Mapping):
        return ConsoleGenerationSettingsReadResult(
            ConsoleGenerationSettingsReadStatus.INVALID
        )
    version = owned.get("version")
    if type(version) is int and version > CONSOLE_GENERATION_SETTINGS_VERSION:
        return ConsoleGenerationSettingsReadResult(
            ConsoleGenerationSettingsReadStatus.UNSUPPORTED_VERSION
        )
    if type(version) is not int or version != CONSOLE_GENERATION_SETTINGS_VERSION:
        return ConsoleGenerationSettingsReadResult(
            ConsoleGenerationSettingsReadStatus.INVALID
        )
    if set(owned) != _OWNED_FIELDS:
        return ConsoleGenerationSettingsReadResult(
            ConsoleGenerationSettingsReadStatus.INVALID
        )
    try:
        persisted = _PersistedConsoleGenerationSettings.model_validate(dict(owned))
    except ValidationError:
        return ConsoleGenerationSettingsReadResult(
            ConsoleGenerationSettingsReadStatus.INVALID
        )
    snapshot = ConsoleGenerationSettingsSnapshot(
        provider=persisted.provider,
        model=persisted.model,
        temperature=persisted.temperature,
        top_p=persisted.top_p,
        min_p=persisted.min_p,
        top_k=persisted.top_k,
        max_tokens=persisted.max_tokens,
        seed=persisted.seed,
        presence_penalty=persisted.presence_penalty,
        frequency_penalty=persisted.frequency_penalty,
        reasoning_effort=persisted.reasoning_effort,
        reasoning_summary=persisted.reasoning_summary,
        verbosity=persisted.verbosity,
        thinking_effort=persisted.thinking_effort,
        thinking_budget_tokens=persisted.thinking_budget_tokens,
        streaming=persisted.streaming,
    )
    return ConsoleGenerationSettingsReadResult(
        ConsoleGenerationSettingsReadStatus.VALID,
        snapshot,
    )


def merge_console_generation_settings(
    metadata: object,
    snapshot: ConsoleGenerationSettingsSnapshot,
) -> dict[str, object]:
    """Replace only this codec's owned key and preserve all metadata siblings."""
    outer = strict_json_metadata_object(metadata, none_as_empty=True)
    existing = parse_console_generation_settings(outer)
    if existing.status is ConsoleGenerationSettingsReadStatus.INVALID:
        raise ValueError(
            "Cannot overwrite invalid Console generation settings metadata."
        )
    if existing.status is ConsoleGenerationSettingsReadStatus.UNSUPPORTED_VERSION:
        owned = outer[CONSOLE_GENERATION_SETTINGS_METADATA_KEY]
        version = owned.get("version") if isinstance(owned, Mapping) else None
        raise ConsoleGenerationSettingsVersionError(
            f"Cannot overwrite Console generation settings at version {version}."
        )
    validated = _validated_snapshot(snapshot)
    outer[CONSOLE_GENERATION_SETTINGS_METADATA_KEY] = {
        "version": CONSOLE_GENERATION_SETTINGS_VERSION,
        "provider": validated.provider,
        "model": validated.model,
        "temperature": validated.temperature,
        "top_p": validated.top_p,
        "min_p": validated.min_p,
        "top_k": validated.top_k,
        "max_tokens": validated.max_tokens,
        "seed": validated.seed,
        "presence_penalty": validated.presence_penalty,
        "frequency_penalty": validated.frequency_penalty,
        "reasoning_effort": validated.reasoning_effort,
        "reasoning_summary": validated.reasoning_summary,
        "verbosity": validated.verbosity,
        "thinking_effort": validated.thinking_effort,
        "thinking_budget_tokens": validated.thinking_budget_tokens,
        "streaming": validated.streaming,
    }
    return outer


def strict_json_metadata_object(
    metadata: object,
    *,
    none_as_empty: bool = False,
) -> dict[str, object]:
    """Normalize strict JSON-object metadata without lossy coercion.

    Args:
        metadata: A JSON object string or a mapping containing JSON values.
        none_as_empty: Whether missing metadata represents an empty object.

    Returns:
        A detached JSON-compatible object with exact string keys.

    Raises:
        ValueError: If the value is not a strict JSON object or contains a
            non-finite number anywhere in the object.
    """

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} is not supported.")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite JSON number {value!r} is not supported.")
        return number

    try:
        if metadata is None and none_as_empty:
            decoded = {}
        elif isinstance(metadata, Mapping):
            candidate = dict(metadata)
            if not _mapping_keys_are_strings(candidate):
                raise ValueError("Mapping keys must be strings.")
            encoded = json.dumps(candidate, allow_nan=False, sort_keys=True)
            decoded = json.loads(
                encoded,
                parse_constant=reject_constant,
                parse_float=finite_float,
            )
        elif type(metadata) is str:
            decoded = json.loads(
                metadata,
                parse_constant=reject_constant,
                parse_float=finite_float,
            )
        else:
            raise ValueError("Unsupported metadata type.")
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise ValueError("metadata must be a valid JSON object.") from exc
    if not isinstance(decoded, dict):
        raise ValueError("metadata must be a valid JSON object.")
    return decoded


def _mapping_keys_are_strings(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(
            type(key) is str and _mapping_keys_are_strings(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_mapping_keys_are_strings(item) for item in value)
    return True


def _validated_snapshot(
    snapshot: ConsoleGenerationSettingsSnapshot,
) -> ConsoleGenerationSettingsSnapshot:
    if not isinstance(snapshot, ConsoleGenerationSettingsSnapshot):
        raise TypeError("snapshot must be ConsoleGenerationSettingsSnapshot.")
    provider = _required_string(snapshot.provider, "provider", _MAX_PROVIDER_CHARS)
    model = _optional_string(snapshot.model, "model", _MAX_MODEL_CHARS)
    return ConsoleGenerationSettingsSnapshot(
        provider=provider,
        model=model,
        temperature=_optional_float(
            snapshot.temperature, "temperature", minimum=0.0, maximum=2.0
        ),
        top_p=_optional_float(snapshot.top_p, "top_p", minimum=0.0, maximum=1.0),
        min_p=_optional_float(snapshot.min_p, "min_p", minimum=0.0, maximum=1.0),
        top_k=_optional_int(snapshot.top_k, "top_k", minimum=0),
        max_tokens=_optional_int(snapshot.max_tokens, "max_tokens", minimum=1),
        seed=_optional_int(snapshot.seed, "seed", minimum=0),
        presence_penalty=_optional_float(
            snapshot.presence_penalty,
            "presence_penalty",
            minimum=-2.0,
            maximum=2.0,
        ),
        frequency_penalty=_optional_float(
            snapshot.frequency_penalty,
            "frequency_penalty",
            minimum=-2.0,
            maximum=2.0,
        ),
        reasoning_effort=_optional_choice(
            snapshot.reasoning_effort,
            "reasoning_effort",
            _REASONING_EFFORT_VALUES,
        ),
        reasoning_summary=_optional_choice(
            snapshot.reasoning_summary,
            "reasoning_summary",
            _REASONING_SUMMARY_VALUES,
        ),
        verbosity=_optional_choice(snapshot.verbosity, "verbosity", _VERBOSITY_VALUES),
        thinking_effort=_optional_choice(
            snapshot.thinking_effort,
            "thinking_effort",
            _THINKING_EFFORT_VALUES,
        ),
        thinking_budget_tokens=_optional_int(
            snapshot.thinking_budget_tokens,
            "thinking_budget_tokens",
            minimum=1024,
        ),
        streaming=_strict_bool(snapshot.streaming, "streaming"),
    )


def _required_string(value: object, name: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{name} must be a non-blank string of at most {maximum} characters."
        )
    return value


def _optional_string(value: object, name: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _required_string(value, name, maximum)


def _optional_choice(
    value: object,
    name: str,
    choices: frozenset[str],
) -> str | None:
    if value is None:
        return None
    if type(value) is not str or value not in choices:
        raise ValueError(f"{name} is invalid.")
    return value


def _optional_float(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number or null.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside its supported range.") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} is outside its supported range.")
    return number


def _optional_int(value: object, name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer or null.")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact boolean.")
    return value
