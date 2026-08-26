"""Typed durable configuration for guided audio.cpp setup.

This module models saved Settings state only. It deliberately does not project a
server configuration, inspect files, contact audio.cpp, or own process lifecycle.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from numbers import Real
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .audio_cpp_config import AudioCppConfig, _nested_audio_cpp_config


_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z", re.ASCII)
_MANAGED_ARTIFACT_COMPONENT = re.compile(
    r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?\Z",
    re.ASCII,
)
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_MAX_JSON_INTEGER = 2**53 - 1
_MAX_INT32 = 2**31 - 1
_COMMON_TIMEOUT_FIELDS = (
    "connect_timeout_seconds",
    "synthesis_timeout_seconds",
)
_COMMON_LIMIT_FIELDS = (
    "max_input_characters",
    "max_response_bytes",
    "max_metadata_bytes",
    "max_catalog_models",
    "max_voices_per_model",
    "max_identifier_characters",
)
_MANAGED_TIMING_BOUNDS = {
    "managed_startup_timeout_seconds": (1.0, 300.0),
    "managed_health_check_interval_seconds": (2.0, 300.0),
    "managed_termination_grace_seconds": (0.1, 60.0),
}


class AudioCppManagedSetupSource(StrEnum):
    """Durable source for the active Managed configuration."""

    USER_JSON = "user_json"
    GUIDED = "guided"


class AudioCppBinarySelectionSource(StrEnum):
    """Reviewed provenance for a guided executable selection."""

    MANUAL = "manual"
    CONFIGURED = "configured"
    PATH = "path"
    HOMEBREW = "homebrew"
    CONVENTIONAL = "conventional"


class AudioCppBackendPreference(StrEnum):
    """Bounded persisted backend preference; Auto is resolved at launch."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    HIP = "hip"


def _contains_unsafe_text(value: str) -> bool:
    return any(
        character in {"\x00", "\r", "\n"}
        or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in value
    )


def _safe_token(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not _IDENTIFIER.fullmatch(value)
        or _contains_unsafe_text(value)
    ):
        raise ValueError(f"audio.cpp {label} is invalid")
    return value


def _safe_managed_artifact_component(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or _MANAGED_ARTIFACT_COMPONENT.fullmatch(value) is None
        or value.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_BASENAMES
    ):
        raise ValueError(f"audio.cpp managed artifact {label} is invalid")
    return value


def _safe_digest(value: object, *, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"audio.cpp {label} must be a lowercase SHA-256 identity")
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 1024
        or "\\" in value
        or _contains_unsafe_text(value)
    ):
        raise ValueError(f"audio.cpp {label} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
        raise ValueError(f"audio.cpp {label} must be a safe relative path")
    if path.as_posix() != value:
        raise ValueError(f"audio.cpp {label} must be a safe relative path")
    return value


def _safe_absolute_path(value: object, *, label: str, allow_empty: bool) -> str:
    if type(value) is not str:
        raise ValueError(f"audio.cpp {label} must be a path string")
    if allow_empty and not value:
        return value
    if (
        not value
        or value != value.strip()
        or len(value) > 4096
        or _contains_unsafe_text(value)
        or not (
            PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
        )
    ):
        raise ValueError(f"audio.cpp {label} must be an absolute path")
    return value


def _strict_integer(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(
            f"audio.cpp {label} must be an integer from {minimum} through {maximum}"
        )
    return value


def _bounded_number(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < minimum
        or float(value) > maximum
    ):
        raise ValueError(
            f"audio.cpp {label} must be a finite number from {minimum:g} "
            f"through {maximum:g}"
        )
    return float(value)


def _defensive_config_copy(value: object) -> object:
    """Copy JSON-like configuration values, including immutable registry views."""

    if isinstance(value, Mapping):
        return {
            deepcopy(key): _defensive_config_copy(nested)
            for key, nested in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_defensive_config_copy(item) for item in value)
    if isinstance(value, list):
        return [_defensive_config_copy(item) for item in value]
    return deepcopy(value)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class AudioCppRecipeOption(_FrozenModel):
    """One immutable string option admitted by an exact recipe."""

    name: str
    value: str

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        return _safe_token(value, label="recipe option name")

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> str:
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or len(value) > 256
            or _contains_unsafe_text(value)
        ):
            raise ValueError("audio.cpp recipe option value is invalid")
        return value


class AudioCppSafeModelProjection(_FrozenModel):
    """Allowlisted model-entry fields frozen into an accepted package."""

    family: str
    task: Literal["tts", "clone"]
    mode: Literal["offline"] = "offline"
    model_relative_path: str | None = None
    model_spec_override_relative_path: str | None = None
    busy_timeout_ms: int | None = None
    load_options: tuple[AudioCppRecipeOption, ...] = ()
    session_options: tuple[AudioCppRecipeOption, ...] = ()

    @field_validator("family", mode="before")
    @classmethod
    def _validate_family(cls, value: object) -> str:
        return _safe_token(value, label="recipe family")

    @field_validator("task", mode="before")
    @classmethod
    def _validate_task(cls, value: object) -> str:
        if value not in {"tts", "clone"}:
            raise ValueError("audio.cpp recipe task must be tts or clone")
        return str(value)

    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, value: object) -> str:
        if value != "offline":
            raise ValueError("audio.cpp recipe mode must be offline")
        return "offline"

    @field_validator(
        "model_relative_path",
        "model_spec_override_relative_path",
        mode="before",
    )
    @classmethod
    def _validate_relative_path(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None
        return _safe_relative_path(value, label=info.field_name)

    @field_validator("busy_timeout_ms", mode="before")
    @classmethod
    def _validate_busy_timeout(cls, value: object) -> int | None:
        if value is None:
            return None
        return _strict_integer(
            value,
            label="model busy_timeout_ms",
            minimum=1,
            maximum=_MAX_INT32,
        )

    @field_validator("load_options", "session_options", mode="before")
    @classmethod
    def _normalize_options(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _unique_options(self) -> AudioCppSafeModelProjection:
        for options in (self.load_options, self.session_options):
            names = tuple(option.name for option in options)
            if len(names) != len(set(names)):
                raise ValueError("audio.cpp recipe option names must be unique")
        return self


class AudioCppManagedArtifactIdentity(_FrozenModel):
    """Exact managed-store identity retained without importing store domains."""

    artifact_id: str
    revision: str
    variant: str

    @field_validator("artifact_id", "revision", "variant", mode="before")
    @classmethod
    def _validate_component(cls, value: object, info: ValidationInfo) -> str:
        return _safe_managed_artifact_component(value, label=info.field_name)


class AudioCppAcceptedPackage(_FrozenModel):
    """Immutable accepted recipe snapshot stored by Guided Settings."""

    package_uuid: str
    recipe_id: str
    recipe_revision: int
    package_variant: str
    public_model_id: str
    canonical_root: str
    canonical_root_identity: str
    configuration_identity: str
    weight_identity: str
    projection: AudioCppSafeModelProjection
    managed_artifact: AudioCppManagedArtifactIdentity | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("package_uuid", mode="before")
    @classmethod
    def _validate_package_uuid(cls, value: object) -> str:
        if type(value) is not str:
            raise ValueError("audio.cpp accepted package UUID is invalid")
        try:
            parsed = UUID(value)
        except ValueError:
            raise ValueError("audio.cpp accepted package UUID is invalid") from None
        if str(parsed) != value:
            raise ValueError("audio.cpp accepted package UUID is invalid")
        return value

    @field_validator("recipe_id", "package_variant", "public_model_id", mode="before")
    @classmethod
    def _validate_identifier(cls, value: object, info: ValidationInfo) -> str:
        return _safe_token(value, label=info.field_name)

    @field_validator("recipe_revision", mode="before")
    @classmethod
    def _validate_revision(cls, value: object) -> int:
        return _strict_integer(
            value,
            label="recipe revision",
            minimum=1,
            maximum=_MAX_INT32,
        )

    @field_validator("canonical_root", mode="before")
    @classmethod
    def _validate_root(cls, value: object) -> str:
        return _safe_absolute_path(
            value, label="canonical package root", allow_empty=False
        )

    @field_validator(
        "canonical_root_identity",
        "configuration_identity",
        "weight_identity",
        mode="before",
    )
    @classmethod
    def _validate_digest(cls, value: object, info: ValidationInfo) -> str:
        return _safe_digest(value, label=info.field_name)


class AudioCppSettingsConfig(_FrozenModel):
    """Full durable audio.cpp Settings state, including dormant source values."""

    mode: Literal["external", "managed"] = "external"
    base_url: str = "http://127.0.0.1:8080"

    managed_setup_source: AudioCppManagedSetupSource = (
        AudioCppManagedSetupSource.USER_JSON
    )
    managed_binary_path: str = ""
    managed_server_json_path: str = ""

    guided_binary_path: str = ""
    guided_binary_source: AudioCppBinarySelectionSource = (
        AudioCppBinarySelectionSource.MANUAL
    )
    guided_packages: tuple[AudioCppAcceptedPackage, ...] = ()
    guided_default_model_id: str | None = None
    guided_backend_preference: AudioCppBackendPreference = (
        AudioCppBackendPreference.AUTO
    )
    guided_device: int | None = None
    guided_threads: int | None = None
    guided_max_request_body_bytes: int = 256 * 1024 * 1024
    guided_busy_timeout_ms: int = 300_000

    managed_startup_timeout_seconds: float = 30.0
    managed_health_check_interval_seconds: float = 10.0
    managed_termination_grace_seconds: float = 5.0
    connect_timeout_seconds: float = 5.0
    synthesis_timeout_seconds: float = 600.0
    max_input_characters: int = 10_000
    max_response_bytes: int = 128 * 1024 * 1024
    max_metadata_bytes: int = 1024 * 1024
    max_catalog_models: int = 1000
    max_voices_per_model: int = 1000
    max_identifier_characters: int = 256

    @field_validator("mode", mode="before")
    @classmethod
    def _validate_outer_mode(cls, value: object) -> str:
        if value not in {"external", "managed"}:
            raise ValueError("audio.cpp mode must be external or managed")
        return str(value)

    @field_validator("base_url", mode="before")
    @classmethod
    def _validate_base_url(cls, value: object) -> str:
        return AudioCppConfig.from_mapping(
            {"mode": "external", "base_url": value}
        ).base_url

    @field_validator("managed_setup_source", mode="before")
    @classmethod
    def _validate_setup_source(cls, value: object) -> AudioCppManagedSetupSource:
        try:
            return AudioCppManagedSetupSource(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(
                "audio.cpp managed_setup_source must be user_json or guided"
            ) from None

    @field_validator("guided_binary_source", mode="before")
    @classmethod
    def _validate_binary_source(cls, value: object) -> AudioCppBinarySelectionSource:
        try:
            return AudioCppBinarySelectionSource(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError("audio.cpp guided binary source is invalid") from None

    @field_validator("guided_backend_preference", mode="before")
    @classmethod
    def _validate_backend(cls, value: object) -> AudioCppBackendPreference:
        try:
            return AudioCppBackendPreference(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError("audio.cpp guided backend preference is invalid") from None

    @field_validator(
        "managed_binary_path",
        "managed_server_json_path",
        "guided_binary_path",
        mode="before",
    )
    @classmethod
    def _validate_selected_path(cls, value: object, info: ValidationInfo) -> str:
        return _safe_absolute_path(value, label=info.field_name, allow_empty=True)

    @field_validator("guided_packages", mode="before")
    @classmethod
    def _normalize_packages(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("guided_default_model_id", mode="before")
    @classmethod
    def _validate_default_model(cls, value: object) -> str | None:
        if value is None:
            return None
        return _safe_token(value, label="guided default model id")

    @field_validator("guided_device", mode="before")
    @classmethod
    def _validate_device(cls, value: object) -> int | None:
        if value is None:
            return None
        return _strict_integer(
            value,
            label="guided device",
            minimum=0,
            maximum=1024,
        )

    @field_validator("guided_threads", mode="before")
    @classmethod
    def _validate_threads(cls, value: object) -> int | None:
        if value is None:
            return None
        return _strict_integer(
            value,
            label="guided threads",
            minimum=1,
            maximum=1024,
        )

    @field_validator("guided_max_request_body_bytes", mode="before")
    @classmethod
    def _validate_body_limit(cls, value: object) -> int:
        return _strict_integer(
            value,
            label="guided max request body bytes",
            minimum=1,
            maximum=_MAX_JSON_INTEGER,
        )

    @field_validator("guided_busy_timeout_ms", mode="before")
    @classmethod
    def _validate_server_busy_timeout(cls, value: object) -> int:
        return _strict_integer(
            value,
            label="guided busy timeout",
            minimum=1,
            maximum=_MAX_INT32,
        )

    @field_validator(*_COMMON_TIMEOUT_FIELDS, mode="before")
    @classmethod
    def _validate_common_timeout(cls, value: object, info: ValidationInfo) -> float:
        return _bounded_number(
            value,
            label=info.field_name,
            minimum=0.001,
            maximum=86_400,
        )

    @field_validator(*_MANAGED_TIMING_BOUNDS, mode="before")
    @classmethod
    def _validate_managed_timing(cls, value: object, info: ValidationInfo) -> float:
        minimum, maximum = _MANAGED_TIMING_BOUNDS[info.field_name]
        return _bounded_number(
            value,
            label=info.field_name,
            minimum=minimum,
            maximum=maximum,
        )

    @field_validator(*_COMMON_LIMIT_FIELDS, mode="before")
    @classmethod
    def _validate_common_limit(cls, value: object, info: ValidationInfo) -> int:
        return _strict_integer(
            value,
            label=info.field_name,
            minimum=1,
            maximum=_MAX_JSON_INTEGER,
        )

    @model_validator(mode="after")
    def _validate_guided_package_set(self) -> AudioCppSettingsConfig:
        package_uuids = tuple(package.package_uuid for package in self.guided_packages)
        if len(package_uuids) != len(set(package_uuids)):
            raise ValueError("audio.cpp guided package internal UUIDs must be unique")
        model_ids = tuple(package.public_model_id for package in self.guided_packages)
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("audio.cpp guided public model IDs must be unique")
        candidate_ids = tuple(
            (
                package.canonical_root,
                package.package_variant,
                package.configuration_identity,
                package.weight_identity,
            )
            for package in self.guided_packages
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("audio.cpp guided package identities must be unique")
        if self.guided_default_model_id is not None and (
            self.guided_default_model_id not in model_ids
        ):
            raise ValueError(
                "audio.cpp guided default model must name an accepted package"
            )
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AudioCppSettingsConfig:
        """Copy approved full-settings fields from one configuration mapping.

        Args:
            values: Mapping containing current, dormant, or unrelated settings.

        Returns:
            A validated immutable copy of approved audio.cpp settings fields.

        Raises:
            ValueError: If the input is not a mapping or an approved value is
                invalid.
        """
        if not isinstance(values, Mapping):
            raise ValueError("audio.cpp settings configuration must be a mapping")
        projected = {
            name: _defensive_config_copy(values[name])
            for name in cls.model_fields
            if name in values
        }
        return cls(**projected)

    def to_mapping(self) -> dict[str, object]:
        """Return one defensive JSON-compatible mapping of approved fields.

        Returns:
            A deep-copied mapping suitable for durable JSON/TOML projection.
        """
        return deepcopy(self.model_dump(mode="json"))


def project_audio_cpp_settings_config(
    app_config: Mapping[str, Any],
) -> AudioCppSettingsConfig:
    """Project the complete durable audio.cpp Settings snapshot.

    The exact nested raw entry takes precedence over the normalized entry, as
    it does for the active transport projection. Dormant manual and guided
    values remain present so later deliberate application can use them.

    Args:
        app_config: Normalized application settings with optional raw config.

    Returns:
        An immutable validated full-settings snapshot.

    Raises:
        ValueError: If the selected Settings entry is invalid.
    """
    selected = _nested_audio_cpp_config(app_config)
    if not isinstance(selected, Mapping):
        raise ValueError("audio.cpp settings configuration must be a mapping")
    return AudioCppSettingsConfig.from_mapping(selected)


__all__ = (
    "AudioCppAcceptedPackage",
    "AudioCppBackendPreference",
    "AudioCppBinarySelectionSource",
    "AudioCppManagedSetupSource",
    "AudioCppManagedArtifactIdentity",
    "AudioCppRecipeOption",
    "AudioCppSafeModelProjection",
    "AudioCppSettingsConfig",
    "project_audio_cpp_settings_config",
)
