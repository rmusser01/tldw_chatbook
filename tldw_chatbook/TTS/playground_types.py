from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

AudioMetadataValue = str | int | float | bool | None


def _freeze_option(value: Any) -> Any:
    """Recursively isolate mutable option containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                deepcopy(key): _freeze_option(nested_value)
                for key, nested_value in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_option(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_option(item) for item in value)
    return deepcopy(value)


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_exact_identifier(
    name: str,
    value: object,
    *,
    nullable: bool = False,
) -> None:
    if value is None and nullable:
        return
    if type(value) is not str or not value:
        raise ValueError(f"{name} must not be empty")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError(f"{name} is invalid") from None


@dataclass(frozen=True, slots=True)
class TTSRequestedSelectionSnapshot:
    """Immutable text-free provenance for one exact admitted native request."""

    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, Any]
    configuration_revision: int

    def __post_init__(self) -> None:
        _require_exact_identifier("provider_id", self.provider_id)
        _require_exact_identifier("model_id", self.model_id)
        _require_exact_identifier("voice_id", self.voice_id, nullable=True)
        _require_exact_identifier("response_format", self.response_format)
        if type(self.speed) not in (int, float):
            raise TypeError("speed must be a number")
        speed = float(self.speed)
        if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
            raise ValueError("speed is invalid")
        if not isinstance(self.options, Mapping):
            raise TypeError("options must be a mapping")
        if type(self.configuration_revision) is not int:
            raise TypeError("configuration_revision must be an integer")
        if self.configuration_revision < 0:
            raise ValueError("configuration_revision must be nonnegative")
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "options", _freeze_option(self.options))


@dataclass(frozen=True, slots=True)
class STTSPlaygroundRequest:
    """Immutable snapshot of one Playground generation request."""

    operation_id: str
    provider_id: str
    model_id: str
    text: str
    voice_id: str | None
    response_format: str
    speed: float = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("operation_id", "provider_id", "model_id", "response_format"):
            _require_identifier(name, getattr(self, name))
        object.__setattr__(
            self,
            "options",
            _freeze_option(self.options),
        )


@dataclass(frozen=True, slots=True)
class STTSGeneratedAudio:
    """Immutable generated-audio artifact with request provenance."""

    path: Path
    provider_id: str
    model_id: str
    voice_id: str | None
    source_text: str
    operation_id: str
    audio_format: str
    content_type: str
    metadata: Mapping[str, AudioMetadataValue] = field(default_factory=dict)
    requested_selection: TTSRequestedSelectionSnapshot | None = None

    def __post_init__(self) -> None:
        for name in (
            "operation_id",
            "provider_id",
            "model_id",
            "audio_format",
            "content_type",
        ):
            _require_identifier(name, getattr(self, name))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )
        if (
            self.requested_selection is not None
            and type(self.requested_selection) is not TTSRequestedSelectionSnapshot
        ):
            raise TypeError(
                "requested_selection must be a requested selection snapshot"
            )

    @property
    def file_suffix(self) -> str:
        """Return the suffix implied by the actual response format."""
        return f".{self.audio_format.removeprefix('.')}"
