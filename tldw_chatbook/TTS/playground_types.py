from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

AudioMetadataValue = str | int | float | bool | None


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


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
            MappingProxyType(deepcopy(dict(self.options))),
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

    @property
    def file_suffix(self) -> str:
        """Return the suffix implied by the actual response format."""
        return f".{self.audio_format.removeprefix('.')}"
