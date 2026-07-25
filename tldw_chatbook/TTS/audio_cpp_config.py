from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from ipaddress import AddressValueError, IPv4Address, IPv6Address
from numbers import Real
from typing import Any, Literal
from unicodedata import category
from urllib.parse import urlsplit

import idna

_CONFIG_DIAGNOSTIC = "audio.cpp configuration must be a mapping"
_MODE_DIAGNOSTIC = "audio.cpp mode must be external"
_URL_DIAGNOSTIC = "audio.cpp base_url must be an absolute HTTP or HTTPS origin"
_TIMEOUT_FIELDS = (
    "connect_timeout_seconds",
    "synthesis_timeout_seconds",
)
_LIMIT_FIELDS = (
    "max_input_characters",
    "max_response_bytes",
    "max_metadata_bytes",
    "max_catalog_models",
    "max_voices_per_model",
    "max_identifier_characters",
)
_CONFIG_FIELDS = (
    "mode",
    "base_url",
    *_TIMEOUT_FIELDS,
    *_LIMIT_FIELDS,
)
_MISSING = object()
_DISALLOWED_HOST_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Cn", "Co"})

AudioCppConfigValue = str | float | int


def _invalid_url() -> ValueError:
    return ValueError(_URL_DIAGNOSTIC)


def _validate_raw_port(netloc: str) -> None:
    if netloc.startswith("["):
        closing_bracket = netloc.find("]")
        if closing_bracket < 0:
            raise _invalid_url()
        suffix = netloc[closing_bracket + 1 :]
        if not suffix:
            return
        if not suffix.startswith(":"):
            raise _invalid_url()
        raw_port = suffix[1:]
    else:
        if ":" not in netloc:
            return
        raw_port = netloc.rsplit(":", maxsplit=1)[1]

    if not raw_port or not raw_port.isascii() or not raw_port.isdecimal():
        raise _invalid_url()


def _canonicalize_hostname(netloc: str, hostname: str) -> str:
    if any(
        category(character) in _DISALLOWED_HOST_CATEGORIES for character in hostname
    ):
        raise _invalid_url()

    if netloc.startswith("["):
        try:
            address = IPv6Address(hostname)
        except AddressValueError:
            raise _invalid_url() from None
        return f"[{address.compressed}]"

    candidate = hostname
    numeric_labels = candidate.split(".")
    if len(numeric_labels) == 4 and all(
        label
        and label.isascii()
        and all("0" <= character <= "9" for character in label)
        for label in numeric_labels
    ):
        try:
            return str(IPv4Address(candidate))
        except AddressValueError:
            raise _invalid_url() from None

    try:
        return idna.encode(candidate.lower()).decode("ascii")
    except (UnicodeError, idna.IDNAError):
        raise _invalid_url() from None


def _canonicalize_http_origin(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        )
        or "?" in value
        or "#" in value
    ):
        raise _invalid_url()

    try:
        parsed = urlsplit(value)
    except (UnicodeError, ValueError):
        raise _invalid_url() from None

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or "@" in parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise _invalid_url()

    _validate_raw_port(parsed.netloc)
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise _invalid_url() from None
    if hostname is None or (port is not None and port < 1):
        raise _invalid_url()
    canonical_hostname = _canonicalize_hostname(parsed.netloc, hostname)
    default_port = 80 if parsed.scheme == "http" else 443
    port_suffix = "" if port is None or port == default_port else f":{port}"
    return f"{parsed.scheme}://{canonical_hostname}{port_suffix}"


def _validate_timeout(field_name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"audio.cpp {field_name} must be a finite positive number")
    return float(value)


def _validate_limit(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"audio.cpp {field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class AudioCppConfig:
    """Validated external audio.cpp connection and safety limits."""

    mode: Literal["external"] = "external"
    base_url: str = "http://127.0.0.1:8080"
    connect_timeout_seconds: float = 5.0
    synthesis_timeout_seconds: float = 600.0
    max_input_characters: int = 10_000
    max_response_bytes: int = 128 * 1024 * 1024
    max_metadata_bytes: int = 1024 * 1024
    max_catalog_models: int = 1000
    max_voices_per_model: int = 1000
    max_identifier_characters: int = 256

    def __post_init__(self) -> None:
        if self.mode != "external":
            raise ValueError(_MODE_DIAGNOSTIC)
        object.__setattr__(
            self,
            "base_url",
            _canonicalize_http_origin(self.base_url),
        )
        for field_name in _TIMEOUT_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _validate_timeout(field_name, getattr(self, field_name)),
            )
        for field_name in _LIMIT_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _validate_limit(field_name, getattr(self, field_name)),
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AudioCppConfig:
        """Copy and validate the approved fields from a registry mapping.

        Args:
            values: Candidate external audio.cpp configuration.

        Returns:
            An immutable validated configuration snapshot.

        Raises:
            ValueError: If the mapping or an approved field is invalid.
        """
        if not isinstance(values, Mapping):
            raise ValueError(_CONFIG_DIAGNOSTIC)
        projected = {
            field_name: deepcopy(values[field_name])
            for field_name in _CONFIG_FIELDS
            if field_name in values
        }
        return cls(**projected)

    def to_mapping(self) -> dict[str, AudioCppConfigValue]:
        """Return an independent registry mapping of approved fields."""
        return {
            "mode": self.mode,
            "base_url": self.base_url,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "synthesis_timeout_seconds": self.synthesis_timeout_seconds,
            "max_input_characters": self.max_input_characters,
            "max_response_bytes": self.max_response_bytes,
            "max_metadata_bytes": self.max_metadata_bytes,
            "max_catalog_models": self.max_catalog_models,
            "max_voices_per_model": self.max_voices_per_model,
            "max_identifier_characters": self.max_identifier_characters,
        }


def _nested_audio_cpp_config(
    app_config: Mapping[str, Any],
) -> object:
    raw_config = app_config.get("COMPREHENSIVE_CONFIG_RAW")
    if isinstance(raw_config, Mapping):
        raw_tts = raw_config.get("app_tts")
        if isinstance(raw_tts, Mapping):
            raw_audio_cpp = raw_tts.get("audio_cpp", _MISSING)
            if raw_audio_cpp is not _MISSING:
                return raw_audio_cpp

    normalized_tts = app_config.get("APP_TTS_CONFIG")
    if isinstance(normalized_tts, Mapping):
        normalized_audio_cpp = normalized_tts.get("audio_cpp", _MISSING)
        if normalized_audio_cpp is not _MISSING:
            return normalized_audio_cpp
    return {}


def project_audio_cpp_config(
    app_config: Mapping[str, Any],
) -> AudioCppConfig:
    """Project the effective external audio.cpp configuration.

    The exact nested raw entry takes precedence over the normalized entry.
    Missing entries use the immutable defaults; environment variables are not
    part of this projection.

    Args:
        app_config: Normalized application settings with optional raw config.

    Returns:
        An immutable validated external configuration snapshot.

    Raises:
        ValueError: If the selected entry or an approved field is invalid.
    """
    selected = _nested_audio_cpp_config(app_config)
    if not isinstance(selected, Mapping):
        raise ValueError(_CONFIG_DIAGNOSTIC)
    return AudioCppConfig.from_mapping(selected)
