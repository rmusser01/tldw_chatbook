"""Pure validation for the pinned audio.cpp HTTP response contract.

The contract was reviewed at audio.cpp commit
``d3d748179e5ace353386fbf17bcaedfacf482d75``. RIFF ancillary chunks are
accepted when their declared payload and required pad byte are complete; their
contents are not interpreted.
"""

from __future__ import annotations

import json
import math
import re
import struct
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Literal, NoReturn, cast


ContractSurface = Literal["health", "models", "voices", "wav"]
TimingMetadata = Mapping[str, float]

_UNSAFE_IDENTIFIER_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})
_TIMING_HEADERS = {
    "x-audiocpp-wall-ms": "wall_ms",
    "x-audiocpp-audio-duration-ms": "audio_duration_ms",
    "x-audiocpp-rtf": "rtf",
}
_DECIMAL_PATTERN = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
_DEFAULT_MAX_TIMING_VALUE_CHARACTERS = 64
_MAX_JSON_NUMBER_CHARACTERS = 128
_MAX_JSON_FLOAT_EXPONENT = 308
_JSON_PARSE_FAILED = object()


class AudioCppContractError(ValueError):
    """A safe, value-independent audio.cpp contract validation failure."""

    __slots__ = ("category", "surface")

    def __init__(self, surface: ContractSurface, category: str) -> None:
        self.surface = surface
        self.category = category
        super().__init__(f"audio.cpp {surface} contract validation failed: {category}")


@dataclass(frozen=True, slots=True)
class AudioCppHealth:
    """Validated audio.cpp readiness metadata."""

    status: str
    backend: str
    models: int


@dataclass(frozen=True, slots=True)
class AudioCppModel:
    """Validated audio.cpp model metadata for a TTS entry."""

    model_id: str
    family: str
    task: str
    mode: str


@dataclass(frozen=True, slots=True)
class Pcm16WavInfo:
    """Authoritative structural metadata from a validated PCM16 WAV body."""

    sample_rate: int
    channels: int
    frame_count: int
    data_size: int
    byte_rate: int
    block_align: int
    bits_per_sample: int


class _InvalidJsonTokenError(ValueError):
    """Internal marker for unsafe JSON tokens."""


def _reject_json_constant(_: str) -> NoReturn:
    raise _InvalidJsonTokenError


def _convert_json_integer(value: str) -> int | None:
    try:
        return int(value)
    except (ValueError, OverflowError):
        return None


def _parse_json_integer(value: str) -> int:
    if len(value) > _MAX_JSON_NUMBER_CHARACTERS:
        raise _InvalidJsonTokenError
    parsed = _convert_json_integer(value)
    if parsed is None:
        raise _InvalidJsonTokenError
    return parsed


def _convert_json_float(value: str) -> float | None:
    try:
        decimal_value = Decimal(value)
        exponent = decimal_value.as_tuple().exponent
        if (
            not decimal_value.is_finite()
            or not isinstance(exponent, int)
            or abs(exponent) > _MAX_JSON_FLOAT_EXPONENT
            or abs(decimal_value.adjusted()) > _MAX_JSON_FLOAT_EXPONENT
        ):
            return None
        parsed = float(decimal_value)
    except (InvalidOperation, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_json_float(value: str) -> float:
    if len(value) > _MAX_JSON_NUMBER_CHARACTERS:
        raise _InvalidJsonTokenError
    parsed = _convert_json_float(value)
    if parsed is None:
        raise _InvalidJsonTokenError
    return parsed


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJsonTokenError
        result[key] = value
    return result


def _decode_utf8(body: bytes) -> str | None:
    try:
        return body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


def _parse_json_document(text: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
            parse_int=_parse_json_integer,
        )
    except AudioCppContractError:
        raise
    except (ValueError, OverflowError, RecursionError):
        return _JSON_PARSE_FAILED


def _fail(surface: ContractSurface, category: str) -> NoReturn:
    raise AudioCppContractError(surface, category)


def _load_json_object(
    body: bytes,
    *,
    max_metadata_bytes: int,
    surface: Literal["health", "models", "voices"],
) -> dict[str, Any]:
    if len(body) > max_metadata_bytes:
        _fail(surface, "size")

    text = _decode_utf8(body)
    if text is None:
        _fail(surface, "encoding")

    value = _parse_json_document(text)
    if value is _JSON_PARSE_FAILED:
        _fail(surface, "json")

    if not isinstance(value, dict):
        _fail(surface, "root")
    return cast(dict[str, Any], value)


def _identifier(
    value: object,
    *,
    max_characters: int,
    surface: Literal["health", "models", "voices"],
) -> str:
    if not isinstance(value, str):
        _fail(surface, "identifier_type")
    if (
        not value
        or len(value) > max_characters
        or value != value.strip()
        or any(
            unicodedata.category(character) in _UNSAFE_IDENTIFIER_CATEGORIES
            for character in value
        )
    ):
        _fail(surface, "identifier")
    return value


def parse_health_response(
    body: bytes,
    max_metadata_bytes: int,
    max_identifier_characters: int,
    max_models: int,
) -> AudioCppHealth:
    """Validate and parse the pinned ``GET /health`` response."""

    value = _load_json_object(
        body,
        max_metadata_bytes=max_metadata_bytes,
        surface="health",
    )
    if value.get("status") != "ok":
        _fail("health", "status")
    backend = _identifier(
        value.get("backend"),
        max_characters=max_identifier_characters,
        surface="health",
    )
    models = value.get("models")
    if type(models) is not int:
        _fail("health", "models_type")
    if models < 0 or models > max_models:
        _fail("health", "models_count")
    return AudioCppHealth(status="ok", backend=backend, models=models)


def parse_models_response(
    body: bytes,
    max_metadata_bytes: int,
    max_identifier_characters: int,
    max_models: int,
) -> tuple[AudioCppModel, ...]:
    """Validate ``GET /v1/models`` and return only its TTS entries."""

    value = _load_json_object(
        body,
        max_metadata_bytes=max_metadata_bytes,
        surface="models",
    )
    if value.get("object") != "list":
        _fail("models", "object")
    entries = value.get("data")
    if not isinstance(entries, list):
        _fail("models", "data")
    if len(entries) > max_models:
        _fail("models", "count")

    seen_ids: set[str] = set()
    tts_models: list[AudioCppModel] = []
    for entry in entries:
        if not isinstance(entry, dict):
            _fail("models", "entry")
        if entry.get("object") != "model":
            _fail("models", "entry_object")
        if entry.get("owned_by") != "engine":
            _fail("models", "owned_by")

        model_id = _identifier(
            entry.get("id"),
            max_characters=max_identifier_characters,
            surface="models",
        )
        family = _identifier(
            entry.get("family"),
            max_characters=max_identifier_characters,
            surface="models",
        )
        task = _identifier(
            entry.get("task"),
            max_characters=max_identifier_characters,
            surface="models",
        )
        mode = _identifier(
            entry.get("mode"),
            max_characters=max_identifier_characters,
            surface="models",
        )

        if model_id in seen_ids:
            _fail("models", "duplicate_id")
        seen_ids.add(model_id)
        if task.casefold() == "tts":
            tts_models.append(
                AudioCppModel(
                    model_id=model_id,
                    family=family,
                    task=task,
                    mode=mode,
                )
            )

    return tuple(tts_models)


def parse_voices_response(
    body: bytes,
    max_metadata_bytes: int,
    max_identifier_characters: int,
    max_voices: int,
) -> tuple[str, ...]:
    """Validate and parse ``GET /v1/audio/voices``."""

    value = _load_json_object(
        body,
        max_metadata_bytes=max_metadata_bytes,
        surface="voices",
    )
    entries = value.get("voices")
    if not isinstance(entries, list):
        _fail("voices", "data")
    if len(entries) > max_voices:
        _fail("voices", "count")

    voices: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        voice = _identifier(
            entry,
            max_characters=max_identifier_characters,
            surface="voices",
        )
        if voice in seen:
            _fail("voices", "duplicate_id")
        seen.add(voice)
        voices.append(voice)
    return tuple(voices)


def _parse_timing_value(
    value: object,
    *,
    max_value_characters: int,
) -> float | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > max_value_characters
        or _DECIMAL_PATTERN.fullmatch(value) is None
    ):
        return None
    try:
        decimal_value = Decimal(value)
    except InvalidOperation:
        return None
    if not decimal_value.is_finite() or decimal_value < 0:
        return None
    try:
        parsed = float(decimal_value)
    except (OverflowError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_timing_headers(
    headers: Mapping[str, str],
    max_value_characters: int = _DEFAULT_MAX_TIMING_VALUE_CHARACTERS,
) -> TimingMetadata:
    """Extract safe scalar timing metadata from the three pinned headers."""

    result: dict[str, float] = {}
    for header_name, raw_value in headers.items():
        if not isinstance(header_name, str):
            continue
        metadata_key = _TIMING_HEADERS.get(header_name.casefold())
        if metadata_key is None:
            continue
        parsed = _parse_timing_value(
            raw_value,
            max_value_characters=max_value_characters,
        )
        if parsed is not None:
            result[metadata_key] = parsed
    return MappingProxyType(result)


def validate_pcm16_wav(body: bytes) -> Pcm16WavInfo:
    """Validate a complete canonical PCM16 RIFF/WAVE body without rewriting it."""

    if len(body) < 12:
        _fail("wav", "header")
    if body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        _fail("wav", "signature")

    declared_size = struct.unpack_from("<I", body, 4)[0]
    if declared_size != len(body) - 8:
        _fail("wav", "riff_size")

    position = 12
    format_info: tuple[int, int, int, int] | None = None
    data_size: int | None = None

    while position < len(body):
        if len(body) - position < 8:
            _fail("wav", "chunk_header")
        chunk_id = body[position : position + 4]
        chunk_size = struct.unpack_from("<I", body, position + 4)[0]
        payload_start = position + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(body):
            _fail("wav", "chunk_size")

        if chunk_id == b"fmt ":
            if format_info is not None:
                _fail("wav", "duplicate_fmt")
            if data_size is not None:
                _fail("wav", "chunk_order")
            if chunk_size != 16:
                _fail("wav", "fmt_size")

            (
                format_tag,
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
            ) = struct.unpack_from("<HHIIHH", body, payload_start)
            if format_tag != 1:
                _fail("wav", "format_tag")
            if channels <= 0:
                _fail("wav", "channels")
            if sample_rate <= 0:
                _fail("wav", "sample_rate")
            if bits_per_sample != 16:
                _fail("wav", "bits_per_sample")
            expected_block_align = channels * 2
            if block_align != expected_block_align:
                _fail("wav", "block_align")
            if byte_rate != sample_rate * expected_block_align:
                _fail("wav", "byte_rate")
            format_info = (
                channels,
                sample_rate,
                byte_rate,
                block_align,
            )
        elif chunk_id == b"data":
            if data_size is not None:
                _fail("wav", "duplicate_data")
            if format_info is None:
                _fail("wav", "chunk_order")
            block_align = format_info[3]
            if chunk_size <= 0:
                _fail("wav", "data_size")
            if chunk_size % block_align:
                _fail("wav", "data_alignment")
            data_size = chunk_size

        position = payload_end
        if chunk_size % 2:
            if position >= len(body):
                _fail("wav", "chunk_padding")
            position += 1

    if format_info is None:
        _fail("wav", "missing_fmt")
    if data_size is None:
        _fail("wav", "missing_data")

    channels, sample_rate, byte_rate, block_align = format_info
    return Pcm16WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        frame_count=data_size // block_align,
        data_size=data_size,
        byte_rate=byte_rate,
        block_align=block_align,
        bits_per_sample=16,
    )


__all__ = [
    "AudioCppContractError",
    "AudioCppHealth",
    "AudioCppModel",
    "Pcm16WavInfo",
    "TimingMetadata",
    "parse_health_response",
    "parse_models_response",
    "parse_timing_headers",
    "parse_voices_response",
    "validate_pcm16_wav",
]
