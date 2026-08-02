"""Bounded, standard-library-only GGUF metadata inspection."""

from __future__ import annotations

import struct
import unicodedata
from dataclasses import dataclass
from typing import Any, BinaryIO


GGUF_VERSION = 3

MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_METADATA_ENTRIES = 4_096
MAX_TENSOR_ENTRIES = 65_536
MAX_STRING_BYTES = 1024 * 1024
MAX_METADATA_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 1_000_000
MAX_ARRAY_DEPTH = 2
MAX_TENSOR_DIMENSIONS = 4
MAX_METADATA_KEY_BYTES = 65_535
MAX_TENSOR_NAME_BYTES = 64
MAX_DISPLAY_CHARACTERS = 256

_GGUF_MAGIC = b"GGUF"
_DEFAULT_ALIGNMENT = 32
_READ_CHUNK_BYTES = 64 * 1024
_LOWERCASE_ALPHANUMERIC = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_METADATA_KEY_CHARACTERS = _LOWERCASE_ALPHANUMERIC | {"_", "."}
# transcribe.cpp v0.1.3 dispatches underscore identifiers such as
# ``granite_speech``, so pinned-runtime admission is wider than generic GGUF prose.
_ARCHITECTURE_CHARACTERS = _LOWERCASE_ALPHANUMERIC | {"_"}

_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

_SCALAR_FORMATS = {
    _TYPE_UINT8: "<B",
    _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H",
    _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I",
    _TYPE_INT32: "<i",
    _TYPE_FLOAT32: "<f",
    _TYPE_BOOL: "<?",
    _TYPE_UINT64: "<Q",
    _TYPE_INT64: "<q",
    _TYPE_FLOAT64: "<d",
}
_SUPPORTED_VALUE_TYPES = frozenset((*_SCALAR_FORMATS, _TYPE_STRING, _TYPE_ARRAY))
_RETAINED_TYPES = {
    "general.architecture": _TYPE_STRING,
    "stt.variant": _TYPE_STRING,
    "general.name": _TYPE_STRING,
    "general.file_type": _TYPE_UINT32,
    "general.alignment": _TYPE_UINT32,
}


class GGUFError(ValueError):
    """Base class for GGUF inspection failures."""


class GGUFParseError(GGUFError):
    """Raised when a GGUF structure is malformed or truncated."""


class GGUFBoundsError(GGUFParseError):
    """Raised when a GGUF structure exceeds an inspection budget."""


class GGUFVersionError(GGUFParseError):
    """Raised when the GGUF version is not supported."""


@dataclass(frozen=True)
class GGUFMetadata:
    """Bounded metadata retained from a GGUF header."""

    architecture: str
    variant: str | None
    model_name: str | None
    file_type: int | None
    data_offset: int


@dataclass
class _MetadataBudget:
    """Track cumulative string and array content retained or skipped."""

    payload_bytes: int = 0

    def consume(self, size: int) -> None:
        if size < 0:
            raise GGUFBoundsError("GGUF metadata payload size cannot be negative")
        if self.payload_bytes + size > MAX_METADATA_PAYLOAD_BYTES:
            raise GGUFBoundsError("GGUF metadata payload exceeds limit")
        self.payload_bytes += size


class _GGUFCursor:
    """Read a GGUF header while enforcing byte and file boundaries."""

    def __init__(self, handle: BinaryIO, *, file_size: int) -> None:
        if file_size < 0:
            raise GGUFParseError("GGUF file size cannot be negative")
        self.handle = handle
        self.file_size = file_size
        self.header_bytes = 0

    def require_available(self, size: int) -> None:
        """Validate a prospective read without touching the handle."""
        if size < 0:
            raise GGUFBoundsError("GGUF read size cannot be negative")
        end = self.header_bytes + size
        if end > MAX_HEADER_BYTES:
            raise GGUFBoundsError("GGUF header exceeds inspection limit")
        if end > self.file_size:
            raise GGUFParseError("GGUF header is truncated")

    def read_exact(self, size: int) -> bytes:
        """Read exactly ``size`` budgeted bytes or raise a typed error."""
        self.require_available(size)
        try:
            data = self.handle.read(size)
        except MemoryError as exc:
            raise GGUFBoundsError("GGUF read could not be bounded") from exc
        if len(data) != size:
            raise GGUFParseError("GGUF header is truncated")
        self.header_bytes += size
        return data

    def skip_exact(self, size: int) -> None:
        """Consume bounded bytes in fixed-size chunks without retaining them."""
        self.require_available(size)
        remaining = size
        while remaining:
            chunk_size = min(remaining, _READ_CHUNK_BYTES)
            self.read_exact(chunk_size)
            remaining -= chunk_size

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        """Unpack one fixed-width value without bypassing read budgeting."""
        try:
            size = struct.calcsize(fmt)
            return struct.unpack(fmt, self.read_exact(size))
        except struct.error as exc:
            raise GGUFParseError("GGUF value could not be decoded") from exc


def _read_limited_utf8(
    cursor: _GGUFCursor,
    *,
    max_bytes: int,
    label: str,
    metadata_budget: _MetadataBudget | None = None,
) -> str:
    (encoded_size,) = cursor.unpack("<Q")
    if encoded_size > max_bytes:
        raise GGUFBoundsError(f"GGUF {label} exceeds limit")
    if metadata_budget is not None:
        metadata_budget.consume(encoded_size)
    encoded = cursor.read_exact(encoded_size)
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFParseError(f"GGUF {label} is not valid UTF-8") from exc


def _read_string(
    cursor: _GGUFCursor,
    *,
    metadata_budget: _MetadataBudget | None = None,
) -> str:
    return _read_limited_utf8(
        cursor,
        max_bytes=MAX_STRING_BYTES,
        label="string",
        metadata_budget=metadata_budget,
    )


def _read_metadata_key(
    cursor: _GGUFCursor,
    metadata_budget: _MetadataBudget,
) -> str:
    key = _read_limited_utf8(
        cursor,
        max_bytes=MAX_METADATA_KEY_BYTES,
        label="metadata key",
        metadata_budget=metadata_budget,
    )
    if (
        not key
        or not key.isascii()
        or any(character not in _METADATA_KEY_CHARACTERS for character in key)
        or any(not segment for segment in key.split("."))
    ):
        raise GGUFParseError("GGUF metadata key has invalid syntax")
    return key


def _read_tensor_name(cursor: _GGUFCursor) -> str:
    return _read_limited_utf8(
        cursor,
        max_bytes=MAX_TENSOR_NAME_BYTES,
        label="tensor name",
    )


def _read_scalar(cursor: _GGUFCursor, value_type: int) -> object:
    if value_type == _TYPE_BOOL:
        encoded = cursor.read_exact(1)
        if encoded not in {b"\x00", b"\x01"}:
            raise GGUFParseError("GGUF BOOL value must be encoded as 0 or 1")
        return encoded == b"\x01"
    fmt = _SCALAR_FORMATS[value_type]
    (value,) = cursor.unpack(fmt)
    return value


def _read_array(
    cursor: _GGUFCursor,
    metadata_budget: _MetadataBudget,
    *,
    depth: int,
) -> None:
    if depth > MAX_ARRAY_DEPTH:
        raise GGUFBoundsError("GGUF array depth exceeds limit")

    element_type, element_count = cursor.unpack("<IQ")
    if element_type not in _SUPPORTED_VALUE_TYPES:
        raise GGUFParseError(f"unsupported GGUF metadata value type: {element_type}")
    if element_count > MAX_ARRAY_ELEMENTS:
        raise GGUFBoundsError("GGUF array element count exceeds limit")

    if element_type in _SCALAR_FORMATS:
        element_size = struct.calcsize(_SCALAR_FORMATS[element_type])
        payload_size = element_count * element_size
        metadata_budget.consume(payload_size)
        if element_type == _TYPE_BOOL:
            cursor.require_available(payload_size)
            remaining = payload_size
            while remaining:
                chunk = cursor.read_exact(min(remaining, _READ_CHUNK_BYTES))
                if any(value > 1 for value in chunk):
                    raise GGUFParseError(
                        "GGUF BOOL array values must be encoded as 0 or 1"
                    )
                remaining -= len(chunk)
        else:
            cursor.skip_exact(payload_size)
        return

    minimum_element_size = 8 if element_type == _TYPE_STRING else 12
    cursor.require_available(element_count * minimum_element_size)
    for _ in range(element_count):
        if element_type == _TYPE_STRING:
            _read_string(cursor, metadata_budget=metadata_budget)
        else:
            _read_array(cursor, metadata_budget, depth=depth + 1)


def _read_metadata_value(
    cursor: _GGUFCursor,
    value_type: int,
    metadata_budget: _MetadataBudget,
) -> object:
    if value_type not in _SUPPORTED_VALUE_TYPES:
        raise GGUFParseError(f"unsupported GGUF metadata value type: {value_type}")
    if value_type in _SCALAR_FORMATS:
        return _read_scalar(cursor, value_type)
    if value_type == _TYPE_STRING:
        return _read_string(cursor, metadata_budget=metadata_budget)
    _read_array(cursor, metadata_budget, depth=1)
    return None


def _validate_retained_type(key: str, value_type: int) -> None:
    expected_type = _RETAINED_TYPES.get(key)
    if expected_type is not None and value_type != expected_type:
        raise GGUFParseError(f"GGUF metadata field {key!r} has the wrong type")


def _sanitize_display(value: str) -> str:
    display: list[str] = []
    for character in value:
        if unicodedata.category(character).startswith("C"):
            continue
        display.append(character)
        if len(display) == MAX_DISPLAY_CHARACTERS:
            break
    return "".join(display)


def _optional_display(retained: dict[str, object], key: str) -> str | None:
    value = retained.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise GGUFParseError(f"GGUF metadata field {key!r} has the wrong type")
    return _sanitize_display(value)


def _validate_architecture(value: str) -> None:
    if (
        not value
        or not value.isascii()
        or any(character not in _ARCHITECTURE_CHARACTERS for character in value)
    ):
        raise GGUFParseError("GGUF general.architecture must match [a-z0-9_]+")


def inspect_gguf(handle: BinaryIO, *, file_size: int) -> GGUFMetadata:
    """Inspect GGUF v3 structure and identity without reading tensor payload.

    The seekable handle must be positioned at byte zero. On success it remains
    positioned at the start of tensor data so the payload is never inspected.
    """
    if handle.tell() != 0:
        raise GGUFParseError("GGUF handle must be positioned at byte zero")
    cursor = _GGUFCursor(handle, file_size=file_size)
    metadata_budget = _MetadataBudget()
    if cursor.read_exact(4) != _GGUF_MAGIC:
        raise GGUFParseError("file is not GGUF")

    (version,) = cursor.unpack("<I")
    if version != GGUF_VERSION:
        raise GGUFVersionError(f"unsupported GGUF version: {version}")

    tensor_count, metadata_count = cursor.unpack("<QQ")
    if tensor_count > MAX_TENSOR_ENTRIES:
        raise GGUFBoundsError("GGUF tensor entry count exceeds limit")
    if metadata_count > MAX_METADATA_ENTRIES:
        raise GGUFBoundsError("GGUF metadata entry count exceeds limit")

    retained: dict[str, object] = {}
    seen_retained: set[str] = set()
    alignment = _DEFAULT_ALIGNMENT

    for _ in range(metadata_count):
        key = _read_metadata_key(cursor, metadata_budget)
        (value_type,) = cursor.unpack("<I")
        if key in _RETAINED_TYPES:
            if key in seen_retained:
                raise GGUFParseError(f"duplicate GGUF metadata field: {key}")
            seen_retained.add(key)
            _validate_retained_type(key, value_type)

        value = _read_metadata_value(cursor, value_type, metadata_budget)
        if key == "general.alignment":
            if not isinstance(value, int) or isinstance(value, bool):
                raise GGUFParseError("GGUF alignment has the wrong type")
            alignment = value
        elif key in _RETAINED_TYPES:
            retained[key] = value

    if alignment <= 0 or alignment % 8:
        raise GGUFParseError("GGUF alignment must be a positive multiple of 8")

    for _ in range(tensor_count):
        _read_tensor_name(cursor)
        (dimension_count,) = cursor.unpack("<I")
        if dimension_count > MAX_TENSOR_DIMENSIONS:
            raise GGUFBoundsError("GGUF tensor dimensions exceed limit")
        cursor.skip_exact(dimension_count * struct.calcsize("<Q"))
        cursor.unpack("<I")
        (tensor_offset,) = cursor.unpack("<Q")
        if tensor_offset % alignment:
            raise GGUFParseError("GGUF tensor offset violates general alignment")

    padding = -cursor.header_bytes % alignment
    data_offset = cursor.header_bytes + padding
    if data_offset > MAX_HEADER_BYTES:
        raise GGUFBoundsError("GGUF data offset exceeds inspection limit")
    if data_offset > file_size:
        raise GGUFParseError("GGUF header is truncated before data offset")
    cursor.skip_exact(padding)

    architecture = retained.get("general.architecture")
    if not isinstance(architecture, str):
        raise GGUFParseError("GGUF is missing general.architecture")
    _validate_architecture(architecture)

    file_type = retained.get("general.file_type")
    if file_type is not None and (
        not isinstance(file_type, int) or isinstance(file_type, bool)
    ):
        raise GGUFParseError("GGUF general.file_type has the wrong type")

    return GGUFMetadata(
        architecture=architecture,
        variant=_optional_display(retained, "stt.variant"),
        model_name=_optional_display(retained, "general.name"),
        file_type=file_type,
        data_offset=data_offset,
    )
