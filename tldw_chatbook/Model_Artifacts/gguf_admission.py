"""Bounded direct-local GGUF admission for the pinned transcribe.cpp runtime."""

from __future__ import annotations

import os
import platform
import stat
import struct
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from ..Utils.path_validation import validate_path_simple


GGUF_VERSION = 3

TRANSCRIBE_CPP_ARCHITECTURES = frozenset(
    {
        "canary",
        "canary_qwen",
        "cohere_asr",
        "funasr_nano",
        "gigaam",
        "granite_speech",
        "granite_speech_nar",
        "medasr",
        "moonshine",
        "moonshine_streaming",
        "parakeet",
        "qwen3_asr",
        "sensevoice",
        "voxtral",
        "voxtral_realtime",
        "whisper",
    }
)
TRANSCRIBE_CPP_WHEEL_TARGETS = frozenset(
    {
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("windows", "x86_64"),
        ("darwin", "arm64"),
        ("darwin", "x86_64"),
    }
)

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


class GGUFCompatibilityError(GGUFError):
    """Base class for pinned-runtime compatibility failures."""


class GGUFArchitectureError(GGUFCompatibilityError, GGUFParseError):
    """Raised when the declared architecture is not admitted by the runtime."""


class GGUFPlatformError(GGUFCompatibilityError):
    """Raised when the pinned runtime has no wheel for a platform target."""


class GGUFPathError(GGUFError):
    """Raised when the selected local GGUF cannot be opened safely."""


class GGUFSourceChangedError(GGUFPathError):
    """Raised when the selected source changes during one admission."""


@dataclass(frozen=True)
class GGUFSourceIdentity:
    """Filesystem identity observed from the admitted open descriptor."""

    device: int
    inode: int
    mode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class GGUFMetadata:
    """Bounded metadata retained from a GGUF header."""

    architecture: str
    variant: str | None
    model_name: str | None
    file_type: int | None
    data_offset: int


@dataclass(frozen=True)
class LocalGGUFAdmission:
    """Bounded result for one explicitly selected local GGUF."""

    path: Path = field(repr=False)
    metadata: GGUFMetadata
    source_identity: GGUFSourceIdentity
    platform_target: tuple[str, str]


@dataclass(frozen=True)
class LocalGGUFInspection:
    """Bounded structure result for one safely opened local GGUF."""

    path: Path = field(repr=False)
    metadata: GGUFMetadata
    source_identity: GGUFSourceIdentity


@dataclass
class OpenedLocalGGUF:
    """One no-follow local GGUF handle with an admitted source identity."""

    path: Path = field(repr=False)
    handle: BinaryIO
    descriptor: int
    identity: GGUFSourceIdentity

    def recheck(self) -> None:
        """Fail if the open node or selected name changed after admission."""
        try:
            opened = _source_identity(os.fstat(self.descriptor))
            named = _source_identity(
                _checked_regular_source_info(
                    self.path,
                    unavailable_message="Selected local GGUF identity could not be verified",
                )
            )
        except OSError:
            raise GGUFPathError(
                "Selected local GGUF identity could not be verified"
            ) from None
        if opened != self.identity or not _named_identity_matches_open(named, opened):
            raise GGUFSourceChangedError(
                "Selected local GGUF changed during validation"
            )


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


def _validate_architecture_identifier(value: str) -> None:
    if (
        not value
        or not value.isascii()
        or any(character not in _ARCHITECTURE_CHARACTERS for character in value)
    ):
        raise GGUFArchitectureError(
            "GGUF general.architecture is not a valid identifier"
        )


def require_transcribe_cpp_architecture(architecture: str) -> None:
    """Reject architecture names outside the pinned transcribe.cpp registry.

    Args:
        architecture: GGUF architecture identifier to validate.

    Raises:
        GGUFArchitectureError: If the identifier is malformed or unsupported.
    """
    _validate_architecture_identifier(architecture)
    if architecture not in TRANSCRIBE_CPP_ARCHITECTURES:
        raise GGUFArchitectureError(
            "GGUF general.architecture is not compatible with transcribe.cpp 0.1.3"
        )


def normalize_platform_target(system: str, machine: str) -> tuple[str, str]:
    """Normalize and admit a platform pair supported by pinned runtime wheels.

    Args:
        system: Operating-system name, such as ``Linux`` or ``Darwin``.
        machine: Machine architecture, such as ``x86_64`` or ``arm64``.

    Returns:
        The normalized operating-system and machine pair.

    Raises:
        GGUFPlatformError: If the pair has no supported transcribe.cpp wheel.
    """
    if not isinstance(system, str) or not isinstance(machine, str):
        raise GGUFPlatformError("transcribe.cpp is unavailable for this platform")

    normalized_system = system.casefold()
    normalized_machine = machine.casefold()
    if normalized_machine in {"amd64", "x86-64", "x64"}:
        normalized_machine = "x86_64"
    elif normalized_machine in {"arm64", "aarch64"}:
        normalized_machine = "aarch64" if normalized_system == "linux" else "arm64"

    target = (normalized_system, normalized_machine)
    if target not in TRANSCRIBE_CPP_WHEEL_TARGETS:
        raise GGUFPlatformError("transcribe.cpp is unavailable for this platform")
    return target


def inspect_gguf_structure(handle: BinaryIO, *, file_size: int) -> GGUFMetadata:
    """Inspect GGUF v3 structure and identity without reading tensor payload.

    The seekable handle must be positioned at byte zero. On success it remains
    positioned at the start of tensor data so the payload is never inspected.

    Args:
        handle: Seekable binary GGUF handle positioned at byte zero.
        file_size: Total file size in bytes.

    Returns:
        Bounded metadata retained from the GGUF header.

    Raises:
        GGUFParseError: If the GGUF structure is malformed or truncated.
        GGUFVersionError: If the file does not use GGUF version 3.
        GGUFBoundsError: If the header exceeds an inspection budget.
        GGUFArchitectureError: If the declared architecture identifier is malformed.
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

    max_tensor_offset: int | None = None
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
        if max_tensor_offset is None or tensor_offset > max_tensor_offset:
            max_tensor_offset = tensor_offset

    padding = -cursor.header_bytes % alignment
    data_offset = cursor.header_bytes + padding
    if data_offset > MAX_HEADER_BYTES:
        raise GGUFBoundsError("GGUF data offset exceeds inspection limit")
    if data_offset > file_size:
        raise GGUFParseError("GGUF header is truncated before data offset")
    if max_tensor_offset is not None and max_tensor_offset >= file_size - data_offset:
        raise GGUFParseError("GGUF tensor offset exceeds available payload")
    cursor.skip_exact(padding)

    architecture = retained.get("general.architecture")
    if not isinstance(architecture, str):
        raise GGUFParseError("GGUF is missing general.architecture")
    _validate_architecture_identifier(architecture)

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


def inspect_gguf(handle: BinaryIO, *, file_size: int) -> GGUFMetadata:
    """Inspect one GGUF accepted by the pinned transcribe.cpp runtime.

    Args:
        handle: Binary handle positioned at the beginning of the GGUF.
        file_size: Authoritative byte size for bounded header inspection.

    Returns:
        Bounded metadata retained from the accepted GGUF header.

    Raises:
        GGUFParseError: If the GGUF header is malformed or exceeds a bound.
        GGUFArchitectureError: If the architecture is unsupported by transcribe.cpp.
    """
    metadata = inspect_gguf_structure(handle, file_size=file_size)
    require_transcribe_cpp_architecture(metadata.architecture)
    return metadata


def _source_identity(info: os.stat_result) -> GGUFSourceIdentity:
    return GGUFSourceIdentity(
        device=info.st_dev,
        inode=info.st_ino,
        mode=info.st_mode,
        size_bytes=info.st_size,
        modified_ns=info.st_mtime_ns,
        changed_ns=info.st_ctime_ns,
    )


def _named_identity_matches_open(
    named: GGUFSourceIdentity,
    opened: GGUFSourceIdentity,
) -> bool:
    """Compare path and descriptor identities across native stat semantics."""
    if platform.system() != "Windows":
        return named == opened
    return (
        named.device,
        named.inode,
        named.mode,
        named.size_bytes,
        named.modified_ns,
    ) == (
        opened.device,
        opened.inode,
        opened.mode,
        opened.size_bytes,
        opened.modified_ns,
    )


def _read_only_no_follow_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _checked_regular_source_info(
    path: Path,
    *,
    unavailable_message: str = "Selected local GGUF is unavailable",
) -> os.stat_result:
    """Return one named local regular-file result or raise a path-safe error."""
    try:
        info = os.lstat(path)
    except OSError:
        raise GGUFPathError(unavailable_message) from None
    identity = _source_identity(info)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(identity.mode)
        or getattr(info, "st_file_attributes", 0) & reparse_point
        or not stat.S_ISREG(identity.mode)
    ):
        raise GGUFPathError("Selected local GGUF is not a regular file")
    return info


@contextmanager
def open_local_gguf(path: str | Path) -> Iterator[OpenedLocalGGUF]:
    """Yield one no-follow regular GGUF handle with stable identity.

    Args:
        path: Explicit local GGUF selected by the user.

    Yields:
        The open binary handle and its descriptor-derived identity.

    Raises:
        GGUFPathError: If the path is invalid, unsafe, or cannot be opened.
        GGUFSourceChangedError: If the selected name or open file changes.
    """
    try:
        validated = validate_path_simple(
            path, require_exists=False, probe_existing=False
        )
        selected = Path(validated).absolute()
    except (OSError, ValueError):
        raise GGUFPathError("Selected local GGUF path is invalid") from None
    if selected.suffix.casefold() != ".gguf":
        raise GGUFPathError("Selected local file must have a .gguf extension")
    initial = _checked_regular_source_info(selected)
    try:
        descriptor = os.open(selected, _read_only_no_follow_flags())
    except OSError:
        raise GGUFPathError("Selected local GGUF could not be opened safely") from None
    try:
        handle = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
    except OSError:
        os.close(descriptor)
        raise GGUFPathError("Selected local GGUF could not be opened safely") from None
    try:
        opened = _source_identity(os.fstat(descriptor))
    except OSError:
        handle.close()
        os.close(descriptor)
        raise GGUFPathError(
            "Selected local GGUF identity could not be verified"
        ) from None
    try:
        if not _named_identity_matches_open(
            _source_identity(initial), opened
        ) or not stat.S_ISREG(opened.mode):
            raise GGUFSourceChangedError(
                "Selected local GGUF changed during validation"
            )
        local = OpenedLocalGGUF(selected, handle, descriptor, opened)
        local.recheck()
        yield local
        local.recheck()
    finally:
        handle.close()
        os.close(descriptor)


def validate_local_gguf_structure(path: str | Path) -> LocalGGUFInspection:
    """Safely inspect bounded GGUF structure through a single descriptor.

    Args:
        path: Explicit local path selected by the user.

    Returns:
        The validated path, metadata, and source identity.

    Raises:
        GGUFPathError: If the path cannot be validated, opened, or inspected.
        GGUFSourceChangedError: If the file changes during validation.
        GGUFParseError: If the GGUF header is malformed.
    """
    with open_local_gguf(path) as opened:
        try:
            metadata = inspect_gguf_structure(
                opened.handle,
                file_size=opened.identity.size_bytes,
            )
        finally:
            opened.recheck()
        return LocalGGUFInspection(opened.path, metadata, opened.identity)


def validate_local_gguf(path: str | Path) -> LocalGGUFAdmission:
    """Inspect one explicit local GGUF for the pinned transcribe.cpp runtime.

    Args:
        path: Explicit local GGUF selected by the user.

    Returns:
        Validated metadata, source identity, path, and runtime platform target.

    Raises:
        GGUFPathError: If the path cannot be validated or opened safely.
        GGUFSourceChangedError: If the selected file changes during inspection.
        GGUFParseError: If the GGUF header is malformed or exceeds a bound.
        GGUFArchitectureError: If the architecture is unsupported by transcribe.cpp.
        GGUFPlatformError: If transcribe.cpp has no wheel for the current platform.
    """
    inspected = validate_local_gguf_structure(path)
    require_transcribe_cpp_architecture(inspected.metadata.architecture)
    platform_target = normalize_platform_target(
        platform.system(),
        platform.machine(),
    )
    return LocalGGUFAdmission(
        inspected.path,
        inspected.metadata,
        inspected.source_identity,
        platform_target,
    )
