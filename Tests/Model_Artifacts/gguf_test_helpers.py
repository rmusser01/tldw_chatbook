"""Deterministic binary fixtures for bounded GGUF reader tests."""

from __future__ import annotations

import struct
from dataclasses import dataclass


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

UINT8 = 0
INT8 = 1
UINT16 = 2
INT16 = 3
UINT32 = 4
INT32 = 5
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9
UINT64 = 10
INT64 = 11
FLOAT64 = 12


@dataclass(frozen=True)
class ArrayFixture:
    """A homogeneous GGUF metadata array."""

    element_type: int
    values: tuple[object, ...]


@dataclass(frozen=True)
class MetadataFixture:
    """One typed GGUF metadata key/value entry."""

    key: str | bytes
    value_type: int
    value: object


@dataclass(frozen=True)
class RawValueFixture:
    """Pre-encoded metadata bytes for intentionally malformed fixtures."""

    data: bytes


@dataclass(frozen=True)
class TensorFixture:
    """One tensor-info record plus a tiny payload fragment."""

    name: str | bytes = "tensor"
    dimensions: tuple[int, ...] = (1,)
    ggml_type: int = 0
    offset: int = 0
    data: bytes = b"\x00"


def gguf_string(value: str) -> bytes:
    """Encode one GGUF length-prefixed UTF-8 string."""
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _raw_gguf_string(value: str | bytes) -> bytes:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<Q", len(encoded)) + encoded


_SCALAR_FORMATS = {
    UINT8: "<B",
    INT8: "<b",
    UINT16: "<H",
    INT16: "<h",
    UINT32: "<I",
    INT32: "<i",
    FLOAT32: "<f",
    BOOL: "<?",
    UINT64: "<Q",
    INT64: "<q",
    FLOAT64: "<d",
}


def _metadata_value(value_type: int, value: object) -> bytes:
    if isinstance(value, RawValueFixture):
        return value.data
    if value_type in _SCALAR_FORMATS:
        return struct.pack(_SCALAR_FORMATS[value_type], value)
    if value_type == STRING:
        if not isinstance(value, (str, bytes)):
            raise TypeError("GGUF string fixtures require str or bytes")
        return _raw_gguf_string(value)
    if value_type == ARRAY:
        if not isinstance(value, ArrayFixture):
            raise TypeError("GGUF array fixtures require ArrayFixture")
        return struct.pack("<IQ", value.element_type, len(value.values)) + b"".join(
            _metadata_value(value.element_type, item) for item in value.values
        )
    if not isinstance(value, bytes):
        raise TypeError("unknown GGUF value types require pre-encoded bytes")
    return value


def _metadata_entry(fixture: MetadataFixture) -> bytes:
    return (
        _raw_gguf_string(fixture.key)
        + struct.pack("<I", fixture.value_type)
        + _metadata_value(fixture.value_type, fixture.value)
    )


def make_gguf(
    *,
    architecture: str = "whisper",
    variant: str | None = None,
    name: str | None = None,
    file_type: int | None = 7,
    tensors: tuple[TensorFixture, ...] = (),
    extra_metadata: tuple[MetadataFixture, ...] = (),
) -> bytes:
    """Build a structurally real, deterministic GGUF v3 byte sequence."""
    metadata = [MetadataFixture("general.architecture", STRING, architecture)]
    if variant is not None:
        metadata.append(MetadataFixture("stt.variant", STRING, variant))
    if name is not None:
        metadata.append(MetadataFixture("general.name", STRING, name))
    if file_type is not None:
        metadata.append(MetadataFixture("general.file_type", UINT32, file_type))

    metadata.extend(extra_metadata)
    if not any(item.key == "general.alignment" for item in metadata):
        metadata.append(MetadataFixture("general.alignment", UINT32, 32))

    return make_raw_gguf(metadata=tuple(metadata), tensors=tensors)


def make_raw_gguf(
    *,
    metadata: tuple[MetadataFixture, ...],
    tensors: tuple[TensorFixture, ...] = (),
    layout_alignment: int | None = None,
) -> bytes:
    """Build GGUF framing around caller-controlled metadata entries."""

    header = bytearray(GGUF_MAGIC)
    header.extend(struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata)))
    for item in metadata:
        header.extend(_metadata_entry(item))

    for tensor in tensors:
        header.extend(_raw_gguf_string(tensor.name))
        header.extend(struct.pack("<I", len(tensor.dimensions)))
        for dimension in tensor.dimensions:
            header.extend(struct.pack("<Q", dimension))
        header.extend(struct.pack("<IQ", tensor.ggml_type, tensor.offset))

    if layout_alignment is None:
        alignment_value = next(
            (
                item.value
                for item in metadata
                if item.key == "general.alignment"
                and item.value_type == UINT32
                and not isinstance(item.value, RawValueFixture)
                and isinstance(item.value, int)
            ),
            32,
        )
        layout_alignment = alignment_value if alignment_value > 0 else 32
    header.extend(b"\x00" * (-len(header) % layout_alignment))

    data_size = max((tensor.offset + len(tensor.data) for tensor in tensors), default=0)
    tensor_data = bytearray(data_size)
    for tensor in tensors:
        tensor_data[tensor.offset : tensor.offset + len(tensor.data)] = tensor.data
    return bytes(header + tensor_data)
