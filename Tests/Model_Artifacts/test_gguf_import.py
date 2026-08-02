"""Tests for bounded local GGUF metadata inspection."""

from __future__ import annotations

import io
import struct
import unicodedata

import pytest

from Tests.Model_Artifacts.gguf_test_helpers import (
    ARRAY,
    BOOL,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    STRING,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    ArrayFixture,
    MetadataFixture,
    RawValueFixture,
    TensorFixture,
    gguf_string,
    make_gguf,
    make_raw_gguf,
)
from tldw_chatbook.Model_Artifacts import gguf_import as gguf


def _gguf_header(*, tensors: int = 0, metadata: int = 0) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", 3, tensors, metadata)


def _architecture_metadata() -> MetadataFixture:
    return MetadataFixture("general.architecture", STRING, "whisper")


class _GuardedReadHandle(io.BytesIO):
    """Fail loudly if a parser requests bytes outside the declared boundary."""

    def __init__(
        self,
        payload: bytes,
        *,
        read_boundary: int | None = None,
        max_request: int | None = None,
    ) -> None:
        super().__init__(payload)
        self.read_boundary = read_boundary
        self.max_request = max_request
        self.requests: list[int] = []

    def read(self, size: int | None = -1) -> bytes:
        request_size = -1 if size is None else size
        self.requests.append(request_size)
        if request_size < 0:
            raise AssertionError("unbounded reads are forbidden")
        if self.max_request is not None and request_size > self.max_request:
            raise AssertionError(f"oversized read requested: {request_size}")
        if (
            self.read_boundary is not None
            and self.tell() + request_size > self.read_boundary
        ):
            raise AssertionError("reader crossed the tensor-data boundary")
        return super().read(size)


def test_inspect_gguf_reads_supported_identity_without_tensor_payload(tmp_path):
    payload = make_gguf(
        architecture="whisper",
        variant="small",
        name="Whisper Small",
    )
    path = tmp_path / "model.gguf"
    path.write_bytes(payload)

    with path.open("rb") as handle:
        metadata = gguf.inspect_gguf(handle, file_size=len(payload))

    assert metadata.architecture == "whisper"
    assert metadata.variant == "small"
    assert metadata.model_name == "Whisper Small"


def test_parser_rejects_handle_not_positioned_at_byte_zero():
    payload = make_gguf()
    handle = io.BytesIO(payload)
    handle.seek(1)

    with pytest.raises(gguf.GGUFParseError, match="byte zero"):
        gguf.inspect_gguf(handle, file_size=len(payload))


def test_parser_rejects_repeated_inspection_without_reading_tensor_bytes():
    tensor_bytes = b"TENSOR-PAYLOAD-SENTINEL"
    payload = make_gguf(tensors=(TensorFixture(data=tensor_bytes),))
    data_offset = len(payload) - len(tensor_bytes)
    handle = _GuardedReadHandle(payload, read_boundary=data_offset)

    gguf.inspect_gguf(handle, file_size=len(payload))

    with pytest.raises(gguf.GGUFParseError, match="byte zero"):
        gguf.inspect_gguf(handle, file_size=len(payload))


def test_parser_rejects_bad_magic_with_typed_error():
    payload = bytearray(make_gguf())
    payload[:4] = b"NOPE"

    with pytest.raises(gguf.GGUFParseError, match="GGUF"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize("version", [1, 2, 4, 0xFFFFFFFF])
def test_parser_rejects_non_v3_versions_with_typed_error(version: int):
    payload = bytearray(make_gguf())
    payload[4:8] = struct.pack("<I", version)

    with pytest.raises(gguf.GGUFVersionError, match=str(version)):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


_TRUNCATED_STRUCTURES = (
    ("magic", b"GGU"),
    ("version", b"GGUF\x03\x00\x00"),
    ("tensor-count", b"GGUF" + struct.pack("<I", 3) + b"\x00" * 7),
    (
        "metadata-count",
        b"GGUF" + struct.pack("<IQ", 3, 0) + b"\x00" * 7,
    ),
    ("metadata-key-length", _gguf_header(metadata=1) + b"\x00" * 7),
    ("metadata-key", _gguf_header(metadata=1) + struct.pack("<Q", 4) + b"key"),
    ("metadata-type", _gguf_header(metadata=1) + gguf_string("key") + b"\x08\0\0"),
    (
        "metadata-scalar",
        _gguf_header(metadata=1)
        + gguf_string("key")
        + struct.pack("<I", UINT32)
        + b"\x01\x00\x00",
    ),
    (
        "metadata-string-length",
        _gguf_header(metadata=1)
        + gguf_string("key")
        + struct.pack("<I", STRING)
        + b"\x00" * 7,
    ),
    (
        "metadata-string",
        _gguf_header(metadata=1)
        + gguf_string("key")
        + struct.pack("<IQ", STRING, 4)
        + b"abc",
    ),
    (
        "array-element-type",
        _gguf_header(metadata=1)
        + gguf_string("array")
        + struct.pack("<I", ARRAY)
        + b"\x00" * 3,
    ),
    (
        "array-length",
        _gguf_header(metadata=1)
        + gguf_string("array")
        + struct.pack("<II", ARRAY, UINT8)
        + b"\x00" * 7,
    ),
    (
        "array-element-payload",
        _gguf_header(metadata=1)
        + gguf_string("array")
        + struct.pack("<IIQ", ARRAY, UINT16, 2)
        + b"\x01\x00\x02",
    ),
    ("tensor-name-length", _gguf_header(tensors=1) + b"\x00" * 7),
    ("tensor-name", _gguf_header(tensors=1) + struct.pack("<Q", 4) + b"abc"),
    (
        "tensor-dimension-count",
        _gguf_header(tensors=1) + gguf_string("t") + b"\x01\x00\x00",
    ),
    (
        "tensor-dimension",
        _gguf_header(tensors=1) + gguf_string("t") + struct.pack("<I", 1) + b"\x01" * 7,
    ),
    (
        "tensor-type",
        _gguf_header(tensors=1)
        + gguf_string("t")
        + struct.pack("<IQ", 1, 1)
        + b"\x00" * 3,
    ),
    (
        "tensor-offset",
        _gguf_header(tensors=1)
        + gguf_string("t")
        + struct.pack("<IQI", 1, 1, 0)
        + b"\x00" * 7,
    ),
    ("alignment-padding", make_gguf(name="forces padding")[:-1]),
)


@pytest.mark.parametrize(("section", "payload"), _TRUNCATED_STRUCTURES)
def test_parser_translates_every_structural_truncation_to_typed_error(
    section: str,
    payload: bytes,
):
    with pytest.raises(gguf.GGUFParseError, match="truncated"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_constants_are_the_approved_parser_limits():
    assert gguf.MAX_HEADER_BYTES == 64 * 1024 * 1024
    assert gguf.MAX_METADATA_ENTRIES == 4_096
    assert gguf.MAX_TENSOR_ENTRIES == 65_536
    assert gguf.MAX_STRING_BYTES == 1024 * 1024
    assert gguf.MAX_METADATA_PAYLOAD_BYTES == 64 * 1024 * 1024
    assert gguf.MAX_ARRAY_ELEMENTS == 1_000_000
    assert gguf.MAX_ARRAY_DEPTH == 2
    assert gguf.MAX_TENSOR_DIMENSIONS == 4
    assert gguf.MAX_METADATA_KEY_BYTES == 65_535
    assert gguf.MAX_TENSOR_NAME_BYTES == 64


@pytest.mark.parametrize(
    ("count_offset", "excessive_count"),
    [
        (8, 65_537),
        (16, 4_097),
    ],
)
def test_bounds_rejects_excessive_header_counts_before_iteration(
    count_offset: int,
    excessive_count: int,
):
    payload = bytearray(make_gguf())
    payload[count_offset : count_offset + 8] = struct.pack("<Q", excessive_count)

    with pytest.raises(gguf.GGUFBoundsError, match="limit"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_rejects_one_oversized_string_before_reading_or_allocating():
    payload = make_raw_gguf(
        metadata=(
            _architecture_metadata(),
            MetadataFixture(
                "oversized",
                STRING,
                RawValueFixture(struct.pack("<Q", 1024 * 1024 + 1)),
            ),
        )
    )
    handle = _GuardedReadHandle(payload, max_request=1024)

    with pytest.raises(gguf.GGUFBoundsError, match="string"):
        gguf.inspect_gguf(handle, file_size=gguf.MAX_HEADER_BYTES)

    assert max(handle.requests) <= 1024


@pytest.mark.parametrize(("size", "is_valid"), [(8, True), (9, False)])
def test_bounds_string_limit_accepts_exact_boundary_and_rejects_one_over(
    monkeypatch,
    size: int,
    is_valid: bool,
):
    monkeypatch.setattr(gguf, "MAX_STRING_BYTES", 8)
    payload = make_gguf(name="x" * size)

    if is_valid:
        metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))
        assert metadata.model_name == "x" * size
    else:
        with pytest.raises(gguf.GGUFBoundsError, match="string"):
            gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_rejects_cumulative_metadata_string_payload(monkeypatch):
    monkeypatch.setattr(gguf, "MAX_METADATA_PAYLOAD_BYTES", 64)
    payload = make_gguf(name="x" * 64)

    with pytest.raises(gguf.GGUFBoundsError, match="metadata payload"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_rejects_cumulative_metadata_array_payload(monkeypatch):
    monkeypatch.setattr(gguf, "MAX_METADATA_PAYLOAD_BYTES", 128)
    payload = make_gguf(
        extra_metadata=(
            MetadataFixture("array", ARRAY, ArrayFixture(UINT8, (1,) * 129)),
        )
    )

    with pytest.raises(gguf.GGUFBoundsError, match="metadata payload"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_cumulative_metadata_payload_accepts_exact_and_rejects_one_over(
    monkeypatch,
):
    exact_payload_bytes = sum(
        len(value.encode("utf-8"))
        for value in (
            "general.architecture",
            "whisper",
            "general.name",
            "",
            "general.file_type",
            "general.alignment",
        )
    )
    monkeypatch.setattr(gguf, "MAX_METADATA_PAYLOAD_BYTES", exact_payload_bytes)

    exact = make_gguf(name="")
    metadata = gguf.inspect_gguf(io.BytesIO(exact), file_size=len(exact))
    assert metadata.model_name == ""

    one_over = make_gguf(name="x")
    with pytest.raises(gguf.GGUFBoundsError, match="metadata payload"):
        gguf.inspect_gguf(io.BytesIO(one_over), file_size=len(one_over))


def test_bounds_rejects_excessive_array_count_before_iteration():
    raw_array = RawValueFixture(struct.pack("<IQ", UINT8, 1_000_001))
    payload = make_gguf(extra_metadata=(MetadataFixture("too_many", ARRAY, raw_array),))

    with pytest.raises(gguf.GGUFBoundsError, match="array"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(("count", "is_valid"), [(2, True), (3, False)])
def test_bounds_array_count_accepts_exact_boundary_and_rejects_one_over(
    monkeypatch,
    count: int,
    is_valid: bool,
):
    monkeypatch.setattr(gguf, "MAX_ARRAY_ELEMENTS", 2)
    payload = make_gguf(
        extra_metadata=(
            MetadataFixture("array", ARRAY, ArrayFixture(UINT8, (1,) * count)),
        )
    )

    if is_valid:
        metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))
        assert metadata.architecture == "whisper"
    else:
        with pytest.raises(gguf.GGUFBoundsError, match="array"):
            gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_rejects_arrays_nested_beyond_depth_two():
    too_deep = ArrayFixture(
        ARRAY,
        (ArrayFixture(ARRAY, (ArrayFixture(UINT8, (1,)),)),),
    )
    payload = make_gguf(extra_metadata=(MetadataFixture("too_deep", ARRAY, too_deep),))

    with pytest.raises(gguf.GGUFBoundsError, match="array depth"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(("nested", "is_valid"), [(False, True), (True, False)])
def test_bounds_array_depth_accepts_exact_boundary_and_rejects_one_over(
    monkeypatch,
    nested: bool,
    is_valid: bool,
):
    monkeypatch.setattr(gguf, "MAX_ARRAY_DEPTH", 1)
    array = (
        ArrayFixture(ARRAY, (ArrayFixture(UINT8, (1,)),))
        if nested
        else ArrayFixture(UINT8, (1,))
    )
    payload = make_gguf(extra_metadata=(MetadataFixture("array", ARRAY, array),))

    if is_valid:
        metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))
        assert metadata.architecture == "whisper"
    else:
        with pytest.raises(gguf.GGUFBoundsError, match="array depth"):
            gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_rejects_tensor_with_more_than_four_dimensions():
    payload = make_gguf(
        tensors=(TensorFixture(dimensions=(1, 1, 1, 1, 1)),),
    )

    with pytest.raises(gguf.GGUFBoundsError, match="dimensions"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(("dimensions", "is_valid"), [(2, True), (3, False)])
def test_bounds_tensor_dimensions_accepts_exact_and_rejects_one_over(
    monkeypatch,
    dimensions: int,
    is_valid: bool,
):
    monkeypatch.setattr(gguf, "MAX_TENSOR_DIMENSIONS", 2)
    payload = make_gguf(
        tensors=(TensorFixture(dimensions=(1,) * dimensions),),
    )

    if is_valid:
        metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))
        assert metadata.architecture == "whisper"
    else:
        with pytest.raises(gguf.GGUFBoundsError, match="dimensions"):
            gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_total_header_accepts_exact_boundary_and_rejects_one_over(
    monkeypatch,
):
    payload = make_gguf()
    monkeypatch.setattr(gguf, "MAX_HEADER_BYTES", len(payload))
    metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))
    assert metadata.data_offset == len(payload)

    monkeypatch.setattr(gguf, "MAX_HEADER_BYTES", len(payload) - 1)
    with pytest.raises(gguf.GGUFBoundsError, match="inspection limit"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_bounds_cursor_rejects_negative_and_over_budget_reads_before_io():
    handle = _GuardedReadHandle(b"")
    cursor = gguf._GGUFCursor(handle, file_size=gguf.MAX_HEADER_BYTES + 1)

    with pytest.raises(gguf.GGUFBoundsError):
        cursor.read_exact(-1)
    with pytest.raises(gguf.GGUFBoundsError, match="inspection limit"):
        cursor.read_exact(gguf.MAX_HEADER_BYTES + 1)

    assert handle.requests == []


def test_inspect_gguf_skips_all_well_formed_scalar_metadata_types():
    scalars = (
        MetadataFixture("u8", UINT8, 255),
        MetadataFixture("i8", INT8, -1),
        MetadataFixture("u16", UINT16, 65_535),
        MetadataFixture("i16", INT16, -2),
        MetadataFixture("u32", UINT32, 42),
        MetadataFixture("i32", INT32, -3),
        MetadataFixture("f32", FLOAT32, 1.5),
        MetadataFixture("bool", BOOL, True),
        MetadataFixture("string", STRING, "ignored"),
        MetadataFixture("u64", UINT64, 2**63),
        MetadataFixture("i64", INT64, -4),
        MetadataFixture("f64", FLOAT64, 2.5),
    )
    payload = make_gguf(extra_metadata=scalars)

    metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))

    assert metadata.architecture == "whisper"


def test_inspect_gguf_skips_homogeneous_arrays_including_depth_two():
    arrays = (
        MetadataFixture("a_u8", ARRAY, ArrayFixture(UINT8, (0, 255))),
        MetadataFixture("a_i8", ARRAY, ArrayFixture(INT8, (-1, 1))),
        MetadataFixture("a_u16", ARRAY, ArrayFixture(UINT16, (0, 65_535))),
        MetadataFixture("a_i16", ARRAY, ArrayFixture(INT16, (-2, 2))),
        MetadataFixture("a_u32", ARRAY, ArrayFixture(UINT32, (0, 42))),
        MetadataFixture("a_i32", ARRAY, ArrayFixture(INT32, (-3, 3))),
        MetadataFixture("a_f32", ARRAY, ArrayFixture(FLOAT32, (1.5,))),
        MetadataFixture("a_bool", ARRAY, ArrayFixture(BOOL, (True, False))),
        MetadataFixture("a_string", ARRAY, ArrayFixture(STRING, ("one", "two"))),
        MetadataFixture("a_u64", ARRAY, ArrayFixture(UINT64, (2**63,))),
        MetadataFixture("a_i64", ARRAY, ArrayFixture(INT64, (-4, 4))),
        MetadataFixture("a_f64", ARRAY, ArrayFixture(FLOAT64, (2.5,))),
        MetadataFixture(
            "nested",
            ARRAY,
            ArrayFixture(ARRAY, (ArrayFixture(UINT8, (1, 2)),)),
        ),
    )
    payload = make_gguf(extra_metadata=arrays)

    metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))

    assert metadata.architecture == "whisper"


def test_parser_rejects_unknown_metadata_value_type():
    payload = make_gguf(
        extra_metadata=(MetadataFixture("unknown", 99, RawValueFixture(b"")),)
    )

    with pytest.raises(gguf.GGUFParseError, match="value type"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_parser_rejects_unknown_array_element_type_even_when_empty():
    payload = make_gguf(
        extra_metadata=(
            MetadataFixture(
                "unknown_array",
                ARRAY,
                RawValueFixture(struct.pack("<IQ", 99, 0)),
            ),
        )
    )

    with pytest.raises(gguf.GGUFParseError, match="value type"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_parser_rejects_noncanonical_bool_scalar_byte():
    payload = make_gguf(
        extra_metadata=(MetadataFixture("bad_bool", BOOL, RawValueFixture(b"\x02")),)
    )

    with pytest.raises(gguf.GGUFParseError, match="BOOL"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_parser_rejects_noncanonical_bool_byte_in_array():
    raw_array = RawValueFixture(struct.pack("<IQ", BOOL, 3) + b"\x00\x02\x01")
    payload = make_gguf(
        extra_metadata=(MetadataFixture("bad_bool_array", ARRAY, raw_array),)
    )

    with pytest.raises(gguf.GGUFParseError, match="BOOL"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(
    "architecture",
    ["whis\nper", "Whisper", "whispér", "whisper-small", "whisper_small", ""],
)
def test_parser_rejects_noncanonical_architecture(architecture: str):
    payload = make_gguf(architecture=architecture)

    with pytest.raises(gguf.GGUFParseError, match="architecture"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize("architecture", ["a", "whisper", "qwen3", "123"])
def test_parser_retains_canonical_architecture_unchanged(architecture: str):
    payload = make_gguf(architecture=architecture)

    metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))

    assert metadata.architecture == architecture


@pytest.mark.parametrize(
    "key",
    [
        "",
        ".general",
        "general.",
        "general..name",
        "General.name",
        "general-name",
        "general name",
        "général.name",
    ],
)
def test_parser_rejects_invalid_metadata_key_syntax(key: str):
    payload = make_gguf(extra_metadata=(MetadataFixture(key, UINT8, 1),))

    with pytest.raises(gguf.GGUFParseError, match="metadata key"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(("size", "is_valid"), [(65_535, True), (65_536, False)])
def test_bounds_metadata_key_accepts_exact_boundary_and_rejects_one_over(
    size: int,
    is_valid: bool,
):
    payload = make_gguf(extra_metadata=(MetadataFixture("a" * size, UINT8, 1),))

    if is_valid:
        metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))
        assert metadata.architecture == "whisper"
    else:
        with pytest.raises(gguf.GGUFBoundsError, match="metadata key"):
            gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(
    ("name", "is_valid"),
    [("t" * 64, True), ("é" * 32, True), ("t" * 65, False)],
)
def test_bounds_tensor_name_accepts_64_encoded_bytes_and_rejects_65(
    name: str,
    is_valid: bool,
):
    payload = make_gguf(tensors=(TensorFixture(name=name),))

    if is_valid:
        metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))
        assert metadata.architecture == "whisper"
    else:
        with pytest.raises(gguf.GGUFBoundsError, match="tensor name"):
            gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(
    "metadata",
    [
        (MetadataFixture(b"\xff", STRING, "ignored"),),
        (MetadataFixture("general.architecture", STRING, b"\xff"),),
        (
            _architecture_metadata(),
            MetadataFixture("ignored", STRING, b"\xff"),
        ),
    ],
)
def test_parser_translates_invalid_utf8_to_typed_error(
    metadata: tuple[MetadataFixture, ...],
):
    payload = make_raw_gguf(metadata=metadata)

    with pytest.raises(gguf.GGUFParseError, match="UTF-8"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(
    ("key", "value_type", "value"),
    [
        ("general.architecture", STRING, "whisper"),
        ("stt.variant", STRING, "small"),
        ("general.name", STRING, "Whisper Small"),
        ("general.file_type", UINT32, 7),
        ("general.alignment", UINT32, 32),
    ],
)
def test_parser_rejects_duplicate_retained_keys(
    key: str,
    value_type: int,
    value: object,
):
    entries: list[MetadataFixture] = []
    if key != "general.architecture":
        entries.append(_architecture_metadata())
    entries.extend(
        [
            MetadataFixture(key, value_type, value),
            MetadataFixture(key, value_type, value),
        ]
    )
    if key != "general.alignment":
        entries.append(MetadataFixture("general.alignment", UINT32, 32))
    payload = make_raw_gguf(metadata=tuple(entries))

    with pytest.raises(gguf.GGUFParseError, match="duplicate"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(
    ("key", "value_type", "value"),
    [
        ("general.architecture", UINT32, 7),
        ("stt.variant", UINT32, 7),
        ("general.name", UINT32, 7),
        ("general.file_type", STRING, "7"),
        ("general.alignment", STRING, "32"),
    ],
)
def test_parser_rejects_wrong_retained_field_types(
    key: str,
    value_type: int,
    value: object,
):
    entries = [] if key == "general.architecture" else [_architecture_metadata()]
    entries.append(MetadataFixture(key, value_type, value))
    if key != "general.alignment":
        entries.append(MetadataFixture("general.alignment", UINT32, 32))
    payload = make_raw_gguf(metadata=tuple(entries))

    with pytest.raises(gguf.GGUFParseError, match="type"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize("alignment", [8, 16, 24, 32])
def test_parser_accepts_positive_eight_byte_alignment_multiples(alignment: int):
    payload = make_raw_gguf(
        metadata=(
            _architecture_metadata(),
            MetadataFixture("general.alignment", UINT32, alignment),
        )
    )

    metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))

    assert metadata.data_offset % alignment == 0


@pytest.mark.parametrize("alignment", [0, 1, 2, 3, 4, 7, 9, 25])
def test_parser_rejects_invalid_alignment(alignment: int):
    payload = make_raw_gguf(
        metadata=(
            _architecture_metadata(),
            MetadataFixture("general.alignment", UINT32, alignment),
        )
    )

    with pytest.raises(gguf.GGUFParseError, match="alignment"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_parser_rejects_tensor_offset_not_aligned_to_general_alignment():
    payload = make_raw_gguf(
        metadata=(
            _architecture_metadata(),
            MetadataFixture("general.alignment", UINT32, 32),
        ),
        tensors=(TensorFixture(offset=1),),
    )

    with pytest.raises(gguf.GGUFParseError, match="tensor offset.*alignment"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_parser_rejects_computed_data_offset_beyond_eof():
    payload = make_gguf(name="padding must exist")
    truncated = payload[:-1]

    with pytest.raises(gguf.GGUFParseError, match="truncated|data offset"):
        gguf.inspect_gguf(io.BytesIO(truncated), file_size=len(truncated))


def test_inspect_gguf_sanitizes_and_caps_display_strings():
    raw_name = "\x00Whis\nper\x7f\u0085" + "x" * 300
    payload = make_gguf(
        architecture="whisper",
        variant="sm\tall",
        name=raw_name,
    )

    metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))

    assert metadata.architecture == "whisper"
    assert metadata.variant == "small"
    assert metadata.model_name == ("Whisper" + "x" * 300)[:256]
    assert len(metadata.model_name) == 256
    assert not any(
        unicodedata.category(char).startswith("C") for char in metadata.model_name
    )


def test_inspect_gguf_never_reads_tensor_payload():
    tensor_bytes = b"TENSOR-PAYLOAD-SENTINEL"
    payload = make_gguf(
        tensors=(TensorFixture(data=tensor_bytes),),
    )
    data_offset = len(payload) - len(tensor_bytes)
    handle = _GuardedReadHandle(payload, read_boundary=data_offset)

    metadata = gguf.inspect_gguf(handle, file_size=len(payload))

    assert metadata.data_offset == data_offset
    assert handle.tell() == data_offset
