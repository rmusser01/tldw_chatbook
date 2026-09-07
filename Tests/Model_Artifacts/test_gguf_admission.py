"""Tests for bounded direct-local GGUF admission."""

from __future__ import annotations

import ast
import io
import os
import stat
import struct
import unicodedata
from pathlib import Path

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
from tldw_chatbook.Model_Artifacts import gguf_admission as gguf


_ADMISSION_NONRELATIVE_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "contextlib",
        "dataclasses",
        "os",
        "pathlib",
        "platform",
        "stat",
        "struct",
        "typing",
        "unicodedata",
    }
)
_ADMISSION_RELATIVE_IMPORT = "..Utils.path_validation"
_DEFERRED_EXPECTED_IMPORT_TARGETS = frozenset(
    {
        ".gguf_admission",
        ".gguf_admission.GGUFError",
        ".gguf_admission.GGUFMetadata",
        ".gguf_admission._sanitize_display",
        ".gguf_admission.normalize_platform_target",
        ".gguf_admission.require_transcribe_cpp_architecture",
        ".service",
        ".service.ArtifactDescriptor",
        ".service.ArtifactFile",
        ".service.ArtifactFormat",
        ".service.ArtifactRef",
        ".service.ArtifactRole",
        ".service.ProvenanceClass",
        "__future__",
        "__future__.annotations",
        "collections.abc",
        "collections.abc.Iterable",
        "re",
    }
)
_DEFERRED_EXPECTED_CONSTANT_NAMES = frozenset(
    {
        "TRANSCRIBE_CPP_VERSION",
        "_PINNED_RUNTIME_RELEASE",
        "_RUNTIME_CONSTRAINT_CLAUSE",
    }
)
_DEFERRED_EXPECTED_CLASS_NAMES = ("GGUFAmbiguousCuratedMatchError",)
_DEFERRED_EXPECTED_FUNCTION_NAMES = (
    "_release_tuple",
    "_compatible_release",
    "runtime_constraint_admits_pinned_version",
    "_eligible_curated_descriptor",
    "_local_model_label",
    "_local_gguf_descriptor",
    "select_gguf_descriptor",
)


def _import_targets(source: str) -> set[str]:
    """Return normalized module and imported-alias targets from Python source."""
    targets: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module is None:
                targets.update(f"{prefix}{alias.name}" for alias in node.names)
                continue
            module_target = f"{prefix}{node.module}"
            targets.add(module_target)
            targets.update(f"{module_target}.{alias.name}" for alias in node.names)
            continue
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
        ):
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            targets.add(node.args[0].value)
        else:
            targets.add("<dynamic __import__>")
    return targets


def _unapproved_admission_import_targets(source: str) -> set[str]:
    unapproved: set[str] = set()
    for target in _import_targets(source):
        if target.endswith(".*"):
            unapproved.add(target)
            continue
        if target.startswith("."):
            if target != _ADMISSION_RELATIVE_IMPORT and not target.startswith(
                f"{_ADMISSION_RELATIVE_IMPORT}."
            ):
                unapproved.add(target)
            continue
        if target.split(".", 1)[0] not in _ADMISSION_NONRELATIVE_IMPORT_ROOTS:
            unapproved.add(target)
    return unapproved


def _assert_deferred_source_contract(source: str) -> None:
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    assert "DEFERRED" in docstring
    assert "TASK-1915" in docstring

    assert _import_targets(source) == _DEFERRED_EXPECTED_IMPORT_TARGETS

    all_bindings: list[ast.expr | None] = []
    constant_bindings: dict[str, list[ast.expr | None]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__all__":
                    all_bindings.append(node.value)
                else:
                    constant_bindings.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                all_bindings.append(node.value)
            else:
                constant_bindings.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                all_bindings.append(node.value)
            else:
                constant_bindings.setdefault(node.target.id, []).append(node.value)

    assert len(all_bindings) == 1
    assert isinstance(all_bindings[0], ast.Tuple)
    assert not all_bindings[0].elts
    assert constant_bindings.keys() == _DEFERRED_EXPECTED_CONSTANT_NAMES
    assert all(len(bindings) == 1 for bindings in constant_bindings.values())
    version = constant_bindings["TRANSCRIBE_CPP_VERSION"][0]
    assert isinstance(version, ast.Constant)
    assert version.value == "0.1.3"

    class_names = tuple(
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    )
    assert class_names == _DEFERRED_EXPECTED_CLASS_NAMES
    functions = tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    assert tuple(function.name for function in functions) == (
        _DEFERRED_EXPECTED_FUNCTION_NAMES
    )
    for function in functions:
        body = function.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        assert body
        assert any(not isinstance(statement, ast.Pass) for statement in body)

    deferred_error = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GGUFAmbiguousCuratedMatchError"
    )
    assert len(deferred_error.bases) == 1
    assert isinstance(deferred_error.bases[0], ast.Name)
    assert deferred_error.bases[0].id == "GGUFError"


def _gguf_header(*, tensors: int = 0, metadata: int = 0) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", 3, tensors, metadata)


def _architecture_metadata() -> MetadataFixture:
    return MetadataFixture("general.architecture", STRING, "whisper")


def _supported_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gguf.platform, "machine", lambda: "arm64")


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
    ["whis\nper", "Whisper", "whispér", "whisper-small", "whisper.small", ""],
)
def test_parser_rejects_noncanonical_architecture(architecture: str):
    payload = make_gguf(architecture=architecture)

    with pytest.raises(gguf.GGUFParseError, match="architecture"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


@pytest.mark.parametrize(
    "architecture",
    ["canary", "whisper", "qwen3_asr", "granite_speech"],
)
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


def test_parser_rejects_aligned_tensor_offset_beyond_available_payload():
    complete = make_gguf(tensors=(TensorFixture(offset=32),))
    data_offset = len(complete) - 33
    truncated = complete[:-2]
    handle = _GuardedReadHandle(truncated, read_boundary=data_offset)

    with pytest.raises(gguf.GGUFParseError, match="tensor offset.*payload"):
        gguf.inspect_gguf(handle, file_size=len(truncated))


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


_TRANSCRIBE_CPP_ARCHITECTURES = frozenset(
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


def test_transcribe_cpp_architecture_declaration_is_exact():
    assert gguf.TRANSCRIBE_CPP_ARCHITECTURES == _TRANSCRIBE_CPP_ARCHITECTURES


@pytest.mark.parametrize("architecture", sorted(_TRANSCRIBE_CPP_ARCHITECTURES))
def test_inspect_gguf_accepts_pinned_transcribe_cpp_architecture(
    architecture: str,
):
    payload = make_gguf(architecture=architecture)

    metadata = gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))

    assert metadata.architecture == architecture


@pytest.mark.parametrize("architecture", ["cohere", "granite", "llama"])
def test_inspect_gguf_rejects_near_miss_architecture_with_typed_error(
    architecture: str,
):
    payload = make_gguf(architecture=architecture)

    with pytest.raises(gguf.GGUFArchitectureError, match="transcribe.cpp 0.1.3"):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))


def test_generic_structure_accepts_llama_without_weakening_transcribe_policy():
    payload = make_gguf(architecture="llama", name="Local LLM", file_type=7)

    metadata = gguf.inspect_gguf_structure(
        io.BytesIO(payload),
        file_size=len(payload),
    )

    assert metadata.architecture == "llama"
    assert metadata.model_name == "Local LLM"
    with pytest.raises(gguf.GGUFArchitectureError):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))

    malformed = make_gguf(architecture="../private")
    with pytest.raises(gguf.GGUFArchitectureError, match="identifier"):
        gguf.inspect_gguf_structure(
            io.BytesIO(malformed),
            file_size=len(malformed),
        )


def test_open_local_gguf_rejects_symlink(tmp_path: Path):
    target = tmp_path / "model.gguf"
    target.write_bytes(make_gguf(architecture="llama"))
    link = tmp_path / "link.gguf"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(gguf.GGUFPathError, match="regular file"):
        with gguf.open_local_gguf(link):
            pytest.fail("symlink must not open")


def test_open_local_gguf_recheck_detects_same_path_replacement(tmp_path: Path):
    source = tmp_path / "model.gguf"
    replacement = tmp_path / "replacement.gguf"
    source.write_bytes(make_gguf(architecture="llama", name="first"))
    replacement.write_bytes(make_gguf(architecture="llama", name="second"))

    with pytest.raises(gguf.GGUFSourceChangedError):
        with gguf.open_local_gguf(source) as opened:
            source.unlink()
            replacement.rename(source)
            opened.recheck()


def test_open_local_gguf_result_redacts_selected_path(tmp_path: Path):
    source = tmp_path / "private-model.gguf"
    source.write_bytes(make_gguf(architecture="llama"))

    with gguf.open_local_gguf(source) as opened:
        assert str(source) not in repr(opened)


def test_open_local_gguf_recheck_maps_post_open_lstat_failure_to_identity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "private-model.gguf"
    source.write_bytes(make_gguf(architecture="llama"))
    real_lstat = gguf.os.lstat
    calls = 0

    def fail_during_recheck(path: str | Path) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("name vanished")
        return real_lstat(path)

    monkeypatch.setattr(gguf.os, "lstat", fail_during_recheck)

    with gguf.open_local_gguf(source) as opened:
        with pytest.raises(gguf.GGUFPathError, match="identity could not be verified"):
            opened.recheck()


def test_open_local_gguf_preserves_caller_oserror(tmp_path: Path):
    source = tmp_path / "model.gguf"
    source.write_bytes(make_gguf(architecture="llama"))
    destination_error = OSError("destination is full")

    with pytest.raises(OSError) as raised:
        with gguf.open_local_gguf(source):
            raise destination_error

    assert raised.value is destination_error


def test_open_local_gguf_rejects_windows_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "model.gguf"
    source.write_bytes(make_gguf(architecture="llama"))
    actual = os.lstat(source)

    class ReparsePointInfo:
        st_dev = actual.st_dev
        st_ino = actual.st_ino
        st_mode = actual.st_mode
        st_size = actual.st_size
        st_mtime_ns = actual.st_mtime_ns
        st_ctime_ns = actual.st_ctime_ns
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    monkeypatch.setattr(gguf.os, "lstat", lambda path: ReparsePointInfo())
    monkeypatch.setattr(
        gguf.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        ReparsePointInfo.st_file_attributes,
        raising=False,
    )

    with pytest.raises(gguf.GGUFPathError, match="regular file"):
        with gguf.open_local_gguf(source):
            pytest.fail("reparse point must not open")


def test_open_local_gguf_accepts_windows_path_birthtime_without_weakening_fstat_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "model.gguf"
    source.write_bytes(make_gguf(architecture="llama"))
    real_lstat = gguf.os.lstat
    real_fstat = gguf.os.fstat

    class WindowsPathInfo:
        def __init__(self, actual: os.stat_result):
            self.st_dev = actual.st_dev
            self.st_ino = actual.st_ino
            self.st_mode = actual.st_mode
            self.st_size = actual.st_size
            self.st_mtime_ns = actual.st_mtime_ns
            self.st_ctime_ns = actual.st_ctime_ns + 1
            self.st_file_attributes = getattr(actual, "st_file_attributes", 0)

    class ChangedDescriptorInfo(WindowsPathInfo):
        def __init__(self, actual: os.stat_result):
            super().__init__(actual)
            self.st_ctime_ns = actual.st_ctime_ns + 1

    monkeypatch.setattr(gguf.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        gguf.os,
        "lstat",
        lambda path: WindowsPathInfo(real_lstat(path)),
    )

    with gguf.open_local_gguf(source) as opened:
        assert opened.identity.changed_ns == real_fstat(opened.descriptor).st_ctime_ns

        monkeypatch.setattr(
            gguf.os,
            "fstat",
            lambda descriptor: ChangedDescriptorInfo(real_fstat(descriptor)),
        )
        with pytest.raises(gguf.GGUFSourceChangedError):
            opened.recheck()
        monkeypatch.setattr(gguf.os, "fstat", real_fstat)


@pytest.mark.parametrize(
    "field",
    ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns"),
)
def test_open_local_gguf_windows_path_comparison_rejects_stable_field_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
):
    source = tmp_path / "model.gguf"
    source.write_bytes(make_gguf(architecture="llama"))
    actual = gguf.os.lstat(source)

    class ChangedPathInfo:
        st_dev = actual.st_dev
        st_ino = actual.st_ino
        st_mode = actual.st_mode
        st_size = actual.st_size
        st_mtime_ns = actual.st_mtime_ns
        st_ctime_ns = actual.st_ctime_ns + 1
        st_file_attributes = getattr(actual, "st_file_attributes", 0)

    setattr(ChangedPathInfo, field, getattr(ChangedPathInfo, field) + 1)
    monkeypatch.setattr(gguf.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gguf.os, "lstat", lambda path: ChangedPathInfo())

    with pytest.raises(gguf.GGUFSourceChangedError):
        with gguf.open_local_gguf(source):
            pytest.fail("changed path identity must not open")


def test_validate_local_gguf_structure_accepts_generic_llm_without_wheel_policy(
    tmp_path: Path,
):
    source = tmp_path / "model.gguf"
    source.write_bytes(make_gguf(architecture="llama", name="Local LLM"))

    inspected = gguf.validate_local_gguf_structure(source)

    assert inspected.metadata.architecture == "llama"
    assert inspected.metadata.model_name == "Local LLM"
    assert inspected.source_identity.inode == os.lstat(source).st_ino


def test_transcribe_cpp_wheel_target_declaration_is_exact():
    assert gguf.TRANSCRIBE_CPP_WHEEL_TARGETS == frozenset(
        {
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("windows", "x86_64"),
            ("darwin", "arm64"),
            ("darwin", "x86_64"),
        }
    )


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", ("linux", "x86_64")),
        ("linux", "AMD64", ("linux", "x86_64")),
        ("LINUX", "arm64", ("linux", "aarch64")),
        ("Linux", "aarch64", ("linux", "aarch64")),
        ("Windows", "AMD64", ("windows", "x86_64")),
        ("Darwin", "arm64", ("darwin", "arm64")),
        ("Darwin", "aarch64", ("darwin", "arm64")),
        ("Darwin", "x86_64", ("darwin", "x86_64")),
    ],
)
def test_normalize_platform_target_accepts_real_spellings_and_aliases(
    system: str,
    machine: str,
    expected: tuple[str, str],
):
    assert gguf.normalize_platform_target(system, machine) == expected


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("Windows", "ARM64"),
        ("Linux", "riscv64"),
        ("Windows", "x86"),
        ("Plan9", "x86_64"),
        ("", ""),
    ],
)
def test_normalize_platform_target_rejects_unsupported_pair(
    system: str,
    machine: str,
):
    with pytest.raises(gguf.GGUFPlatformError, match="unavailable"):
        gguf.normalize_platform_target(system, machine)


def test_validate_local_gguf_returns_path_private_admission_with_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    model_path = tmp_path / "chosen.gguf"
    payload = make_gguf(architecture="whisper", name="Whisper Small")
    model_path.write_bytes(payload)
    expected = os.lstat(model_path)

    result = gguf.validate_local_gguf(model_path)

    assert result.path == model_path.absolute()
    assert result.metadata.architecture == "whisper"
    assert result.source_identity.device == expected.st_dev
    assert result.source_identity.inode == expected.st_ino
    assert result.source_identity.mode == expected.st_mode
    assert result.source_identity.size_bytes == len(payload)
    assert result.source_identity.modified_ns == expected.st_mtime_ns
    assert result.source_identity.changed_ns == expected.st_ctime_ns
    assert result.platform_target == ("darwin", "arm64")
    assert str(model_path) not in repr(result)


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_validate_local_gguf_rejects_non_regular_sources_without_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    _supported_runtime(monkeypatch)
    secret_path = tmp_path / "private-model.gguf"
    if kind == "directory":
        secret_path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.gguf"
        target.write_bytes(make_gguf())
        try:
            secret_path.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is unavailable")

    with pytest.raises(gguf.GGUFPathError) as raised:
        gguf.validate_local_gguf(secret_path)

    assert str(secret_path) not in str(raised.value)
    assert str(secret_path) not in repr(raised.value)


def test_validate_local_gguf_uses_project_validator_without_resolving_final_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    model_path = tmp_path / "chosen.gguf"
    model_path.write_bytes(make_gguf())
    observed: dict[str, object] = {}

    def validate(
        value: str | Path,
        require_exists: bool | None = None,
        *,
        probe_existing: bool | None = None,
    ) -> Path:
        observed.update(
            value=value,
            require_exists=require_exists,
            probe_existing=probe_existing,
        )
        return Path(value)

    monkeypatch.setattr(gguf, "validate_path_simple", validate, raising=False)

    gguf.validate_local_gguf(model_path)

    assert observed == {
        "value": model_path,
        "require_exists": False,
        "probe_existing": False,
    }


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_validate_local_gguf_rejects_fifo_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "private-model.gguf"
    os.mkfifo(selected)

    def unexpected_open(path: str | Path, flags: int) -> int:
        raise AssertionError("os.open must not be called for an irregular source")

    monkeypatch.setattr(gguf.os, "open", unexpected_open)

    with pytest.raises(gguf.GGUFPathError) as raised:
        gguf.validate_local_gguf(selected)

    assert type(raised.value) is gguf.GGUFPathError


def test_validate_local_gguf_accepts_case_insensitive_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.GGUF"
    selected.write_bytes(make_gguf())

    result = gguf.validate_local_gguf(selected)

    assert result.path == selected.absolute()


def test_validate_local_gguf_rejects_nonterminal_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf.tmp"
    selected.write_bytes(make_gguf())

    with pytest.raises(gguf.GGUFPathError):
        gguf.validate_local_gguf(selected)


def test_validate_local_gguf_redacts_project_validator_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "private-model.gguf"
    raw_message = f"validator exposed {selected}"

    def reject(*args: object, **kwargs: object) -> Path:
        raise ValueError(raw_message)

    monkeypatch.setattr(gguf, "validate_path_simple", reject, raising=False)

    with pytest.raises(gguf.GGUFPathError) as raised:
        gguf.validate_local_gguf(selected)

    assert str(selected) not in str(raised.value)
    assert str(selected) not in repr(raised.value)
    assert raw_message not in str(raised.value)
    assert raised.value.__suppress_context__


def test_validate_local_gguf_rejects_replacement_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    replacement = tmp_path / "replacement.gguf"
    selected.write_bytes(make_gguf(architecture="whisper"))
    replacement.write_bytes(make_gguf(architecture="parakeet"))
    real_open = gguf.os.open
    opened: list[int] = []

    def replace_then_open(path: str | Path, flags: int) -> int:
        replacement.replace(selected)
        descriptor = real_open(path, flags)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(gguf.os, "open", replace_then_open)

    with pytest.raises(gguf.GGUFSourceChangedError):
        gguf.validate_local_gguf(selected)

    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="nonblocking FIFO replacement is unavailable",
)
def test_validate_local_gguf_classifies_regular_to_irregular_replacement_as_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    replacement = tmp_path / "replacement.gguf"
    selected.write_bytes(make_gguf())
    os.mkfifo(replacement)
    real_open = gguf.os.open
    opened: list[int] = []

    def replace_then_open(path: str | Path, flags: int) -> int:
        assert flags & os.O_NONBLOCK
        selected.unlink()
        replacement.replace(selected)
        descriptor = real_open(path, flags)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(gguf.os, "open", replace_then_open)

    with pytest.raises(gguf.GGUFSourceChangedError):
        gguf.validate_local_gguf(selected)

    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_validate_local_gguf_inspects_same_open_descriptor_and_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    selected.write_bytes(make_gguf(tensors=(TensorFixture(data=b"x" * (64 * 1024)),)))
    real_open = gguf.os.open
    real_inspect = gguf.inspect_gguf_structure
    opened: list[int] = []
    inspected: list[int] = []

    def capture_open(path: str | Path, flags: int) -> int:
        descriptor = real_open(path, flags)
        opened.append(descriptor)
        return descriptor

    def inspect(handle: object, *, file_size: int) -> gguf.GGUFMetadata:
        descriptor = handle.fileno()
        inspected.append(descriptor)
        assert os.fstat(descriptor).st_size == file_size
        metadata = real_inspect(handle, file_size=file_size)
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == metadata.data_offset
        return metadata

    monkeypatch.setattr(gguf.os, "open", capture_open)
    monkeypatch.setattr(gguf, "inspect_gguf_structure", inspect)

    gguf.validate_local_gguf(selected)

    assert inspected == opened
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_validate_local_gguf_rechecks_name_when_nofollow_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    backing = tmp_path / "backing.gguf"
    selected.write_bytes(make_gguf())
    real_open = gguf.os.open
    real_flags = gguf._read_only_no_follow_flags()
    opened: list[int] = []

    def flags_without_nofollow() -> int:
        return real_flags & ~getattr(os, "O_NOFOLLOW", 0)

    def replace_with_same_inode_symlink(path: str | Path, flags: int) -> int:
        selected.replace(backing)
        try:
            selected.symlink_to(backing)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        descriptor = real_open(path, flags)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(gguf, "_read_only_no_follow_flags", flags_without_nofollow)
    monkeypatch.setattr(gguf.os, "open", replace_with_same_inode_symlink)

    with pytest.raises(gguf.GGUFPathError):
        gguf.validate_local_gguf(selected)

    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_validate_local_gguf_source_change_wins_when_inspection_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    selected.write_bytes(make_gguf())
    real_fstat = gguf.os.fstat
    calls = 0
    inspected: list[int] = []

    def changing_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        info = real_fstat(descriptor)
        if calls < 3:
            return info
        values = list(info)
        values[6] = info.st_size + 1
        return os.stat_result(values)

    def malformed(handle: object, *, file_size: int) -> gguf.GGUFMetadata:
        inspected.append(handle.fileno())
        raise gguf.GGUFParseError("malformed test fixture")

    monkeypatch.setattr(gguf.os, "fstat", changing_fstat)
    monkeypatch.setattr(gguf, "inspect_gguf_structure", malformed)

    with pytest.raises(gguf.GGUFSourceChangedError):
        gguf.validate_local_gguf(selected)

    assert len(inspected) == 1
    with pytest.raises(OSError):
        real_fstat(inspected[0])


def test_validate_local_gguf_preserves_parser_error_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _supported_runtime(monkeypatch)
    selected = tmp_path / "chosen.gguf"
    selected.write_bytes(make_gguf())
    parser_error = gguf.GGUFParseError("malformed test fixture")
    inspected: list[int] = []

    def malformed(handle: object, *, file_size: int) -> gguf.GGUFMetadata:
        inspected.append(handle.fileno())
        raise parser_error

    monkeypatch.setattr(gguf, "inspect_gguf_structure", malformed)

    with pytest.raises(gguf.GGUFParseError) as raised:
        gguf.validate_local_gguf(selected)

    assert raised.value is parser_error
    assert len(inspected) == 1
    with pytest.raises(OSError):
        os.fstat(inspected[0])


def test_admission_import_boundary_allows_only_parser_dependencies():
    source = Path(gguf.__file__).read_text(encoding="utf-8")

    assert _unapproved_admission_import_targets(source) == set()
    assert not hasattr(gguf, "select_gguf_descriptor")


@pytest.mark.parametrize(
    "source",
    [
        "from . import store",
        "from . import os",
        "from .dataclasses import dataclass",
        "from tldw_chatbook.Model_Artifacts import service",
        "import httpx",
        "import urllib.request",
        "import socket",
        "import ctypes",
        "import cffi",
        "from textual.app import App",
        '__import__("httpx")',
        "from dataclasses import *",
    ],
)
def test_admission_import_boundary_rejects_disallowed_targets(source: str):
    assert _unapproved_admission_import_targets(source)


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "from dataclasses import dataclass",
        "from ..Utils.path_validation import validate_path_simple",
    ],
)
def test_admission_import_boundary_allows_approved_targets(source: str):
    assert _unapproved_admission_import_targets(source) == set()


def test_deferred_gguf_managed_import_is_source_only_and_unreferenced():
    model_artifacts = Path(gguf.__file__).parent
    deferred_path = model_artifacts / "_deferred_gguf_managed_import.py"

    source = deferred_path.read_text(encoding="utf-8")
    _assert_deferred_source_contract(source)

    init_source = (model_artifacts / "__init__.py").read_text(encoding="utf-8")
    assert "_deferred_gguf_managed_import" not in init_source

    references = [
        path
        for path in model_artifacts.parent.rglob("*.py")
        if path != deferred_path
        and "_deferred_gguf_managed_import" in path.read_text(encoding="utf-8")
    ]
    assert references == []
