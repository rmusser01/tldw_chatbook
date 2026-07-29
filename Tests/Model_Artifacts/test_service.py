"""Tests for immutable model-artifact descriptor contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import pytest

from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactDescriptorParseError,
    ArtifactDescriptorValidationError,
    ArtifactFile,
    ArtifactFormat,
    ArtifactLeaseKey,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
    closure_fingerprint,
)


def ref(
    artifact_id: str = "parakeet-v2",
    revision: str = "a" * 40,
    variant: str = "int8",
) -> ArtifactRef:
    """Build a small valid artifact reference."""

    return ArtifactRef(artifact_id, revision, variant)


def artifact_file(
    content: bytes = b"x",
    path: str = "model.onnx",
) -> ArtifactFile:
    """Build file metadata matching a small byte payload."""

    return ArtifactFile(path, len(content), hashlib.sha256(content).hexdigest())


def descriptor(**overrides: Any) -> ArtifactDescriptor:
    """Build a valid descriptor with selected fields replaced."""

    values: dict[str, object] = {
        "reference": ref(),
        "model_id": "nvidia/parakeet-tdt-0.6b-v2",
        "role": ArtifactRole.ROOT,
        "format": ArtifactFormat.ONNX,
        "consumer": "stt",
        "model_family": "parakeet",
        "upstream_repository": "nvidia/parakeet-tdt-0.6b-v2",
        "upstream_revision": "main-a1b2c3",
        "source_url": "https://models.example.test/parakeet-v2",
        "precision": "int8",
        "license_id": "cc-by-4.0",
        "license_url": "https://licenses.example.test/cc-by-4.0",
        "usage_notice": "Review the upstream model card before use.",
        "runtime_name": "onnx-asr",
        "runtime_version_constraint": "==0.12.0",
        "supported_os": ("linux", "macos", "windows"),
        "supported_architectures": ("x86-64", "arm64"),
        "provenance": (
            ProvenanceClass.CHATBOOK_CURATED,
            ProvenanceClass.INTEGRITY_VERIFIED,
        ),
        "files": (artifact_file(),),
        "dependencies": (),
    }
    values.update(overrides)
    if "expected_installed_bytes" not in overrides:
        files = values["files"]
        assert isinstance(files, tuple)
        values["expected_installed_bytes"] = sum(item.size_bytes for item in files)
    return ArtifactDescriptor(**values)  # type: ignore[arg-type]


def test_ref_requires_canonical_portable_components() -> None:
    assert ArtifactRef("parakeet-v2", "a" * 40, "int8").variant == "int8"

    for value in ("", " ", "../x", "Parakeet", "con", "x ", "x/y", r"x\y"):
        with pytest.raises(ArtifactDescriptorValidationError):
            ArtifactRef(value, "a" * 40, "int8")
    with pytest.raises(ArtifactDescriptorValidationError):
        ArtifactRef("parakeet-v2", "../revision", "int8")
    with pytest.raises(ArtifactDescriptorValidationError):
        ArtifactRef("parakeet-v2", "a" * 40, "INT8")


@pytest.mark.parametrize(
    "reserved_name",
    ("aux", "nul.txt", "com1", "lpt9.data"),
)
def test_ref_rejects_windows_reserved_device_aliases(reserved_name: str) -> None:
    with pytest.raises(ArtifactDescriptorValidationError):
        ArtifactRef(reserved_name, "revision", "int8")


def test_artifact_ref_maps_exactly_to_lease_key() -> None:
    reference = ref("parakeet-v2", "immutable-revision", "int8")

    assert reference.lease_key() == ArtifactLeaseKey(
        artifact_id="parakeet-v2",
        revision="immutable-revision",
        variant="int8",
    )


@pytest.mark.parametrize(
    "path",
    (
        "",
        ".",
        "../model.onnx",
        "nested/../model.onnx",
        "/model.onnx",
        r"nested\model.onnx",
        "model.onnx/",
        "manifest.json",
        "active/state.json",
        "ready/state.json",
        "staging/model.onnx",
        "locks/model.onnx",
        "CON.onnx",
        "nested/Lpt1.bin",
    ),
)
def test_artifact_file_rejects_unsafe_or_reserved_paths(path: str) -> None:
    with pytest.raises(ArtifactDescriptorValidationError):
        ArtifactFile(path, 1, "0" * 64)


def test_descriptor_rejects_duplicate_and_casefold_file_paths() -> None:
    with pytest.raises(ArtifactDescriptorValidationError, match="duplicate"):
        descriptor(
            files=(
                ArtifactFile("model.onnx", 1, "0" * 64),
                ArtifactFile("model.onnx", 1, "1" * 64),
            ),
            expected_installed_bytes=2,
        )

    with pytest.raises(
        ArtifactDescriptorValidationError,
        match="case-insensitive",
    ):
        descriptor(
            files=(
                ArtifactFile("Model.onnx", 1, "0" * 64),
                ArtifactFile("model.onnx", 1, "1" * 64),
            ),
            expected_installed_bytes=2,
        )


@pytest.mark.parametrize("size_bytes", (-1, 1.0, True, "1"))
def test_artifact_file_requires_nonnegative_integer_size(size_bytes: object) -> None:
    with pytest.raises(ArtifactDescriptorValidationError):
        ArtifactFile("model.onnx", size_bytes, "0" * 64)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sha256",
    ("", "not-a-sha256", "A" * 64, "0" * 63, "0" * 65),
)
def test_artifact_file_requires_exact_lowercase_sha256(sha256: str) -> None:
    with pytest.raises(ArtifactDescriptorValidationError):
        ArtifactFile("model.onnx", 1, sha256)


def test_descriptor_rejects_installed_byte_mismatch() -> None:
    with pytest.raises(
        ArtifactDescriptorValidationError,
        match="installed bytes",
    ):
        descriptor(expected_installed_bytes=2, files=(artifact_file(b"x"),))


def test_descriptor_rejects_provenance_conflict_or_empty_provenance() -> None:
    with pytest.raises(ArtifactDescriptorValidationError, match="provenance"):
        descriptor(provenance=())
    with pytest.raises(ArtifactDescriptorValidationError, match="provenance"):
        descriptor(
            provenance=(
                ProvenanceClass.INTEGRITY_VERIFIED,
                ProvenanceClass.LOCAL_INTEGRITY_RECORDED,
            )
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("source_url", "https://token@example.test/model"),
        ("source_url", "https://example.test/model?sig=secret"),
        ("source_url", "https://example.test/model#section"),
        ("source_url", "https://example .test/model"),
        ("source_url", "file:///tmp/model"),
        ("source_url", "https:///model"),
        ("license_url", "https://token@example.test/license"),
        ("license_url", "https://example.test/license?token=secret"),
    ),
)
def test_descriptor_rejects_unsafe_provenance_urls(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ArtifactDescriptorValidationError, match=field):
        descriptor(**{field: value})


def test_descriptor_accepts_valid_url_with_ipv6_hostname() -> None:
    item = descriptor(
        source_url="https://[2001:db8::1]/model",
        license_url="http://[2001:db8::2]:8080/license",
    )

    assert item.source_url == "https://[2001:db8::1]/model"
    assert ArtifactDescriptor.from_dict(item.to_dict()) == item


@pytest.mark.parametrize("field", ("source_url", "license_url"))
@pytest.mark.parametrize(
    "value",
    (
        r"https://example.test\evil.com/model",
        "https://./model",
        "https://example.test/%zz",
    ),
)
def test_descriptor_rejects_malformed_http_urls(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ArtifactDescriptorValidationError, match=field):
        descriptor(**{field: value})


@pytest.mark.parametrize("field", ("source_url", "license_url"))
@pytest.mark.parametrize(
    "value",
    (
        r"https://example.test\evil.com/model",
        "https://./model",
        "https://example.test/%zz",
    ),
)
def test_descriptor_parser_rejects_malformed_http_urls(
    field: str,
    value: str,
) -> None:
    encoded = descriptor().to_dict()
    encoded[field] = value

    with pytest.raises(ArtifactDescriptorParseError, match=field):
        ArtifactDescriptor.from_dict(encoded)


def test_descriptor_rejects_duplicate_or_conflicting_dependencies() -> None:
    dependency = ref("silero-vad")
    with pytest.raises(ArtifactDescriptorValidationError, match="duplicate"):
        descriptor(dependencies=(dependency, dependency))

    conflicting = (
        ArtifactRef("silero-vad", "a" * 40, "int8"),
        ArtifactRef("silero-vad", "b" * 40, "int8"),
    )
    with pytest.raises(ArtifactDescriptorValidationError, match="conflicting"):
        descriptor(dependencies=conflicting)

    conflicting_variant = (
        ArtifactRef("silero-vad", "a" * 40, "int8"),
        ArtifactRef("silero-vad", "a" * 40, "fp32"),
    )
    with pytest.raises(ArtifactDescriptorValidationError, match="conflicting"):
        descriptor(dependencies=conflicting_variant)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda values: values.__setitem__("role", "root"),
        lambda values: values.__setitem__("format", "onnx"),
        lambda values: values.__setitem__("supported_os", ["linux"]),
        lambda values: values.__setitem__("files", [artifact_file()]),
        lambda values: values.__setitem__("expected_installed_bytes", True),
        lambda values: values.__setitem__("consumer", " stt"),
        lambda values: values.__setitem__("precision", "fp32"),
    ),
)
def test_descriptor_direct_construction_is_strict(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    item = descriptor()
    values = {
        field: getattr(item, field) for field in ArtifactDescriptor.__dataclass_fields__
    }
    mutation(values)

    with pytest.raises(ArtifactDescriptorValidationError):
        ArtifactDescriptor(**values)  # type: ignore[arg-type]


def test_descriptor_serialization_round_trips_deterministically() -> None:
    item = descriptor(
        dependencies=(ref("silero-vad", "vad-revision", "int8"),),
        files=(
            artifact_file(b"model", "models/Model.onnx"),
            artifact_file(b"weights", "models/weights.bin"),
        ),
    )

    encoded = item.to_dict()

    assert encoded["schema_version"] == 1
    assert ArtifactDescriptor.from_dict(encoded).to_dict() == encoded
    assert ArtifactDescriptor.from_dict(encoded) == item


def test_descriptor_parser_rejects_unsupported_schema() -> None:
    encoded = descriptor().to_dict()
    encoded["schema_version"] = 2

    with pytest.raises(ArtifactDescriptorParseError, match="schema_version"):
        ArtifactDescriptor.from_dict(encoded)


@pytest.mark.parametrize("field", ("reference", "model_id", "dependencies"))
def test_descriptor_parser_rejects_missing_fields(field: str) -> None:
    encoded = descriptor().to_dict()
    del encoded[field]

    with pytest.raises(ArtifactDescriptorParseError, match="keys"):
        ArtifactDescriptor.from_dict(encoded)


def test_descriptor_parser_rejects_unknown_fields() -> None:
    encoded = descriptor().to_dict()
    encoded["unexpected"] = "value"

    with pytest.raises(ArtifactDescriptorParseError, match="keys"):
        ArtifactDescriptor.from_dict(encoded)


@pytest.mark.parametrize(
    "field,value",
    (
        ("model_id", 123),
        ("role", True),
        ("expected_installed_bytes", True),
        ("supported_os", "linux"),
        ("files", "model.onnx"),
        ("dependencies", {}),
    ),
)
def test_descriptor_parser_rejects_mistyped_fields(
    field: str,
    value: object,
) -> None:
    encoded = descriptor().to_dict()
    encoded[field] = value

    with pytest.raises(ArtifactDescriptorParseError):
        ArtifactDescriptor.from_dict(encoded)


def test_descriptor_parser_rejects_unknown_or_mistyped_nested_fields() -> None:
    encoded = descriptor().to_dict()
    reference = encoded["reference"]
    assert isinstance(reference, dict)
    reference["extra"] = "unsafe"
    with pytest.raises(ArtifactDescriptorParseError, match="reference"):
        ArtifactDescriptor.from_dict(encoded)

    encoded = descriptor().to_dict()
    files = encoded["files"]
    assert isinstance(files, list)
    first_file = files[0]
    assert isinstance(first_file, dict)
    first_file["size_bytes"] = True
    with pytest.raises(ArtifactDescriptorParseError, match="files"):
        ArtifactDescriptor.from_dict(encoded)


@pytest.mark.parametrize("target", ("descriptor", "reference", "file"))
def test_descriptor_parser_rejects_mixed_type_extra_keys(
    target: str,
) -> None:
    encoded = descriptor().to_dict()
    if target == "descriptor":
        mapping = encoded
    elif target == "reference":
        mapping = encoded["reference"]
    else:
        files = encoded["files"]
        assert isinstance(files, list)
        mapping = files[0]
    assert isinstance(mapping, dict)
    mapping[1] = "unexpected"
    mapping[None] = "unexpected"

    with pytest.raises(ArtifactDescriptorParseError, match="keys"):
        ArtifactDescriptor.from_dict(encoded)


def test_descriptor_parser_reruns_value_validation() -> None:
    encoded = descriptor().to_dict()
    encoded["source_url"] = "https://token@example.test/model"

    with pytest.raises(ArtifactDescriptorParseError, match="source_url"):
        ArtifactDescriptor.from_dict(encoded)


def test_closure_fingerprint_is_stable_for_order_and_duplicates() -> None:
    root = ref("parakeet-v2")
    vad = ref("silero-vad", "vad-revision", "int8")
    tokenizer = ref("parakeet-tokenizer", "tokenizer-revision", "v1")

    canonical = closure_fingerprint(root, (vad, tokenizer))

    assert (
        canonical == "35cef45d39b7eee2189f60a1130fe8b75953cfedc9c4a4abfefc32ad48c1effa"
    )
    assert canonical == closure_fingerprint(root, (tokenizer, vad))
    assert canonical == closure_fingerprint(root, (vad, root, tokenizer, vad))
    assert len(canonical) == 64
    assert set(canonical) <= set("0123456789abcdef")


def test_closure_fingerprint_changes_with_exact_closure() -> None:
    root = ref("parakeet-v2")
    vad = ref("silero-vad", "vad-revision", "int8")

    assert closure_fingerprint(root, ()) != closure_fingerprint(root, (vad,))
    assert closure_fingerprint(root, (vad,)) != closure_fingerprint(
        root,
        (ref("silero-vad", "new-vad-revision", "int8"),),
    )
