"""Tests for immutable model-artifact descriptor contracts."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Model_Artifacts import service as service_module
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


def source_tree(
    tmp_path: Path,
    files: dict[str, bytes],
) -> tuple[Path, tuple[ArtifactFile, ...]]:
    """Create a source directory and matching descriptor file metadata."""

    source = tmp_path / "source"
    source.mkdir()
    expected = []
    for relative_path, content in files.items():
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        expected.append(artifact_file(content, relative_path))
    return source, tuple(expected)


def install_inputs(
    tmp_path: Path,
    files: dict[str, bytes] | None = None,
) -> tuple[object, ArtifactDescriptor, Path]:
    """Build a service, descriptor, and matching source directory."""

    service = service_module.ModelArtifactService(tmp_path / "store")
    source, expected = source_tree(
        tmp_path,
        files or {"model.onnx": b"model"},
    )
    return service, descriptor(files=expected), source


def installed_artifact(
    tmp_path: Path,
) -> tuple[object, ArtifactDescriptor, Path, Path]:
    """Install one valid artifact and return service, descriptor, source, final."""

    service, item, source = install_inputs(tmp_path)
    service.install(item, source)
    return service, item, source, service.artifact_path(item.reference)


def symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool) -> None:
    """Create a test symlink or skip when the platform forbids it."""

    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")


def regular_tree_size(root: Path) -> int:
    """Count logical regular-file bytes without following links."""

    total = 0
    for entry in os.scandir(root):
        mode = entry.stat(follow_symlinks=False).st_mode
        if stat.S_ISREG(mode):
            total += entry.stat(follow_symlinks=False).st_size
        elif stat.S_ISDIR(mode):
            total += regular_tree_size(Path(entry.path))
    return total


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


def test_descriptor_accepts_printable_unicode_metadata() -> None:
    item = descriptor(usage_notice="Modèle validé — prêt.")

    assert item.usage_notice == "Modèle validé — prêt."


@pytest.mark.parametrize("field", ("source_url", "license_url"))
def test_descriptor_rejects_zero_width_url_hostname(field: str) -> None:
    value = "https://exa\u200bmple.test/model"

    with pytest.raises(ArtifactDescriptorValidationError, match=field):
        descriptor(**{field: value})


@pytest.mark.parametrize("field", ("source_url", "license_url"))
def test_descriptor_parser_rejects_zero_width_url_hostname(field: str) -> None:
    encoded = descriptor().to_dict()
    encoded[field] = "https://exa\u200bmple.test/model"

    with pytest.raises(ArtifactDescriptorParseError, match=field):
        ArtifactDescriptor.from_dict(encoded)


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


def test_install_verifies_then_promotes_immutable_directory(tmp_path: Path) -> None:
    service, item, source = install_inputs(
        tmp_path,
        {"models/model.onnx": b"model", "tokens.txt": b"tokens"},
    )

    assert service.install(item, source) == item.reference
    final = service.artifact_path(item.reference)
    assert (final / "models" / "model.onnx").read_bytes() == b"model"
    assert (final / "tokens.txt").read_bytes() == b"tokens"
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1,
        "descriptor": item.to_dict(),
    }


def test_service_validates_root_and_creates_only_owned_layout(tmp_path: Path) -> None:
    root = tmp_path / "store"
    service = service_module.ModelArtifactService(root)

    assert service.artifacts_path == root.resolve() / "artifacts"
    assert service.staging_path == root.resolve() / "staging"
    assert {path.name for path in root.iterdir()} == {
        "active",
        "artifacts",
        "locks",
        "ready",
        "staging",
    }

    with pytest.raises(TypeError):
        service_module.ModelArtifactService(str(root))  # type: ignore[arg-type]
    for invalid_timeout in (True, -1.0, float("inf"), float("nan"), "5"):
        with pytest.raises(ValueError):
            service_module.ModelArtifactService(
                tmp_path / f"invalid-{invalid_timeout!s}",
                lease_timeout_seconds=invalid_timeout,  # type: ignore[arg-type]
            )


def test_service_rejects_invalid_or_symlinked_managed_paths(tmp_path: Path) -> None:
    with pytest.raises(service_module.ArtifactPathError):
        service_module.ModelArtifactService(tmp_path / "bad\0root")

    root = tmp_path / "store"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    symlink_or_skip(root / "artifacts", external, target_is_directory=True)
    with pytest.raises(service_module.ArtifactPathError):
        service_module.ModelArtifactService(root)

    assert tuple(external.iterdir()) == ()


def test_install_validates_argument_types_before_staging_mutation(
    tmp_path: Path,
) -> None:
    service, item, source = install_inputs(tmp_path)

    with pytest.raises(TypeError):
        service.install(object(), source)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        service.install(item, str(source))  # type: ignore[arg-type]
    with pytest.raises(service_module.ArtifactPathError):
        service.install(item, tmp_path / "bad\0source")

    assert tuple(service.staging_path.iterdir()) == ()


@pytest.mark.parametrize(
    "expected_file",
    (
        ArtifactFile("model.onnx", 5, "0" * 64),
        ArtifactFile("model.onnx", 4, hashlib.sha256(b"nope").hexdigest()),
    ),
    ids=("size", "hash"),
)
def test_install_integrity_failure_never_creates_final_directory(
    tmp_path: Path,
    expected_file: ArtifactFile,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    source, _ = source_tree(tmp_path, {"model.onnx": b"fail"})
    item = descriptor(files=(expected_file,))

    with pytest.raises(service_module.ArtifactIntegrityError):
        service.install(item, source)

    assert service.artifact_path(item.reference).exists() is False


@pytest.mark.parametrize(
    "unsafe_entry",
    (
        "missing",
        "extra_file",
        "extra_empty_directory",
        "extra_symlink",
        "declared_file_symlink",
        "nested_file_symlink",
        "declared_directory_symlink",
        "nested_directory_symlink",
    ),
)
def test_install_rejects_incomplete_or_unsafe_source_tree(
    tmp_path: Path,
    unsafe_entry: str,
) -> None:
    paths = {
        "nested_directory_symlink": "outer/inner/model.onnx",
        "declared_directory_symlink": "nested/model.onnx",
        "nested_file_symlink": "nested/model.onnx",
    }
    relative = paths.get(unsafe_entry, "model.onnx")
    service, item, source = install_inputs(tmp_path, {relative: b"model"})
    declared = source / relative
    external = tmp_path / "external"

    if unsafe_entry == "missing":
        declared.unlink()
    elif unsafe_entry == "extra_file":
        (source / "extra.bin").write_bytes(b"extra")
    elif unsafe_entry == "extra_empty_directory":
        (source / "empty").mkdir()
    elif unsafe_entry == "extra_symlink":
        symlink_or_skip(
            source / "extra-link",
            declared,
            target_is_directory=False,
        )
    elif unsafe_entry in {"declared_file_symlink", "nested_file_symlink"}:
        external.write_bytes(b"model")
        declared.unlink()
        symlink_or_skip(declared, external, target_is_directory=False)
    elif unsafe_entry == "declared_directory_symlink":
        external.mkdir()
        (external / "model.onnx").write_bytes(b"model")
        shutil.rmtree(source / "nested")
        symlink_or_skip(source / "nested", external, target_is_directory=True)
    else:
        external.mkdir()
        (external / "model.onnx").write_bytes(b"model")
        shutil.rmtree(source / "outer" / "inner")
        symlink_or_skip(
            source / "outer" / "inner",
            external,
            target_is_directory=True,
        )

    with pytest.raises(service_module.ArtifactPathError):
        service.install(item, source)

    assert service.artifact_path(item.reference).exists() is False


def test_install_rejects_symlinked_source_directory(tmp_path: Path) -> None:
    service, item, source = install_inputs(tmp_path)
    linked_source = tmp_path / "linked-source"
    symlink_or_skip(linked_source, source, target_is_directory=True)

    with pytest.raises(service_module.ArtifactPathError):
        service.install(item, linked_source)

    assert service.artifact_path(item.reference).exists() is False
    assert tuple(service.staging_path.iterdir()) == ()


def test_install_rejects_symlinked_source_ancestor(tmp_path: Path) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    real_parent = tmp_path / "real-parent"
    source = real_parent / "source"
    source.mkdir(parents=True)
    (source / "model.onnx").write_bytes(b"model")
    linked_parent = tmp_path / "linked-parent"
    symlink_or_skip(linked_parent, real_parent, target_is_directory=True)
    item = descriptor(files=(artifact_file(b"model"),))

    with pytest.raises(service_module.ArtifactPathError):
        service.install(item, linked_parent / "source")

    assert service.artifact_path(item.reference).exists() is False
    assert tuple(service.staging_path.iterdir()) == ()


def test_install_rejects_source_directory_identity_change_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, item, source = install_inputs(tmp_path)
    abandoned = service.staging_path / "abandoned"
    abandoned.mkdir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "model.onnx").write_bytes(b"model")
    original_copy = service._copy_payload

    def swap_then_copy(
        copied_descriptor: ArtifactDescriptor,
        copied_source: Path,
        staging: Path,
    ) -> None:
        copied_source.rename(tmp_path / "original-source")
        replacement.rename(copied_source)
        original_copy(copied_descriptor, copied_source, staging)

    monkeypatch.setattr(service, "_copy_payload", swap_then_copy)

    with pytest.raises(service_module.ArtifactPathError):
        service.install(item, source)

    assert service.artifact_path(item.reference).exists() is False
    assert tuple(service.staging_path.iterdir()) == (abandoned,)


def test_install_rejects_special_source_entry_when_supported(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    service, item, source = install_inputs(tmp_path)
    try:
        os.mkfifo(source / "pipe")
    except OSError as error:
        pytest.skip(f"FIFO creation is unavailable: {error}")

    with pytest.raises(service_module.ArtifactPathError):
        service.install(item, source)

    assert service.artifact_path(item.reference).exists() is False


def test_identical_reinstall_is_idempotent_and_rehashes_payload(
    tmp_path: Path,
) -> None:
    service, item, source, final = installed_artifact(tmp_path)

    assert service.install(item, source) == item.reference
    assert tuple(service.staging_path.iterdir()) == ()

    (final / "model.onnx").write_bytes(b"x" * item.files[0].size_bytes)
    with pytest.raises(service_module.ArtifactIntegrityError):
        service.install(item, source)

    assert (final / "model.onnx").read_bytes() == b"x" * item.files[0].size_bytes


@pytest.mark.parametrize("corruption", ("missing", "extra", "symlink"))
def test_matching_manifest_with_invalid_payload_is_integrity_failure(
    tmp_path: Path,
    corruption: str,
) -> None:
    service, item, source, final = installed_artifact(tmp_path)
    payload = final / "model.onnx"
    if corruption == "missing":
        payload.unlink()
    elif corruption == "extra":
        (final / "unexpected.bin").write_bytes(b"unexpected")
    else:
        payload.unlink()
        symlink_or_skip(payload, source / "model.onnx", target_is_directory=False)

    with pytest.raises(service_module.ArtifactIntegrityError):
        service.install(item, source)


@pytest.mark.parametrize("populated", (False, True))
def test_install_never_replaces_existing_destination(
    tmp_path: Path,
    populated: bool,
) -> None:
    service, item, source = install_inputs(tmp_path)
    destination = service.artifact_path(item.reference)
    destination.mkdir(parents=True)
    if populated:
        (destination / "keep").write_bytes(b"existing")

    with pytest.raises(service_module.ArtifactConflictError):
        service.install(item, source)

    assert destination.is_dir()
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == ({"keep": b"existing"} if populated else {})


@pytest.mark.parametrize("conflict", ("invalid_manifest", "different_descriptor"))
def test_install_preserves_conflicting_existing_artifact(
    tmp_path: Path,
    conflict: str,
) -> None:
    service, item, source = install_inputs(tmp_path)
    destination = service.artifact_path(item.reference)
    destination.mkdir(parents=True)
    (destination / "keep").write_bytes(b"existing")
    if conflict == "invalid_manifest":
        (destination / "manifest.json").write_text("{}", encoding="utf-8")
    else:
        different = descriptor(
            reference=item.reference,
            files=item.files,
            model_id="other/model",
        )
        (destination / "manifest.json").write_text(
            json.dumps(
                {"schema_version": 1, "descriptor": different.to_dict()},
            ),
            encoding="utf-8",
        )
    before = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.iterdir()
    }

    with pytest.raises(service_module.ArtifactConflictError):
        service.install(item, source)

    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.iterdir()
    } == before


def test_install_rejects_managed_ancestor_symlink_before_external_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, item, source = install_inputs(tmp_path)
    external = tmp_path / "external-artifact"
    external_final = external / item.reference.revision / item.reference.variant
    external_final.mkdir(parents=True)
    (external_final / "model.onnx").write_bytes(b"model")
    (external_final / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "descriptor": item.to_dict()},
        ),
        encoding="utf-8",
    )
    managed_ancestor = service.artifacts_path / item.reference.artifact_id
    symlink_or_skip(managed_ancestor, external, target_is_directory=True)
    before = {
        path.relative_to(external).as_posix(): path.read_bytes()
        for path in external.rglob("*")
        if path.is_file()
    }

    def forbid_manifest_read(_directory: Path) -> ArtifactDescriptor:
        raise AssertionError("external manifest must not be read")

    monkeypatch.setattr(service, "_read_manifest", forbid_manifest_read)

    with pytest.raises(service_module.ArtifactPathError):
        service.install(item, source)

    assert {
        path.relative_to(external).as_posix(): path.read_bytes()
        for path in external.rglob("*")
        if path.is_file()
    } == before
    assert tuple(service.staging_path.iterdir()) == ()


@pytest.mark.parametrize("failure", ("copy", "hash", "promotion"))
def test_failed_install_removes_only_operation_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    service, item, source = install_inputs(tmp_path)
    abandoned = service.staging_path / "pre-existing" / "part"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b"keep")
    method_name = {
        "copy": "_copy_payload",
        "hash": "_verify_payload",
        "promotion": "_promote",
    }[failure]

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"injected {failure} failure")

    monkeypatch.setattr(service, method_name, fail)

    with pytest.raises(service_module.ArtifactError) as caught:
        service.install(item, source)

    assert isinstance(caught.value.__cause__, OSError)
    assert tuple(service.staging_path.iterdir()) == (abandoned.parent,)
    assert abandoned.read_bytes() == b"keep"
    assert service.artifact_path(item.reference).exists() is False


@pytest.mark.parametrize("blocked_key", ("lifecycle", "target"))
def test_services_contend_on_lifecycle_and_target_writer_leases(
    tmp_path: Path,
    blocked_key: str,
) -> None:
    root = tmp_path / "store"
    first = service_module.ModelArtifactService(root)
    second = service_module.ModelArtifactService(
        root,
        lease_timeout_seconds=0.01,
    )
    source, files = source_tree(tmp_path, {"model.onnx": b"model"})
    item = descriptor(files=files)
    key = (
        ArtifactLeaseKey("!lifecycle", "1", "writer")
        if blocked_key == "lifecycle"
        else item.reference.lease_key()
    )

    with service_module.ArtifactOperationLease(
        root / "locks",
        key,
        service_module.LeaseMode.EXCLUSIVE,
    ):
        with pytest.raises(service_module.ArtifactStateError) as caught:
            second.install(item, source)

    assert isinstance(
        caught.value.__cause__,
        service_module.ArtifactLeaseError,
    )
    assert first.artifact_path(item.reference).exists() is False
    assert tuple(second.staging_path.iterdir()) == ()


def test_install_acquires_exact_writer_leases_in_fixed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, item, source = install_inputs(tmp_path)
    acquired: list[tuple[ArtifactLeaseKey, object]] = []

    class RecordingLease:
        def __init__(
            self,
            _lock_root: Path,
            key: ArtifactLeaseKey,
            mode: object,
            **_kwargs: object,
        ) -> None:
            self._entry = (key, mode)

        def __enter__(self) -> RecordingLease:
            acquired.append(self._entry)
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(service_module, "ArtifactOperationLease", RecordingLease)

    service.install(item, source)

    assert acquired == [
        (
            ArtifactLeaseKey("!lifecycle", "1", "writer"),
            service_module.LeaseMode.EXCLUSIVE,
        ),
        (item.reference.lease_key(), service_module.LeaseMode.EXCLUSIVE),
    ]


def test_inventory_is_deterministic_visible_strict_and_hash_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, item, _source, final = installed_artifact(tmp_path)
    malformed = service.artifact_path(ref("malformed", "revision", "v1"))
    malformed.mkdir(parents=True)
    (malformed / "manifest.json").write_text("{}", encoding="utf-8")
    mismatch = service.artifact_path(ref("mismatch", "revision", "v1"))
    shutil.copytree(final, mismatch)
    unexpected = service.artifacts_path / "orphan"
    unexpected.write_bytes(b"not a directory")
    incomplete = service.artifacts_path / "partial"
    incomplete.mkdir()
    linked = service.artifacts_path / "symlinked"
    symlink_or_skip(linked, final, target_is_directory=True)

    def reject_hashing(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("inventory must not hash payloads")

    real_import = builtins.__import__

    def reject_runtime_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] in {
            "httpx",
            "llama_cpp",
            "onnxruntime",
            "transformers",
        }:
            raise AssertionError(f"inventory imported runtime/client {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(service_module.hashlib, "sha256", reject_hashing)
    monkeypatch.setattr(builtins, "__import__", reject_runtime_import)

    installed = service.list_installed()

    assert [entry.path for entry in installed] == sorted(
        (final, malformed, mismatch, unexpected, incomplete, linked),
        key=lambda path: path.as_posix(),
    )
    by_path = {entry.path: entry for entry in installed}
    assert by_path[final] == service_module.InstalledArtifact(
        path=final,
        descriptor=item,
        ready=False,
        active=False,
    )
    for path in (malformed, mismatch, unexpected, incomplete, linked):
        assert by_path[path].descriptor is None
        assert by_path[path].error
        assert by_path[path].ready is False
        assert by_path[path].active is False


def test_disk_usage_counts_regular_bytes_without_following_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _item, _source, final = installed_artifact(tmp_path)
    abandoned = service.staging_path / "abandoned" / "part"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b"staging")
    external_file = tmp_path / "external.bin"
    external_file.write_bytes(b"x" * 10_000)
    external_directory = tmp_path / "external-directory"
    external_directory.mkdir()
    (external_directory / "large.bin").write_bytes(b"y" * 20_000)
    symlink_or_skip(
        final / "external-file-link",
        external_file,
        target_is_directory=False,
    )
    symlink_or_skip(
        service.staging_path / "external-directory-link",
        external_directory,
        target_is_directory=True,
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: (100, 40, 60))

    usage = service.disk_usage()

    assert usage == service_module.ArtifactDiskUsage(
        installed_bytes=regular_tree_size(service.artifacts_path),
        staging_bytes=regular_tree_size(service.staging_path),
        free_bytes=60,
    )
    assert usage.installed_bytes < external_file.stat().st_size
    assert usage.staging_bytes == len(b"staging")


@pytest.mark.parametrize("owned_root", ("artifacts", "staging"))
def test_disk_usage_rejects_replaced_owned_root_symlink(
    tmp_path: Path,
    owned_root: str,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "outside.bin"
    external_file.write_bytes(b"outside")
    replaced = (
        service.artifacts_path if owned_root == "artifacts" else service.staging_path
    )
    replaced.rmdir()
    symlink_or_skip(replaced, external, target_is_directory=True)

    with pytest.raises(service_module.ArtifactPathError):
        service.disk_usage()

    assert external_file.read_bytes() == b"outside"


def test_disk_usage_rejects_directory_identity_change_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    original_scandir = service_module.os.scandir
    previous_artifacts = tmp_path / "previous-artifacts"
    swapped = False

    def swap_artifacts(path: str | os.PathLike[str]) -> object:
        nonlocal swapped
        if Path(path) == service.artifacts_path and not swapped:
            entries = list(original_scandir(path))
            service.artifacts_path.rename(previous_artifacts)
            service.artifacts_path.mkdir()
            swapped = True
            return iter(entries)
        return original_scandir(path)

    monkeypatch.setattr(service_module.os, "scandir", swap_artifacts)

    with pytest.raises(service_module.ArtifactPathError):
        service.disk_usage()

    assert swapped is True
