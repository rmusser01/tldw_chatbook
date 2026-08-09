"""Focused tests for descriptor-verified external Parakeet roots."""

from __future__ import annotations

import hashlib
import os
import traceback
from dataclasses import replace
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
)
from tldw_chatbook.STT.parakeet_external import (
    ExternalParakeetErrorCode,
    ExternalParakeetVerificationError,
    ExternalParakeetVerifier,
)


V2_MODEL_ID = "nemo-parakeet-tdt-0.6b-v2"
V3_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"
HASH_CHUNK_BYTES = 64 * 1024


def _artifact_file(path: str, payload: bytes) -> ArtifactFile:
    return ArtifactFile(
        path=path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _descriptor(
    files: dict[str, bytes],
    *,
    model_id: str = V2_MODEL_ID,
    precision: str = "int8",
) -> ArtifactDescriptor:
    version = "v3" if model_id == V3_MODEL_ID else "v2"
    artifact_files = tuple(
        _artifact_file(path, payload) for path, payload in files.items()
    )
    return ArtifactDescriptor(
        reference=ArtifactRef(f"parakeet-{version}", "tiny-revision", precision),
        model_id=model_id,
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="stt",
        model_family="parakeet",
        upstream_repository=f"example/parakeet-{version}",
        upstream_revision="tiny-revision",
        source_url="https://example.invalid/model.onnx",
        precision=precision,
        expected_installed_bytes=sum(file.size_bytes for file in artifact_files),
        license_id="cc-by-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        usage_notice="Tiny verifier fixture.",
        runtime_name="onnx-asr",
        runtime_version_constraint="==0.12.0",
        supported_os=("linux", "darwin", "windows"),
        supported_architectures=("x86-64", "arm64"),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=artifact_files,
        dependencies=(ArtifactRef("silero-vad", "tiny-revision", "f32"),),
    )


def _materialize(directory: Path, files: dict[str, bytes]) -> None:
    directory.mkdir()
    for relative_path, payload in files.items():
        path = directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool = False,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")


def _assert_error(
    code: ExternalParakeetErrorCode,
    descriptor: ArtifactDescriptor,
    directory: Path,
    **kwargs: object,
) -> ExternalParakeetVerificationError:
    with pytest.raises(ExternalParakeetVerificationError) as caught:
        ExternalParakeetVerifier().verify(descriptor, directory, **kwargs)

    assert caught.value.code is code
    assert str(caught.value) == f"External Parakeet verification failed: {code.value}"
    assert str(directory.absolute()) not in str(caught.value)
    assert str(directory.absolute()) not in repr(caught.value)
    return caught.value


def test_verifies_v2_int8_declared_files_and_ignores_unrelated_regular_files(
    tmp_path: Path,
) -> None:
    files = {
        "config.json": b"{}",
        "encoder-model.int8.onnx": b"encoder",
        "decoder_joint-model.int8.onnx": b"decoder",
    }
    root = tmp_path / "external-v2"
    descriptor = _descriptor(files)
    _materialize(root, files)
    unrelated = root / "README.md"
    unrelated.write_text("user owned", encoding="utf-8")

    verified = ExternalParakeetVerifier().verify(descriptor, root)

    assert verified.reference == descriptor.reference
    assert verified.directory == root.absolute()
    assert verified.snapshot.paths == tuple(root / path for path in files)
    assert unrelated.read_text(encoding="utf-8") == "user owned"
    assert unrelated not in verified.snapshot.paths
    assert repr(verified) == (
        f"VerifiedExternalParakeet(reference={descriptor.reference!r})"
    )


def test_verifies_v3_f32_external_data_file(tmp_path: Path) -> None:
    files = {
        "encoder-model.onnx": b"graph",
        "encoder-model.onnx.data": b"external-weights",
        "decoder_joint-model.onnx": b"decoder",
    }
    root = tmp_path / "external-v3"
    descriptor = _descriptor(files, model_id=V3_MODEL_ID, precision="f32")
    _materialize(root, files)

    verified = ExternalParakeetVerifier().verify(descriptor, root)

    assert {path.name for path in verified.snapshot.paths} == {
        "encoder-model.onnx",
        "encoder-model.onnx.data",
        "decoder_joint-model.onnx",
    }


@pytest.mark.parametrize(
    ("replacement", "value"),
    (
        ("role", ArtifactRole.DEPENDENCY),
        ("format", ArtifactFormat.GGUF),
        ("consumer", "image"),
        ("model_family", "other"),
        ("model_id", "unknown-parakeet"),
    ),
)
def test_rejects_unsupported_descriptor_shapes(
    tmp_path: Path,
    replacement: str,
    value: object,
) -> None:
    files = {"model.onnx": b"model"}
    root = tmp_path / "external-unsupported"
    _materialize(root, files)
    descriptor = replace(_descriptor(files), **{replacement: value})

    _assert_error(ExternalParakeetErrorCode.UNSUPPORTED, descriptor, root)


def test_rejects_missing_declared_file(tmp_path: Path) -> None:
    files = {"missing.onnx": b"expected"}
    root = tmp_path / "external-missing"
    root.mkdir()

    _assert_error(ExternalParakeetErrorCode.MISSING, _descriptor(files), root)


def test_formatted_missing_file_error_omits_selected_absolute_path(
    tmp_path: Path,
) -> None:
    files = {"missing.onnx": b"expected"}
    root = tmp_path / "private-external-missing"
    root.mkdir()

    with pytest.raises(ExternalParakeetVerificationError) as caught:
        ExternalParakeetVerifier().verify(_descriptor(files), root)

    formatted = "".join(traceback.format_exception(caught.value))
    assert caught.value.code is ExternalParakeetErrorCode.MISSING
    assert str(root.absolute()) not in formatted


def test_rejects_wrong_size_before_hashing(tmp_path: Path) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-size"
    _materialize(root, {"model.onnx": b"short"})
    observed: list[tuple[int, int]] = []

    _assert_error(
        ExternalParakeetErrorCode.CORRUPT,
        _descriptor(files),
        root,
        progress=lambda done, total: observed.append((done, total)),
    )
    assert observed == [(0, len(files["model.onnx"]))]


def test_rejects_wrong_sha256(tmp_path: Path) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-sha"
    _materialize(root, {"model.onnx": b"tampered"})

    _assert_error(ExternalParakeetErrorCode.CORRUPT, _descriptor(files), root)


def test_rejects_required_file_symlink(tmp_path: Path) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-symlink"
    root.mkdir()
    target = tmp_path / "target.onnx"
    target.write_bytes(files["model.onnx"])
    _symlink_or_skip(root / "model.onnx", target)

    _assert_error(ExternalParakeetErrorCode.IRREGULAR, _descriptor(files), root)


def test_rejects_required_directory_node(tmp_path: Path) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-directory-node"
    root.mkdir()
    (root / "model.onnx").mkdir()

    _assert_error(ExternalParakeetErrorCode.IRREGULAR, _descriptor(files), root)


def test_rejects_selected_root_with_redirected_ancestor(tmp_path: Path) -> None:
    files = {"model.onnx": b"expected"}
    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    root = actual_parent / "model-root"
    _materialize(root, files)
    redirected_parent = tmp_path / "redirected-parent"
    _symlink_or_skip(
        redirected_parent,
        actual_parent,
        target_is_directory=True,
    )
    selected = redirected_parent / "model-root"

    _assert_error(ExternalParakeetErrorCode.IRREGULAR, _descriptor(files), selected)


def test_rejects_declared_path_with_redirected_ancestor(tmp_path: Path) -> None:
    files = {"nested/model.onnx": b"expected"}
    root = tmp_path / "external-contained"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.onnx").write_bytes(files["nested/model.onnx"])
    _symlink_or_skip(root / "nested", outside, target_is_directory=True)

    _assert_error(ExternalParakeetErrorCode.IRREGULAR, _descriptor(files), root)


def test_rejects_file_metadata_mutation_during_hashing(tmp_path: Path) -> None:
    payload = b"x" * (HASH_CHUNK_BYTES + 1)
    files = {"model.onnx": payload}
    root = tmp_path / "external-changing"
    _materialize(root, files)
    model_path = root / "model.onnx"
    changed = False

    def mutate_after_first_chunk(bytes_done: int, bytes_total: int) -> None:
        nonlocal changed
        assert bytes_total == len(payload)
        if bytes_done and not changed:
            metadata = model_path.stat()
            os.utime(
                model_path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 5_000_000_000),
            )
            changed = True

    _assert_error(
        ExternalParakeetErrorCode.CHANGED,
        _descriptor(files),
        root,
        progress=mutate_after_first_chunk,
    )
    assert changed


def test_rejects_earlier_file_mutated_while_later_file_hashes(tmp_path: Path) -> None:
    first_payload = b"original"
    second_payload = b"x" * (HASH_CHUNK_BYTES + 1)
    files = {
        "first.onnx": first_payload,
        "second.onnx": second_payload,
    }
    root = tmp_path / "external-cross-file-change"
    _materialize(root, files)
    first_path = root / "first.onnx"
    mutated = False

    def mutate_first_during_second(bytes_done: int, _bytes_total: int) -> None:
        nonlocal mutated
        if bytes_done > len(first_payload) and not mutated:
            metadata = first_path.stat()
            first_path.write_bytes(b"tampered")
            os.utime(
                first_path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 5_000_000_000),
            )
            mutated = True

    _assert_error(
        ExternalParakeetErrorCode.CHANGED,
        _descriptor(files),
        root,
        progress=mutate_first_during_second,
    )
    assert mutated


def test_polls_cancellation_within_hash_chunk_loop(tmp_path: Path) -> None:
    payload = b"x" * (HASH_CHUNK_BYTES + 1)
    files = {"model.onnx": payload}
    root = tmp_path / "external-cancelled"
    _materialize(root, files)
    cancel = False
    observed: list[tuple[int, int]] = []

    def progress(bytes_done: int, bytes_total: int) -> None:
        nonlocal cancel
        observed.append((bytes_done, bytes_total))
        if bytes_done:
            cancel = True

    _assert_error(
        ExternalParakeetErrorCode.CANCELLED,
        _descriptor(files),
        root,
        cancelled=lambda: cancel,
        progress=progress,
    )
    assert observed == [(0, len(payload)), (HASH_CHUNK_BYTES, len(payload))]


def test_reports_determinate_cumulative_byte_progress(tmp_path: Path) -> None:
    files = {"first.onnx": b"abc", "second.onnx": b"defgh"}
    root = tmp_path / "external-progress"
    _materialize(root, files)
    observed: list[tuple[int, int]] = []

    ExternalParakeetVerifier().verify(
        _descriptor(files),
        root,
        progress=lambda done, total: observed.append((done, total)),
    )

    assert observed == [(0, 8), (3, 8), (8, 8)]


def test_broken_progress_callback_does_not_corrupt_verification(tmp_path: Path) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-broken-progress"
    _materialize(root, files)

    def broken_progress(_bytes_done: int, _bytes_total: int) -> None:
        raise RuntimeError("broken callback")

    verified = ExternalParakeetVerifier().verify(
        _descriptor(files),
        root,
        progress=broken_progress,
    )

    assert verified.reference == _descriptor(files).reference


def test_path_private_logs_omit_selected_absolute_path(tmp_path: Path) -> None:
    files = {"model.onnx": b"expected"}
    actual = tmp_path / "actual-root"
    _materialize(actual, files)
    selected = tmp_path / "private-selected-root"
    _symlink_or_skip(selected, actual, target_is_directory=True)
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))

    try:
        _assert_error(
            ExternalParakeetErrorCode.IRREGULAR,
            _descriptor(files),
            selected,
        )
    finally:
        logger.remove(sink)

    assert str(selected.absolute()) not in "".join(records)
