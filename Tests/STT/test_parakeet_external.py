"""Focused tests for descriptor-verified external Parakeet roots."""

from __future__ import annotations

import hashlib
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

import tldw_chatbook.STT.parakeet_external as parakeet_external
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
    format_external_parakeet_recovery,
)


V2_MODEL_ID = "nemo-parakeet-tdt-0.6b-v2"
V3_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"
HASH_CHUNK_BYTES = 64 * 1024


@pytest.mark.parametrize(
    ("code", "message", "is_error"),
    (
        (
            ExternalParakeetErrorCode.MISSING,
            "Required model files are missing. Choose a complete model directory.",
            True,
        ),
        (
            ExternalParakeetErrorCode.IRREGULAR,
            "Model files must be regular files without links. Choose a safe model directory.",
            True,
        ),
        (
            ExternalParakeetErrorCode.CHANGED,
            "Model files changed during verification. Wait for file changes to finish, then retry.",
            True,
        ),
        (
            ExternalParakeetErrorCode.CORRUPT,
            "Model files do not match the curated model. Choose an unmodified model directory.",
            True,
        ),
        (
            ExternalParakeetErrorCode.UNSUPPORTED,
            "This curated model does not support an external directory.",
            True,
        ),
        (
            ExternalParakeetErrorCode.CANCELLED,
            "Verification cancelled. The prior source is unchanged.",
            False,
        ),
    ),
)
def test_recovery_formatter_preserves_path_free_lab_copy(
    tmp_path: Path,
    code: ExternalParakeetErrorCode,
    message: str,
    is_error: bool,
) -> None:
    assert format_external_parakeet_recovery(code) == (message, is_error)
    assert str(tmp_path) not in message


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


def _hardlink_or_skip(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as error:
        pytest.skip(f"hardlink creation is unavailable: {error}")


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


def test_accepts_windows_path_and_handle_ctime_representation_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-windows-ctime"
    descriptor = _descriptor(files)
    _materialize(root, files)
    real_fstat = os.fstat

    def windows_fstat(descriptor_fd: int) -> SimpleNamespace:
        metadata = real_fstat(descriptor_fd)
        return SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mode=metadata.st_mode,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns,
            st_ctime_ns=metadata.st_ctime_ns + 1,
        )

    monkeypatch.setattr(parakeet_external.os, "fstat", windows_fstat)

    verified = ExternalParakeetVerifier().verify(descriptor, root)

    assert verified.reference == descriptor.reference


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


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not getattr(os, "O_NONBLOCK", 0),
    reason="requires POSIX FIFOs and nonblocking opens",
)
def test_opens_required_file_nonblocking_to_reject_fifo_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-fifo-swap"
    _materialize(root, files)
    model_path = root / "model.onnx"
    original_path = root / "model.original"
    real_open = os.open
    opened = False

    def swap_to_fifo_and_open(path: Path, flags: int) -> int:
        nonlocal opened
        assert Path(path) == model_path
        model_path.rename(original_path)
        try:
            try:
                os.mkfifo(model_path)
            except OSError as error:
                pytest.skip(f"FIFO creation is unavailable: {error}")
            assert flags & os.O_NONBLOCK
            descriptor_fd = real_open(path, flags)
            opened = True
            return descriptor_fd
        finally:
            model_path.unlink(missing_ok=True)
            original_path.rename(model_path)

    monkeypatch.setattr(parakeet_external.os, "open", swap_to_fifo_and_open)

    error = _assert_error(
        ExternalParakeetErrorCode.CHANGED,
        _descriptor(files),
        root,
    )
    assert error.diagnostic_code == "open_file_identity"
    assert opened
    assert str(root.absolute()) not in "".join(traceback.format_exception(error))


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

    error = _assert_error(
        ExternalParakeetErrorCode.CHANGED,
        _descriptor(files),
        root,
        progress=mutate_after_first_chunk,
    )
    assert error.diagnostic_code == "post_read_file_identity"
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


def test_rejects_earlier_file_mutated_after_snapshot_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "first.onnx": b"original",
        "second.onnx": b"second",
    }
    root = tmp_path / "external-post-snapshot-change"
    _materialize(root, files)
    first_path = root / "first.onnx"
    real_snapshot = parakeet_external.snapshot_local_source
    mutated = False

    def snapshot_then_mutate(paths: tuple[Path, ...]):
        nonlocal mutated
        snapshot = real_snapshot(paths)
        metadata = first_path.stat()
        first_path.write_bytes(b"tampered")
        os.utime(
            first_path,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 5_000_000_000),
        )
        mutated = True
        return snapshot

    monkeypatch.setattr(
        parakeet_external, "snapshot_local_source", snapshot_then_mutate
    )

    _assert_error(ExternalParakeetErrorCode.CHANGED, _descriptor(files), root)
    assert mutated


def test_rejects_snapshot_of_temporary_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "nested/first.onnx": b"original",
        "second.onnx": b"second",
    }
    root = tmp_path / "external-temporary-snapshot"
    _materialize(root, files)
    first_path = root / "nested" / "first.onnx"
    original_parent = root / "original-nested"
    real_snapshot = parakeet_external.snapshot_local_source
    substituted = False

    def snapshot_temporary_file(paths: tuple[Path, ...]):
        nonlocal substituted
        first_path.parent.rename(original_parent)
        first_path.parent.mkdir()
        try:
            first_path.write_bytes(b"replaced")
            snapshot = real_snapshot(paths)
        finally:
            first_path.unlink(missing_ok=True)
            first_path.parent.rmdir()
            original_parent.rename(first_path.parent)
        substituted = True
        return snapshot

    monkeypatch.setattr(
        parakeet_external, "snapshot_local_source", snapshot_temporary_file
    )

    _assert_error(ExternalParakeetErrorCode.CHANGED, _descriptor(files), root)
    assert substituted


@pytest.mark.parametrize("metadata_probe_fallback", (False, True))
def test_waiter_cancellation_wins_when_hash_finishes_before_next_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_probe_fallback: bool,
) -> None:
    payload = b"x" * HASH_CHUNK_BYTES
    files = {"model.onnx": payload}
    root = tmp_path / "external-cancelled"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    if metadata_probe_fallback:
        monkeypatch.setattr(verifier, "_cache_key", lambda *_args: None)
    cancel = False
    observed: list[tuple[int, int]] = []

    def progress(bytes_done: int, bytes_total: int) -> None:
        nonlocal cancel
        observed.append((bytes_done, bytes_total))
        if bytes_done:
            cancel = True

    try:
        with pytest.raises(ExternalParakeetVerificationError) as caught:
            verifier.verify(
                _descriptor(files),
                root,
                cancelled=lambda: cancel,
                progress=progress,
            )
    finally:
        verifier.close()

    assert caught.value.code is ExternalParakeetErrorCode.CANCELLED
    assert observed == [(0, len(payload)), (len(payload), len(payload))]


@pytest.mark.parametrize("metadata_probe_fallback", (False, True))
def test_cancel_callback_runs_only_on_waiter_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_probe_fallback: bool,
) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-cancel-thread"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    if metadata_probe_fallback:
        monkeypatch.setattr(verifier, "_cache_key", lambda *_args: None)
    waiter_thread = threading.get_ident()
    worker_entered = threading.Event()
    caller_entered = threading.Event()
    state_lock = threading.Lock()
    callback_threads: set[int] = set()
    active_calls = 0
    max_active_calls = 0

    def cancelled() -> bool:
        nonlocal active_calls, max_active_calls
        current = threading.get_ident()
        with state_lock:
            callback_threads.add(current)
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        try:
            if current != waiter_thread:
                worker_entered.set()
                caller_entered.wait(timeout=2)
            elif worker_entered.is_set():
                caller_entered.set()
            return False
        finally:
            with state_lock:
                active_calls -= 1

    verifier.verify(_descriptor(files), root, cancelled=cancelled)
    verifier.close()

    assert callback_threads == {waiter_thread}
    assert max_active_calls == 1


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


def test_concurrent_waiters_share_one_hash_pass(tmp_path: Path, monkeypatch) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-coalesced"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    real_open = parakeet_external.os.open
    hashing_started = threading.Event()
    second_started = threading.Event()
    release_hash = threading.Event()
    open_count = 0
    second_cancel_polls = 0

    def counted_open(path, flags):
        nonlocal open_count
        open_count += 1
        hashing_started.set()
        assert release_hash.wait(timeout=2)
        return real_open(path, flags)

    monkeypatch.setattr(parakeet_external.os, "open", counted_open)

    def second_cancelled() -> bool:
        nonlocal second_cancel_polls
        second_cancel_polls += 1
        if second_cancel_polls >= 2:
            second_started.set()
        return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(verifier.verify, _descriptor(files), root)
        assert hashing_started.wait(timeout=2)
        second = pool.submit(
            verifier.verify,
            _descriptor(files),
            root,
            cancelled=second_cancelled,
        )
        assert second_started.wait(timeout=2)
        release_hash.set()
        assert first.result(timeout=2) == second.result(timeout=2)

    verifier.close()
    assert open_count == 1


def test_cancelling_one_waiter_keeps_shared_hash_for_other_waiter(
    tmp_path: Path, monkeypatch
) -> None:
    files = {"model.onnx": b"expected"}
    root = tmp_path / "external-waiter-cancel"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    real_open = parakeet_external.os.open
    hashing_started = threading.Event()
    release_hash = threading.Event()
    cancel_first = threading.Event()

    def gated_open(path, flags):
        hashing_started.set()
        assert release_hash.wait(timeout=2)
        return real_open(path, flags)

    monkeypatch.setattr(parakeet_external.os, "open", gated_open)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            verifier.verify,
            _descriptor(files),
            root,
            cancelled=cancel_first.is_set,
        )
        assert hashing_started.wait(timeout=2)
        second = pool.submit(verifier.verify, _descriptor(files), root)
        cancel_first.set()
        with pytest.raises(ExternalParakeetVerificationError) as caught:
            first.result(timeout=2)
        assert caught.value.code is ExternalParakeetErrorCode.CANCELLED
        assert not second.done()
        release_hash.set()
        assert second.result(timeout=2).reference == _descriptor(files).reference

    verifier.close()


def test_cancelling_last_waiter_stops_shared_hash(tmp_path: Path, monkeypatch) -> None:
    payload = b"x" * 4096
    files = {"model.onnx": payload}
    root = tmp_path / "external-last-waiter-cancel"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    cancel = threading.Event()
    progressed = threading.Event()
    waiter_dropped = threading.Event()
    observed: list[int] = []
    monkeypatch.setattr(parakeet_external, "_HASH_CHUNK_BYTES", 1)
    real_drop_waiter = verifier._drop_waiter

    def observed_drop_waiter(*args) -> None:
        real_drop_waiter(*args)
        waiter_dropped.set()

    monkeypatch.setattr(verifier, "_drop_waiter", observed_drop_waiter)

    def progress(done: int, _total: int) -> None:
        observed.append(done)
        if done:
            progressed.set()
            assert waiter_dropped.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            verifier.verify,
            _descriptor(files),
            root,
            cancelled=cancel.is_set,
            progress=progress,
        )
        assert progressed.wait(timeout=2)
        cancel.set()
        with pytest.raises(ExternalParakeetVerificationError) as caught:
            pending.result(timeout=2)

    verifier.close()
    assert caught.value.code is ExternalParakeetErrorCode.CANCELLED
    assert observed[-1] < len(payload)


def test_configured_and_scope_owners_bound_cache_lifetime(
    tmp_path: Path, monkeypatch
) -> None:
    files = {"model.onnx": b"expected"}
    descriptor = _descriptor(files)
    root = tmp_path / "external-owned-cache"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    real_open = parakeet_external.os.open
    open_count = 0

    def counted_open(path, flags):
        nonlocal open_count
        open_count += 1
        return real_open(path, flags)

    monkeypatch.setattr(parakeet_external.os, "open", counted_open)
    verifier.verify(descriptor, root, owner=("configured", "v2_int8"))
    verifier.set_configured_owners({"v2_int8": (descriptor.reference, root.absolute())})
    verifier.verify(descriptor, root)
    assert open_count == 1

    verifier.verify(descriptor, root, owner=("scope", "batch-1"))
    verifier.set_configured_owners({})
    verifier.verify(descriptor, root, owner=("scope", "batch-1"))
    assert open_count == 1
    verifier.release_scope("batch-1")
    verifier.verify(descriptor, root)
    assert open_count == 2
    verifier.close()


def test_hardlinked_roots_keep_the_selected_directory_and_snapshot(
    tmp_path: Path,
) -> None:
    files = {"model.onnx": b"expected"}
    descriptor = _descriptor(files)
    first_root = tmp_path / "first-root"
    second_root = tmp_path / "second-root"
    _materialize(first_root, files)
    second_root.mkdir()
    _hardlink_or_skip(first_root / "model.onnx", second_root / "model.onnx")
    verifier = ExternalParakeetVerifier()

    first = verifier.verify(descriptor, first_root, owner=("scope", "batch"))
    second = verifier.verify(descriptor, second_root, owner=("scope", "batch"))
    verifier.close()

    assert first.directory == first_root.absolute()
    assert first.snapshot.paths == (first_root / "model.onnx",)
    assert second.directory == second_root.absolute()
    assert second.snapshot.paths == (second_root / "model.onnx",)


def test_metadata_change_forces_rehash(tmp_path: Path, monkeypatch) -> None:
    files = {"model.onnx": b"expected"}
    descriptor = _descriptor(files)
    root = tmp_path / "external-cache-metadata"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    real_open = parakeet_external.os.open
    open_count = 0

    def counted_open(path, flags):
        nonlocal open_count
        open_count += 1
        return real_open(path, flags)

    monkeypatch.setattr(parakeet_external.os, "open", counted_open)
    verifier.verify(descriptor, root, owner=("scope", "batch"))
    model_path = root / "model.onnx"
    metadata = model_path.stat()
    os.utime(
        model_path,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 5_000_000_000),
    )
    verifier.verify(descriptor, root, owner=("scope", "batch"))
    verifier.close()

    assert open_count == 2


def test_configured_owner_reverification_prunes_its_old_metadata_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"model.onnx": b"expected"}
    descriptor = _descriptor(files)
    root = tmp_path / "external-configured-reverification"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    real_open = parakeet_external.os.open
    open_count = 0

    def counted_open(path, flags):
        nonlocal open_count
        open_count += 1
        return real_open(path, flags)

    monkeypatch.setattr(parakeet_external.os, "open", counted_open)
    owner = ("configured", "v2_int8")
    verifier.verify(descriptor, root, owner=owner)
    model_path = root / "model.onnx"
    original = model_path.stat()
    os.utime(
        model_path,
        ns=(original.st_atime_ns, original.st_mtime_ns + 5_000_000_000),
    )
    verifier.verify(descriptor, root, owner=owner)
    verifier.verify(descriptor, root)
    assert open_count == 2

    current = model_path.stat()
    os.utime(model_path, ns=(current.st_atime_ns, original.st_mtime_ns))
    verifier.verify(descriptor, root)
    verifier.close()

    assert open_count == 3


def test_scope_released_during_hash_is_not_retained(
    tmp_path: Path, monkeypatch
) -> None:
    files = {"model.onnx": b"expected"}
    descriptor = _descriptor(files)
    root = tmp_path / "external-release-race"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    real_open = parakeet_external.os.open
    hashing_started = threading.Event()
    release_hash = threading.Event()
    open_count = 0

    def gated_open(path, flags):
        nonlocal open_count
        open_count += 1
        hashing_started.set()
        assert release_hash.wait(timeout=2)
        return real_open(path, flags)

    monkeypatch.setattr(parakeet_external.os, "open", gated_open)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            verifier.verify,
            descriptor,
            root,
            owner=("scope", "batch"),
        )
        assert hashing_started.wait(timeout=2)
        verifier.release_scope("batch")
        release_hash.set()
        pending.result(timeout=2)

    verifier.verify(descriptor, root)
    verifier.close()
    assert open_count == 2


def test_close_cancels_in_flight_hashing(tmp_path: Path, monkeypatch) -> None:
    payload = b"x" * 16384
    files = {"model.onnx": payload}
    root = tmp_path / "external-close"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    progressed = threading.Event()
    shutdown_started = threading.Event()
    monkeypatch.setattr(parakeet_external, "_HASH_CHUNK_BYTES", 1)
    real_shutdown = verifier._executor.shutdown

    def observed_shutdown(*args, **kwargs):
        shutdown_started.set()
        return real_shutdown(*args, **kwargs)

    monkeypatch.setattr(verifier._executor, "shutdown", observed_shutdown)

    def progress(done: int, _total: int) -> None:
        if done:
            progressed.set()
            assert shutdown_started.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            verifier.verify,
            _descriptor(files),
            root,
            progress=progress,
        )
        assert progressed.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as close_pool:
            closed = close_pool.submit(verifier.close)
            assert shutdown_started.wait(timeout=2)
            closed.result(timeout=2)
        with pytest.raises(ExternalParakeetVerificationError) as caught:
            pending.result(timeout=2)

    assert caught.value.code is ExternalParakeetErrorCode.CANCELLED


def test_close_cancels_metadata_probe_fallback_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"x" * 4096
    files = {"model.onnx": payload}
    root = tmp_path / "external-close-fallback"
    _materialize(root, files)
    verifier = ExternalParakeetVerifier()
    progressed = threading.Event()
    shutdown_started = threading.Event()
    observed: list[int] = []
    monkeypatch.setattr(parakeet_external, "_HASH_CHUNK_BYTES", 1)
    monkeypatch.setattr(verifier, "_cache_key", lambda *_args: None)
    real_shutdown = verifier._executor.shutdown

    def observed_shutdown(*args, **kwargs):
        shutdown_started.set()
        return real_shutdown(*args, **kwargs)

    monkeypatch.setattr(verifier._executor, "shutdown", observed_shutdown)

    def progress(done: int, _total: int) -> None:
        observed.append(done)
        if done:
            progressed.set()
            assert shutdown_started.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            verifier.verify,
            _descriptor(files),
            root,
            progress=progress,
        )
        assert progressed.wait(timeout=2)
        with ThreadPoolExecutor(max_workers=1) as close_pool:
            closed = close_pool.submit(verifier.close)
            assert shutdown_started.wait(timeout=2)
            closed.result(timeout=2)
        with pytest.raises(ExternalParakeetVerificationError) as caught:
            pending.result(timeout=2)

    assert caught.value.code is ExternalParakeetErrorCode.CANCELLED
    assert observed[-1] < len(payload)
    assert not verifier._fallback_stops
