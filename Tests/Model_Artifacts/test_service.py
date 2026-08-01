"""Tests for immutable model-artifact descriptor contracts."""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import json
import multiprocessing
import os
import shutil
import stat
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from Tests.Model_Artifacts.lease_processes import hold_set
import tldw_chatbook.Model_Artifacts as artifacts_module
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


def test_package_exports_the_complete_public_artifact_api() -> None:
    """TASK-595 Task 10 expanded ``__all__`` with the async acquisition
    surface (``ArtifactAcquisitionService`` et al. and ``stream_fetch``),
    resolved lazily via ``__getattr__`` so plain ``import
    tldw_chatbook.Model_Artifacts`` still never touches ``.acquisition``/
    ``.fetch`` -- see ``test_package_import_does_not_load_inference_or_http_runtimes``
    below and ``test_credentials_and_boundaries.py``'s
    ``test_stt_and_transcription_worker_modules_never_import_acquisition_or_fetch``,
    both of which this expansion must keep passing."""

    expected = {
        "ACQUISITION_SESSION_LEASE_KEY",
        "AcquisitionBusyError",
        "AcquisitionConsent",
        "AcquisitionError",
        "AcquisitionProgress",
        "ArtifactAcquisitionService",
        "ArtifactCatalog",
        "ArtifactConflictError",
        "ArtifactDependencyError",
        "ArtifactDescriptor",
        "ArtifactDescriptorError",
        "ArtifactDescriptorParseError",
        "ArtifactDescriptorValidationError",
        "ArtifactDiskUsage",
        "ArtifactError",
        "ArtifactFile",
        "ArtifactFormat",
        "ArtifactHandle",
        "ArtifactInUseError",
        "ArtifactIntegrityError",
        "ArtifactLeaseCancelledError",
        "ArtifactLeaseError",
        "ArtifactLeaseKey",
        "ArtifactLeaseTimeoutError",
        "ArtifactNotReadyError",
        "ArtifactOperationLease",
        "ArtifactOperationLeaseSet",
        "ArtifactPathError",
        "ArtifactPreflightEntry",
        "ArtifactRef",
        "ArtifactRole",
        "ArtifactStateError",
        "CatalogError",
        "ConsentMismatchError",
        "CredentialResolver",
        "EnvConfigCredentialResolver",
        "FetchResult",
        "FetchValidators",
        "GatedRepositoryError",
        "InsufficientSpaceError",
        "InstalledArtifact",
        "LeasedArtifactHandle",
        "LeaseMode",
        "ModelArtifactService",
        "PreflightNotGrantableError",
        "PreflightReport",
        "ProvenanceClass",
        "ReconcileReport",
        "TransferError",
        "closure_fingerprint",
        "stream_fetch",
    }

    assert set(artifacts_module.__all__) == expected
    assert all(hasattr(artifacts_module, name) for name in expected)
    assert "_ReadinessRecord" not in artifacts_module.__all__
    assert not hasattr(artifacts_module, "_ReadinessRecord")


def test_package_import_does_not_load_inference_or_http_runtimes() -> None:
    code = """
import sys
import tldw_chatbook.Model_Artifacts

for name in ("onnxruntime", "onnx_asr", "ctranslate2", "faster_whisper", "httpx"):
    assert name not in sys.modules, name
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        f"fresh package import failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
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


def test_delete_and_reconcile_expose_stable_frozen_contracts() -> None:
    assert issubclass(
        service_module.ArtifactInUseError,
        service_module.ArtifactStateError,
    )
    assert callable(service_module.ModelArtifactService.delete)
    assert callable(service_module.ModelArtifactService.reconcile)

    report = service_module.ReconcileReport(
        readiness_created=0,
        state_removed=0,
        corrupt_artifacts=(Path("/a"), Path("/b")),
        staging_entries=(Path("/c"),),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.state_removed = 1  # type: ignore[misc]


def test_loaded_root_blocks_delete_without_mutation_then_closes_cleanly(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    leased = service.acquire(root.reference)
    contender = service_module.ModelArtifactService(
        tmp_path / "store",
        lease_timeout_seconds=0.01,
    )
    ready_path = service.readiness_path(root.reference)
    active_path = service.active_path(root.reference.artifact_id)
    payload_path = service.artifact_path(root.reference) / root.files[0].path
    before = (
        ready_path.read_bytes(),
        active_path.read_bytes(),
        payload_path.read_bytes(),
        tuple(sorted(service.artifact_path(root.reference).iterdir())),
    )

    with pytest.raises(service_module.ArtifactInUseError):
        contender.delete(root.reference)

    assert (
        ready_path.read_bytes(),
        active_path.read_bytes(),
        payload_path.read_bytes(),
        tuple(sorted(service.artifact_path(root.reference).iterdir())),
    ) == before

    leased.close()
    contender.delete(root.reference)

    assert service.artifact_path(root.reference).exists() is False
    assert ready_path.exists() is False
    assert active_path.exists() is False
    assert service.artifact_path(dependency.reference).exists()


def test_loaded_dependency_blocks_delete_then_invalidates_every_affected_root(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    leased = service.acquire(root.reference)
    contender = service_module.ModelArtifactService(
        tmp_path / "store",
        lease_timeout_seconds=0.01,
    )

    with pytest.raises(service_module.ArtifactInUseError):
        contender.delete(dependency.reference)
    assert service.readiness_path(root.reference).exists()
    assert service.active_path(root.reference.artifact_id).exists()
    assert service.artifact_path(dependency.reference).exists()

    leased.close()
    contender.delete(dependency.reference)

    assert service.readiness_path(root.reference).exists() is False
    assert service.active_path(root.reference.artifact_id).exists() is False
    assert service.artifact_path(dependency.reference).exists() is False
    assert service.artifact_path(root.reference).exists()


def test_delete_inactive_revision_preserves_other_active_revision(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    first_ref = ref("shared-root", "revision-one", "int8")
    second_ref = ref("shared-root", "revision-two", "int8")
    first = single_file_descriptor(first_ref, ArtifactRole.ROOT, b"first")
    second = single_file_descriptor(second_ref, ArtifactRole.ROOT, b"second")
    install_descriptor_payload(service, tmp_path, first, b"first")
    install_descriptor_payload(service, tmp_path, second, b"second")
    service.activate(first_ref)
    service.activate(second_ref)
    active_bytes = service.active_path(first_ref.artifact_id).read_bytes()

    service.delete(first_ref)

    assert service.artifact_path(first_ref).exists() is False
    assert service.readiness_path(first_ref).exists() is False
    assert service.artifact_path(second_ref).exists()
    assert service.readiness_path(second_ref).exists()
    assert service.active_path(second_ref.artifact_id).read_bytes() == active_bytes
    assert service._read_active(second_ref.artifact_id) == second_ref


def test_spawned_shared_closure_blocks_delete_until_forced_process_death(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    contender = service_module.ModelArtifactService(
        tmp_path / "store",
        lease_timeout_seconds=0.1,
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    never_release = context.Event()
    raw_keys = tuple(
        (
            reference.artifact_id,
            reference.revision,
            reference.variant,
        )
        for reference in sorted((root.reference, dependency.reference))
    )
    process = context.Process(
        target=hold_set,
        args=(
            str(tmp_path / "store" / "locks"),
            raw_keys,
            service_module.LeaseMode.SHARED.value,
            ready,
            never_release,
        ),
    )
    process.start()
    try:
        assert ready.wait(10.0)

        with pytest.raises(service_module.ArtifactInUseError):
            contender.delete(dependency.reference)
        assert service.readiness_path(root.reference).exists()
        assert service.active_path(root.reference.artifact_id).exists()
        assert service.artifact_path(dependency.reference).exists()

        process.terminate()
        process.join(10.0)
        assert process.is_alive() is False

        contender.delete(dependency.reference)
        assert service.artifact_path(dependency.reference).exists() is False
        assert service.readiness_path(root.reference).exists() is False
        assert service.active_path(root.reference.artifact_id).exists() is False
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        assert process.is_alive() is False


def test_delete_corrupt_target_but_reject_missing_and_symlinked_target(
    tmp_path: Path,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    target = service.artifact_path(root.reference)
    (target / "manifest.json").write_text("{", encoding="utf-8")

    service.delete(root.reference)

    assert target.exists() is False
    with pytest.raises(service_module.ArtifactStateError, match="does not exist"):
        service.delete(root.reference)

    external = tmp_path / "external-target"
    external.mkdir()
    external_payload = external / "keep.bin"
    external_payload.write_bytes(b"keep")
    target.parent.mkdir(parents=True, exist_ok=True)
    symlink_or_skip(target, external, target_is_directory=True)

    with pytest.raises(service_module.ArtifactPathError):
        service.delete(root.reference)
    assert external_payload.read_bytes() == b"keep"


@pytest.mark.parametrize("failure", ("state", "payload"))
def test_delete_failure_fails_closed_without_touching_unrelated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    unrelated_ref = ref("unrelated", "unrelated-revision", "int8")
    unrelated = single_file_descriptor(
        unrelated_ref,
        ArtifactRole.ROOT,
        b"unrelated",
    )
    install_descriptor_payload(service, tmp_path, unrelated, b"unrelated")
    service.activate(unrelated_ref)
    unrelated_ready = service.readiness_path(unrelated_ref).read_bytes()
    unrelated_active = service.active_path(unrelated_ref.artifact_id).read_bytes()
    unrelated_payload = (
        service.artifact_path(unrelated_ref) / unrelated.files[0].path
    ).read_bytes()
    target = service.artifact_path(dependency.reference)

    if failure == "state":
        original_remove = service._remove_state_path

        def fail_affected_state(path: Path, message: str) -> None:
            if path == service.readiness_path(root.reference):
                raise service_module.ArtifactStateError("injected state failure")
            original_remove(path, message)

        monkeypatch.setattr(service, "_remove_state_path", fail_affected_state)
    else:
        original_rmtree = shutil.rmtree

        def fail_target(path: Path, *args: object, **kwargs: object) -> None:
            if Path(path) == target:
                raise OSError("injected payload failure")
            original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", fail_target)

    with pytest.raises(service_module.ArtifactStateError):
        service.delete(dependency.reference)

    assert target.exists()
    if failure == "state":
        assert service.readiness_path(root.reference).exists()
        assert service.active_path(root.reference.artifact_id).exists()
    else:
        assert service.readiness_path(root.reference).exists() is False
        assert service.active_path(root.reference.artifact_id).exists() is False
    assert service.readiness_path(unrelated_ref).read_bytes() == unrelated_ready
    assert (
        service.active_path(unrelated_ref.artifact_id).read_bytes() == unrelated_active
    )
    assert (
        service.artifact_path(unrelated_ref) / unrelated.files[0].path
    ).read_bytes() == unrelated_payload


def test_delete_failure_before_target_exclusive_does_not_mutate_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    ready_path = service.readiness_path(root.reference)
    active_path = service.active_path(root.reference.artifact_id)
    target = service.artifact_path(root.reference)
    before = (
        ready_path.read_bytes(),
        active_path.read_bytes(),
        (target / root.files[0].path).read_bytes(),
    )
    original_lease = service_module.ArtifactOperationLease

    def fail_target_construction(
        lock_root: Path,
        key: ArtifactLeaseKey,
        mode: object,
        **kwargs: object,
    ) -> object:
        if key == root.reference.lease_key():
            raise service_module.ArtifactLeaseError("injected target lease failure")
        return original_lease(lock_root, key, mode, **kwargs)

    monkeypatch.setattr(
        service_module,
        "ArtifactOperationLease",
        fail_target_construction,
    )

    with pytest.raises(service_module.ArtifactStateError, match="deletion leases"):
        service.delete(root.reference)

    assert (
        ready_path.read_bytes(),
        active_path.read_bytes(),
        (target / root.files[0].path).read_bytes(),
    ) == before


def test_delete_preserves_body_error_over_target_release_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    body_error = service_module.ArtifactStateError("injected deletion body failure")

    class FailingTargetReleaseLease:
        def __init__(
            self,
            _lock_root: Path,
            key: ArtifactLeaseKey,
            _mode: object,
            **_kwargs: object,
        ) -> None:
            self.key = key

        def __enter__(self) -> FailingTargetReleaseLease:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def acquire(self) -> FailingTargetReleaseLease:
            return self

        def release(self) -> None:
            raise service_module.ArtifactLeaseError("injected target release failure")

    monkeypatch.setattr(
        service_module,
        "ArtifactOperationLease",
        FailingTargetReleaseLease,
    )
    monkeypatch.setattr(
        service,
        "_delete_under_leases",
        lambda _reference: (_ for _ in ()).throw(body_error),
    )

    with pytest.raises(service_module.ArtifactStateError) as caught:
        service.delete(root.reference)

    assert caught.value is body_error
    assert any(
        "lease cleanup failed" in note and "target release failure" in note
        for note in getattr(body_error, "__notes__", ())
    )


def test_reconcile_rebuilds_valid_readiness_without_creating_active_and_is_idempotent(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    expected_closure = tuple(sorted((root.reference, dependency.reference)))

    first = service.reconcile()

    assert first == service_module.ReconcileReport(1, 0, (), ())
    assert service._read_readiness(root.reference) == service_module._ReadinessRecord(
        root=root.reference,
        closure=expected_closure,
        closure_fingerprint=closure_fingerprint(root.reference, expected_closure),
    )
    assert service.active_path(root.reference.artifact_id).exists() is False
    leased = service.acquire(root.reference)
    leased.close()

    second = service.reconcile()

    assert second == service_module.ReconcileReport(0, 0, (), ())


@pytest.mark.parametrize("prior_state", ("stale", "malformed", "symlink"))
def test_reconcile_write_failure_preserves_replaceable_readiness_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_state: str,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    ready_path = service.readiness_path(root.reference)
    ready_path.parent.mkdir(parents=True)
    external = tmp_path / "external-readiness.json"
    external.write_bytes(b"external-state")
    if prior_state == "stale":
        ready_path.write_text(
            json.dumps(readiness_state(root.reference, (root.reference,))),
            encoding="utf-8",
        )
    elif prior_state == "malformed":
        ready_path.write_bytes(b"{malformed")
    else:
        symlink_or_skip(ready_path, external, target_is_directory=False)
    prior_bytes = ready_path.read_bytes()
    prior_link = os.readlink(ready_path) if ready_path.is_symlink() else None
    original_write = service_module.atomic_write_json

    def fail_readiness_write(path: Path, value: dict[str, object]) -> None:
        if Path(path) == ready_path:
            raise OSError("injected replacement failure")
        original_write(path, value)

    monkeypatch.setattr(service_module, "atomic_write_json", fail_readiness_write)

    with pytest.raises(service_module.ArtifactStateError, match="write"):
        service.reconcile()

    assert ready_path.read_bytes() == prior_bytes
    assert ready_path.is_symlink() is (prior_state == "symlink")
    if prior_link is not None:
        assert os.readlink(ready_path) == prior_link
    assert external.read_bytes() == b"external-state"


def test_reconcile_atomically_replaces_readiness_symlink_with_consistent_counts(
    tmp_path: Path,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    ready_path = service.readiness_path(root.reference)
    ready_path.parent.mkdir(parents=True)
    external = tmp_path / "external-readiness.json"
    external.write_bytes(b"external-state")
    symlink_or_skip(ready_path, external, target_is_directory=False)

    report = service.reconcile()

    assert report == service_module.ReconcileReport(1, 1, (), ())
    assert ready_path.is_symlink() is False
    assert service._read_readiness(root.reference).root == root.reference
    assert external.read_bytes() == b"external-state"


def test_reconcile_may_remove_directory_readiness_before_stable_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    ready_path = service.readiness_path(root.reference)
    ready_path.mkdir(parents=True)
    (ready_path / "partial.json").write_bytes(b"partial")
    payload = service.artifact_path(root.reference) / root.files[0].path
    payload_bytes = payload.read_bytes()
    monkeypatch.setattr(
        service_module,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected directory replacement failure")
        ),
    )

    with pytest.raises(service_module.ArtifactStateError, match="write"):
        service.reconcile()

    assert ready_path.exists() is False
    assert payload.read_bytes() == payload_bytes


def test_reconcile_corrupt_dependency_invalidates_state_without_deleting_payload(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    corrupt_path = service.artifact_path(dependency.reference)
    payload = corrupt_path / dependency.files[0].path
    payload.write_bytes(b"x" * dependency.files[0].size_bytes)

    report = service.reconcile()

    assert report == service_module.ReconcileReport(
        readiness_created=0,
        state_removed=2,
        corrupt_artifacts=(corrupt_path,),
        staging_entries=(),
    )
    assert corrupt_path.exists()
    assert payload.exists()
    assert service.readiness_path(root.reference).exists() is False
    assert service.active_path(root.reference.artifact_id).exists() is False
    with pytest.raises(service_module.ArtifactNotReadyError):
        service.acquire(root.reference)


@pytest.mark.parametrize(
    "corruption",
    ("payload", "manifest", "symlink", "missing", "extra"),
)
def test_reconcile_reports_corrupt_root_variants_and_keeps_installed_bytes(
    tmp_path: Path,
    corruption: str,
) -> None:
    service, item, _source, target = installed_artifact(tmp_path)
    service.activate(item.reference)
    payload = target / item.files[0].path
    external = tmp_path / "external-payload"
    external.write_bytes(b"external")
    if corruption == "payload":
        payload.write_bytes(b"x" * item.files[0].size_bytes)
    elif corruption == "manifest":
        (target / "manifest.json").write_text("{", encoding="utf-8")
    elif corruption == "symlink":
        payload.unlink()
        symlink_or_skip(payload, external, target_is_directory=False)
    elif corruption == "missing":
        payload.unlink()
    else:
        (target / "undeclared.bin").write_bytes(b"extra")

    report = service.reconcile()

    assert report.corrupt_artifacts == (target,)
    assert report.readiness_created == 0
    assert report.state_removed == 2
    assert target.exists()
    assert any(entry.path == target for entry in service.list_installed())
    assert service.readiness_path(item.reference).exists() is False
    assert service.active_path(item.reference.artifact_id).exists() is False
    assert external.read_bytes() == b"external"


def test_reconcile_removes_exact_depth_state_directories_once_and_is_idempotent(
    tmp_path: Path,
) -> None:
    service, item, _source, target = installed_artifact(tmp_path)
    service.activate(item.reference)
    ready_directory = service._ready_path / "ghost" / "revision" / "int8.json"
    ready_directory.mkdir(parents=True)
    active_directory = service._active_path / "ghost.json"
    active_directory.mkdir()
    external = tmp_path / "external-state-target"
    external.mkdir()
    keep = external / "keep"
    keep.write_bytes(b"keep")
    try:
        (ready_directory / "external-link").symlink_to(
            external,
            target_is_directory=True,
        )
    except OSError:
        pass
    payload = target / item.files[0].path
    payload_bytes = payload.read_bytes()

    first = service.reconcile()

    assert first == service_module.ReconcileReport(0, 2, (), ())
    assert ready_directory.exists() is False
    assert active_directory.exists() is False
    assert payload.read_bytes() == payload_bytes
    assert keep.read_bytes() == b"keep"

    second = service.reconcile()

    assert second == service_module.ReconcileReport(0, 0, (), ())


def test_reconcile_removes_all_malformed_derived_files_and_rebuilds_valid_root(
    tmp_path: Path,
) -> None:
    service, item, _source, _target = installed_artifact(tmp_path)
    ready_path = service.readiness_path(item.reference)
    ready_path.parent.mkdir(parents=True)
    ready_path.write_text("{", encoding="utf-8")
    active_path = service.active_path(item.reference.artifact_id)
    invalid_files = (
        service._ready_path / "empty.json",
        service._ready_path / "unsupported.json",
        service._ready_path / "interrupted.tmp",
        service._ready_path / "deep" / "duplicate" / "state.json",
        active_path,
        service._active_path / "empty.json",
        service._active_path / "unsupported.json",
        service._active_path / "interrupted.tmp",
        service._active_path / "deep" / "duplicate" / "state.json",
    )
    for path in invalid_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"schema_version":2}' if "unsupported" in path.name else "",
            encoding="utf-8",
        )

    report = service.reconcile()

    assert report == service_module.ReconcileReport(1, 10, (), ())
    assert service._read_readiness(item.reference).root == item.reference
    assert active_path.exists() is False
    for path in invalid_files[1:]:
        assert path.exists() is False


def test_reconcile_removes_orphan_readiness_and_active_state(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    missing = ref("missing", "missing-revision", "int8")
    ready_path = service.readiness_path(missing)
    ready_path.parent.mkdir(parents=True)
    ready_path.write_text(
        json.dumps(readiness_state(missing, (missing,))),
        encoding="utf-8",
    )
    active_path = service.active_path(missing.artifact_id)
    active_path.write_text(
        json.dumps({"schema_version": 1, "root": missing.to_dict()}),
        encoding="utf-8",
    )

    report = service.reconcile()

    assert report == service_module.ReconcileReport(0, 2, (), ())
    assert ready_path.exists() is False
    assert active_path.exists() is False


@pytest.mark.parametrize("failure", ("missing", "cycle", "role"))
def test_reconcile_does_not_ready_invalid_dependency_graphs(
    tmp_path: Path,
    failure: str,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    root_ref = ref()
    child_ref = ref("child", "child-revision", "int8")
    root = single_file_descriptor(
        root_ref,
        ArtifactRole.ROOT,
        b"root",
        dependencies=(child_ref,),
    )
    install_descriptor_payload(service, tmp_path, root, b"root")
    if failure == "cycle":
        child = single_file_descriptor(
            child_ref,
            ArtifactRole.DEPENDENCY,
            b"child",
            dependencies=(root_ref,),
        )
        install_descriptor_payload(service, tmp_path, child, b"child")
    elif failure == "role":
        child = single_file_descriptor(child_ref, ArtifactRole.ROOT, b"child")
        install_descriptor_payload(service, tmp_path, child, b"child")

    report = service.reconcile()

    assert report.readiness_created == (1 if failure == "role" else 0)
    assert service.readiness_path(root_ref).exists() is False


def test_reconcile_reports_observed_staging_entries_without_touching_them(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    directory = service.staging_path / "operation"
    directory.mkdir()
    nested = directory / "part"
    nested.write_bytes(b"part")
    file_path = service.staging_path / "interrupted.tmp"
    file_path.write_bytes(b"temp")
    external = tmp_path / "external-staging"
    external.mkdir()
    link = service.staging_path / "external-link"
    symlink_or_skip(link, external, target_is_directory=True)
    expected = tuple(
        sorted((directory, file_path, link), key=lambda path: path.as_posix())
    )

    first = service.reconcile()
    second = service.reconcile()

    assert first == service_module.ReconcileReport(0, 0, (), expected)
    assert second == first
    assert nested.read_bytes() == b"part"
    assert file_path.read_bytes() == b"temp"
    assert link.is_symlink()


def test_reconcile_reports_live_pre_lifecycle_install_staging_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, item, source = install_inputs(tmp_path)
    copy_finished = threading.Event()
    release_install = threading.Event()
    original_copy = service._copy_payload
    installed: list[ArtifactRef] = []
    failures: list[BaseException] = []

    def blocking_copy(
        copied_item: ArtifactDescriptor,
        copied_source: Path,
        staging: Path,
    ) -> None:
        original_copy(copied_item, copied_source, staging)
        copy_finished.set()
        if not release_install.wait(10.0):
            raise AssertionError("test did not release staged install")

    def run_install() -> None:
        try:
            installed.append(service.install(item, source))
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(service, "_copy_payload", blocking_copy)
    install_thread = threading.Thread(target=run_install)
    install_thread.start()
    try:
        assert copy_finished.wait(10.0)
        expected = tuple(
            sorted(service.staging_path.iterdir(), key=lambda path: path.as_posix())
        )
        assert len(expected) == 1

        report = service.reconcile()

        assert report.staging_entries == expected
        assert expected[0].is_dir()
    finally:
        release_install.set()
        install_thread.join(10.0)

    assert install_thread.is_alive() is False
    assert failures == []
    assert installed == [item.reference]
    assert tuple(service.staging_path.iterdir()) == ()


@pytest.mark.parametrize(
    "managed_root",
    ("artifacts", "ready", "active", "staging", "locks"),
)
@pytest.mark.parametrize("replacement", ("symlink", "directory"))
def test_reconcile_rejects_replaced_managed_roots_without_external_mutation(
    tmp_path: Path,
    managed_root: str,
    replacement: str,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    path = tmp_path / "store" / managed_root
    previous = tmp_path / f"previous-{managed_root}"
    path.rename(previous)
    external = tmp_path / f"external-{managed_root}"
    external.mkdir()
    keep = external / "keep"
    keep.write_bytes(b"keep")
    if replacement == "symlink":
        symlink_or_skip(path, external, target_is_directory=True)
    else:
        path.mkdir()

    with pytest.raises(service_module.ArtifactPathError):
        service.reconcile()

    assert keep.read_bytes() == b"keep"
    if replacement == "directory":
        assert tuple(path.iterdir()) == ()


def test_reconcile_uses_shared_closure_leases_with_resident_handle(
    tmp_path: Path,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    leased = service.acquire(root.reference)

    try:
        report = service.reconcile()
    finally:
        leased.close()

    assert report == service_module.ReconcileReport(0, 0, (), ())
    assert service._read_active(root.reference.artifact_id) == root.reference


@pytest.mark.parametrize("failure", ("verify", "write", "unlink"))
def test_reconcile_injected_failures_are_stable_and_never_delete_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    service, item, _source, target = installed_artifact(tmp_path)
    payload = target / item.files[0].path
    payload_bytes = payload.read_bytes()
    if failure == "verify":
        monkeypatch.setattr(
            service,
            "_verify_payload",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected verification failure")
            ),
        )
    elif failure == "write":
        monkeypatch.setattr(
            service_module,
            "atomic_write_json",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected readiness write failure")
            ),
        )
    else:
        ready_path = service._ready_path / "interrupted.tmp"
        ready_path.write_text("{", encoding="utf-8")
        monkeypatch.setattr(
            service,
            "_remove_state_path",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                service_module.ArtifactStateError("injected unlink failure")
            ),
        )

    with pytest.raises(service_module.ArtifactStateError):
        service.reconcile()

    assert target.exists()
    assert payload.read_bytes() == payload_bytes


def install_descriptor_payload(
    service: object,
    tmp_path: Path,
    item: ArtifactDescriptor,
    content: bytes,
) -> None:
    """Install one single-file descriptor from an isolated source directory."""

    source = (
        tmp_path
        / "sources"
        / item.reference.artifact_id
        / item.reference.revision
        / item.reference.variant
    )
    source.mkdir(parents=True)
    (source / item.files[0].path).write_bytes(content)
    service.install(item, source)  # type: ignore[attr-defined]


def installed_root_and_dependency(
    tmp_path: Path,
) -> tuple[object, ArtifactDescriptor, ArtifactDescriptor]:
    """Install one root and its exact dependency."""

    service = service_module.ModelArtifactService(tmp_path / "store")
    dependency_reference = ref("silero-vad", "vad-revision", "int8")
    dependency = descriptor(
        reference=dependency_reference,
        role=ArtifactRole.DEPENDENCY,
        precision=dependency_reference.variant,
        files=(artifact_file(b"dependency"),),
        model_id="snakers4/silero-vad",
    )
    root = descriptor(
        dependencies=(dependency.reference,),
        files=(artifact_file(b"root"),),
    )
    install_descriptor_payload(service, tmp_path, dependency, b"dependency")
    install_descriptor_payload(service, tmp_path, root, b"root")
    return service, root, dependency


def single_file_descriptor(
    reference: ArtifactRef,
    role: ArtifactRole,
    content: bytes,
    *,
    dependencies: tuple[ArtifactRef, ...] = (),
) -> ArtifactDescriptor:
    """Build a descriptor for dependency-graph tests."""

    return descriptor(
        reference=reference,
        role=role,
        precision=reference.variant,
        files=(artifact_file(content),),
        dependencies=dependencies,
        model_id=f"example/{reference.artifact_id}",
    )


def readiness_state(
    root: ArtifactRef,
    closure: tuple[ArtifactRef, ...],
) -> dict[str, object]:
    """Build the exact version 1 readiness JSON value."""

    return {
        "schema_version": 1,
        "root": root.to_dict(),
        "closure": [reference.to_dict() for reference in closure],
        "closure_fingerprint": closure_fingerprint(root, closure),
    }


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


# ---------------------------------------------------------------------------
# TASK-1694: service-owned download-stage seam.
#
# Ported (unmodified in intent) from codex/task-595-managed-downloads-v2's
# Tests/Model_Artifacts/test_service.py -- that branch's own tests for the
# stage API just ported into service.py above (see the reconciliation
# review, item 1). All fixture helpers used below (``descriptor``,
# ``artifact_file``, ``install_inputs``, ``symlink_or_skip``) already exist
# in this file with matching signatures.
# ---------------------------------------------------------------------------


def test_download_stage_finalizes_only_verified_payload(tmp_path: Path) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor(files=(artifact_file(b"model"),))

    stage = service._download_stage_for(item, create=True)

    assert stage is not None
    assert stage.operation.parent == service.staging_path
    assert stage.payload.parent == stage.operation
    assert stage.state.parent == stage.operation
    assert stage.marker.parent == stage.operation
    assert service.artifact_path(item.reference).exists() is False

    (stage.payload / "model.onnx").write_bytes(b"model")
    (stage.state / "model.json").write_text("{}", encoding="utf-8")

    assert service._finalize_download_stage(item, stage) == item.reference

    final = service.artifact_path(item.reference)
    assert (final / "model.onnx").read_bytes() == b"model"
    assert (final / "manifest.json").is_file()
    assert not (final / "model.json").exists()
    assert not stage.operation.exists()


def test_download_stage_create_false_is_non_mutating_when_absent(tmp_path: Path) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()

    assert service._download_stage_for(item, create=False) is None
    assert tuple(service.staging_path.iterdir()) == ()


def test_download_stage_reopens_its_exact_marked_operation(tmp_path: Path) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()

    stage = service._download_stage_for(item, create=True)

    assert stage is not None
    assert service._download_stage_for(item, create=False) == stage


@pytest.mark.parametrize("corruption", ("extra", "size", "digest", "symlink"))
def test_invalid_download_stage_payload_never_creates_final_directory(
    tmp_path: Path,
    corruption: str,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor(files=(artifact_file(b"model"),))
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    (stage.payload / "model.onnx").write_bytes(b"model")
    if corruption == "extra":
        (stage.payload / "extra.bin").write_bytes(b"extra")
    elif corruption == "size":
        (stage.payload / "model.onnx").write_bytes(b"wrong")
    elif corruption == "digest":
        (stage.payload / "model.onnx").write_bytes(b"xxxxx")
    else:
        (stage.payload / "model.onnx").unlink()
        external = tmp_path / "external-model"
        external.write_bytes(b"model")
        symlink_or_skip(
            stage.payload / "model.onnx",
            external,
            target_is_directory=False,
        )

    with pytest.raises((service_module.ArtifactIntegrityError, service_module.ArtifactPathError)):
        service._finalize_download_stage(item, stage)

    assert service.artifact_path(item.reference).exists() is False


def test_download_stage_finalization_does_not_copy_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor(files=(artifact_file(b"model"),))
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    (stage.payload / "model.onnx").write_bytes(b"model")

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("finalization must not copy the payload")

    monkeypatch.setattr(service, "_copy_payload", fail_copy)

    assert service._finalize_download_stage(item, stage) == item.reference


def test_download_stage_recovers_a_matching_staged_manifest(tmp_path: Path) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor(files=(artifact_file(b"model"),))
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    (stage.payload / "model.onnx").write_bytes(b"model")
    (stage.payload / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "descriptor": item.to_dict()}),
        encoding="utf-8",
    )

    reopened = service._download_stage_for(item, create=False)

    assert reopened == stage
    assert service._finalize_download_stage(item, reopened) == item.reference
    assert not (service.artifact_path(item.reference) / "state").exists()


def test_discard_download_stage_refuses_substituted_operation(tmp_path: Path) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    shutil.rmtree(stage.operation)
    stage.operation.mkdir()
    (stage.operation / "keep").write_bytes(b"existing")

    with pytest.raises(service_module.ArtifactPathError):
        service._discard_download_stage(stage)

    assert (stage.operation / "keep").read_bytes() == b"existing"


def test_download_stage_rejects_a_changed_descriptor_or_malformed_marker(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()
    changed = descriptor(reference=item.reference, model_id="example/changed")
    stage = service._download_stage_for(item, create=True)
    assert stage is not None

    assert service._download_stage_for(changed, create=False) is None
    with pytest.raises(service_module.ArtifactPathError):
        service._finalize_download_stage(changed, stage)

    stage.marker.write_text("{", encoding="utf-8")
    with pytest.raises(service_module.ArtifactPathError):
        service._download_stage_for(item, create=False)
    assert service.artifact_path(item.reference).exists() is False


def test_download_stage_rejects_special_state_node_without_finalizing(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor(files=(artifact_file(b"model"),))
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    (stage.payload / "model.onnx").write_bytes(b"model")
    try:
        os.mkfifo(stage.state / "resume.json")
    except OSError as error:
        pytest.skip(f"FIFO creation is unavailable: {error}")

    with pytest.raises(service_module.ArtifactPathError):
        service._finalize_download_stage(item, stage)

    assert service.artifact_path(item.reference).exists() is False


def test_download_stage_converges_on_identical_destination_without_state_changes(
    tmp_path: Path,
) -> None:
    service, item, source = install_inputs(tmp_path, {"model.onnx": b"model"})
    service.install(item, source)
    service.activate(item.reference)
    readiness_before = service.readiness_path(item.reference).read_bytes()
    active_before = service.active_path(item.reference.artifact_id).read_bytes()
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    (stage.payload / "model.onnx").write_bytes(b"model")

    assert service._finalize_download_stage(item, stage) == item.reference

    assert not stage.operation.exists()
    assert service.readiness_path(item.reference).read_bytes() == readiness_before
    assert service.active_path(item.reference.artifact_id).read_bytes() == active_before


@pytest.mark.parametrize(
    "corruption",
    ("missing", "extra", "size", "digest", "symlink"),
)
def test_download_stage_validates_before_identical_destination_convergence(
    tmp_path: Path,
    corruption: str,
) -> None:
    service, item, source = install_inputs(tmp_path, {"model.onnx": b"model"})
    service.install(item, source)
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    payload = stage.payload / "model.onnx"
    payload.write_bytes(b"model")
    if corruption == "missing":
        payload.unlink()
    elif corruption == "extra":
        (stage.payload / "extra.bin").write_bytes(b"extra")
    elif corruption == "size":
        payload.write_bytes(b"wrong")
    elif corruption == "digest":
        payload.write_bytes(b"xxxxx")
    else:
        payload.unlink()
        external = tmp_path / "external-model"
        external.write_bytes(b"model")
        symlink_or_skip(payload, external, target_is_directory=False)

    with pytest.raises((service_module.ArtifactIntegrityError, service_module.ArtifactPathError)):
        service._finalize_download_stage(item, stage)

    assert stage.operation.exists()
    assert service.artifact_path(item.reference).is_dir()


def test_download_stage_preserves_a_conflicting_destination(tmp_path: Path) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor(files=(artifact_file(b"model"),))
    destination = service.artifact_path(item.reference)
    destination.mkdir(parents=True)
    (destination / "keep").write_bytes(b"existing")
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    (stage.payload / "model.onnx").write_bytes(b"model")

    with pytest.raises(service_module.ArtifactConflictError):
        service._finalize_download_stage(item, stage)

    assert (destination / "keep").read_bytes() == b"existing"
    assert stage.operation.exists()


def test_discard_download_stage_removes_only_its_contained_marked_operation(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    unrelated = service.staging_path / "unrelated"
    unrelated.mkdir()

    service._discard_download_stage(stage)

    assert not stage.operation.exists()
    assert unrelated.is_dir()
    escaped = dataclasses.replace(stage, operation=tmp_path / "outside")
    with pytest.raises(service_module.ArtifactPathError):
        service._discard_download_stage(escaped)


def test_discard_download_stage_does_not_change_readiness_or_active_state(
    tmp_path: Path,
) -> None:
    service, item, source = install_inputs(tmp_path, {"model.onnx": b"model"})
    service.install(item, source)
    service.activate(item.reference)
    readiness_before = service.readiness_path(item.reference).read_bytes()
    active_before = service.active_path(item.reference.artifact_id).read_bytes()
    stage = service._download_stage_for(item, create=True)
    assert stage is not None

    service._discard_download_stage(stage)

    assert service.readiness_path(item.reference).read_bytes() == readiness_before
    assert service.active_path(item.reference.artifact_id).read_bytes() == active_before


def test_failed_discard_cleanup_never_poison_canonical_stage_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "store"
    service = service_module.ModelArtifactService(root)
    item = descriptor()
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    original_rmtree = service_module.shutil.rmtree

    def partially_remove_retired(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent == service.staging_path and path.name.startswith(
            ".download-retired-"
        ):
            original_rmtree(path / "state")
            raise OSError("injected retired cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(service_module.shutil, "rmtree", partially_remove_retired)

    with pytest.raises(service_module.ArtifactStateError):
        service._discard_download_stage(stage)

    assert not stage.operation.exists()
    monkeypatch.setattr(service_module.shutil, "rmtree", original_rmtree)
    restarted = service_module.ModelArtifactService(root)
    assert restarted._download_stage_for(item, create=True) is not None


def test_failed_final_stage_cleanup_never_leaves_canonical_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor(files=(artifact_file(b"model"),))
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    (stage.payload / "model.onnx").write_bytes(b"model")
    original_rmtree = service_module.shutil.rmtree

    def fail_retired_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent == service.staging_path and path.name.startswith(
            ".download-retired-"
        ):
            raise OSError("injected retired cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(service_module.shutil, "rmtree", fail_retired_cleanup)

    with pytest.raises(service_module.ArtifactStateError):
        service._finalize_download_stage(item, stage)

    assert not stage.operation.exists()
    monkeypatch.setattr(service_module.shutil, "rmtree", original_rmtree)
    restarted = service_module.ModelArtifactService(tmp_path / "store")
    assert restarted._download_stage_for(item, create=True) is not None


def test_fresh_service_recovers_and_discards_post_promotion_stage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    service = service_module.ModelArtifactService(root)
    item = descriptor(files=(artifact_file(b"model"),))
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    shutil.rmtree(stage.payload)

    restarted = service_module.ModelArtifactService(root)
    recovered = restarted._download_stage_for(item, create=False)

    assert recovered is not None
    assert not recovered.payload.exists()
    restarted._discard_download_stage(recovered)
    assert not stage.operation.exists()


def test_fresh_service_discards_marker_only_post_promotion_stage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    service = service_module.ModelArtifactService(root)
    item = descriptor()
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    shutil.rmtree(stage.payload)
    shutil.rmtree(stage.state)

    restarted = service_module.ModelArtifactService(root)
    recovered = restarted._download_stage_for(item, create=False)

    assert recovered is not None
    assert not recovered.payload.exists()
    assert not recovered.state.exists()
    restarted._discard_download_stage(recovered)
    assert not stage.operation.exists()


def test_download_stage_never_publishes_partial_canonical_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "store"
    service = service_module.ModelArtifactService(root)
    item = descriptor()
    operation, _marker, _payload, _state, _fingerprint = service._download_stage_paths(
        item,
    )
    original_identity = service._download_stage_node_identity

    def fail_temporary_identity(path: Path, *, directory: bool) -> tuple[int, int, int]:
        if path.parent == service.staging_path and path.name.startswith(".download-"):
            raise service_module.ArtifactPathError("injected temporary identity failure")
        return original_identity(path, directory=directory)

    monkeypatch.setattr(service, "_download_stage_node_identity", fail_temporary_identity)
    with pytest.raises(service_module.ArtifactPathError):
        service._download_stage_for(item, create=True)

    assert not operation.exists()
    restarted = service_module.ModelArtifactService(root)
    assert restarted._download_stage_for(item, create=True) is not None


def test_download_stage_discards_losing_temporary_publication_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()
    operation, _marker, _payload, _state, _fingerprint = service._download_stage_paths(
        item,
    )
    original_exists = service._managed_path_exists
    winner_published = False

    def publish_winner(path: Path) -> bool:
        nonlocal winner_published
        if path == operation and not winner_published:
            candidates = tuple(
                candidate
                for candidate in service.staging_path.iterdir()
                if candidate.name.startswith(".download-")
            )
            if candidates:
                shutil.copytree(candidates[0], operation)
                winner_published = True
        return original_exists(path)

    monkeypatch.setattr(service, "_managed_path_exists", publish_winner)

    stage = service._download_stage_for(item, create=True)

    assert stage is not None
    assert stage.operation == operation
    assert tuple(
        candidate
        for candidate in service.staging_path.iterdir()
        if candidate.name.startswith(".download-")
    ) == ()


def test_download_stage_retries_after_marker_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()
    original_write = service_module.atomic_write_json

    def fail_marker_write(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "download-stage.json":
            raise OSError("injected marker write failure")
        original_write(path, *args, **kwargs)

    monkeypatch.setattr(service_module, "atomic_write_json", fail_marker_write)
    with pytest.raises(service_module.ArtifactStateError):
        service._download_stage_for(item, create=True)

    assert tuple(service.staging_path.iterdir()) == ()
    monkeypatch.setattr(service_module, "atomic_write_json", original_write)
    assert service._download_stage_for(item, create=True) is not None


@pytest.mark.parametrize("failed_directory", ("payload", "state"))
def test_download_stage_retries_after_layout_directory_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_directory: str,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()
    original_mkdir = Path.mkdir

    def fail_selected_mkdir(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if (
            path.name == failed_directory
            and path.parent.parent == service.staging_path
            and path.parent.name.startswith(".download-")
        ):
            raise OSError(f"injected {failed_directory} mkdir failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_selected_mkdir)
    with pytest.raises(service_module.ArtifactStateError):
        service._download_stage_for(item, create=True)

    assert tuple(service.staging_path.iterdir()) == ()
    monkeypatch.setattr(Path, "mkdir", original_mkdir)
    assert service._download_stage_for(item, create=True) is not None


def test_discard_rejects_forged_post_promotion_handle_outside_staging(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor()
    stage = service._download_stage_for(item, create=True)
    assert stage is not None
    outside = service.artifacts_path / stage.operation.name
    outside.mkdir()
    marker = outside / stage.marker.name
    marker.write_bytes(stage.marker.read_bytes())
    state = outside / stage.state.name
    state.mkdir()
    payload = outside / stage.payload.name

    def identity(path: Path) -> tuple[int, int, int]:
        info = path.stat(follow_symlinks=False)
        return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))

    forged = dataclasses.replace(
        stage,
        operation=outside,
        marker=marker,
        payload=payload,
        state=state,
        operation_identity=identity(outside),
        marker_identity=identity(marker),
        payload_identity=None,
        state_identity=identity(state),
    )

    with pytest.raises(service_module.ArtifactPathError):
        service._discard_download_stage(forged)

    assert marker.is_file()


def test_service_validates_root_and_creates_only_owned_layout(tmp_path: Path) -> None:
    root = tmp_path / "store"
    service = service_module.ModelArtifactService(root)

    assert service.artifacts_path == root.resolve() / "artifacts"
    assert service.staging_path == root.resolve() / "staging"
    assert service.locks_path == root.resolve() / "locks"
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


def test_install_rejects_source_paths_blocked_by_central_validation(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source;unsafe"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"model")
    item = descriptor(files=(artifact_file(b"model"),))

    with pytest.raises(service_module.ArtifactPathError):
        service.install(item, source)

    assert service.artifact_path(item.reference).exists() is False
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


def test_idempotent_install_reports_operation_staging_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, item, source, final = installed_artifact(tmp_path)
    abandoned = service.staging_path / "abandoned"
    abandoned.mkdir()
    cleanup_calls: list[Path] = []
    cleanup_error = OSError("injected cleanup failure")

    def fail_cleanup(path: Path) -> None:
        cleanup_calls.append(Path(path))
        raise cleanup_error

    monkeypatch.setattr(service_module.shutil, "rmtree", fail_cleanup)

    with pytest.raises(service_module.ArtifactStateError) as caught:
        service.install(item, source)

    assert caught.value.__cause__ is cleanup_error
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0].parent == service.staging_path
    assert cleanup_calls[0] != abandoned
    assert abandoned.is_dir()
    assert final.is_dir()


def test_primary_install_error_preserves_cleanup_failure_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, item, source = install_inputs(tmp_path)
    abandoned = service.staging_path / "abandoned"
    abandoned.mkdir()
    primary = service_module.ArtifactIntegrityError("primary install failure")
    cleanup_error = OSError("injected cleanup failure")

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise primary

    def fail_cleanup(_path: Path) -> None:
        raise cleanup_error

    monkeypatch.setattr(service, "_copy_payload", fail_copy)
    monkeypatch.setattr(service_module.shutil, "rmtree", fail_cleanup)

    with pytest.raises(service_module.ArtifactIntegrityError) as caught:
        service.install(item, source)

    assert caught.value is primary
    assert any(
        "operation staging cleanup failed" in note
        and "injected cleanup failure" in note
        for note in getattr(primary, "__notes__", ())
    )
    assert abandoned.is_dir()
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


def test_inventory_rejects_replaced_artifacts_root_before_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = descriptor(files=(artifact_file(b"model"),))
    external_final = (
        tmp_path
        / "external"
        / item.reference.artifact_id
        / item.reference.revision
        / item.reference.variant
    )
    external_final.mkdir(parents=True)
    (external_final / "model.onnx").write_bytes(b"model")
    (external_final / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "descriptor": item.to_dict()},
        ),
        encoding="utf-8",
    )
    service.artifacts_path.rmdir()
    symlink_or_skip(
        service.artifacts_path,
        tmp_path / "external",
        target_is_directory=True,
    )
    original_scandir = service_module.os.scandir
    scan_calls: list[Path] = []

    def record_scandir(path: str | os.PathLike[str]) -> object:
        scan_calls.append(Path(path))
        return original_scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(service_module.os, "scandir", record_scandir)
        installed = service.list_installed()

    assert scan_calls == []
    assert len(installed) == 1
    assert installed[0].path == service.artifacts_path
    assert installed[0].descriptor is None
    assert installed[0].ready is False
    assert installed[0].active is False
    assert installed[0].error


def test_inventory_reports_deeply_nested_manifest_as_invalid(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    final = service.artifact_path(ref())
    final.mkdir(parents=True)
    depth = 1_000
    while depth <= 65_536:
        nested = "[" * depth + "0" + "]" * depth
        try:
            json.loads(nested)
        except RecursionError:
            break
        depth *= 2
    else:
        pytest.fail("JSON decoder accepted the bounded nesting probe")
    (final / "manifest.json").write_text(
        '{"schema_version":1,"descriptor":' + nested + "}",
        encoding="utf-8",
    )

    installed = service.list_installed()

    assert len(installed) == 1
    assert installed[0].path == final
    assert installed[0].descriptor is None
    assert installed[0].ready is False
    assert installed[0].active is False
    assert installed[0].error


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

    with monkeypatch.context() as patch:
        patch.setattr(service_module.os, "scandir", swap_artifacts)
        with pytest.raises(service_module.ArtifactPathError):
            service.disk_usage()

    assert swapped is True


def test_activate_writes_exact_state_and_returns_canonical_leased_handle(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    expected_closure = tuple(sorted((root.reference, dependency.reference)))
    expected_fingerprint = closure_fingerprint(
        root.reference,
        (dependency.reference,),
    )

    assert service.activate(root.reference) == root.reference
    assert json.loads(
        service.readiness_path(root.reference).read_text(encoding="utf-8")
    ) == {
        "schema_version": 1,
        "root": root.reference.to_dict(),
        "closure": [reference.to_dict() for reference in expected_closure],
        "closure_fingerprint": expected_fingerprint,
    }
    assert json.loads(
        service.active_path(root.reference.artifact_id).read_text(encoding="utf-8")
    ) == {
        "schema_version": 1,
        "root": root.reference.to_dict(),
    }

    leased = service.acquire(root.reference)
    with leased as entered:
        assert entered is leased
        assert leased.handle.root == root.reference
        assert leased.handle.closure == expected_closure
        assert leased.handle.closure_fingerprint == expected_fingerprint
        assert leased.handle.paths == tuple(
            (reference, service.artifact_path(reference))
            for reference in expected_closure
        )
        assert leased.handle.lease_keys == tuple(
            reference.lease_key() for reference in expected_closure
        )
        assert leased.handle.resident_identity == (
            root.reference,
            expected_fingerprint,
        )


def test_activate_resolves_transitive_closure_in_canonical_order(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    leaf_ref = ref("aaa-tokenizer", "leaf-revision", "int8")
    middle_ref = ref("zzz-vad", "middle-revision", "int8")
    leaf = single_file_descriptor(
        leaf_ref,
        ArtifactRole.DEPENDENCY,
        b"leaf",
    )
    middle = single_file_descriptor(
        middle_ref,
        ArtifactRole.DEPENDENCY,
        b"middle",
        dependencies=(leaf_ref,),
    )
    root = descriptor(
        files=(artifact_file(b"root"),),
        dependencies=(middle_ref,),
    )
    for item, content in (
        (leaf, b"leaf"),
        (middle, b"middle"),
        (root, b"root"),
    ):
        install_descriptor_payload(service, tmp_path, item, content)

    service.activate(root.reference)

    state = json.loads(
        service.readiness_path(root.reference).read_text(encoding="utf-8")
    )
    assert state["closure"] == [
        reference.to_dict()
        for reference in sorted((leaf_ref, root.reference, middle_ref))
    ]


@pytest.mark.parametrize(
    "failure",
    (
        "missing",
        "dependency-role",
        "root-role",
        "cycle",
        "root-as-dependency",
        "conflict",
    ),
)
def test_activate_rejects_invalid_exact_dependency_graph(
    tmp_path: Path,
    failure: str,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    child_ref = ref("child", "child-revision", "int8")
    root_ref = ref()

    if failure == "missing":
        root = single_file_descriptor(
            root_ref,
            ArtifactRole.ROOT,
            b"root",
            dependencies=(child_ref,),
        )
        install_descriptor_payload(service, tmp_path, root, b"root")
    elif failure == "dependency-role":
        child = single_file_descriptor(child_ref, ArtifactRole.ROOT, b"child")
        root = single_file_descriptor(
            root_ref,
            ArtifactRole.ROOT,
            b"root",
            dependencies=(child_ref,),
        )
        install_descriptor_payload(service, tmp_path, child, b"child")
        install_descriptor_payload(service, tmp_path, root, b"root")
    elif failure == "root-role":
        root = single_file_descriptor(
            root_ref,
            ArtifactRole.DEPENDENCY,
            b"root",
        )
        install_descriptor_payload(service, tmp_path, root, b"root")
    elif failure == "cycle":
        second_ref = ref("second", "second-revision", "int8")
        child = single_file_descriptor(
            child_ref,
            ArtifactRole.DEPENDENCY,
            b"child",
            dependencies=(second_ref,),
        )
        second = single_file_descriptor(
            second_ref,
            ArtifactRole.DEPENDENCY,
            b"second",
            dependencies=(child_ref,),
        )
        root = single_file_descriptor(
            root_ref,
            ArtifactRole.ROOT,
            b"root",
            dependencies=(child_ref,),
        )
        for item, content in (
            (child, b"child"),
            (second, b"second"),
            (root, b"root"),
        ):
            install_descriptor_payload(service, tmp_path, item, content)
    elif failure == "root-as-dependency":
        child = single_file_descriptor(
            child_ref,
            ArtifactRole.DEPENDENCY,
            b"child",
            dependencies=(root_ref,),
        )
        root = single_file_descriptor(
            root_ref,
            ArtifactRole.ROOT,
            b"root",
            dependencies=(child_ref,),
        )
        install_descriptor_payload(service, tmp_path, child, b"child")
        install_descriptor_payload(service, tmp_path, root, b"root")
    else:
        left_ref = ref("left", "left-revision", "int8")
        right_ref = ref("right", "right-revision", "int8")
        shared_v1_ref = ref("shared", "revision-one", "int8")
        shared_v2_ref = ref("shared", "revision-two", "int8")
        graph = (
            single_file_descriptor(
                shared_v1_ref,
                ArtifactRole.DEPENDENCY,
                b"shared-one",
            ),
            single_file_descriptor(
                shared_v2_ref,
                ArtifactRole.DEPENDENCY,
                b"shared-two",
            ),
            single_file_descriptor(
                left_ref,
                ArtifactRole.DEPENDENCY,
                b"left",
                dependencies=(shared_v1_ref,),
            ),
            single_file_descriptor(
                right_ref,
                ArtifactRole.DEPENDENCY,
                b"right",
                dependencies=(shared_v2_ref,),
            ),
        )
        root = single_file_descriptor(
            root_ref,
            ArtifactRole.ROOT,
            b"root",
            dependencies=(left_ref, right_ref),
        )
        for item, content in zip(
            graph,
            (b"shared-one", b"shared-two", b"left", b"right"),
            strict=True,
        ):
            install_descriptor_payload(service, tmp_path, item, content)
        install_descriptor_payload(service, tmp_path, root, b"root")

    with pytest.raises(service_module.ArtifactDependencyError):
        service.activate(root.reference)

    assert service.readiness_path(root.reference).exists() is False


def test_activate_verification_failure_leaves_no_loadable_readiness(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    ready_path = service.readiness_path(root.reference)
    ready_path.write_text('{"schema_version":2}', encoding="utf-8")
    (service.artifact_path(dependency.reference) / "model.onnx").write_bytes(
        b"x" * dependency.files[0].size_bytes
    )

    with pytest.raises(service_module.ArtifactIntegrityError):
        service.activate(root.reference)

    assert ready_path.exists() is False
    assert service.active_path(root.reference.artifact_id).exists() is False


def test_failed_activation_preserves_another_active_revision(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    first_ref = ref("shared-root", "revision-one", "int8")
    second_ref = ref("shared-root", "revision-two", "int8")
    first = single_file_descriptor(first_ref, ArtifactRole.ROOT, b"first")
    second = single_file_descriptor(second_ref, ArtifactRole.ROOT, b"second")
    install_descriptor_payload(service, tmp_path, first, b"first")
    install_descriptor_payload(service, tmp_path, second, b"second")
    service.activate(second_ref)
    active_path = service.active_path(first_ref.artifact_id)
    active_bytes = active_path.read_bytes()
    (service.artifact_path(first_ref) / "model.onnx").write_bytes(b"wrong")

    with pytest.raises(service_module.ArtifactIntegrityError):
        service.activate(first_ref)

    assert active_path.read_bytes() == active_bytes
    assert service._read_active(first_ref.artifact_id) == second_ref


@pytest.mark.parametrize(
    "mutation",
    (
        "malformed",
        "duplicate-key",
        "extra-key",
        "missing-key",
        "unsupported-schema",
        "bool-schema",
        "mistyped-root",
        "wrong-root",
        "mistyped-closure",
        "empty-closure",
        "unsorted-closure",
        "duplicate-closure",
        "wrong-closure",
        "wrong-fingerprint",
        "mistyped-fingerprint",
    ),
)
def test_activate_rebuilds_every_invalid_readiness_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    path = service.readiness_path(root.reference)
    closure = tuple(sorted((root.reference, dependency.reference)))
    raw = readiness_state(root.reference, closure)

    if mutation == "malformed":
        path.write_text("{", encoding="utf-8")
    elif mutation == "duplicate-key":
        path.write_text(
            json.dumps(raw)[:-1] + ',"schema_version":1}',
            encoding="utf-8",
        )
    else:
        if mutation == "extra-key":
            raw["extra"] = "value"
        elif mutation == "missing-key":
            raw.pop("closure_fingerprint")
        elif mutation == "unsupported-schema":
            raw["schema_version"] = 2
        elif mutation == "bool-schema":
            raw["schema_version"] = True
        elif mutation == "mistyped-root":
            raw["root"] = "root"
        elif mutation == "wrong-root":
            raw["root"] = ref("other", "revision", "int8").to_dict()
        elif mutation == "mistyped-closure":
            raw["closure"] = "closure"
        elif mutation == "empty-closure":
            raw["closure"] = []
        elif mutation == "unsorted-closure":
            raw["closure"] = [reference.to_dict() for reference in reversed(closure)]
        elif mutation == "duplicate-closure":
            raw["closure"] = [
                root.reference.to_dict(),
                root.reference.to_dict(),
            ]
        elif mutation == "wrong-closure":
            raw["closure"] = [root.reference.to_dict()]
            raw["closure_fingerprint"] = closure_fingerprint(root.reference, ())
        elif mutation == "wrong-fingerprint":
            raw["closure_fingerprint"] = "0" * 64
        else:
            raw["closure_fingerprint"] = 1
        path.write_text(json.dumps(raw), encoding="utf-8")

    calls: list[ArtifactRef] = []
    original = service._verify_installed

    def count_verification(
        reference: ArtifactRef,
        expected_role: ArtifactRole,
    ) -> None:
        calls.append(reference)
        original(reference, expected_role)

    monkeypatch.setattr(service, "_verify_installed", count_verification)

    service.activate(root.reference)

    assert calls == list(closure)
    assert json.loads(path.read_text(encoding="utf-8")) == readiness_state(
        root.reference,
        closure,
    )


def test_activate_reuses_fresh_matching_readiness_without_payload_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)

    def reject_verification(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("matching readiness must bypass payload verification")

    monkeypatch.setattr(service, "_verify_installed", reject_verification)

    assert service.activate(root.reference) == root.reference


def test_active_atomic_write_failure_preserves_prior_selector_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    previous_ref = ref("shared-root", "revision-one", "int8")
    replacement_ref = ref("shared-root", "revision-two", "int8")
    previous = single_file_descriptor(
        previous_ref,
        ArtifactRole.ROOT,
        b"previous",
    )
    replacement = single_file_descriptor(
        replacement_ref,
        ArtifactRole.ROOT,
        b"replacement",
    )
    install_descriptor_payload(service, tmp_path, previous, b"previous")
    install_descriptor_payload(service, tmp_path, replacement, b"replacement")
    service.activate(previous_ref)
    active_path = service.active_path(previous_ref.artifact_id)
    previous_bytes = active_path.read_bytes()
    original = service_module.atomic_write_json

    def fail_active(path: Path, value: dict[str, object]) -> None:
        if Path(path) == active_path:
            raise OSError("injected active write failure")
        original(path, value)

    monkeypatch.setattr(service_module, "atomic_write_json", fail_active)

    with pytest.raises(service_module.ArtifactStateError):
        service.activate(replacement_ref)

    assert active_path.read_bytes() == previous_bytes
    assert service.readiness_path(replacement_ref).exists()


@pytest.mark.parametrize("replacement", ("symlink", "new-directory"))
def test_activate_rejects_replaced_active_root_before_state_write(
    tmp_path: Path,
    replacement: str,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    active_path = service.active_path(root.reference.artifact_id)
    previous_bytes = active_path.read_bytes()
    active_root = active_path.parent
    previous_root = tmp_path / "previous-active"
    active_root.rename(previous_root)
    external = tmp_path / "external-active"
    external.mkdir()
    if replacement == "symlink":
        symlink_or_skip(active_root, external, target_is_directory=True)
    else:
        active_root.mkdir()

    with pytest.raises(service_module.ArtifactPathError):
        service.activate(root.reference)

    assert (previous_root / active_path.name).read_bytes() == previous_bytes
    assert tuple(external.iterdir()) == ()
    if replacement == "new-directory":
        assert tuple(active_root.iterdir()) == ()


@pytest.mark.parametrize("operation", ("activate", "acquire"))
@pytest.mark.parametrize("replacement", ("symlink", "new-directory"))
def test_operations_reject_replaced_locks_root_before_lease_construction(
    tmp_path: Path,
    operation: str,
    replacement: str,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    if operation == "acquire":
        service.activate(root.reference)
    locks_root = tmp_path / "store" / "locks"
    previous_root = tmp_path / "previous-locks"
    locks_root.rename(previous_root)
    external = tmp_path / "external-locks"
    external.mkdir()
    if replacement == "symlink":
        symlink_or_skip(locks_root, external, target_is_directory=True)
    else:
        locks_root.mkdir()

    with pytest.raises(service_module.ArtifactPathError):
        if operation == "activate":
            service.activate(root.reference)
        else:
            service.acquire(root.reference)

    assert tuple(external.iterdir()) == ()
    if replacement == "new-directory":
        assert tuple(locks_root.iterdir()) == ()


def test_activate_acquires_lifecycle_before_shared_canonical_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    events: list[tuple[str, object]] = []

    class RecordingLifecycleLease:
        def __init__(
            self,
            _lock_root: Path,
            key: ArtifactLeaseKey,
            mode: object,
            **_kwargs: object,
        ) -> None:
            events.append(("lifecycle-constructed", (key, mode)))

        def __enter__(self) -> RecordingLifecycleLease:
            events.append(("lifecycle-acquired", None))
            return self

        def __exit__(self, *_args: object) -> None:
            events.append(("lifecycle-released", None))

    class RecordingLeaseSet:
        def __init__(
            self,
            _lock_root: Path,
            keys: tuple[ArtifactLeaseKey, ...],
            mode: object,
            **_kwargs: object,
        ) -> None:
            events.append(("set-constructed", (tuple(keys), mode)))

        def __enter__(self) -> RecordingLeaseSet:
            events.append(("set-acquired", None))
            return self

        def __exit__(self, *_args: object) -> None:
            events.append(("set-released", None))

    monkeypatch.setattr(
        service_module,
        "ArtifactOperationLease",
        RecordingLifecycleLease,
    )
    monkeypatch.setattr(
        service_module,
        "ArtifactOperationLeaseSet",
        RecordingLeaseSet,
    )

    service.activate(root.reference)

    assert events == [
        (
            "lifecycle-constructed",
            (
                ArtifactLeaseKey("!lifecycle", "1", "writer"),
                service_module.LeaseMode.EXCLUSIVE,
            ),
        ),
        ("lifecycle-acquired", None),
        (
            "set-constructed",
            (
                tuple(
                    reference.lease_key()
                    for reference in sorted((root.reference, dependency.reference))
                ),
                service_module.LeaseMode.SHARED,
            ),
        ),
        ("set-acquired", None),
        ("set-released", None),
        ("lifecycle-released", None),
    ]


def test_acquire_holds_exact_shared_closure_until_idempotent_close(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    leased = service.acquire(root.reference)
    keys = tuple(
        reference.lease_key()
        for reference in sorted((root.reference, dependency.reference))
    )

    with pytest.raises(service_module.ArtifactLeaseError):
        service_module.ArtifactOperationLeaseSet(
            tmp_path / "store" / "locks",
            keys,
            service_module.LeaseMode.EXCLUSIVE,
            timeout_seconds=0.01,
        ).acquire()

    with leased as entered:
        assert entered is leased

    leased.close()
    with service_module.ArtifactOperationLeaseSet(
        tmp_path / "store" / "locks",
        keys,
        service_module.LeaseMode.EXCLUSIVE,
        timeout_seconds=0.1,
    ):
        pass


def test_closed_leased_handle_cannot_be_reentered(tmp_path: Path) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    leased = service.acquire(root.reference)
    leased.close()
    leased.close()

    with pytest.raises(service_module.ArtifactStateError, match="closed"):
        with leased:
            pass


@pytest.mark.parametrize("mutation", ("changed", "removed"))
def test_acquire_reread_failure_releases_every_shared_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    path = service.readiness_path(root.reference)
    original = service._read_readiness
    reads = 0

    def mutate_after_first_read(reference: ArtifactRef) -> object:
        nonlocal reads
        record = original(reference)
        reads += 1
        if reads == 1:
            if mutation == "removed":
                path.unlink()
            else:
                extra = ref("other", "other-revision", "int8")
                closure = tuple(sorted((*record.closure, extra)))
                path.write_text(
                    json.dumps(readiness_state(root.reference, closure)),
                    encoding="utf-8",
                )
        return record

    monkeypatch.setattr(service, "_read_readiness", mutate_after_first_read)

    with pytest.raises(service_module.ArtifactStateError, match="changed"):
        service.acquire(root.reference)

    keys = tuple(
        reference.lease_key()
        for reference in sorted((root.reference, dependency.reference))
    )
    with service_module.ArtifactOperationLeaseSet(
        tmp_path / "store" / "locks",
        keys,
        service_module.LeaseMode.EXCLUSIVE,
        timeout_seconds=0.1,
    ):
        pass


def test_acquire_missing_or_invalid_readiness_uses_stable_errors_without_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)

    def reject_lease_construction(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid readiness must not construct leases")

    monkeypatch.setattr(
        service_module,
        "ArtifactOperationLeaseSet",
        reject_lease_construction,
    )

    with pytest.raises(service_module.ArtifactNotReadyError):
        service.acquire(root.reference)

    path = service.readiness_path(root.reference)
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(service_module.ArtifactStateError):
        service.acquire(root.reference)


def test_acquire_does_not_verify_payload_or_read_active_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)

    def reject_expensive_work(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("acquire must trust strict readiness under leases")

    monkeypatch.setattr(service, "_verify_payload", reject_expensive_work)
    monkeypatch.setattr(service, "_read_manifest", reject_expensive_work)
    service.active_path(root.reference.artifact_id).write_text("{", encoding="utf-8")

    leased = service.acquire(root.reference)
    leased.close()


def test_leased_handle_release_error_preserves_body_exception() -> None:
    root = ref()
    handle = service_module.ArtifactHandle(
        root=root,
        closure=(root,),
        closure_fingerprint=closure_fingerprint(root, ()),
        paths=((root, Path("/managed/root")),),
    )

    class FailingLeaseSet:
        def release(self) -> None:
            raise service_module.ArtifactLeaseError("injected release failure")

    leased = service_module.LeasedArtifactHandle(handle, FailingLeaseSet())
    body_error = ValueError("body failure")

    with pytest.raises(ValueError) as caught:
        with leased:
            raise body_error

    assert caught.value is body_error
    assert any(
        "lease context cleanup failed" in note and "release failure" in note
        for note in getattr(body_error, "__notes__", ())
    )
    leased.close()

    cleanup_only = service_module.LeasedArtifactHandle(handle, FailingLeaseSet())
    with pytest.raises(service_module.ArtifactLeaseError, match="release failure"):
        with cleanup_only:
            pass


def test_inventory_reports_exact_ready_and_active_revision_flags(
    tmp_path: Path,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    first_ref = ref("shared-root", "revision-one", "int8")
    second_ref = ref("shared-root", "revision-two", "int8")
    first = single_file_descriptor(first_ref, ArtifactRole.ROOT, b"first")
    second = single_file_descriptor(second_ref, ArtifactRole.ROOT, b"second")
    install_descriptor_payload(service, tmp_path, first, b"first")
    install_descriptor_payload(service, tmp_path, second, b"second")
    service.activate(first_ref)
    service.activate(second_ref)

    installed = {
        entry.descriptor.reference: entry
        for entry in service.list_installed()
        if entry.descriptor is not None
    }

    assert installed[first_ref].ready is True
    assert installed[first_ref].active is False
    assert installed[first_ref].error is None
    assert installed[second_ref].ready is True
    assert installed[second_ref].active is True
    assert installed[second_ref].error is None


def test_inventory_dependency_is_not_root_ready_or_active(tmp_path: Path) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)

    installed = {
        entry.descriptor.reference: entry
        for entry in service.list_installed()
        if entry.descriptor is not None
    }

    assert installed[root.reference].ready is True
    assert installed[root.reference].active is True
    assert installed[dependency.reference].ready is False
    assert installed[dependency.reference].active is False


def test_inventory_ignores_forged_valid_state_for_dependency_role(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    dependency_readiness = service.readiness_path(dependency.reference)
    dependency_readiness.parent.mkdir(parents=True)
    dependency_readiness.write_text(
        json.dumps(readiness_state(dependency.reference, (dependency.reference,))),
        encoding="utf-8",
    )
    service.active_path(dependency.reference.artifact_id).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": dependency.reference.to_dict(),
            }
        ),
        encoding="utf-8",
    )

    installed = {
        entry.descriptor.reference: entry
        for entry in service.list_installed()
        if entry.descriptor is not None
    }

    assert installed[root.reference].ready is True
    assert installed[root.reference].active is True
    assert installed[dependency.reference].ready is False
    assert installed[dependency.reference].active is False


def test_inventory_reports_malformed_state_for_dependency_role(
    tmp_path: Path,
) -> None:
    service, _root, dependency = installed_root_and_dependency(tmp_path)
    dependency_readiness = service.readiness_path(dependency.reference)
    dependency_readiness.parent.mkdir(parents=True)
    dependency_readiness.write_text("{", encoding="utf-8")
    service.active_path(dependency.reference.artifact_id).write_text(
        "{",
        encoding="utf-8",
    )

    installed = {
        entry.descriptor.reference: entry
        for entry in service.list_installed()
        if entry.descriptor is not None
    }
    dependency_entry = installed[dependency.reference]

    assert dependency_entry.ready is False
    assert dependency_entry.active is False
    assert dependency_entry.error
    assert "readiness" in dependency_entry.error
    assert "active" in dependency_entry.error


def test_inventory_malformed_state_is_nonfatal_and_does_not_resolve_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, root, _dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    service.readiness_path(root.reference).write_text("{", encoding="utf-8")
    service.active_path(root.reference.artifact_id).write_text("{", encoding="utf-8")

    def reject_graph_or_payload_work(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inventory must not resolve or verify dependency payloads")

    monkeypatch.setattr(service, "_resolve_closure", reject_graph_or_payload_work)
    monkeypatch.setattr(service, "_verify_payload", reject_graph_or_payload_work)

    installed = {
        entry.descriptor.reference: entry
        for entry in service.list_installed()
        if entry.descriptor is not None
    }
    root_entry = installed[root.reference]

    assert root_entry.ready is False
    assert root_entry.active is False
    assert root_entry.error
    assert "readiness" in root_entry.error
    assert "active" in root_entry.error


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate-key",
        "extra-key",
        "missing-key",
        "unsupported-schema",
        "bool-schema",
        "mistyped-root",
        "wrong-artifact-id",
    ),
)
def test_inventory_strict_active_selector_errors_are_nonfatal(
    tmp_path: Path,
    mutation: str,
) -> None:
    service = service_module.ModelArtifactService(tmp_path / "store")
    item = single_file_descriptor(ref(), ArtifactRole.ROOT, b"root")
    install_descriptor_payload(service, tmp_path, item, b"root")
    service.activate(item.reference)
    path = service.active_path(item.reference.artifact_id)
    raw: dict[str, object] = {
        "schema_version": 1,
        "root": item.reference.to_dict(),
    }
    if mutation == "duplicate-key":
        path.write_text(
            json.dumps(raw)[:-1] + ',"schema_version":1}',
            encoding="utf-8",
        )
    else:
        if mutation == "extra-key":
            raw["extra"] = "value"
        elif mutation == "missing-key":
            raw.pop("root")
        elif mutation == "unsupported-schema":
            raw["schema_version"] = 2
        elif mutation == "bool-schema":
            raw["schema_version"] = True
        elif mutation == "mistyped-root":
            raw["root"] = "root"
        else:
            raw["root"] = ref("other", "revision", "int8").to_dict()
        path.write_text(json.dumps(raw), encoding="utf-8")

    entries = [
        entry for entry in service.list_installed() if entry.descriptor is not None
    ]

    assert len(entries) == 1
    assert entries[0].ready is True
    assert entries[0].active is False
    assert entries[0].error
    assert "active" in entries[0].error
