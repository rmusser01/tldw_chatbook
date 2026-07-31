"""TASK-595 Task 2: reconcile deletes staging orphans, preserves resumable state."""

import json
from pathlib import Path

import pytest

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactLeaseKey,
    ArtifactOperationLease,
    LeaseMode,
    ModelArtifactService,
)


@pytest.fixture()
def service(tmp_path):
    return ModelArtifactService(tmp_path / "root")


def _managed_dir(service, artifact="m1", rev="r1", variant="int8") -> Path:
    d = Path(service.staging_path) / "managed" / artifact / rev / variant
    d.mkdir(parents=True)
    return d


def test_orphan_install_staging_is_removed(service):
    orphan = Path(service.staging_path) / "install-deadbeef"
    orphan.mkdir()
    (orphan / "partial.bin").write_bytes(b"x" * 10)
    report = service.reconcile()
    assert not orphan.exists()
    assert any("install-deadbeef" in item for item in report.staging_removed)


def test_managed_entry_without_sidecar_is_removed(service):
    d = _managed_dir(service)
    (d / "model.onnx").write_bytes(b"partial")
    report = service.reconcile()
    assert not d.exists()
    assert report.staging_removed


def test_managed_entry_with_valid_sidecar_survives(service):
    d = _managed_dir(service)
    (d / "model.onnx").write_bytes(b"partial")
    (d / "fetch-state.json").write_text(json.dumps({
        "files": {"model.onnx": {"etag": "\"abc\"", "last_modified": None,
                                   "bytes_done": 7, "complete": False}}
    }))
    report = service.reconcile()
    assert d.exists()
    assert (d / "model.onnx").exists()


def test_managed_entry_with_corrupt_sidecar_is_removed(service):
    d = _managed_dir(service)
    (d / "fetch-state.json").write_text("{not json")
    service.reconcile()
    assert not d.exists()


def test_gc_never_escapes_staging(service, tmp_path):
    """Containment: a symlink inside staging pointing outside must not
    cause deletion outside the root (extends the 594 containment tests)."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    link = Path(service.staging_path) / "managed" / "evil"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(victim)
    service.reconcile()
    assert (victim / "keep.txt").exists()


def test_gc_skips_install_staging_dir_held_by_a_live_installer(service, tmp_path):
    """A held per-directory staging lease simulates a live install() call in
    another process/thread. GC must not touch its staging directory while
    the lease is held, and must remove it once the lease is released --
    this is the race reconcile() vs. a pre-lifecycle-lock install() must
    not lose (see test_reconcile_reports_live_pre_lifecycle_install_staging_entry
    in test_service.py, which reconcile() must keep passing)."""
    live = Path(service.staging_path) / "install-livecase"
    live.mkdir()
    (live / "model.onnx").write_bytes(b"copied-but-not-yet-promoted")
    key = ArtifactLeaseKey("#install-staging", live.name, "lock")
    lock_root = tmp_path / "root" / "locks"

    with ArtifactOperationLease(lock_root, key, LeaseMode.EXCLUSIVE):
        held_report = service.reconcile()

    assert live.exists()
    assert (live / "model.onnx").exists()
    assert held_report.staging_removed == ()

    freed_report = service.reconcile()

    assert not live.exists()
    assert any("install-livecase" in item for item in freed_report.staging_removed)
