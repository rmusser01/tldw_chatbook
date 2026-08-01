"""TASK-595 Task 2: reconcile deletes staging orphans, preserves resumable state."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactLeaseKey,
    ArtifactLeaseTimeoutError,
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


def _sidecar_path(managed_dir: Path) -> Path:
    """The fetch-state sidecar's path for ``managed_dir`` -- a SIBLING
    file (``<variant>.fetch-state.json``), never a child of the payload
    directory it describes (see acquisition.py's ``_fetch_sidecar_path``
    and this file's drift-guard test)."""
    return managed_dir.parent / f"{managed_dir.name}.fetch-state.json"


def _install_inputs(tmp_path: Path) -> tuple[ArtifactDescriptor, Path]:
    """Build a minimal valid descriptor + matching source directory.

    A small local stand-in for test_service.py's ``install_inputs`` --
    that helper (and every other in that file) is private to the sealed
    TASK-594 suite and must not be imported from here (see this repo's
    "not cross-importing test-private helpers" convention, e.g.
    test_preflight.py's ``_two_file_descriptor`` docstring).
    """
    content = b"model-bytes"
    source = tmp_path / "install-source"
    source.mkdir()
    (source / "model.onnx").write_bytes(content)
    file = ArtifactFile("model.onnx", len(content), hashlib.sha256(content).hexdigest())
    item = ArtifactDescriptor(
        reference=ArtifactRef("race-model", "r" * 40, "int8"),
        model_id="test/model",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="test",
        model_family="test-family",
        upstream_repository="test/repo",
        upstream_revision="main",
        source_url="https://example.test/model",
        precision="int8",
        license_id="test-license",
        license_url="https://example.test/license",
        usage_notice="Test model",
        runtime_name="test-runtime",
        runtime_version_constraint="==1.0.0",
        supported_os=("linux",),
        supported_architectures=("x86-64",),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=(file,),
        expected_installed_bytes=len(content),
        dependencies=(),
    )
    return item, source


def test_install_staging_directory_creation_is_atomic_with_lease_acquisition(
    service, tmp_path, monkeypatch
):
    """install()'s ``install-*`` staging directory must never become
    visible on disk BEFORE its per-directory orphan-detection lease is
    held.

    P1 RACE (TASK-595 final review): reconcile()'s staging GC
    (``_gc_install_staging_orphan``) treats an ``install-*`` directory as
    abandoned -- safe to delete -- whenever ``_install_staging_lease_key``
    for its name is free. If install() creates the directory FIRST and
    only acquires that lease AFTERWARD, a concurrent reconcile() pass can
    observe the directory in that window, acquire the (still-free) lease
    itself, and delete a staging dir install() is about to copy files
    into.

    Proven by intercepting the actual ``os.mkdir`` call install() uses to
    create the staging directory -- this covers BOTH implementations:
    pre-fix, ``tempfile.mkdtemp`` creates the directory via its own
    internal ``os.mkdir`` call (the ``tempfile`` module binds ``_os`` to
    the SAME ``os`` module object, so patching ``os.mkdir`` intercepts it
    too); post-fix, the directory is created via a direct ``os.mkdir``
    call. At the exact moment either version is about to create the
    directory, this test independently attempts a NON-BLOCKING acquire of
    that same directory's orphan-detection lease from a separate lease
    handle -- modeling reconcile()'s own non-blocking probe. If install()
    already holds the lease (the fix), the probe's acquire must fail
    (busy). If it does not yet hold it (the bug), the probe's acquire
    succeeds -- proving the directory would be visible on disk with
    nothing yet protecting it from GC.
    """
    item, source = _install_inputs(tmp_path)
    real_mkdir = os.mkdir
    probe_acquired: list[bool] = []

    def probing_mkdir(path, mode=0o777, *args, **kwargs):
        path_obj = Path(path)
        if (
            path_obj.parent == Path(service.staging_path)
            and path_obj.name.startswith("install-")
        ):
            probe_lease = ArtifactOperationLease(
                service.locks_path,
                ArtifactLeaseKey("#install-staging", path_obj.name, "lock"),
                LeaseMode.EXCLUSIVE,
                timeout_seconds=0.01,
            )
            try:
                probe_lease.acquire()
            except ArtifactLeaseTimeoutError:
                # Busy: install() already holds this directory's lease
                # before the directory itself exists -- the fix.
                probe_acquired.append(False)
            else:
                # Free: the directory is about to become visible on disk
                # with NOTHING yet protecting it from a concurrent
                # reconcile() pass -- the race.
                probe_acquired.append(True)
                probe_lease.release()
        return real_mkdir(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", probing_mkdir)
    service.install(item, source)

    # os.mkdir on the staging directory's own path fires more than once in
    # practice -- the real creation, plus later redundant
    # Path.mkdir(parents=True, exist_ok=True) calls from the copy phase
    # that no-op once the directory exists. Every one of them must find
    # the lease already held: the FIRST entry is the directory's actual
    # creation moment (the one this test targets), and any True anywhere
    # in this list still proves an unprotected instant existed.
    assert probe_acquired, "the install-* staging mkdir must have been observed"
    assert not any(probe_acquired), (
        "install()'s per-staging lease must already be held BEFORE its "
        "staging directory becomes visible on disk, or a concurrent "
        "reconcile() pass can delete a live staging directory"
    )


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
    _sidecar_path(d).write_text(json.dumps({
        "files": {"model.onnx": {"etag": "\"abc\"", "last_modified": None,
                                   "bytes_done": 7, "complete": False}}
    }))
    report = service.reconcile()
    assert d.exists()
    assert (d / "model.onnx").exists()
    assert _sidecar_path(d).exists()


def test_managed_entry_with_sidecar_inside_payload_dir_is_ignored_and_removed(service):
    """A sidecar written in the OLD in-tree location (``<variant>/
    fetch-state.json``, a child of the payload directory) no longer
    counts -- the classifier only ever looks at the SIBLING path. This
    also means the payload directory itself is invalid (no sidecar at its
    new location), so it is removed like any other unprotected entry;
    the misplaced sidecar file is just an ordinary file inside it that
    goes with it.
    """
    d = _managed_dir(service)
    (d / "model.onnx").write_bytes(b"partial")
    (d / "fetch-state.json").write_text(json.dumps({
        "files": {"model.onnx": {"etag": "\"abc\"", "last_modified": None,
                                   "bytes_done": 7, "complete": False}}
    }))
    report = service.reconcile()
    assert not d.exists()
    assert report.staging_removed


def test_managed_entry_with_corrupt_sidecar_is_removed(service):
    d = _managed_dir(service)
    _sidecar_path(d).write_text("{not json")
    service.reconcile()
    assert not d.exists()


def test_managed_entry_with_non_mapping_sidecar_is_removed(service):
    """A sidecar that parses as valid JSON but isn't a ``{"files": {...}}``
    mapping (the exact shape Task 7's fetch phase always writes) is still
    garbage, not resumable state.

    Regression test for the review finding: the old check only required
    ``json.loads`` to succeed, so a bare JSON list or string -- neither a
    usable fetch-state record -- would survive as "valid" forever, keeping
    dead staging alive indefinitely.
    """
    list_dir = _managed_dir(service, artifact="m-list")
    _sidecar_path(list_dir).write_text(json.dumps([]))

    string_dir = _managed_dir(service, artifact="m-string")
    _sidecar_path(string_dir).write_text(json.dumps("x"))

    report = service.reconcile()

    assert not list_dir.exists()
    assert not string_dir.exists()
    assert any("m-list" in item for item in report.staging_removed)
    assert any("m-string" in item for item in report.staging_removed)


def test_managed_entry_sidecar_without_payload_dir_is_removed(service):
    """A sidecar file surviving with no matching payload directory (e.g. a
    crash between _install_artifact deleting the payload but before the
    sidecar unlink, or simply a hand-placed stray file) is not resumable
    state either -- both members of the pair must be real."""
    managed_root = Path(service.staging_path) / "managed" / "m1" / "r1"
    managed_root.mkdir(parents=True)
    orphan_sidecar = managed_root / "int8.fetch-state.json"
    orphan_sidecar.write_text(json.dumps({
        "files": {"model.onnx": {"etag": None, "last_modified": None,
                                   "bytes_done": 7, "complete": False}}
    }))
    report = service.reconcile()
    assert not orphan_sidecar.exists()
    assert any("int8.fetch-state.json" in item for item in report.staging_removed)


def test_fetch_sidecar_suffix_mirror_matches_acquisition():
    """Drift guard: service.py's sibling-sidecar suffix must equal
    acquisition.py's -- see both modules' constants for the rationale."""
    from tldw_chatbook.Model_Artifacts import acquisition
    from tldw_chatbook.Model_Artifacts import service as service_module

    assert acquisition._FETCH_SIDECAR_SUFFIX == service_module._MANAGED_FETCH_SIDECAR_SUFFIX


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
