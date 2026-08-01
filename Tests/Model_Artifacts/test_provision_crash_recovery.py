"""TASK-595 Task 10: crash recovery, containment, and public exports.

Real subprocess crashes (``multiprocessing`` + SIGKILL), not asyncio
cancellation -- Task 7's cancellation test already covers the in-process
case (``test_provision_cancel_mid_fetch_releases_lease_and_preserves_prior_active``
in ``test_provision_fetch.py``). These tests prove the same guarantees hold
when the OS, not asyncio, tears the process down: the exclusive session
lease is a real ``portalocker`` flock the kernel releases on process death;
a durable, sidecar-recorded checkpoint on disk is what a *different* process
picks back up.

Each scenario freezes the child at an exact point via
``provision_processes.provision_signal_on_phase`` -- a progress-callback
hook blocking on a local, never-set ``threading.Event`` -- so the crash
lands deterministically rather than racing a sleep against I/O speed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
from pathlib import Path
from urllib.parse import urlparse

import pytest

from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from Tests.Model_Artifacts.provision_processes import (
    DictCatalog,
    build_descriptor,
    provision_signal_on_phase,
)
from tldw_chatbook.Model_Artifacts import ArtifactRef, closure_fingerprint
from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionConsent,
    ArtifactAcquisitionService,
)
from tldw_chatbook.Model_Artifacts.leases import ArtifactOperationLease, LeaseMode
from tldw_chatbook.Model_Artifacts.service import (
    ACQUISITION_SESSION_LEASE_KEY,
    ModelArtifactService,
)

pytestmark = pytest.mark.integration


def _trusted_hostname(srv: FixtureArtifactServer) -> str:
    """Bare hostname a fixture server answers on (egress trust needs this,
    not a full URL -- see test_stream_fetch.py's identical helper)."""

    return urlparse(srv.url("/")).hostname


def _spawn_and_kill_at(
    root_dir: Path,
    artifact_specs: tuple[dict, ...],
    root_ref_parts: tuple[str, str, str],
    trusted_origin: str,
    *,
    signal_phase: str,
    signal_artifact_id: str | None,
) -> None:
    """Run ``provision_signal_on_phase`` in a real subprocess; SIGKILL it
    the instant it freezes at ``signal_phase``.

    Raises via assertion if the child never reaches the freeze point (a
    real bug, not a flake -- the child's own 30s safety-net timeout would
    otherwise let it silently finish normally and this helper's caller
    would then be asserting about the WRONG process state).
    """

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=provision_signal_on_phase,
        args=(
            str(root_dir),
            artifact_specs,
            root_ref_parts,
            trusted_origin,
            signal_phase,
            signal_artifact_id,
            ready,
        ),
    )
    process.start()
    try:
        assert ready.wait(10.0), (
            f"child never reached phase={signal_phase!r}, exit={process.exitcode}"
        )
        process.kill()  # SIGKILL -- kill -9, not a graceful terminate()
        process.join(10.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        assert process.is_alive() is False
    finally:
        if process.is_alive():
            process.kill()
            process.join(5.0)


async def _assert_session_lease_is_free(core: ModelArtifactService) -> None:
    """The exclusive acquisition-session lease must be immediately
    acquirable -- the killed child's flock was released by the kernel,
    not left dangling."""

    loop = asyncio.get_running_loop()
    probe = ArtifactOperationLease(
        core.locks_path,
        ACQUISITION_SESSION_LEASE_KEY,
        LeaseMode.EXCLUSIVE,
        timeout_seconds=1.0,
    )
    await loop.run_in_executor(None, probe.acquire)
    await loop.run_in_executor(None, probe.release)


# ---------------------------------------------------------------------------
# Scenario 1: kill mid-fetch (frozen at pre-verify, right after a genuine
# partial-but-durable fetch completed) -- valid sidecar survives reconcile,
# session lease frees, a fresh provision resumes via Range and completes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_mid_fetch_valid_sidecar_survives_and_fresh_provision_resumes(
    tmp_path,
):
    # The descriptor declares the FULL body's size/hash; the fixture route
    # initially serves only a leading slice of it with a matching (short)
    # Content-Length -- a genuine, unforced stream_fetch success (no
    # protocol violation) that durably records a real bytes_done < declared
    # size in the sidecar. That is the only way a first-ever fetch attempt
    # leaves a partial-but-valid checkpoint behind: a SIGKILL landing
    # mid-transfer of a file's ONLY attempt leaves no sidecar at all (see
    # this suite's docstring), so this test freezes AFTER that attempt
    # legitimately finished (pre-verify has not yet compared the hash),
    # which is exactly the "fetched but not yet installed" crash window
    # Task 2's staging GC rule exists for.
    full_body = b"AB" * 3000  # 6000 bytes
    initial_body = full_body[:2400]
    sha256 = hashlib.sha256(full_body).hexdigest()
    root_dir = tmp_path / "root"
    ref_parts = ("crash-model", "r" * 40, "int8")

    with FixtureArtifactServer() as srv:
        srv.serve("/model.bin", initial_body, etag='"v1"', support_range=True)
        trusted = _trusted_hostname(srv)

        spec = {
            "artifact_id": ref_parts[0],
            "revision": ref_parts[1],
            "variant": ref_parts[2],
            "role": "root",
            "source_url": srv.url("/model.bin"),
            "size_bytes": len(full_body),
            "sha256": sha256,
            "dependencies": (),
        }

        _spawn_and_kill_at(
            root_dir,
            (spec,),
            ref_parts,
            trusted,
            signal_phase="pre-verify",
            signal_artifact_id=ref_parts[0],
        )

        core = ModelArtifactService(root_dir)
        staging_dir = (
            core.staging_path / "managed" / ref_parts[0] / ref_parts[1] / ref_parts[2]
        )
        sidecar_path = staging_dir.parent / f"{staging_dir.name}.fetch-state.json"

        # The fetch phase durably completed before the freeze: a real,
        # parseable checkpoint sits on disk.
        assert sidecar_path.exists()
        sidecar_before = json.loads(sidecar_path.read_text())
        assert sidecar_before["files"]["model.bin"] == {
            "etag": '"v1"',
            "last_modified": None,
            "bytes_done": len(initial_body),
            "complete": False,
        }
        assert (staging_dir / "model.bin").stat().st_size == len(initial_body)

        # (a) valid-sidecar managed staging survives reconcile() (Task 2's rule).
        report = core.reconcile()
        assert staging_dir.exists()
        assert (staging_dir / "model.bin").exists()
        assert sidecar_path.exists()
        assert report.staging_removed == ()

        # (b) the session lease is free -- the OS released the killed
        # child's flock.
        await _assert_session_lease_is_free(core)

        # (c) a fresh provision resumes via Range and completes.
        srv.serve("/model.bin", full_body, etag='"v1"', support_range=True)
        descriptor = build_descriptor(
            *ref_parts,
            role="root",
            source_url=srv.url("/model.bin"),
            size_bytes=len(full_body),
            sha256=sha256,
            dependencies=(),
        )
        catalog = DictCatalog({descriptor.reference: descriptor})
        root_ref = ArtifactRef(*ref_parts)
        consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(root_ref, ()))
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda _p: 10**12, trusted_origins=frozenset({trusted})
        )

        activated = await svc.provision(root_ref, consent, catalog)

        assert activated == root_ref
        range_headers = [headers.get("Range") for headers in srv.requests["/model.bin"]]
        assert f"bytes={len(initial_body)}-" in range_headers
        assert not staging_dir.exists()
        with core.acquire(root_ref) as handle:
            assert handle.handle.root == root_ref


# ---------------------------------------------------------------------------
# Scenario 2: kill between install and activate -- both artifacts already
# installed, activation not yet run; a fresh provision activates the
# already-installed closure with zero fixture requests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_between_install_and_activate_fresh_provision_activates_with_zero_requests(
    tmp_path,
):
    dep_body = b"dependency-payload-bytes"
    root_body = b"root-payload-bytes-here"
    dep_sha256 = hashlib.sha256(dep_body).hexdigest()
    root_sha256 = hashlib.sha256(root_body).hexdigest()
    root_dir = tmp_path / "root"
    # Deliberately sorts before the root ref: resolve_catalog_closure walks
    # in stable ArtifactRef order, so the dependency installs first and the
    # freeze on root's own "verify-install" event lands only once BOTH
    # artifacts are fully installed.
    dep_ref_parts = ("aaa-dep-model", "d" * 40, "int8")
    root_ref_parts = ("zzz-root-model", "r" * 40, "int8")

    with FixtureArtifactServer() as srv:
        srv.serve("/dep.bin", dep_body, etag='"dep-v1"', support_range=True)
        srv.serve("/root.bin", root_body, etag='"root-v1"', support_range=True)
        trusted = _trusted_hostname(srv)

        dep_spec = {
            "artifact_id": dep_ref_parts[0],
            "revision": dep_ref_parts[1],
            "variant": dep_ref_parts[2],
            "role": "dependency",
            "source_url": srv.url("/dep.bin"),
            "size_bytes": len(dep_body),
            "sha256": dep_sha256,
            "dependencies": (),
        }
        root_spec = {
            "artifact_id": root_ref_parts[0],
            "revision": root_ref_parts[1],
            "variant": root_ref_parts[2],
            "role": "root",
            "source_url": srv.url("/root.bin"),
            "size_bytes": len(root_body),
            "sha256": root_sha256,
            "dependencies": (dep_ref_parts,),
        }

        _spawn_and_kill_at(
            root_dir,
            (dep_spec, root_spec),
            root_ref_parts,
            trusted,
            signal_phase="verify-install",
            signal_artifact_id=root_ref_parts[0],
        )

        core = ModelArtifactService(root_dir)
        dep_ref = ArtifactRef(*dep_ref_parts)
        root_ref = ArtifactRef(*root_ref_parts)

        # Both artifacts are genuinely installed; activation never ran.
        installed_refs = {
            item.descriptor.reference
            for item in core.list_installed()
            if item.descriptor is not None
        }
        assert installed_refs == {dep_ref, root_ref}
        assert not core.active_path(root_ref.artifact_id).exists()

        await _assert_session_lease_is_free(core)

        requests_before = {path: len(reqs) for path, reqs in srv.requests.items()}

        dep_descriptor = build_descriptor(
            *dep_ref_parts,
            role="dependency",
            source_url=srv.url("/dep.bin"),
            size_bytes=len(dep_body),
            sha256=dep_sha256,
            dependencies=(),
        )
        root_descriptor = build_descriptor(
            *root_ref_parts,
            role="root",
            source_url=srv.url("/root.bin"),
            size_bytes=len(root_body),
            sha256=root_sha256,
            dependencies=(dep_ref_parts,),
        )
        catalog = DictCatalog({dep_ref: dep_descriptor, root_ref: root_descriptor})
        consent = AcquisitionConsent(
            closure_fingerprint=closure_fingerprint(root_ref, (dep_ref,))
        )
        svc = ArtifactAcquisitionService(
            core, free_bytes_probe=lambda _p: 10**12, trusted_origins=frozenset({trusted})
        )

        events = []
        activated = await svc.provision(root_ref, consent, catalog, progress=events.append)

        assert activated == root_ref
        assert [event.phase for event in events] == ["activate"]
        assert {path: len(reqs) for path, reqs in srv.requests.items()} == requests_before

        with core.acquire(root_ref) as handle:
            assert handle.handle.root == root_ref
            assert set(handle.handle.closure) == {root_ref, dep_ref}


# ---------------------------------------------------------------------------
# Scenario 3: containment -- reconcile()'s staging_removed names only
# orphans; a crash-surviving valid entry and unrelated tmp_path content are
# both left alone.
# ---------------------------------------------------------------------------


def test_reconcile_after_crash_removes_only_orphans_leaves_everything_else(tmp_path):
    survivor_body = b"survivor-bytes" * 20
    survivor_sha256 = hashlib.sha256(survivor_body).hexdigest()
    root_dir = tmp_path / "root"
    survivor_ref_parts = ("survivor-model", "s" * 40, "int8")

    # Unrelated content sitting alongside the managed store root -- must be
    # completely untouched by reconcile()'s staging GC.
    unrelated_file = tmp_path / "user_notes.txt"
    unrelated_file.write_text("do not touch")
    unrelated_dir = tmp_path / "unrelated_project"
    unrelated_dir.mkdir()
    (unrelated_dir / "keep.bin").write_bytes(b"keep-me")

    with FixtureArtifactServer() as srv:
        srv.serve("/model.bin", survivor_body, etag='"v1"', support_range=True)
        trusted = _trusted_hostname(srv)

        survivor_spec = {
            "artifact_id": survivor_ref_parts[0],
            "revision": survivor_ref_parts[1],
            "variant": survivor_ref_parts[2],
            "role": "root",
            "source_url": srv.url("/model.bin"),
            "size_bytes": len(survivor_body),
            "sha256": survivor_sha256,
            "dependencies": (),
        }

        # A real crash mid-pipeline: the fetch fully (and durably) completed,
        # but the process dies before pre-verify's hash comparison -- the
        # entry must read as "resumable", not "orphaned".
        _spawn_and_kill_at(
            root_dir,
            (survivor_spec,),
            survivor_ref_parts,
            trusted,
            signal_phase="pre-verify",
            signal_artifact_id=survivor_ref_parts[0],
        )

    core = ModelArtifactService(root_dir)
    survivor_dir = (
        core.staging_path
        / "managed"
        / survivor_ref_parts[0]
        / survivor_ref_parts[1]
        / survivor_ref_parts[2]
    )
    assert (survivor_dir.parent / f"{survivor_dir.name}.fetch-state.json").exists()

    # A hand-crafted orphan: no sidecar at all, unrelated to the crash above.
    orphan_dir = core.staging_path / "managed" / "orphan-model" / "rev1" / "int8"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "model.bin").write_bytes(b"abandoned")

    report = core.reconcile()

    # Only the orphan is named as removed.
    assert report.staging_removed == ("managed/orphan-model/rev1/int8",)
    assert not orphan_dir.exists()

    # The crash-surviving valid entry is untouched.
    assert survivor_dir.exists()
    assert (survivor_dir / "model.bin").exists()
    assert (survivor_dir.parent / f"{survivor_dir.name}.fetch-state.json").exists()

    # Content entirely outside the managed store is untouched.
    assert unrelated_file.read_text() == "do not touch"
    assert (unrelated_dir / "keep.bin").read_bytes() == b"keep-me"
