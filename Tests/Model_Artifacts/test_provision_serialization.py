"""TASK-595 Task 6: provision() skeleton.

Session lease + in-process lock serialization, consent drift detection, and
the idempotent already-installed completion path. The fetch/pre-verify/
install phases are method stubs filled in by Tasks 7-8; none of these tests
need them to succeed.
"""

from __future__ import annotations

import asyncio
import multiprocessing
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from Tests.Model_Artifacts.lease_processes import hold_one
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog, make_descriptor
from tldw_chatbook.Model_Artifacts import ArtifactDependencyError, ArtifactRef, closure_fingerprint
from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionBusyError,
    AcquisitionConsent,
    ArtifactAcquisitionService,
    ConsentMismatchError,
)
from tldw_chatbook.Model_Artifacts.leases import (
    ArtifactLeaseTimeoutError,
    ArtifactOperationLease,
    LeaseMode,
)
from tldw_chatbook.Model_Artifacts.service import (
    ACQUISITION_SESSION_LEASE_KEY,
    ModelArtifactService,
)


@contextmanager
def holding_process(
    target: Callable[..., None],
    args: tuple[object, ...],
) -> Iterator[multiprocessing.Process]:
    """Spawn a subprocess holding a lease, mirroring test_operation_leases_process.py.

    Duplicated locally rather than imported: that module's helper is a
    test-private detail, not a shared utility, and the two are small enough
    that duplicating is cheaper than coupling two unrelated test modules.
    """

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=target, args=(*args, ready, release))
    process.start()
    try:
        assert ready.wait(10.0), f"child failed to acquire lease, exit={process.exitcode}"
        yield process
    finally:
        release.set()
        process.join(10.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        assert process.is_alive() is False
        assert process.exitcode == 0


def _installed_root(tmp_path: Path) -> tuple[ModelArtifactService, ArtifactRef, object]:
    """Build a core with one artifact installed and activated."""

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("root-model", "r1", "int8")
    desc = make_descriptor(ref=root)
    source = tmp_path / "source"
    source.mkdir()
    (source / desc.files[0].path).write_bytes(b"x")
    core.install(desc, source)
    core.activate(root)
    return core, root, desc


@pytest.mark.asyncio
async def test_provision_serializes_concurrent_calls_in_process(tmp_path, monkeypatch):
    """A second in-process provision() call blocks on the asyncio.Lock.

    A naive version of this test (assert the fetch stub was not re-entered
    right after starting task2) passed even with the ``asyncio.Lock``
    entirely removed -- confirmed by mutation. The reason: the non-blocking
    session lease ALSO serializes (with ``AcquisitionBusyError`` instead of
    a wait), and if task2 raced straight to that lease attempt, its 0.1s
    timeout could still resolve (fail) before the test's short
    ``asyncio.sleep(0)`` assertion, or -- worse -- could coincidentally
    succeed if task1 happened to release within that same 0.1s poll window,
    making the mutation's absence of a lock invisible. The fix: hold task1
    paused (via ``gate``, entirely test-controlled) for LONGER than the
    session lease's own timeout, then assert task2 is not even **done**
    yet. An ``asyncio.Lock`` wait has no timeout and blocks indefinitely;
    a raced, un-queued session-lease attempt would have already resolved
    (successfully or as ``AcquisitionBusyError``) well within that window.
    """

    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    root = ArtifactRef("root-model", "r1", "int8")
    desc = make_descriptor(ref=root)
    catalog = DictCatalog({root: desc})
    consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(root, ()))

    events: list[str] = []
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def fake_fetch(descriptor, staging_dir, progress_state):
        events.append("enter")
        entered.set()
        await gate.wait()
        events.append("exit")

    async def fake_preverify(descriptor, staging_dir, progress_state):
        pass

    async def fake_install(descriptor, staging_dir):
        pass

    monkeypatch.setattr(svc, "_fetch_artifact", fake_fetch)
    monkeypatch.setattr(svc, "_preverify_artifact", fake_preverify)
    monkeypatch.setattr(svc, "_install_artifact", fake_install)

    async def run_and_capture() -> BaseException | ArtifactRef:
        try:
            return await svc.provision(root, consent, catalog)
        except BaseException as error:  # noqa: BLE001 -- asserted by caller, not swallowed
            return error

    task1 = asyncio.create_task(run_and_capture())
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    assert events == ["enter"]

    entered.clear()
    task2 = asyncio.create_task(run_and_capture())
    # Comfortably longer than _SESSION_LEASE_TIMEOUT_SECONDS (0.1s): if
    # task2 reached the session-lease acquire attempt at all (i.e. the
    # in-process Lock did not queue it first), it would have resolved --
    # successfully or as AcquisitionBusyError -- well within this window,
    # since task1 (still paused on `gate`) has not released anything yet.
    await asyncio.sleep(0.4)
    assert task2.done() is False, (
        "second provision() call must block on the asyncio.Lock, "
        "not race the session lease"
    )
    assert events == ["enter"]
    assert entered.is_set() is False

    gate.set()
    result1 = await task1
    assert events[:2] == ["enter", "exit"]

    await asyncio.wait_for(entered.wait(), timeout=5.0)
    result2 = await task2

    assert events == ["enter", "exit", "enter", "exit"]
    # Both calls reach the real core.activate() (fetch/preverify/install were
    # faked as no-ops, so nothing was actually installed) and fail there --
    # confirms no OTHER exception masked the ordering proof above.
    assert isinstance(result1, ArtifactDependencyError)
    assert isinstance(result2, ArtifactDependencyError)


@pytest.mark.integration
def test_provision_cross_process_busy_raises(tmp_path):
    """An externally held session lease makes provision() fail fast, busy.

    Mirrors Tests/Model_Artifacts/test_operation_leases_process.py's own
    style (plain sync test, real subprocess) rather than nesting blocking
    subprocess start/join calls inside an async test function.
    """

    core, root, desc = _installed_root(tmp_path)
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    catalog = DictCatalog({root: desc})
    consent = asyncio.run(svc.preflight(root, catalog)).grant()

    raw_key = (
        ACQUISITION_SESSION_LEASE_KEY.artifact_id,
        ACQUISITION_SESSION_LEASE_KEY.revision,
        ACQUISITION_SESSION_LEASE_KEY.variant,
    )
    with holding_process(
        hold_one,
        (str(core.locks_path), raw_key, LeaseMode.EXCLUSIVE.value),
    ):
        with pytest.raises(AcquisitionBusyError):
            asyncio.run(svc.provision(root, consent, catalog))

    # Once released, the identical call succeeds (busy was transient).
    result = asyncio.run(svc.provision(root, consent, catalog))
    assert result == root


@pytest.mark.asyncio
async def test_provision_fingerprint_drift_raises_consent_mismatch(tmp_path):
    """Consent minted from one catalog does not carry over a mutated one.

    Both entries are pre-installed so preflight() never needs the network
    gating probe; the drift is purely in the CLOSURE SHAPE (root+dependency
    vs. root alone), which is what closure_fingerprint actually hashes (it
    does not hash descriptor content -- only the set of ArtifactRefs in the
    closure).
    """

    core = ModelArtifactService(tmp_path / "root")
    root = ArtifactRef("root-model", "r1", "int8")
    dep = ArtifactRef("dep-model", "r1", "int8")
    body = b"m" * 16

    root_desc_with_dep = make_descriptor(ref=root, dependencies=(dep,), files_body=body)
    dep_desc = make_descriptor(ref=dep, files_body=body)

    root_source = tmp_path / "root-source"
    root_source.mkdir()
    (root_source / root_desc_with_dep.files[0].path).write_bytes(body)
    core.install(root_desc_with_dep, root_source)

    dep_source = tmp_path / "dep-source"
    dep_source.mkdir()
    (dep_source / dep_desc.files[0].path).write_bytes(body)
    core.install(dep_desc, dep_source)

    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    catalog_v1 = DictCatalog({root: root_desc_with_dep, dep: dep_desc})
    report = await svc.preflight(root, catalog_v1)
    consent = report.grant()

    root_desc_without_dep = make_descriptor(ref=root, files_body=body)
    catalog_v2 = DictCatalog({root: root_desc_without_dep})

    with pytest.raises(ConsentMismatchError):
        await svc.provision(root, consent, catalog_v2)


@pytest.mark.asyncio
async def test_provision_fully_installed_closure_skips_stubs(tmp_path, monkeypatch):
    """A fully-installed closure activates without touching any phase stub."""

    core, root, desc = _installed_root(tmp_path)
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    catalog = DictCatalog({root: desc})
    report = await svc.preflight(root, catalog)
    consent = report.grant()

    calls = {"fetch": 0, "preverify": 0, "install": 0}

    async def counting_fetch(descriptor, staging_dir, progress_state):
        calls["fetch"] += 1

    async def counting_preverify(descriptor, staging_dir, progress_state):
        calls["preverify"] += 1

    async def counting_install(descriptor, staging_dir):
        calls["install"] += 1

    monkeypatch.setattr(svc, "_fetch_artifact", counting_fetch)
    monkeypatch.setattr(svc, "_preverify_artifact", counting_preverify)
    monkeypatch.setattr(svc, "_install_artifact", counting_install)

    result = await svc.provision(root, consent, catalog)

    assert result == root
    assert calls == {"fetch": 0, "preverify": 0, "install": 0}


@pytest.mark.asyncio
async def test_provision_holds_session_lease_across_phase_stub(tmp_path, monkeypatch):
    """The session lease stays held while paused inside a phase stub.

    Regression guard for the Task 2 carry-over: the lease must be held for
    provision()'s ENTIRE run, not released between the pre-checks and the
    per-artifact phase loop -- releasing early would let reconcile()'s
    managed-staging GC (``_gc_managed_staging``) race a live download the
    same way an earlier draft let it race a live ``install()``.

    In-process exclusive locks on this lock file DO conflict across
    different ``ArtifactOperationLease`` handles (confirmed: portalocker's
    flock-based exclusive lock is scoped to the open file description, not
    the process), so a same-process non-blocking probe is a faithful test
    of "is the lease currently held" without needing a subprocess.
    """

    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    root = ArtifactRef("root-model", "r1", "int8")
    desc = make_descriptor(ref=root)
    catalog = DictCatalog({root: desc})
    consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(root, ()))

    entered = asyncio.Event()
    release = asyncio.Event()

    async def paused_fetch(descriptor, staging_dir, progress_state):
        entered.set()
        await release.wait()
        raise NotImplementedError("stop here; only the pause matters")

    monkeypatch.setattr(svc, "_fetch_artifact", paused_fetch)

    task = asyncio.create_task(svc.provision(root, consent, catalog))
    await asyncio.wait_for(entered.wait(), timeout=5.0)

    loop = asyncio.get_running_loop()
    probe = ArtifactOperationLease(
        core.locks_path,
        ACQUISITION_SESSION_LEASE_KEY,
        LeaseMode.EXCLUSIVE,
        timeout_seconds=0.1,
    )
    with pytest.raises(ArtifactLeaseTimeoutError):
        await loop.run_in_executor(None, probe.acquire)

    release.set()
    with pytest.raises(NotImplementedError):
        await task

    # Released in provision()'s finally: a fresh probe now succeeds.
    probe2 = ArtifactOperationLease(
        core.locks_path,
        ACQUISITION_SESSION_LEASE_KEY,
        LeaseMode.EXCLUSIVE,
        timeout_seconds=1.0,
    )
    await loop.run_in_executor(None, probe2.acquire)
    probe2.release()


@pytest.mark.asyncio
async def test_provision_not_yet_installed_reaches_fetch_stub(tmp_path):
    """A not-yet-installed artifact really does reach the fetch stub.

    Regression guard for the stub contract itself (lessons-testing-evidence:
    a guard/extension-point that no test ever calls is unverified plumbing).
    Consent is constructed directly rather than via preflight().grant() to
    stay network-free -- preflight() would otherwise gating-probe this
    artifact's placeholder source_url.
    """

    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    root = ArtifactRef("root-model", "r1", "int8")
    desc = make_descriptor(ref=root)
    catalog = DictCatalog({root: desc})
    consent = AcquisitionConsent(closure_fingerprint=closure_fingerprint(root, ()))

    with pytest.raises(NotImplementedError):
        await svc.provision(root, consent, catalog)
