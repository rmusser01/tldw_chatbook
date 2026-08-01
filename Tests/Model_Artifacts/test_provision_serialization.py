"""TASK-595 Task 6: provision() skeleton.

Session lease + in-process lock serialization, consent drift detection, and
the idempotent already-installed completion path. The fetch/pre-verify/
install phases are method stubs filled in by Tasks 7-8; none of these tests
need them to succeed.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from Tests.Model_Artifacts.acquisition_test_helpers import grant_consent
from Tests.Model_Artifacts.lease_processes import hold_one
from Tests.Model_Artifacts.test_acquisition_types import DictCatalog, make_descriptor
from tldw_chatbook.Model_Artifacts import ArtifactDependencyError, ArtifactRef
from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionBusyError,
    ArtifactAcquisitionService,
    ConsentMismatchError,
    TransferError,
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
    consent = grant_consent(svc, root, catalog)

    events: list[str] = []
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def fake_fetch(descriptor, staging_dir, progress_state, resolved_sources=None):
        events.append("enter")
        entered.set()
        await gate.wait()
        events.append("exit")

    async def fake_preverify(descriptor, staging_dir, progress_state, resolved_sources=None):
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
    # confirms no OTHER exception masked the ordering proof above. The raw
    # ArtifactDependencyError core.activate() raises is wrapped by
    # _run_core_call into a retryable TransferError (never-trap rule); the
    # original is still reachable as __cause__.
    assert isinstance(result1, TransferError)
    assert result1.retryable is True
    assert isinstance(result1.__cause__, ArtifactDependencyError)
    assert isinstance(result2, TransferError)
    assert result2.retryable is True
    assert isinstance(result2.__cause__, ArtifactDependencyError)


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

    async def counting_fetch(descriptor, staging_dir, progress_state, resolved_sources=None):
        calls["fetch"] += 1

    async def counting_preverify(descriptor, staging_dir, progress_state, resolved_sources=None):
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
    consent = grant_consent(svc, root, catalog)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def paused_fetch(descriptor, staging_dir, progress_state, resolved_sources=None):
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
async def test_provision_releases_session_lease_when_cancelled_during_acquire(
    tmp_path, monkeypatch
):
    """Cancelling provision() while the session lease is still being
    acquired in its worker thread must not leak the OS-backed flock.

    Regression test for a review finding: ``await run_in_executor(None,
    lease.acquire)`` sat OUTSIDE the try/finally that released the lease,
    so a ``CancelledError`` delivered while that call was still running in
    its worker thread (not yet returned) left the lease's file handle open
    for the rest of the process's life -- every later ``provision()`` call
    would then raise ``AcquisitionBusyError`` forever, and
    ``reconcile()``'s managed-staging GC (which also requires the session
    lease to be free) would be permanently disabled.

    ``lease.acquire`` cannot be interrupted once its worker thread has
    started, so the fix must WAIT for it to actually finish before
    deciding whether to release -- a naive ``if lease.acquired`` check
    performed immediately after catching the cancellation (without first
    waiting for the in-flight acquire to settle) would still read False
    here, since ``gated_acquire`` below deliberately keeps the acquire in
    flight, blocked on ``gate``, at the moment ``task.cancel()`` is called.

    ``captured["lease"]`` deliberately keeps its own reference to the
    abandoned lease object, independent of ``provision()``'s own (now-dead)
    frame: confirmed by trial that without a held reference, CPython's
    reference counting can close the abandoned handle as an incidental side
    effect of collecting the ``CancelledError``'s traceback shortly after
    ``pytest.raises`` releases it -- closing a file descriptor releases its
    flock too, which would silently "self-heal" this exact leak in-process
    and produce a false pass. A real caller that catches and logs/reports
    the ``CancelledError`` (holding its traceback, and therefore this
    frame, alive far longer) would not get that same accidental rescue, so
    the test must not rely on it either.
    """

    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    root = ArtifactRef("root-model", "r1", "int8")
    desc = make_descriptor(ref=root)
    catalog = DictCatalog({root: desc})
    consent = grant_consent(svc, root, catalog)

    started = threading.Event()
    gate = threading.Event()
    # Set once the gated acquire attempt has fully settled (succeeded OR
    # itself failed) -- awaited below before probing. Without this, the
    # probe's own acquire could race ahead of the background attempt and
    # spuriously "win" the flock first, which would make the background
    # attempt fail out on ITS OWN contention timeout (never setting
    # ``lease._handle``) and mask a real leak as a false pass.
    settled = threading.Event()
    original_acquire = ArtifactOperationLease.acquire
    captured = {}

    def gated_acquire(self):
        is_session_lease = self.key == ACQUISITION_SESSION_LEASE_KEY
        if is_session_lease:
            captured["lease"] = self
            started.set()
            assert gate.wait(timeout=5.0), "test gate was never released"
        try:
            return original_acquire(self)
        finally:
            if is_session_lease:
                settled.set()

    monkeypatch.setattr(ArtifactOperationLease, "acquire", gated_acquire)

    loop = asyncio.get_running_loop()
    task = asyncio.create_task(svc.provision(root, consent, catalog))
    assert await loop.run_in_executor(None, started.wait, 5.0), "acquire never started"

    task.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await loop.run_in_executor(None, settled.wait, 5.0), (
        "the in-flight acquire attempt never settled"
    )
    # ``captured["lease"]`` is not asserted on directly here: a correct fix
    # releases it as part of its own cancellation handling (before
    # ``await task`` above even returns), while the buggy version leaves it
    # acquired forever -- ".acquired" legitimately differs between the two,
    # so the probe below (not this snapshot) is the single source of truth.

    # Restore the real acquire before probing -- the gate above only
    # applies to the cancelled attempt.
    monkeypatch.setattr(ArtifactOperationLease, "acquire", original_acquire)

    # Direct, authoritative evidence the OS-backed flock was released: a
    # fresh non-blocking probe must succeed instead of timing out busy.
    probe = ArtifactOperationLease(
        core.locks_path,
        ACQUISITION_SESSION_LEASE_KEY,
        LeaseMode.EXCLUSIVE,
        timeout_seconds=1.0,
    )
    await loop.run_in_executor(None, probe.acquire)
    probe.release()

    # And per the review's own framing: a subsequent provision() call must
    # not raise AcquisitionBusyError (it still fails, but for the unrelated
    # reason that the placeholder source_url never resolves).
    with pytest.raises(TransferError):
        await svc.provision(root, consent, catalog)


@pytest.mark.asyncio
async def test_provision_not_yet_installed_reaches_fetch_phase(tmp_path):
    """A not-yet-installed artifact really does reach the fetch phase.

    Regression guard for the extension point itself (lessons-testing-
    evidence: a guard/extension-point that no test ever calls is unverified
    plumbing). Originally asserted the Task 6 stub's ``NotImplementedError``;
    Task 7 replaced that stub with a real implementation, so reaching it now
    means a genuine network attempt against the placeholder ``source_url``
    (an RFC 2606 ``.test`` domain, guaranteed to never resolve) -- which the
    egress policy's DNS-failure path rejects, wrapped as ``TransferError``.
    Consent is granted via ``grant_consent`` (the network-free
    ``_aggregate_closure(...).grant()`` path) rather than the real
    ``preflight()`` to keep this test's OWN setup network-free --
    ``preflight()`` would otherwise gating-probe this same placeholder URL
    itself.
    """

    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    root = ArtifactRef("root-model", "r1", "int8")
    desc = make_descriptor(ref=root)
    catalog = DictCatalog({root: desc})
    consent = grant_consent(svc, root, catalog)

    with pytest.raises(TransferError) as excinfo:
        await svc.provision(root, consent, catalog)
    assert excinfo.value.retryable is False
