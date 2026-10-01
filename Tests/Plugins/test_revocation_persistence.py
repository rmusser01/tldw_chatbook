"""Revocation through the real facade while its storage worker is unavailable."""

import asyncio
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
from tldw_chatbook.Plugins.service import PluginService


@pytest.fixture
async def revocation_case(tmp_path, native_package):
    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: SimpleNamespace(archived=False),
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    await service.bootstrap("test passphrase")
    review = await service.review_install(
        native_package(), selection=("skill:review",), workspace_id="a"
    )
    await service.commit(review, review.operation_id)
    trust = await service.review_trust(review.installation_id)
    await service.commit(trust, trust.operation_id)

    async def enable(workspace, operation_id):
        active = await service.review_activation(
            review.installation_id, workspace_id=workspace, intent="enabled"
        )
        return await service.commit(active, active.operation_id)

    entries = {}
    for workspace in ("a", "b"):
        await enable(workspace, "enable-" + workspace)
        admitted = await service.admit(service.capture_maximum(workspace), workspace)
        entries[workspace] = admitted["available_skills"]
        await service.check_entries(entries[workspace])
    # This child executes only stdlib: no app/profile import in the child.
    child = await asyncio.to_thread(
        subprocess.Popen,
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; print('ready', flush=True); sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert await asyncio.to_thread(child.stdout.readline) == "ready\n"
    cleanup_started = asyncio.Event()
    loop = asyncio.get_running_loop()
    gate = threading.Event()
    blocked = threading.Event()
    cancelled_b = threading.Event()
    survivor = False
    cleanup_error = None

    def cancel_a():
        loop.call_soon_threadsafe(cleanup_started.set)
        if not survivor:
            child.terminate()
        if cleanup_error is not None:
            raise cleanup_error

    await asyncio.to_thread(service.bind_run, entries["a"], "root-a", cancel_a)
    await asyncio.to_thread(service.bind_run, entries["b"], "root-b", cancelled_b.set)

    def block_persistence():
        def block():
            blocked.set()
            assert gate.wait(15), "test storage gate timed out"

        service._loop.call_soon_threadsafe(block)
        assert blocked.wait(2)

    requests = {}

    async def disable(target, label):
        if label not in requests:
            requests[label] = service.begin_disable(target)
        return await service.finish_revocation(requests[label])

    async def uninstall(installation_id, label):
        if label not in requests:
            requests[label] = service.begin_uninstall(installation_id)
        return await service.finish_revocation(requests[label])

    def start_disable_here():
        from tldw_chatbook.Plugins.revocation import RevocationTarget

        return asyncio.create_task(
            disable(RevocationTarget(review.installation_id, "a", False), "disable-a")
        )

    def admission_allows(workspace):
        try:
            service.check_entries_live(entries[workspace])
            return True
        except PermissionError:
            return False

    case = SimpleNamespace(
        service=service,
        requests=requests,
        disable=disable,
        uninstall=uninstall,
        installation=review.installation_id,
        entries=entries,
        child=child,
        cleanup_started=cleanup_started,
        cancelled_b=cancelled_b,
        block_persistence=block_persistence,
        release_persistence=gate.set,
        start_disable_here=start_disable_here,
        admission_allows_a=lambda: admission_allows("a"),
        admission_allows_b=lambda: admission_allows("b"),
        enable=enable,
    )

    def keep_alive():
        nonlocal survivor
        survivor = True

    case.keep_alive = keep_alive

    def fail_cleanup(error):
        nonlocal cleanup_error
        cleanup_error = error

    case.fail_cleanup = fail_cleanup
    try:
        yield case
    finally:
        gate.set()
        if child.poll() is None:
            child.terminate()
        await asyncio.to_thread(child.wait, 5)
        child.stdin.close()
        child.stdout.close()
        await asyncio.to_thread(service.complete_run, "root-a")
        await asyncio.to_thread(service.complete_run, "root-b")
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase,committed",
    [
        ("materialized", False),
        ("prepared", False),
        ("registry_committed", True),
        ("certified", True),
        ("marker_advanced", True),
        ("published", True),
    ],
)
async def test_each_durable_boundary_preserves_live_seal_and_commit_truth(
    revocation_case, phase, committed
):
    from tldw_chatbook.Plugins.revocation import RevocationFailure

    case = revocation_case
    failure = OSError("private storage diagnostic")

    def fail(current):
        if current == phase:
            raise failure

    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", fail)
    )
    with pytest.raises(RevocationFailure) as caught:
        await case.start_disable_here()
    assert caught.value.original_error is failure
    assert caught.value.receipt.committed is committed
    assert "private" not in str(caught.value) and "private" not in repr(
        caught.value.receipt
    )
    assert not case.admission_allows_a()
    assert case.admission_allows_b()
    assert not case.cancelled_b.is_set()
    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", None)
    )
    if phase == "registry_committed":
        retry = await case.start_disable_here()
        assert retry.phase == "recovery_required" and retry.committed
    elif phase == "prepared":
        retry = await case.start_disable_here()
        assert retry.phase == "aborted" and not retry.committed
    else:
        assert (await case.start_disable_here()).committed


@pytest.mark.asyncio
async def test_locked_store_preserves_original_failure_and_cleanup(revocation_case):
    from tldw_chatbook.Plugins.revocation import RevocationFailure

    case = revocation_case
    await case.service._call(lambda: case.service._coordinator.authority.lock())
    with pytest.raises(RevocationFailure) as caught:
        await case.start_disable_here()
    assert not caught.value.receipt.committed
    assert caught.value.receipt.persistence_error
    await asyncio.wait_for(case.cleanup_started.wait(), 2)
    assert not case.admission_allows_a() and case.admission_allows_b()
    await case.service.unlock("test passphrase")
    assert (await case.start_disable_here()).committed


@pytest.mark.asyncio
async def test_uninstall_retains_data_processes_and_defers_files_for_survivor(
    revocation_case, tmp_path
):
    case = revocation_case
    case.keep_alive()
    data = tmp_path / "saved-data"
    data.mkdir()
    (data / "precious").write_text("retained")

    # Seed a data-root record through the SAME authenticated commit transaction.
    # Root issuance is F8; this test qualifies F6's retention of existing roots.
    def seed_on_next_review():
        coordinator = case.service._coordinator
        original = coordinator._apply_review

        def apply(cursor, review, retained):
            original(cursor, review, retained)
            cursor.execute(
                "INSERT INTO data_roots VALUES (?, ?, ?, ?, ?, ?)",
                ("saved", case.installation, "a", str(data), 1, 0),
            )
            coordinator._apply_review = original

        coordinator._apply_review = apply

    await case.service._call(seed_on_next_review)
    await case.enable("b", "root-fixture")
    receipt = await case.uninstall(case.installation, "uninstall")
    assert receipt.committed and receipt.cleanup_pending and not receipt.runtime_stopped
    assert case.cancelled_b.is_set()
    package = case.service.profile_root / "plugins/packages" / case.installation
    assert package.exists() and (data / "precious").read_text() == "retained"
    snapshot = await case.service._call(
        lambda: case.service._coordinator.published_snapshot()
    )
    assert (
        not snapshot["installations"]
        and not snapshot["revision_trust"]
        and not snapshot["mappings"]
    )
    assert snapshot["tombstones"][0]["installation_id"] == case.installation
    assert snapshot["data_roots"][0]["root_id"] == "saved"
    assert (
        len(
            await case.service._call(
                lambda: case.service._coordinator.owner.list_processes(
                    limit=10, offset=0
                )
            )
        )
        == 2
    )
    case.child.terminate()
    await asyncio.to_thread(case.child.wait, 2)
    await asyncio.to_thread(case.service.complete_run, "root-a")
    await asyncio.to_thread(case.service.complete_run, "root-b")
    receipt = await case.uninstall(case.installation, "uninstall")
    assert receipt.runtime_stopped and not receipt.cleanup_pending
    assert not package.exists() and data.exists()
    assert (await case.uninstall(case.installation, "uninstall")).committed


@pytest.mark.asyncio
async def test_uninstall_keeps_files_for_unregistered_surviving_process(
    revocation_case,
):
    case = revocation_case

    def reserve_unknown():
        owner = case.service._coordinator.owner
        return owner.reserve_launch(
            "orphan",
            case.installation,
            "b",
            case.service.live_runs()[0].revision_digest,
        )

    token = await case.service._call(reserve_unknown)
    await asyncio.to_thread(case.service.complete_run, "root-a")
    await asyncio.to_thread(case.service.complete_run, "root-b")
    receipt = await case.uninstall(case.installation, "uninstall-orphan")
    assert receipt.committed and receipt.cleanup_pending and not receipt.runtime_stopped
    package = case.service.profile_root / "plugins/packages" / case.installation
    assert package.exists()
    assert token in await case.service._call(
        lambda: case.service._coordinator.owner.unsettled_tokens(case.installation)
    )


@pytest.mark.asyncio
async def test_new_request_after_prepared_abort_can_reconcile_for_fresh_enable(
    revocation_case,
):
    from tldw_chatbook.Plugins.revocation import RevocationFailure, RevocationTarget

    case = revocation_case

    def fail(phase):
        if phase == "prepared":
            raise OSError("full disk")

    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", fail)
    )
    with pytest.raises(RevocationFailure):
        await case.start_disable_here()
    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", None)
    )
    assert (await case.start_disable_here()).phase == "aborted"
    with pytest.raises(PermissionError):
        await case.enable("a", "too-early")
    assert (
        await case.disable(
            RevocationTarget(case.installation, "a", False), "fresh-disable"
        )
    ).committed
    await case.enable("a", "reconciled-enable")
    fresh = await case.service.admit(case.service.capture_maximum("a"), "fresh")
    await case.service.check_entries(fresh["available_skills"])
    with pytest.raises(PermissionError):
        await case.service.check_entries(case.entries["a"])


@pytest.mark.asyncio
async def test_original_storage_failure_survives_cleanup_failure(revocation_case):
    from tldw_chatbook.Plugins.revocation import RevocationFailure

    case = revocation_case
    cleanup = RuntimeError("private cleanup")
    original = OSError("private storage")
    case.fail_cleanup(cleanup)

    def fail(phase):
        raise original

    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", fail)
    )
    with pytest.raises(RevocationFailure) as caught:
        await case.start_disable_here()
    assert (
        caught.value.original_error is original and caught.value.__cause__ is original
    )
    assert caught.value.receipt.cleanup_errors == ("RuntimeError",)
    assert case.service.fences.operations[
        case.requests["disable-a"].request_id
    ].cleanup_errors == [cleanup]
    assert not case.admission_allows_a() and case.admission_allows_b()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [
        "materialized",
        "prepared",
        "registry_committed",
        "certified",
        "marker_advanced",
        "published",
    ],
)
async def test_cleanup_runs_while_each_durable_boundary_is_stalled(
    revocation_case, phase
):
    case = revocation_case
    blocked, release = threading.Event(), threading.Event()

    def stall(current):
        if current == phase:
            blocked.set()
            assert release.wait(5)

    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", stall)
    )
    task = case.start_disable_here()
    try:
        await asyncio.wait_for(case.cleanup_started.wait(), 2)
        assert await asyncio.to_thread(blocked.wait, 2)
        await asyncio.to_thread(case.child.wait, 2)
        assert not task.done()
        assert not case.admission_allows_a() and case.admission_allows_b()
        assert not case.cancelled_b.is_set()
    finally:
        release.set()
    assert (await task).committed


@pytest.mark.asyncio
async def test_unobserved_process_inventory_never_claims_stop(revocation_case):
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    case = revocation_case

    def reserve_unknown():
        owner = case.service._coordinator.owner
        return owner.reserve_launch(
            "unregistered",
            case.installation,
            "a",
            case.service.live_runs()[0].revision_digest,
        )

    await case.service._call(reserve_unknown)
    await asyncio.to_thread(case.service.complete_run, "root-a")
    case.block_persistence()
    operation = asyncio.create_task(
        case.disable(RevocationTarget(case.installation, "a", False), "unobserved")
    )
    try:
        await asyncio.sleep(0)
        status = case.service.revocation_status(case.requests["unobserved"])
        assert status.runtime_stopped is None and status.cleanup_pending
    finally:
        case.release_persistence()
        receipt = await operation
    assert receipt.runtime_stopped is False and receipt.cleanup_pending


@pytest.mark.asyncio
async def test_superseded_locked_request_cannot_revoke_fresh_run(revocation_case):
    from tldw_chatbook.Plugins.revocation import RevocationFailure, RevocationTarget

    case = revocation_case
    target = RevocationTarget(case.installation, "a", False)
    await case.service._call(lambda: case.service._coordinator.authority.lock())
    with pytest.raises(RevocationFailure):
        await case.disable(target, "old-locked")
    await case.service.unlock("test passphrase")
    assert (await case.disable(target, "replacement")).committed
    await case.enable("a", "resume-after-replacement")
    fresh = await case.service.admit(
        case.service.capture_maximum("a"), "fresh-replacement"
    )
    cancelled = threading.Event()
    await asyncio.to_thread(
        case.service.bind_run, fresh["available_skills"], "fresh-root", cancelled.set
    )
    try:
        with pytest.raises(ValueError, match="superseded"):
            await case.disable(target, "old-locked")
        assert not cancelled.is_set()
        await case.service.check_entries(fresh["available_skills"])
    finally:
        await asyncio.to_thread(case.service.complete_run, "fresh-root")


@pytest.mark.asyncio
async def test_uninstall_does_not_follow_replaced_package_parent(
    revocation_case, tmp_path
):
    case = revocation_case
    await asyncio.to_thread(case.service.complete_run, "root-a")
    await asyncio.to_thread(case.service.complete_run, "root-b")
    packages = case.service.profile_root / "plugins/packages"
    outside = tmp_path / "outside-packages"

    def replace_parent(phase):
        if phase == "published":
            packages.rename(outside)
            packages.symlink_to(outside, target_is_directory=True)

    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", replace_parent)
    )
    receipt = await case.uninstall(case.installation, "uninstall-replaced-parent")
    assert receipt.committed and receipt.cleanup_pending and receipt.cleanup_errors
    assert (outside / case.installation / "plugin.json").exists()
    packages.unlink()
    outside.rename(packages)
    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", None)
    )
    assert not (
        await case.uninstall(case.installation, "uninstall-replaced-parent")
    ).cleanup_pending
