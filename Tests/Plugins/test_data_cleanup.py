"""Exact-root cleanup through real reviewed authority and retained host ownership."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

from Tests.hooks_v2_process_support import child_argv

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.fixture
def data_cleanup_case(plugin_stack, native_package):
    from types import SimpleNamespace

    from tldw_chatbook.Plugins.inspection import inspect_package

    stack = plugin_stack
    review = stack.call(
        lambda: stack.coordinator.review(
            inspect_package(native_package()),
            selection=("skill:review",),
            workspace_id="a",
        )
    )
    receipt = stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert receipt.committed
    assert (
        stack.call(stack.coordinator.published_snapshot)["installations"][0][
            "installation_id"
        ]
        == review.installation_id
    )
    return SimpleNamespace(
        stack=stack,
        installation=review.installation_id,
        revision=review.inspection.effective_digest,
    )


def create_root(case, workspace_id=None):
    stack = case.stack
    review = stack.call(
        lambda: stack.coordinator.review_data_creation(
            case.installation, workspace_id=workspace_id
        )
    )
    return stack.call(
        lambda: stack.coordinator.create_data(review, review.operation_id)
    )


def submit(stack, callback):
    async def invoke():
        return await callback()

    return asyncio.run_coroutine_threadsafe(invoke(), stack.loop)


@pytest.mark.asyncio
async def test_idle_writer_blocks_data_deletion(data_cleanup_case):
    """Removing revision leases must never let cleanup erase an idle writer's root."""
    case = data_cleanup_case
    stack = case.stack
    ref = create_root(case)
    token = stack.call(
        lambda: stack.owner.reserve_launch(
            "writer", case.installation, "a", case.revision, roots=(ref,)
        )
    )
    script = """
import json, os, socket, sys
from pathlib import Path
socket.socket.connect = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network refused"))
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve().parents[1] == Path(sys.argv[2])
assert os.environ["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
print(json.dumps({"pid": os.getpid(), "source": tldw_chatbook.__file__}), flush=True)
for line in sys.stdin:
    if line.strip() == "write":
        (Path(sys.argv[1]) / "writer.txt").write_text("owned idle writer")
        print("wrote", flush=True)
    else:
        break
"""
    child = subprocess.Popen(  # noqa: ASYNC220 -- retained real child, explicitly reaped below.
        child_argv(script) + [str(ref.path), str(Path.cwd())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pending = None
    try:
        provenance = json.loads(child.stdout.readline())
        stack.call(lambda: stack.owner.publish_process(token, provenance))
        stack.call(lambda: stack.owner.set_process_kind(token, "idle_connection"))
        assert (
            stack.call(
                lambda: stack.owner.active_revision_leases(
                    case.installation, case.revision
                )
            )
            == 0
        )
        review = stack.call(lambda: stack.coordinator.review_data_deletion((ref,)))
        pending = submit(
            stack, lambda: stack.coordinator.delete_data((ref,), review.operation_id)
        )
        for _ in range(200):
            fenced = stack.call(
                lambda: stack.coordinator.root_usage.is_fenced(ref.root_id)
            )
            if fenced:
                break
            await asyncio.sleep(0.01)
        assert fenced
        assert ref.path.is_dir() and not pending.done()
        child.stdin.write("write\n")
        child.stdin.flush()
        assert child.stdout.readline().strip() == "wrote"
        assert (ref.path / "writer.txt").read_text() == "owned idle writer"
        child.stdin.write("stop\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        stack.call(lambda: stack.owner.settle_process(token, True))
        receipt = await asyncio.wrap_future(pending)
        assert receipt.committed and not receipt.cleanup_pending
        assert not ref.path.exists()
        root = stack.call(stack.coordinator.published_snapshot)["data_roots"][0]
        assert root["generation"] == ref.generation + 1
        assert root["custody"]["state"] == "cleaned_absent"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
            stack.call(lambda: stack.owner.settle_process(token, True))
        if pending is not None and not pending.done():
            await asyncio.wrap_future(pending)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()


def reserve(case, ref, workspace="a"):
    return case.stack.call(
        lambda: case.stack.owner.reserve_launch(
            "host-user", case.installation, workspace, case.revision, roots=(ref,)
        )
    )


@pytest.mark.asyncio
async def test_shared_users_reader_and_cancel_wait_preserve_other_owners(
    data_cleanup_case,
):
    case = data_cleanup_case
    ref = create_root(case)
    a, b = reserve(case, ref), reserve(case, ref, "b")
    reader = open(ref.path / "reader", "w+")  # noqa: ASYNC230, SIM115
    try:
        review = case.stack.call(
            lambda: case.stack.coordinator.review_data_deletion((ref,))
        )
        pending = submit(
            case.stack,
            lambda: case.stack.coordinator.delete_data((ref,), review.operation_id),
        )
        for _ in range(200):
            rows = case.stack.call(
                lambda: case.stack.coordinator.root_usage.blockers((ref,))
            )
            if case.stack.call(
                lambda: case.stack.coordinator.root_usage.is_fenced(ref.root_id)
            ):
                break
            await asyncio.sleep(0.01)
        assert {row.get("workspace_id") for row in rows} >= {"a", "b"}
        case.stack.call(lambda: case.stack.owner.settle_process(a, True))
        assert not pending.done() and ref.path.exists()
        reader.write("held reader")
        reader.flush()
        case.stack.call(
            lambda: case.stack.coordinator.cancel_data_deletion(review.operation_id)
        )
        receipt = await asyncio.wrap_future(pending)
        assert receipt.phase == "cancelled" and ref.path.exists()
        assert case.stack.call(
            lambda: case.stack.owner.unsettled_tokens(case.installation)
        ) == (b,)
    finally:
        reader.close()
        case.stack.call(lambda: case.stack.owner.settle_process(b, True))


@pytest.mark.parametrize("replacement", ["leaf", "anchor", "link", "boot", "birth"])
def test_reviewed_replacements_refuse_and_preserve_external_sentinel(
    data_cleanup_case, monkeypatch, tmp_path, replacement
):
    import tldw_chatbook.Plugins.data_cleanup as module

    case = data_cleanup_case
    ref = create_root(case)
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("untouched")
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_deletion((ref,))
    )
    if replacement == "boot":
        monkeypatch.setattr(
            module, "boot_identity", lambda: "00000000-0000-0000-0000-000000000001"
        )
    elif replacement == "birth":
        real = module.physical_identity

        def changed(fd):
            value = real(fd)
            return dict(
                value, birth_nanoseconds=(value["birth_nanoseconds"] + 1) % 10**9
            )

        monkeypatch.setattr(module, "physical_identity", changed)
    else:
        target = ref.path.parent if replacement == "anchor" else ref.path
        target.rename(target.with_name(target.name + "-original"))
        if replacement == "link":
            target.symlink_to(tmp_path, target_is_directory=True)
        else:
            target.mkdir()
    with pytest.raises((PermissionError, OSError)):
        case.stack.call(
            lambda: case.stack.coordinator.delete_data((ref,), review.operation_id)
        )
    assert sentinel.read_text() == "untouched"
    assert case.stack.call(
        lambda: case.stack.coordinator.root_usage.is_fenced(ref.root_id)
    )


@pytest.mark.asyncio
async def test_missing_join_and_uncertain_terminal_never_prove_drain(data_cleanup_case):
    case = data_cleanup_case
    ref = create_root(case)
    token = reserve(case, ref)
    case.stack.call(lambda: case.stack.owner.settle_process(token, False))
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_deletion((ref,))
    )
    pending = submit(
        case.stack,
        lambda: case.stack.coordinator.delete_data((ref,), review.operation_id),
    )
    await asyncio.sleep(0.03)
    assert not pending.done()

    def lose_join():
        with case.stack.registry.transaction() as cursor:
            cursor.execute("DELETE FROM root_users WHERE owner_token=?", (token,))

    case.stack.call(lose_join)
    rows = case.stack.call(lambda: case.stack.coordinator.root_usage.blockers((ref,)))
    assert any(row["reason"] == "root_owner_coverage_incomplete" for row in rows)
    with pytest.raises(PermissionError):
        case.stack.call(lambda: case.stack.owner.settle_process(token, True))
    case.stack.call(
        lambda: case.stack.coordinator.cancel_data_deletion(review.operation_id)
    )
    await asyncio.wrap_future(pending)
    assert ref.path.exists()


def test_uninstall_retains_original_owner_and_explicit_attachment_changes_generation(
    data_cleanup_case, native_package
):
    from tldw_chatbook.Plugins.inspection import inspect_package

    case = data_cleanup_case
    ref = create_root(case)
    (ref.path / "keep").write_text("retained")
    case.stack.call(lambda: case.stack.coordinator.uninstall(case.installation))
    assert ref.path.is_dir()
    new = case.stack.call(
        lambda: case.stack.coordinator.review(
            inspect_package(native_package()),
            selection=("skill:review",),
            workspace_id=None,
        )
    )
    case.stack.call(lambda: case.stack.coordinator.commit(new, new.operation_id))
    assert new.installation_id != ref.installation_id
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_attachment(
            (ref,), new.installation_id
        )
    )
    case.stack.call(
        lambda: case.stack.coordinator.delete_data((ref,), review.operation_id)
    )
    row = case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][0]
    assert row["installation_id"] == ref.installation_id
    assert row["custody"]["attached_installation_id"] == new.installation_id
    assert row["generation"] == ref.generation + 1
    with pytest.raises(PermissionError):
        reserve(case, ref)
    assert (ref.path / "keep").read_text() == "retained"


@pytest.mark.asyncio
async def test_actual_root_membership_and_generation_invalidate_managed_resume(
    native_console,
):
    import threading

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )

    case = native_console
    installed = await case.install()
    service = case.service
    creation = await service.review_data_creation(
        installed.installation_id, workspace_id="workspace-a"
    )
    ref = await service.create_data(creation, creation.operation_id)
    entries = (
        await service.admit(service.capture_maximum("workspace-a"), "data-pending")
    )["available_skills"]
    await asyncio.to_thread(
        service.bind_run, entries, "data-run", threading.Event().set
    )
    try:
        pin = await asyncio.to_thread(
            service.capture_resume_pin, entries, "data-run", "conversation", "message"
        )
        assert json.loads(pin)["installations"][0]["data_coverage"] == "known"
        sealed = await asyncio.to_thread(
            service.seal_resume_checkpoint,
            pin,
            parse_provider_continuation_json(json.dumps(_checkpoint())),
            "conversation",
            "message",
        )
        maximum = await service.resume_maximum(
            service.capture_maximum("workspace-a"), sealed, "conversation", "message"
        )
        assert (await service.admit(maximum, "data-control"))["available_skills"]
        attachment = await service.review_data_attachment((ref,), None)
        await service.delete_data((ref,), attachment.operation_id)
        with pytest.raises(PermissionError):
            await service.resume_maximum(
                service.capture_maximum("workspace-a"),
                sealed,
                "conversation",
                "message",
            )
        with pytest.raises(PermissionError):
            await service.check_entries(entries)
    finally:
        await asyncio.to_thread(service.complete_run, "data-run")


@pytest.mark.asyncio
async def test_pending_grant_storage_does_not_block_live_fence(
    data_cleanup_case, monkeypatch
):
    import threading

    case = data_cleanup_case
    ref = create_root(case)
    entered, proceed = threading.Event(), threading.Event()
    real = case.stack.owner._reserve_launch

    def hold(*args):
        entered.set()
        assert proceed.wait(10)
        return real(*args)

    monkeypatch.setattr(case.stack.owner, "_reserve_launch", hold)
    task = asyncio.create_task(asyncio.to_thread(reserve, case, ref))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        fence = case.stack.coordinator.root_usage.fence((ref,))
        assert case.stack.coordinator.root_usage.is_fenced(ref.root_id)
    finally:
        proceed.set()
    with pytest.raises(PermissionError):
        await task
    assert (
        case.stack.call(lambda: case.stack.owner.unsettled_tokens(case.installation))
        == ()
    )
    assert not case.stack.coordinator.root_usage.pending
    case.stack.coordinator.root_usage.unfence(fence)
    token = reserve(case, ref)
    case.stack.call(lambda: case.stack.owner.settle_process(token, True))


@pytest.mark.asyncio
async def test_late_process_publication_retains_owner_after_root_fence(
    data_cleanup_case,
):
    case = data_cleanup_case
    ref = create_root(case)
    token = reserve(case, ref)
    fence = case.stack.coordinator.root_usage.fence((ref,))
    with pytest.raises(PermissionError):
        case.stack.call(
            lambda: case.stack.owner.publish_process(
                token,
                {
                    "host_handle": "launched before publication",
                    "pid": os.getpid(),
                    "start_identity": "not this live pid",
                },
            )
        )
    assert token in case.stack.call(
        lambda: case.stack.owner.unsettled_tokens(case.installation)
    )
    assert case.stack.call(lambda: case.stack.coordinator.root_usage.blockers((ref,)))
    case.stack.call(lambda: case.stack.owner.settle_process(token, True))
    case.stack.coordinator.root_usage.unfence(fence)


@pytest.mark.parametrize(
    "cut", ["data_fenced", "data_deleting", "data_unlinked", "data_root_removed"]
)
def test_cleanup_cut_retains_phase_and_resumes_only_after_reviewed_reconciliation(
    data_cleanup_case, cut
):
    from Tests.Plugins.test_surviving_process_recovery import reopen

    case = data_cleanup_case
    ref = create_root(case)
    (ref.path / "one").write_text("one")
    (ref.path / "two").write_text("two")
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_deletion((ref,))
    )

    def stop(phase):
        if phase == cut:
            raise OSError("controlled cleanup interruption")

    case.stack.call(lambda: setattr(case.stack.coordinator, "progress", stop))
    with pytest.raises(OSError):
        case.stack.call(
            lambda: case.stack.coordinator.delete_data((ref,), review.operation_id)
        )
    case.stack.call(lambda: setattr(case.stack.coordinator, "progress", None))
    current = case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][
        0
    ]
    assert current["generation"] == ref.generation and current["deletion_fenced"]
    assert current["cleanup"]["phase"] == (
        "waiting" if cut == "data_fenced" else "deleting"
    )
    lookup = case.stack.call(
        lambda: case.stack.coordinator.lookup_operation(review.operation_id)
    )
    assert lookup.cleanup_pending
    existed = ref.path.exists()
    reopen(case.stack, clean=False)
    lookup = case.stack.call(
        lambda: case.stack.coordinator.lookup_operation(review.operation_id)
    )
    assert lookup.cleanup_pending and ref.path.exists() == existed
    reconciliation = case.stack.call(
        lambda: case.stack.coordinator.review_data_reconciliation(
            (ref,), confirm_quiescence=lambda refs: True
        )
    )
    case.stack.call(
        lambda: case.stack.coordinator.reconcile_data(
            reconciliation, reconciliation.operation_id
        )
    )
    resume = case.stack.call(
        lambda: case.stack.coordinator.review_data_cleanup_resume((ref,))
    )
    assert resume.operation_id != review.operation_id
    receipt = case.stack.call(
        lambda: case.stack.coordinator.delete_data((ref,), resume.operation_id)
    )
    assert receipt.phase == "complete" and not ref.path.exists()
    assert (
        case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][0][
            "generation"
        ]
        == ref.generation + 1
    )


@pytest.mark.asyncio
async def test_explicit_cancel_work_closes_only_reviewed_root_handles(
    data_cleanup_case,
):
    case = data_cleanup_case
    ref, other = create_root(case), create_root(case)
    handles = [
        open(ref.path / "a", "w"),  # noqa: ASYNC230, SIM115
        open(ref.path / "b", "w"),  # noqa: ASYNC230, SIM115
        open(other.path / "c", "w"),  # noqa: ASYNC230, SIM115
    ]
    tokens = [reserve(case, ref), reserve(case, ref, "b"), reserve(case, other)]
    calls = []
    for index, token in enumerate(tokens):

        def close(index=index, token=token):
            calls.append(index)
            handles[index].close()
            case.stack.owner.settle_process(token, True)

        case.stack.call(
            lambda token=token, close=close: (
                case.stack.coordinator.root_usage.retain_cancel(token, close)
            )
        )
    try:
        review = case.stack.call(
            lambda: case.stack.coordinator.review_data_deletion((ref,))
        )
        deletion = submit(
            case.stack,
            lambda: case.stack.coordinator.delete_data((ref,), review.operation_id),
        )
        for _ in range(200):
            if case.stack.call(
                lambda: (
                    review.operation_id in case.stack.coordinator.root_usage.operations
                )
            ):
                break
            await asyncio.sleep(0.01)
        assert calls == [] and not deletion.done()
        case.stack.call(
            lambda: case.stack.coordinator.cancel_data_work(review.operation_id)
        )
        receipt = await asyncio.wrap_future(deletion)
        assert receipt.phase == "complete" and sorted(calls) == [0, 1]
        assert not handles[2].closed and other.path.exists()
    finally:
        for index, handle in enumerate(handles):
            if not handle.closed:
                handle.close()
                case.stack.call(
                    lambda index=index: case.stack.owner.settle_process(
                        tokens[index], True
                    )
                )


def test_generic_commit_cannot_bypass_root_reconciliation_entry(data_cleanup_case):
    case = data_cleanup_case
    ref = create_root(case)
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_reconciliation(
            (ref,), confirm_quiescence=lambda refs: True
        )
    )
    with pytest.raises(PermissionError):
        case.stack.call(
            lambda: case.stack.coordinator.commit(review, review.operation_id)
        )
    assert (
        case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][0][
            "generation"
        ]
        == ref.generation
    )


@pytest.mark.parametrize(
    "cut", ["prepared", "registry_committed", "certified", "marker_advanced"]
)
def test_fence_persistence_cut_never_unlinks_before_committed_deleting(
    data_cleanup_case, cut
):
    case = data_cleanup_case
    ref = create_root(case)
    (ref.path / "untouched").write_text("saved")
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_deletion((ref,))
    )

    def fail(phase):
        if phase == cut:
            raise OSError("controlled fence publication cut")

    case.stack.call(lambda: setattr(case.stack.coordinator, "progress", fail))
    with pytest.raises(OSError):
        case.stack.call(
            lambda: case.stack.coordinator.delete_data((ref,), review.operation_id)
        )
    case.stack.call(lambda: setattr(case.stack.coordinator, "progress", None))
    assert (ref.path / "untouched").read_text() == "saved"
    assert case.stack.coordinator.root_usage.is_fenced(ref.root_id)
    if cut == "registry_committed":
        assert any(
            receipt.phase == "recovery_required"
            for receipt in case.stack.call(case.stack.coordinator.recover)
        )
    else:
        case.stack.call(case.stack.coordinator.recover)
        case.stack.call(
            lambda: case.stack.coordinator.cancel_data_deletion(review.operation_id)
        )
        assert not case.stack.coordinator.root_usage.is_fenced(ref.root_id)
        assert (ref.path / "untouched").read_text() == "saved"


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_owned_deletion_running(data_cleanup_case):
    case = data_cleanup_case
    ref = create_root(case)
    token = reserve(case, ref)
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_deletion((ref,))
    )
    future = submit(
        case.stack,
        lambda: case.stack.coordinator.delete_data((ref,), review.operation_id),
    )
    waiter = asyncio.wrap_future(future)
    for _ in range(200):
        operation = case.stack.coordinator.root_usage.operations.get(
            review.operation_id
        )
        if operation is not None and operation.receipt.phase == "waiting":
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("deletion never reached wait")
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    case.stack.call(lambda: case.stack.owner.settle_process(token, True))
    for _ in range(200):
        if operation.task.done():
            break
        await asyncio.sleep(0.01)
    assert operation.task.done() and operation.task.result().phase == "complete"
    assert not ref.path.exists()


@pytest.mark.asyncio
async def test_service_root_fence_precedes_worker_io(native_console):
    import threading

    case = native_console
    installed = await case.install()
    creation = await case.service.review_data_creation(installed.installation_id)
    ref = await case.service.create_data(creation, creation.operation_id)
    review = await case.service.review_data_deletion((ref,))
    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(5)

    blocked = asyncio.create_task(case.service._call(block))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    deletion = asyncio.create_task(
        case.service.delete_data((ref,), review.operation_id)
    )
    try:
        await asyncio.sleep(0.02)
        assert case.service._coordinator.root_usage.is_fenced(ref.root_id)
        assert not deletion.done()
    finally:
        release.set()
        await blocked
    assert (await deletion).phase == "complete"


@pytest.mark.parametrize("fence_timing", [None, "before_publication", "during_acquire"])
def test_later_grant_preserves_all_original_publication_epochs(
    data_cleanup_case, monkeypatch, fence_timing
):
    case = data_cleanup_case
    a, b = create_root(case), create_root(case)
    usage = case.stack.coordinator.root_usage
    token = reserve(case, a)
    old_epoch = usage.grant_epochs[token]
    fence = None
    real = usage._insert

    def fence_first_root(cursor, owner_token, grant):
        nonlocal fence
        value = real(cursor, owner_token, grant)
        if fence_timing == "during_acquire":
            fence = usage.fence((a,))
        return value

    monkeypatch.setattr(usage, "_insert", fence_first_root)
    try:
        if fence_timing == "during_acquire":
            with pytest.raises(PermissionError, match="root_access_fenced"):
                case.stack.call(lambda: usage.acquire(b, token))
        else:
            case.stack.call(lambda: usage.acquire(b, token))
        assert dict(usage.grant_epochs[token])[a.root_id] == dict(old_epoch)[a.root_id]
        if fence_timing == "before_publication":
            fence = usage.fence((a,))
        if fence is not None:
            usage.unfence(fence)
            assert not usage.is_fenced(a.root_id)
        publication = lambda: case.stack.owner.publish_process(
            token,
            {
                "host_handle": "exact pending owner",
                "pid": os.getpid(),
                "start_identity": "not live PID proof",
            },
        )
        if fence_timing:
            with pytest.raises(PermissionError, match="root_late_publication_fenced"):
                case.stack.call(publication)
        else:
            case.stack.call(publication)
        assert case.stack.call(lambda: usage.blockers((a,)))
        assert case.stack.call(lambda: usage.blockers((b,)))
    finally:
        case.stack.call(lambda: case.stack.owner.settle_process(token, True))
        if fence:
            usage.unfence(fence)


@pytest.mark.parametrize("expire", [False, True])
def test_final_cleanup_preserves_original_review_deadline(
    data_cleanup_case, monkeypatch, expire
):
    import types

    import tldw_chatbook.Plugins.coordinator as coordinator_module
    import tldw_chatbook.Plugins.data_cleanup as cleanup_module

    case = data_cleanup_case
    ref = create_root(case)
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_deletion((ref,))
    )
    now = [review.expires_at - 1]
    clock = types.SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(cleanup_module, "time", clock)
    monkeypatch.setattr(coordinator_module, "time", clock)

    def cut(phase):
        if phase == "data_root_removed" and expire:
            now[0] = review.expires_at + 1

    case.stack.call(lambda: setattr(case.stack.coordinator, "progress", cut))
    if expire:
        with pytest.raises(ValueError, match="expired review"):
            case.stack.call(
                lambda: case.stack.coordinator.delete_data((ref,), review.operation_id)
            )
        row = case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][
            0
        ]
        assert (
            row["cleanup"]["phase"] == "deleting"
            and row["generation"] == ref.generation
        )
        assert row["deletion_fenced"] and not ref.path.exists()
        fresh = case.stack.call(
            lambda: case.stack.coordinator.review_data_cleanup_resume((ref,))
        )
        assert fresh.expires_at > now[0]
        receipt = case.stack.call(
            lambda: case.stack.coordinator.delete_data((ref,), fresh.operation_id)
        )
    else:
        receipt = case.stack.call(
            lambda: case.stack.coordinator.delete_data((ref,), review.operation_id)
        )
    assert receipt.phase == "complete"
    row = case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][0]
    assert (
        row["generation"] == ref.generation + 1
        and row["custody"]["state"] == "cleaned_absent"
    )


def test_later_grants_preserve_released_history_without_refreshing_held_epochs(
    data_cleanup_case,
):
    case = data_cleanup_case
    a, b, c = create_root(case), create_root(case), create_root(case)
    usage = case.stack.coordinator.root_usage
    token = reserve(case, a)
    case.stack.call(lambda: usage.acquire(b, token))
    old_b_epoch = dict(usage.grant_epochs[token])[b.root_id]
    a_usage = case.stack.call(
        lambda: case.stack.registry._connection.execute(
            "SELECT usage_token FROM root_users WHERE owner_token=? AND root_id=?",
            (token, a.root_id),
        ).fetchone()[0]
    )
    case.stack.call(lambda: usage.release(a_usage, confirmed=True))
    fence = usage.fence((a,))
    try:
        case.stack.call(lambda: usage.acquire(c, token))
        assert dict(usage.grant_epochs[token]) == {b.root_id: old_b_epoch, c.root_id: 0}
        case.stack.call(
            lambda: case.stack.owner.publish_process(
                token, {"host_handle": "exact remaining B/C"}
            )
        )
        rows = case.stack.call(
            lambda: case.stack.registry._connection.execute(
                "SELECT root_id, state FROM root_users WHERE owner_token=?", (token,)
            ).fetchall()
        )
        assert {row[0]: row[1] for row in rows} == {
            a.root_id: "released",
            b.root_id: "held",
            c.root_id: "held",
        }
        assert not case.stack.call(lambda: usage.blockers((a,)))
    finally:
        case.stack.call(lambda: case.stack.owner.settle_process(token, True))
        usage.unfence(fence)
