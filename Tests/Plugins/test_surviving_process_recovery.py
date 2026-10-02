"""Clean/dirty checkpoint and actual-process root identity recovery controls."""

import json
import subprocess
from pathlib import Path

import pytest

from Tests.hooks_v2_process_support import child_argv

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Plugins.test_data_cleanup import create_root, reserve
from Tests.Plugins.test_data_cleanup import data_cleanup_case as _data_cleanup_case

data_cleanup_case = _data_cleanup_case


def reopen(stack, *, clean):
    from tldw_chatbook.Plugins.authority_store import PluginAuthorityStore
    from tldw_chatbook.Plugins.coordinator import PluginCoordinator
    from tldw_chatbook.Plugins.registry import PluginRegistry
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    async def operation():
        if clean:
            stack.coordinator.root_usage.shutdown()
        root, trust, marker = (
            stack.owner.root,
            stack.authority.store_dir,
            stack.authority.marker_store,
        )
        stack.registry.close()
        stack.owner.close()
        stack.owner = PluginRuntimeOwner(root)
        assert stack.owner.try_acquire()
        stack.registry = PluginRegistry(root / "registry.sqlite3", owner=stack.owner)
        stack.authority = PluginAuthorityStore(
            trust, marker, accept_reduced_protection=True
        )
        stack.authority.unlock("test passphrase")
        stack.coordinator = PluginCoordinator(
            stack.registry, stack.authority, stack.owner
        )
        assert not any(
            item.phase == "recovery_required"
            for item in await stack.coordinator.recover()
        )

    return stack.call(operation)


def test_actual_native_fields_stable_in_separate_processes(data_cleanup_case):
    case = data_cleanup_case
    ref = create_root(case)
    code = """
import json, os, socket, sys
from pathlib import Path
socket.socket.connect = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network refused"))
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve().parents[1] == Path(sys.argv[2])
assert os.environ["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
from tldw_chatbook.Plugins.data_cleanup import physical_identity, boot_identity, DIR_FLAGS
fd = os.open(sys.argv[1], DIR_FLAGS)
try:
 print(json.dumps({"identity": physical_identity(fd), "boot": boot_identity()}))
finally:
 os.close(fd)
"""
    outputs = []
    for index in range(3):
        result = subprocess.run(
            child_argv(code) + [str(ref.path), str(Path.cwd())],
            text=True,
            capture_output=True,
            timeout=20,
            check=True,
        )
        outputs.append(json.loads(result.stdout))
        (ref.path / str(index)).write_text("entry changes do not change incarnation")
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0]["identity"]["birth_seconds"] > 0
    print(json.dumps(outputs))


def test_clean_restart_after_ordinary_authority_commit_keeps_roots_usable(
    data_cleanup_case,
):
    case = data_cleanup_case
    ref = create_root(case)
    token = reserve(case, ref)
    case.stack.call(lambda: case.stack.owner.settle_process(token, True))
    review = case.stack.call(
        lambda: case.stack.coordinator.review_trust(case.installation)
    )
    case.stack.call(lambda: case.stack.coordinator.commit(review, review.operation_id))
    reopen(case.stack, clean=True)
    token = reserve(case, ref)
    case.stack.call(lambda: case.stack.owner.settle_process(token, True))
    deletion = case.stack.call(
        lambda: case.stack.coordinator.review_data_deletion((ref,))
    )
    case.stack.call(
        lambda: case.stack.coordinator.delete_data((ref,), deletion.operation_id)
    )
    assert not ref.path.exists()


@pytest.mark.parametrize(
    "checkpoint", ["dirty", "missing", "malformed", "marker_mismatch"]
)
def test_unqualified_prior_checkpoint_fences_retained_roots(
    data_cleanup_case, checkpoint
):
    case = data_cleanup_case
    ref = create_root(case)
    path = case.stack.authority.marker_store.path.with_name("runtime_checkpoint.json")
    if checkpoint == "missing":
        path.unlink()
    elif checkpoint == "malformed":
        path.write_text('{"phase":"clean"}')
    elif checkpoint == "marker_mismatch":
        value = json.loads(path.read_text())
        value["phase"] = "clean"
        value["marker"]["recovery_snapshot_digest"] = "0" * 64
        path.write_text(json.dumps(value))
    reopen(case.stack, clean=False)
    assert case.stack.call(
        lambda: case.stack.coordinator.root_usage.is_fenced(ref.root_id)
    )
    with pytest.raises(PermissionError):
        reserve(case, ref)
    with pytest.raises(PermissionError):
        case.stack.call(lambda: case.stack.coordinator.review_data_deletion((ref,)))
    assert ref.path.exists()


@pytest.mark.asyncio
async def test_service_shutdown_drains_root_users_before_final_clean(native_console):
    case = native_console
    installed = await case.install()
    review = await case.service._call(
        lambda: case.service._coordinator.review_data_creation(
            installed.installation_id
        )
    )
    ref = await case.service._call(
        lambda: case.service._coordinator.create_data(review, review.operation_id)
    )
    token = await case.service._call(
        lambda: case.service._coordinator.owner.reserve_launch(
            "idle",
            installed.installation_id,
            "workspace-a",
            installed.inspection.effective_digest,
            roots=(ref,),
        )
    )
    with pytest.raises(PermissionError):
        await case.service.aclose()
    assert case.service._thread.is_alive()
    await case.service.settle_data_user(token, confirmed=True)
    await case.service.aclose()
    marker = case.service._marker_factory(None)
    assert marker.load_runtime_checkpoint()["phase"] == "clean"


def test_coherent_sqlite_rollback_missing_owner_and_joins_still_blocks_live_child(
    data_cleanup_case, tmp_path
):
    import sqlite3

    case = data_cleanup_case
    stack = case.stack
    ref = create_root(case)
    backup = tmp_path / "older.sqlite"

    def snapshot():
        with sqlite3.connect(backup) as destination:
            stack.registry._connection.backup(destination)

    stack.call(snapshot)
    token = reserve(case, ref)
    code = """
import os, socket, sys
from pathlib import Path
socket.socket.connect = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network refused"))
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve().parents[1] == Path(sys.argv[2])
assert os.environ["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
print("ready", flush=True)
for line in sys.stdin:
 if line.strip() == "stop": break
 (Path(sys.argv[1]) / "survived").write_text("still alive")
 print("wrote", flush=True)
"""
    child = subprocess.Popen(
        child_argv(code) + [str(ref.path), str(Path.cwd())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        stack.call(
            lambda: stack.owner.publish_process(
                token, {"pid": child.pid, "start_identity": "exact controlled child"}
            )
        )

        def restore():
            with sqlite3.connect(backup) as source:
                source.backup(stack.registry._connection)

        stack.call(restore)
        assert (
            stack.call(
                lambda: stack.registry._connection.execute(
                    "SELECT count(*) FROM root_users"
                ).fetchone()[0]
            )
            == 0
        )
        # Same-session live grants survive the coherent SQL loss as well.
        assert stack.call(lambda: stack.coordinator.root_usage.blockers((ref,)))
        reopen(stack, clean=False)
        with pytest.raises(PermissionError):
            stack.call(lambda: stack.coordinator.review_data_deletion((ref,)))
        child.stdin.write("write\n")
        child.stdin.flush()
        assert child.stdout.readline().strip() == "wrote"
        assert (ref.path / "survived").read_text() == "still alive"
        with pytest.raises(PermissionError):
            stack.call(
                lambda: stack.coordinator.review_data_reconciliation(
                    (ref,), confirm_quiescence=lambda refs: child.poll() is not None
                )
            )
        child.stdin.write("stop\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        reconciliation = stack.call(
            lambda: stack.coordinator.review_data_reconciliation(
                (ref,), confirm_quiescence=lambda refs: child.poll() == 0
            )
        )
        updated = stack.call(
            lambda: stack.coordinator.reconcile_data(
                reconciliation, reconciliation.operation_id
            )
        )
        deletion = stack.call(lambda: stack.coordinator.review_data_deletion(updated))
        stack.call(
            lambda: stack.coordinator.delete_data(updated, deletion.operation_id)
        )
        assert not ref.path.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()


def test_failed_dirty_readback_refuses_before_any_grant(data_cleanup_case, monkeypatch):
    case = data_cleanup_case
    ref = create_root(case)
    reopen(case.stack, clean=True)
    marker = case.stack.authority.marker_store
    old = marker.load_runtime_checkpoint()
    monkeypatch.setattr(marker, "load_runtime_checkpoint", lambda: old)
    with pytest.raises(ValueError, match="publication mismatch"):
        reserve(case, ref)
    assert (
        case.stack.call(
            lambda: case.stack.registry._connection.execute(
                "SELECT count(*) FROM root_users WHERE state!='released'"
            ).fetchone()[0]
        )
        == 0
    )
    assert case.stack.call(
        lambda: case.stack.coordinator.root_usage.is_fenced(ref.root_id)
    )


def test_rebind_after_boot_change_requires_reviewed_quiescence(
    data_cleanup_case, monkeypatch
):
    import tldw_chatbook.Plugins.data_cleanup as module

    case = data_cleanup_case
    ref = create_root(case)
    monkeypatch.setattr(
        module, "boot_identity", lambda: "00000000-0000-0000-0000-000000000001"
    )
    with pytest.raises(PermissionError, match="boot_identity_changed"):
        reserve(case, ref)
    with pytest.raises(PermissionError):
        case.stack.call(
            lambda: case.stack.coordinator.review_data_reconciliation(
                (ref,), confirm_quiescence=lambda refs: False
            )
        )
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_reconciliation(
            (ref,), confirm_quiescence=lambda refs: True
        )
    )
    fresh = case.stack.call(
        lambda: case.stack.coordinator.reconcile_data(review, review.operation_id)
    )[0]
    assert fresh.generation == ref.generation + 1
    token = reserve(case, fresh)
    case.stack.call(lambda: case.stack.owner.settle_process(token, True))


def test_dirty_orphan_and_reset_never_mint_clean_over_retained_data(data_cleanup_case):
    case = data_cleanup_case
    ref = create_root(case)
    case.stack.call(lambda: case.stack.coordinator.uninstall(case.installation))
    reopen(case.stack, clean=False)
    assert case.stack.call(
        lambda: case.stack.coordinator.root_usage.is_fenced(ref.root_id)
    )
    case.stack.call(
        lambda: case.stack.coordinator.reset(operation_id="explicit-root-reset")
    )
    with pytest.raises((PermissionError, ValueError)):
        case.stack.call(lambda: case.stack.coordinator.bootstrap("new passphrase"))
    assert ref.path.exists()
    assert (
        case.stack.authority.marker_store.load_runtime_checkpoint()["phase"] == "dirty"
    )


@pytest.mark.asyncio
async def test_failed_clean_write_keeps_service_owned_and_root_admission_closed(
    native_console, monkeypatch
):
    case = native_console
    installed = await case.install()
    creation = await case.service.review_data_creation(installed.installation_id)
    ref = await case.service.create_data(creation, creation.operation_id)
    marker = case.service._coordinator.authority.marker_store
    real = marker.save_runtime_checkpoint

    def refuse_clean(value):
        if value["phase"] == "clean":
            raise OSError("protected backend temporarily unavailable")
        real(value)

    with monkeypatch.context() as patch:
        patch.setattr(marker, "save_runtime_checkpoint", refuse_clean)
        with pytest.raises(OSError):
            await case.service.aclose()
        assert case.service._thread.is_alive()
        assert marker.load_runtime_checkpoint()["phase"] == "dirty"
        with pytest.raises(PermissionError):
            await case.service.reserve_data_user(
                "late",
                installed.installation_id,
                "workspace-a",
                installed.inspection.effective_digest,
                roots=(ref,),
            )
    await case.service.aclose()
    assert not case.service._thread.is_alive()


@pytest.mark.parametrize("payload", [None, b"not-a-uuid", b"0" * 65])
def test_unavailable_or_malformed_native_boot_value_refuses(monkeypatch, payload):
    import ctypes
    from types import SimpleNamespace

    import tldw_chatbook.Plugins.data_cleanup as module

    class Probe:
        def __call__(self, name, buffer, length, new, size):
            if payload is None:
                return -1
            if len(payload) <= len(buffer):
                buffer.value = payload
            ctypes.cast(length, ctypes.POINTER(ctypes.c_size_t))[0] = len(payload)
            return 0

    monkeypatch.setattr(
        module.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(sysctlbyname=Probe()),
    )
    with pytest.raises(PermissionError, match="boot_identity_unavailable"):
        module.boot_identity()


def test_restart_preserves_pending_attachment_intent(data_cleanup_case):
    case = data_cleanup_case
    ref = create_root(case)
    (ref.path / "saved").write_text("keep")
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_attachment((ref,), None)
    )

    def fail(phase):
        if phase == "data_fenced":
            raise OSError("controlled attachment cut")

    case.stack.call(lambda: setattr(case.stack.coordinator, "progress", fail))
    with pytest.raises(OSError):
        case.stack.call(
            lambda: case.stack.coordinator.delete_data((ref,), review.operation_id)
        )
    reopen(case.stack, clean=False)
    proof = case.stack.call(
        lambda: case.stack.coordinator.review_data_reconciliation(
            (ref,), confirm_quiescence=lambda roots: True
        )
    )
    case.stack.call(
        lambda: case.stack.coordinator.reconcile_data(proof, proof.operation_id)
    )
    resumed = case.stack.call(
        lambda: case.stack.coordinator.review_data_cleanup_resume((ref,))
    )
    assert resumed.action == "attach" and resumed.attachment is None
    result = case.stack.call(
        lambda: case.stack.coordinator.delete_data((ref,), resumed.operation_id)
    )
    assert result.phase == "attached" and (ref.path / "saved").read_text() == "keep"
    row = case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][0]
    assert row["generation"] == ref.generation + 1
    assert row["custody"]["attached_installation_id"] is None


@pytest.mark.parametrize("replacement", [None, "leaf", "ancestor", "missing_ancestor"])
def test_completed_absence_dirty_restart_reconciles_only_exact_ancestry(
    data_cleanup_case, replacement
):
    from tldw_chatbook.Plugins.data_cleanup import root_ref

    case = data_cleanup_case
    ref = create_root(case)
    deletion = case.stack.call(
        lambda: case.stack.coordinator.review_data_deletion((ref,))
    )
    assert (
        case.stack.call(
            lambda: case.stack.coordinator.delete_data((ref,), deletion.operation_id)
        ).phase
        == "complete"
    )
    original = case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][
        0
    ]
    ref = root_ref(original)
    assert original["custody"]["state"] == "cleaned_absent" and not ref.path.exists()
    sentinel = None
    if replacement == "leaf":
        ref.path.mkdir()
        sentinel = ref.path / "external"
    elif replacement in {"ancestor", "missing_ancestor"}:
        ref.path.parent.rename(ref.path.parent.with_name("retained-data-anchor"))
        if replacement == "ancestor":
            ref.path.parent.mkdir()
            sentinel = ref.path.parent / "external"
    if sentinel is not None:
        sentinel.write_text("preserve")
    reopen(case.stack, clean=False)
    if replacement is not None:
        with pytest.raises((PermissionError, FileNotFoundError)):
            case.stack.call(
                lambda: case.stack.coordinator.review_data_reconciliation(
                    (ref,), confirm_quiescence=lambda roots: True
                )
            )
        with pytest.raises(PermissionError):
            case.stack.call(case.stack.coordinator.root_usage.shutdown)
        if sentinel is not None:
            assert sentinel.read_text() == "preserve"
        return
    review = case.stack.call(
        lambda: case.stack.coordinator.review_data_reconciliation(
            (ref,), confirm_quiescence=lambda roots: True
        )
    )
    reconciled = case.stack.call(
        lambda: case.stack.coordinator.reconcile_data(review, review.operation_id)
    )
    assert reconciled[0].generation == ref.generation + 1
    after = case.stack.call(case.stack.coordinator.published_snapshot)["data_roots"][0]
    assert (
        after["custody"]["state"] == "cleaned_absent"
        and after["path"] == original["path"]
    )
    assert (
        after["installation_id"] == original["installation_id"]
        and not ref.path.exists()
    )
    case.stack.call(case.stack.coordinator.root_usage.shutdown)
    assert (
        case.stack.authority.marker_store.load_runtime_checkpoint()["phase"] == "clean"
    )
    reopen(case.stack, clean=False)
    assert not case.stack.coordinator.root_usage.recovery
    case.stack.call(case.stack.coordinator.root_usage.shutdown)


def test_unexpected_missing_present_root_is_not_completed_absence(data_cleanup_case):
    case = data_cleanup_case
    ref = create_root(case)
    ref.path.rmdir()
    reopen(case.stack, clean=False)
    with pytest.raises(PermissionError, match="root_missing_without_deleting_intent"):
        case.stack.call(
            lambda: case.stack.coordinator.review_data_reconciliation(
                (ref,), confirm_quiescence=lambda roots: True
            )
        )


@pytest.mark.asyncio
async def test_final_clean_seals_queued_and_new_service_mutations(
    native_console, monkeypatch
):
    import asyncio
    import threading

    from tldw_chatbook.Plugins.service import PluginService

    case = native_console
    installed = await case.install()
    creation = await case.service.review_data_creation(installed.installation_id)
    ref = await case.service.create_data(creation, creation.operation_id)
    trust = await case.service.review_trust(installed.installation_id)
    marker = case.service._coordinator.authority.marker_store
    real = marker.save_runtime_checkpoint
    entered, release = threading.Event(), threading.Event()

    def hold_clean(value):
        real(value)
        if value["phase"] == "clean":
            entered.set()
            assert release.wait(5)

    with monkeypatch.context() as patch:
        patch.setattr(marker, "save_runtime_checkpoint", hold_clean)
        closing = asyncio.create_task(case.service.aclose())
        assert await asyncio.to_thread(entered.wait, 5)
        mutation = asyncio.create_task(case.service.commit(trust, trust.operation_id))
        await asyncio.sleep(0.02)
        release.set()
        results = await asyncio.gather(closing, mutation, return_exceptions=True)
    assert results[0] is None
    assert isinstance(results[1], PermissionError), results[1]
    assert marker.load_runtime_checkpoint()["marker"] == marker.load_marker()
    reopened = PluginService(
        case.service.profile_root,
        workspace_lookup=case.service.workspace_lookup,
        marker_store_factory=case.service._marker_factory,
        accept_reduced_protection=True,
    )
    try:
        await reopened.unlock("test passphrase")
        token = await reopened.reserve_data_user(
            "after-clean",
            installed.installation_id,
            "workspace-a",
            installed.inspection.effective_digest,
            roots=(ref,),
        )
        await reopened.settle_data_user(token, confirmed=True)
    finally:
        await reopened.aclose()


@pytest.mark.asyncio
async def test_refused_shutdown_seals_mutations_but_accepts_terminal_settlement(
    native_console,
):
    case = native_console
    installed = await case.install()
    creation = await case.service.review_data_creation(installed.installation_id)
    ref = await case.service.create_data(creation, creation.operation_id)
    trust = await case.service.review_trust(installed.installation_id)
    token = await case.service.reserve_data_user(
        "pending-close",
        installed.installation_id,
        "workspace-a",
        installed.inspection.effective_digest,
        roots=(ref,),
    )
    with pytest.raises(PermissionError):
        await case.service.aclose()
    try:
        with pytest.raises(PermissionError, match="plugin_service_closing"):
            await case.service.commit(trust, trust.operation_id)
    finally:
        await case.service.settle_data_user(token, confirmed=True)
        await case.service.aclose()


@pytest.mark.asyncio
async def test_shutdown_waits_for_already_running_authority_mutation(
    native_console, monkeypatch
):
    import asyncio
    import threading

    case = native_console
    installed = await case.install()
    creation = await case.service.review_data_creation(installed.installation_id)
    await case.service.create_data(creation, creation.operation_id)
    trust = await case.service.review_trust(installed.installation_id)
    real = case.service._coordinator.commit
    entered, release = threading.Event(), threading.Event()

    async def hold(review, operation_id):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await real(review, operation_id)

    with monkeypatch.context() as patch:
        patch.setattr(case.service._coordinator, "commit", hold)
        mutation = asyncio.create_task(case.service.commit(trust, trust.operation_id))
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            with pytest.raises(PermissionError, match="plugin_calls_not_drained"):
                await case.service.aclose()
            assert (
                case.service._coordinator.authority.marker_store.load_runtime_checkpoint()[
                    "phase"
                ]
                == "dirty"
            )
        finally:
            release.set()
        assert (await asyncio.wait_for(mutation, 5)).committed
    await case.service.aclose()
    marker = case.service._marker_factory(None)
    assert marker.load_runtime_checkpoint()["marker"] == marker.load_marker()


@pytest.mark.asyncio
async def test_shutdown_rejects_previously_queued_service_mutation(native_console):
    import asyncio
    import threading

    case = native_console
    installed = await case.install()
    creation = await case.service.review_data_creation(installed.installation_id)
    await case.service.create_data(creation, creation.operation_id)
    trust = await case.service.review_trust(installed.installation_id)
    entered, release = threading.Event(), threading.Event()

    def block_worker():
        entered.set()
        assert release.wait(5)

    blocker = asyncio.create_task(case.service._call(block_worker))
    assert await asyncio.to_thread(entered.wait, 5)
    mutation = asyncio.create_task(case.service.commit(trust, trust.operation_id))
    await asyncio.sleep(0.02)
    closing = asyncio.create_task(case.service.aclose())
    await asyncio.sleep(0.02)
    release.set()
    assert await blocker is None
    with pytest.raises(PermissionError, match="plugin_service_closing"):
        await mutation
    await closing
    marker = case.service._marker_factory(None)
    assert marker.load_runtime_checkpoint()["marker"] == marker.load_marker()
