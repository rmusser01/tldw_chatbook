"""Actual preinstallation custody and bounded materialization controls."""

from pathlib import Path

import pytest


def acquire(stack, source, operation="candidate"):
    from tldw_chatbook.Plugins.acquisition import acquire_snapshot
    from tldw_chatbook.Plugins.git_source import select_local_source

    owner = stack.owner
    return stack.call(
        lambda: acquire_snapshot(
            select_local_source(source),
            operation,
            owner.acquisition_destination(operation),
            owner=owner,
            authority=stack.authority,
        )
    )


def test_local_snapshot_uses_owned_staging_without_installation(
    plugin_stack, native_package
):
    from tldw_chatbook.Plugins.acquisition import acquire_snapshot
    from tldw_chatbook.Plugins.git_source import select_local_source

    source = native_package()
    owner = plugin_stack.owner
    destination = owner.acquisition_destination("local-success")
    result = plugin_stack.call(
        lambda: acquire_snapshot(
            select_local_source(source),
            "local-success",
            destination,
            owner=owner,
            authority=plugin_stack.authority,
        )
    )
    assert Path(result.materialized_identity) == destination
    assert (destination / "plugin.json").read_bytes() == (
        source / "plugin.json"
    ).read_bytes()
    assert result.inspection.content_digest
    assert not plugin_stack.call(lambda: owner.list_processes(limit=50, offset=0))
    with pytest.raises(ValueError):
        result.commit = "changed"


def test_materializer_checkpoint_stops_before_creating_output(native_package, tmp_path):
    from tldw_chatbook.Plugins.package_files import materialize_package

    calls = []

    def checkpoint():
        calls.append(True)
        raise InterruptedError("controlled stop")

    destination = tmp_path / "output"
    with pytest.raises((InterruptedError, ValueError)):
        materialize_package(native_package(), destination, checkpoint=checkpoint)
    assert calls
    assert not destination.exists()


def test_exact_production_limits():
    from tldw_chatbook.Plugins import acquisition, package_files, retention

    assert acquisition.MAX_ACQUISITIONS == 2
    assert acquisition.ACQUISITION_SECONDS == 120
    assert acquisition.MAX_STAGING_BYTES == 500 * 1024**2
    assert package_files.MAX_PACKAGE_BYTES == 100 * 1024**2
    assert package_files.MAX_FILE_BYTES == 10 * 1024**2
    assert package_files.MAX_FILES == 10_000
    assert package_files.MAX_PATH_DEPTH == 32
    assert retention.MANAGED_BYTES == 2 * 1024**3
    assert retention.FREE_RESERVE_BYTES == 100 * 1024**2


def test_destination_and_operation_cannot_be_reused(
    plugin_stack, native_package, tmp_path
):
    from tldw_chatbook.Plugins.acquisition import acquire_snapshot
    from tldw_chatbook.Plugins.git_source import select_local_source

    source = native_package()
    sentinel = tmp_path / "untouched"
    sentinel.mkdir()
    (sentinel / "valuable").write_text("original")
    with pytest.raises(ValueError, match="destination_invalid"):
        plugin_stack.call(
            lambda: acquire_snapshot(
                select_local_source(source),
                "one",
                sentinel,
                owner=plugin_stack.owner,
                authority=plugin_stack.authority,
            )
        )
    result = acquire(plugin_stack, source, "one")
    with pytest.raises(ValueError, match="operation_reused"):
        acquire(plugin_stack, source, "one")
    assert (sentinel / "valuable").read_text() == "original"
    assert Path(result.materialized_identity).is_dir()


@pytest.mark.parametrize(
    "bound", ["MAX_FILE_BYTES", "MAX_PACKAGE_BYTES", "MAX_FILES", "MAX_PATH_DEPTH"]
)
def test_real_local_materialization_enforces_each_bound(
    plugin_stack, native_package, monkeypatch, bound
):
    from tldw_chatbook.Plugins import package_files

    source = native_package()
    control = acquire(plugin_stack, source, "control")
    assert control.inspection.content_digest
    monkeypatch.setattr(package_files, bound, 1)
    with pytest.raises(ValueError, match="limit"):
        acquire(plugin_stack, source, "bounded")
    assert Path(control.materialized_identity).exists()


def test_full_disk_does_not_change_installed_revision(
    plugin_stack, native_package, monkeypatch
):
    from collections import namedtuple

    from Tests.Plugins.test_coordinator import reviewed
    from tldw_chatbook.Plugins import retention

    review = reviewed(plugin_stack, native_package())
    plugin_stack.call(
        lambda: plugin_stack.coordinator.commit(review, review.operation_id)
    )
    before = plugin_stack.call(plugin_stack.authority.verify_current)
    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda _: usage(10**12, 10**12, retention.FREE_RESERVE_BYTES - 1),
    )
    with pytest.raises(OSError, match="reserve"):
        acquire(plugin_stack, native_package())
    assert plugin_stack.call(plugin_stack.authority.verify_current) == before


def test_terminal_custody_cleanup_requires_release_age_and_exact_target(
    plugin_stack, native_package
):
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins.acquisition import cleanup_acquisitions, release_snapshot

    result = acquire(plugin_stack, native_package())
    future = datetime.now(UTC) + timedelta(hours=25)
    cleanup = lambda now: plugin_stack.call(
        lambda: cleanup_acquisitions(
            plugin_stack.owner, plugin_stack.authority, now=now
        )
    )
    assert cleanup(future) == ()
    plugin_stack.call(
        lambda: release_snapshot(plugin_stack.owner, plugin_stack.authority, result)
    )
    assert cleanup(datetime.now(UTC)) == ()
    assert cleanup(future) == ("candidate",)
    assert not Path(result.materialized_identity).parent.exists()
    assert cleanup(future) == ()


@pytest.mark.parametrize("damage", ["tamper", "lost", "replace"])
def test_restart_never_cleans_uncertain_custody(plugin_stack, native_package, damage):
    import json
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins.acquisition import (
        _record_path,
        cleanup_acquisitions,
        release_snapshot,
    )
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    result = acquire(plugin_stack, native_package())
    plugin_stack.call(
        lambda: release_snapshot(plugin_stack.owner, plugin_stack.authority, result)
    )
    path = _record_path(plugin_stack.owner, result.operation_id)
    leaf = Path(result.materialized_identity).parent
    if damage == "tamper":
        payload = json.loads(path.read_text())
        payload["body"]["nonce"] = "tampered"
        path.write_text(json.dumps(payload))
    elif damage == "lost":
        path.unlink()
    else:
        leaf.rename(leaf.with_name("original"))
        leaf.mkdir()

    def restart():
        plugin_stack.owner.close()
        new = PluginRuntimeOwner(plugin_stack.owner.root)
        assert new.try_acquire()
        try:
            try:
                removed = cleanup_acquisitions(
                    new,
                    plugin_stack.authority,
                    now=datetime.now(UTC) + timedelta(hours=25),
                )
            except (ValueError, PermissionError):
                removed = ()
            assert removed == ()
        finally:
            new.close()
            assert plugin_stack.owner.try_acquire()

    plugin_stack.call(restart)
    assert leaf.exists()


def test_unknown_staging_and_custody_overflow_refuse_without_deletion(
    plugin_stack, native_package, monkeypatch
):
    from tldw_chatbook.Plugins import acquisition

    anchor = plugin_stack.owner.root / "staging"
    anchor.mkdir()
    for name in ("unknown-a", "unknown-b"):
        (anchor / name).mkdir()
    with pytest.raises(ValueError, match="concurrency_limit"):
        acquire(plugin_stack, native_package())
    assert {p.name for p in anchor.iterdir()} == {"unknown-a", "unknown-b"}
    for child in anchor.iterdir():
        child.rmdir()
    assert acquire(plugin_stack, native_package()).inspection.content_digest
    monkeypatch.setattr(acquisition, "MAX_CUSTODY_RECORDS", 1)
    with pytest.raises(ValueError, match="custody_limit"):
        acquire(plugin_stack, native_package(), "overflow")


@pytest.mark.asyncio
async def test_public_service_retains_materializer_after_waiter_cancel(
    tmp_path, native_package, monkeypatch
):
    import asyncio
    import threading

    from tldw_chatbook.Plugins import package_files
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.service import PluginService, PluginUnavailable

    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    await service.bootstrap("test passphrase")
    original = package_files.materialize_package
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(package_files, "materialize_package", blocked)
    destination = await service.acquisition_destination("cancelled")
    task = asyncio.create_task(
        service.acquire_snapshot(
            select_local_source(native_package()), "cancelled", destination
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(PluginUnavailable, match="not_drained"):
            await service.aclose()
        release.set()
        for _ in range(100):
            if not service._coordinator.owner.acquisition_tasks:
                break
            await asyncio.sleep(0.01)
        assert not service._coordinator.owner.acquisition_tasks
        assert not destination.exists()
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_error", [False, True])
async def test_actual_git_tree_cancel_retains_slot_until_terminal(
    plugin_stack, native_package, tmp_path, monkeypatch, probe_error
):
    import asyncio
    import json
    import os

    from Tests.Plugins.acquisition_process_support import repository, stubborn_git
    from tldw_chatbook.Plugins import acquisition
    from tldw_chatbook.Plugins.git_source import select_local_source

    source = repository(native_package(), tmp_path / "source")
    marker = tmp_path / "tree.json"
    executable = stubborn_git(tmp_path / "fake-git", marker)
    original = acquisition.shutil.which
    monkeypatch.setattr(
        acquisition.shutil,
        "which",
        lambda name: str(executable) if name == "git" else original(name),
    )
    owner = plugin_stack.owner
    proof = asyncio.Event()
    original_wait = acquisition.ProcessTreeController.wait

    async def withheld_proof(controller, tree, *, timeout):
        terminal = await original_wait(controller, tree, timeout=timeout)
        if probe_error and not proof.is_set():
            raise OSError("controlled native probe unavailable")
        return terminal and proof.is_set()

    monkeypatch.setattr(acquisition.ProcessTreeController, "wait", withheld_proof)
    future = asyncio.run_coroutine_threadsafe(
        acquisition.acquire_snapshot(
            select_local_source(source),
            "cancel-tree",
            owner.acquisition_destination("cancel-tree"),
            owner=owner,
            authority=plugin_stack.authority,
        ),
        plugin_stack.loop,
    )
    try:
        for _ in range(500):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        evidence = json.loads(marker.read_text())
        assert (
            Path(evidence["package"]).resolve()
            == Path(__file__).resolve().parents[2] / "tldw_chatbook/__init__.py"
        )
        future.cancel()
        await asyncio.sleep(0.4)
        assert owner.acquisition_tasks
        records = plugin_stack.call(
            lambda: acquisition._inventory(owner, plugin_stack.authority)
        )
        assert records[0]["phase"] == "pending"
        proof.set()
        for _ in range(500):
            if not owner.acquisition_tasks:
                break
            await asyncio.sleep(0.01)
        assert not owner.acquisition_tasks
        with pytest.raises(ProcessLookupError):
            os.killpg(evidence["pid"], 0)
        records = plugin_stack.call(
            lambda: acquisition._inventory(owner, plugin_stack.authority)
        )
        assert records[0]["phase"] == "terminal"
    finally:
        proof.set()
        future.cancel()


def test_native_restart_cleans_only_released_terminal_receipt(
    plugin_stack, native_package
):
    import subprocess

    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Plugins.acquisition import release_snapshot

    result = acquire(plugin_stack, native_package())
    plugin_stack.call(
        lambda: release_snapshot(plugin_stack.owner, plugin_stack.authority, result)
    )
    root = plugin_stack.owner.root
    parent = root.parent
    code = (
        "from pathlib import Path;from datetime import UTC,datetime,timedelta;"
        "from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner;"
        "from tldw_chatbook.Plugins.authority_store import PluginAuthorityStore,FilePluginMarkerStore;"
        "from tldw_chatbook.Plugins.acquisition import cleanup_acquisitions;"
        f"owner=PluginRuntimeOwner(Path({str(root)!r}));assert owner.try_acquire();"
        f"authority=PluginAuthorityStore(Path({str(parent / 'trust/plugins')!r}),FilePluginMarkerStore(Path({str(parent / 'marker')!r})),accept_reduced_protection=True);"
        "authority.unlock('test passphrase');"
        "assert cleanup_acquisitions(owner,authority,now=datetime.now(UTC)+timedelta(hours=25))==('candidate',);"
        "owner.close();print('RESTART_TERMINAL_CLEANED')"
    )
    plugin_stack.call(plugin_stack.owner.close)
    try:
        completed = subprocess.run(
            child_argv(code), capture_output=True, text=True, timeout=20, check=False
        )
        assert completed.returncode == 0, completed.stderr
        assert "RESTART_TERMINAL_CLEANED" in completed.stdout
    finally:
        plugin_stack.call(plugin_stack.owner.try_acquire)
    assert not Path(result.materialized_identity).parent.exists()


@pytest.mark.asyncio
async def test_two_actual_producers_refuse_third_and_release_slots(
    plugin_stack, native_package, monkeypatch
):
    import asyncio
    import threading

    from tldw_chatbook.Plugins import acquisition, package_files
    from tldw_chatbook.Plugins.git_source import select_local_source

    original = package_files.materialize_package
    entered = threading.Barrier(3)
    release = threading.Event()

    def blocked(*args, **kwargs):
        entered.wait(timeout=5)
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(package_files, "materialize_package", blocked)
    owner = plugin_stack.owner
    source = select_local_source(native_package())
    futures = [
        asyncio.run_coroutine_threadsafe(
            acquisition.acquire_snapshot(
                source,
                str(index),
                owner.acquisition_destination(str(index)),
                owner=owner,
                authority=plugin_stack.authority,
            ),
            plugin_stack.loop,
        )
        for index in range(2)
    ]
    try:
        await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(ValueError, match="concurrency_limit"):
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    acquisition.acquire_snapshot(
                        source,
                        "third",
                        owner.acquisition_destination("third"),
                        owner=owner,
                        authority=plugin_stack.authority,
                    ),
                    plugin_stack.loop,
                )
            )
        assert len(owner.acquisition_tasks) == 2
        release.set()
        results = await asyncio.gather(*(asyncio.wrap_future(f) for f in futures))
        assert all(Path(result.materialized_identity).exists() for result in results)
    finally:
        release.set()


def test_deadline_and_staging_quota_stop_real_local_git_copy(
    plugin_stack, native_package, tmp_path, monkeypatch
):
    from Tests.Plugins.acquisition_process_support import repository
    from tldw_chatbook.Plugins import acquisition

    source = repository(native_package(), tmp_path / "source")
    with monkeypatch.context() as patch:
        patch.setattr(acquisition, "ACQUISITION_SECONDS", 0)
        with pytest.raises(ValueError, match="deadline_limit"):
            acquire(plugin_stack, source, "deadline")
    with monkeypatch.context() as patch:
        patch.setattr(acquisition, "MAX_STAGING_BYTES", 1)
        with pytest.raises(ValueError, match="staging_limit"):
            acquire(plugin_stack, source, "staging")
    assert acquire(plugin_stack, source, "control").commit


def test_nonterminal_restart_never_infers_quiescence(plugin_stack, native_package):
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins import acquisition
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = plugin_stack.owner
    for operation in ("lost-a", "lost-b"):
        plugin_stack.call(
            lambda operation=operation: acquisition._reserve(
                owner,
                plugin_stack.authority,
                operation,
                owner.acquisition_destination(operation),
            )
        )

    def restart():
        owner.close()
        new = PluginRuntimeOwner(owner.root)
        assert new.try_acquire()
        try:
            assert (
                acquisition.cleanup_acquisitions(
                    new,
                    plugin_stack.authority,
                    now=datetime.now(UTC) + timedelta(hours=25),
                )
                == ()
            )
            with pytest.raises(ValueError, match="concurrency_limit"):
                acquisition._reserve(
                    new,
                    plugin_stack.authority,
                    "new",
                    new.acquisition_destination("new"),
                )
        finally:
            new.close()
            assert owner.try_acquire()

    plugin_stack.call(restart)
    assert all(
        owner.acquisition_destination(operation).parent.exists()
        for operation in ("lost-a", "lost-b")
    )


@pytest.mark.asyncio
async def test_uninitialized_and_locked_service_leave_no_acquisition_custody(
    tmp_path, native_package
):
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.service import PluginService

    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    source = select_local_source(native_package())
    destination = await service.acquisition_destination("after-unlock")
    try:
        for _ in range(2):
            with pytest.raises((ValueError, PermissionError)):
                await service.acquire_snapshot(source, "after-unlock", destination)
            assert not destination.parent.exists()
        await service.bootstrap("test passphrase")
        await service._call(service._coordinator.authority.lock)
        with pytest.raises((ValueError, PermissionError)):
            await service.acquire_snapshot(source, "after-unlock", destination)
        assert not destination.parent.exists()
        await service.unlock("test passphrase")
        result = await service.acquire_snapshot(source, "after-unlock", destination)
        assert Path(result.materialized_identity).exists()
    finally:
        await service.aclose()


@pytest.mark.parametrize("mode", ["quota", "deadline", "output"])
def test_live_process_limits_terminate_actual_producer(
    plugin_stack, native_package, tmp_path, monkeypatch, mode
):
    import shlex
    import time

    from Tests.hooks_v2_process_support import child_argv
    from Tests.Plugins.acquisition_process_support import repository
    from tldw_chatbook.Plugins import acquisition

    source = repository(native_package(), tmp_path / "source")
    executable = tmp_path / "bounded-git"
    if mode == "quota":
        code = "import time;from pathlib import Path;f=Path('large-staging-object').open('wb');f.truncate(500*1024*1024+1);f.close();time.sleep(60)"
        expected = "staging_limit"
    elif mode == "output":
        code = "import os,time;os.write(1,b'x'*100000);time.sleep(60)"
        expected = "output_limit"
    else:
        code = "import time;time.sleep(60)"
        expected = "deadline_limit"
        monkeypatch.setattr(acquisition, "ACQUISITION_SECONDS", 2)
    executable.write_text("#!/bin/sh\nexec " + shlex.join(child_argv(code)) + ' "$@"\n')
    executable.chmod(0o700)
    original = acquisition.shutil.which
    monkeypatch.setattr(
        acquisition.shutil,
        "which",
        lambda name: str(executable) if name == "git" else original(name),
    )
    started = time.monotonic()
    with pytest.raises(ValueError, match=expected):
        acquire(plugin_stack, source)
    assert time.monotonic() - started < 6
    records = plugin_stack.call(
        lambda: acquisition._inventory(plugin_stack.owner, plugin_stack.authority)
    )
    assert records[0]["phase"] == "terminal"
    assert not plugin_stack.owner.acquisition_tasks


@pytest.mark.parametrize("retire_by", ["capacity", "age"])
def test_cleaned_receipt_retirement_preserves_new_incarnation(
    plugin_stack, native_package, monkeypatch, retire_by
):
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins import acquisition

    owner, authority = plugin_stack.owner, plugin_stack.authority
    old = acquire(plugin_stack, native_package(), "label")
    path = acquisition._record_path(owner, "label")
    terminal_receipt = path.read_bytes()
    plugin_stack.call(lambda: acquisition.release_snapshot(owner, authority, old))
    assert plugin_stack.call(
        lambda: acquisition.cleanup_acquisitions(
            owner, authority, now=datetime.now(UTC) + timedelta(hours=25)
        )
    ) == ("label",)
    old_receipt = path.read_bytes()
    if retire_by == "capacity":
        monkeypatch.setattr(acquisition, "MAX_CUSTODY_RECORDS", 1)
    else:
        assert (
            plugin_stack.call(
                lambda: acquisition.cleanup_acquisitions(
                    owner, authority, now=datetime.now(UTC) + timedelta(days=31)
                )
            )
            == ()
        )
        assert not path.exists()
    fresh = acquire(plugin_stack, native_package(), "label")
    assert fresh.acquisition_nonce != old.acquisition_nonce
    with pytest.raises(ValueError, match="release_mismatch"):
        plugin_stack.call(lambda: acquisition.release_snapshot(owner, authority, old))
    new_receipt = path.read_bytes()
    path.write_bytes(terminal_receipt)
    with pytest.raises(ValueError, match="staging_changed|custody_limit"):
        acquire(plugin_stack, native_package(), "must-not-admit")
    with pytest.raises(ValueError, match="staging_changed"):
        plugin_stack.call(
            lambda: acquisition._qualify(
                owner, acquisition._inventory(owner, authority)[0]
            )
        )
    assert Path(fresh.materialized_identity).exists()
    path.write_bytes(old_receipt)
    assert (
        plugin_stack.call(
            lambda: acquisition.cleanup_acquisitions(
                owner, authority, now=datetime.now(UTC) + timedelta(days=31)
            )
        )
        == ()
    )
    assert Path(fresh.materialized_identity).exists()
    assert path.exists()  # Replayed cleaned history cannot retire over an extant leaf.
    path.write_bytes(new_receipt)
    plugin_stack.call(lambda: acquisition.release_snapshot(owner, authority, fresh))


def test_receipt_retirement_protects_changed_or_extant_staging(
    plugin_stack, native_package, monkeypatch
):
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins import acquisition

    owner, authority = plugin_stack.owner, plugin_stack.authority
    old = acquire(plugin_stack, native_package(), "protected")
    plugin_stack.call(lambda: acquisition.release_snapshot(owner, authority, old))
    plugin_stack.call(
        lambda: acquisition.cleanup_acquisitions(
            owner, authority, now=datetime.now(UTC) + timedelta(hours=25)
        )
    )
    leaf = Path(old.materialized_identity).parent
    leaf.mkdir()
    (leaf / "unknown").write_text("keep")
    monkeypatch.setattr(acquisition, "MAX_CUSTODY_RECORDS", 1)
    with pytest.raises(ValueError, match="custody_limit"):
        acquire(plugin_stack, native_package(), "new-label")
    assert (leaf / "unknown").read_text() == "keep"
    assert acquisition._record_path(owner, "protected").exists()


@pytest.mark.parametrize("bound", ["staging", "managed", "reserve"])
@pytest.mark.parametrize("cross_limit", [False, True])
def test_final_materialization_quota_acceptance(
    plugin_stack, native_package, monkeypatch, bound, cross_limit
):
    from collections import namedtuple
    from datetime import UTC, datetime, timedelta

    from Tests.Plugins.test_coordinator import reviewed
    from tldw_chatbook.Plugins import acquisition, retention

    source = native_package()
    review = reviewed(plugin_stack, source)
    plugin_stack.call(
        lambda: plugin_stack.coordinator.commit(review, review.operation_id)
    )
    before = plugin_stack.call(plugin_stack.authority.verify_current)
    owner, authority = plugin_stack.owner, plugin_stack.authority
    size = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    initial = retention.managed_usage(owner.root)
    allowance = size + (-1 if cross_limit else 1)
    # A controlled polling interval puts every small final write after the last
    # real scan. The real F1 copy and real byte accounting remain active.
    monkeypatch.setattr(acquisition, "POLL_SECONDS", 60)
    if bound == "staging":
        monkeypatch.setattr(acquisition, "MAX_STAGING_BYTES", allowance)
    elif bound == "managed":
        monkeypatch.setattr(retention, "MANAGED_BYTES", initial + allowance)
    else:
        usage = namedtuple("Usage", "total used free")
        monkeypatch.setattr(
            "shutil.disk_usage",
            lambda _: usage(
                10**12,
                0,
                retention.FREE_RESERVE_BYTES
                + allowance
                - (retention.managed_usage(owner.root) - initial),
            ),
        )
    if cross_limit:
        with pytest.raises((ValueError, OSError), match="staging_limit|quota|reserve"):
            acquire(plugin_stack, source, "final-copy")
    else:
        result = acquire(plugin_stack, source, "final-copy")
        assert result.inspection.content_digest
    record = plugin_stack.call(lambda: acquisition._inventory(owner, authority))[0]
    assert record["phase"] == "terminal"
    assert record["recovery_referenced"] is (not cross_limit)
    assert not owner.acquisition_tasks
    assert plugin_stack.call(authority.verify_current) == before
    if cross_limit:
        assert owner.acquisition_destination("final-copy").exists()
        assert (
            plugin_stack.call(
                lambda: acquisition.cleanup_acquisitions(owner, authority)
            )
            == ()
        )
        assert plugin_stack.call(
            lambda: acquisition.cleanup_acquisitions(
                owner, authority, now=datetime.now(UTC) + timedelta(hours=25)
            )
        ) == ("final-copy",)
        assert not owner.acquisition_destination("final-copy").parent.exists()


@pytest.mark.asyncio
async def test_final_capacity_scan_retains_worker_after_waiter_cancel(
    tmp_path, native_package, monkeypatch
):
    import asyncio
    import json
    import shutil
    import threading

    from tldw_chatbook.Plugins import acquisition
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.service import PluginService, PluginUnavailable

    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    await service.bootstrap("test passphrase")
    destination = await service.acquisition_destination("final-scan")
    entered, release = threading.Event(), threading.Event()
    original = shutil.disk_usage

    def blocked_usage(path):
        if destination.exists():
            entered.set()
            assert release.wait(5)
        return original(path)

    monkeypatch.setattr(acquisition, "POLL_SECONDS", 60)
    monkeypatch.setattr(shutil, "disk_usage", blocked_usage)
    task = asyncio.create_task(
        service.acquire_snapshot(
            select_local_source(native_package()), "final-scan", destination
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert destination.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        owner = service._coordinator.owner
        assert owner.acquisition_tasks
        receipt = acquisition._record_path(owner, "final-scan")
        assert json.loads(receipt.read_text())["body"]["phase"] == "pending"
        with pytest.raises(PluginUnavailable, match="not_drained"):
            await service.aclose()
        release.set()
        for _ in range(500):
            if not owner.acquisition_tasks:
                break
            await asyncio.sleep(0.01)
        assert not owner.acquisition_tasks
        record = json.loads(receipt.read_text())["body"]
        assert record["phase"] == "terminal"
        assert record["recovery_referenced"] is False
    finally:
        release.set()
        await service.aclose()


@pytest.mark.parametrize("recipe", ["dev_20261001", "branch_7b4d5c", "integrated_2"])
def test_checkpoint_materialization_preserves_expected_recipe(
    native_package, tmp_path, recipe
):
    from tldw_chatbook.Plugins.inspection import _inspect_capture
    from tldw_chatbook.Plugins.package_files import capture_package, materialize_package

    source = native_package()
    expected = _inspect_capture(capture_package(source), recipe=recipe)
    checkpoints = []
    destination = tmp_path / "retained-output"
    actual = materialize_package(
        source,
        destination,
        expected=expected,
        dialect=expected.dialect,
        checkpoint=lambda: checkpoints.append(True),
    )
    assert checkpoints
    assert actual.adapter_version == expected.adapter_version
    assert actual.effective_digest == expected.effective_digest
    assert actual.inventory == expected.inventory
    assert actual.materialized_identity == str(destination)
    assert (destination / "plugin.json").read_bytes() == (
        source / "plugin.json"
    ).read_bytes()


@pytest.mark.parametrize("mismatch", ["dialect", "source"])
def test_checkpoint_materialization_cannot_bypass_expected_binding(
    native_package, tmp_path, mismatch
):
    from tldw_chatbook.Plugins.inspection import _inspect_capture
    from tldw_chatbook.Plugins.package_files import capture_package, materialize_package

    source = native_package()
    expected = _inspect_capture(capture_package(source), recipe="dev_20261001")
    dialect = expected.dialect
    if mismatch == "source":
        (source / "skills/review/SKILL.md").write_text("changed after inspection")
    else:
        dialect = "cursor"
    checkpoints = []
    destination = tmp_path / "refused-output"
    with pytest.raises(ValueError):
        materialize_package(
            source,
            destination,
            expected=expected,
            dialect=dialect,
            checkpoint=lambda: checkpoints.append(True),
        )
    assert checkpoints
    assert not destination.exists()
