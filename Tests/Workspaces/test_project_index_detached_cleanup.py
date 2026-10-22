"""Native cleanup ownership survives removal without restoring source authority."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from Tests.Agents.test_project_index_tool import provider, invoke
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.DB.Workspace_Project_Index_DB import (
    WorkspaceProjectIndexDB,
    index_store_path,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.project_index import runtime
from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy


async def build(service, key):
    off = service.status(key)
    enabled = service.enable(key, ProjectIndexPolicy((), ()), expected_scope=off.scope)
    job = service.refresh(
        key, expected_scope=enabled.scope, expected_job_revision=enabled.job_revision
    )
    await asyncio.to_thread(service._runs[job].supervisor.join, 30)
    assert service.status(key).state == "Ready"


@pytest.mark.parametrize("remove", ["detach", "archive"])
@private_profile_test
async def test_detached_cleanup_failure_restarts_and_spares_replacement(
    tmp_path, request, monkeypatch, remove
):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("detachedmarker " * 12000, encoding="utf-8")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "registry.db"))
    registry.create_workspace(workspace_id="removed", name="Removed")
    binding = registry.add_folder_binding("removed", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    service = runtime.get_project_index_service(app)
    key = runtime.project_index_key(app, "removed", binding.binding_id, root)
    try:
        await build(service, key)
        tool = provider(SimpleNamespace(root=root, service=service, key=key))
        path = index_store_path(service._factory.data_root, key)
        before = path.stat().st_size
        clear = WorkspaceProjectIndexDB.clear_owned_namespace

        def denied(*args, **kwargs):
            raise OSError("held cleanup failure")

        monkeypatch.setattr(WorkspaceProjectIndexDB, "clear_owned_namespace", denied)
        if remove == "detach":
            registry.remove_runtime_binding(binding.binding_id)
        else:
            registry.archive_workspace("removed")
        result = await asyncio.to_thread(invoke, tool, {"query": "detachedmarker"})
        assert not json.loads(result.content)["hits"]
        result = await asyncio.to_thread(service.retry_detached_cleanup)
        assert result["pending"] == 1 and result["completed"] == 0
        await runtime.close_project_index_service(app)
        monkeypatch.setattr(WorkspaceProjectIndexDB, "clear_owned_namespace", clear)
        app = SimpleNamespace(workspace_registry_service=registry)
        service = runtime.get_project_index_service(app)
        registry.create_workspace(workspace_id="replacement", name="Replacement")
        replacement = registry.add_folder_binding("replacement", root)
        new_key = runtime.project_index_key(
            app, "replacement", replacement.binding_id, root
        )
        await build(service, new_key)
        new_path = index_store_path(service._factory.data_root, new_key)
        new_bytes = new_path.read_bytes()

        # Cleanup must never inspect or readmit the removed source.
        def source_forbidden(*args, **kwargs):
            raise AssertionError("cleanup attempted source authority")

        monkeypatch.setattr(service, "_reader", source_forbidden)
        monkeypatch.setattr(service, "_fence", source_forbidden)
        result = await asyncio.to_thread(service.retry_detached_cleanup)
        assert (
            result["completed"] == 1
            and result["pending"] == 0
            and result["refused"] == 0
        )
        assert path.stat().st_size < before
        assert new_path.read_bytes() == new_bytes
        with service._factory.retain() as pin, pin.bind():
            store = WorkspaceProjectIndexDB(path, cleanup_only=True)
            try:
                with store.connection() as conn:
                    assert conn.execute("SELECT count(*) FROM files").fetchone()[0] == 0
                    assert (
                        conn.execute("SELECT count(*) FROM chunk_fts").fetchone()[0]
                        == 0
                    )
            finally:
                store.close_native()
    finally:
        await runtime.close_project_index_service(app)
        registry.db.close()


@private_profile_test
async def test_receipt_native_existing_only_bounds_and_alias_refusal(tmp_path, request):
    import os
    from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime
    from tldw_chatbook.Utils.private_paths import secure_private_directory

    group = tmp_path / "receipt"
    assert secure_private_directory(
        group, create=True, application_owned=True
    ).verified_private
    path = group / "cleanup-owner-v1.lock"
    with pytest.raises(FileNotFoundError):
        BindingLifetime.acquire(path, existing_only=True)
    assert not path.exists()
    handle = BindingLifetime.acquire(path)
    try:
        handle.write_cleanup_receipt(b"x" * 4096)
        assert handle.read_cleanup_receipt() == b"x" * 4096
        handle.write_cleanup_receipt(b"{}")
        assert handle.read_cleanup_receipt() == b"{}" and path.stat().st_size == 2
        with pytest.raises(ValueError, match="receipt"):
            handle.write_cleanup_receipt(b"x" * 4097)
    finally:
        handle.close()
    assert handle.closed
    os.link(path, group / "alias")
    with pytest.raises(ValueError, match="alias"):
        BindingLifetime.acquire(path, existing_only=True)


@private_profile_test
async def test_missing_or_corrupt_detached_receipt_refuses_without_content_loss(
    tmp_path, request
):
    from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("ownedmarker", encoding="utf-8")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "registry.db"))
    registry.create_workspace(workspace_id="ws", name="Workspace")
    binding = registry.add_folder_binding("ws", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    service = runtime.get_project_index_service(app)
    key = runtime.project_index_key(app, "ws", binding.binding_id, root)
    try:
        await build(service, key)
        service._unsubscribe()
        path = index_store_path(service._factory.data_root, key)
        receipt = path.parent.parent / "cleanup-owner-v1.lock"
        lease = BindingLifetime.acquire(receipt, existing_only=True)
        try:
            lease.write_cleanup_receipt(b"{}")
        finally:
            lease.close()
        registry.remove_runtime_binding(binding.binding_id)
        before = path.read_bytes()
        result = service.retry_detached_cleanup()
        assert result == {"completed": 0, "pending": 0, "refused": 1}
        assert path.read_bytes() == before
        receipt.unlink()
        result = service.retry_detached_cleanup()
        assert result["refused"] == 1 and not receipt.exists()
        assert path.read_bytes() == before
    finally:
        await runtime.close_project_index_service(app)
        registry.db.close()


@private_profile_test
async def test_same_id_recreation_cannot_enable_during_detached_reclamation(
    tmp_path, request, monkeypatch
):
    import threading
    from tldw_chatbook.Workspaces.project_index.lifetime import LifetimeBusy

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("racemarker", encoding="utf-8")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "registry.db"))
    registry.create_workspace(workspace_id="ws", name="Workspace")
    binding = registry.add_folder_binding("ws", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    service = runtime.get_project_index_service(app)
    key = runtime.project_index_key(app, "ws", binding.binding_id, root)
    entered, release = threading.Event(), threading.Event()
    task = None
    try:
        await build(service, key)
        service._unsubscribe()
        registry.remove_runtime_binding(binding.binding_id)
        reclaim = service._reclaim_namespaces

        def held(*args, **kwargs):
            entered.set()
            assert release.wait(30)
            return reclaim(*args, **kwargs)

        monkeypatch.setattr(service, "_reclaim_namespaces", held)
        task = asyncio.create_task(asyncio.to_thread(service.retry_detached_cleanup))
        assert await asyncio.to_thread(entered.wait, 10)
        registry.save_runtime_binding(binding)
        current = service.status(key)
        with pytest.raises(LifetimeBusy):
            service.enable(
                key, ProjectIndexPolicy((), ()), expected_scope=current.scope
            )
        release.set()
        assert (await task)["completed"] == 1
        off = service.status(key)
        assert off.state == "Off" and off.indexed == 0
        enabled = service.enable(
            key, ProjectIndexPolicy((), ()), expected_scope=off.scope
        )
        assert enabled.state == "Paused"
    finally:
        release.set()
        if task is not None:
            await task
        await runtime.close_project_index_service(app)
        registry.db.close()


@private_profile_test
async def test_failed_receipt_close_retains_real_capability_pin_and_operation(
    tmp_path, request, monkeypatch
):
    import threading
    from tldw_chatbook.Workspaces.project_index.lifetime import (
        BindingLifetime,
        LifetimeBusy,
    )
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexRetirementUnproven,
    )

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("closemarker", encoding="utf-8")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "registry.db"))
    registry.create_workspace(workspace_id="ws", name="Workspace")
    binding = registry.add_folder_binding("ws", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    service = runtime.get_project_index_service(app)
    key = runtime.project_index_key(app, "ws", binding.binding_id, root)
    failed, repair = threading.Event(), threading.Event()
    owners, errors = [], []
    thread = None
    close = BindingLifetime.close
    try:
        await build(service, key)
        service._unsubscribe()
        registry.remove_runtime_binding(binding.binding_id)

        def denied(lease):
            if lease.path.name == "cleanup-owner-v1.lock":
                owners.append(lease)
                raise OSError("native close refused")
            return close(lease)

        monkeypatch.setattr(BindingLifetime, "close", denied)

        def retry():
            try:
                service.retry_detached_cleanup()
            except BaseException as error:
                errors.append(error)
            finally:
                failed.set()
                assert repair.wait(30)
                for operation in tuple(service._operations):
                    for owner in operation.retained:
                        owner.close()

        thread = threading.Thread(target=retry)
        thread.start()
        assert await asyncio.to_thread(failed.wait, 15)
        assert isinstance(errors[0], ProjectIndexRetirementUnproven)
        assert owners and not owners[0].closed
        with pytest.raises(LifetimeBusy):
            BindingLifetime.acquire(owners[0].path, existing_only=True)
        assert any(
            not owner.closed
            for operation in service._operations
            for owner in operation.retained
        )
        with pytest.raises(ProjectIndexRetirementUnproven):
            service.management_snapshot(key)
        with pytest.raises(ProjectIndexRetirementUnproven):
            await service.aclose()
    finally:
        monkeypatch.setattr(BindingLifetime, "close", close)
        repair.set()
        if thread is not None:
            await asyncio.to_thread(thread.join, 10)
        registry.db.close()


@private_profile_test
async def test_detached_cleanup_cold_child_uses_recorded_owner_only(tmp_path, request):
    import subprocess
    import sys

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("coldmarker " * 15000, encoding="utf-8")
    registry_path = tmp_path / "registry.db"
    registry = LocalWorkspaceRegistryService(WorkspaceDB(registry_path))
    registry.create_workspace(workspace_id="ws", name="Workspace")
    binding = registry.add_folder_binding("ws", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    service = runtime.get_project_index_service(app)
    key = runtime.project_index_key(app, "ws", binding.binding_id, root)
    await build(service, key)
    service._unsubscribe()
    registry.remove_runtime_binding(binding.binding_id)
    path = index_store_path(service._factory.data_root, key)
    before = path.stat().st_size
    await runtime.close_project_index_service(app)
    registry.db.close()
    receipt = tmp_path / "cold-cleanup.json"
    script = "\n".join(
        [
            "import Tests.conftest",
            "import asyncio,json",
            "from pathlib import Path",
            "from types import SimpleNamespace",
            "from tldw_chatbook.DB.Workspace_DB import WorkspaceDB",
            "from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService",
            "from tldw_chatbook.Workspaces.project_index.runtime import get_project_index_service",
            f"registry=LocalWorkspaceRegistryService(WorkspaceDB(Path({str(registry_path)!r})))",
            "service=get_project_index_service(SimpleNamespace(workspace_registry_service=registry))",
            "def forbidden(*args): raise AssertionError('source authority attempted')",
            "service._reader=service._fence=forbidden",
            "result=service.retry_detached_cleanup()",
            f"Path({str(receipt)!r}).write_text(json.dumps(result),encoding='utf-8')",
            "asyncio.run(service.aclose())",
            "registry.db.close()",
        ]
    )
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "completed": 1,
        "pending": 0,
        "refused": 0,
    }
    assert path.stat().st_size < before


@private_profile_test
async def test_detach_during_real_scan_revokes_then_retries_after_retirement(
    tmp_path, request, monkeypatch
):
    import threading

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("inflightmarker", encoding="utf-8")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "registry.db"))
    registry.create_workspace(workspace_id="ws", name="Workspace")
    binding = registry.add_folder_binding("ws", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    service = runtime.get_project_index_service(app)
    key = runtime.project_index_key(app, "ws", binding.binding_id, root)
    entered, release = threading.Event(), threading.Event()
    scan = service._scanner.scan

    def held(authority, scope, job, frontier, emit, stop, **kwargs):
        def publish(unit):
            accepted = emit(unit)
            if accepted and unit.captures:
                entered.set()
                assert release.wait(30)
            return accepted

        return scan(authority, scope, job, frontier, publish, stop, **kwargs)

    monkeypatch.setattr(service._scanner, "scan", held)
    try:
        off = service.status(key)
        enabled = service.enable(
            key, ProjectIndexPolicy((), ()), expected_scope=off.scope
        )
        job = service.refresh(
            key,
            expected_scope=enabled.scope,
            expected_job_revision=enabled.job_revision,
        )
        run = service._runs[job]
        assert await asyncio.to_thread(entered.wait, 15)
        tool = provider(SimpleNamespace(root=root, service=service, key=key))
        registry.remove_runtime_binding(binding.binding_id)
        assert run.stop.is_set()
        await asyncio.to_thread(service._cleanup_thread.join, 10)
        assert service.retry_detached_cleanup()["pending"] == 1
        result = await asyncio.to_thread(invoke, tool, {"query": "inflightmarker"})
        assert json.loads(result.content)["status"] == "unavailable"
        release.set()
        await asyncio.to_thread(run.supervisor.join, 15)
        assert (
            run.native_retired
            and service._scanner.retired
            and service._scanner.pipes_closed
        )
        assert service.retry_detached_cleanup()["completed"] == 1
    finally:
        release.set()
        await runtime.close_project_index_service(app)
        registry.db.close()
