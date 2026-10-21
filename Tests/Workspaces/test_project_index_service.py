"""Application index lifecycle uses real private stores and contained scanners."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import importlib
import threading
import time

import pytest
from Tests.private_profile import private_profile_test

from Tests.Workspaces.test_project_index_scanner import authority
from tldw_chatbook.DB.Workspace_Project_Index_DB import (
    WorkspaceProjectIndexDB,
    index_store_path,
)
from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner

POLICY = ProjectIndexPolicy(("",), (".txt", ".py"))


class Rig:
    def __init__(self, tmp_path, scanner=None):
        self.root = tmp_path / "project"
        self.root.mkdir()
        self.data = tmp_path / "cache"
        self.live = authority(self.root)
        self.key = self.live.key
        self.lock = threading.RLock()
        cls = importlib.import_module(
            "tldw_chatbook.Workspaces.project_index.service"
        ).ProjectIndexService
        self.service = cls(
            self.store, scanner or ProjectScanner(), self.read, self.fence
        )

    def store(self, key):
        path = index_store_path(self.data, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return WorkspaceProjectIndexDB(path)

    @contextmanager
    def fence(self, key):
        with self.lock:
            yield self.live if self.live and self.live.key == key else None

    def read(self, key):
        with self.fence(key) as current:
            return current

    def enable(self):
        status = self.service.status(self.key)
        return self.service.enable(self.key, POLICY, expected_scope=status.scope)

    def refresh(self):
        status = self.service.status(self.key)
        return self.service.refresh(
            self.key,
            expected_scope=status.scope,
            expected_job_revision=status.job_revision,
        )

    def settle(self, state="Ready", *, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            status = self.service.status(self.key)
            if status.state == state:
                return status
            time.sleep(0.02)
        raise AssertionError(f"expected {state}, got {status}")

    async def close(self):
        await self.service.aclose()


@private_profile_test
async def test_incremental_refresh_retires_deleted_and_renamed_membership(
    tmp_path, request
):
    rig = Rig(tmp_path)
    try:
        (rig.root / "a.txt").write_text("first marker")
        (rig.root / "nested").mkdir()
        (rig.root / "nested/b.txt").write_text("second marker")
        rig.enable()
        rig.refresh()
        assert rig.settle().indexed == 2
        (rig.root / "a.txt").rename(rig.root / "renamed.txt")
        (rig.root / "nested/b.txt").unlink()
        rig.refresh()
        assert rig.settle().indexed == 1
        store = rig.store(rig.key)
        with store.transaction() as conn:
            assert [row[0] for row in conn.execute("SELECT path FROM files")] == [
                "renamed.txt"
            ]
        store.close()
    finally:
        await rig.close()


@private_profile_test
async def test_delayed_management_cannot_reverse_clear_or_new_enable(tmp_path, request):
    rig = Rig(tmp_path)
    try:
        off = rig.service.status(rig.key)
        enabled = rig.enable()
        with pytest.raises(ValueError, match="stale_scope"):
            rig.service.clear(rig.key, expected_scope=off.scope)
        cleared = rig.service.clear(rig.key, expected_scope=enabled.scope)
        with pytest.raises(ValueError, match="stale_scope"):
            rig.service.enable(rig.key, POLICY, expected_scope=off.scope)
        with pytest.raises(ValueError, match="stale_scope"):
            rig.service.disable(rig.key, expected_scope=enabled.scope)
        assert cleared.state == "Off"
        assert rig.service.status(rig.key).scope == cleared.scope
    finally:
        await rig.close()


class BarrierScanner(ProjectScanner):
    def __init__(self):
        super().__init__()
        self.before_publish = threading.Event()
        self.release = threading.Event()
        self.published = None

    def scan(self, *args, **kwargs):
        emit = args[4]

        def held(page):
            self.before_publish.set()
            assert self.release.wait(10)
            self.published = emit(page)
            return self.published

        return super().scan(*args[:4], held, *args[5:], **kwargs)


@pytest.mark.parametrize("mutation", ["detach", "revision"])
@private_profile_test
async def test_late_publication_rechecks_authority_and_never_returns_ready(
    tmp_path, request, mutation
):
    scanner = BarrierScanner()
    rig = Rig(tmp_path, scanner)
    try:
        (rig.root / "a.txt").write_text("held marker")
        rig.enable()
        rig.refresh()
        assert scanner.before_publish.wait(10)
        with rig.lock:
            rig.live = (
                None
                if mutation == "detach"
                else replace(rig.live, authority_revision=2)
            )
        scanner.release.set()
        rig.settle("Off" if mutation == "detach" else "Paused")
        assert scanner.published is False
        assert rig.service.status(rig.key).indexed == 0
    finally:
        scanner.release.set()
        await rig.close()


@private_profile_test
async def test_pause_and_explicit_resume_keep_exact_job_cas(tmp_path, request):
    scanner = BarrierScanner()
    rig = Rig(tmp_path, scanner)
    try:
        (rig.root / "a.txt").write_text("held marker")
        rig.enable()
        job = rig.refresh()
        assert scanner.before_publish.wait(10)
        current = rig.service.status(rig.key)
        scanner.release.set()
        paused = rig.service.pause(
            job,
            expected_scope=current.scope,
            expected_job_revision=current.job_revision,
        )
        replacement = rig.service.resume(
            job, expected_scope=paused.scope, expected_job_revision=paused.job_revision
        )
        assert replacement != job
        with pytest.raises(ValueError, match="stale_job"):
            rig.service.pause(
                job,
                expected_scope=paused.scope,
                expected_job_revision=paused.job_revision,
            )
        assert rig.settle().indexed == 1
    finally:
        scanner.release.set()
        await rig.close()


@private_profile_test
async def test_runtime_is_lazy_identity_is_stable_and_old_profile_is_refused(
    tmp_path, request, monkeypatch
):
    from types import SimpleNamespace
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    runtime = importlib.import_module("tldw_chatbook.Workspaces.project_index.runtime")
    app = SimpleNamespace()
    await runtime.close_project_index_service(app)
    assert not hasattr(app, "_project_index_service")
    app = SimpleNamespace(
        workspace_registry_service=LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.db")
        )
    )
    try:
        first = runtime.get_project_index_service(app)
        assert first is runtime.get_project_index_service(app)
        factory = first._factory
        with factory.retain() as pin:
            identity = pin.identity
        from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path

        config = effective_config_path()
        with config.open("a", encoding="utf-8") as stream:
            stream.write("\n# unrelated preference save\n")
        with factory.retain() as pin:
            assert pin.identity == identity
        with monkeypatch.context() as changed:
            installation, generation = runtime.verified_data_identity()
            changed.setattr(
                runtime,
                "verified_data_identity",
                lambda: (installation, generation + "-restored"),
            )
            with pytest.raises(ValueError, match="index_data_authority_changed"):
                factory.retain()
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "other-config.toml"))
        with pytest.raises(Exception):
            factory.retain()
    finally:
        await runtime.close_project_index_service(app)
        app.workspace_registry_service.db.close()


@private_profile_test
async def test_whole_binding_quota_counts_prior_authority_and_root(tmp_path, request):
    from tldw_chatbook.Workspaces.project_index.service import binding_bytes

    rig = Rig(tmp_path)
    try:
        rig.enable()
        old_key = replace(
            rig.key,
            data_authority_id="previous-authority",
            root_identity_fingerprint="previous-root",
        )
        old_path = index_store_path(rig.data, old_key)
        old_path.parent.mkdir(parents=True)
        (old_path.parent / "obsolete-vector.tmp").write_bytes(b"x" * 300)
        assert (
            old_path.parent.parent == index_store_path(rig.data, rig.key).parent.parent
        )
        before = binding_bytes(old_path.parent.parent)
        (old_path.parent / "obsolete-vector.tmp").write_bytes(b"x" * 800)
        assert binding_bytes(old_path.parent.parent) == before + 500
    finally:
        await rig.close()


@private_profile_test
async def test_cleanup_failure_stays_off_and_no_worker_reopens(
    tmp_path, request, monkeypatch
):
    rig = Rig(tmp_path)
    try:
        enabled = rig.enable()

        def fail(store, scope):
            raise OSError("synthetic cleanup failure")

        monkeypatch.setattr(WorkspaceProjectIndexDB, "clear_revoked", fail)
        with pytest.raises(OSError, match="synthetic cleanup"):
            rig.service.clear(rig.key, expected_scope=enabled.scope)
        assert rig.service.status(rig.key).state == "Off"
        assert not rig.service._runs
    finally:
        await rig.close()


@private_profile_test
async def test_shutdown_retains_lifetime_through_native_db_closure_and_repeated_cancel(
    tmp_path, request, monkeypatch
):
    import sqlite3
    from tldw_chatbook.Workspaces.project_index.lifetime import (
        BindingLifetime,
        LifetimeBusy,
    )

    rig = Rig(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = WorkspaceProjectIndexDB.close
    native = []

    def held_close(store):
        if (
            threading.current_thread().name == "project-index-writer"
            and not entered.is_set()
        ):
            connection = getattr(store._thread_local, "conn", None)
            if connection is not None:
                native.append(connection)
                entered.set()
                assert release.wait(15)
        original(store)

    monkeypatch.setattr(WorkspaceProjectIndexDB, "close", held_close)
    closer = None
    try:
        (rig.root / "a.txt").write_text("closure marker")
        rig.enable()
        rig.refresh()
        assert await asyncio.to_thread(entered.wait, 10)
        group = index_store_path(rig.data, rig.key).parent.parent
        with pytest.raises(LifetimeBusy):
            BindingLifetime.acquire(group / rig.key.lifetime_name)
        closer = asyncio.create_task(rig.service.aclose())
        await asyncio.sleep(0.01)
        closer.cancel()
        await asyncio.sleep(0.01)
        closer.cancel()
        await asyncio.sleep(0.01)
        assert not closer.done()
        assert rig.service._scanner.retired
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closer
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.in_transaction.__get__(native[0])
        assert all(
            not run.writer.is_alive() and not run.supervisor.is_alive()
            for run in rig.service._runs.values()
        )
        lease = BindingLifetime.acquire(group / rig.key.lifetime_name)
        lease.close()
    finally:
        release.set()
        if closer is not None and not closer.done():
            await closer
        await rig.close()


@pytest.mark.parametrize("mutation", ["detach", "exclude", "retarget", "clear"])
@private_profile_test
async def test_cross_process_mutation_prevents_held_production_publication(
    tmp_path, request, mutation
):
    import subprocess
    import sys
    from pathlib import Path
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    scanner = BarrierScanner()
    rig = Rig(tmp_path, scanner)
    registry_path = tmp_path / "registry.db"
    registry = LocalWorkspaceRegistryService(WorkspaceDB(registry_path))
    registry.create_workspace(workspace_id="project-ws", name="Project")
    binding = registry.add_folder_binding("project-ws", rig.root)
    rig.key = replace(rig.key, workspace_id="project-ws", binding_id=binding.binding_id)
    reader, fence = registry.project_index_authority_adapters(lambda: "test")
    rig.service = type(rig.service)(rig.store, scanner, reader, fence)
    (rig.root / "a.txt").write_text("held cross process marker")
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    try:
        rig.enable()
        rig.refresh()
        assert scanner.before_publish.wait(10)
        code = f"""from pathlib import Path
from dataclasses import replace
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
from tldw_chatbook.Workspaces.project_index.models import ProjectIndexKey
registry=LocalWorkspaceRegistryService(WorkspaceDB(Path({str(registry_path)!r})))
bid={binding.binding_id!r}
action={mutation!r}
if action=='detach': registry.remove_runtime_binding(bid)
elif action=='exclude': registry.add_binding_exclusion('project-ws',bid,'a.txt')
elif action=='retarget': registry.save_runtime_binding(replace(registry.get_runtime_binding(bid),locator={str(alternate)!r}))
else:
    key=ProjectIndexKey({rig.key.data_authority_id!r},{rig.key.workspace_id!r},{rig.key.binding_id!r},{rig.key.locator_fingerprint!r},{rig.key.root_identity_fingerprint!r})
    store=WorkspaceProjectIndexDB(Path({str(index_store_path(rig.data, rig.key))!r}))
    with registry.binding_authority_fence(bid):
        store.revoke(key,clear=False,expected_scope=store.load_status(key).scope)
    store.close()
registry.db.close()
print('mutation committed')
"""
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "mutation committed" in result.stdout
        scanner.release.set()
        rig.settle("Paused" if mutation == "exclude" else "Off")
        await rig.close()
        assert scanner.published is False
        assert scanner.retired and scanner.pipes_closed
    finally:
        scanner.release.set()
        await rig.close()
        registry.db.close()


@private_profile_test
async def test_legacy_claim_cannot_be_laundered_through_disable_enable(
    tmp_path, request
):
    rig = Rig(tmp_path)
    try:
        status = rig.enable()
        store = rig.store(rig.key)
        store.claim_job(
            status.scope,
            owner_token="pre-native-owner",
            authority_revision=1,
            expected_job_revision=status.job_revision,
        )
        store.close()
        off = rig.service.disable(rig.key, expected_scope=status.scope)
        with pytest.raises(ValueError, match="legacy_retirement_unproven"):
            rig.service.enable(rig.key, POLICY, expected_scope=off.scope)
    finally:
        await rig.close()


@private_profile_test
async def test_real_app_shutdown_without_console_drains_index_writer(tmp_path, request):
    from tldw_chatbook.app import TldwCli

    scanner = BarrierScanner()
    rig = Rig(tmp_path, scanner)
    app = TldwCli()
    app._project_index_service = rig.service
    try:
        (rig.root / "a.txt").write_text("app shutdown marker")
        rig.enable()
        rig.refresh()
        assert await asyncio.to_thread(scanner.before_publish.wait, 10)
        assert not any(
            type(screen).__name__ == "ChatScreen" for screen in app.screen_stack
        )
        closing = asyncio.create_task(app._shutdown_app_owned_lifecycles())
        await asyncio.sleep(0.05)
        assert not closing.done()
        scanner.release.set()
        await closing
        assert scanner.retired and scanner.pipes_closed
        assert all(not run.writer.is_alive() for run in rig.service._runs.values())
    finally:
        scanner.release.set()
        await rig.close()


@private_profile_test
async def test_failed_supervisor_dispatch_waits_for_started_writer_retirement(
    tmp_path, request, monkeypatch
):
    scanner = BarrierScanner()
    rig = Rig(tmp_path, scanner)
    failed = threading.Event()
    start = threading.Thread.start

    def fail_owner(thread):
        if thread.name == "project-index-owner":
            assert scanner.before_publish.wait(10)
            failed.set()
            raise RuntimeError("synthetic owner dispatch failed")
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_owner)
    task = None
    try:
        (rig.root / "a.txt").write_text("dispatch marker")
        rig.enable()
        task = asyncio.create_task(asyncio.to_thread(rig.refresh))
        assert await asyncio.to_thread(failed.wait, 10)
        await asyncio.sleep(0.03)
        assert not task.done(), "caller completed while its real writer was still alive"
        scanner.release.set()
        with pytest.raises(RuntimeError, match="synthetic owner dispatch"):
            await task
        assert scanner.retired
        assert all(not run.writer.is_alive() for run in rig.service._runs.values())
    finally:
        scanner.release.set()
        if task is not None and not task.done():
            try:
                await task
            except RuntimeError:
                pass
        await rig.close()


@private_profile_test
async def test_crashed_writer_stays_interrupted_until_explicit_resume_without_duplicates(
    tmp_path, request
):
    import subprocess
    import sys
    from pathlib import Path
    from tldw_chatbook.Workspaces.project_index.lifetime import LifetimeBusy

    rig = Rig(tmp_path)
    for index in range(300):
        (rig.root / f"{index:04}.txt").write_text("restart marker")
    rig.enable()
    paths = [
        str(Path(__file__).resolve().parents[2]),
        *[path for path in sys.path if "site-packages" in path],
    ]
    code = f"""import sys
sys.path[:0]={paths!r}
from pathlib import Path
from contextlib import contextmanager
import threading
from Tests.Workspaces.test_project_index_scanner import authority
from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB,index_store_path
from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService
from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
root=Path({str(rig.root)!r}); data=Path({str(rig.data)!r}); auth=authority(root)
@contextmanager
def fence(key): yield auth
class Held(ProjectScanner):
    def scan(self,*args,**kwargs):
        emit=args[4]
        def hold(page):
            result=emit(page)
            print('committed',flush=True)
            threading.Event().wait()
            return result
        return super().scan(*args[:4],hold,*args[5:],**kwargs)
service=ProjectIndexService(lambda key:WorkspaceProjectIndexDB(index_store_path(data,key)),Held(),lambda key:auth,fence)
s=service.status(auth.key)
service.refresh(auth.key,expected_scope=s.scope,expected_job_revision=s.job_revision)
threading.Event().wait()
"""
    child = subprocess.Popen(
        [getattr(sys, "_base_executable", sys.executable), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        line = await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 20)
        assert line.strip() == b"committed"
        current = rig.service.status(rig.key)
        assert current.state == "Building" and current.indexed > 0
        with pytest.raises(LifetimeBusy):
            rig.service.refresh(
                rig.key,
                expected_scope=current.scope,
                expected_job_revision=current.job_revision,
            )
        child.kill()
        assert await asyncio.to_thread(child.wait, 10) is not None
        interrupted = rig.settle("Interrupted")
        assert interrupted.job_id == current.job_id
        assert not rig.service._runs
        rig.service.resume(
            interrupted.job_id,
            expected_scope=interrupted.scope,
            expected_job_revision=interrupted.job_revision,
        )
        ready = rig.settle()
        assert ready.indexed == 300
        store = rig.store(rig.key)
        with store.connection() as conn:
            assert conn.execute(
                "SELECT COUNT(*),COUNT(DISTINCT path) FROM files"
            ).fetchone()[:] == (300, 300)
        store.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdout.close()
        await rig.close()


@private_profile_test
async def test_refresh_preserves_unfinished_subtree_and_can_restart_partial(
    tmp_path, request
):
    class Incomplete(ProjectScanner):
        incomplete = False

        def scan(self, *args, **kwargs):
            emit = args[4]

            def partial(page):
                if self.incomplete and page.directory == "nested":
                    page = replace(page, directory_complete=False)
                return emit(page)

            return super().scan(*args[:4], partial, *args[5:], **kwargs)

    scanner = Incomplete()
    rig = Rig(tmp_path, scanner)
    try:
        (rig.root / "nested").mkdir()
        (rig.root / "nested/a.txt").write_text("unchanged marker")
        rig.enable()
        rig.refresh()
        rig.settle()
        store = rig.store(rig.key)
        with store.connection() as conn:
            original = tuple(conn.execute("SELECT id,text FROM chunks").fetchone())
        store.close()
        rig.refresh()
        rig.settle()
        store = rig.store(rig.key)
        with store.connection() as conn:
            assert (
                tuple(conn.execute("SELECT id,text FROM chunks").fetchone()) == original
            )
        store.close()
        (rig.root / "nested/a.txt").unlink()
        scanner.incomplete = True
        rig.refresh()
        assert rig.settle("Partial").indexed == 1
        for run in rig.service._runs.values():
            run.supervisor.join()
        scanner.incomplete = False
        rig.refresh()
        assert rig.settle().indexed == 0
    finally:
        await rig.close()


@private_profile_test
async def test_policy_narrowing_filters_existing_text_and_widening_needs_refresh(
    tmp_path, request
):
    from Tests.Workspaces.project_index_support import search_authority

    rig = Rig(tmp_path)
    try:
        (rig.root / "a.txt").write_text("policy marker")
        (rig.root / "b.py").write_text("policy marker")
        rig.enable()
        rig.refresh()
        ready = rig.settle()
        narrow = rig.service.enable(
            rig.key, ProjectIndexPolicy(("",), (".py",)), expected_scope=ready.scope
        )
        store = rig.store(rig.key)
        assert [
            c.relative_path
            for c in store.lexical_candidates(
                search_authority(rig.root, narrow.scope), "marker", limit=10
            )
        ] == ["b.py"]
        store.close()
        rig.refresh()
        assert rig.settle().indexed == 1
        widened = rig.service.enable(rig.key, POLICY, expected_scope=narrow.scope)
        assert widened.state == "Paused" and widened.indexed == 1
        rig.refresh()
        assert rig.settle().indexed == 2
    finally:
        await rig.close()


@private_profile_test
async def test_production_runtime_pin_drains_idle_and_spans_writer_retirement(
    tmp_path, request
):
    from types import SimpleNamespace
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.project_index.runtime import (
        get_project_index_service,
        project_index_key,
        close_project_index_service,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _retire_current_thread_caches

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("production pin marker")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "registry.db"))
    registry.create_workspace(workspace_id="project-ws", name="Project")
    binding = registry.add_folder_binding("project-ws", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    scanner = BarrierScanner()
    service = get_project_index_service(app)
    assert service is get_project_index_service(app)
    service._scanner = scanner
    key = project_index_key(app, "project-ws", binding.binding_id, root)
    pause = None
    try:
        current = service.status(key)
        enabled = service.enable(key, POLICY, expected_scope=current.scope)
        pause = storage._begin_local_pause()
        _retire_current_thread_caches(pause)
        assert pause.drain(time.monotonic() + 2), "idle index retained a storage lease"
        pause.resume()
        pause = None
        service.refresh(
            key,
            expected_scope=enabled.scope,
            expected_job_revision=enabled.job_revision,
        )
        assert await asyncio.to_thread(scanner.before_publish.wait, 10)
        runs = tuple(service._runs.values())
        assert len(runs) == 1 and not runs[0].pin.closed
        pause = storage._begin_local_pause()
        _retire_current_thread_caches(pause)
        assert not pause.drain(
            time.monotonic()
        ), "held writer lost retained storage admission"
        pause.resume()
        pause = None
        scanner.release.set()
        await close_project_index_service(app)
        assert runs[0].native_retired and runs[0].pin.closed
        assert not runs[0].writer.is_alive() and not runs[0].supervisor.is_alive()
    finally:
        if pause is not None:
            pause.resume()
        scanner.release.set()
        await close_project_index_service(app)
        registry.db.close()


@private_profile_test
async def test_obsolete_namespace_bytes_force_partial_without_publication(
    tmp_path, request, monkeypatch
):
    import tldw_chatbook.DB.Workspace_Project_Index_DB as db_module
    import tldw_chatbook.Workspaces.project_index.service as service_module

    rig = Rig(tmp_path)
    try:
        rig.enable()
        obsolete = index_store_path(
            rig.data, replace(rig.key, data_authority_id="retired-generation")
        )
        obsolete.parent.mkdir(parents=True)
        (obsolete.parent / "old.db-journal").write_bytes(b"x" * (256 * 1024))
        monkeypatch.setattr(db_module, "MAX_CACHE_BYTES", 256 * 1024)
        monkeypatch.setattr(service_module, "MAX_CACHE_BYTES", 256 * 1024)
        (rig.root / "a.txt").write_text("quota marker")
        rig.refresh()
        status = rig.settle("Partial")
        assert status.indexed == 0 and status.cache_bytes > 256 * 1024
    finally:
        await rig.close()


@private_profile_test
async def test_clear_reclaims_prior_roots_and_authorities_preserving_control_and_leases(
    tmp_path, request
):
    from Tests.Workspaces.project_index_support import page
    from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime
    from tldw_chatbook.Workspaces.project_index.service import binding_bytes

    rig = Rig(tmp_path)
    keys = [
        replace(rig.key, root_identity_fingerprint="previous-root"),
        replace(rig.key, data_authority_id="previous-authority"),
    ]
    scopes = []
    stamps = []
    try:
        current = rig.enable()
        for n, key in enumerate(keys):
            store = rig.store(key)
            scope = store.configure(
                key, POLICY, enabled=True, expected_scope=store.load_status(key).scope
            )
            lease = BindingLifetime.acquire(
                index_store_path(rig.data, key).parent.parent / key.lifetime_name
            )
            claim = store.claim_job(
                scope,
                owner_token="native-v1:" + key.binding_namespace + ":" + str(n),
                authority_revision=1,
                expected_job_revision=store.load_status(key).job_revision,
            )
            assert store.commit_scan_page(
                scope,
                claim.job_id,
                page("reclamation marker " * 20000, path="old.txt"),
                expected_job_revision=claim.job_revision,
                authority_revision=1,
            )
            store.retire_job(
                scope,
                claim.job_id,
                owner_token=claim.owner_token,
                expected_job_revision=claim.job_revision,
                state="Ready",
            )
            store.close()
            lease.close()
            scopes.append(scope)
            stamps.append(index_store_path(rig.data, key).stat().st_ino)
        group = index_store_path(rig.data, rig.key).parent.parent
        before = binding_bytes(group)
        cleared = rig.service.clear(rig.key, expected_scope=current.scope)
        assert cleared.state == "Off"
        assert binding_bytes(group) < before // 2
        for key, scope, inode in zip(keys, scopes, stamps):
            assert index_store_path(rig.data, key).stat().st_ino == inode
            assert (group / key.lifetime_name).exists()
            store = rig.store(key)
            after = store.load_status(key)
            assert after.state == "Off" and after.indexed == 0
            assert after.scope.epoch > scope.epoch
            assert store.read_claim(after.scope) is None
            store.close()
        with pytest.raises(ValueError, match="stale_scope"):
            rig.service.enable(rig.key, POLICY, expected_scope=current.scope)
        rig.enable()
        (rig.root / "a.txt").write_text("after clear marker")
        rig.refresh()
        assert rig.settle().indexed == 1
    finally:
        await rig.close()


@private_profile_test
async def test_admitted_capacity_generations_clear_and_rebuild_with_bounded_peak(
    tmp_path, request, monkeypatch
):
    from hashlib import sha256
    import tldw_chatbook.DB.Workspace_Project_Index_DB as db_module
    import tldw_chatbook.Workspaces.project_index.service as service_module

    limit = 12 * 1024**2
    monkeypatch.setattr(db_module, "MAX_CACHE_BYTES", limit)
    monkeypatch.setattr(service_module, "MAX_CACHE_BYTES", limit)
    rig = Rig(tmp_path)
    original_key = rig.key
    samples = []
    end = threading.Event()
    group = index_store_path(rig.data, rig.key).parent.parent

    def sample():
        while not end.wait(0.002):
            try:
                samples.append(service_module.binding_bytes(group))
            except FileNotFoundError:
                pass

    sampler = threading.Thread(target=sample)
    try:
        rig.key = replace(original_key, data_authority_id="prior-authority")
        rig.live = replace(rig.live, key=rig.key)
        (rig.root / "prior.txt").write_text("prior marker " * 10000)
        rig.enable()
        rig.refresh()
        rig.settle()
        for run in rig.service._runs.values():
            run.supervisor.join()
        prior_key = rig.key
        rig.key = original_key
        rig.live = replace(rig.live, key=original_key)
        for n in range(20):
            directory = rig.root / f"d{n:02}"
            directory.mkdir()
            body = " ".join(
                sha256(f"{n}:{i}".encode()).hexdigest() for i in range(2000)
            )
            (directory / "data.txt").write_text(body, encoding="utf-8")
        rig.enable()
        sampler.start()
        rig.refresh()
        rig.settle("Partial", seconds=60)
        for run in rig.service._runs.values():
            run.supervisor.join()
        old_size = index_store_path(rig.data, original_key).stat().st_size
        admitted_pages = (
            limit - index_store_path(rig.data, prior_key).stat().st_size - 1024**2
        ) // 2
        assert old_size > admitted_pages * 0.8
        rig.root = tmp_path / "new-root"
        rig.root.mkdir()
        rig.live = authority(rig.root)
        rig.key = rig.live.key
        current = rig.enable()
        before = service_module.binding_bytes(group)
        cleared = rig.service.clear(rig.key, expected_scope=current.scope)
        after = service_module.binding_bytes(group)
        assert cleared.state == "Off" and after < before // 2
        assert (group / prior_key.lifetime_name).exists()
        assert (group / original_key.lifetime_name).exists()
        rig.enable()
        (rig.root / "after.txt").write_text("rebuilt marker")
        rig.refresh()
        assert rig.settle().indexed == 1
        assert samples and max(samples) <= limit
        (tmp_path / "capacity-receipt.txt").write_text(
            f"capacity={limit} old_database={old_size} before_clear={before} after_clear={after} sampled_highwater={max(samples)}",
            encoding="utf-8",
        )
        print(
            f"capacity={limit} old_database={old_size} before_clear={before} after_clear={after} sampled_highwater={max(samples)}"
        )
    finally:
        end.set()
        if sampler.ident is not None:
            sampler.join()
        await rig.close()


@private_profile_test
async def test_compaction_failure_is_off_retryable_and_stale_clear_cannot_undo_enable(
    tmp_path, request, monkeypatch
):
    rig = Rig(tmp_path)
    entered, release = threading.Event(), threading.Event()
    task = None
    try:
        current = rig.enable()

        def fail_once(store):
            raise OSError("synthetic disk failure")

        with monkeypatch.context() as patch:
            patch.setattr(WorkspaceProjectIndexDB, "reclaim_free_pages", fail_once)
            with pytest.raises(OSError, match="synthetic disk"):
                rig.service.clear(rig.key, expected_scope=current.scope)
        off = rig.service.status(rig.key)
        assert off.state == "Off"
        from tldw_chatbook.Workspaces.project_index.lifetime import LifetimeBusy

        compact = WorkspaceProjectIndexDB.reclaim_free_pages

        def held_compact(store):
            entered.set()
            assert release.wait(10)
            return compact(store)

        with monkeypatch.context() as patch:
            patch.setattr(WorkspaceProjectIndexDB, "reclaim_free_pages", held_compact)
            task = asyncio.create_task(
                asyncio.to_thread(rig.service.clear, rig.key, expected_scope=off.scope)
            )
            assert await asyncio.to_thread(entered.wait, 10)
            revoked = rig.service.status(rig.key)
            with pytest.raises(LifetimeBusy):
                rig.service.enable(rig.key, POLICY, expected_scope=revoked.scope)
            release.set()
            retry = await task
        assert retry.state == "Off"
        entered.clear()
        release.clear()
        cleanup = rig.service._clear_binding

        def held(*args):
            entered.set()
            assert release.wait(10)
            return cleanup(*args)

        monkeypatch.setattr(rig.service, "_clear_binding", held)
        task = asyncio.create_task(
            asyncio.to_thread(rig.service.clear, rig.key, expected_scope=retry.scope)
        )
        assert await asyncio.to_thread(entered.wait, 10)
        revoked = rig.service.status(rig.key)
        enabled = rig.service.enable(rig.key, POLICY, expected_scope=revoked.scope)
        release.set()
        with pytest.raises(ValueError, match="stale_scope"):
            await task
        assert rig.service.status(rig.key).scope == enabled.scope
    finally:
        release.set()
        if task is not None and not task.done():
            try:
                await task
            except ValueError:
                pass
        await rig.close()


@private_profile_test
async def test_legacy_nonincremental_cache_clear_refuses_off_without_replacement(
    tmp_path, request
):
    rig = Rig(tmp_path)
    try:
        current = rig.enable()
        store = rig.store(rig.key)
        with store.connection() as conn:
            conn.execute("PRAGMA auto_vacuum=NONE")
            conn.execute("VACUUM")  # Fixture only: emulate an older unreleased cache.
        store.close()
        inode = index_store_path(rig.data, rig.key).stat().st_ino
        with pytest.raises(ValueError, match="index_cleanup_rebuild_required"):
            rig.service.clear(rig.key, expected_scope=current.scope)
        assert rig.service.status(rig.key).state == "Off"
        assert index_store_path(rig.data, rig.key).stat().st_ino == inode
    finally:
        await rig.close()


@private_profile_test
async def test_delayed_old_root_clear_cannot_revoke_new_root_enable(tmp_path, request):
    rig = Rig(tmp_path)
    try:
        old_key = rig.key
        old = rig.enable()
        rig.root = tmp_path / "replacement-root"
        rig.root.mkdir()
        rig.live = authority(rig.root)
        rig.key = rig.live.key
        current = rig.enable()
        with pytest.raises(ValueError, match="authority_unavailable"):
            rig.service.clear(old_key, expected_scope=old.scope)
        assert rig.service.status(rig.key).scope == current.scope
        assert rig.service.status(rig.key).state == "Paused"
    finally:
        await rig.close()


@private_profile_test
async def test_failed_writer_dispatch_becomes_explicitly_resumable(
    tmp_path, request, monkeypatch
):
    rig = Rig(tmp_path)
    start = threading.Thread.start

    def fail_writer(thread):
        if thread.name == "project-index-writer":
            raise RuntimeError("synthetic writer dispatch failed")
        return start(thread)

    try:
        rig.enable()
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", fail_writer)
            with pytest.raises(RuntimeError, match="synthetic writer dispatch"):
                rig.refresh()
        assert rig.service.status(rig.key).state == "Interrupted"
    finally:
        await rig.close()


class AdmissionBarrierScanner(ProjectScanner):
    """Hold before the second real Directory admission, after a committed page."""

    def __init__(self):
        super().__init__()
        self.before_admit = threading.Event()
        self.release = threading.Event()
        self.admissions = []
        self.directory_requests = []

    def scan(self, *args, **kwargs):
        admit = kwargs["admit_unit"]

        def held(directory):
            if self.admissions:
                self.before_admit.set()
                assert self.release.wait(30)
            result = admit(directory)
            self.admissions.append(result)
            return result

        kwargs["admit_unit"] = held
        return super().scan(*args, **kwargs)

    def _unit(self, process, request):
        if request["type"] == "Directory":
            self.directory_requests.append(request["directory"])
        return super()._unit(process, request)


@pytest.mark.parametrize("action", ["pause", "disable", "clear"])
@private_profile_test
async def test_remote_management_denies_next_scanner_admission(
    tmp_path, request, action
):
    from tldw_chatbook.Workspaces.project_index.lifetime import LifetimeBusy

    scanner = AdmissionBarrierScanner()
    rig = Rig(tmp_path, scanner)
    other = type(rig.service)(rig.store, ProjectScanner(), rig.read, rig.fence)
    try:
        (rig.root / "nested").mkdir()
        (rig.root / "nested/secret.txt").write_text("must not read after revocation")
        rig.enable()
        job = rig.refresh()
        assert await asyncio.to_thread(scanner.before_admit.wait, 30)
        current = other.status(rig.key)
        assert scanner.directory_requests == [""]
        if action == "pause":
            other.pause(
                job,
                expected_scope=current.scope,
                expected_job_revision=current.job_revision,
            )
        elif action == "clear":
            # Clear revokes before refusing cleanup of the still-live scanner.
            with pytest.raises(LifetimeBusy):
                other.clear(rig.key, expected_scope=current.scope)
        else:
            other.disable(rig.key, expected_scope=current.scope)
        scanner.release.set()
        await asyncio.to_thread(rig.service._runs[job].supervisor.join)
        assert scanner.admissions[-1] is None
        assert scanner.directory_requests == [""]
        assert scanner.retired
        assert rig.service.status(rig.key).indexed == 0
        if action == "clear":
            off = other.status(rig.key)
            assert other.clear(rig.key, expected_scope=off.scope).state == "Off"
    finally:
        scanner.release.set()
        await other.aclose()
        await rig.close()


@pytest.mark.parametrize("action", ["pause", "disable", "clear"])
@private_profile_test
async def test_delayed_local_stop_cannot_cancel_replacement_job(
    tmp_path, request, monkeypatch, action
):
    from tldw_chatbook.Workspaces.project_index.lifetime import LifetimeBusy

    scanner = BarrierScanner()
    rig = Rig(tmp_path, scanner)
    entered, release, replacement_stopped = (threading.Event() for _ in range(3))
    management = stop_waiter = None
    try:
        (rig.root / "a.txt").write_text("replacement must finish")
        current = rig.enable()
        old_job = rig.refresh()
        assert await asyncio.to_thread(scanner.before_publish.wait, 10)
        current = rig.service.status(rig.key)
        stop_local = rig.service._stop_local

        def delayed_stop(target):
            entered.set()
            assert release.wait(30)
            return stop_local(target)

        monkeypatch.setattr(rig.service, "_stop_local", delayed_stop)

        def operation():
            if action == "pause":
                return rig.service.pause(
                    old_job,
                    expected_scope=current.scope,
                    expected_job_revision=current.job_revision,
                )
            return getattr(rig.service, action)(rig.key, expected_scope=current.scope)

        management = asyncio.create_task(asyncio.to_thread(operation))
        assert await asyncio.to_thread(entered.wait, 30)
        scanner.release.set()
        await asyncio.to_thread(rig.service._runs[old_job].supervisor.join)
        assert rig.service._runs[old_job].native_retired
        scanner.before_publish.clear()
        scanner.release.clear()
        if action == "pause":
            paused = rig.service.status(rig.key)
            new_job = rig.service.resume(
                old_job,
                expected_scope=paused.scope,
                expected_job_revision=paused.job_revision,
            )
        else:
            rig.enable()
            new_job = rig.refresh()
        assert await asyncio.to_thread(scanner.before_publish.wait, 10)
        replacement = rig.service._runs[new_job]
        set_stop = replacement.stop.set

        def observe_stop():
            set_stop()
            replacement_stopped.set()

        monkeypatch.setattr(replacement.stop, "set", observe_stop)
        stop_waiter = asyncio.create_task(
            asyncio.to_thread(replacement_stopped.wait, 30)
        )
        release.set()
        done, _ = await asyncio.wait(
            (management, stop_waiter), timeout=30, return_when=asyncio.FIRST_COMPLETED
        )
        assert done, "management did not finish or target a run"
        assert not replacement.stop.is_set()
        if action == "clear":
            with pytest.raises((LifetimeBusy, ValueError)):
                await management
        else:
            await management
        scanner.release.set()
        assert rig.settle().indexed == 1
    finally:
        release.set()
        scanner.release.set()
        replacement_stopped.set()
        if management is not None:
            await asyncio.gather(management, return_exceptions=True)
        if stop_waiter is not None:
            await stop_waiter
        await rig.close()
