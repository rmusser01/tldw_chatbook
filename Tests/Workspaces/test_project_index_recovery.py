"""Native lifetime proofs survive both the parent and early child bootstrap."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from Tests.Backup_Recovery.conftest import helper_resource_root  # noqa: F401


def lifetime_module():
    return importlib.import_module("tldw_chatbook.Workspaces.project_index.lifetime")


@pytest.mark.parametrize("early", [True, False])
def test_inherited_capability_blocks_replacement_until_native_child_exit(
    tmp_path, early
):
    mod = lifetime_module()
    path = tmp_path / "lifetime.lock"
    lease = mod.BindingLifetime.acquire(path)
    # The child holds only the inherited OS capability, before importing any
    # application code or completing a scanner handshake.
    code = "import sys; print('alive',flush=True); sys.stdin.buffer.read(1)"
    child = lease.spawn(
        [getattr(sys, "_base_executable", sys.executable), "-I", "-S", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        if not early:
            assert child.stdout.readline().strip() == b"alive"
        lease.close()
        with pytest.raises(mod.LifetimeBusy):
            mod.BindingLifetime.acquire(path)
        child.stdin.write(b"x")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        acquired = mod.BindingLifetime.acquire(path)
        acquired.close()
        assert path.exists()
    finally:
        lease.close()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()


def test_lifetime_refuses_alias_and_is_stable_across_release(tmp_path):
    mod = lifetime_module()
    path = tmp_path / "lifetime.lock"
    lease = mod.BindingLifetime.acquire(path)
    stamp = path.stat()
    with pytest.raises(mod.LifetimeBusy):
        mod.BindingLifetime.acquire(path)
    lease.close()
    second = mod.BindingLifetime.acquire(path)
    second.close()
    assert path.stat().st_ino == stamp.st_ino


@pytest.mark.parametrize("child_started", [False, True])
def test_hard_parent_death_does_not_release_child_inherited_ownership(
    tmp_path, child_started
):
    mod = lifetime_module()
    path = tmp_path / "lifetime.lock"
    release = tmp_path / "release"
    repo = Path(__file__).resolve().parents[2]
    child_code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); print('child',flush=True);\nwhile not p.exists(): time.sleep(.01)"
    parent_code = f"""import sys,subprocess
sys.path.insert(0,{str(repo)!r})
from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime
from pathlib import Path
lease=BindingLifetime.acquire(Path({str(path)!r}))
child=lease.spawn([sys.executable,'-I','-S','-c',{child_code!r},{str(release)!r}],stdout=subprocess.PIPE,creationflags={4 if os.name == 'nt' and not child_started else 0})
print(child.pid,flush=True)
if {child_started!r}:
    child.stdout.readline()
print('created',flush=True)
sys.stdin.buffer.read(1)
"""
    parent = subprocess.Popen(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-I",
            "-S",
            "-c",
            parent_code,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    handle = None
    try:
        child_pid = int(parent.stdout.readline())
        assert parent.stdout.readline().strip() == b"created"
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel.OpenProcess.restype = wintypes.HANDLE
            handle = kernel.OpenProcess(0x100001, False, child_pid)
            assert handle
        parent.kill()
        parent.wait(timeout=10)
        with pytest.raises(mod.LifetimeBusy):
            mod.BindingLifetime.acquire(path)
        release.touch()
        if handle is not None:
            if not child_started:
                assert kernel.TerminateProcess(wintypes.HANDLE(handle), 1)
            assert kernel.WaitForSingleObject(wintypes.HANDLE(handle), 10000) == 0
        else:
            import time

            deadline = time.monotonic() + 10
            while True:
                try:
                    acquired = mod.BindingLifetime.acquire(path)
                    acquired.close()
                    break
                except mod.LifetimeBusy:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
        acquired = mod.BindingLifetime.acquire(path)
        acquired.close()
    finally:
        release.touch()
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        parent.stdin.close()
        parent.stdout.close()
        if handle is not None:
            kernel.CloseHandle(wintypes.HANDLE(handle))


# Reuse the existing real encrypted-recovery fixture; no alternate parser or
# restoration simulation is involved in this identity qualification.


def test_actual_restore_and_rollback_issue_distinct_verified_generations(
    tmp_path,
    monkeypatch,
    helper_resource_root,  # noqa: F811
):  # noqa: F811
    from threading import Event
    from Tests.Backup_Recovery.test_later_rollback import _completed
    from tldw_chatbook.Backup_Recovery.isolated_restore import verified_data_identity
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
    from tldw_chatbook.Backup_Recovery import recovery_copies

    with _completed(tmp_path, monkeypatch, helper_resource_root) as (
        case,
        operation,
        original,
    ):
        restored = verified_data_identity()
        assert restored[1] != "ordinary"
        control = tmp_path / "control"
        approved = preview_rollback(
            operation,
            control_root=control,
            old_password=b"original",
            target=case[1].target,
            cancel=Event(),
        )
        recovery_copies.rollback(
            operation,
            control_root=control,
            old_password=b"original",
            new_password=b"new",
            cancel=Event(),
            approved_plan=approved,
        )
        rolled_back = verified_data_identity()
        assert rolled_back[1] != restored[1]
        assert rolled_back[1] != "ordinary"


def test_native_lifetime_refuses_hardlinked_capability(tmp_path):
    path = tmp_path / "original.lock"
    lease = lifetime_module().BindingLifetime.acquire(path)
    lease.close()
    alias = tmp_path / "alias.lock"
    os.link(path, alias)
    with pytest.raises(ValueError, match="lifetime_alias_refused"):
        lifetime_module().BindingLifetime.acquire(alias)


@pytest.mark.parametrize("legacy_claim", [False, True])
def test_v1_cache_migration_preserves_control_and_quarantines_unknown_claim(
    tmp_path, legacy_claim
):
    from Tests.Workspaces.project_index_support import open_index_store, make_index_key
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy

    store = open_index_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.configure(
        key,
        ProjectIndexPolicy((), ()),
        enabled=True,
        expected_scope=store.load_status(key).scope,
    )
    if legacy_claim:
        store.claim_job(
            scope,
            owner_token="old-writer",
            authority_revision=1,
            expected_job_revision=store.load_status(key).job_revision,
        )
    with store.transaction() as conn:
        conn.execute("DROP TABLE chunk_fts")
        conn.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(text)")
        conn.execute("DROP INDEX directories_eligible")
        conn.execute("ALTER TABLE directories DROP COLUMN attempt_revision")
        for table in (
            "authority_witness",
            "scan_seen",
            "legacy_retirement",
            "schema_version",
        ):
            conn.execute(f"DROP TABLE {table}")
        conn.execute(
            "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=1))"
        )
        conn.execute("INSERT INTO schema_version VALUES(1)")
    store.close()
    reopened = open_index_store(tmp_path)
    try:
        assert reopened.load_status(key).scope == scope
        with reopened.connection() as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 3
        if legacy_claim:
            with pytest.raises(ValueError, match="legacy_retirement_unproven"):
                reopened.assert_native_lifetime()
        else:
            reopened.assert_native_lifetime()
    finally:
        reopened.close()


def test_pending_frontier_continues_past_full_incomplete_page(tmp_path):
    from Tests.Workspaces.project_index_support import open_index_store, make_index_key

    store = open_index_store(tmp_path)
    try:
        scope = store.load_status(make_index_key(tmp_path)).scope
        with store.transaction() as conn:
            conn.executemany(
                "INSERT INTO directories(path,identity,complete) VALUES(?,?,0)",
                ((f"d{n:03}", "identity") for n in range(257)),
            )
        first = store.pending_directories(scope)
        assert len(first) == 256
        assert store.pending_directories(scope, after=first[-1]) == ("d256",)
    finally:
        store.close()


def _held_query_recovery_child(base):
    """Fresh selected-profile process; imports runtime only after real publication."""
    import asyncio
    import json
    import threading
    import time
    from types import SimpleNamespace
    import psutil
    from tldw_chatbook import config
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.project_index.runtime import (
        get_project_index_service,
        project_index_key,
    )
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _retire_current_thread_caches
    from Tests.Agents.test_project_index_tool import provider, invoke

    live = config.get_user_data_dir()
    assert live.is_relative_to(base / "live")
    root = base / "query-project"
    root.mkdir()
    (root / "marker.txt").write_text("rollbackpinmarker", encoding="utf-8")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(live / "workspaces.db"))
    registry.create_workspace(workspace_id="query-proof", name="Query proof")
    binding = registry.add_folder_binding("query-proof", root)
    app = SimpleNamespace(workspace_registry_service=registry)
    service = get_project_index_service(app)
    key = project_index_key(app, "query-proof", binding.binding_id, root)
    off = service.status(key)
    enabled = service.enable(
        key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=off.scope
    )
    job = service.refresh(
        key, expected_scope=enabled.scope, expected_job_revision=enabled.job_revision
    )
    service._runs[job].supervisor.join(60)
    ready = service.status(key)
    assert ready.state == "Ready", repr(service._runs[job].error)
    tool = provider(SimpleNamespace(service=service, root=root, key=key))
    entered, release = threading.Event(), threading.Event()
    capture = []
    unit = ProjectScanner._unit

    def held(self, process, command):
        result = unit(self, process, command)
        if command["type"] == "Capture":
            capture.append(self)
            entered.set()
            assert release.wait(30)
        return result

    ProjectScanner._unit = held
    output = []

    def query_call():
        output.append(invoke(tool, {"query": "rollbackpinmarker"}))

    query = threading.Thread(target=query_call)
    pause = None
    try:
        query.start()
        assert entered.wait(15)
        (search,) = tuple(service._searches)
        assert search.pin is not None and not search.pin.closed
        scanner = capture[0]
        assert psutil.pid_exists(scanner.worker_pid) and not scanner.retired
        pause = storage._begin_local_pause()
        _retire_current_thread_caches(pause)
        assert not pause.drain(time.monotonic()), "real query lost its storage pin"
        release.set()
        query.join(15)
        assert not query.is_alive()
        assert scanner.retired and scanner.pipes_closed
        assert not psutil.pid_exists(scanner.worker_pid)
        assert search.pin.closed and search.done.is_set()
        if not pause.drain(time.monotonic() + 3):
            leases = [
                {
                    "path": str(item.path),
                    "owner": item.owner_id,
                    "connections": [
                        str(lease.resource_thread)
                        for lease in item.connections.values()
                    ],
                }
                for item in __import__(
                    "tldw_chatbook.Backup_Recovery.participants",
                    fromlist=["_installed_repositories"],
                )._installed_repositories
                if item.connections
            ]
            (base / "remaining-leases.json").write_text(
                json.dumps(leases, indent=2), encoding="utf-8"
            )
            raise AssertionError(leases)
        pause.resume()
        pause = None
        ProjectScanner._unit = unit
        registry.db.close()
        (base / "query-retired.json").write_text(
            json.dumps(
                {
                    "key": __import__("dataclasses").asdict(key),
                    "scope": __import__("dataclasses").asdict(ready.scope),
                    "held_capture_pid": scanner.worker_pid,
                    "worker_gone": True,
                    "pin_blocked_drain": True,
                    "pin_closed_after_native_exit": True,
                    "pipes_closed": scanner.pipes_closed,
                    "drained_after_retirement": True,
                }
            ),
            encoding="utf-8",
        )
    finally:
        release.set()
        query.join(15)
        if pause is not None:
            pause.resume()
        asyncio.run(service.aclose())
        registry.db.close()


def _old_key_recovery_child(base):
    """A new admitted process refuses the key saved before actual rollback."""
    import asyncio
    import json
    from types import SimpleNamespace
    from tldw_chatbook import config
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.project_index.runtime import get_project_index_service
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexKey,
        ProjectIndexScope,
    )

    receipt = json.loads((base / "query-retired.json").read_text(encoding="utf-8"))
    key = ProjectIndexKey(**receipt["key"])
    scope = ProjectIndexScope(
        key, receipt["scope"]["epoch"], receipt["scope"]["processing_revision"]
    )
    live = config.get_user_data_dir()
    assert live.is_relative_to(base / "live")
    registry = LocalWorkspaceRegistryService(WorkspaceDB(live / "workspaces.db"))
    service = get_project_index_service(
        SimpleNamespace(workspace_registry_service=registry)
    )
    try:
        assert service._factory.identity != key.data_authority_id
        for operation in (lambda: service.status(key), lambda: service.policy(scope)):
            try:
                operation()
            except ValueError as error:
                assert str(error) == "index_data_authority_changed"
            else:
                raise AssertionError(
                    "old key regained authority after encrypted rollback"
                )
        (base / "old-key-refused").touch()
    finally:
        asyncio.run(service.aclose())
        registry.db.close()


def test_real_query_pin_blocks_rollback_until_capture_native_retirement(
    tmp_path,
    monkeypatch,
    helper_resource_root,  # noqa: F811 - imported fixture
):
    import threading
    import time
    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )
    from tldw_chatbook.Backup_Recovery import crypto, credentials, replacement
    from tldw_chatbook.Backup_Recovery import bootstrap, recovery_copies
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore
    from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
    from tldw_chatbook.Backup_Recovery.isolated_restore import verified_data_identity
    from tldw_chatbook.Workspaces.project_index.models import OWNER_ID

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    credential_store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: credential_store)

    def existing_data(live):
        (live / "data/Local").mkdir(parents=True)
        return ()

    # The fixture normally supplies a deliberately corrupt original config.
    # Install valid original bytes before its REAL planner fingerprints them.
    import Tests.Backup_Recovery.test_held_sqlite_rollback as fixture_module

    original_plan = fixture_module.plan_restore

    def plan_with_valid_original(*args, **kwargs):
        selector = next(
            item.path for item in kwargs["target"].items if item.owner == "config"
        )
        selector.write_text(
            '[general]\nusers_name="Local"\n[paths]\ndata_dir="'
            + (tmp_path / "live/data").as_posix()
            + '"\n',
            encoding="utf-8",
        )
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(fixture_module, "plan_restore", plan_with_valid_original)
    with replacement_case(
        tmp_path, monkeypatch, prepared=False, extras=existing_data
    ) as case:
        operation = replacement.replace(
            case[1],
            case[0],
            control_root=tmp_path / "control",
            rollback_password=b"original",
            cancel=threading.Event(),
        )
        approvals = []

        def review_index_owner():
            _, profiles, _ = bootstrap._control_records(
                bootstrap.default_bootstrap_root()
            )
            profile = next(p for p in profiles if p["selector"] == str(case[5]))
            witness = profile["activation"]
            assert OWNER_ID in witness["owners"]
            store = ActivationStore(Path(witness["store_root"]))
            assert not store.allowed(witness["generation"], OWNER_ID)
            store.approve(witness["generation"], OWNER_ID)
            assert store.allowed(witness["generation"], OWNER_ID)
            approvals.append(witness)

        review_index_owner()
        script = (
            "from pathlib import Path; from tldw_chatbook.Backup_Recovery import bootstrap; bootstrap.default_bootstrap_root=lambda:Path("
            + repr(str(tmp_path / "bootstrap"))
            + "); from Tests.Workspaces.test_project_index_recovery import _held_query_recovery_child; _held_query_recovery_child(Path("
            + repr(str(tmp_path))
            + "))"
        )
        with (tmp_path / "query-child.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-c", script], stdout=log, stderr=log
            )
            try:
                deadline = time.monotonic() + 100
                while not (tmp_path / "query-retired.json").exists():
                    assert child.poll() is None, (
                        tmp_path / "query-child.log"
                    ).read_text(encoding="utf-8")
                    assert time.monotonic() < deadline
                    time.sleep(0.05)
                assert child.wait(30) == 0, (tmp_path / "query-child.log").read_text(
                    encoding="utf-8"
                )
                old_generation = verified_data_identity()
                approved = preview_rollback(
                    operation,
                    control_root=tmp_path / "control",
                    old_password=b"original",
                    target=case[1].target,
                    cancel=threading.Event(),
                )
                recovery_copies.rollback(
                    operation,
                    control_root=tmp_path / "control",
                    old_password=b"original",
                    new_password=b"new",
                    cancel=threading.Event(),
                    approved_plan=approved,
                )
                assert verified_data_identity() != old_generation
                review_index_owner()
                verify_script = script.replace(
                    "_held_query_recovery_child", "_old_key_recovery_child"
                )
                checked = subprocess.run(
                    [sys.executable, "-c", verify_script],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                (tmp_path / "old-key-child.log").write_text(
                    checked.stdout + checked.stderr, encoding="utf-8"
                )
                assert checked.returncode == 0, checked.stdout + checked.stderr
                assert (tmp_path / "old-key-refused").exists()
                assert approvals[0]["generation"] != approvals[1]["generation"]
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(10)
