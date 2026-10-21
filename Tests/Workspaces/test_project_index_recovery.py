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
            assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 2
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
                "INSERT INTO directories VALUES(?,?,0)",
                ((f"d{n:03}", "identity") for n in range(257)),
            )
        first = store.pending_directories(scope)
        assert len(first) == 256
        assert store.pending_directories(scope, after=first[-1]) == ("d256",)
    finally:
        store.close()
