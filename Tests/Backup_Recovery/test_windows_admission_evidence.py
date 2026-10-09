"""Native NTFS admission reuse, compared with the unchanged derivation."""

from __future__ import annotations

import os
import subprocess

import pytest

from Tests.windows_custody import unavailable_custody
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Utils.windows_files import WindowsOS

pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows NTFS evidence")


@pytest.fixture
def native_scope(tmp_path, monkeypatch):
    win = WindowsOS()
    parent = tmp_path / "explicit-user-private"
    win.mkdir(parent, 0o700)
    root, selector, data = (parent / n for n in ("bootstrap", "config", "data"))
    fd = win.open(selector, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    win.write(fd, b"scope1")
    win.close(fd)
    win.mkdir(data, 0o700)
    authority = admission_authority(root)
    authority.register("profile", (selector, data))
    bind_profile(root, selector, ("profile",), root / "admission")
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    startup = storage.acquire_storage()
    try:
        yield root, selector, data, startup
    finally:
        startup.close()


def _verdict(path, related_paths=()):
    try:
        with storage.acquire_storage(path, related_paths=related_paths) as lease:
            return "allowed", lease.execution_context(path)[1]
    except bootstrap.RecoveryRequired as error:
        return "refused", str(error)


def _warm(scope):
    root, selector, data, _ = scope
    target = data / "store.db"
    for _ in range(3):
        assert _verdict(target)[0] == "allowed"
    hold = storage._holds[(os.getpid(), str(root))]
    evidence = hold.evidence.get(str(selector))
    assert evidence is not None and evidence.confirmed
    return target, hold, evidence


def test_native_unchanged_admission_reuses_counted_evidence(native_scope, monkeypatch):
    _, _, data, _ = native_scope
    target = data / "store.db"
    # Current Windows runs a full derivation on every call, even after warming.
    for _ in range(3):
        assert _verdict(target)[0] == "allowed"
    calls = []
    original = storage._scope

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "_scope", counted)
    assert _verdict(target)[0] == "allowed"
    assert calls == [], "unchanged native evidence must avoid full scope derivation"


def test_native_reuse_reduces_open_handle_work(native_scope, monkeypatch):
    target, _, _ = _warm(native_scope)
    native = windows_files._native()
    original = native.open_handle
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "open_handle", counted)
    assert _verdict(target)[0] == "allowed"
    reused = len(calls)
    calls.clear()
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
    assert _verdict(target)[0] == "allowed"
    derived = len(calls)
    print(f"NATIVE_ADMISSION_OPEN_COUNTS reuse={reused} derivation={derived}")
    assert reused * 4 < derived


@pytest.mark.parametrize(
    "mutation",
    [
        "acl",
        "owner",
        "selector-in-place",
        "pending-file",
        "intent",
        "directory-replace",
        "junction",
    ],
)
def test_native_mutation_reuse_matches_full_derivation(
    native_scope, monkeypatch, mutation
):
    root, selector, data, _ = native_scope
    target, _, _ = _warm(native_scope)
    win = WindowsOS()
    if mutation == "acl":
        subprocess.run(
            ["icacls", str(data.parent), "/grant", "*S-1-1-0:(M)"],
            check=True,
            capture_output=True,
        )  # nosec B603 B607 -- fixed native tool and private synthetic fixture.
    elif mutation == "owner":
        result = subprocess.run(
            ["icacls", str(data), "/setowner", "*S-1-5-32-544"],
            check=False,
            capture_output=True,
        )  # nosec B603 B607 -- fixed owner SID and synthetic fixture.
        if result.returncode:
            unavailable_custody("token cannot assign the Administrators owner SID")
    elif mutation == "junction":
        target_dir = data.with_name("junction-target")
        win.mkdir(target_dir, 0o700)
        data.rename(data.with_name("old-data"))
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(data), str(target_dir)],
            check=True,
            capture_output=True,
        )  # nosec B603 B607 -- fixed tool and synthetic fixture.
    elif mutation == "selector-in-place":
        before = win.stat(selector)
        selector.write_bytes(b"scope2")
        win.utime(selector, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert win.stat(selector).st_ctime_ns != before.st_ctime_ns
    elif mutation in {"pending-file", "intent"}:
        parent = root if mutation == "pending-file" else root / "admission"
        before = win.stat(parent)
        name = (
            "pending-external.json"
            if mutation == "pending-file"
            else "registry.pending.json"
        )
        fd = win.open(parent / name, win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600)
        win.write(fd, b"{}")
        win.close(fd)
        # A genuine NTFS directory ChangeTime must notice even restored mtime.
        win.utime(parent, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert win.stat(parent).st_ctime_ns != before.st_ctime_ns
    else:
        data.rename(data.with_name("old-data"))
        win.mkdir(data, 0o700)
    try:
        reused = _verdict(target)
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        derived = _verdict(target)
        assert reused == derived
        if mutation in {"acl", "pending-file", "intent"}:
            assert derived[0] == "refused"
    finally:
        if mutation == "acl":
            win.chmod(data.parent, 0o700)


def test_native_change_between_count_and_observation_falls_back(
    native_scope, monkeypatch
):
    root, _, _, _ = native_scope
    target, hold, _ = _warm(native_scope)
    before = hold.count
    original = storage._observe_stamps
    changed = False

    def change_on_observation(*paths):
        nonlocal changed
        if not changed:
            assert hold.count == before + 1
            changed = True
            fd = WindowsOS().open(
                root / "pending-between-count.json",
                WindowsOS().O_CREAT | WindowsOS().O_EXCL | WindowsOS().O_WRONLY,
                0o600,
            )
            WindowsOS().write(fd, b"{}")
            WindowsOS().close(fd)
        return original(*paths)

    monkeypatch.setattr(storage, "_observe_stamps", change_on_observation)
    reused = _verdict(target)
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
    assert reused == _verdict(target)
    assert changed and reused[0] == "refused"
    assert hold.count == before


def test_native_reuse_keeps_pause_and_maintenance_refusals(native_scope, monkeypatch):
    target, hold, _ = _warm(native_scope)
    before = hold.count
    pause = storage._begin_local_pause()
    try:
        assert _verdict(target) == ("refused", "storage_locally_paused")
        assert hold.count == before
    finally:
        pause.resume()
    monkeypatch.setattr(storage._local, "admitted", True, raising=False)
    assert _verdict(target) == ("refused", "maintenance_requires_owner_capability")
    assert hold.count == before


def test_native_acl_edit_changes_posture_even_with_same_projected_permissions(
    native_scope,
):
    _, _, data, _ = native_scope
    before = storage._posture(data)
    # A redundant trusted administrator ACE preserves projected mode/UID;
    # Exact native owner/DACL bytes still distinguish the descriptor edit.
    subprocess.run(
        ["icacls", str(data), "/grant", "*S-1-5-32-544:(R)"],
        check=True,
        capture_output=True,
    )  # nosec B603 B607 -- fixed native tool and private synthetic fixture.
    after = storage._posture(data)
    assert before[:5] == after[:5]
    assert before != after, "fresh native owner/DACL bytes must fence an ACL edit"


def test_native_snapshot_detects_ancestor_replacement_after_child_observation(
    native_scope, monkeypatch
):
    _, _, data, _ = native_scope
    win, native = WindowsOS(), windows_files._native()
    child = data / "snapshot-child"
    fd = win.open(child, win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600)
    win.close(fd)
    original_open, original_close = native.open_handle, native.kernel.CloseHandle
    child_opens = 0
    observed_child_handle = None
    mutated = False

    def remember_child_validation(name, *args, **kwargs):
        nonlocal child_opens, observed_child_handle
        handle = original_open(name, *args, **kwargs)
        if name == child.name:
            child_opens += 1
            if child_opens == 2:
                observed_child_handle = handle
        return handle

    def replace_after_child_validation(handle):
        nonlocal mutated
        result = original_close(handle)
        if handle == observed_child_handle and not mutated:
            mutated = True
            win.rename(data, data.with_name("old-pinned-data"))
            win.mkdir(data, 0o700)
        return result

    monkeypatch.setattr(native, "open_handle", remember_child_validation)
    monkeypatch.setattr(native.kernel, "CloseHandle", replace_after_child_validation)
    with pytest.raises(OSError, match="windows_admission_snapshot_changed"):
        win.stat_many_for_admission((data.parent, data, child))
    assert mutated


def test_native_related_evidence_shares_one_fresh_ancestor_walk(
    native_scope, monkeypatch
):
    _, _, data, _ = native_scope
    target = data / "store.db"
    related = tuple(
        data / name
        for name in ("lock", "backup", "temporary", "model", "dictionary", "selector")
    )
    for _ in range(3):
        assert _verdict(target, related)[0] == "allowed"
    native = windows_files._native()
    original_open, original_scope = native.open_handle, storage._scope
    calls, derivations = [], []

    def counted_open(*args, **kwargs):
        calls.append(True)
        return original_open(*args, **kwargs)

    def counted_scope(*args, **kwargs):
        derivations.append(True)
        return original_scope(*args, **kwargs)

    monkeypatch.setattr(native, "open_handle", counted_open)
    monkeypatch.setattr(storage, "_scope", counted_scope)
    assert _verdict(target)[0] == "allowed"
    single = len(calls)
    calls.clear()
    assert _verdict(target, related)[0] == "allowed"
    grouped = len(calls)
    print(
        f"NATIVE_RELATED_EVIDENCE_OPEN_COUNTS single={single} grouped={grouped} related={len(related)}"
    )
    assert not derivations
    # Each additional absent sibling needs fresh descent and validation only.
    assert grouped <= single + 2 * len(related)


@pytest.mark.parametrize("change", ["related-appeared", "unreadable-observation"])
def test_native_related_change_after_count_rederives_without_leaking(
    native_scope, monkeypatch, change
):
    root, _, data, _ = native_scope
    target, related = data / "store.db", data / "related-control"
    for _ in range(3):
        assert _verdict(target, (related,))[0] == "allowed"
    hold = storage._holds[(os.getpid(), str(root))]
    before = hold.count
    original_observe, original_scope = storage._observe_stamps, storage._scope
    observed, derivations = [], []

    def changed_observation(*paths):
        if not observed:
            assert hold.count == before + 1
            observed.append(True)
            if change == "unreadable-observation":
                raise PermissionError("optional native evidence is unreadable")
            win = WindowsOS()
            fd = win.open(related, win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600)
            win.close(fd)
        return original_observe(*paths)

    def full_derivation(*args, **kwargs):
        derivations.append(True)
        return original_scope(*args, **kwargs)

    monkeypatch.setattr(storage, "_observe_stamps", changed_observation)
    monkeypatch.setattr(storage, "_scope", full_derivation)
    reused = _verdict(target, (related,))
    assert observed and derivations
    assert hold.count == before
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
    assert reused == _verdict(target, (related,))
    assert hold.count == before
