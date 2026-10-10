"""Change-notified Windows admission evidence reuse (TASK-34601 ADR-126 amendment).

A warm acquisition may skip the full re-observation only while notifications
armed before the confirming observation stay quiet. Every test first proves the
fast path is really taken, then that each change, failure or limit returns to
the original full observation with the full derivation's verdict.
"""

from __future__ import annotations

import ctypes as C
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from Tests.Backup_Recovery import test_windows_admission_evidence as oracle
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Utils.windows_files import WindowsOS

pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows NTFS evidence")

# The oracle's real-NTFS profile fixture and helpers, shared unchanged.
native_scope = oracle.native_scope
_verdict = oracle._verdict
_warm = oracle._warm


def _count_observations(patch):
    """Record path counts of every stamp observation and every qualification walk."""
    observed, qualified = [], []
    stamps, qualify = storage._observe_stamps, storage.qualified_for

    def counted_stamps(posture_paths, content_paths, links=None):
        observed.append(len(set(posture_paths) | set(content_paths)))
        return stamps(posture_paths, content_paths, links)

    def counted_qualify(*args, **kwargs):
        qualified.append(True)
        return qualify(*args, **kwargs)

    patch.setattr(storage, "_observe_stamps", counted_stamps)
    patch.setattr(storage, "qualified_for", counted_qualify)
    return observed, qualified


def _verified(hold, evidence):
    """The hold's verified watch over ``evidence`` (selector evidence first)."""
    for watch in hold.watches.values():
        if (
            watch.entries[0] is evidence
            and watch.watch is not None
            and watch.verified_at is not None
        ):
            return watch
    return None


def _watched(scope):
    """Warm, then one more reuse so a verified watch covers the evidence."""
    target, hold, evidence = _warm(scope)
    assert _verdict(target)[0] == "allowed"
    assert _verified(hold, evidence) is not None, "arming must succeed and verify"
    return target, hold, evidence


def _drop_watches(hold):
    """Retire every watch of ``hold`` so the next full observation arms afresh."""
    with storage._lock:
        closable = storage._retire_watches(hold)
    storage._close_retired_watches(closable)


def _root_count(target):
    return len({Path(target.anchor), Path(storage._QUALIFICATION_FILE.anchor)})


def _assert_fast(target, monkeypatch):
    # A scoped context: undoing the whole monkeypatch would drop the fixture's
    # bootstrap-root and selector patches too.
    with monkeypatch.context() as patch:
        observed, qualified = _count_observations(patch)
        assert _verdict(target)[0] == "allowed"
    # Every content drive root (which no parent can watch) is re-stamped.
    assert observed == [_root_count(target)] and qualified == []


def _assert_full(target, monkeypatch):
    with monkeypatch.context() as patch:
        observed, _ = _count_observations(patch)
        verdict = _verdict(target)
    assert observed and max(observed) > _root_count(
        target
    ), "this change must be re-observed"
    return verdict


def _derived(target, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(storage, "_EVIDENCE_REUSE", False)
        return _verdict(target)


def test_quiet_watch_skips_full_observation_and_qualification(
    native_scope, monkeypatch
):
    target, _, _ = _watched(native_scope)
    for _ in range(3):
        _assert_fast(target, monkeypatch)


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
def test_notified_mutation_returns_to_full_observation_with_derived_verdict(
    native_scope, monkeypatch, mutation
):
    root, selector, data, _ = native_scope
    target, _, _ = _watched(native_scope)
    _assert_fast(target, monkeypatch)
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
            pytest.skip("token cannot assign the Administrators owner SID")
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
    elif mutation in {"pending-file", "intent"}:
        parent = root if mutation == "pending-file" else root / "admission"
        name = (
            "pending-external.json"
            if mutation == "pending-file"
            else "registry.pending.json"
        )
        fd = win.open(parent / name, win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600)
        win.write(fd, b"{}")
        win.close(fd)
    else:
        data.rename(data.with_name("old-data"))
        win.mkdir(data, 0o700)
    try:
        reused = _assert_full(target, monkeypatch)
        assert reused == _derived(target, monkeypatch)
    finally:
        if mutation == "acl":
            win.chmod(data.parent, 0o700)


def test_change_reported_during_root_restamp_falls_back(native_scope, monkeypatch):
    root, _, _, _ = native_scope
    target, hold, _ = _watched(native_scope)
    before = hold.count
    original = storage._observe_stamps
    changed = []

    def change_on_root_restamp(posture_paths, content_paths, links=None):
        if (
            not changed
            and posture_paths
            and all(path.parent == path for path in posture_paths)
            and not content_paths
        ):
            assert hold.count == before + 1  # counted before the final check
            changed.append(True)
            win = WindowsOS()
            fd = win.open(
                root / "pending-during-restamp.json",
                win.O_CREAT | win.O_EXCL | win.O_WRONLY,
                0o600,
            )
            win.write(fd, b"{}")
            win.close(fd)
        return original(posture_paths, content_paths, links)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_observe_stamps", change_on_root_restamp)
        reused = _verdict(target)
    assert changed and reused == _derived(target, monkeypatch)
    assert reused[0] == "refused"
    assert hold.count == before


def test_change_just_after_confirming_observation_is_not_trusted(
    native_scope, monkeypatch
):
    """Arm-before-observe: a change after the confirming observation is seen."""
    root, _, _, _ = native_scope
    target, hold, _ = _warm(native_scope)
    _drop_watches(hold)  # the next call must ARM, then confirm
    original = storage._observe_evidence
    changed = []

    def change_after_observation(entries, links=None):
        result = original(entries, links)
        if links is not None and not changed:
            changed.append(True)
            win = WindowsOS()
            fd = win.open(
                root / "pending-after-observe.json",
                win.O_CREAT | win.O_EXCL | win.O_WRONLY,
                0o600,
            )
            win.write(fd, b"{}")
            win.close(fd)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_observe_evidence", change_after_observation)
        _verdict(target)  # arms, observes; the change lands after the observation
    assert changed
    reused = _verdict(target)
    assert reused == _derived(target, monkeypatch) and reused[0] == "refused"


def test_in_process_by_id_write_invalidates_the_watch(native_scope, monkeypatch):
    """Facade writes through by-id handles notify nothing; they bump a generation."""
    _, selector, _, _ = native_scope
    target, _, _ = _watched(native_scope)
    _assert_fast(target, monkeypatch)
    WindowsOS().chmod(selector, 0o600)  # reopens the selector by id with WRITE_DAC
    _assert_full(target, monkeypatch)


def _write_by_id(path, data):
    """Write ``path`` through a raw OpenFileById handle, outside the facade."""
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.WriteFile.argtypes = [
        C.c_void_p,
        C.c_char_p,
        C.c_uint32,
        C.POINTER(C.c_uint32),
        C.c_void_p,
    ]
    native = windows_files._native()
    with windows_files._parent(str(path)) as (parent, leaf):
        pinned = native.open_handle(leaf, parent=parent, metadata=True)
    try:
        info = native.info(pinned)
        descriptor = windows_files._FileIdDescriptor(
            C.sizeof(windows_files._FileIdDescriptor),
            0,
            (C.c_uint64 * 2)((info.index_high << 32) | info.index_low, 0),
        )
        handle = native.kernel.OpenFileById(
            pinned, C.byref(descriptor), 0x40000000, 7, None, 0
        )
        assert handle not in {None, C.c_void_p(-1).value}, C.get_last_error()
        try:
            written = C.c_uint32()
            assert kernel.WriteFile(handle, data, len(data), C.byref(written), None)
        finally:
            native.kernel.CloseHandle(handle)
    finally:
        native.kernel.CloseHandle(pinned)


def test_external_by_id_write_is_bounded_by_the_backstop(native_scope, monkeypatch):
    """Accepted gap: another process's by-id write is seen after <= the backstop."""
    root, _, _, _ = native_scope
    target, hold, _ = _watched(native_scope)
    _write_by_id(root / "admission" / "registry.json", b"XXXXXXXX")
    time.sleep(storage._EVIDENCE_WATCH_BACKSTOP_S + 0.05)
    reused = _assert_full(target, monkeypatch)
    assert reused == _derived(target, monkeypatch) and reused[0] == "refused"
    # The full observation that found the change left no watch verified.
    assert all(watch.verified_at is None for watch in hold.watches.values())


def test_watch_armed_on_an_exited_thread_still_serves(native_scope, monkeypatch):
    """Notification reads belong to a persistent issuer, not the arming thread."""
    target, hold, evidence = _warm(native_scope)
    _drop_watches(hold)  # the worker below must be the one that arms

    def arm_and_exit():
        assert _verdict(target)[0] == "allowed"

    worker = threading.Thread(target=arm_and_exit)
    worker.start()
    worker.join()
    assert _verified(hold, evidence) is not None
    _assert_fast(target, monkeypatch)


def test_alternating_paths_keep_their_own_watches(native_scope, monkeypatch):
    _, _, data, _ = native_scope
    WindowsOS().mkdir(data / "sub", 0o700)
    first, second = data / "a.db", data / "sub" / "b.db"
    for _ in range(4):
        assert _verdict(first)[0] == "allowed"
        assert _verdict(second)[0] == "allowed"
    for _ in range(3):
        _assert_fast(first, monkeypatch)
        _assert_fast(second, monkeypatch)


def test_hard_linked_content_is_never_served_from_notifications(
    native_scope, monkeypatch
):
    _, selector, data, _ = native_scope
    os.link(selector, data / "selector-alias.toml")
    target, hold, evidence = _warm(native_scope)
    for _ in range(3):
        assert _verdict(target)[0] == "allowed"
        _assert_full(target, monkeypatch)
    assert _verified(hold, evidence) is None


def test_backstop_expiry_forces_full_observation(native_scope, monkeypatch):
    target, _, _ = _watched(native_scope)
    with monkeypatch.context() as patch:
        patch.setattr(storage, "_EVIDENCE_WATCH_BACKSTOP_S", 0.0)
        observed, qualified = _count_observations(patch)
        assert _verdict(target)[0] == "allowed"
    assert max(observed) > _root_count(target) and qualified


def test_arm_failure_keeps_full_observation_without_retrying(native_scope, monkeypatch):
    attempts = []

    def refuse(_directories):
        attempts.append(True)
        raise OSError("controlled arm failure")

    monkeypatch.setattr(windows_files, "DirectoryWatch", refuse)
    target, hold, _ = _warm(native_scope)
    armed = len(attempts)
    for _ in range(3):
        _assert_full(target, monkeypatch)
    assert armed >= 1 and len(attempts) == armed, "a failed arm must not retry"
    assert all(watch.watch is None for watch in hold.watches.values())


def test_pause_releases_watch_handles_and_resume_rearms(native_scope, monkeypatch):
    target, hold, evidence = _watched(native_scope)
    native_watch = _verified(hold, evidence).watch
    pause = storage._begin_local_pause()
    try:
        assert not hold.watches and native_watch._closed
        assert _verdict(target) == ("refused", "storage_locally_paused")
    finally:
        pause.resume()
    assert _verdict(target)[0] == "allowed"  # full observation re-arms
    assert all(watch.watch is not native_watch for watch in hold.watches.values())


def test_hold_retirement_closes_watch_handles(native_scope):
    root, _, _, startup = native_scope
    _, hold, evidence = _watched(native_scope)
    native_watch = _verified(hold, evidence).watch
    startup.close()
    assert storage._holds.get((os.getpid(), str(root))) is None
    assert native_watch._closed


def test_separate_qualification_drive_is_watched_and_restamped(
    native_scope, monkeypatch, capsys
):
    """Use the real installed policy file and a private fixture on another drive."""
    root, _, _, startup = native_scope
    fixture_anchor = Path(root.anchor)
    policy_anchor = Path(storage._QUALIFICATION_FILE.anchor)
    with capsys.disabled():
        print(
            "\n::notice::Native watch fixture drive letters: "
            f"profile={root.drive[:1].upper()}; "
            f"qualification={storage._QUALIFICATION_FILE.drive[:1].upper()}"
        )
    if fixture_anchor == policy_anchor:
        pytest.skip(
            "Installed qualification source and private fixture are on the same drive"
        )
    native_watches = []
    handles = []
    try:
        target, hold, evidence = _warm(native_scope)
        roots = {path for path, _ in evidence.posture if path.parent == path}
        before = hold.count
        with monkeypatch.context() as patch:
            observed, qualified = _count_observations(patch)
            assert _verdict(target)[0] == "allowed"
        assert hold.count == before
        verified = _verified(hold, evidence)
        native_watches = [item.watch for item in hold.watches.values() if item.watch]
        handles = [
            (watch._native.kernel, slot[0], slot[1])
            for watch in native_watches
            for slot in watch._slots
        ]
        for kernel, directory, event in handles:
            for handle in (directory, event):
                flags = C.c_uint32()
                assert kernel.GetHandleInformation(C.c_void_p(handle), C.byref(flags))
    finally:
        startup.close()
    assert storage._holds.get((os.getpid(), str(root))) is None
    assert all(watch._closed for watch in native_watches)
    for kernel, directory, event in handles:
        for handle in (directory, event):
            flags = C.c_uint32()
            C.set_last_error(0)
            assert not kernel.GetHandleInformation(C.c_void_p(handle), C.byref(flags))
            assert C.get_last_error() == 6  # ERROR_INVALID_HANDLE
    assert policy_anchor in roots, "The actual policy drive needs posture evidence"
    assert verified is not None and handles, "Original watches must arm and verify"
    assert observed == [len(roots)] and qualified == []
