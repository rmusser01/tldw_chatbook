"""Stage2 metadata-only native contracts; absent APIs are capability failures.

Original counter/TokenOwner and bootstrap tree-refusal controls remain separate.
These tests never enable admission reuse or modify production security policy.
"""

import ctypes as C
import os

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Utils.windows_files import WindowsOS

pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="real Windows metadata prerequisite"
)


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


@pytest.mark.parametrize("ancestor", ["NUL", "bad:stream", "trailing.", "bad\x01name"])
def test_snapshot_refuses_invalid_ancestor_before_native_open(
    tmp_path, monkeypatch, ancestor
):
    def unexpected_native():
        pytest.fail("invalid ancestor reached the native filesystem")

    monkeypatch.setattr(windows_files, "_native", unexpected_native)
    selected = tmp_path / ancestor / "valid-leaf.bin"
    with pytest.raises(ValueError, match="(?:invalid|reserved)_windows_component"):
        windows_files.WindowsOS().stat_many_for_admission((selected,))


@pytest.mark.parametrize(
    "outcome", ["before-close-false", "after-close-false", "raised", "native-protected"]
)
def test_fresh_snapshot_retains_exact_uncertain_close_incarnation(
    tmp_path, monkeypatch, outcome
):
    win, native = WindowsOS(), windows_files._native()
    selected = tmp_path / "uncertain-close.bin"
    selected.write_bytes(b"private metadata")
    original_open = native.open_handle
    original_close = native.kernel.CloseHandle
    kernel = C.WinDLL("kernel32", use_last_error=True)
    protect = kernel.SetHandleInformation
    protect.argtypes = [windows_files._HANDLE, windows_files._U32, windows_files._U32]
    protect.restype = windows_files._I32
    births = []
    current = {}
    captured = []
    retries = []
    native_failures = []
    failure = RuntimeError("controlled close failure")

    def opened(name, *args, **kwargs):
        handle = original_open(name, *args, **kwargs)
        assert handle not in current or not current[handle]["live"]
        info = native.info(handle)
        birth = {
            "handle": handle,
            "identity": (info.volume, (info.index_high << 32) | info.index_low),
            "name": name,
            "live": True,
            "attempts": [],
            "protected": False,
        }
        births.append(birth)
        current[handle] = birth
        if name == selected.name:
            captured.append(birth)
            if outcome == "native-protected":
                native.check(protect(handle, 2, 2))
                birth["protected"] = True
        return handle

    def close(handle):
        birth = current.get(handle)
        if birth is not None:
            birth["attempts"].append(handle)
        if captured and birth is captured[0]:
            if len(birth["attempts"]) > 1:
                # Reject blind retry before it can close a reused numeric value.
                retries.append(birth)
                C.set_last_error(5)
                return 0
            if outcome == "raised":
                raise failure
            if outcome == "after-close-false":
                assert original_close(handle)
                birth["live"] = False
            if outcome != "native-protected":
                C.set_last_error(5)
                return 0
        # Native-protected deliberately reaches the genuine failing Win32 call.
        result = original_close(handle)
        error = C.get_last_error() if not result else 0
        if (
            outcome == "native-protected"
            and captured
            and birth is captured[0]
            and not result
        ):
            native_failures.append(error)
        if result and birth is not None:
            birth["live"] = False
        return result

    monkeypatch.setattr(native, "open_handle", opened)
    monkeypatch.setattr(native.kernel, "CloseHandle", close)
    try:
        with pytest.raises(windows_files._AdmissionMetadataCloseError) as caught:
            win.stat_many_for_admission((selected,))
        assert len(captured) == 1 and not retries
        birth = captured[0]
        assert len(birth["attempts"]) == 1
        assert current[birth["handle"]] is birth
        assert len(caught.value.failed_handles) == 1
        retained = caught.value.failed_handles[0]
        assert retained.native is native and retained.handle == birth["handle"]
        assert retained.identity == birth["identity"] and retained.path == selected
        if outcome == "raised":
            assert retained.close_error is failure
        elif outcome == "native-protected":
            assert len(native_failures) == 1 and native_failures[0] != 0
            assert retained.close_error == native_failures[0]
        else:
            assert retained.close_error == 5
        if outcome == "after-close-false":
            assert not birth["live"]
            with pytest.raises(OSError) as invalid:
                native.info(retained.handle)
            assert invalid.value.winerror == 6
        else:
            assert birth["live"]
            info = native.info(retained.handle)
            assert (
                info.volume,
                (info.index_high << 32) | info.index_low,
            ) == retained.identity
    finally:
        # Numeric handle identity alone never authorizes cleanup. Every native
        # open birth/close attempt is observed; only the still-current captured
        # live birth can be unprotected/reclaimed, never after-close uncertainty.
        if captured:
            birth = captured[0]
            handle = birth["handle"]
            if (
                outcome != "after-close-false"
                and birth["live"]
                and current.get(handle) is birth
            ):
                info = native.info(handle)
                assert (info.volume, (info.index_high << 32) | info.index_low) == birth[
                    "identity"
                ]
                if birth["protected"]:
                    native.check(protect(handle, 2, 0))
                assert original_close(handle)
                birth["live"] = False
    assert births and not any(birth["live"] for birth in births)
    for handle in current:
        with pytest.raises(OSError) as invalid:
            native.info(handle)
        assert invalid.value.winerror == 6


def test_fresh_snapshot_preserves_order_missing_values_and_retires_native_handles(
    tmp_path, monkeypatch
):
    native, win = windows_files._native(), WindowsOS()
    left, right, missing = (tmp_path / name for name in ("left", "right", "missing"))
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    expected = {path: win.stat(path, follow_symlinks=False) for path in (left, right)}
    opened = []
    original = native.open_handle

    def observe(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(native, "open_handle", observe)
    result = win.stat_many_for_admission((right, missing, left, right))
    assert list(result) == [right, missing, left]
    assert result[missing] is None
    assert opened, "original native bodies must actually run"
    for path in (left, right):
        info, descriptor = result[path]
        assert type(info) is os.stat_result
        assert tuple(info) == tuple(expected[path])
        assert info.st_mtime_ns == expected[path].st_mtime_ns
        assert type(descriptor) is bytes and 20 <= len(descriptor) <= 4096
    for handle in opened:
        with pytest.raises(OSError) as invalid:
            native.info(handle)
        assert invalid.value.winerror == 6


def test_empty_snapshot_performs_no_native_open(monkeypatch):
    native, win = windows_files._native(), WindowsOS()

    def unexpected(*args, **kwargs):
        pytest.fail("empty metadata snapshot opened a native handle")

    monkeypatch.setattr(native, "open_handle", unexpected)
    assert win.stat_many_for_admission(()) == {}


def test_scalar_fresh_metadata_preserves_current_projection_and_exact_descriptor(
    tmp_path, monkeypatch
):
    win = WindowsOS()
    selected = tmp_path / "scalar.bin"
    selected.write_bytes(b"metadata")
    ordinary = win.stat(selected, follow_symlinks=False)
    native = windows_files._native()
    original = native.advapi.GetSecurityInfo
    descriptors = []

    def observe_descriptor(*args):
        result = original(*args)
        if result == 0:
            pointer = C.cast(args[-1], C.POINTER(C.c_void_p)).contents.value
            size = native.advapi.GetSecurityDescriptorLength(pointer)
            assert 20 <= size <= 4096
            descriptors.append(C.string_at(pointer, size))
        return result

    monkeypatch.setattr(native.advapi, "GetSecurityInfo", observe_descriptor)
    info, descriptor = win.stat_for_admission(selected, follow_symlinks=False)
    assert descriptors and descriptor == descriptors[-1]
    assert type(info) is type(ordinary) is os.stat_result
    assert tuple(info) == tuple(ordinary)
    assert info.st_mtime_ns == ordinary.st_mtime_ns
    assert type(descriptor) is bytes and 20 <= len(descriptor) <= 4096
    selected.unlink()
    with pytest.raises(FileNotFoundError):
        win.stat_for_admission(selected, follow_symlinks=False)
