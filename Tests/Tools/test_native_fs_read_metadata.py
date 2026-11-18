"""Windows bounded reads compare native change time, including the open boundary."""

import ctypes
import os
import sys
from ctypes import wintypes
from types import SimpleNamespace

import pytest

from tldw_chatbook.Tools import local_tool_impls as local
from tldw_chatbook.Tools import workspace_root_pin as pin

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.skipif(os.name != "nt", reason="actual Windows file metadata"),
]


def _distinct_clocks(target):
    """Set real, distinct creation/change clocks without waiting for a timer tick."""
    class BasicInfo(ctypes.Structure):
        _fields_ = [
            ("created", ctypes.c_int64), ("accessed", ctypes.c_int64),
            ("written", ctypes.c_int64), ("changed", ctypes.c_int64),
            ("attributes", wintypes.DWORD),
        ]

    import msvcrt

    descriptor = os.open(target, os.O_RDWR | os.O_BINARY)
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ]
        kernel.SetFileInformationByHandle.restype = wintypes.BOOL
        # January 2000 and January 2001, in Windows 100 ns units. A subsequent
        # real write changes the latter even on a filesystem with coarse ticks.
        basic = BasicInfo(125911584000000000, 0, 0, 126227808000000000, 0)
        assert kernel.SetFileInformationByHandle(
            msvcrt.get_osfhandle(descriptor), 0, ctypes.byref(basic),
            ctypes.sizeof(basic),
        )
    finally:
        os.close(descriptor)


def _native_stat(target):
    with target.open("rb") as stream:
        return os.fstat(stream.fileno())


def _identity_and_write_time(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_birthtime_ns)


def test_native_read_accepts_distinct_path_and_handle_clocks(tmp_path):
    target = tmp_path / "distinct.txt"
    content = b"first\r\nsecond\x1aend\r\n"
    target.write_bytes(content)
    _distinct_clocks(target)
    named, opened = target.stat(follow_symlinks=False), _native_stat(target)
    assert type(opened.st_birthtime_ns) is int
    assert named.st_birthtime_ns == opened.st_birthtime_ns
    assert opened.st_birthtime_ns != opened.st_ctime_ns
    if sys.version_info[:2] == (3, 12):
        assert named.st_ctime_ns != opened.st_ctime_ns
    assert local._read_bounded_bytes(target) == content


@pytest.mark.parametrize("boundary", ["open", "read"])
def test_native_read_refuses_same_identity_write_with_restored_mtime(
    tmp_path, monkeypatch, boundary,
):
    target = tmp_path / "mutating.txt"
    target.write_bytes(b"before\r\n")
    _distinct_clocks(target)
    before = _native_stat(target)
    original_open, original_read, original_close = os.open, os.read, os.close
    body_descriptors, body_closes, mutations = [], [], []

    def mutate():
        target.write_bytes(b"after!\r\n")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = _native_stat(target)
        assert _identity_and_write_time(after) == _identity_and_write_time(before)
        assert after.st_ctime_ns != before.st_ctime_ns
        mutations.append(True)

    def body_open(path, *args, **kwargs):
        if path == target and boundary == "open":
            mutate()
        descriptor = original_open(path, *args, **kwargs)
        if path == target:
            body_descriptors.append(descriptor)
        return descriptor

    def body_read(descriptor, count):
        result = original_read(descriptor, count)
        if descriptor in body_descriptors and boundary == "read" and not mutations:
            mutate()
        return result

    def close(descriptor):
        if descriptor in body_descriptors:
            body_closes.append(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(local.os, "open", body_open)
    monkeypatch.setattr(local.os, "read", body_read)
    monkeypatch.setattr(local.os, "close", close)
    with pytest.raises(local.LocalToolError, match=f"while {boundary}ing"):
        local._read_bounded_bytes(target)
    assert mutations == [True]
    assert body_descriptors and body_closes == body_descriptors


def test_native_read_refuses_named_replacement_before_body_open(tmp_path, monkeypatch):
    target, replacement = tmp_path / "first.txt", tmp_path / "second.txt"
    target.write_bytes(b"original")
    replacement.write_bytes(b"replaced")
    original_open = os.open
    replaced = []

    def replace(path, *args, **kwargs):
        if path == target:
            os.replace(replacement, target)
            replaced.append(True)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(local.os, "open", replace)
    with pytest.raises(local.LocalToolError, match="while opening"):
        local._read_bounded_bytes(target)
    assert replaced == [True]


def test_native_metadata_refuses_reparse_and_retires_its_handle(tmp_path, monkeypatch):
    target, link = tmp_path / "real.txt", tmp_path / "link.txt"
    target.write_bytes(b"original")
    try:
        link.symlink_to(target)
    except OSError as error:
        if error.winerror != 1314:
            raise
        # A junction exercises the same native reparse rejection without the
        # optional Windows privilege required to create a file symlink.
        import _winapi

        directory = tmp_path / "real-directory"
        directory.mkdir()
        _winapi.CreateJunction(str(directory), str(link))
    original_close = pin._windows_close_metadata
    closes = []

    def close(handle):
        original_close(handle)
        closes.append(handle)

    monkeypatch.setattr(local, "_windows_close_metadata", close)
    with pytest.raises(local.LocalToolError, match="not regular"):
        local._read_bounded_bytes(link)
    assert len(closes) == 1


def test_native_metadata_close_failure_never_returns_contents(tmp_path, monkeypatch):
    target = tmp_path / "close.txt"
    target.write_bytes(b"original")
    original_close = os.close
    closes = []

    def close(descriptor):
        original_close(descriptor)
        closes.append(descriptor)
        raise OSError("injected close uncertainty after actual retirement")

    monkeypatch.setattr(local.os, "close", close)
    with pytest.raises(OSError, match="close uncertainty"):
        local._read_bounded_bytes(target)
    assert len(closes) == 1


def test_native_read_refuses_unavailable_creation_time(tmp_path, monkeypatch):
    target = tmp_path / "incomplete.txt"
    target.write_bytes(b"original")
    original = os.fstat

    def incomplete(descriptor):
        actual = original(descriptor)
        return SimpleNamespace(**{
            name: getattr(actual, name)
            for name in (
                "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns",
            )
        })

    monkeypatch.setattr(local.os, "fstat", incomplete)
    with pytest.raises(local.LocalToolError, match="creation time unavailable"):
        local._read_bounded_bytes(target)


def test_missing_native_read_keeps_file_error_and_root_pin_keeps_root_error(tmp_path):
    target = tmp_path / "missing-private-name.txt"
    with pytest.raises(FileNotFoundError) as caught:
        local._read_bounded_bytes(target)
    assert target.name not in str(caught.value)
    assert caught.value.filename is None
    with pytest.raises(pin.WorkspaceRootPinError, match="workspace root open failed"):
        pin._windows_open_directory(target)
