"""Original native Windows directory identity and retained-root controls."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Tools import workspace_root_pin as pin
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows HANDLE identity")


@contextmanager
def _original_handles():
    opened = Counter()
    closed = Counter()
    open_code = pin._windows_open_directory.__code__
    close_code = pin._windows_close_handle.__code__
    previous = sys.getprofile()

    def observe(frame, event, value):
        if event != "return":
            return
        if frame.f_code is open_code and isinstance(value, tuple):
            opened[value[0]] += 1
        elif frame.f_code is close_code:
            closed[frame.f_locals["handle"]] += 1

    sys.setprofile(observe)
    try:
        yield opened, closed
    finally:
        sys.setprofile(previous)
        assert opened, "the actual native opener was not observed"
        assert not (opened - closed), "an original native handle did not retire"
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetHandleInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetHandleInformation.restype = wintypes.BOOL
        for handle in opened:
            flags = wintypes.DWORD()
            ctypes.set_last_error(0)
            assert not kernel32.GetHandleInformation(handle, ctypes.byref(flags))
            assert ctypes.get_last_error() == 6  # ERROR_INVALID_HANDLE


def test_actual_handle_identity_matches_current_python_stat(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    expected = capture_directory_chain(root).identities[0]
    with _original_handles():
        handle, observed, reparse = pin._windows_open_directory(root)
        try:
            assert (observed.device, observed.inode) == (
                expected.device,
                expected.inode,
            )
            assert reparse is False
        finally:
            pin._windows_close_handle(handle)


@pytest.mark.parametrize("body_error", [False, True])
def test_retained_pin_restores_cwd_and_retires_actual_handles(
    tmp_path: Path, body_error: bool
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "sentinel.txt").write_text("original", encoding="utf-8")
    chain = capture_directory_chain(root)
    previous_directory = Path.cwd()

    def visit():
        with pin.pin_workspace_root(root, chain) as pinned:
            assert (
                pinned.relative_path("sentinel.txt").read_text(encoding="utf-8")
                == "original"
            )
            if body_error:
                raise RuntimeError("body failure")

    with _original_handles() as (opened, _closed):
        if body_error:
            with pytest.raises(RuntimeError, match="body failure"):
                visit()
        else:
            visit()
        assert (
            sum(opened.values()) == 2
        )  # retained root and current-directory verification
    assert Path.cwd() == previous_directory


@pytest.mark.parametrize("field, bit", [("device", 1 << 48), ("inode", 1 << 96)])
def test_high_identity_bits_are_compared_without_truncation(
    tmp_path: Path, field: str, bit: int
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    chain = capture_directory_chain(root)
    identity = chain.identities[0]
    changed_identity = replace(identity, **{field: getattr(identity, field) ^ bit})
    changed_chain = replace(chain, identities=(changed_identity, *chain.identities[1:]))
    previous_directory = Path.cwd()
    with _original_handles():
        with pytest.raises(pin.WorkspaceRootPinError, match="identity mismatch"):
            with pin.pin_workspace_root(root, changed_chain):
                pytest.fail("changed high identity bits admitted")
    assert Path.cwd() == previous_directory


def test_real_junction_is_refused_and_its_handle_retires(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    junction = tmp_path / "junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(root)],
        check=True,
        capture_output=True,
    )  # nosec B603 B607 -- fixed command and synthetic temporary paths
    expected = capture_directory_chain(root).identities[0]
    previous_directory = Path.cwd()
    with _original_handles():
        with pytest.raises(
            pin.WorkspaceRootPinError, match="unsafe workspace root metadata"
        ):
            pin._pin_windows(junction, expected)
    assert Path.cwd() == previous_directory
