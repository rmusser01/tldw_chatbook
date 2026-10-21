from __future__ import annotations

import ctypes
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_chatbook.Tools.workspace_root_pin as workspace_root_pin
from tldw_chatbook.Tools.workspace_root_pin import (
    WorkspaceRootPinError,
    pin_workspace_root,
)
from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryChain,
    capture_directory_chain,
)


def _pre_pin_child(
    locator: str,
    chain: DirectoryChain,
    ready: Any,
    resume: Any,
    output: Any,
) -> None:
    try:
        ready.set()
        if not resume.wait(5):
            raise RuntimeError("test barrier timed out")
        with pin_workspace_root(Path(locator), chain):
            output.put(("pinned", ""))
    except WorkspaceRootPinError as error:
        output.put(("refused", str(error)))


def _post_pin_child(
    locator: str,
    chain: DirectoryChain,
    ready: Any,
    resume: Any,
    output: Any,
) -> None:
    try:
        with pin_workspace_root(Path(locator), chain) as pinned:
            ready.set()
            if not resume.wait(5):
                raise RuntimeError("test barrier timed out")
            output.put(
                (
                    "read",
                    pinned.relative_path("sentinel.txt").read_text(encoding="utf-8"),
                )
            )
    except BaseException as error:
        output.put(("error", type(error).__name__))


def _post_pin_write_child(
    locator: str,
    chain: DirectoryChain,
    ready: Any,
    resume: Any,
    output: Any,
) -> None:
    try:
        with pin_workspace_root(Path(locator), chain) as pinned:
            ready.set()
            if not resume.wait(5):
                raise RuntimeError("test barrier timed out")
            pinned.relative_path("written.txt").write_text("A_WRITE", encoding="utf-8")
            output.put(("written", "A_WRITE"))
    except BaseException as error:
        output.put(("error", type(error).__name__))


def _spawn_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


def _join_child(process: Any) -> None:
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("root-pin child did not exit")
    assert process.exitcode == 0


def test_root_replacement_before_pin_is_refused_by_identity(tmp_path: Path) -> None:
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "sentinel.txt").write_text("A", encoding="utf-8")
    chain = capture_directory_chain(locator)

    context = _spawn_context()
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_pre_pin_child,
        args=(str(locator), chain, ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "root-pin child did not reach the pre-pin barrier"

    retained_a = tmp_path / "retained-a"
    locator.rename(retained_a)
    locator.mkdir()
    (locator / "sentinel.txt").write_text("B", encoding="utf-8")
    resume.set()
    _join_child(process)

    assert output.get(timeout=2) == ("refused", "workspace root identity mismatch")


def test_root_replacement_after_pin_never_redirects_relative_io(tmp_path: Path) -> None:
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "sentinel.txt").write_text("A", encoding="utf-8")
    replacement_b = tmp_path / "replacement-b"
    replacement_b.mkdir()
    (replacement_b / "sentinel.txt").write_text("B", encoding="utf-8")
    chain = capture_directory_chain(locator)

    context = _spawn_context()
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_post_pin_child,
        args=(str(locator), chain, ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "root-pin child did not reach the barrier"

    retained_a = tmp_path / "retained-a"
    replacement_refused = False
    try:
        os.replace(locator, retained_a)
        os.replace(replacement_b, locator)
    except OSError:
        replacement_refused = True
        if retained_a.exists() and not locator.exists():
            os.replace(retained_a, locator)
    finally:
        resume.set()

    _join_child(process)
    outcome = output.get(timeout=2)
    assert outcome == ("read", "A")
    if os.name == "nt":
        assert replacement_refused, "Windows should lock the retained current directory"


def test_root_replacement_after_pin_never_redirects_relative_write(
    tmp_path: Path,
) -> None:
    locator = tmp_path / "workspace"
    locator.mkdir()
    replacement_b = tmp_path / "replacement-b"
    replacement_b.mkdir()
    chain = capture_directory_chain(locator)

    context = _spawn_context()
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_post_pin_write_child,
        args=(str(locator), chain, ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "root-pin child did not reach the write barrier"

    retained_a = tmp_path / "retained-a"
    replacement_refused = False
    try:
        os.replace(locator, retained_a)
        os.replace(replacement_b, locator)
    except OSError:
        replacement_refused = True
        if retained_a.exists() and not locator.exists():
            os.replace(retained_a, locator)
    finally:
        resume.set()

    _join_child(process)
    assert output.get(timeout=2) == ("written", "A_WRITE")
    written_in_a = (
        (retained_a / "written.txt")
        if retained_a.exists()
        else (locator / "written.txt")
    )
    assert written_in_a.read_text(encoding="utf-8") == "A_WRITE"
    if replacement_b.exists():
        assert not (replacement_b / "written.txt").exists()
    else:
        assert not (locator / "written.txt").exists()
    if os.name == "nt":
        assert replacement_refused, "Windows should lock the retained current directory"


def test_relative_path_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    chain = capture_directory_chain(root)

    with pin_workspace_root(root, chain) as pinned:
        with pytest.raises(WorkspaceRootPinError, match="relative path"):
            pinned.relative_path(str(tmp_path / "outside.txt"))
        with pytest.raises(WorkspaceRootPinError, match="relative path"):
            pinned.relative_path("../outside.txt")


def test_root_locator_uses_central_path_validator_before_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    chain = capture_directory_chain(root)
    calls: list[tuple[Path, Path, bool, bool]] = []
    validated = [root.resolve()]

    def validate(
        user_path: Path,
        base_directory: Path,
        *,
        redact_paths: bool,
        allow_hidden: bool,
    ) -> Path:
        calls.append(
            (Path(user_path), Path(base_directory), redact_paths, allow_hidden)
        )
        return validated[0]

    monkeypatch.setattr(workspace_root_pin, "validate_path", validate, raising=False)

    with pin_workspace_root(root, chain) as pinned:
        assert pinned.canonical_locator == root.resolve()

    validated[0] = other.resolve()
    with pytest.raises(WorkspaceRootPinError, match="invalid admitted"):
        with pin_workspace_root(root, chain):
            pass

    assert calls == [(root, root, True, True), (root, root, True, True)]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(r"\outside.txt", id="rooted-current-drive"),
        pytest.param("C:outside.txt", id="drive-relative"),
        pytest.param(r"C:\outside.txt", id="drive-rooted"),
        pytest.param(r"\\server\share\outside.txt", id="unc"),
    ],
)
def test_relative_path_rejects_windows_drive_root_and_anchor_on_every_platform(
    tmp_path: Path,
    value: str,
) -> None:
    windows_path = PureWindowsPath(value)
    assert windows_path.drive or windows_path.root or windows_path.anchor
    root = tmp_path / "workspace"
    root.mkdir()
    chain = capture_directory_chain(root)

    with pin_workspace_root(root, chain) as pinned:
        with pytest.raises(WorkspaceRootPinError, match="relative path"):
            pinned.relative_path(value)


@pytest.mark.skipif(os.name != "nt", reason="native Windows handle identity")
def test_windows_handle_identity_matches_full_python_identity(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    expected = os.lstat(root)

    handle, actual, reparse = workspace_root_pin._windows_open_directory(root)
    try:
        assert actual.device == expected.st_dev
        assert actual.inode == expected.st_ino
        assert not reparse
    finally:
        workspace_root_pin._windows_close_handle(handle)


@pytest.mark.skipif(os.name != "nt", reason="native Windows current-directory pin")
def test_windows_unchanged_root_pins_and_reads_relative_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sentinel.txt").write_text("ADMITTED", encoding="utf-8")
    chain = capture_directory_chain(root)
    previous = Path.cwd()

    with pin_workspace_root(root, chain) as pinned:
        assert pinned.identity == chain.identities[0]
        assert (
            pinned.relative_path("sentinel.txt").read_text(encoding="utf-8")
            == "ADMITTED"
        )
    assert Path.cwd() == previous


@pytest.mark.skipif(os.name != "nt", reason="native Windows handle identity")
def test_windows_pin_rejects_device_with_same_low_bits(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    chain = capture_directory_chain(root)
    expected = chain.identities[0]
    changed = replace(expected, device=expected.device ^ (1 << 32))
    assert changed.device & 0xFFFFFFFF == expected.device & 0xFFFFFFFF
    altered_chain = replace(chain, identities=(changed, *chain.identities[1:]))
    previous = Path.cwd()

    with pytest.raises(WorkspaceRootPinError, match="identity mismatch"):
        with pin_workspace_root(root, altered_chain):
            pytest.fail("different high device bits were admitted")
    assert Path.cwd() == previous


@pytest.mark.skipif(os.name != "nt", reason="native Windows metadata failure")
def test_windows_identity_metadata_failure_closes_open_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ctypes import wintypes

    root = tmp_path / "workspace"
    root.mkdir()
    native = ctypes.WinDLL("kernel32", use_last_error=True)
    native.GetHandleInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    native.GetHandleInformation.restype = wintypes.BOOL
    opened: list[int] = []

    def unavailable(handle, _kind, _information, _size):
        opened.append(handle)
        return False

    kernel32 = SimpleNamespace(
        CreateFileW=native.CreateFileW,
        GetFileInformationByHandle=native.GetFileInformationByHandle,
        GetFileInformationByHandleEx=unavailable,
        CloseHandle=native.CloseHandle,
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: kernel32)

    with pytest.raises(WorkspaceRootPinError, match="metadata unavailable"):
        handle, _identity, _reparse = workspace_root_pin._windows_open_directory(root)
        workspace_root_pin._windows_close_handle(handle)
    assert len(opened) == 1
    flags = wintypes.DWORD()
    assert not native.GetHandleInformation(opened[0], ctypes.byref(flags))
    assert ctypes.get_last_error() == 6  # ERROR_INVALID_HANDLE


@pytest.mark.skipif(os.name != "nt", reason="native Windows identity layout")
def test_windows_handle_identity_preserves_all_file_id_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    native = ctypes.WinDLL("kernel32", use_last_error=True)
    # FILE_ID_INFO: ULONGLONG followed by BYTE Identifier[16].
    payload = bytes.fromhex("efcdab8967452301ffeeddccbbaa99887766554433221100")

    def full_identity(_handle, kind, information, size):
        assert kind == 18  # FileIdInfo
        assert size == 24
        ctypes.memmove(information, payload, len(payload))
        return True

    kernel32 = SimpleNamespace(
        CreateFileW=native.CreateFileW,
        GetFileInformationByHandle=native.GetFileInformationByHandle,
        GetFileInformationByHandleEx=full_identity,
        CloseHandle=native.CloseHandle,
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: kernel32)

    handle, actual, reparse = workspace_root_pin._windows_open_directory(root)
    try:
        assert actual.device == 0x0123456789ABCDEF
        assert actual.inode == 0x00112233445566778899AABBCCDDEEFF
        assert not reparse
    finally:
        workspace_root_pin._windows_close_handle(handle)


@pytest.mark.skipif(os.name != "nt", reason="native Windows reparse refusal")
def test_windows_pin_refuses_junction_root(tmp_path: Path) -> None:
    import _winapi

    root = tmp_path / "workspace"
    root.mkdir()
    expected = capture_directory_chain(root).identities[0]
    retained = tmp_path / "retained"
    root.rename(retained)
    _winapi.CreateJunction(str(retained), str(root))

    handle, _identity, reparse = workspace_root_pin._windows_open_directory(root)
    try:
        assert reparse
    finally:
        workspace_root_pin._windows_close_handle(handle)
    with pytest.raises(WorkspaceRootPinError, match="unsafe workspace root metadata"):
        workspace_root_pin._pin_windows(root, expected)
