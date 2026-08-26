"""Contracts for the narrow native Windows audio.cpp artifact boundary."""

from __future__ import annotations

import asyncio
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pytest

from tldw_chatbook.TTS import windows_artifact_fs


class FakeWindowsKernel:
    """Deterministic kernel boundary without host Win32 calls."""

    def __init__(self) -> None:
        self.next_handle = 40
        self.identities: dict[int, windows_artifact_fs.WindowsFileIdentity] = {}
        self.open_calls: list[tuple[str, Literal["directory", "file"], bool, bool]] = []
        self.inheritable_calls: list[tuple[int, bool]] = []
        self.closed: list[int] = []
        self.close_failures = 0
        self.set_acl_calls: list[tuple[int, bool]] = []
        self.verify_acl_calls: list[tuple[int, bool]] = []
        self.acl_matches = True
        self.writes: list[tuple[int, bytes]] = []
        self.flushes: list[int] = []
        self.read_only: list[int] = []
        self.locks: list[int] = []
        self.unlocks: list[int] = []
        self.deleted: list[int] = []

    def open_handle(
        self,
        path: str,
        *,
        kind: Literal["directory", "file"],
        create_new: bool,
        writable: bool,
    ) -> int:
        self.open_calls.append((path, kind, create_new, writable))
        handle = self.next_handle
        self.next_handle += 1
        self.identities[handle] = windows_artifact_fs.WindowsFileIdentity(
            volume_serial_number=9,
            file_id=handle.to_bytes(16, "little"),
            kind=kind,
            reparse_tag=0,
        )
        return handle

    def set_inheritable(self, handle: int, inheritable: bool) -> None:
        self.inheritable_calls.append((handle, inheritable))

    def identity(self, handle: int) -> windows_artifact_fs.WindowsFileIdentity:
        return self.identities[handle]

    def current_user_sid(self) -> bytes:
        return b"PRIVATE-TEST-SID"

    def set_private_acl(self, handle: int, user_sid: bytes, directory: bool) -> None:
        assert user_sid == b"PRIVATE-TEST-SID"
        self.set_acl_calls.append((handle, directory))

    def verify_private_acl(
        self,
        handle: int,
        user_sid: bytes,
        directory: bool,
    ) -> bool:
        assert user_sid == b"PRIVATE-TEST-SID"
        self.verify_acl_calls.append((handle, directory))
        return self.acl_matches

    def write_all(self, handle: int, data: bytes) -> None:
        self.writes.append((handle, data))

    def read(self, handle: int, count: int, offset: int) -> bytes:
        del handle, count, offset
        return b"fixture"

    def flush(self, handle: int) -> None:
        self.flushes.append(handle)

    def set_read_only(self, handle: int) -> None:
        self.read_only.append(handle)

    def lock_exclusive_nonblocking(self, handle: int) -> None:
        self.locks.append(handle)

    def unlock(self, handle: int) -> None:
        self.unlocks.append(handle)

    def delete_handle(self, handle: int) -> None:
        self.deleted.append(handle)

    def close_handle(self, handle: int) -> None:
        if self.close_failures:
            self.close_failures -= 1
            raise OSError("PRIVATE CLOSE PATH")
        self.closed.append(handle)


def test_windows_artifact_filesystem_module_is_available() -> None:
    assert find_spec("tldw_chatbook.TTS.windows_artifact_fs") is not None


def test_process_default_filesystem_is_fail_closed_off_windows() -> None:
    if os.name == "nt":
        assert isinstance(
            windows_artifact_fs.OS_WINDOWS_ARTIFACT_FILESYSTEM,
            windows_artifact_fs.NativeWindowsArtifactFilesystem,
        )
    else:
        assert isinstance(
            windows_artifact_fs.OS_WINDOWS_ARTIFACT_FILESYSTEM,
            windows_artifact_fs.UnavailableWindowsArtifactFilesystem,
        )


@pytest.mark.parametrize(
    ("selected", "extended"),
    [
        (r"C:\Models\Voice 模型", r"\\?\C:\Models\Voice 模型"),
        (
            r"\\model-host\shared models\Voice",
            r"\\?\UNC\model-host\shared models\Voice",
        ),
    ],
)
def test_normalize_windows_artifact_path_preserves_absolute_unicode_paths(
    selected: str,
    extended: str,
) -> None:
    assert windows_artifact_fs.normalize_windows_artifact_path(selected) == extended


@pytest.mark.parametrize(
    "selected",
    [
        "",
        r"relative\model",
        r"C:relative\model",
        r"\\?\C:\private\model",
        r"\\.\C:\private\model",
        r"\Device\HarddiskVolume1\private\model",
        r"C:\private\..\model",
        r"C:\private\NUL\model",
        "C:\\private\\trailing.\\model",
        "C:\\private\\trailing \\model",
        r"C:\private\asset:stream",
    ],
)
def test_normalize_windows_artifact_path_rejects_ambiguous_namespaces(
    selected: str,
) -> None:
    with pytest.raises(ValueError, match="Windows artifact path is unavailable"):
        windows_artifact_fs.normalize_windows_artifact_path(selected)


@pytest.mark.parametrize(
    ("machine", "pointer_bits", "expected"),
    [
        ("AMD64", 64, "x86_64"),
        ("x86_64", 64, "x86_64"),
        ("AMD64", 32, "x86"),
        ("x86", 32, "x86"),
        ("i686", 32, "x86"),
        ("ARM64", 64, None),
        ("aarch64", 64, None),
        ("x86", 64, None),
        ("AMD64", 16, None),
    ],
)
def test_normalize_windows_process_architecture_is_pointer_width_exact(
    machine: str,
    pointer_bits: int,
    expected: str | None,
) -> None:
    assert (
        windows_artifact_fs.normalize_windows_process_architecture(
            machine, pointer_bits
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "system_name",
        "windows_major",
        "python_version",
        "machine",
        "pointer_bits",
        "supported",
    ),
    [
        ("Windows", 10, (3, 12), "AMD64", 64, True),
        ("Windows", 11, (3, 13), "AMD64", 32, True),
        ("Windows", 9, (3, 12), "AMD64", 64, False),
        ("Windows", 10, (3, 11), "AMD64", 64, False),
        ("Windows", 10, (3, 12), "ARM64", 64, False),
        ("Darwin", 15, (3, 12), "AMD64", 64, False),
    ],
)
def test_windows_audio_cpp_platform_support_is_explicit(
    system_name: str,
    windows_major: int,
    python_version: tuple[int, int],
    machine: str,
    pointer_bits: int,
    supported: bool,
) -> None:
    assert (
        windows_artifact_fs.windows_audio_cpp_platform_supported(
            system_name=system_name,
            windows_major=windows_major,
            python_version=python_version,
            machine=machine,
            pointer_bits=pointer_bits,
        )
        is supported
    )


def _filesystem(
    kernel: FakeWindowsKernel,
) -> windows_artifact_fs.NativeWindowsArtifactFilesystem:
    return windows_artifact_fs.NativeWindowsArtifactFilesystem(kernel=kernel)


def test_pinned_directory_is_non_inheritable_and_opaque() -> None:
    kernel = FakeWindowsKernel()

    pinned = _filesystem(kernel).pin_directory_no_reparse(r"C:\private\models")

    assert kernel.open_calls == [(r"\\?\C:\private\models", "directory", False, False)]
    assert kernel.inheritable_calls == [(40, False)]
    assert pinned.identity == kernel.identities[40]
    assert "40" not in repr(pinned)
    assert "private" not in repr(pinned).casefold()
    pinned.close()
    pinned.close()
    assert kernel.closed == [40]


def test_reparse_directory_is_rejected_and_closed() -> None:
    kernel = FakeWindowsKernel()
    filesystem = _filesystem(kernel)
    original_open = kernel.open_handle

    def open_reparse(*args: object, **kwargs: object) -> int:
        handle = original_open(*args, **kwargs)  # type: ignore[arg-type]
        kernel.identities[handle] = windows_artifact_fs.WindowsFileIdentity(
            volume_serial_number=9,
            file_id=handle.to_bytes(16, "little"),
            kind="directory",
            reparse_tag=0xA0000003,
        )
        return handle

    kernel.open_handle = open_reparse  # type: ignore[method-assign]

    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as raised:
        filesystem.pin_directory_no_reparse(r"C:\private\junction")

    assert raised.value.code == "changed"
    assert str(raised.value) == "Windows artifact identity changed"
    assert kernel.closed == [40]


def test_private_directory_installs_and_reverifies_protected_acl() -> None:
    kernel = FakeWindowsKernel()

    owner = _filesystem(kernel).create_private_directory(r"C:\private\owner")

    assert kernel.open_calls == [(r"\\?\C:\private\owner", "directory", True, True)]
    assert kernel.set_acl_calls == [(40, True)]
    assert kernel.verify_acl_calls == [(40, True)]
    assert owner.privacy_posture == "windows_account_protected"


def test_existing_directory_can_be_protected_without_recreation() -> None:
    kernel = FakeWindowsKernel()

    owner = _filesystem(kernel).protect_private_directory(r"C:\private\runtime")

    assert kernel.open_calls == [(r"\\?\C:\private\runtime", "directory", False, True)]
    assert kernel.set_acl_calls == [(40, True)]
    assert kernel.verify_acl_calls == [(40, True)]
    assert owner.privacy_posture == "windows_account_protected"


def test_private_file_is_flushed_hardened_and_verified_before_publication() -> None:
    kernel = FakeWindowsKernel()

    owner = _filesystem(kernel).create_private_file(
        r"C:\private\owner\server.json",
        b'{"host":"127.0.0.1"}\n',
        read_only=True,
    )

    assert kernel.writes == [(40, b'{"host":"127.0.0.1"}\n')]
    assert kernel.flushes == [40]
    assert kernel.set_acl_calls == [(40, False)]
    assert kernel.read_only == [40]
    assert kernel.verify_acl_calls == [(40, False)]
    assert owner.privacy_posture == "windows_account_protected"


def test_failed_acl_verification_never_publishes_the_handle() -> None:
    kernel = FakeWindowsKernel()
    kernel.acl_matches = False

    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as raised:
        _filesystem(kernel).create_private_directory(r"C:\private\owner")

    assert raised.value.code == "privacy_unavailable"
    assert str(raised.value) == "Windows artifact privacy is unavailable"
    assert kernel.closed == [40]


def test_owner_lock_and_unlock_target_only_the_exact_handle() -> None:
    kernel = FakeWindowsKernel()
    owner = _filesystem(kernel).create_private_file(
        r"C:\private\owner\owner.lock",
        b"",
    )

    owner.lock_exclusive_nonblocking()
    owner.unlock()

    assert kernel.locks == [40]
    assert kernel.unlocks == [40]


def test_exact_delete_revalidates_identity_and_acl() -> None:
    kernel = FakeWindowsKernel()
    owner = _filesystem(kernel).create_private_directory(r"C:\private\owner")
    initial_verifications = len(kernel.verify_acl_calls)

    owner.delete_exact()

    assert len(kernel.verify_acl_calls) == initial_verifications + 1
    assert kernel.deleted == [40]


def test_exact_delete_preserves_a_substituted_identity() -> None:
    kernel = FakeWindowsKernel()
    owner = _filesystem(kernel).create_private_directory(r"C:\private\owner")
    kernel.identities[40] = windows_artifact_fs.WindowsFileIdentity(
        volume_serial_number=10,
        file_id=b"replacement".ljust(16, b"\0"),
        kind="directory",
        reparse_tag=0,
    )

    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as raised:
        owner.delete_exact()

    assert raised.value.code == "changed"
    assert kernel.deleted == []


def test_failed_close_retains_the_exact_handle_for_retry() -> None:
    kernel = FakeWindowsKernel()
    owner = _filesystem(kernel).create_private_directory(r"C:\private\owner")
    kernel.close_failures = 1

    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as raised:
        owner.close()

    assert raised.value.code == "cleanup_failed"
    assert raised.value.take_cleanup_owner() is owner
    assert raised.value.take_cleanup_owner() is None
    owner.close()
    assert kernel.closed == [40]


def test_native_failures_are_bounded_and_path_free() -> None:
    kernel = FakeWindowsKernel()

    def fail_open(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise OSError("PRIVATE-TOKEN C:\\Users\\owner\\secret")

    kernel.open_handle = fail_open  # type: ignore[method-assign]

    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as raised:
        _filesystem(kernel).pin_directory_no_reparse(r"C:\Users\owner\secret")

    rendered = repr(raised.value) + str(raised.value)
    assert raised.value.code == "unavailable"
    assert "PRIVATE-TOKEN" not in rendered
    assert "Users" not in rendered
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_post_open_failure_and_close_failure_retains_cleanup_owner() -> None:
    kernel = FakeWindowsKernel()
    kernel.close_failures = 1

    def fail_identity(handle: int) -> windows_artifact_fs.WindowsFileIdentity:
        del handle
        raise OSError("PRIVATE IDENTITY PATH")

    kernel.identity = fail_identity  # type: ignore[method-assign]

    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as raised:
        _filesystem(kernel).pin_directory_no_reparse(r"C:\private\owner")

    assert raised.value.code == "cleanup_failed"
    cleanup = raised.value.take_cleanup_owner()
    assert cleanup is not None
    cleanup.close()
    assert kernel.closed == [40]


def test_rejected_reparse_and_close_failure_retains_cleanup_owner() -> None:
    kernel = FakeWindowsKernel()
    filesystem = _filesystem(kernel)
    original_open = kernel.open_handle

    def open_reparse(*args: object, **kwargs: object) -> int:
        handle = original_open(*args, **kwargs)  # type: ignore[arg-type]
        kernel.identities[handle] = windows_artifact_fs.WindowsFileIdentity(
            volume_serial_number=9,
            file_id=handle.to_bytes(16, "little"),
            kind="directory",
            reparse_tag=0xA0000003,
        )
        kernel.close_failures = 1
        return handle

    kernel.open_handle = open_reparse  # type: ignore[method-assign]

    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as raised:
        filesystem.pin_directory_no_reparse(r"C:\private\junction")

    assert raised.value.code == "cleanup_failed"
    cleanup = raised.value.take_cleanup_owner()
    assert cleanup is not None
    cleanup.close()
    assert kernel.closed == [40]


def test_failed_private_creation_retirement_is_retryable() -> None:
    kernel = FakeWindowsKernel()

    def fail_acl(handle: int, user_sid: bytes, directory: bool) -> None:
        del handle, user_sid, directory
        kernel.close_failures = 1
        raise OSError("PRIVATE ACL PATH")

    kernel.set_private_acl = fail_acl  # type: ignore[method-assign]

    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as raised:
        _filesystem(kernel).create_private_directory(r"C:\private\owner")

    assert raised.value.code == "cleanup_failed"
    cleanup = raised.value.take_cleanup_owner()
    assert cleanup is not None
    cleanup.close()
    assert kernel.deleted == [40]
    assert kernel.closed == [40]


@pytest.mark.parametrize(
    "signal",
    [KeyboardInterrupt("PRIVATE INTERRUPT"), SystemExit(23), asyncio.CancelledError()],
)
def test_control_flow_preserves_family_and_retains_failed_cleanup(
    signal: BaseException,
) -> None:
    kernel = FakeWindowsKernel()
    kernel.close_failures = 1

    def fail_write(handle: int, data: bytes) -> None:
        del handle, data
        raise signal

    kernel.write_all = fail_write  # type: ignore[method-assign]

    with pytest.raises(BaseException) as raised:
        _filesystem(kernel).create_private_file(
            r"C:\private\owner\server.json",
            b"{}",
        )

    assert raised.value is signal
    rendered = repr(raised.value) + str(raised.value)
    assert "PRIVATE" not in rendered
    if isinstance(signal, SystemExit):
        assert signal.code == 23
    cleanup = windows_artifact_fs.take_windows_artifact_cleanup_owner(signal)
    assert cleanup is not None
    assert windows_artifact_fs.take_windows_artifact_cleanup_owner(signal) is None
    cleanup.close()
    assert kernel.deleted == [40]
    assert kernel.closed == [40]


def test_close_control_flow_preserves_signal_and_exact_owner() -> None:
    kernel = FakeWindowsKernel()
    owner = _filesystem(kernel).create_private_directory(r"C:\private\owner")
    signal = KeyboardInterrupt("PRIVATE CLOSE CONTROL")
    original_close = kernel.close_handle

    def fail_close(handle: int) -> None:
        del handle
        raise signal

    kernel.close_handle = fail_close  # type: ignore[method-assign]

    with pytest.raises(BaseException) as raised:
        owner.close()

    assert raised.value is signal
    assert "PRIVATE" not in (repr(signal) + str(signal))
    assert windows_artifact_fs.take_windows_artifact_cleanup_owner(signal) is owner
    kernel.close_handle = original_close  # type: ignore[method-assign]
    owner.close()
    assert kernel.closed == [40]


def test_control_group_is_recursively_bounded_and_retains_cleanup() -> None:
    kernel = FakeWindowsKernel()
    kernel.close_failures = 1
    signal = BaseExceptionGroup(
        "PRIVATE GROUP PATH",
        [KeyboardInterrupt("PRIVATE CHILD"), SystemExit("PRIVATE EXIT")],
    )

    def fail_write(handle: int, data: bytes) -> None:
        del handle, data
        raise signal

    kernel.write_all = fail_write  # type: ignore[method-assign]

    with pytest.raises(BaseExceptionGroup) as raised:
        _filesystem(kernel).create_private_file(
            r"C:\private\owner\server.json",
            b"{}",
        )

    rendered = repr(raised.value) + str(raised.value)
    rendered += "".join(repr(child) + str(child) for child in raised.value.exceptions)
    assert "PRIVATE" not in rendered
    cleanup = windows_artifact_fs.take_windows_artifact_cleanup_owner(raised.value)
    assert cleanup is not None
    cleanup.close()
    assert kernel.deleted == [40]
    assert kernel.closed == [40]


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handles")
def test_native_windows_private_handles_round_trip_and_lock(tmp_path: Path) -> None:
    filesystem = windows_artifact_fs.NativeWindowsArtifactFilesystem()
    owner_path = tmp_path / "owned artifacts"
    file_path = owner_path / "server.json"

    directory = filesystem.create_private_directory(owner_path)
    first = filesystem.create_private_file(file_path, b'{"ok":true}', read_only=True)
    second = filesystem.open_file_no_reparse(file_path, writable=True)

    assert directory.privacy_posture == "windows_account_protected"
    assert first.privacy_posture == "windows_account_protected"
    assert first.read(64) == b'{"ok":true}'
    first.lock_exclusive_nonblocking()
    with pytest.raises(windows_artifact_fs.WindowsArtifactError) as busy:
        second.lock_exclusive_nonblocking()
    assert busy.value.code == "busy"
    first.unlock()
    second.close()
    first.delete_exact()
    first.close()
    assert not file_path.exists()
    directory.delete_exact()
    directory.close()
    assert not owner_path.exists()
