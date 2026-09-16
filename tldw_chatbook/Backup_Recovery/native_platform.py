"""Platform-native publication and persistence barriers for recovery storage."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from pathlib import Path


def flush_file(fd: int) -> None:
    """Persist file contents and metadata using the host's native barrier."""
    if os.name == "nt":
        from ..Utils.windows_files import flush_file as native_flush

        native_flush(fd)
        return
    os.fsync(fd)
    if platform.system() == "Darwin":
        import fcntl

        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)


def flush_directory(fd: int) -> None:
    """Persist directory changes; failures remain ambiguous to journal callers."""
    if os.name == "nt":
        from ..Utils.windows_files import flush_directory as native_flush

        native_flush(fd)
    else:
        flush_file(fd)


def rename_noreplace(
    source_parent: int, source: str, target_parent: int, target: str
) -> None:
    """Atomically move a file or directory while preserving an existing target."""
    if os.name == "nt":
        from ..Utils.windows_files import rename_noreplace as native_rename

        native_rename(source_parent, source, target_parent, target)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    system = platform.system()
    if system == "Darwin":
        fn, flags = libc.renameatx_np, 0x4  # RENAME_EXCL
    elif system == "Linux":
        fn, flags = libc.renameat2, 1  # RENAME_NOREPLACE
    else:
        raise OSError("native_publication_unavailable")
    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    fn.restype = ctypes.c_int
    if fn(
        source_parent, os.fsencode(source), target_parent, os.fsencode(target), flags
    ):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError("destination_exists")
        raise OSError(code, "native_publication_failed")


def _linux_mount_type(fd: int) -> str:
    """Resolve the pinned mount; detached or incomplete mount records fail closed."""
    try:
        descriptor = Path(f"/proc/self/fdinfo/{fd}").read_text()
        mount_id = next(
            line.split()[1]
            for line in descriptor.splitlines()
            if line.startswith("mnt_id:")
        )
        mount = next(
            line
            for line in Path("/proc/self/mountinfo").read_text().splitlines()
            if line.split()[0] == mount_id
        )
        return mount.split(" - ", 1)[1].split()[0]
    except (StopIteration, IndexError, UnicodeError) as error:
        raise OSError("native_mount_identity_unavailable") from error


def linux_identity(fd: int) -> dict[str, str | int]:
    """Read filesystem type from fstatfs and flags from the pinned descriptor."""
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise OSError("native_rename_unavailable")
    # f_type is the first native long on Linux; reserve more than the complete
    # statfs structure to avoid coupling subsequent fields to a specific ABI.
    data = (ctypes.c_long * 128)()
    fn = libc.fstatfs
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    if fn(fd, ctypes.byref(data)):
        raise OSError(ctypes.get_errno(), "native_identity_unavailable")
    # ext2/ext3/ext4 share f_type. Bind the mount-table type to this descriptor's
    # mount ID instead of labelling every EXT superblock as ext4.
    filesystem = _linux_mount_type(fd)
    expected = {"ext4": 0xEF53, "xfs": 0x58465342, "btrfs": 0x9123683E}
    if expected.get(filesystem) != data[0] & 0xFFFFFFFF:
        filesystem = "unsupported"
    return {
        "os": "Linux",
        "release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "filesystem": filesystem,
        "flags": os.fstatvfs(fd).f_flag,
    }
