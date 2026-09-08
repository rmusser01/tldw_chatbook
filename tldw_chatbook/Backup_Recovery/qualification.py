"""Read-only, operation-specific native evidence. No runtime self-qualification."""

from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path


# Darwin's statfs64 ABI from the installed sys/mount.h; no pathname shell probe.
class _StatFS(ctypes.Structure):
    _fields_ = [
        ("bsize", ctypes.c_uint32),
        ("iosize", ctypes.c_int32),
        ("counts", ctypes.c_uint64 * 5),
        ("fsid", ctypes.c_int32 * 2),
        ("owner", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subtype", ctypes.c_uint32),
        ("fstype", ctypes.c_char * 16),
        ("mount", ctypes.c_char * 1024),
        ("source", ctypes.c_char * 1024),
        ("reserved", ctypes.c_uint32 * 8),
    ]


def native_identity(fd: int) -> dict[str, str | int]:
    """Read current native identity from a pinned directory descriptor."""
    if platform.system() != "Darwin":
        raise OSError("native_platform_unqualified")
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.fstatfs
    fn.argtypes = [ctypes.c_int, ctypes.POINTER(_StatFS)]
    fn.restype = ctypes.c_int
    info = _StatFS()
    if fn(fd, ctypes.byref(info)) != 0:
        raise OSError("native_identity_unavailable")
    if not hasattr(libc, "renameatx_np"):
        raise OSError("native_rename_unavailable")
    return {
        "os": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "filesystem": info.fstype.decode("ascii"),
        "flags": info.flags,
    }


def _qualified_identity(
    operation: str, identity: dict[str, str | int]
) -> tuple[bool, str]:
    evidence = json.loads(
        Path(__file__).with_name("native_qualification.json").read_text()
    )
    if evidence.get("schema_version") != 1:
        return False, "qualification_evidence_invalid"
    for row in evidence["evidence"]:
        if row["identity"] == identity and operation in row["operations"]:
            return True, "qualified_native_evidence"
    return False, "operation_not_qualified"


def qualified_for(operation: str, root: Path) -> tuple[bool, str]:
    """Match installed test evidence and current identity without writing probes."""
    from .native_files import pinned_directory

    try:
        with pinned_directory(root) as fd:
            return _qualified_identity(operation, native_identity(fd))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False, "qualification_unavailable"
