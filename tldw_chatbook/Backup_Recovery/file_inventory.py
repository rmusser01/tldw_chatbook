"""Bounded metadata enumeration of an exact root, without capture authority."""

import hashlib
import os
from pathlib import Path
import stat
import sys
import unicodedata

from tldw_chatbook.Utils.path_validation import validate_recovery_relative_path

from . import bootstrap
from .models import FileMetadata, StorageItem

MAX_ENTRIES = 100_000
MAX_PATH_BYTES = 1_024
MAX_EXPANDED_BYTES = 1024**4
MAX_MEMBER_BYTES = 256 * 1024**3


def _metadata_supported(fd: int, info: os.stat_result) -> bool:
    """Detect unsupported host metadata without loading content or engines.

    macOS ACLs have a native interface separate from listxattr. Linux POSIX ACLs
    are extended attributes. Other platforms remain capability-unavailable.
    """
    if sys.platform not in {"darwin", "linux"}:
        return False
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o7000
        or getattr(info, "st_flags", 0)
    ):
        return False
    if sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.flistxattr.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        libc.flistxattr.restype = ctypes.c_ssize_t
        if libc.flistxattr(fd, None, 0, 0) != 0:
            return False
        libc.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.acl_get_fd_np.restype = ctypes.c_void_p
        libc.acl_get_entry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        libc.acl_get_entry.restype = ctypes.c_int
        libc.acl_free.argtypes = [ctypes.c_void_p]
        libc.acl_free.restype = ctypes.c_int
        acl = libc.acl_get_fd_np(fd, 0x100)  # ACL_TYPE_EXTENDED, installed Darwin ABI
        if not acl:
            return ctypes.get_errno() == 2  # ENOENT: no extended ACL on this inode
        try:
            entry = ctypes.c_void_p()
            result = libc.acl_get_entry(acl, 0, ctypes.byref(entry))
            # Darwin uses 0 for an entry, -1/EINVAL for end of the ACL.
            return result == -1 and ctypes.get_errno() == 22
        finally:
            if libc.acl_free(acl) != 0:
                raise OSError("metadata_unavailable")
    return hasattr(os, "listxattr") and not os.listxattr(fd)


def inventory_tree(
    root: Path, *, owner: str, external: bool
) -> tuple[StorageItem, ...]:
    """Enumerate metadata with pinned no-follow traversal and explicit refusals.

    IDs are tree-local; installed callers apply the selected profile context.
    Child dependencies preserve directory topology. This does not qualify external
    metadata or grant filesystem authority. Metadata is preview evidence only.
    """
    return _inventory_tree(root, owner=owner, external=external)


def _inventory_root(root: Path, *, owner: str, external: bool) -> StorageItem:
    """Inspect one pinned root's kind/metadata without visiting its children."""
    return _inventory_tree(root, owner=owner, external=external, root_only=True)[0]


def _inventory_tree(
    root: Path,
    *,
    owner: str,
    external: bool,
    selected_paths: frozenset[str] | None = None,
    root_only: bool = False,
) -> tuple[StorageItem, ...]:
    if type(root_only) is not bool or (root_only and selected_paths is not None):
        raise ValueError("invalid_root_inspection")
    if selected_paths is not None:
        if type(selected_paths) is not frozenset or len(selected_paths) > MAX_ENTRIES:
            raise ValueError("invalid_selected_paths")
        selected = {""}
        folded = {}
        for relative in selected_paths:
            validate_recovery_relative_path(relative)
            if not relative:
                raise ValueError("invalid_selected_paths")
            for path in (Path(relative), *Path(relative).parents):
                name = "" if str(path) == "." else path.as_posix()
                normalized = unicodedata.normalize("NFC", name).casefold()
                if normalized in folded and folded[normalized] != name:
                    raise ValueError("tree_path_collision")
                folded[normalized] = name
                selected.add(name)
                if len(selected) > MAX_ENTRIES:
                    raise ValueError("tree_inventory_limit")
        if not selected_paths:
            return ()
    else:
        selected = None
    root = Path(os.path.abspath(root))
    root_id = owner + ":" + hashlib.sha256(os.fsencode(root)).hexdigest()
    items = []
    total = 0
    root_device = None

    def walk(parent, leaf, relative, parent_id, depth):
        nonlocal total, root_device
        logical_id = (
            root_id
            if not relative
            else root_id + ":" + hashlib.sha256(relative.encode()).hexdigest()
        )
        validate_recovery_relative_path(relative)
        dependencies = (parent_id,) if parent_id else ()
        path = root / relative
        if (
            len(items) >= MAX_ENTRIES
            or depth > 64
            or len(relative.encode()) > MAX_PATH_BYTES
        ):
            raise ValueError("tree_inventory_limit")
        try:
            info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            items.append(
                StorageItem(
                    owner,
                    logical_id,
                    path,
                    "unused" if not relative and not external else "unavailable",
                    dependencies,
                )
            )
            return
        if root_device is None:
            root_device = info.st_dev
        regular = stat.S_ISREG(info.st_mode)
        directory = stat.S_ISDIR(info.st_mode)
        if (
            (not regular and not directory)
            or (regular and info.st_nlink != 1)
            or info.st_dev != root_device
        ):
            items.append(
                StorageItem(owner, logical_id, path, "unsupported", dependencies)
            )
            return
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        metadata_fd = os.open(leaf, flags, dir_fd=parent)
        try:
            held = os.fstat(metadata_fd)
            if (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
            ) != (
                held.st_dev,
                held.st_ino,
                held.st_mode,
                held.st_size,
                held.st_mtime_ns,
            ):
                raise ValueError("tree_inventory_changed")
            supported = _metadata_supported(metadata_fd, held)
        finally:
            os.close(metadata_fd)
        metadata = FileMetadata(
            1,
            root_id,
            relative,
            parent_id,
            "directory" if directory else "file",
            stat.S_IMODE(info.st_mode) & 0o777,
            info.st_mtime_ns,
            "external" if external else "private",
        )
        if regular:
            total += info.st_size
            status = (
                "included"
                if supported
                and info.st_size <= MAX_MEMBER_BYTES
                and total <= MAX_EXPANDED_BYTES
                else "unsupported"
            )
            items.append(
                StorageItem(
                    owner, logical_id, path, status, dependencies, metadata=metadata
                )
            )
            return
        items.append(
            StorageItem(
                owner,
                logical_id,
                path,
                "included_directory" if supported else "unsupported",
                dependencies,
                metadata=metadata,
            )
        )
        if root_only:
            return
        fd = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            held = os.fstat(fd)
            if (info.st_dev, info.st_ino) != (held.st_dev, held.st_ino):
                raise ValueError("tree_inventory_changed")
            if selected is None:
                with os.scandir(fd) as entries:
                    children = []
                    for entry in entries:
                        if len(children) + len(items) >= MAX_ENTRIES:
                            raise ValueError("tree_inventory_limit")
                        children.append(entry.name)
            else:
                children = [
                    Path(path).name
                    for path in selected
                    if path
                    and (
                        ""
                        if str(Path(path).parent) == "."
                        else Path(path).parent.as_posix()
                    )
                    == relative
                ]
            folded = {}
            for child in children:
                key = unicodedata.normalize("NFC", child).casefold()
                if (
                    key in folded
                    or "\\" in child
                    or ":" in child
                    or any(ord(c) < 32 for c in child)
                ):
                    raise ValueError("tree_path_collision")
                folded[key] = child
            for child in sorted(children):
                walk(fd, child, str(Path(relative) / child), logical_id, depth + 1)
            current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if (info.st_dev, info.st_ino, info.st_mtime_ns) != (
                current.st_dev,
                current.st_ino,
                current.st_mtime_ns,
            ):
                raise ValueError("tree_inventory_changed")
        finally:
            os.close(fd)

    try:
        with bootstrap.pinned_directory(root.parent) as parent:
            walk(parent, root.name, "", None, 0)
    except (OSError, ValueError, RuntimeError):
        return (StorageItem(owner, root_id, root, "unavailable", ()),)
    return tuple(items)
