"""Bounded metadata enumeration of an exact root, without capture authority."""

import hashlib
import os
from pathlib import Path
import stat

from . import bootstrap
from .models import StorageItem

MAX_ENTRIES = 100_000
MAX_PATH_BYTES = 1_024
MAX_EXPANDED_BYTES = 1024**4
MAX_MEMBER_BYTES = 256 * 1024**3


def inventory_tree(
    root: Path, *, owner: str, external: bool
) -> tuple[StorageItem, ...]:
    """Enumerate metadata with pinned no-follow traversal and explicit refusals.

    IDs are tree-local; installed callers apply the selected profile context.
    Child dependencies preserve directory topology. This does not qualify external
    metadata or grant filesystem authority; the task9 policy remains necessary.
    """
    root = Path(os.path.abspath(root))
    root_id = owner + ":" + hashlib.sha256(os.fsencode(root)).hexdigest()
    items = []
    total = 0

    def walk(parent, leaf, relative, parent_id, depth):
        nonlocal total
        logical_id = (
            root_id
            if not relative
            else root_id + ":" + hashlib.sha256(relative.encode()).hexdigest()
        )
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
                    "unused" if not relative else "unavailable",
                    dependencies,
                )
            )
            return
        if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            total += info.st_size
            status = (
                "included"
                if info.st_size <= MAX_MEMBER_BYTES and total <= MAX_EXPANDED_BYTES
                else "unsupported"
            )
            items.append(StorageItem(owner, logical_id, path, status, dependencies))
            return
        if not stat.S_ISDIR(info.st_mode):
            items.append(
                StorageItem(owner, logical_id, path, "unsupported", dependencies)
            )
            return
        items.append(
            StorageItem(owner, logical_id, path, "included_directory", dependencies)
        )
        fd = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            held = os.fstat(fd)
            if (info.st_dev, info.st_ino) != (held.st_dev, held.st_ino):
                raise ValueError("tree_inventory_changed")
            with os.scandir(fd) as entries:
                children = []
                for entry in entries:
                    if len(children) + len(items) >= MAX_ENTRIES:
                        raise ValueError("tree_inventory_limit")
                    children.append(entry.name)
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
