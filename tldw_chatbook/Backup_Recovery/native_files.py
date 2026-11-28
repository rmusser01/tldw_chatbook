"""Pinned private storage and native no-replace publication (ADR-126)."""

from __future__ import annotations

import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from .native_platform import flush_directory, flush_file, rename_noreplace
from .qualification import _qualified_identity, native_identity, qualified_for


#: TASK-40608 D1 (owner decisions O1, O3), Windows only; the same rule as
#: ``Utils.private_paths._walk_verified_parent``. A pinned profile folder and
#: every folder above it cannot be renamed, replaced or deleted while the pin
#: is held, so once a full walk here has checked the folders above a pin, a
#: later walk to that pin or below it may start at the pin for this long. Only
#: an owner or permission change on a folder above the pin can go unseen, and
#: only inside this window. The pinned folder and everything below it are
#: checked on every walk, as before. Values: pin path -> (pin release
#: generation, time.monotonic() when the full walk began).
#:
#: Measured 2026-11-26: the admission folder is walked about 200 times in a
#: Console turn, 12 folders each from the drive root: 3,600 native opens and as
#: many checked stats to reach a folder two levels below a held pin.
_PINNED_ANCESTORS_SECONDS = 1.0
_pinned_ancestors: dict[str, tuple[int, float]] = {}
_pinned_ancestors_lock = threading.Lock()


def _pinned_walk_start(root: Path):
    """The held pin a walk to ``root`` passes through or ends at, if any.

    Returns:
        ``(pin path, components covered, release generation)`` when a held pin
        is ``root`` or a folder above it under the same spelling, else None.
    """
    if sys.platform != "win32":
        return None
    from tldw_chatbook.Utils.windows_files import pinned_start

    found = pinned_start(root)
    if found is None:
        return None
    pin_path, covered, generation = found
    if tuple(root.parts[1 : 1 + len(covered)]) != covered:
        return None
    return pin_path, covered, generation


def _open_at_pin(root: Path, pinned, started: float, flags: int) -> int | None:
    """Open the pin a recent full walk went through, or None to walk in full."""
    with _pinned_ancestors_lock:
        known = _pinned_ancestors.get(pinned[0])
    if (
        known is None
        or known[0] != pinned[2]
        or not started - known[1] < _PINNED_ANCESTORS_SECONDS
    ):
        return None
    try:
        fd = os.open(pinned[0], flags)
    except OSError:
        return None
    # The pin must still be held: the folders above it stay put only while it
    # is, and a released pin is opened by name like any other folder.
    if _pinned_walk_start(root) != pinned:
        os.close(fd)
        return None
    return fd


@contextmanager
def pinned_directory(
    root: Path, *, _visit: Callable[[int], None] | None = None
) -> Iterator[int]:
    """Pin a trusted directory; optionally visit each validated ancestor.

    The internal visitor borrows each descriptor only during its call. It must
    not close or retain it. Visitor failures propagate through native cleanup.
    """
    if not root.is_absolute() or ".." in root.parts:
        raise OSError("absolute_directory_required")
    if not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise OSError("native_nofollow_unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    # A visitor is shown every folder of the chain, so its walk is always full.
    pinned = _pinned_walk_start(root) if _visit is None else None
    started = time.monotonic()
    fd = _open_at_pin(root, pinned, started, flags) if pinned is not None else None
    skipped = 0 if fd is None else len(pinned[1])
    if fd is None:
        fd = os.open(root.anchor, flags)
    try:
        for component in root.parts[1 + skipped :]:
            info = os.fstat(fd)
            if info.st_uid not in (0, os.geteuid()) or (
                info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX
            ):
                raise OSError("unsafe_directory")
            if _visit is not None:
                _visit(fd)
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid not in (0, os.geteuid()) or (
            info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX
        ):
            raise OSError("unsafe_directory")
        if _visit is not None:
            _visit(fd)
        if pinned is not None and not skipped:
            # This walk checked every folder above the pin.
            with _pinned_ancestors_lock:
                _pinned_ancestors[pinned[0]] = (pinned[2], started)
        yield fd
    finally:
        os.close(fd)


def create_private_directory(destination: Path) -> None:
    """Create one new owner-private directory under a pinned existing parent."""
    with pinned_directory(destination.parent) as parent:
        os.mkdir(destination.name, mode=0o700, dir_fd=parent)
        flush_directory(parent)


@contextmanager
def create_private_file(destination: Path) -> Iterator[int]:
    """Exclusively create a private regular file; flush on successful close."""
    with pinned_directory(destination.parent) as parent:
        fd = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        try:
            yield fd
            os.fsync(fd)
            flush_directory(parent)
        finally:
            os.close(fd)


def _rename_new(
    source_parent: int, source: str, target_parent: int, target: str
) -> None:
    """Native primitive, also exercised directly by the qualification suite."""
    rename_noreplace(source_parent, source, target_parent, target)


def _flush_private_tree(fd: int, device: int) -> None:
    """Flush only owned no-follow objects on one volume, leaves before parents."""
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_dev != device:
        raise OSError("staged_tree_not_private")
    if stat.S_ISREG(info.st_mode):
        if info.st_nlink != 1:
            raise OSError("staged_tree_linked")
        os.fchmod(fd, 0o600)
        flush_file(fd)
        return
    if not stat.S_ISDIR(info.st_mode):
        raise OSError("staged_tree_not_regular")
    os.fchmod(fd, 0o700)
    for name in os.listdir(fd):
        child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            _flush_private_tree(child, device)
        finally:
            os.close(child)
    flush_directory(fd)


def publish_new(
    staged: Path,
    destination: Path,
    *,
    parent_identities: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> None:
    """Durably move an operation-owned file/directory without replacing any name.

    Callers own and freeze staged bytes/tree, and journal intent before calling.
    A flush failure after rename is ambiguous: preserve/journal actual evidence;
    never delete a destination or retry using replace. No hard-link fallback.
    """
    allowed, reason = qualified_for("publish_new", destination.parent)
    if not allowed:
        raise OSError(reason)
    with (
        pinned_directory(staged.parent) as source_parent,
        pinned_directory(destination.parent) as target_parent,
    ):
        if parent_identities is not None:
            actual = tuple(
                (info.st_dev, info.st_ino)
                for info in (os.fstat(source_parent), os.fstat(target_parent))
            )
            if actual != parent_identities:
                raise OSError("publication_parent_changed")
        allowed, reason = _qualified_identity(
            "publish_new", native_identity(target_parent)
        )
        if not allowed:
            raise OSError(reason)
        before = os.stat(staged.name, dir_fd=source_parent, follow_symlinks=False)
        directory = stat.S_ISDIR(before.st_mode)
        operation = "publish_directory" if directory else "publish_file"
        allowed, reason = _qualified_identity(operation, native_identity(target_parent))
        if not allowed:
            raise OSError(reason)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        fd = os.open(staged.name, flags, dir_fd=source_parent)
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != (
                before.st_dev,
                before.st_ino,
            ) or info.st_uid != os.geteuid():
                raise OSError("staged_identity_changed")
            if not (stat.S_ISREG(info.st_mode) or directory) or (
                not directory and info.st_nlink != 1
            ):
                raise OSError("staged_not_private_regular_object")
            if info.st_dev != os.fstat(target_parent).st_dev:
                raise OSError("cross_volume_publication_unqualified")
            _flush_private_tree(fd, info.st_dev)
            if os.stat(
                staged.name, dir_fd=source_parent, follow_symlinks=False
            ) != os.fstat(fd):
                raise OSError("staged_identity_changed")
            _rename_new(source_parent, staged.name, target_parent, destination.name)
            flush_directory(target_parent)
            flush_directory(source_parent)
        finally:
            os.close(fd)
