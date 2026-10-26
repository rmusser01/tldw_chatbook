"""Pinned private storage and native no-replace publication (ADR-126)."""

from __future__ import annotations

import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from .native_platform import flush_directory, flush_file, rename_noreplace
from .qualification import _qualified_identity, native_identity, qualified_for


@contextmanager
def pinned_directory(root: Path, *, _custody=None) -> Iterator[int]:
    """Pin a trusted absolute directory with no symlink components."""
    if not root.is_absolute() or ".." in root.parts:
        raise OSError("absolute_directory_required")
    if not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise OSError("native_nofollow_unavailable")
    fd = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    if _custody is not None:
        _custody.fds.append(fd)
    try:
        for component in root.parts[1:]:
            info = os.fstat(fd)
            if info.st_uid not in (0, os.geteuid()) or (
                info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX
            ):
                raise OSError("unsafe_directory")
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            if _custody is not None:
                _custody.fds.append(child)
            _close_owned(fd, _custody)
            fd = child
        info = os.fstat(fd)
        if info.st_uid not in (0, os.geteuid()) or (
            info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX
        ):
            raise OSError("unsafe_directory")
        yield fd
    finally:
        _close_owned(fd, _custody)


def create_private_directory(destination: Path, *, _custody=None) -> None:
    """Create one new owner-private directory under a pinned existing parent."""
    pinned = (
        pinned_directory(destination.parent)
        if _custody is None
        else pinned_directory(destination.parent, _custody=_custody)
    )
    with pinned as parent:
        os.mkdir(destination.name, mode=0o700, dir_fd=parent)
        flush_directory(parent)


@contextmanager
def create_private_file(destination: Path, *, _custody=None) -> Iterator[int]:
    """Exclusively create a private regular file; flush on successful close."""
    with (
        pinned_directory(destination.parent, _custody=_custody) as parent,
        _create_private_file_at(parent, destination.name, _custody=_custody) as fd,
    ):
        yield fd


@contextmanager
def _create_private_file_at(parent: int, name: str, *, _custody=None) -> Iterator[int]:
    """Create a native private leaf under an already owned parent pin."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("private_leaf_required")
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent,
    )
    if _custody is not None:
        _custody.fds.append(fd)
    try:
        yield fd
        os.fsync(fd)
        flush_directory(parent)
    finally:
        _close_owned(fd, _custody)


def _rename_new(
    source_parent: int, source: str, target_parent: int, target: str
) -> None:
    """Native primitive, also exercised directly by the qualification suite."""
    rename_noreplace(source_parent, source, target_parent, target)


def _close_owned(fd, custody):
    if custody is None:
        os.close(fd)
    else:
        custody.close_descriptor(fd)
        custody.fds.remove(fd)


def _flush_private_tree(fd: int, device: int, *, _custody=None) -> None:
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
        if _custody is not None:
            _custody.fds.append(child)
        try:
            _flush_private_tree(child, device, _custody=_custody)
        finally:
            _close_owned(child, _custody)
    flush_directory(fd)


def publish_new(
    staged: Path,
    destination: Path,
    *,
    parent_identities: tuple[tuple[int, int], tuple[int, int]] | None = None,
    _custody=None,
) -> None:
    """Durably move an operation-owned file/directory without replacing any name.

    Callers own and freeze staged bytes/tree, and journal intent before calling.
    A flush failure after rename is ambiguous: preserve/journal actual evidence;
    never delete a destination or retry using replace. No hard-link fallback.
    """
    allowed, reason = (
        qualified_for("publish_new", destination.parent)
        if _custody is None
        else qualified_for("publish_new", destination.parent, _custody=_custody)
    )
    if not allowed:
        raise OSError(reason)
    with (
        pinned_directory(staged.parent, _custody=_custody) as source_parent,
        pinned_directory(destination.parent, _custody=_custody) as target_parent,
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
        if _custody is not None:
            _custody.fds.append(fd)
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
            _flush_private_tree(fd, info.st_dev, _custody=_custody)
            if os.stat(
                staged.name, dir_fd=source_parent, follow_symlinks=False
            ) != os.fstat(fd):
                raise OSError("staged_identity_changed")
            _rename_new(source_parent, staged.name, target_parent, destination.name)
            flush_directory(target_parent)
            flush_directory(source_parent)
        finally:
            _close_owned(fd, _custody)
