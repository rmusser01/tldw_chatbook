"""Exact descriptor-relative namespace transitions for profile migration."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
from pathlib import Path

from tldw_chatbook.Utils import private_paths


_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAME_NOREPLACE = 1 if sys.platform.startswith("linux") else 0x00000004


def _same_parent(current: os.stat_result, expected: os.stat_result) -> bool:
    return (
        private_paths._same_identity(current, expected)
        and stat.S_IFMT(current.st_mode) == stat.S_IFMT(expected.st_mode)
        and stat.S_IMODE(current.st_mode) == stat.S_IMODE(expected.st_mode)
        and current.st_uid == expected.st_uid
        and stat.S_ISDIR(current.st_mode)
        and current.st_uid == os.geteuid()
        and stat.S_IMODE(current.st_mode) == 0o700
    )


def _valid_file(value: os.stat_result, *, links: frozenset[int]) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and value.st_uid == os.geteuid()
        and stat.S_IMODE(value.st_mode) == 0o600
        and value.st_nlink in links
    )


def _rename_noreplace(
    parent_fd: int,
    source_leaf: str,
    destination_leaf: str,
) -> None:
    source = os.fsencode(source_leaf)
    destination = os.fsencode(destination_leaf)
    if sys.platform == "darwin" and hasattr(_LIBC, "renameatx_np"):
        result = _LIBC.renameatx_np(
            parent_fd,
            source,
            parent_fd,
            destination,
            _RENAME_NOREPLACE,
        )
    elif hasattr(_LIBC, "renameat2"):
        result = _LIBC.renameat2(
            parent_fd,
            source,
            parent_fd,
            destination,
            _RENAME_NOREPLACE,
        )
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename unavailable")
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _open_parent(path: Path, expected: os.stat_result) -> int:
    parent_fd, _leaf = private_paths._open_verified_parent(
        path,
        missing_leaf_allowed=True,
    )
    if not _same_parent(os.fstat(parent_fd), expected):
        os.close(parent_fd)
        raise ValueError
    return parent_fd


def _sidecars_absent(parent_fd: int, leaf: str) -> bool:
    for suffix in ("-wal", "-shm", "-journal"):
        try:
            os.stat(f"{leaf}{suffix}", dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        return False
    return True


def move_exact_noreplace(
    source: Path,
    destination: Path,
    *,
    parent_identity: os.stat_result,
    file_identity: os.stat_result,
    allowed_links: frozenset[int] = frozenset({1}),
) -> os.stat_result:
    """Atomically move the exact source inode or restore a substituted leaf."""

    if source.parent != destination.parent or source.name == destination.name:
        raise ValueError
    parent_fd = _open_parent(source, parent_identity)
    file_fd = -1
    moved = False
    try:
        file_fd = os.open(
            source.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_fd)
        entry = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(opened, file_identity)
            or not private_paths._same_identity(entry, file_identity)
            or not _valid_file(opened, links=allowed_links)
            or not _valid_file(entry, links=allowed_links)
            or not _sidecars_absent(parent_fd, source.name)
            or not _sidecars_absent(parent_fd, destination.name)
        ):
            raise ValueError
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError
        _rename_noreplace(parent_fd, source.name, destination.name)
        moved = True
        current = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(current, opened)
            or not private_paths._same_identity(current, file_identity)
            or not _valid_file(current, links=allowed_links)
        ):
            try:
                _rename_noreplace(parent_fd, destination.name, source.name)
                moved = False
                os.fsync(parent_fd)
            except BaseException:
                pass
            raise ValueError
        os.fsync(parent_fd)
        reopened = _open_parent(source, parent_identity)
        try:
            if os.fstat(reopened).st_nlink != os.fstat(parent_fd).st_nlink:
                raise ValueError
        finally:
            os.close(reopened)
        return current
    except BaseException:
        if moved:
            # A verified exact object remains preserved at the destination.
            try:
                os.fsync(parent_fd)
            except BaseException:
                pass
        raise
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def remove_exact(
    path: Path,
    *,
    parent_identity: os.stat_result,
    file_identity: os.stat_result,
    allowed_links: frozenset[int] = frozenset({1}),
) -> None:
    """Quarantine then remove the exact inode, preserving a swapped source."""

    holding = path.with_name(f".{path.name}.migration-hold")
    try:
        moved = move_exact_noreplace(
            path,
            holding,
            parent_identity=parent_identity,
            file_identity=file_identity,
            allowed_links=allowed_links,
        )
    except FileNotFoundError:
        parent_fd = _open_parent(holding, parent_identity)
        try:
            moved = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not private_paths._same_identity(moved, file_identity)
                or not _valid_file(moved, links=allowed_links)
                or not _sidecars_absent(parent_fd, holding.name)
            ):
                raise ValueError
        finally:
            os.close(parent_fd)
    parent_fd = _open_parent(holding, parent_identity)
    try:
        current = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
        if not private_paths._same_identity(current, moved) or not _valid_file(
            current, links=allowed_links
        ):
            raise ValueError
        os.unlink(holding.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        reopened = _open_parent(path, parent_identity)
        try:
            if os.fstat(reopened).st_nlink != os.fstat(parent_fd).st_nlink:
                raise ValueError
        finally:
            os.close(reopened)
    finally:
        os.close(parent_fd)


__all__ = ["move_exact_noreplace", "remove_exact"]
