"""Exact descriptor-relative namespace transitions for profile migration."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tldw_chatbook.Utils import private_paths


_LIBC = None if sys.platform == "win32" else ctypes.CDLL(None, use_errno=True)
_RENAME_NOREPLACE = 1 if sys.platform.startswith("linux") else 0x00000004


class MigrationTombstoneKey(Enum):
    """Closed finite keys for non-authoritative migration tombstones."""

    JOURNAL = "journal"
    ACTIVE_CANDIDATE = "active-candidate"
    ACTIVE_ROLLBACK = "active-rollback"
    PRE_V3_CANDIDATE = "pre-v3-candidate"
    PRE_V3_ROLLBACK = "pre-v3-rollback"
    PRE_V4_CANDIDATE = "pre-v4-candidate"
    PRE_V4_ROLLBACK = "pre-v4-rollback"
    LIVE_WAL = "live-wal"
    LIVE_SHM = "live-shm"


@dataclass(frozen=True, slots=True, repr=False)
class ParentAuthority:
    """Immutable exact parent metadata pinned before namespace mutation."""

    identity: os.stat_result

    def __repr__(self) -> str:
        return "ParentAuthority(<private>)"


def _same_parent(current: os.stat_result, expected: os.stat_result) -> bool:
    return (
        private_paths._same_identity(current, expected)
        and stat.S_IFMT(current.st_mode) == stat.S_IFMT(expected.st_mode)
        and stat.S_IMODE(current.st_mode) == stat.S_IMODE(expected.st_mode)
        and current.st_uid == expected.st_uid
        and current.st_nlink == expected.st_nlink
        and stat.S_ISDIR(current.st_mode)
        and current.st_uid == os.geteuid()
        and stat.S_IMODE(current.st_mode) == 0o700
    )


def _same_parent_with_link_delta(
    current: os.stat_result,
    expected: os.stat_result,
    *,
    link_delta: int,
) -> bool:
    return (
        private_paths._same_identity(current, expected)
        and stat.S_IFMT(current.st_mode) == stat.S_IFMT(expected.st_mode)
        and stat.S_IMODE(current.st_mode) == stat.S_IMODE(expected.st_mode)
        and current.st_uid == expected.st_uid
        and current.st_nlink == expected.st_nlink + link_delta
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


def rename_noreplace_at(
    parent_fd: int,
    source_leaf: str,
    destination_leaf: str,
) -> None:
    """Atomically move one sibling leaf without replacing its destination."""

    _rename_noreplace(parent_fd, source_leaf, destination_leaf)


def _open_parent(path: Path, authority: ParentAuthority) -> int:
    parent_fd, _leaf = private_paths._open_verified_parent(
        path,
        missing_leaf_allowed=True,
    )
    if not _same_parent(os.fstat(parent_fd), authority.identity):
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


def _holding_path(path: Path, key: MigrationTombstoneKey) -> Path:
    if type(key) is not MigrationTombstoneKey:
        raise TypeError
    return path.with_name(f".profile-migration-{key.value}.tombstone")


def require_reusable_tombstone(
    path: Path,
    *,
    parent_authority: ParentAuthority,
    tombstone_key: MigrationTombstoneKey,
) -> None:
    """Require an existing bounded tombstone to remain exact and unaliased."""

    holding = _holding_path(path, tombstone_key)
    parent_fd = _open_parent(path, parent_authority)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                holding.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return
        opened = os.fstat(descriptor)
        named = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(opened, named)
            or not _valid_file(opened, links=frozenset({1}))
            or not _valid_file(named, links=frozenset({1}))
            or not _sidecars_absent(parent_fd, holding.name)
        ):
            raise ValueError
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def admit_zero_reusable_tombstone(
    path: Path,
    *,
    parent_authority: ParentAuthority,
    tombstone_key: MigrationTombstoneKey,
) -> os.stat_result:
    """Admit one restart-surviving zero tombstone through retained descriptors."""

    holding = _holding_path(path, tombstone_key)
    parent_fd = _open_parent(path, parent_authority)
    descriptor = -1
    try:
        descriptor = os.open(
            holding.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        named = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            opened.st_size != 0
            or named.st_size != 0
            or not private_paths._same_identity(opened, named)
            or not _valid_file(opened, links=frozenset({1}))
            or not _valid_file(named, links=frozenset({1}))
            or not _sidecars_absent(parent_fd, holding.name)
        ):
            raise ValueError
        os.fsync(descriptor)
        settled = os.fstat(descriptor)
        current = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            settled.st_size != 0
            or current.st_size != 0
            or not private_paths._same_identity(settled, opened)
            or not private_paths._same_identity(current, opened)
            or not _valid_file(settled, links=frozenset({1}))
            or not _valid_file(current, links=frozenset({1}))
        ):
            raise ValueError
        os.fsync(parent_fd)
        return settled
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def prepare_reusable_tombstone(
    path: Path,
    *,
    parent_authority: ParentAuthority,
    file_identity: os.stat_result,
    source_key: MigrationTombstoneKey,
    destination_key: MigrationTombstoneKey,
) -> os.stat_result:
    """Move exact retained cleanup evidence into one zero reusable leaf."""

    source = _holding_path(path, source_key)
    destination = _holding_path(path, destination_key)
    parent_fd = _open_parent(path, parent_authority)
    descriptor = -1
    try:
        descriptor = os.open(
            source.name,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        named = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(opened, file_identity)
            or not private_paths._same_identity(named, file_identity)
            or not _valid_file(opened, links=frozenset({1}))
            or not _valid_file(named, links=frozenset({1}))
            or not _sidecars_absent(parent_fd, source.name)
        ):
            raise ValueError
        if source != destination:
            _rename_noreplace(parent_fd, source.name, destination.name)
            os.fsync(parent_fd)
        current = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
        if (
            not private_paths._same_identity(opened, file_identity)
            or not private_paths._same_identity(current, file_identity)
            or not _valid_file(opened, links=frozenset({1}))
            or not _valid_file(current, links=frozenset({1}))
            or not _sidecars_absent(parent_fd, destination.name)
        ):
            raise ValueError
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
        settled = os.fstat(descriptor)
        current = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            settled.st_size != 0
            or current.st_size != 0
            or not private_paths._same_identity(settled, file_identity)
            or not private_paths._same_identity(current, file_identity)
            or not _valid_file(settled, links=frozenset({1}))
            or not _valid_file(current, links=frozenset({1}))
        ):
            raise ValueError
        os.fsync(parent_fd)
        return settled
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def remove_zero_reusable_tombstone(
    path: Path,
    *,
    parent_authority: ParentAuthority,
    file_identity: os.stat_result,
    tombstone_key: MigrationTombstoneKey,
) -> None:
    """Remove exact zero cleanup evidence so its fixed slot can be refilled."""

    holding = _holding_path(path, tombstone_key)
    parent_fd = _open_parent(path, parent_authority)
    descriptor = -1
    try:
        descriptor = os.open(
            holding.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        named = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            opened.st_size != 0
            or named.st_size != 0
            or not private_paths._same_identity(opened, file_identity)
            or not private_paths._same_identity(named, file_identity)
            or not _valid_file(opened, links=frozenset({1}))
            or not _valid_file(named, links=frozenset({1}))
        ):
            raise ValueError
        os.unlink(holding.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def move_exact_noreplace(
    source: Path,
    destination: Path,
    *,
    parent_authority: ParentAuthority,
    file_identity: os.stat_result,
    allowed_links: frozenset[int] = frozenset({1}),
) -> os.stat_result:
    """Atomically move the exact source inode or restore a substituted leaf."""

    if source.parent != destination.parent or source.name == destination.name:
        raise ValueError
    parent_fd = _open_parent(source, parent_authority)
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
        deferred: BaseException | None = None
        try:
            _rename_noreplace(parent_fd, source.name, destination.name)
            moved = True
        except BaseException as error:
            # A control-flow signal can arrive immediately after the atomic
            # syscall.  Prove that exact transition before deferring it.
            transitioned: os.stat_result | None = None
            try:
                transitioned = os.stat(
                    destination.name, dir_fd=parent_fd, follow_symlinks=False
                )
                os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                if transitioned is not None and private_paths._same_identity(
                    transitioned, opened
                ):
                    moved = True
                    deferred = error
                else:
                    raise error
            else:
                raise error
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
        reopened = _open_parent(source, parent_authority)
        try:
            if not _same_parent(os.fstat(reopened), parent_authority.identity):
                raise ValueError
        finally:
            os.close(reopened)
        if deferred is not None:
            raise deferred
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
    parent_authority: ParentAuthority,
    file_identity: os.stat_result,
    tombstone_key: MigrationTombstoneKey,
    allowed_links: frozenset[int] = frozenset({1}),
) -> None:
    """Atomically quarantine exact bytes in one bounded private tombstone."""

    holding = _holding_path(path, tombstone_key)
    deferred: BaseException | None = None
    try:
        move_exact_noreplace(
            path,
            holding,
            parent_authority=parent_authority,
            file_identity=file_identity,
            allowed_links=allowed_links,
        )
    except BaseException as error:
        if not isinstance(error, Exception):
            deferred = error
    parent_fd = _open_parent(holding, parent_authority)
    file_fd = -1
    try:
        file_fd = os.open(
            holding.name,
            os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_fd)
        current = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(opened, file_identity)
            or not private_paths._same_identity(current, file_identity)
            or not _valid_file(opened, links=allowed_links)
            or not _valid_file(current, links=allowed_links)
            or not _sidecars_absent(parent_fd, holding.name)
        ):
            raise ValueError
        os.fsync(file_fd)
        settled = os.fstat(file_fd)
        current = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(settled, file_identity)
            or not private_paths._same_identity(current, file_identity)
            or not _valid_file(settled, links=allowed_links)
            or not _valid_file(current, links=allowed_links)
        ):
            raise ValueError
        os.fsync(parent_fd)
        reopened = _open_parent(path, parent_authority)
        try:
            try:
                os.stat(path.name, dir_fd=reopened, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                # A substitution at the disposed logical leaf is foreign.
                raise ValueError
        finally:
            os.close(reopened)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)
    if deferred is not None:
        raise deferred
    # The retained leaf is cleanup evidence only.  Nonzero content is never
    # reused; keeping it bounds cleanup without risking a hardlink alias.


def open_new_or_reused_private_file(
    path: Path,
    *,
    parent_authority: ParentAuthority,
    tombstone_key: MigrationTombstoneKey,
) -> tuple[int, int, os.stat_result, ParentAuthority]:
    """Open a new private file, reusing only its exact zero tombstone."""

    holding = _holding_path(path, tombstone_key)
    parent_fd = _open_parent(path, parent_authority)
    file_fd = -1
    created = False
    try:
        try:
            holding_fd = os.open(
                holding.name,
                os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOCTTY", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            holding_fd = -1
        if holding_fd >= 0:
            try:
                held = os.fstat(holding_fd)
                entry = os.stat(holding.name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    held.st_size != 0
                    or entry.st_size != 0
                    or not private_paths._same_identity(held, entry)
                    or not _valid_file(held, links=frozenset({1}))
                    or not _valid_file(entry, links=frozenset({1}))
                    or not _sidecars_absent(parent_fd, holding.name)
                ):
                    raise ValueError
                try:
                    _rename_noreplace(parent_fd, holding.name, path.name)
                except BaseException as error:
                    acquired: os.stat_result | None = None
                    try:
                        acquired = os.stat(
                            path.name,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                        os.stat(
                            holding.name,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    if acquired is None or not private_paths._same_identity(
                        acquired, held
                    ):
                        raise
                    # Tuple assignment in the caller has not completed. Put
                    # the exact zero inode back under its bounded tombstone,
                    # durably verify both namespaces, then redeliver.
                    _rename_noreplace(parent_fd, path.name, holding.name)
                    os.fsync(parent_fd)
                    restored = os.stat(
                        holding.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    try:
                        os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise ValueError
                    reopened = _open_parent(path, parent_authority)
                    os.close(reopened)
                    if (
                        restored.st_size != 0
                        or not private_paths._same_identity(restored, held)
                        or not _valid_file(restored, links=frozenset({1}))
                    ):
                        raise ValueError
                    raise error
                file_fd = holding_fd
                holding_fd = -1
                os.fsync(parent_fd)
            finally:
                if holding_fd >= 0:
                    os.close(holding_fd)
        else:
            file_fd = os.open(
                path.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            created = True
        opened = os.fstat(file_fd)
        entry = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(opened, entry)
            or not _valid_file(opened, links=frozenset({1}))
            or not _valid_file(entry, links=frozenset({1}))
            or opened.st_size != 0
            or entry.st_size != 0
        ):
            raise ValueError
        current_parent = os.fstat(parent_fd)
        reopened, _leaf = private_paths._open_verified_parent(
            path,
            missing_leaf_allowed=True,
        )
        try:
            reopened_parent = os.fstat(reopened)
            link_delta = 1 if created and sys.platform == "darwin" else 0
            if not _same_parent_with_link_delta(
                current_parent,
                parent_authority.identity,
                link_delta=link_delta,
            ) or not _same_parent(reopened_parent, current_parent):
                raise ValueError
        finally:
            os.close(reopened)
        return parent_fd, file_fd, opened, ParentAuthority(current_parent)
    except BaseException:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)
        raise


__all__ = [
    "MigrationTombstoneKey",
    "ParentAuthority",
    "admit_zero_reusable_tombstone",
    "move_exact_noreplace",
    "open_new_or_reused_private_file",
    "prepare_reusable_tombstone",
    "rename_noreplace_at",
    "remove_zero_reusable_tombstone",
    "require_reusable_tombstone",
    "remove_exact",
]
