"""Host-native durability barriers shared by every atomic writer.

``flush_file``/``flush_directory`` were moved here from
``Backup_Recovery/native_platform.py`` (task-32896). ``Utils`` is the leaf
layer -- ``Backup_Recovery`` already reached *into* it for the Windows half
(``Utils/windows_files.py``), so the barriers belong on this side of that
edge, and ``Utils/atomic_file_ops.py`` can now use them without ``Utils``
importing a feature package. ``native_platform`` re-exports both names, so
the seven recovery modules that import from it are unchanged.

Kept stdlib-only on purpose: ``native_platform`` sits on the recovery
bootstrap import path, which must not pull in loguru/tempfile/shutil.

Cost note (measured, qa/tier2-code-review-2026-09-21/validation/S25): the
Darwin ``F_FULLFSYNC`` barrier in ``flush_file`` is ~300x a plain
``os.fsync``. That is the price of actually reaching the platter; a write
that skips it is atomic but not durable.
"""

from __future__ import annotations

import errno
import os
import platform

#: Directory fsync is genuinely unavailable on some filesystems. Only these
#: errnos are tolerated -- anything else is a real durability failure and is
#: raised, not swallowed.
UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
)


def flush_file(fd: int) -> None:
    """Persist file contents and metadata using the host's native barrier.

    On Darwin this is ``fsync`` followed by ``F_FULLFSYNC``: a plain ``fsync``
    returns before the drive's own write cache reaches the platter, so it is
    not a durability barrier there at all.

    Args:
        fd: An OPEN file descriptor for a regular file, opened for writing.
            The caller keeps ownership; this never closes it. User-space
            buffers must already have been flushed into it (``f.flush()``) --
            this barrier only moves what the kernel holds.

    Raises:
        OSError: The barrier failed. Never tolerated: unlike a directory
            fsync, there is no filesystem on which a file fsync is merely
            unsupported, so a failure here is a real durability failure and
            the caller must treat the write as unpersisted.
    """
    if os.name == "nt":
        from .windows_files import flush_file as native_flush

        native_flush(fd)
        return
    os.fsync(fd)
    if platform.system() == "Darwin":
        import fcntl

        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)


def flush_directory(fd: int) -> None:
    """Persist directory changes; failures remain ambiguous to journal callers.

    Args:
        fd: An OPEN descriptor for a DIRECTORY (POSIX: ``os.open`` with
            ``O_RDONLY | O_DIRECTORY``). The caller keeps ownership.

    Raises:
        OSError: The barrier failed. Raised unfiltered here -- deciding which
            errnos mean "this filesystem cannot do it" is
            :func:`fsync_parent_directory`'s job, because only a caller that
            opened the directory itself knows whether the failure is
            recoverable. A journal caller must treat a failure as an
            AMBIGUOUS publication, not as a completed or an abandoned one.
    """
    if os.name == "nt":
        from .windows_files import flush_directory as native_flush

        native_flush(fd)
    else:
        flush_file(fd)


def fsync_parent_directory(directory: str | os.PathLike[str]) -> None:
    """Persist the *directory entry* so a completed rename survives power loss.

    ``os.replace`` plus a file fsync is atomic but not durable: until the
    parent directory is itself fsynced the rename can be lost. Windows cannot
    open a directory via ``os.open``, so the barrier is skipped there rather
    than crashing; filesystems that reject a directory fsync outright are
    tolerated via ``UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS``.

    Args:
        directory: The directory whose entries must be persisted -- for a
            publication, the PARENT of the renamed file, not the file. It is
            opened read-only and closed again here.

    Raises:
        OSError: The directory barrier genuinely failed, i.e. the errno is not
            in ``UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS``. Because this runs AFTER
            ``os.replace``, such a failure means the new file is already
            visible and only its survival across a power loss is unconfirmed;
            a caller that reports it as a failed write must not also claim the
            old contents are intact.
    """
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError as exc:
        if exc.errno in UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise
    try:
        flush_directory(fd)
    except OSError as exc:
        if exc.errno not in UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            raise
    finally:
        os.close(fd)
