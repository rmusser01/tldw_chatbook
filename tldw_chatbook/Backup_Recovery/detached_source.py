"""Read-only, stable acquisition of detached files; never live capture authority."""

from __future__ import annotations

import hashlib
import json
import stat
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from tldw_chatbook.Utils.path_validation import validate_recovery_relative_path
from tldw_chatbook.Utils.platform_files import os

from .file_inventory import (
    MAX_ENTRIES,
    MAX_EXPANDED_BYTES,
    MAX_MEMBER_BYTES,
    MAX_PATH_BYTES,
    _metadata_supported,
)
from .native_files import (
    create_private_directory,
    create_private_file,
    pinned_directory,
)


@dataclass(frozen=True)
class SourceMember:
    relative: str
    directory: bool
    identity: tuple[int, ...]
    sha256: str | None


@dataclass(frozen=True)
class SourceReview:
    root: Path
    members: tuple[SourceMember, ...]
    digest: str
    excluded_roots: tuple[str, ...] = ()


def _identity(info) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _hash(fd: int, size: int, destination: int | None = None) -> str:
    digest = hashlib.sha256()
    total = 0
    while chunk := os.read(fd, min(1024 * 1024, size - total + 1)):
        total += len(chunk)
        if total > size:
            raise ValueError("detached_source_changed")
        digest.update(chunk)
        if destination is not None:
            view = memoryview(chunk)
            while view:
                written = os.write(destination, view)
                if written <= 0:
                    raise OSError("detached_copy_incomplete")
                view = view[written:]
    return digest.hexdigest()


def review_source(root: Path, *, excluded_roots: tuple[str, ...] = ()) -> SourceReview:
    """Bind a complete bounded census to no-follow identities and content hashes.

    Args:
        root: Explicit local source directory. Record paths are never consulted.
        excluded_roots: Installed caller's relative transient/work directories.

    Returns:
        Immutable review of every in-scope member, including empty directories.

    Raises:
        ValueError: A source object, alias, permission or resource limit is unsafe.
    """
    root = Path(os.path.abspath(root))
    if type(excluded_roots) is not tuple or len(set(excluded_roots)) != len(
        excluded_roots
    ):
        raise ValueError("invalid_detached_exclusions")
    for name in excluded_roots:
        validate_recovery_relative_path(name)
        if not name or Path(name).as_posix() != name:
            raise ValueError("invalid_detached_exclusions")
    members = []
    total = 0
    device = None

    def walk(parent, name, relative, depth):
        nonlocal total, device
        validate_recovery_relative_path(relative)
        if (
            len(members) >= MAX_ENTRIES
            or depth > 64
            or len(relative.encode()) > MAX_PATH_BYTES
        ):
            raise ValueError("detached_source_limit")
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        directory = stat.S_ISDIR(before.st_mode)
        if device is None:
            device = before.st_dev
        if (
            not (directory or stat.S_ISREG(before.st_mode))
            or before.st_dev != device
            or (not directory and before.st_nlink != 1)
        ):
            raise ValueError("detached_source_unsafe")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        fd = os.open(name, flags, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if _identity(info) != _identity(before):
                raise ValueError("detached_source_changed")
            if (
                not _metadata_supported(fd, info)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o022
            ):
                raise ValueError("detached_source_unsafe")
            digest = None
            if directory:
                with os.scandir(fd) as entries:
                    children = []
                    folded = set()
                    for entry in entries:
                        key = unicodedata.normalize("NFC", entry.name).casefold()
                        if key in folded or len(children) + len(members) >= MAX_ENTRIES:
                            raise ValueError("detached_source_collision")
                        folded.add(key)
                        validate_recovery_relative_path(entry.name)
                        children.append(entry.name)
                # Record parents before children for deterministic private creation.
                members.append(SourceMember(relative, True, _identity(info), None))
                for child in sorted(children):
                    child_relative = (Path(relative) / child).as_posix()
                    if child_relative in excluded_roots:
                        excluded = os.stat(child, dir_fd=fd, follow_symlinks=False)
                        if not stat.S_ISDIR(excluded.st_mode):
                            raise ValueError("detached_source_unsafe")
                        continue
                    walk(fd, child, child_relative, depth + 1)
            else:
                total += info.st_size
                if info.st_size > MAX_MEMBER_BYTES or total > MAX_EXPANDED_BYTES:
                    raise ValueError("detached_source_limit")
                digest = _hash(fd, info.st_size)
                members.append(SourceMember(relative, False, _identity(info), digest))
            if _identity(os.fstat(fd)) != _identity(info) or _identity(
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            ) != _identity(info):
                raise ValueError("detached_source_changed")
        finally:
            os.close(fd)

    try:
        with pinned_directory(root.parent) as parent:
            walk(parent, root.name, "", 0)
    except (OSError, RuntimeError) as error:
        raise ValueError("detached_source_unsafe") from error
    payload = {"members": [asdict(row) for row in members], "excluded": excluded_roots}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return SourceReview(root, tuple(members), digest, excluded_roots)


def recheck_source(review: SourceReview) -> None:
    """Refuse any identity, byte or member-list change since explicit review."""
    try:
        current = review_source(review.root, excluded_roots=review.excluded_roots)
    except (ValueError, OSError) as error:
        raise ValueError("detached_source_changed") from error
    if current != review:
        raise ValueError("detached_source_changed")


def acquire_source(review: SourceReview, destination: Path) -> Path:
    """Copy reviewed members privately without writing source files or sidecars."""
    destination = Path(destination)
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("invalid_detached_destination")
    overlap = destination == review.root or review.root in destination.parents
    excluded_destination = any(
        destination == review.root / name or review.root / name in destination.parents
        for name in review.excluded_roots
    )
    if (
        review.root == destination
        or destination in review.root.parents
        or (overlap and not excluded_destination)
    ):
        raise ValueError("detached_destination_overlap")
    recheck_source(review)
    create_private_directory(destination)
    for row in review.members:
        if not row.relative:
            continue
        target = destination / row.relative
        if row.directory:
            create_private_directory(target)
            continue
        source = review.root / row.relative
        with pinned_directory(source.parent) as parent:
            fd = os.open(
                source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            try:
                if _identity(os.fstat(fd)) != row.identity:
                    raise ValueError("detached_source_changed")
                with create_private_file(target) as output:
                    if _hash(fd, row.identity[3], output) != row.sha256:
                        raise ValueError("detached_source_changed")
                if _identity(os.fstat(fd)) != row.identity:
                    raise ValueError("detached_source_changed")
            finally:
                os.close(fd)
    recheck_source(review)
    return destination
