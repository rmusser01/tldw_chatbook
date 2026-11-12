"""Fail-closed canonical directory identity capture shared by workspace code."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

_WINDOWS = os.name == "nt"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", None)


class DirectoryIdentityError(ValueError):
    """Raised when a directory identity cannot be established safely."""


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    """Stable metadata needed to recognize one directory."""

    device: int
    inode: int
    mode: int
    reparse: bool


@dataclass(frozen=True, slots=True)
class DirectoryChain:
    """Canonical root and its root-first directory ancestor identities."""

    canonical_root: Path = field(repr=False)
    identities: tuple[DirectoryIdentity, ...] = field(repr=False)


def directory_identity_from_stat(value: object) -> DirectoryIdentity:
    """Build one directory identity, refusing incomplete platform metadata."""
    try:
        device = int(getattr(value, "st_dev"))
        inode = int(getattr(value, "st_ino"))
        mode = int(getattr(value, "st_mode"))
    except (AttributeError, TypeError, ValueError) as error:
        raise DirectoryIdentityError("directory metadata unavailable") from error

    reparse = _reparse_from_stat(value)
    return DirectoryIdentity(device=device, inode=inode, mode=mode, reparse=reparse)


def capture_directory_chain(root: Path) -> DirectoryChain:
    """Resolve ``root`` once and capture its root-first safe ancestor chain."""
    try:
        canonical_root = root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise DirectoryIdentityError("canonical directory unavailable") from error

    identities: list[DirectoryIdentity] = []
    for ancestor in (canonical_root, *canonical_root.parents):
        try:
            metadata = os.lstat(ancestor)
        except OSError as error:
            raise DirectoryIdentityError("directory metadata unavailable") from error
        identity = directory_identity_from_stat(metadata)
        if (
            not stat.S_ISDIR(identity.mode)
            or stat.S_ISLNK(identity.mode)
            or identity.reparse
        ):
            raise DirectoryIdentityError("unsafe directory metadata")
        identities.append(identity)
    return DirectoryChain(canonical_root=canonical_root, identities=tuple(identities))


def _reparse_from_stat(value: object) -> bool:
    if not _WINDOWS:
        return False
    try:
        attributes = getattr(value, "st_file_attributes")
        if attributes is None or _REPARSE_POINT is None:
            raise TypeError
        return bool(int(attributes) & int(_REPARSE_POINT))
    except (AttributeError, TypeError, ValueError) as error:
        raise DirectoryIdentityError("directory file attributes unavailable") from error
