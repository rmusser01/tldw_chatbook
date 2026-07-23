"""Private local-file lifecycle primitives.

This module is deliberately dependency-leaf: callers choose failure policy and
diagnostics while this module performs lexical selection and filesystem checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

PathInput: TypeAlias = str | os.PathLike[str]


class PrivatePathStatus(StrEnum):
    CREATED_PRIVATE = "created_private"
    HARDENED_PRIVATE = "hardened_private"
    ALREADY_PRIVATE = "already_private"
    UNSAFE_PARENT = "unsafe_parent"
    WRONG_OWNER = "wrong_owner"
    LINK_OR_NON_REGULAR = "link_or_non_regular"
    OPERATION_FAILED = "operation_failed"
    UNVERIFIED_PLATFORM = "unverified_platform"


@dataclass(frozen=True)
class PrivatePathResult:
    lexical_path: Path
    status: PrivatePathStatus
    reason: str | None = None

    @property
    def verified_private(self) -> bool:
        return self.status in {
            PrivatePathStatus.CREATED_PRIVATE,
            PrivatePathStatus.HARDENED_PRIVATE,
            PrivatePathStatus.ALREADY_PRIVATE,
        }

    @property
    def usable(self) -> bool:
        return self.verified_private or (
            self.status is PrivatePathStatus.UNVERIFIED_PLATFORM
        )


class PrivatePathError(OSError):
    def __init__(self, result: PrivatePathResult) -> None:
        self.result = result
        reason = f": {result.reason}" if result.reason else ""
        super().__init__(f"{result.status.value}{reason}")


def lexical_path(path: PathInput) -> Path:
    raw = os.fspath(path)
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    expanded = os.path.expanduser(raw)
    return Path(os.path.abspath(os.path.normpath(expanded)))
