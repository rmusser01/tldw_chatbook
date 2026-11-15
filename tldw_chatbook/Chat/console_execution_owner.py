"""Canonical compatibility exports for the native run-store execution lease."""

from __future__ import annotations

from tldw_chatbook.DB.console_execution_owner import (
    ClassVar,
    ConsoleExecutionBusy,
    ConsoleExecutionOwner,
    ConsoleExecutionUnavailable,
    Path,
    PrivatePathStatus,
    RLock,
    Self,
    TextIO,
    _identity,  # noqa: F401 - retained original private lookup
    _store_identity,  # noqa: F401 - retained original private lookup
    contextmanager,
    errno,
    fcntl,
    lexical_path,
    open_private_binary,
    open_private_lock_stream,
    os,
    stat,
    verify_trusted_directory,
)

__all__ = [
    "ClassVar",
    "ConsoleExecutionBusy",
    "ConsoleExecutionOwner",
    "ConsoleExecutionUnavailable",
    "Path",
    "PrivatePathStatus",
    "RLock",
    "Self",
    "TextIO",
    "annotations",
    "contextmanager",
    "errno",
    "fcntl",
    "lexical_path",
    "open_private_binary",
    "open_private_lock_stream",
    "os",
    "stat",
    "verify_trusted_directory",
]
