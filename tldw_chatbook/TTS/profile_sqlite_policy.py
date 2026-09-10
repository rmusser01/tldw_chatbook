"""Public SQLite close-policy capability admission for TTS profile stores."""

from __future__ import annotations

import sqlite3

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


def _raise_runtime_unsupported() -> None:
    raise ProfileRepositoryError("runtime_unsupported") from None


def configure_native_close_policy(connection: sqlite3.Connection) -> None:
    """Enable and verify the required policy on one borrowed SQLite handle.

    Args:
        connection: Caller-owned connection that has not executed SQL yet.

    Raises:
        ProfileRepositoryError: If the public capability is missing, rejected,
            or cannot be verified as enabled.
    """
    try:
        option = sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE
        setconfig = connection.setconfig
        getconfig = connection.getconfig
    except AttributeError:
        _raise_runtime_unsupported()

    try:
        setconfig(option, True)
        configured = getconfig(option)
    except sqlite3.Error:
        _raise_runtime_unsupported()
    if configured is not True:
        _raise_runtime_unsupported()


def require_native_close_policy_support() -> None:
    """Probe required public SQLite support before TTS store initialization.

    Raises:
        ProfileRepositoryError: If the runtime lacks or rejects the required
            close-policy capability.
    """
    try:
        connection_type = sqlite3.Connection
        option = sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE
    except AttributeError:
        _raise_runtime_unsupported()
    if (
        not callable(getattr(connection_type, "setconfig", None))
        or not callable(getattr(connection_type, "getconfig", None))
        or type(option) is not int
    ):
        _raise_runtime_unsupported()

    connection = sqlite3.connect(":memory:")
    try:
        configure_native_close_policy(connection)
    finally:
        connection.close()
