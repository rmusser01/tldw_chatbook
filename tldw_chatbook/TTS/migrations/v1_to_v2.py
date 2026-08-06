"""v1 -> v2: version fence only. v2 stores may contain non-audio.cpp providers;
pre-expansion builds must refuse them (they reject user_version > 1)."""

from __future__ import annotations

import sqlite3

TARGET_VERSION = 2


def migrate(connection: sqlite3.Connection) -> None:
    """Apply the version-one to version-two migration transactionally.

    There is no DDL change between v1 and v2: profile rows may now carry any
    of the seven provider ids instead of only ``audio_cpp``. Bumping the
    stored schema version is itself the fence -- pre-expansion builds already
    refuse to open a store whose ``user_version`` exceeds their own constant,
    so this migration's only job is to advance that fence.
    """

    connection.execute("PRAGMA user_version = 2")
