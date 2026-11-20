"""Exec entry: python -I -S <installed absolute file>; no application startup."""

import os
import sys


def _close_inherited_descriptors() -> None:
    """Close every descriptor above stderr that this process inherited.

    On macOS the launcher spawns this entry with ``posix_spawn``
    (TASK-33620.15.55), which, unlike the earlier ``close_fds=True`` fork, keeps
    any inheritable parent descriptor open across exec. Python and SQLite create
    their descriptors close-on-exec, so normally there is nothing to close; either
    way the helper then works with exactly its three pipes, as before. It runs
    before any other statement, so every descriptor above 2 is an inherited one.
    """
    directory = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"
    try:
        names = os.listdir(directory)
    except OSError:
        os.closerange(3, os.sysconf("SC_OPEN_MAX"))
        return
    for name in names:
        if not name.isdecimal() or int(name) <= 2:
            continue
        try:
            os.close(int(name))
        except OSError:
            pass  # the listing's own directory descriptor is already closed


if __name__ == "__main__":
    try:
        _close_inherited_descriptors()
    except BaseException:  # noqa: BLE001 - never emit a private startup traceback
        raise SystemExit(1) from None

# Imported only after the inherited-descriptor close above.
from pathlib import Path
from types import ModuleType


def main() -> int:
    """Bootstrap only the fixed installed package namespaces, without __init__."""
    if not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 1:
        return 1
    # Required launch-only metadata. Never adopt a parent discovered after
    # interpreter startup: a subreaper may already have inherited this child.
    parent_metadata = os.environ.pop("_TLDW_PRIVATE_SQLITE_PARENT_PID", None)
    if (
        parent_metadata is None
        or not 1 <= len(parent_metadata) <= 20
        or not parent_metadata.isascii()
        or not parent_metadata.isdecimal()
        or parent_metadata.startswith("0")
    ):
        return 1
    parent_pid = int(parent_metadata)
    if os.getppid() != parent_pid:
        return 1
    entry = Path(__file__).resolve(strict=True)
    package = entry.parent.parent
    if (
        entry.name != "private_sqlite_helper_entry.py"
        or entry.parent.name != "DB"
        or package.name != "tldw_chatbook"
    ):
        return 1
    for name, directory in (
        ("tldw_chatbook", package),
        ("tldw_chatbook.DB", package / "DB"),
        ("tldw_chatbook.Utils", package / "Utils"),
        ("tldw_chatbook.TTS", package / "TTS"),
        ("tldw_chatbook.TTS.migrations", package / "TTS" / "migrations"),
    ):
        namespace = ModuleType(name)
        namespace.__path__ = [str(directory)]
        namespace.__package__ = name
        sys.modules[name] = namespace
    from tldw_chatbook.DB.private_sqlite_helper import run

    return run(parent_pid)


if __name__ == "__main__":
    try:
        result = main()
    except BaseException:  # noqa: BLE001 - never emit a private startup traceback
        result = 1
    raise SystemExit(result)
