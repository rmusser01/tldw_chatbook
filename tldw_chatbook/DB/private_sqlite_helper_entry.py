"""Exec entry: python -I -S <installed absolute file>; no application startup."""

import os
import sys


#: Highest descriptor the no-listing fallback closes. Without a listing and a
#: limit at or below this, the helper refuses to start rather than guess.
_FALLBACK_DESCRIPTOR_LIMIT = 65536


def _close_inherited_descriptors() -> None:
    """Close every descriptor this process inherited across exec.

    On macOS the launcher spawns this entry with ``posix_spawn``
    (TASK-33620.15.55), which, unlike the earlier ``close_fds=True`` fork, keeps
    every inheritable parent descriptor open across exec. Python and SQLite open
    their descriptors close-on-exec, so normally this finds only descriptors the
    app itself inherited from its own launcher. Either way the helper then works
    with exactly its three pipes, as before. Only an inheritable descriptor can
    have crossed exec, so a close-on-exec one (anything this process opened) is
    never touched.

    Raises:
        SystemExit: The descriptors can be neither listed nor bounded, so an
            inherited one cannot be shown closed (fail closed).
    """
    directory = "/dev/fd" if sys.platform == "darwin" else "/proc/self/fd"
    try:
        candidates = [int(name) for name in os.listdir(directory) if name.isdecimal()]
    except OSError:
        try:
            limit = os.sysconf("SC_OPEN_MAX")
        except (OSError, ValueError, OverflowError):
            limit = -1
        if not 3 <= limit <= _FALLBACK_DESCRIPTOR_LIMIT:
            raise SystemExit(1) from None
        candidates = range(3, limit)
    for descriptor in candidates:
        if descriptor <= 2:
            continue
        try:
            if os.get_inheritable(descriptor):
                os.close(descriptor)
        except OSError:
            pass  # not open (the listing's own descriptor is already closed)


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
