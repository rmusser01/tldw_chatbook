"""Exec entry: python -I -S <installed absolute file>; no application startup."""

import os
import sys
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
