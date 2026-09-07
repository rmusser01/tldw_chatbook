"""Exec entry: python -I -S <installed absolute file>; no application startup."""

import sys
from pathlib import Path
from types import ModuleType


def main() -> int:
    """Bootstrap only the fixed installed package namespaces, without __init__."""
    if not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 1:
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
    ):
        namespace = ModuleType(name)
        namespace.__path__ = [str(directory)]
        namespace.__package__ = name
        sys.modules[name] = namespace
    from tldw_chatbook.DB.private_sqlite_helper import run

    return run()


if __name__ == "__main__":
    try:
        result = main()
    except BaseException:  # noqa: BLE001 - never emit a private startup traceback
        result = 1
    raise SystemExit(result)
