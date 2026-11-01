"""One-shot detached validation process; release all runtime leases on exit."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from threading import Event


def main(argv: list[str]) -> int:
    """Produce a bounded private reply, never publishing or approving a profile."""
    from .portable_relocation import _write

    if len(argv) != 3:
        return 2
    source, work, reply = map(Path, argv)
    if (
        not all(path.is_absolute() for path in (source, work, reply))
        or reply.parent != work.parent
        or reply.name != "producer-result.json"
    ):
        return 2
    try:
        from .detached_layout import produce_detached_archive

        result = produce_detached_archive(source, work, Event())
        if result.archive.parent != work.parent:
            raise ValueError("detached_archive_location_invalid")
        value = {
            "version": 1,
            "status": "succeeded",
            "archive": result.archive.name,
            "review_lines": list(result.review_lines),
        }
    except (OSError, ValueError, RuntimeError, SystemExit) as error:
        code = str(error)
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]{0,255}", code):
            code = "detached_import_failed"
        _write(
            reply.parent, reply.name, {"version": 1, "status": "refused", "issue": code}
        )
        return 1
    _write(reply.parent, reply.name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
