#!/usr/bin/env python3
"""Guard: each ADR number in ``backlog/decisions`` belongs to one decision file.

ADR files are named ``<NNN>-<title>.md``. Two branches minting the same number
left 50 numbers shared by 117 files (renumbered under TASK-36943.3), so a reference such as ``ADR-228`` named
more than one decision. Owner rule (2026-10-25): on a collision the oldest file
keeps the number and each younger file takes the next free number with a
renumbering-provenance note (see ``backlog/decisions/README.md``).

Every collision is fatal. The grandfathered list that TASK-36943.2 introduced
was emptied and removed by TASK-36943.3.

Stdlib-only and install-free, like the other derived-artifact checkers.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DECISIONS_DIR = REPO_ROOT / "backlog" / "decisions"
ADR_FILE_RE = re.compile(r"^(\d+)-.*\.md$")

RESOLUTION = (
    "Resolve per the 2026-10-25 owner rule: the OLDEST file keeps the number "
    "(Date field, else add-commit time); each younger file renumbers to the "
    "next free number with a renumbering-provenance note. New ADRs take the "
    "highest number in use plus 1 and never backfill gaps."
)


def adr_numbers(decisions_dir: Path) -> dict[int, list[str]]:
    """Map each ADR number to the sorted names of the files that use it.

    Args:
        decisions_dir: Directory holding ``<NNN>-<title>.md`` files.

    Returns:
        Number -> file names. ``000-template.md`` is skipped.
    """
    numbers: dict[int, list[str]] = defaultdict(list)
    for path in sorted(decisions_dir.glob("*.md")):
        match = ADR_FILE_RE.match(path.name)
        if match and int(match.group(1)) != 0:
            numbers[int(match.group(1))].append(path.name)
    return dict(numbers)


def problems(numbers: dict[int, list[str]]) -> tuple[list[str], int]:
    """Return ``(violations, next_free_number)``.

    Next free is the highest number in use plus 1.
    """
    out = [
        f"ADR-{number:03d} is used by {len(files)} files: " + ", ".join(files)
        for number, files in sorted(numbers.items())
        if len(files) > 1
    ]
    return out, max(numbers, default=0) + 1


def main(argv: list[str] | None = None) -> int:
    """Fail on any ADR number collision.

    Returns:
        int: ``0`` when clean, ``1`` otherwise (``::error::`` lines name each
            violation and the next free ADR number).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-dir", type=Path, default=DECISIONS_DIR)
    args = parser.parse_args(argv)
    if not args.decisions_dir.is_dir():
        print(f"::error::no ADR directory at {args.decisions_dir}")
        return 1
    numbers = adr_numbers(args.decisions_dir)
    found, next_free = problems(numbers)
    if found:
        print("::error::ADR numbering violations in backlog/decisions:")
        for line in found:
            print(f"  {line}")
        print(f"Next free ADR number: {next_free:03d}")
        print(RESOLUTION, file=sys.stderr)
        return 1
    print(
        f"ADR numbers unique across {sum(map(len, numbers.values()))} decision "
        f"files. Next free: {next_free:03d}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
