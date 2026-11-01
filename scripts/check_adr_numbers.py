#!/usr/bin/env python3
"""Guard: each ADR number in ``backlog/decisions`` belongs to one decision file.

ADR files are named ``<NNN>-<title>.md``. Two branches minting the same number
left 50 numbers shared by 117 files, so a reference such as ``ADR-228`` named
more than one decision. Owner rule (2026-10-25): on a collision the oldest file
keeps the number and each younger file takes the next free number with a
renumbering-provenance note (see ``backlog/decisions/README.md``).

``scripts/adr_shared_numbers_frozen.txt`` lists today's shared numbers with
their file counts. It can only shrink: a number outside it, or a count above
it, is a new collision; a listed number that is no longer shared, or whose
count dropped, fails until the list is updated. TASK-36943.3 empties it.

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
FROZEN_FILE = Path(__file__).resolve().with_name("adr_shared_numbers_frozen.txt")
ADR_FILE_RE = re.compile(r"^(\d+)-.*\.md$")
FROZEN_LINE_RE = re.compile(r"^(\d+)\s+(\d+)$")

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


def read_frozen(frozen_file: Path) -> dict[int, int]:
    """Parse ``<number> <count>`` lines; ``#`` comments and blanks are ignored."""
    frozen: dict[int, int] = {}
    for line in frozen_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        match = FROZEN_LINE_RE.match(line)
        if match:
            frozen[int(match.group(1))] = int(match.group(2))
    return frozen


def problems(
    numbers: dict[int, list[str]], frozen: dict[int, int]
) -> tuple[list[str], int]:
    """Return ``(violations, next_free_number)``.

    Next free is the highest number in use (files or frozen list) plus 1.
    """
    out: list[str] = []
    for number in sorted(numbers):
        files = numbers[number]
        if len(files) > 1 and len(files) > frozen.get(number, 1):
            out.append(
                f"ADR-{number:03d} is used by {len(files)} files "
                f"(frozen allowance {frozen.get(number, 0)}): " + ", ".join(files)
            )
    for number in sorted(frozen):
        count = len(numbers.get(number, []))
        if count < frozen[number] and count <= 1:
            out.append(
                f"ADR-{number:03d} is no longer shared ({count} file); "
                f"remove it from {FROZEN_FILE.name}"
            )
        elif count < frozen[number]:
            out.append(
                f"ADR-{number:03d} has {count} files, frozen at {frozen[number]}; "
                f"lower its count in {FROZEN_FILE.name}"
            )
    return out, max([*numbers, *frozen], default=0) + 1


def main(argv: list[str] | None = None) -> int:
    """Fail on a new ADR number collision or a stale frozen entry.

    Returns:
        int: ``0`` when clean, ``1`` otherwise (``::error::`` lines name each
            violation and the next free ADR number).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-dir", type=Path, default=DECISIONS_DIR)
    parser.add_argument("--frozen-file", type=Path, default=FROZEN_FILE)
    args = parser.parse_args(argv)
    if not args.decisions_dir.is_dir():
        print(f"::error::no ADR directory at {args.decisions_dir}")
        return 1
    frozen = read_frozen(args.frozen_file) if args.frozen_file.is_file() else {}
    numbers = adr_numbers(args.decisions_dir)
    found, next_free = problems(numbers, frozen)
    if found:
        print("::error::ADR numbering violations in backlog/decisions:")
        for line in found:
            print(f"  {line}")
        print(f"Next free ADR number: {next_free:03d}")
        print(RESOLUTION, file=sys.stderr)
        return 1
    print(
        f"ADR numbers unique across {sum(map(len, numbers.values()))} decision "
        f"files ({len(frozen)} frozen shared numbers). Next free: {next_free:03d}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
