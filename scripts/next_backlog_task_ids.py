#!/usr/bin/env python3
"""Allocate free backlog task ids with a lead buffer (TASK-35131).

``dev`` mints task ids continuously across active sessions (~2/hour observed
2026-10-14/15). A branch that picks "next free id" at task-creation time races
that minting and loses before the PR lands: the ``Backlog task IDs are unique``
required check evicted four consecutive allocations on PRs #3083 and #3119
(34948-34952 -> 34949-34951 -> 34991/34992 -> 35026/35027), each renumbered to
the then-next-free id and each re-colliding within hours. A +30 lead survived
~15 hours of minting. This helper makes that discipline the default instead of
folklore.

Scans the same three directories the uniqueness guard resolves ids in --
``backlog/tasks``, ``backlog/completed``, ``backlog/archive/tasks`` -- because
archiving a file does not free its id. Refuses to print any id that exists in
any of them, filename prefix or frontmatter ``id:`` alike.

Stdlib-only: it runs with no dependency install, like the other
derived-artifacts checkers.

Usage:
    python scripts/next_backlog_task_ids.py [N] [--buffer B] [--root PATH]

    N        number of ids to allocate (default 1)
    --buffer lead past the current max id, across ALL three directories
             (default 30; ~15h of headroom at dev's observed mint rate)
    --root   repository root (default: this script's parent)

Prints one integer per line, ascending. The FIRST printed id is
``max(existing) + buffer``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIRS = (
    REPO_ROOT / "backlog" / "tasks",
    REPO_ROOT / "backlog" / "completed",
    REPO_ROOT / "backlog" / "archive" / "tasks",
)
FILENAME_ID_RE = re.compile(r"^task-(\d+)(?:\.\d+)* - .*\.md$")
FRONTMATTER_ID_RE = re.compile(r"^id:\s*TASK-(\d+)", re.IGNORECASE)
DEFAULT_BUFFER = 30


def existing_task_ids(root: Path) -> set[int]:
    """Every integer id claimed in the three id-resolving directories.

    Integer ids only: dotted subtask ids (``task-3.1``) never collide with
    fresh integer allocations, so their stems already count via the integer
    part. Both claim surfaces are scanned -- filename prefix and frontmatter
    ``id:`` -- matching the uniqueness guard's rule that a hand-edited rename
    can leave the two disagreeing.
    """
    claimed: set[int] = set()
    for directory in (
        root / "backlog" / "tasks",
        root / "backlog" / "completed",
        root / "backlog" / "archive" / "tasks",
    ):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix != ".md":
                continue
            match = FILENAME_ID_RE.match(path.name)
            if match:
                claimed.add(int(match.group(1)))
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Frontmatter sits at the top; the first id: line is the task's
            # identity, later ones are prose (same rule as the guard).
            for line in text.splitlines()[:50]:
                match = FRONTMATTER_ID_RE.match(line)
                if match:
                    claimed.add(int(match.group(1)))
                    break
    return claimed


def allocate(count: int, buffer: int, root: Path) -> list[int]:
    """Return ``count`` free ids starting at ``max(existing) + buffer``."""
    if count < 1:
        raise SystemExit("count must be >= 1")
    if buffer < 1:
        raise SystemExit("buffer must be >= 1 (a 0 buffer is the race this fixes)")
    claimed = existing_task_ids(root)
    candidate = (max(claimed) if claimed else 0) + buffer
    allocated: list[int] = []
    while len(allocated) < count:
        if candidate in claimed:
            # A dotted-subtask stem or a sparse high id occupies the slot;
            # skip past it rather than ever printing a claimed id.
            candidate += 1
            continue
        allocated.append(candidate)
        candidate += 1
    return allocated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Allocate free backlog task ids with a lead buffer (TASK-35131)."
    )
    parser.add_argument(
        "count", nargs="?", type=int, default=1, help="number of ids (default 1)"
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=DEFAULT_BUFFER,
        help=f"lead past current max across all task dirs (default {DEFAULT_BUFFER})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: this script's parent)",
    )
    args = parser.parse_args(argv)
    for task_id in allocate(args.count, args.buffer, args.root):
        print(task_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
