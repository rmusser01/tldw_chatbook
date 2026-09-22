#!/usr/bin/env python3
"""Guard: the Tests/UI slice that gates pull requests can only grow.

TASK-32908. Before this, ``Tests/UI`` -- 25,917 collected tests, the largest
single directory in the tree -- gated **nothing** on a pull request:

* ``.github/workflows/test.yml`` does have a 12-shard ``ui-tests`` job, but
  that workflow is ``on: push: branches: ["main"]`` plus
  ``workflow_dispatch``. It does not accept ``pull_request`` events at all,
  so neither its core shards nor its UI shards are a PR gate.
* ``.github/workflows/nightly-deep.yml`` runs ``pytest ./Tests/`` on a
  schedule, which is where the UI suite was assumed to be covered. It is
  not: with no ``--continue-on-collection-errors`` a single bad import
  aborts the whole session, and it has been doing exactly that -- measured
  on run 35706024071 (2026-09-22), ``collected 101851 items / 1 error``
  then ``Interrupted: 1 error during collection``, **zero tests executed**,
  six consecutive nights.

So the only PR gate is ``derived-artifacts.yml``, and the cost of that gap
is not theoretical. ``Tests/UI/test_console_library_tool_setting.py``
asserted ``service._collections is app.local_library_collections_service``
against a ``LocalLibraryToolService`` that has had no ``_collections``
attribute since 5dd1077df6 retired it. The assertion could not pass. It sat
red and blocked nothing -- and the sibling copy of that same factory in
``Chat/console_runtime.py`` went on passing a now-rejected
``collections_service=`` keyword, i.e. a live ``TypeError`` on the
Console-direct Library tool path, for three weeks.

The full directory cannot go in the PR gate: measured at 25,917 tests and
~5.5 CPU-hours, against a fast lane budgeted in minutes. What goes in is a
**verified-green subset**, listed one path per line in
``scripts/ui_pr_gate_census.txt`` and run by the ``ui-fast-lane`` job.

This checker is what stops that subset from quietly evaporating. The tests
themselves are the ratchet on *behaviour* -- a censused file that goes red
turns the job red, which is the whole point. What tests cannot catch is the
census being edited instead of the bug:

* a listed path that no longer exists (renamed or deleted) makes pytest
  collect fewer files while still exiting 0 -- a gate that silently tests
  less than it claims;
* deleting lines is the path of least resistance when a censused file goes
  red, and nothing else would notice.

Hence: every listed path must exist, no duplicates, and the census may
never fall below ``MINIMUM_FILES``. Growing it is free; shrinking it
requires editing this file, which is the review checkpoint.

Exits 0 when the census is intact, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CENSUS_PATH = REPO_ROOT / "scripts" / "ui_pr_gate_census.txt"

#: Ratchet floor. Raise it when the census grows; never lower it without
#: saying why in the commit message. This is deliberately a literal rather
#: than "len(census) at HEAD" -- a floor derived from the file it guards
#: guards nothing.
MINIMUM_FILES = 120  # TASK-32908: 120 files / 851 tests, verified green


def read_census(path: Path) -> list[str]:
    """Return the census entries, in file order, ignoring blanks/comments."""
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def main() -> int:
    if not CENSUS_PATH.exists():
        print(f"FAIL: census file is missing: {CENSUS_PATH}", file=sys.stderr)
        return 1

    entries = read_census(CENSUS_PATH)
    problems: list[str] = []

    seen: set[str] = set()
    for entry in entries:
        if entry in seen:
            problems.append(f"duplicate entry: {entry}")
        seen.add(entry)
        if not entry.startswith("Tests/UI/"):
            problems.append(f"not a Tests/UI path: {entry}")
            continue
        if not (REPO_ROOT / entry).is_file():
            problems.append(
                f"listed file does not exist: {entry}\n"
                "    A renamed or deleted censused file makes the gate collect "
                "fewer tests while still exiting 0.\n"
                "    Update the census to the new path, or remove the line and "
                "lower MINIMUM_FILES with a reason."
            )

    if len(entries) < MINIMUM_FILES:
        problems.append(
            f"census has shrunk: {len(entries)} files, floor is {MINIMUM_FILES}.\n"
            "    If a censused file genuinely had to leave the gate, lower\n"
            f"    MINIMUM_FILES in {Path(__file__).name} in the SAME commit and say why.\n"
            "    Deleting the line on its own is how a gate rots to nothing."
        )

    if problems:
        print(
            f"FAIL: {CENSUS_PATH.relative_to(REPO_ROOT)} is not intact "
            f"({len(problems)} problem(s)):",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(entries)} Tests/UI files in the PR gate "
        f"(floor {MINIMUM_FILES}); every listed path exists."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
