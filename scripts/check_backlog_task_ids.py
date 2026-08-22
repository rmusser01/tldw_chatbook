#!/usr/bin/env python3
"""Guard: no two backlog task files may claim the same task id.

Task files are named ``task-<id> - <Title>.md``; subtask ids may be multi-dotted
(``task-3.1``, ``task-10.6.1``). Two branches minting the same id and merging
separately produces two files that share an id, and backlog CLI lookups become
ambiguous. This has happened ten-plus times (ids 152+, 196-203, 246-256, ...).

Both namespaces the repo relies on are checked, because they can disagree after
a hand-edited rename and the backlog CLI resolves by frontmatter:

* the filename prefix, and
* the YAML frontmatter ``id:`` field.

Extracted from the inline shell in ``.github/workflows/backlog-guard.yml``
(TASK-19572) so that workflow and ``derived-artifacts.yml`` cannot drift apart.
Stdlib-only: it runs with no dependency install, like the other derived-artifact
checkers.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "backlog" / "tasks"

FILENAME_ID_RE = re.compile(r"^(task-\d+(?:\.\d+)*) - .*\.md$")
FRONTMATTER_ID_RE = re.compile(r"^id:\s*(\S+)", re.IGNORECASE)

RESOLUTION = (
    "Resolve per the 2026-08-21 owner rule (TASK-19601): the OLDER arrival "
    "keeps the id; the younger task(s) renumber -- regardless of Done status -- "
    "with a Renumbering provenance section, updating dependencies: and doc/code "
    "references."
)


def _first_frontmatter_id(path: Path) -> str | None:
    """Return the file's first ``id:`` value, lowercased, or None.

    Only the first match is used: frontmatter sits at the top of the file, and a
    later ``id:`` inside prose (a code block, a quoted example) must not be
    mistaken for the task's identity.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        match = FRONTMATTER_ID_RE.match(line)
        if match:
            return match.group(1).strip().casefold()
    return None


def duplicate_ids(tasks_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Collect ids claimed by more than one task file.

    Args:
        tasks_dir: Directory holding ``task-<id> - <Title>.md`` files.

    Returns:
        tuple: ``(by_filename, by_frontmatter)`` -- each maps a duplicated id to
            the sorted filenames claiming it. Empty dicts when all ids are
            unique.
    """
    by_filename: dict[str, list[str]] = defaultdict(list)
    by_frontmatter: dict[str, list[str]] = defaultdict(list)
    for path in sorted(tasks_dir.glob("*.md")):
        match = FILENAME_ID_RE.match(path.name)
        if match:
            by_filename[match.group(1)].append(path.name)
        frontmatter_id = _first_frontmatter_id(path)
        if frontmatter_id:
            by_frontmatter[frontmatter_id].append(path.name)
    return (
        {key: sorted(v) for key, v in by_filename.items() if len(v) > 1},
        {key: sorted(v) for key, v in by_frontmatter.items() if len(v) > 1},
    )


def _report(label: str, duplicates: dict[str, list[str]]) -> None:
    print(f"::error::Duplicate backlog task IDs in {label}:")
    for task_id in sorted(duplicates):
        print(f"--- {task_id} ---")
        for name in duplicates[task_id]:
            print(f"  {name}")


def main(argv: list[str] | None = None) -> int:
    """Fail when any task id is claimed twice.

    Returns:
        int: ``0`` when every id is unique, ``1`` otherwise (with ``::error::``
            annotations naming each colliding id and its files).
    """
    # NOTE (Qodo PR #1947 finding 2, "Unvalidated --tasks-dir used"): this is
    # deliberately NOT routed through Utils/path_validation.py. Three reasons:
    #   1. path_validation.py imports Metrics.metrics_logger, which imports
    #      psutil -- a third-party package. This script (and the other
    #      derived-artifact checkers) are stdlib-only and install-free by
    #      design (TASK-19572's own AC; see the docstring above and
    #      .github/workflows/derived-artifacts.yml), so importing it would
    #      quietly break that contract.
    #   2. Neither CI workflow that runs this script ever passes --tasks-dir
    #      (backlog-guard.yml and derived-artifacts.yml both invoke it bare),
    #      so there is no CI-reachable, externally-controlled input here --
    #      only a developer's own CLI argument, typed in their own shell,
    #      reading files they already have OS-level access to. There is no
    #      privilege boundary for a "traversal" to cross.
    #   3. --tasks-dir is intentionally usable with a directory outside the
    #      repo: Tests/Architecture/test_derived_artifact_checkers.py passes
    #      a pytest `tmp_path` fixture (outside REPO_ROOT) to exercise this
    #      function in isolation. Confining it to the repo root would break
    #      that test and the flag's own purpose.
    # The operations here (`glob`, `read_text`) are read-only regardless.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=TASKS_DIR,
        help="directory of backlog task files (defaults to backlog/tasks)",
    )
    args = parser.parse_args(argv)

    if not args.tasks_dir.is_dir():
        print(f"::error::no backlog task directory at {args.tasks_dir}")
        return 1

    filename_dupes, frontmatter_dupes = duplicate_ids(args.tasks_dir)
    if filename_dupes:
        _report("filenames", filename_dupes)
    if frontmatter_dupes:
        _report("frontmatter id: fields", frontmatter_dupes)
    if filename_dupes or frontmatter_dupes:
        print(RESOLUTION, file=sys.stderr)
        return 1

    total = sum(1 for path in args.tasks_dir.glob("task-*.md"))
    print(
        f"No duplicate task IDs across {total} task files "
        "(filenames + frontmatter)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
