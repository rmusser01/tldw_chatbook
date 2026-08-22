"""Local gate for duplicate backlog task ids (TASK-19573's last AC).

Two branches minting the same id and merging separately produce two task
files that share an id; `backlog` CLI lookups then resolve ambiguously.
This has happened repeatedly -- ids 152+, 196-203, 246-256, `task-3401`
(TASK-13158), and most recently a batch of seven (TASK-19573).

`.github/workflows/backlog-guard.yml` already fails on this, but only
*after* a push: the collision costs a CI cycle and a PR round-trip to
discover, and the workflow's output is the only place it is visible.
TASK-19573 shipped its renumbering with this AC open:

    "A **local** gate exists -- a pytest that fails on duplicate ids -- so
     a collision is caught before push rather than by a workflow nobody
     can run."

This is that gate. It checks BOTH namespaces the guard checks, because
they can disagree after a hand-edited rename and the backlog CLI resolves
by frontmatter:

* the filename prefix (``task-<id> - <Title>.md``, ids may be multi-dotted)
* the first ``id:`` line of the frontmatter (stored as ``TASK-NNN``)
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = PROJECT_ROOT / "backlog" / "tasks"

#: Mirrors the workflow's `sed -nE 's/^(task-[0-9]+(\.[0-9]+)*) - .*\.md$/\1/p'`.
_FILENAME_ID = re.compile(r"^(task-[0-9]+(?:\.[0-9]+)*) - .*\.md$")


def _task_files() -> list[Path]:
    return sorted(TASKS_DIR.glob("task-*.md"))


def _format(dupes: dict[str, list[str]]) -> str:
    lines = []
    for task_id, names in sorted(dupes.items()):
        lines.append(f"--- {task_id} ---")
        lines.extend(f"    {name}" for name in sorted(names))
    lines.append(
        "Resolve per the owner rule (TASK-19601): the OLDER arrival keeps the "
        "id; the younger task(s) renumber -- regardless of Done status -- with "
        "a Renumbering provenance section, updating dependencies: and "
        "doc/code references."
    )
    return "\n".join(lines)


@pytest.fixture(scope="module")
def task_files() -> list[Path]:
    files = _task_files()
    # Guard the guard: a wrong PROJECT_ROOT would make every assertion below
    # vacuously true, which is how a green gate hides a real collision.
    assert files, f"no task files found under {TASKS_DIR}"
    return files


def test_no_duplicate_task_ids_in_filenames(task_files: list[Path]) -> None:
    by_id: dict[str, list[str]] = defaultdict(list)
    for path in task_files:
        match = _FILENAME_ID.match(path.name)
        if match:
            by_id[match.group(1)].append(path.name)
    dupes = {k: v for k, v in by_id.items() if len(v) > 1}
    assert not dupes, "Duplicate backlog task IDs in filenames:\n" + _format(dupes)


def test_no_duplicate_task_ids_in_frontmatter(task_files: list[Path]) -> None:
    by_id: dict[str, list[str]] = defaultdict(list)
    for path in task_files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("id:"):
                by_id[line.split(":", 1)[1].strip().lower()].append(path.name)
                break  # first `id:` only, matching the workflow's awk
    dupes = {k: v for k, v in by_id.items() if len(v) > 1}
    assert not dupes, (
        "Duplicate backlog task IDs in frontmatter id: fields:\n" + _format(dupes)
    )


def test_every_task_file_declares_a_frontmatter_id(task_files: list[Path]) -> None:
    """A file with no ``id:`` is invisible to the frontmatter check above.

    Without this, deleting an id line would *silence* a collision rather
    than surface it -- the failure mode the guard exists to prevent.
    """
    missing = [
        path.name
        for path in task_files
        if not any(
            line.startswith("id:")
            for line in path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        )
    ]
    assert not missing, "Task files with no frontmatter id:\n" + "\n".join(
        f"    {name}" for name in sorted(missing)
    )
