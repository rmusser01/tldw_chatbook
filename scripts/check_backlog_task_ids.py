#!/usr/bin/env python3
"""Guard: backlog task ids are unique and task paths are Windows-compatible.

Task files are named ``task-<id> - <Title>.md``; subtask ids may be multi-dotted
(``task-3.1``, ``task-10.6.1``). Two branches minting the same id and merging
separately produces two files that share an id, and backlog CLI lookups become
ambiguous. This has happened ten-plus times (ids 152+, 196-203, 246-256, ...).

Both namespaces the repo relies on are checked, because they can disagree after
a hand-edited rename and the backlog CLI resolves by frontmatter:

* the filename prefix, and
* the YAML frontmatter ``id:`` field.

All three directories the CLI resolves an id in are scanned -- ``backlog/tasks``,
``backlog/completed``, and ``backlog/archive/tasks`` -- because archiving a file
does not free its id. Scanning only ``backlog/tasks`` made TASK-2157 invisible
here while ``backlog task 2157`` stayed ambiguous on dev.

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
# Every directory the backlog CLI resolves an id in. Scanning only
# ``backlog/tasks`` let an archived file keep its id while hiding the collision:
# upstream Backlog.md 1.44.0 hands the id of an archived task straight back to
# the next ``task create`` (reproduced 2026-08-22), and the resulting pair then
# passed this guard. TASK-2157 was born exactly that way -- its older claimant
# was archived at 18:08 and the id reissued at 18:09.
TASK_DIRS = (
    REPO_ROOT / "backlog" / "tasks",
    REPO_ROOT / "backlog" / "completed",
    REPO_ROOT / "backlog" / "archive" / "tasks",
)

FILENAME_ID_RE = re.compile(r"^(task-\d+(?:\.\d+)*) - .*\.md$")
FRONTMATTER_ID_RE = re.compile(r"^id:\s*(\S+)", re.IGNORECASE)
WINDOWS_RESERVED_CHARACTERS = frozenset('<>:"/\\|?*')
WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{suffix}" for suffix in (*map(str, range(1, 10)), "¹", "²", "³")}
    | {f"lpt{suffix}" for suffix in (*map(str, range(1, 10)), "¹", "²", "³")}
)
WINDOWS_PATH_RESOLUTION = (
    "Keep punctuation in task content, but rename the task file with a "
    "Windows-safe spelling and update live path references."
)

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


def _label(path: Path) -> str:
    """Repo-relative path when there is one, else the absolute path.

    With several buckets in scope, a bare name cannot say whether the duplicate
    is a live task or an archived one -- which is the whole question. Paths
    outside the repo (``--tasks-dir`` pointed at a scratch directory) keep their
    full form for the same reason: two external buckets can both be named
    ``tasks``, and a basename would collapse them into identical rows.
    """
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def windows_incompatible_reason(name: str) -> str | None:
    """Return why a basename cannot be represented by Win32, else ``None``."""
    for character in name:
        if ord(character) <= 0x1F:
            return f"contains ASCII control U+{ord(character):04X}"
        if character in WINDOWS_RESERVED_CHARACTERS:
            return f"contains Windows-reserved character {character!r}"
    if name.endswith((".", " ")):
        return "ends with a dot or space"
    device_stem = name.split(".", 1)[0].casefold()
    if device_stem in WINDOWS_RESERVED_DEVICE_NAMES:
        return f"uses reserved Windows device name {device_stem.upper()!r}"
    return None


def windows_incompatible_paths(*task_dirs: Path) -> dict[str, str]:
    """Return directly contained files whose basenames Win32 rejects."""
    invalid: dict[str, str] = {}
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        for path in sorted(task_dir.iterdir()):
            if not path.is_file():
                continue
            reason = windows_incompatible_reason(path.name)
            if reason:
                invalid[_label(path)] = reason
    return invalid


def duplicate_ids(
    *task_dirs: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Collect ids claimed by more than one task file, across every bucket given.

    Args:
        *task_dirs: Directories holding ``task-<id> - <Title>.md`` files. A
            directory that does not exist is skipped, so a project without
            ``completed/`` or ``archive/tasks/`` is not an error.

    Returns:
        tuple: ``(by_filename, by_frontmatter)`` -- each maps a duplicated id to
            the sorted paths claiming it. Empty dicts when all ids are unique.
    """
    by_filename: dict[str, list[str]] = defaultdict(list)
    by_frontmatter: dict[str, list[str]] = defaultdict(list)
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        for path in sorted(task_dir.glob("*.md")):
            match = FILENAME_ID_RE.match(path.name)
            if match:
                by_filename[match.group(1)].append(_label(path))
            frontmatter_id = _first_frontmatter_id(path)
            if frontmatter_id:
                by_frontmatter[frontmatter_id].append(_label(path))
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


def _report_windows_incompatible(invalid: dict[str, str]) -> None:
    print("::error::Windows-incompatible Backlog task paths:")
    for path, reason in sorted(invalid.items()):
        print(f"  {path}: {reason}")


def main(argv: list[str] | None = None) -> int:
    """Fail when ids collide or a task path is incompatible with Windows.

    Returns:
        int: ``0`` when ids are unique and paths are Windows-compatible, ``1``
            otherwise (with ``::error::`` annotations naming each violation).
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
        action="append",
        dest="tasks_dirs",
        help=(
            "directory of backlog task files; repeatable. Defaults to every "
            "bucket the CLI resolves ids in: backlog/tasks, backlog/completed, "
            "backlog/archive/tasks"
        ),
    )
    args = parser.parse_args(argv)

    task_dirs = tuple(args.tasks_dirs) if args.tasks_dirs else TASK_DIRS
    if not any(task_dir.is_dir() for task_dir in task_dirs):
        print(
            f"::error::no backlog task directory at {', '.join(str(d) for d in task_dirs)}"
        )
        return 1

    filename_dupes, frontmatter_dupes = duplicate_ids(*task_dirs)
    invalid_paths = windows_incompatible_paths(*task_dirs)
    if filename_dupes:
        _report("filenames", filename_dupes)
    if frontmatter_dupes:
        _report("frontmatter id: fields", frontmatter_dupes)
    if invalid_paths:
        _report_windows_incompatible(invalid_paths)
    if filename_dupes or frontmatter_dupes:
        print(RESOLUTION, file=sys.stderr)
    if invalid_paths:
        print(WINDOWS_PATH_RESOLUTION, file=sys.stderr)
    if filename_dupes or frontmatter_dupes or invalid_paths:
        return 1

    scanned = [task_dir for task_dir in task_dirs if task_dir.is_dir()]
    total = sum(1 for task_dir in scanned for _ in task_dir.glob("task-*.md"))
    print(
        f"No duplicate task IDs across {total} task files in "
        f"{', '.join(_label(task_dir) for task_dir in scanned)} "
        "(filenames + frontmatter); all task paths are Windows-compatible."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
