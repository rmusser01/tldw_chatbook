#!/usr/bin/env python3
"""Guard: every backlog task file can actually be read.

A task file with broken frontmatter is not a loud failure. Every backlog reader
-- ``task list``, the board, search, the MCP read tools -- skips it with one log
line and carries on, so the task simply stops existing. Thirty-two files had
accumulated in that state before anyone noticed (repaired in PR #1954); the
warnings interleave with output, so the damage read as two or three files.

Only the shapes that provably break a reader are checked, and only with the
standard library, because the workflows that run this install nothing:

* the frontmatter block must open on line 1 and close;
* a plain (unquoted) scalar must not contain ``": "`` -- YAML reads the second
  colon as another mapping key, which is what ``title: Console transcript: skip
  move_child`` did to twenty-eight files;
* a quoted scalar must actually close, with interior quotes escaped -- what
  ``title: 'Impersonate drafts the user's next reply'`` did to three more;
* an owned ``<!-- SECTION:NAME:BEGIN -->`` must be closed, since an
  unterminated section is a hard parse error and swallows every heading below
  it. A stray ``:END`` with no BEGIN is *not* flagged: the real parser ignores
  it, and nine files on dev carry one.

Deliberately NOT a YAML parser. PyYAML is a project dependency but is not
installed in `derived-artifacts.yml` (see its "nothing is installed" note), and
a partial re-implementation would fail differently from the real readers. These
four rules are the ones with evidence behind them; anything subtler is left to
``backlog-py doctor``, which uses the real parser.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The buckets a task file can live in, mirroring check_backlog_task_ids.py.
TASK_DIRS = (
    REPO_ROOT / "backlog" / "tasks",
    REPO_ROOT / "backlog" / "completed",
    REPO_ROOT / "backlog" / "archive" / "tasks",
)

FRONTMATTER_DELIMITER = "---"
# A frontmatter mapping line: an unindented key, then the value.
MAPPING_LINE_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*):(?P<value>.*)$")
SECTION_BEGIN_RE = re.compile(r"^<!-- SECTION:(?P<name>[A-Z0-9_ -]+):BEGIN -->\s*$")
SECTION_END_RE = re.compile(r"^<!-- SECTION:(?P<name>[A-Z0-9_ -]+):END -->\s*$")

RESOLUTION = (
    "Run `backlog-py doctor --fix` (backlog-md-py >= 2.0.2), which repairs these "
    "shapes and re-parses each file before writing it, or quote the title by hand."
)


def _label(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _scalar_problem(value: str) -> str | None:
    """Why a single-line frontmatter value would not load, or None.

    Multi-line values are handled by the caller, which owns the surrounding
    lines; this only judges what fits on one.
    """
    scalar = value.strip()
    if not scalar or scalar in {">", ">-", "|", "|-", "[]", "{}"}:
        return None
    quote = scalar[0]
    if quote in "'\"":
        interior = scalar[1:-1]
        # A single-quoted scalar escapes a quote by doubling it, a double-quoted
        # one with a backslash. An odd one out means the scalar ended early and
        # the remainder is stray tokens -- `title: 'the user's next reply'`.
        if quote == "'" and interior.replace("''", "").count("'"):
            return f"single-quoted value has an unescaped apostrophe: {scalar[:60]}"
        if quote == '"' and re.search(r'(?<!\\)"', interior):
            return f"double-quoted value has an unescaped quote: {scalar[:60]}"
        return None
    if ": " in scalar or scalar.endswith(":"):
        return f"unquoted value contains a colon and must be quoted: {scalar[:60]}"
    return None


def _closes_on_a_continuation_line(lines: list[str], start: int, stop: int, quote: str) -> bool:
    """Whether a wrapped quoted scalar closes before the next key.

    YAML lets a quoted scalar run across indented continuation lines, which is
    how the backlog writers emit a long title. Judging only the first line would
    report every wrapped title as unterminated.
    """
    for index in range(start, stop):
        line = lines[index]
        if line[:1] not in (" ", "\t"):
            return False
        if line.strip().endswith(quote):
            return True
    return False


def frontmatter_problems(text: str) -> list[str]:
    """Find frontmatter that would not survive a YAML load.

    Args:
        text: Full task-file source, frontmatter included.

    Returns:
        list[str]: One reason per problem, in reading order. Empty when the
            frontmatter block is well formed.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        return ["no frontmatter block: the file does not open with ---"]
    closing = next(
        (i for i in range(1, len(lines)) if lines[i].strip() == FRONTMATTER_DELIMITER),
        None,
    )
    if closing is None:
        return ["frontmatter block is never closed by a second ---"]

    problems: list[str] = []
    for index in range(1, closing):
        match = MAPPING_LINE_RE.match(lines[index])
        if match is None:
            continue
        value = match.group("value").strip()
        quote = value[:1]
        # `quote and` matters: "" is a substring of every string, so an empty
        # value (a key introducing a block sequence) would take the quoted path.
        if quote and quote in "'\"" and not (len(value) > 1 and value.endswith(quote)):
            if not _closes_on_a_continuation_line(lines, index + 1, closing, quote):
                problems.append(
                    f"{match.group('key')}: quoted value is never closed: {value[:60]}"
                )
            continue
        problem = _scalar_problem(value)
        if problem is not None:
            problems.append(f"{match.group('key')}: {problem}")
    return problems


def section_problems(text: str) -> list[str]:
    """Find owned sections that never close.

    A stray ``:END`` without a BEGIN is deliberately not reported: the real
    parser ignores it, and flagging it would fail nine files on dev that every
    reader handles fine.

    Args:
        text: Full task-file source.

    Returns:
        list[str]: One reason per unterminated section, in reading order.
    """
    open_names: list[str] = []
    for line in text.split("\n"):
        begin = SECTION_BEGIN_RE.match(line)
        if begin is not None:
            open_names.append(begin.group("name"))
            continue
        end = SECTION_END_RE.match(line)
        if end is not None and end.group("name") in open_names:
            open_names.remove(end.group("name"))
    return [f"SECTION:{name}:BEGIN is never closed" for name in open_names]


def unreadable_task_files(*task_dirs: Path) -> dict[str, list[str]]:
    """Collect the task files no backlog reader can parse.

    Args:
        *task_dirs: Directories of ``task-<id> - <Title>.md`` files. One that
            does not exist is skipped, so a project without ``completed/`` or
            ``archive/tasks/`` is not an error.

    Returns:
        dict[str, list[str]]: Labeled path -> every reason that file is
            unreadable. Empty when every file parses.
    """
    unreadable: dict[str, list[str]] = {}
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        for path in sorted(task_dir.glob("task-*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                # A file that is not valid UTF-8 is exactly what this checker
                # exists to name. Letting UnicodeDecodeError escape would fail
                # the required check with a traceback and hide which file it was.
                unreadable[_label(path)] = [str(error)]
                continue
            problems = frontmatter_problems(text) + section_problems(text)
            if problems:
                unreadable[_label(path)] = problems
    return unreadable


def main(argv: list[str] | None = None) -> int:
    """Fail when any task file cannot be read by the backlog tooling.

    Args:
        argv: Command-line arguments, or None to read ``sys.argv``.

    Returns:
        int: ``0`` when every task file parses, ``1`` otherwise (with
            ``::error::`` annotations naming each file and its problems).
    """
    # NOTE (Qodo, "Unvalidated --tasks-dir used"): deliberately NOT routed
    # through Utils/path_validation.py, for the same three reasons recorded in
    # check_backlog_task_ids.py, where this finding was first declined on
    # PR #1947:
    #   1. path_validation.py imports Metrics.metrics_logger -> psutil. These
    #      checkers are stdlib-only and install-free by design; derived-
    #      artifacts.yml installs nothing.
    #   2. Neither workflow passes --tasks-dir; both invoke the script bare, so
    #      there is no CI-reachable externally-controlled input and no privilege
    #      boundary for a traversal to cross.
    #   3. --tasks-dir is intentionally usable outside the repo, which is how
    #      Tests/Architecture/test_derived_artifact_checkers.py exercises it
    #      with a pytest tmp_path.
    # Every operation here (glob, read_text) is read-only.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        action="append",
        dest="tasks_dirs",
        help="directory of backlog task files; repeatable. Defaults to every bucket.",
    )
    args = parser.parse_args(argv)

    task_dirs = tuple(args.tasks_dirs) if args.tasks_dirs else TASK_DIRS
    if not any(task_dir.is_dir() for task_dir in task_dirs):
        print(f"::error::no backlog task directory at {', '.join(str(d) for d in task_dirs)}")
        return 1

    unreadable = unreadable_task_files(*task_dirs)
    if unreadable:
        print("::error::Backlog task files that no reader can parse:")
        for label in sorted(unreadable):
            print(f"--- {label} ---")
            for problem in unreadable[label]:
                print(f"  {problem}")
        print(RESOLUTION, file=sys.stderr)
        return 1

    scanned = [task_dir for task_dir in task_dirs if task_dir.is_dir()]
    total = sum(1 for task_dir in scanned for _ in task_dir.glob("task-*.md"))
    print(
        f"All {total} task files in {', '.join(_label(d) for d in scanned)} are readable "
        "(frontmatter + owned-section markers)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
