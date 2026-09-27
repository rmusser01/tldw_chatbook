#!/usr/bin/env python3
"""Measure how often real dev->PR-branch sync merges conflict, and on which files.

Replays every sync merge (a merge commit inside a PR branch whose second parent
brought `dev` in) found in the last N first-parent merges of `origin/dev`, with
`git merge-tree`, and tallies conflicting files. This is the method behind the
2026-09-27 CI spec's baseline (547 syncs, 48% conflicted).

Caveat: `git merge-tree` honours the LOCAL `.gitattributes` merge drivers, while
GitHub's server-side merge does not, so a local `merge=` attribute would make the
rate look better than GitHub sees it.

A git error on any sync aborts the run: a silently skipped merge would understate the rate.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def _sync_merges(first_parent_merges: int) -> list[str]:
    merges = _git(
        "log", "origin/dev", "--first-parent", "--merges", "--format=%H",
        "-n", str(first_parent_merges),
    ).stdout.split()
    syncs: set[str] = set()
    for merge in merges:
        syncs.update(
            _git("log", "--merges", "--format=%H", f"{merge}^1..{merge}^2").stdout.split()
        )
    return sorted(syncs)


def _conflicted_files(sync: str) -> list[str] | None:
    result = _git("merge-tree", "--write-tree", "--name-only", f"{sync}^1", f"{sync}^2")
    if result.returncode == 0:
        return None
    if result.returncode != 1:
        raise RuntimeError(
            f"git merge-tree failed for {sync} (exit {result.returncode}): {result.stderr.strip()}"
        )
    lines = result.stdout.splitlines()[1:]
    return [
        line for line in lines
        if line and not line.startswith(("Auto-merging", "CONFLICT"))
    ]


def main() -> int:
    """Print the sync-merge conflict rate and the files most often in conflict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merges", type=int, default=300,
                        help="first-parent merges of origin/dev to scan (default 300)")
    args = parser.parse_args()
    syncs = _sync_merges(args.merges)
    per_file: Counter[str] = Counter()
    conflicted = 0
    for sync in syncs:
        files = _conflicted_files(sync)
        if files is None:
            continue
        conflicted += 1
        per_file.update(set(files))
    pct = (100 * conflicted // len(syncs)) if syncs else 0
    print(f"sync merges: {len(syncs)}")
    print(f"conflicted: {conflicted} ({pct}%)")
    for path, count in per_file.most_common(15):
        print(f"  {count:5d}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
