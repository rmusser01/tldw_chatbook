#!/usr/bin/env python3
"""Measure how often real dev->PR-branch sync merges conflict, and on which files.

Replays every sync merge (a merge commit inside a PR branch whose second parent
brought `dev` in) found in the last N first-parent merges of `origin/dev`, with
`git merge-tree`, and tallies conflicting files. This is the method behind the
2026-09-27 CI spec's baseline (547 syncs, 48% conflicted).

Pass `--since DATE` (any date format git accepts) to only count syncs committed
on or after DATE, e.g. `--since 2026-09-27` to isolate syncs against the new
schema without recounting the pre-change history -- this is how the post-merge
review measures the rate drop.

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


def _sync_merges(first_parent_merges: int, since: str | None = None) -> list[str]:
    merges = _git(
        "log", "origin/dev", "--first-parent", "--merges", "--format=%H",
        "-n", str(first_parent_merges),
    ).stdout.split()
    # A candidate is only a real sync if its second parent actually came from
    # dev -- otherwise a branch merging some unrelated side branch into itself
    # gets miscounted as a dev sync (Qodo #4).
    dev_chain = set(_git("rev-list", "--first-parent", "origin/dev").stdout.split())
    since_args = [f"--since={since}"] if since else []
    syncs: set[str] = set()
    for merge in merges:
        candidates = _git(
            "log", "--merges", "--format=%H", *since_args, f"{merge}^1..{merge}^2",
        ).stdout.split()
        for candidate in candidates:
            second_parent = _git("rev-parse", f"{candidate}^2").stdout.strip()
            if second_parent in dev_chain:
                syncs.add(candidate)
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


def _positive_int(value: str) -> int:
    """argparse type: reject a --merges value below 1 (an empty scan is not success)."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return parsed


def main() -> int:
    """Print the sync-merge conflict rate and the files most often in conflict.

    Returns:
        int: process exit status (0); the measurement is printed to stdout.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merges", type=_positive_int, default=300,
                        help="first-parent merges of origin/dev to scan (default 300)")
    parser.add_argument("--since", default=None,
                        help="only count syncs committed on or after this date "
                             "(any format git accepts)")
    args = parser.parse_args()
    syncs = _sync_merges(args.merges, since=args.since)
    if args.since:
        print(f"window: last {args.merges} dev merges, syncs since {args.since}")
    else:
        print(f"window: last {args.merges} dev merges")
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
