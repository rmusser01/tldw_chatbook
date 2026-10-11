#!/usr/bin/env python3
"""Replay UI Fast Lane shard logs over the census (TASK-33007).

The UI lane runs ``scripts/ui_pr_gate_census.txt`` as round-robin shards, so
where a slow file sits decides which shard it lands in. This replays the
per-file seconds of real CI runs over the current census and prints the
pytest minutes each shard would carry; ``--write`` re-orders the census tail
(the last ``--tail`` entries) for the flattest split.

Per-file seconds move about 40% between runs on the hosted runners, so each
file is scored at the WORST of the runs given. Pass more than one run.

Usage:
    # one directory per CI run, one file per shard job:
    gh run view <run-id> --job <job-id> --log > run1/shard1.log
    python scripts/ui_lane_shard_replay.py --shards 8 run1 run2
    python scripts/ui_lane_shard_replay.py --shards 8 --tail 11 --write run1 run2
"""

from __future__ import annotations

import argparse
import itertools
import random
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CENSUS_PATH = REPO_ROOT / "scripts" / "ui_pr_gate_census.txt"
#: Seconds charged to an entry no log has timed (a new file or node id).
UNTIMED_SECONDS = 3.0

_STAMPED = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z (.*)$")
_FILE_LINE = re.compile(r"(Tests/UI/\S+\.py) ")
_DOTS_LINE = re.compile(r"^[.FEsxX]+\s")


def file_seconds(log_text: str) -> dict[str, float]:
    """Seconds each test file took in one shard's job log.

    pytest prints a file's name when it starts it and its last progress line
    when it ends, so a file's time runs from the previous file's last line
    (or the "collected" line) to its own.

    Args:
        log_text: A ``gh run view --log`` dump of one UI Fast Lane shard.

    Returns:
        Seconds by ``Tests/UI/...`` path; a file listed by several node ids
        is one entry.
    """
    last: dict[str, datetime] = {}
    order: list[str] = []
    current: str | None = None
    start: datetime | None = None
    for line in log_text.splitlines():
        match = _STAMPED.search(line)
        if match is None:
            continue
        stamp = datetime.fromisoformat(match.group(1)[:26])
        body = match.group(2)
        if "collected " in body and " item" in body:
            start = stamp
        named = _FILE_LINE.match(body)
        if named:
            current = named.group(1)
            order.append(current)
            last[current] = stamp
        elif current and _DOTS_LINE.match(body):
            last[current] = stamp
    seconds: dict[str, float] = {}
    previous = start
    for name in order:
        if previous is not None:
            seconds[name] = (
                seconds.get(name, 0.0) + (last[name] - previous).total_seconds()
            )
        previous = last[name]
    return seconds


def run_seconds(run_dir: Path) -> dict[str, float]:
    """Per-file seconds of one CI run: every shard log in ``run_dir``."""
    seconds: dict[str, float] = {}
    for log in sorted(run_dir.glob("*.log")):
        seconds.update(file_seconds(log.read_text(encoding="utf-8", errors="replace")))
    return seconds


def worst_case(runs: list[dict[str, float]]) -> dict[str, float]:
    """Each file at the slowest of the runs that timed it."""
    names = set().union(*runs) if runs else set()
    return {name: max(run.get(name, 0.0) for run in runs) for name in names}


def shard_loads(
    entries: list[str], seconds: dict[str, float], total: int
) -> list[float]:
    """Seconds each of ``total`` round-robin shards carries.

    A file listed by node ids is timed once in a log, so its time is charged
    to its first entry and its other entries cost nothing.
    """
    loads = [0.0] * total
    charged: set[str] = set()
    for index, entry in enumerate(entries):
        path = entry.split("::", 1)[0]
        if "::" in entry:
            if path in charged:
                continue
            charged.add(path)
        loads[index % total] += seconds.get(path, UNTIMED_SECONDS)
    return loads


def flattest_tail(
    head: list[str], tail: list[str], seconds: dict[str, float], total: int
) -> list[str]:
    """Order ``tail`` so the heaviest shard is as light as a swap search finds."""

    def score(order: list[str]) -> tuple[float, float]:
        loads = shard_loads(head + order, seconds, total)
        return max(loads), max(loads) - min(loads)

    rng = random.Random(11)
    best: tuple[tuple[float, float], list[str]] | None = None
    for _ in range(300):
        order = tail[:]
        rng.shuffle(order)
        current = score(order)
        improved = True
        while improved:
            improved = False
            for a, b in itertools.combinations(range(len(order)), 2):
                order[a], order[b] = order[b], order[a]
                swapped = score(order)
                if swapped < current:
                    current, improved = swapped, True
                else:
                    order[a], order[b] = order[b], order[a]
        if best is None or current < best[0]:
            best = (current, order[:])
    return best[1] if best else tail


def _minutes(loads: list[float]) -> str:
    return " ".join(f"{load / 60:.1f}" for load in loads) + (
        f"  max {max(loads) / 60:.1f} min"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="one directory per CI run")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--tail", type=int, default=0, help="entries to re-order")
    parser.add_argument("--write", action="store_true", help="rewrite the census tail")
    args = parser.parse_args(argv)

    seconds = worst_case([run_seconds(run) for run in args.runs])
    if not seconds:
        print("no shard logs with pytest file lines found", file=sys.stderr)
        return 2
    lines = CENSUS_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    entries = [
        line.strip() for line in lines if line.strip() and not line.startswith("#")
    ]
    print(
        f"{args.shards} shards, as listed:  {_minutes(shard_loads(entries, seconds, args.shards))}"
    )
    if not args.tail:
        return 0
    head, tail = entries[: -args.tail], entries[-args.tail :]
    order = flattest_tail(head, tail, seconds, args.shards)
    print(
        f"{args.shards} shards, best tail:  {_minutes(shard_loads(head + order, seconds, args.shards))}"
    )
    if args.write:
        kept = [line for line in lines if line.strip() not in tail]
        CENSUS_PATH.write_text(
            "".join(kept) + "".join(f"{entry}\n" for entry in order), encoding="utf-8"
        )
        print(f"rewrote the last {args.tail} entries of {CENSUS_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
