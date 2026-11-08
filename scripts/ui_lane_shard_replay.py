#!/usr/bin/env python3
"""Replay UI Fast Lane shard logs over the census (TASK-33007).

The UI lane runs ``scripts/ui_pr_gate_census.txt`` as round-robin shards, so
where a slow file sits decides which shard it lands in. This replays the
file and node seconds of real CI runs over the current census and prints the
pytest minutes each shard would carry; ``--write`` re-orders the census tail
(the last ``--tail`` entries) for the flattest split. Each tail entry moves
with the comment lines directly above it (the "what it pins" note); the tail
block's own header stays put, so end it with a blank line before the first
entry, or that entry takes the header with it.

Timings move about 40% between hosted runs. Each file/node is scored at its
WORST observed run. Pass more than one run. Node costs use pytest duration
rows (all phases and parameter cases); old top-25 logs leave an evenly split
unmeasured remainder. This is an estimate of pytest work, not job wall time:
collection, shutdown and runner setup must be measured separately. The UI
lane prints all durations so future runs do not need the old even-split guess.

Usage:
    # one directory per CI run, one file per shard job; --shards is the UI
    # lane's matrix size, --tail the number of entries to re-order, counted
    # back from the end of the census (the block's announced count plus any
    # entries appended after it; such entries must not be blank-separated,
    # or --write carries the blank into the block):
    gh run view <run-id> --job <job-id> --log > run1/shard1.log
    python scripts/ui_lane_shard_replay.py --shards 34 run1 run2
    python scripts/ui_lane_shard_replay.py --shards 34 --tail 92 --write run1 run2

TASK-39144: ``--rebalance MINUTES`` swaps entries between shards, across the
whole census, until no shard's replay passes MINUTES of pytest (the census
header names the value that keeps a job 5 min under its cap); ``--charge
ENTRY=SECONDS`` prices an entry no log timed. ``--write`` then swaps those
entries in the census, each with the comment lines directly above it::

    python scripts/ui_lane_shard_replay.py --shards 34 --rebalance 13.2 \\
        --charge Tests/UI/test_new.py=40 --write run1 run2
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import random
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Loaded by path, as the other scripts load their siblings, so a `scripts`
# package elsewhere on sys.path can never shadow the checker.
_CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_ui_pr_gate_census", REPO_ROOT / "scripts" / "check_ui_pr_gate_census.py"
)
_checker = importlib.util.module_from_spec(_CHECKER_SPEC)
_CHECKER_SPEC.loader.exec_module(_checker)
read_census = _checker.read_census

CENSUS_PATH = REPO_ROOT / "scripts" / "ui_pr_gate_census.txt"
#: Seconds charged to an entry no log has timed (a new file or node id).
UNTIMED_SECONDS = 3.0

_STAMPED = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z (.*)$")
_FILE_LINE = re.compile(r"(Tests/UI/\S+\.py) ")
_DOTS_LINE = re.compile(r"^[.FEsxX]+\s")
_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)s\s+(?:setup|call|teardown)\s+(Tests/UI/\S+\.py::.+)$"
)


def file_seconds(log_text: str) -> dict[str, float]:
    """Seconds each test file took in one shard's job log.

    pytest prints a file's name when it starts it and its last progress line
    when it ends, so a file's time runs from the previous file's last line to
    its own. The first file's runs from the "collected" line or, in a log
    without one (``pytest -q`` prints none), from the last line stamped
    before it.

    Args:
        log_text: A ``gh run view --log`` dump of one UI Fast Lane shard.

    Returns:
        Seconds by ``Tests/UI/...`` path; a file listed by several node ids
        is one entry.
    """
    occurrences: list[tuple[str, datetime]] = []
    collected: datetime | None = None
    before_first: datetime | None = None
    for line in log_text.splitlines():
        match = _STAMPED.search(line)
        if match is None:
            continue
        stamp = datetime.fromisoformat(match.group(1)[:26])
        body = match.group(2)
        named = _FILE_LINE.match(body)
        if named:
            occurrences.append((named.group(1), stamp))
        elif not occurrences:
            before_first = stamp
            if "collected " in body and " item" in body:
                collected = stamp
        elif _DOTS_LINE.match(body):
            occurrences[-1] = (occurrences[-1][0], stamp)
    seconds: dict[str, float] = {}
    previous = collected or before_first
    for name, ended in occurrences:
        if previous is not None:
            seconds[name] = seconds.get(name, 0.0) + (ended - previous).total_seconds()
        previous = ended
    return seconds


def run_seconds(run_dir: Path) -> dict[str, float]:
    """File and per-node phase seconds of one CI run in ``run_dir``.

    A file whose node ids land in several shards is timed in each of their
    logs; its seconds are the sum.
    """
    seconds: dict[str, float] = {}
    for log in sorted(run_dir.glob("*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        for name, spent in file_seconds(text).items():
            seconds[name] = seconds.get(name, 0.0) + spent
        for line in text.splitlines():
            stamped = _STAMPED.search(line)
            duration = _DURATION.match(stamped.group(2)) if stamped else None
            if duration:
                node = duration.group(2).rstrip()
                seconds[node] = seconds.get(node, 0.0) + float(duration.group(1))
    return seconds


def worst_case(runs: list[dict[str, float]]) -> dict[str, float]:
    """Each file/node at the slowest of the runs that timed it."""
    names = set().union(*runs) if runs else set()
    return {name: max(run.get(name, 0.0) for run in runs) for name in names}


def entry_seconds(entries: list[str], seconds: dict[str, float]) -> dict[str, float]:
    """Charge measured nodes directly; share only unmeasured file remainder.

    Function/class selectors include their parameter cases and descendants.
    Old logs report only the slowest tests: unknown selectors share the file's
    remaining seconds. With all selectors measured, any file overhead is
    shared equally. Per-node worst cases may exceed a file's worst case; keep
    those costs rather than scale measured work down.
    """
    by_file: dict[str, list[str]] = {}
    for entry in entries:
        by_file.setdefault(entry.split("::", 1)[0], []).append(entry)
    costs: dict[str, float] = {}
    for path, selectors in by_file.items():
        measured: dict[str, float] = {}
        for entry in selectors:
            if entry == path:
                continue
            matches = [
                spent
                for node, spent in seconds.items()
                if node == entry
                or node.startswith(entry + "[")
                or node.startswith(entry + "::")
            ]
            if matches:
                measured[entry] = sum(matches)
        remainder = max(
            seconds.get(path, UNTIMED_SECONDS) - sum(measured.values()), 0.0
        )
        unknown = [entry for entry in selectors if entry not in measured]
        for entry in selectors:
            costs[entry] = measured.get(entry, 0.0)
            if entry in unknown:
                costs[entry] += remainder / len(unknown)
            elif not unknown:
                costs[entry] += remainder / len(selectors)
    return costs


def shard_loads(
    entries: list[str], seconds: dict[str, float], total: int
) -> list[float]:
    """Seconds each round-robin shard carries, using node times when available.

    Dealt by the checker's own ``shard``, the function the UI lane calls.
    """
    costs = entry_seconds(entries, seconds)
    return [
        sum(costs[entry] for entry in _checker.shard(entries, index, total))
        for index in range(total)
    ]


def rebalanced(
    entries: list[str],
    costs: dict[str, float],
    total: int,
    target: float,
    fixed: frozenset[str] = frozenset(),
) -> tuple[list[str], int]:
    """Swap entries between shards until each carries at most ``target`` s.

    TASK-39144. A swap trades two entries' census positions, so each shard
    keeps its slot count and every other entry its slot; the checker's own
    ``shard`` still deals the result. Each step takes the heaviest shard and
    the swap with another shard that leaves the pair's heavier side lightest,
    and only while that shard is over ``target``, so a census already under
    it does not move. Every swap lowers the sum of squared loads, so the
    search ends; it stops early when no swap lowers the heaviest shard.

    Args:
        entries: The census, in file order.
        costs: Seconds per entry (``entry_seconds``).
        total: The UI lane's shard count.
        target: The most seconds a shard may carry.
        fixed: Entries that must keep their slot.

    Returns:
        The new census order and the number of swaps it took.
    """
    order = list(entries)
    slots = [_checker.shard(list(range(len(order))), index, total) for index in range(total)]
    loads = [sum(costs[order[slot]] for slot in shard) for shard in slots]
    swaps = 0
    while True:
        heavy = max(range(total), key=loads.__getitem__)
        if loads[heavy] <= target:
            return order, swaps
        best = None
        for light in range(total):
            gap = loads[heavy] - loads[light]
            for a in slots[heavy] if gap > 0 else ():
                for b in slots[light]:
                    moved = costs[order[a]] - costs[order[b]]
                    if not 0 < moved < gap or {order[a], order[b]} & fixed:
                        continue
                    peak = max(loads[heavy] - moved, loads[light] + moved)
                    if best is None or peak < best[0]:
                        best = (peak, a, b, light, moved)
        if best is None:
            return order, swaps
        _, a, b, light, moved = best
        order[a], order[b] = order[b], order[a]
        loads[heavy] -= moved
        loads[light] += moved
        swaps += 1


def flattest_tail(
    head: list[str], tail: list[str], seconds: dict[str, float], total: int
) -> list[str]:
    """Order ``tail`` so the heaviest shard is as light as a swap search finds."""

    costs = entry_seconds(head + tail, seconds)

    def score(order: list[str]) -> tuple[float, float]:
        loads = [0.0] * total
        for index, entry in enumerate(head + order):
            loads[index % total] += costs[entry]
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
    parser.add_argument("--census", type=Path, default=CENSUS_PATH)
    parser.add_argument(
        "--charge",
        action="append",
        default=[],
        metavar="ENTRY=SECONDS",
        help="seconds for an entry no log timed (charge it its local serial time; see the census header)",
    )
    parser.add_argument(
        "--rebalance",
        type=float,
        metavar="MINUTES",
        help="swap entries across the whole census until no shard carries more pytest minutes",
    )
    args = parser.parse_args(argv)

    seconds = worst_case([run_seconds(run) for run in args.runs])
    if not seconds:
        print("no shard logs with pytest file lines found", file=sys.stderr)
        return 2
    for charge in args.charge:
        entry, _, spent = charge.rpartition("=")
        seconds[entry] = float(spent)
    entries = read_census(args.census)  # what the lane deals
    print(
        f"{args.shards} shards, as listed:  {_minutes(shard_loads(entries, seconds, args.shards))}"
    )
    if args.rebalance is not None:
        text = args.census.read_text(encoding="utf-8")
        order, swaps = rebalanced(
            entries,
            entry_seconds(entries, seconds),
            args.shards,
            args.rebalance * 60,
            anchored(text),
        )
        print(
            f"{args.shards} shards, {swaps} swaps:  {_minutes(shard_loads(order, seconds, args.shards))}"
        )
        if args.write and swaps:
            args.census.write_text(_swapped(text, order), encoding="utf-8")
            print(f"swapped {swaps} pairs of entries in {args.census.name}")
        return 0
    if not args.tail:
        return 0
    head, tail = entries[: -args.tail], entries[-args.tail :]
    order = flattest_tail(head, tail, seconds, args.shards)
    print(
        f"{args.shards} shards, best tail:  {_minutes(shard_loads(head + order, seconds, args.shards))}"
    )
    if args.write:
        if len(set(entries)) != len(entries):
            print("the census lists an entry twice: fix it first", file=sys.stderr)
            return 2
        args.census.write_text(_reordered(args.census, tail, order), encoding="utf-8")
        print(f"rewrote the last {args.tail} entries of {args.census.name}")
    return 0


def _entry_spans(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Each entry's lines: the comment run right above it, then the entry."""
    spans = {}
    for index, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            start = index
            while start and lines[start - 1].lstrip().startswith("#"):
                start -= 1
            spans[line.strip()] = (start, index + 1)
    return spans


def anchored(text: str) -> frozenset[str]:
    """Entries whose comments open a block (after a blank line or the file's
    start): those comments head the block, so --rebalance leaves them be."""
    lines = text.splitlines(keepends=True)
    return frozenset(
        entry
        for entry, (start, _) in _entry_spans(lines).items()
        if start == 0 or not lines[start - 1].strip()
    )


def _swapped(text: str, order: list[str]) -> str:
    """The census text with its entries in ``order``, each with its comments.

    Blank lines and block-opening entries (``anchored``) keep their place.
    """
    lines = (text if text.endswith("\n") else text + "\n").splitlines(keepends=True)
    spans = _entry_spans(lines)
    out: list[str] = []
    at = 0
    for old, new in zip(spans, order):
        start, end = spans[old]
        out += lines[at:start]
        out += lines[slice(*spans[new])]
        at = end
    return "".join(out + lines[at:])


def _reordered(census: Path, tail: list[str], order: list[str]) -> str:
    """The census text with ``tail`` dealt in ``order``.

    An entry carries the comment lines (and blanks) between it and the entry
    before it. The first tail entry carries only its own comment run: a blank
    line above that run ends the block header, which stays where it is.
    """
    text = census.read_text(encoding="utf-8")
    lines = (text if text.endswith("\n") else text + "\n").splitlines(keepends=True)
    tail_set = set(tail)
    first = next(i for i, line in enumerate(lines) if line.strip() in tail_set)
    start = first
    while start and lines[start - 1].strip().startswith("#"):
        start -= 1
    groups: dict[str, list[str]] = {}
    pending: list[str] = []
    for line in lines[start:]:
        pending.append(line)
        if line.strip() in tail_set:
            groups[line.strip()], pending = pending, []
    return (
        "".join(lines[:start])
        + "".join("".join(groups[entry]) for entry in order)
        + "".join(pending)  # comments after the last entry stay last
    )


if __name__ == "__main__":
    raise SystemExit(main())
