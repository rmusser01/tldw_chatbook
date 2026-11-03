#!/usr/bin/env python3
"""Ruff ratchet gate (TASK-38600, ADR 287).

Runs ruff over ``tldw_chatbook`` with the project selection (F, B, ASYNC,
PLE), aggregates findings PER RULE, and compares them against the checked-in
baseline ``Tests/lint/ruff_baseline.txt``:

- any rule whose count EXCEEDS the baseline fails the gate;
- a rule that is lower than the baseline (or absent entirely) passes, and the
  delta is printed so shrinking is visible;
- a rule code that does not appear in the baseline at all is treated as
  baseline 0 (new debt fails immediately).

The ruff version is pinned in CI (see ``.github/workflows/lint.yml``) and
recorded in the baseline header; regenerate the baseline with the SAME
version:

    python Tests/lint/ruff_gate.py --update-baseline

Exit codes: 0 = gate passed (all rules at or under baseline), 1 = gate
failed (some rule exceeded its baseline or ruff could not run), 2 = usage
error.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = "tldw_chatbook"
SELECT = "F,B,ASYNC,PLE"
BASELINE_PATH = Path(__file__).resolve().parent / "ruff_baseline.txt"

_RULE_RE = re.compile(r"\b((?:F|B|ASYNC|PLE)\d+)\b")
_BASELINE_RE = re.compile(r"^\s*([A-Z]+\d+)\s+(\d+)\s*$")
# Ruff exits 1 when it has findings; anything above that is a real failure.
_RUFF_FINDINGS_EXIT = 1


def _run_ruff() -> dict[str, int]:
    """Run ruff and return per-rule finding counts (empty dict = clean)."""
    ruff = shutil.which("ruff")
    if ruff is None:
        print("ERROR: ruff not found on PATH", file=sys.stderr)
        sys.exit(1)
    proc = subprocess.run(
        [ruff, "check", TARGET, "--select", SELECT, "--output-format", "concise"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, _RUFF_FINDINGS_EXIT):
        print(f"ERROR: ruff exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)
    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        match = _RULE_RE.search(line)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    return counts


def _load_baseline() -> dict[str, int]:
    baseline: dict[str, int] = {}
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        match = _BASELINE_RE.match(line)
        if match:
            baseline[match.group(1)] = int(match.group(2))
    return baseline


def _write_baseline(counts: dict[str, int]) -> None:
    ruff_version = subprocess.run(
        [shutil.which("ruff") or "ruff", "--version"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    lines = [
        "# ruff ratchet baseline (TASK-38600; ADR 287 "
        "backlog/decisions/287-ruff-lint-gate.md)",
        f"# generated with: {ruff_version}",
        "# selection: F,B,ASYNC,PLE over tldw_chatbook (F is ZERO by policy",
        "# and intentionally absent here -- the gate-new CI job enforces it)",
        "# format: <RULE> <count>, one per line",
        "# regenerate (with the pinned CI version!): "
        "python Tests/lint/ruff_gate.py --update-baseline",
    ]
    for rule in sorted(counts):
        lines.append(f"{rule} {counts[rule]}")
    BASELINE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the baseline file from the current ruff counts",
    )
    args = parser.parse_args(argv)

    counts = _run_ruff()

    if args.update_baseline:
        _write_baseline(counts)
        print(f"baseline rewritten: {sum(counts.values())} findings across "
              f"{len(counts)} rules -> {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    baseline = _load_baseline()
    rules = sorted(set(baseline) | set(counts))
    exceeded: list[str] = []
    improved: list[str] = []
    for rule in rules:
        current = counts.get(rule, 0)
        allowed = baseline.get(rule, 0)
        if current > allowed:
            exceeded.append(f"  {rule}: {current} > baseline {allowed} (+{current - allowed})")
        elif current < allowed:
            improved.append(f"  {rule}: {current} < baseline {allowed} (-{allowed - current})")

    if exceeded:
        print("RATCHET FAILED -- rules above baseline:")
        print("\n".join(exceeded))
        print("Fix the new findings, or lower the baseline counts in "
              f"{BASELINE_PATH.relative_to(REPO_ROOT)} only as part of a "
              "deliberate debt decision.")
        return 1
    print(f"RATCHET OK -- {sum(counts.values())} findings across {len(counts)} "
          f"rules, all at or under baseline "
          f"({sum(baseline.values())} in baseline).")
    if improved:
        print("improved (please shrink the baseline to lock the gain):")
        print("\n".join(improved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
