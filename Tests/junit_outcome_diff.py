#!/usr/bin/env python3
"""Diff two pytest junit XML reports by per-test outcome.

Usage:
    python Tests/junit_outcome_diff.py baseline.xml candidate.xml [--allow-missing PATTERN ...]

Compares the (classname, name) -> outcome mapping of two junit files produced by
``pytest --junitxml=...`` and reports:

    REGRESSED   passed in baseline, fails/errors in candidate
    VANISHED    present in baseline, absent from candidate (deleted or no longer collected)
    RECOVERED   failed/errored in baseline, passes in candidate
    NEW         absent from baseline, present in candidate
    NOW-SKIPPED passed in baseline, skipped in candidate

Exit status is non-zero when there are REGRESSED, NOW-SKIPPED, or unexplained
VANISHED entries, so CI and PR verification can gate on it. Intentional
deletions are declared with ``--allow-missing`` (a substring match against the
test id); anything vanished that matches no pattern counts as a failure.

junit is used (rather than pytest's terminal output) because it records every
test's outcome individually; counts alone cannot distinguish "one test fixed,
one broken" from "no change" (see backlog/docs/lessons-testing-evidence.md —
compare failure *sets*, never counts).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def load_outcomes(path: Path) -> dict[str, str]:
    """Map ``classname::name`` -> outcome (pass | fail | error | skip)."""
    outcomes: dict[str, str] = {}
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        key = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("error") is not None:
            outcome = "error"
        elif case.find("failure") is not None:
            outcome = "fail"
        elif case.find("skipped") is not None:
            outcome = "skip"
        else:
            outcome = "pass"
        # Reruns/duplicates: keep the worst outcome so a flaky pass cannot mask a failure.
        rank = {"error": 3, "fail": 2, "skip": 1, "pass": 0}
        if key not in outcomes or rank[outcome] > rank[outcomes[key]]:
            outcomes[key] = outcome
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--allow-missing",
        action="append",
        default=[],
        metavar="PATTERN",
        help="substring of test ids whose disappearance is intentional (repeatable)",
    )
    parser.add_argument(
        "--max-print", type=int, default=50, help="cap printed ids per category"
    )
    args = parser.parse_args(argv)

    base = load_outcomes(args.baseline)
    cand = load_outcomes(args.candidate)

    regressed = sorted(
        k for k, v in cand.items() if v in ("fail", "error") and base.get(k) == "pass"
    )
    now_skipped = sorted(
        k for k, v in cand.items() if v == "skip" and base.get(k) == "pass"
    )
    recovered = sorted(
        k for k, v in cand.items() if v == "pass" and base.get(k) in ("fail", "error")
    )
    vanished = sorted(k for k in base if k not in cand)
    allowed_vanished = [
        k for k in vanished if any(p in k for p in args.allow_missing)
    ]
    unexplained_vanished = [k for k in vanished if k not in set(allowed_vanished)]
    new = sorted(k for k in cand if k not in base)

    def emit(label: str, ids: list[str]) -> None:
        print(f"{label}: {len(ids)}")
        for k in ids[: args.max_print]:
            print(f"  {k}")
        if len(ids) > args.max_print:
            print(f"  ... and {len(ids) - args.max_print} more")

    print(f"baseline:  {len(base)} tests ({args.baseline})")
    print(f"candidate: {len(cand)} tests ({args.candidate})")
    emit("REGRESSED (pass -> fail/error)", regressed)
    emit("NOW-SKIPPED (pass -> skip)", now_skipped)
    emit("VANISHED unexplained", unexplained_vanished)
    emit("VANISHED allowed", allowed_vanished)
    emit("RECOVERED (fail/error -> pass)", recovered)
    emit("NEW", new)

    return 1 if (regressed or now_skipped or unexplained_vanished) else 0


if __name__ == "__main__":
    sys.exit(main())
