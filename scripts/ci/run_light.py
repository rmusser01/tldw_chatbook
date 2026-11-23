"""Run the light tier's selected tests: one pytest process per file (TASK-42521).

bootstrap_profile and sandboxed suites must not share a process (TASK-32873),
so each file gets its own. Every failure is reported before the job fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

NO_TESTS_COLLECTED = 5


def plan(targets: object) -> list[list[str]]:
    """Group `Tests/CI` plus the targets by file; a whole file wins over its nodes."""
    if (
        not isinstance(targets, list)
        or not targets
        or not all(
            isinstance(t, str) and t.startswith("Tests/") and ".." not in t.split("/")
            for t in targets
        )
    ):
        raise SystemExit("Refusing missing or invalid test selection")
    files: dict[str, list[str]] = {}
    for target in ["Tests/CI", *targets]:
        files.setdefault(target.split("::", 1)[0], []).append(target)
    # pytest narrows `f f::n` to `f::n`, so a whole file must run alone.
    return [[path] if path in nodes else nodes for path, nodes in files.items()]


def main(run=subprocess.run) -> int:
    failed = []
    for nodes in plan(json.loads(os.environ["CI_TEST_TARGETS"])):
        # Same flags as the PR lanes: plugin autoload on, 180 s per test.
        code = run(
            [sys.executable, "-m", "pytest", *nodes, "--timeout=180", "--tb=short"]
        ).returncode
        if code == NO_TESTS_COLLECTED:
            print(f"::warning::{nodes[0]} collected no tests", flush=True)
        elif code:
            failed.append(nodes[0].split("::", 1)[0])
    if failed:
        print("Failed: " + ", ".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
