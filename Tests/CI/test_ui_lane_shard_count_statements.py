"""Every statement of the UI Fast Lane's shard count agrees with its matrix.

The count runs in one place: the ``shard`` matrix of ``ui-fast-lane`` in
``.github/workflows/derived-artifacts.yml``. Comments and documents repeat it
in words, and the review of #3057 found them pinned by nothing (TASK-33007
took the lane from four shards to seven while the replay's usage said eight).
This fails when the matrix changes and one of them is left behind.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ".github/workflows/derived-artifacts.yml"
ADR = "backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md"
_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve"
    " thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()

#: (file, pattern whose one group is the stated number, what it counts).
#: "runners" is the shards plus the other test lanes the required check needs;
#: "runners+perf" adds perf-guard's jobs, which a PR touching code also starts.
_STATEMENTS = (
    (WORKFLOW, r"TASK-35834 runs (\w+) shards", "shards"),
    ("scripts/ui_pr_gate_census.txt", r"round-robin over (\w+) shards", "shards"),
    ("scripts/ui_pr_gate_census.txt", r"TASK-35834 runs (\w+):", "shards"),
    (
        "scripts/check_ui_pr_gate_census.py",
        r"TASK-35834: lane now runs (\w+)\.",
        "shards",
    ),
    ("scripts/ui_lane_shard_replay.py", r"--shards (\d+) run1 run2", "shards"),
    ("scripts/ui_lane_shard_replay.py", r"--shards (\d+) --tail", "shards"),
    (ADR, r"TASK-35834: (\w+) UI\s+shards", "shards"),
    (ADR, r"current total is (\w+) concurrent test runners per PR", "runners"),
    (ADR, r"TASK-34353; (\w+) since\s+TASK-35834", "shards"),
    (ADR, r"now runs (\w+) test runners at once", "runners"),
    (ADR, r"(\w+) with perf-guard", "runners+perf"),
)


def _counts() -> dict[str, int]:
    """The UI lane's shards, the test runners an ordinary PR starts, and
    those runners with perf-guard's.

    The runners are the shards plus every other lane the required check
    ``needs`` (the PR Fast Lane's three jobs, the Console P0 gate, the Dreams
    tracking tests and the Dreams and Guardian boundary contracts since
    TASK-34916, TASK-35575, TASK-34929 and TASK-34986), each one runner:
    a second matrix job among them fails here instead of miscounting.
    """
    jobs = _jobs(WORKFLOW)
    needs = jobs["derived-artifacts"]["needs"]
    assert [job for job in needs if "strategy" in jobs[job]] == ["ui-fast-lane"]
    shards = len(jobs["ui-fast-lane"]["strategy"]["matrix"]["shard"])
    runners = shards + len(needs) - 1
    perf = _jobs(".github/workflows/perf-guard.yml")
    assert not [job for job in perf.values() if "strategy" in job]
    return {"shards": shards, "runners": runners, "runners+perf": runners + len(perf)}


def _jobs(workflow: str) -> dict:
    return yaml.safe_load((REPO_ROOT / workflow).read_text(encoding="utf-8"))["jobs"]


@pytest.mark.parametrize(("path", "pattern", "counts"), _STATEMENTS)
def test_a_stated_shard_count_is_the_matrix_size(path, pattern, counts):
    number = _counts()[counts]
    stated = re.findall(pattern, (REPO_ROOT / path).read_text(encoding="utf-8"))

    assert len(stated) == 1, f"{path}: expected one match of {pattern!r}, got {stated}"
    assert stated[0] in (str(number), _WORDS[number]), (
        f"{path} says {stated[0]!r} where the ui-fast-lane matrix makes it "
        f"{number} ({counts})"
    )
