"""Every statement of the UI Fast Lane's shard count agrees with its matrix.

The count runs in one place: the ``shard`` matrix of ``ui-fast-lane`` in
``.github/workflows/derived-artifacts.yml``. Comments and documents repeat it
in words or numerals, and the review of #3057 found them pinned by nothing (TASK-33007
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
#: "eligible" counts all shards plus the other required test lanes.
#: "runners" applies the UI matrix's concurrency ceiling to that count;
#: "runners+perf" adds perf-guard's jobs, which a PR touching code also starts.
_STATEMENTS = (
    (WORKFLOW, r"Current UI lane: (\d+) shards", "shards"),
    ("scripts/ui_pr_gate_census.txt", r"round-robin over (\w+) shards", "shards"),
    ("scripts/ui_pr_gate_census.txt", r"Current UI lane: (\d+) shards", "shards"),
    ("scripts/check_ui_pr_gate_census.py", r"Current UI lane: (\d+) shards", "shards"),
    ("scripts/ui_lane_shard_replay.py", r"--shards (\d+) run1 run2", "shards"),
    ("scripts/ui_lane_shard_replay.py", r"--shards (\d+) --tail", "shards"),
    (ADR, r"Current\s+UI lane: (\d+) shards", "shards"),
    (ADR, r"current total is (\d+) eligible test jobs per PR", "eligible"),
    (ADR, r"UI lane currently runs (\d+) shards", "shards"),
    (ADR, r"now requests (\d+) test runners in parallel", "runners"),
    (ADR, r"test runners in parallel, (\d+) with perf-guard", "runners+perf"),
)


def _counts() -> dict[str, int]:
    """The UI lane's shards, the test runners an ordinary PR starts, and
    those runners with perf-guard's.

    Eligible jobs are all shards plus every other lane the required check
    ``needs``. Peak runners apply the UI matrix's max-parallel ceiling first.
    The PR Fast Lane's three jobs, the Console P0 gate, the Dreams
    tracking tests and the Dreams and Guardian boundary contracts since
    TASK-34916, TASK-35575, TASK-34929 and TASK-34986 each use one runner:
    a second matrix job among them fails here instead of miscounting.
    """
    jobs = _jobs(WORKFLOW)
    needs = jobs["derived-artifacts"]["needs"]
    assert [job for job in needs if "strategy" in jobs[job]] == ["ui-fast-lane"]
    shards = len(jobs["ui-fast-lane"]["strategy"]["matrix"]["shard"])
    eligible = shards + len(needs) - 1
    limit = jobs["ui-fast-lane"]["strategy"].get("max-parallel", shards)
    runners = min(shards, limit) + len(needs) - 1
    perf = _jobs(".github/workflows/perf-guard.yml")
    assert not [job for job in perf.values() if "strategy" in job]
    return {
        "shards": shards,
        "eligible": eligible,
        "runners": runners,
        "runners+perf": runners + len(perf),
    }


def _jobs(workflow: str) -> dict:
    return yaml.safe_load((REPO_ROOT / workflow).read_text(encoding="utf-8"))["jobs"]


@pytest.mark.parametrize(
    ("shards", "limit", "eligible", "parallel"),
    [
        (28, 17, 34, 23),
        (28, 4, 34, 10),
        (17, None, 23, 23),
    ],
)
def test_counts_distinguish_eligible_jobs_from_parallel_runners(
    monkeypatch, shards, limit, eligible, parallel
):
    needs = ["ui-fast-lane", *[f"lane-{index}" for index in range(6)]]
    jobs = {name: {} for name in needs}
    strategy = {"matrix": {"shard": list(range(1, shards + 1))}}
    if limit is not None:
        strategy["max-parallel"] = limit
    jobs["ui-fast-lane"] = {"strategy": strategy}
    jobs["derived-artifacts"] = {"needs": needs}
    monkeypatch.setitem(
        globals(), "_jobs", lambda name: jobs if name == WORKFLOW else {"perf": {}}
    )
    counts = _counts()
    assert counts["runners"] == parallel
    assert counts["runners+perf"] == parallel + 1
    assert counts["eligible"] == eligible


@pytest.mark.parametrize(("path", "pattern", "counts"), _STATEMENTS)
def test_a_stated_shard_count_is_the_matrix_size(path, pattern, counts):
    number = _counts()[counts]
    stated = re.findall(pattern, (REPO_ROOT / path).read_text(encoding="utf-8"))

    assert len(stated) == 1, f"{path}: expected one match of {pattern!r}, got {stated}"
    assert stated[0] == str(number) or (
        number < len(_WORDS) and stated[0] == _WORDS[number]
    ), (
        f"{path} says {stated[0]!r} where the ui-fast-lane matrix makes it "
        f"{number} ({counts})"
    )
