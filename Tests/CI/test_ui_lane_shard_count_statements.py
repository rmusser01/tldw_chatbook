"""Every statement of the UI Fast Lane's shard count agrees with its matrix.

The count runs in one place: the ``shard`` matrix of ``ui-fast-lane`` in
``.github/workflows/ci-qualification.yml``. Comments and documents repeat it
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
WORKFLOW = ".github/workflows/ci-qualification.yml"
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
    (ADR, r"Background qualifications request (\d+) application runners", "background"),
    (ADR, r"combined application ceiling of (\d+)", "combined"),
)


def _counts() -> dict[str, int]:
    """The UI lane's shards, the test runners an ordinary PR starts, and
    those runners with perf-guard's.

    Eligible application jobs exclude the serialized preflight. Peak runners
    apply the UI matrix's max-parallel ceiling after preflight succeeds.
    The PR Fast Lane's three jobs, the Console P0 gate, the Dreams
    tracking tests and the Dreams and Guardian boundary contracts since
    TASK-34916, TASK-35575, TASK-34929 and TASK-34986 each use one runner:
    a second matrix job among them fails here instead of miscounting.
    """
    jobs = _jobs(WORKFLOW)
    assert [name for name, job in jobs.items() if "strategy" in job] == ["ui-fast-lane"]
    strategy = jobs["ui-fast-lane"]["strategy"]
    shards = len(strategy["matrix"]["shard"])
    eligible = shards + len(jobs) - 1
    limits = re.fullmatch(
        r"\$\{\{ inputs.queue_front && (\d+) \|\| (\d+) \}\}", strategy["max-parallel"]
    )
    assert limits, "Expected explicit front/background matrix limits"
    front, background = map(int, limits.groups())
    runners = min(shards, front) + len(jobs) - 1
    background_runners = min(shards, background) + len(jobs) - 1
    perf = _jobs(".github/workflows/perf-guard.yml")
    assert not [job for job in perf.values() if "strategy" in job]
    return {
        "shards": shards,
        "eligible": eligible,
        "runners": runners,
        "background": background_runners,
        "combined": runners + background_runners,
        "runners+perf": runners + len(perf),
    }


def _jobs(workflow: str) -> dict:
    return yaml.safe_load((REPO_ROOT / workflow).read_text(encoding="utf-8"))["jobs"]


@pytest.mark.parametrize(
    ("shards", "front", "background", "eligible", "parallel", "combined"),
    [(28, 17, 8, 34, 23, 37), (28, 4, 2, 34, 10, 18), (10, 17, 8, 16, 16, 30)],
)
def test_counts_distinguish_eligible_jobs_from_parallel_runners(
    monkeypatch, shards, front, background, eligible, parallel, combined
):
    jobs = {name: {} for name in ["ui-fast-lane", *[f"lane-{i}" for i in range(6)]]}
    jobs["ui-fast-lane"] = {
        "strategy": {
            "matrix": {"shard": list(range(1, shards + 1))},
            "max-parallel": f"${{{{ inputs.queue_front && {front} || {background} }}}}",
        }
    }
    monkeypatch.setitem(
        globals(), "_jobs", lambda name: jobs if name == WORKFLOW else {"perf": {}}
    )
    counts = _counts()
    assert counts["runners"] == parallel
    assert counts["runners+perf"] == parallel + 1
    assert counts["eligible"] == eligible
    assert counts["combined"] == combined


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
