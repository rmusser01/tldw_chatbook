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
_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve").split()

#: (file, pattern whose one group is the stated number, what it counts).
#: "runners" is the shards plus the PR Fast Lane and the Console P0 gate.
_STATEMENTS = (
    (WORKFLOW, r"pr-fast-lane \+ (\d+) UI shards", "shards"),
    (WORKFLOW, r"\+ this = (\d+) test runners", "runners"),
    (WORKFLOW, r"the UI lane from four shards to (\w+)\)", "shards"),
    ("scripts/ui_pr_gate_census.txt", r"round-robin over (\w+) shards", "shards"),
    ("scripts/ui_pr_gate_census.txt", r"TASK-33007 runs (\w+):", "shards"),
    ("scripts/check_ui_pr_gate_census.py", r"so the lane runs (\w+)\.", "shards"),
    ("scripts/ui_lane_shard_replay.py", r"--shards (\d+) run1 run2", "shards"),
    ("scripts/ui_lane_shard_replay.py", r"--shards (\d+) --tail", "shards"),
    (ADR, r"took the UI lane to (\w+) shards", "shards"),
    (ADR, r"makes (\w+) concurrent test\s+runners per PR", "runners"),
    (ADR, r"TASK-34353; (\w+) since\s+TASK-33007", "shards"),
    (ADR, r"now runs (\w+) test runners at once", "runners"),
)


def _shard_count() -> int:
    workflow = yaml.safe_load((REPO_ROOT / WORKFLOW).read_text(encoding="utf-8"))
    return len(workflow["jobs"]["ui-fast-lane"]["strategy"]["matrix"]["shard"])


@pytest.mark.parametrize(("path", "pattern", "counts"), _STATEMENTS)
def test_a_stated_shard_count_is_the_matrix_size(path, pattern, counts):
    number = _shard_count() + (2 if counts == "runners" else 0)
    stated = re.findall(pattern, (REPO_ROOT / path).read_text(encoding="utf-8"))

    assert len(stated) == 1, f"{path}: expected one match of {pattern!r}, got {stated}"
    assert stated[0] in (str(number), _WORDS[number]), (
        f"{path} says {stated[0]!r} where the ui-fast-lane matrix makes it "
        f"{number} ({counts})"
    )
