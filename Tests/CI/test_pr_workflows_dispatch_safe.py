"""Every pull_request workflow must also run correctly from workflow_dispatch.

The merge queue re-runs, on the rebased head, every workflow that ran on the old head.
A rebase made with GITHUB_TOKEN gets only approval-pending pull_request runs (spec F4),
so dispatch is the only way they run again (spec section 5.4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
# The queue's own entry point reacts to PR events but is never re-dispatched (it is not on
# main, so it cannot be dispatched, and scripts/merge_queue.py excludes it by design).
QUEUE = "merge-queue.yml"


def _pr_workflows():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        if path.name == QUEUE:
            continue
        wf = yaml.safe_load(path.read_text(encoding="utf-8"))
        on = wf.get(True, wf.get("on"))
        if isinstance(on, dict) and "pull_request" in on:
            yield path, wf, on


def test_there_are_pull_request_workflows():
    assert len(list(_pr_workflows())) >= 10


def test_every_pull_request_workflow_is_dispatchable():
    missing = [p.name for p, wf, on in _pr_workflows() if "workflow_dispatch" not in on]
    assert missing == []


def test_pr_context_job_guards_also_admit_dispatch():
    bad = []
    for path, wf, on in _pr_workflows():
        for job_id, job in wf["jobs"].items():
            cond = str(job.get("if", ""))
            if ("github.event.pull_request" in cond or "github.event.label" in cond) and "workflow_dispatch" not in cond:
                bad.append(f"{path.name}:{job_id}")
    assert bad == []


def test_pr_head_checkouts_fall_back_to_github_sha():
    bad = []
    for path, wf, on in _pr_workflows():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "github.event.pull_request.head.sha" in line and "github.sha" not in line:
                bad.append(f"{path.name}: {line.strip()}")
    assert bad == []
