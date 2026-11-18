"""Every pull_request workflow must also run correctly from workflow_dispatch.

A maintainer can then run any of them by hand (for example on a fork or bot PR, which the
merge queue skips). The queue itself no longer dispatches: since the 2026-10-06 revision it
approves the held pull_request runs after its rebase (ADR-218 amendment, spec V4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
# The queue's own entry point reacts to PR events and is never dispatched (it is not on main,
# so it cannot be).
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
