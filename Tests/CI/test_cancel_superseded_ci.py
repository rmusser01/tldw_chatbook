"""Execute the actual checkout-free cleanup script against a recording API."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/merge-queue.yml"
REPOSITORY = "owner/project"
CURRENT = "a" * 40
OLD = "b" * 40


def _job():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "cancel-superseded"
    ]


def _pr():
    return {
        "number": 42,
        "state": "open",
        "user": {"type": "User"},
        "head": {"sha": CURRENT, "ref": "feature", "repo": {"full_name": REPOSITORY}},
        "base": {"ref": "dev", "repo": {"full_name": REPOSITORY}},
    }


def _run(**changes):
    return {
        "id": 10,
        "event": "pull_request",
        "path": ".github/workflows/derived-artifacts.yml",
        "head_branch": "feature",
        "head_sha": OLD,
        "head_repository": {"full_name": REPOSITORY},
        "pull_requests": [{"number": 42}],
        "status": "queued",
        **changes,
    }


HARNESS = r"""
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const calls = [], cancelled = [];
let reads = 0;
const github = {
  rest: {
    pulls: {get: async args => {
      calls.push(['read', args]);
      if (input.readError && reads === input.readError.at) {
        throw Object.assign(new Error('read failed'), {status: input.readError.status});
      }
      return {data: input.prs[Math.min(reads++, input.prs.length - 1)]};
    }},
    actions: {
      listWorkflowRuns: 'listWorkflowRuns',
      listJobsForWorkflowRun: 'listJobsForWorkflowRun',
      cancelWorkflowRun: async args => {
        calls.push(['cancel', args]);
        if (input.cancelError) {
          throw Object.assign(new Error('cancel failed'), {status: input.cancelError});
        }
        cancelled.push(args.run_id);
        return {status: 202};
      }
    }
  },
  paginate: async (method, args) => {
    calls.push(['paginate', method, args]);
    if (method === 'listJobsForWorkflowRun') {
      if (input.jobsError) throw new Error('jobs failed');
      return input.jobs?.[args.run_id] ?? [];
    }
    return input.runs;
  }
};
const core = {info: () => {}, warning: () => {}, notice: () => {}};
const context = {repo: {owner: 'owner', repo: 'project'},
  payload: {pull_request: input.event}};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  let error = null;
  try { await new AsyncFunction('github', 'context', 'core', input.script)(github, context, core); }
  catch (e) { error = e.message; }
  process.stdout.write(JSON.stringify({calls, cancelled, error}));
})();
"""


def _execute(runs, *, prs=None, event=None, **errors):
    node = shutil.which("node")
    assert node, "Node is required to execute the real GitHub Actions cleanup script"
    script = _job()["steps"][0]["with"]["script"]
    result = subprocess.run(
        [node, "-e", HARNESS],
        input=json.dumps(
            {
                "script": script,
                "runs": runs,
                "prs": prs or [_pr()],
                "event": event or _pr(),
                **errors,
            }
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=True,
    )
    return json.loads(result.stdout)


def test_cleanup_is_independent_and_has_no_checkout_or_queue_mutation():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = _job()
    assert job["permissions"] == {"actions": "write", "pull-requests": "read"}
    assert job["timeout-minutes"] == 5
    assert "needs" not in job and "concurrency" not in job
    assert "concurrency" not in workflow
    assert len(job["steps"]) == 1
    assert job["steps"][0]["uses"] == "actions/github-script@v7"
    assert "github.event.action == 'synchronize'" in job["if"]
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in job["if"]
    )
    assert "github.event.pull_request.user.type != 'Bot'" in job["if"]
    assert "MERGE_QUEUE" not in job["if"]
    assert "github.event.action != 'synchronize'" in workflow["jobs"]["queue"]["if"]
    assert "${{" not in job["steps"][0]["with"]["script"]


@pytest.mark.parametrize(
    "status", ["queued", "pending", "in_progress", "waiting", "requested"]
)
def test_only_obsolete_head_is_cancelled_and_runs_are_paginated(status):
    result = _execute([_run(status=status), _run(id=11, head_sha=CURRENT)])
    assert result["error"] is None
    assert result["cancelled"] == [10]
    assert result["calls"][1] == [
        "paginate",
        "listWorkflowRuns",
        {
            "owner": "owner",
            "repo": "project",
            "workflow_id": "derived-artifacts.yml",
            "branch": "feature",
            "event": "pull_request",
            "per_page": 100,
        },
    ]
    assert result["calls"][2] == [
        "paginate",
        "listJobsForWorkflowRun",
        {
            "owner": "owner",
            "repo": "project",
            "run_id": 10,
            "filter": "latest",
            "per_page": 100,
        },
    ]
    assert [call[0] for call in result["calls"]] == [
        "read",
        "paginate",
        "paginate",
        "read",
        "cancel",
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "completed"},
        {"status": "unknown"},
        {"head_sha": CURRENT},
        {"head_sha": ""},
        {"head_sha": None},
        {"head_branch": "other"},
        {"head_repository": {"full_name": "outsider/project"}},
        {"head_repository": None},
        {"pull_requests": []},
        {"pull_requests": [{"number": 43}]},
        {"pull_requests": [{"number": 42}, {"number": 43}]},
        {"id": None},
        {"id": -1},
        {"head_sha": "malformed"},
        {"event": "workflow_dispatch"},
        {"event": "push"},
        {"path": ".github/workflows/perf-guard.yml"},
    ],
)
def test_unrelated_current_or_ambiguous_runs_are_preserved(changes):
    result = _execute([_run(**changes)])
    assert result["error"] is None
    assert result["cancelled"] == []


@pytest.mark.parametrize("change", ["head", "closed", "fork", "base", "bot", "number"])
@pytest.mark.parametrize("during_cleanup", [False, True])
def test_stale_event_or_changed_live_pr_stops_cleanup(change, during_cleanup):
    changed = copy.deepcopy(_pr())
    if change == "head":
        changed["head"]["sha"] = OLD
    elif change == "closed":
        changed["state"] = "closed"
    elif change == "fork":
        changed["head"]["repo"]["full_name"] = "outsider/project"
    elif change == "base":
        changed["base"]["ref"] = "main"
    elif change == "bot":
        changed["user"]["type"] = "Bot"
    else:
        changed["number"] = 43
    result = _execute([_run()], prs=[_pr(), changed] if during_cleanup else [changed])
    assert result["error"] is None
    assert result["cancelled"] == []


def test_head_change_between_two_cancellations_stops_the_second():
    changed = _pr()
    changed["head"]["sha"] = OLD
    result = _execute([_run(), _run(id=11)], prs=[_pr(), _pr(), changed])
    assert result["cancelled"] == [10]


def test_same_head_evidence_survives_base_movement():
    moved_base = _pr()
    moved_base["base"]["sha"] = "c" * 40
    result = _execute([_run(head_sha=CURRENT)], prs=[moved_base])
    assert result["error"] is None
    assert result["cancelled"] == []


@pytest.mark.parametrize(
    "name", ["Derived artifacts reproduce from their sources", "Merge queue tick"]
)
@pytest.mark.parametrize("status", ["queued", "in_progress", "completed"])
def test_preserves_the_run_that_may_be_approving_replacement_ci(name, status):
    result = _execute(
        [_run(status="in_progress")], jobs={"10": [{"name": name, "status": status}]}
    )
    assert result["error"] is None
    assert result["cancelled"] == []


def test_job_inventory_error_never_cancels_a_possible_queue_controller():
    result = _execute([_run()], jobsError=503)
    assert result["error"] == "jobs failed"
    assert result["cancelled"] == []


def test_empty_branch_never_widens_the_run_query():
    missing_branch = _pr()
    missing_branch["head"]["ref"] = ""
    result = _execute(
        [_run(head_branch="")], prs=[missing_branch], event=missing_branch
    )
    assert result["cancelled"] == []
    assert all(call[0] != "paginate" for call in result["calls"])


def test_concurrent_completion_is_harmless_but_other_errors_are_visible():
    assert _execute([_run()], cancelError=409)["error"] is None
    assert _execute([_run()], cancelError=403)["error"] == "cancel failed"
    result = _execute([_run()], readError={"at": 1, "status": 503})
    assert result["error"] == "read failed"
    assert result["cancelled"] == []
