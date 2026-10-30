"""Run the real CI cost report against recorded Actions API responses."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github/workflows/derived-artifacts.yml"
)
REPORT = "Report qualification cost"
HARNESS = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const calls = [], warnings = [], rows = [];
const summary = {addHeading(s) {rows.push(s); return this;},
  addRaw(s) {rows.push(s); return this;},
  addTable(s) {rows.push(s); return this;},
  async write() { if (input.summaryError) throw new Error('summary unavailable'); }};
const all = Symbol('all'), attempt = Symbol('attempt');
const github = {rest: {actions: {listJobsForWorkflowRun: all,
  listJobsForWorkflowRunAttempt: attempt}},
  async paginate(endpoint, args) {
    calls.push({endpoint: endpoint === all ? 'all' : 'attempt', ...args});
    if (input.apiError) throw new Error('API unavailable');
    return endpoint === all ? input.all : input.current;
  }};
const core = {summary, warning(s) {warnings.push(s);}};
const context = {repo: {owner: 'owner', repo: 'repo'}, runId: input.runId || 123};
process.env.GITHUB_RUN_ATTEMPT = input.attempt || '1';
(async () => {
  let error = null;
  try {await new (Object.getPrototypeOf(async function(){}).constructor)(
    'github', 'context', 'core', input.script)(github, context, core);}
  catch(e) {error = e.message;}
  process.stdout.write(JSON.stringify({calls, warnings, rows, error}));
})();
"""


def _job():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "derived-artifacts"
    ]


def _report():
    return next(step for step in _job()["steps"] if step.get("name") == REPORT)


def _run(current, all_jobs=None, **extra):
    node = shutil.which("node")
    assert node, "Node executes the actual Actions report"
    result = subprocess.run(
        [node, "-e", HARNESS],
        input=json.dumps(
            {
                "script": _report()["with"]["script"],
                "current": current,
                "all": current if all_jobs is None else all_jobs,
                **extra,
            }
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=True,
    )
    return json.loads(result.stdout)


def _record(job_id=1, **changes):
    return {
        "id": job_id,
        "run_id": 123,
        "name": "PR Fast Lane",
        "status": "completed",
        "conclusion": "success",
        "created_at": "2026-10-29T00:00:00Z",
        "started_at": "2026-10-29T00:10:00Z",
        "completed_at": "2026-10-29T00:40:00Z",
        **changes,
    }


def _text(result):
    return json.dumps(result["rows"])


def _current_row(result):
    table = next(row for row in result["rows"] if isinstance(row, list))
    return table[1]


def test_reporting_is_bounded_and_only_diagnostic_failure_is_tolerated():
    report = _report()
    assert _job()["steps"][-1] == report
    assert report["uses"] == "actions/github-script@v7"
    assert report["if"] == "${{ always() }}"
    assert report["continue-on-error"] is True
    assert report["timeout-minutes"] == 1
    assert _job()["permissions"] == {"actions": "read"}
    assert not _job().get("continue-on-error", False)
    assert all(
        not step.get("continue-on-error", False) for step in _job()["steps"][:-1]
    )
    assert "${{" not in report["with"]["script"]


def test_retry_cost_is_separate_and_repeated_job_ids_are_counted_once():
    old, new = _record(), _record(2)
    result = _run([new, new], [old, new, old], attempt="2")
    assert result["error"] is None
    text = _text(result)
    assert "Attempt 2" in text and "30.0" in text and "60.0" in text
    assert "10.0" in text
    assert "not billed minutes" in text
    assert "aggregate and queue tick" in text
    assert result["calls"] == [
        {
            "endpoint": "attempt",
            "owner": "owner",
            "repo": "repo",
            "run_id": 123,
            "attempt_number": 2,
            "per_page": 100,
        },
        {
            "endpoint": "all",
            "owner": "owner",
            "repo": "repo",
            "run_id": 123,
            "filter": "all",
            "per_page": 100,
        },
    ]


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_completed_unsuccessful_jobs_still_consume_runner_time(conclusion):
    assert "30.0" in _text(_run([_record(conclusion=conclusion)]))


def test_skip_control_jobs_and_reject_cross_run_or_invalid_metadata():
    result = _run(
        [
            _record(),
            _record(2, conclusion="skipped"),
            _record(3, name="Derived artifacts reproduce from their sources"),
            _record(4, name="Merge queue tick"),
            _record(5, run_id=999),
            _record("bad"),
        ]
    )
    text = _text(result)
    assert "30.0" in text and "60.0" not in text
    assert _current_row(result) == ["Attempt 1", "30.0", "1", "10.0", "10.0", "2", "2"]


@pytest.mark.parametrize(
    "changes,expected",
    [
        (
            {"started_at": None},
            ["Unavailable", "0", "Unavailable", "Unavailable", "1", "1"],
        ),
        ({"completed_at": "invalid"}, ["Unavailable", "0", "10.0", "10.0", "1", "0"]),
        (
            {"completed_at": "2026-10-28T00:00:00Z"},
            ["Unavailable", "0", "10.0", "10.0", "1", "0"],
        ),
        ({"created_at": None}, ["30.0", "1", "Unavailable", "Unavailable", "0", "1"]),
        (
            {"created_at": "2026-10-29T00:20:00Z"},
            ["30.0", "1", "Unavailable", "Unavailable", "0", "1"],
        ),
        (
            {"status": "in_progress"},
            ["Unavailable", "0", "Unavailable", "Unavailable", "1", "1"],
        ),
    ],
)
def test_missing_or_inverted_times_are_disclosed(changes, expected):
    assert _current_row(_run([_record(**changes)]))[1:] == expected


def test_wait_median_and_max_use_all_measured_jobs():
    result = _run([_record(), _record(2, created_at="2026-10-28T23:40:00Z")])
    assert _current_row(result)[1:5] == ["60.0", "2", "20.0", "30.0"]


def test_empty_sample_is_unavailable():
    text = _text(_run([]))
    assert "Unavailable" in text
    assert "0.0" not in text


@pytest.mark.parametrize("minutes,warns", [(500, False), (501, True)])
def test_budget_warns_only_above_500_cumulative_minutes(minutes, warns):
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 10, 29, tzinfo=timezone.utc)
    old = _record(
        started_at=start.isoformat(),
        completed_at=(start + timedelta(minutes=minutes)).isoformat(),
    )
    result = _run([], [old], attempt="2")
    assert any("budget" in warning for warning in result["warnings"]) is warns
    assert result["error"] is None


@pytest.mark.parametrize("extra", [{"apiError": True}, {"summaryError": True}])
def test_reporting_errors_cannot_change_the_qualification_verdict(extra):
    result = _run([_record()], **extra)
    assert result["warnings"] or result["error"]
    assert _report()["continue-on-error"] is True


def test_invalid_attempt_does_not_read_unrelated_api_data():
    result = _run([_record()], attempt="bad")
    assert result["calls"] == []
    assert result["warnings"] or result["error"]
