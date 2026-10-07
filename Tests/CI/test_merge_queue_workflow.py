"""Shape of the merge-queue entry point (spec sections 3, 5 and 9)."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "merge-queue.yml"


def _wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_triggers_are_dev_events_only():
    on = _wf()[True]
    assert set(on) == {"pull_request", "push", "workflow_dispatch"}
    assert on["pull_request"] == {"types": ["auto_merge_enabled", "auto_merge_disabled", "closed"], "branches": ["dev"]}
    assert on["push"] == {"branches": ["dev"]}
    # The queue wakes itself to re-run a failed run once it completes.
    assert set(on["workflow_dispatch"]["inputs"]) == {"wait_run"}


def test_never_reads_main():
    text = WORKFLOW.read_text(encoding="utf-8")
    for trigger in ("pull_request_target", "schedule", "workflow_run", "check_suite", "check_run"):
        assert f"{trigger}:" not in text, trigger


def test_write_permissions_are_job_level_only():
    wf = _wf()
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"]["queue"]
    assert job["permissions"] == {"contents": "write", "pull-requests": "write", "actions": "write"}


def test_runs_devs_script_only_when_enabled_and_never_for_forks():
    job = _wf()["jobs"]["queue"]
    assert "vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on'" in job["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in job["if"]
    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    assert job["steps"][0] == {"uses": "actions/checkout@v4", "with": {"ref": "dev", "persist-credentials": False}}
    assert job["steps"][1]["run"] == "python3 scripts/merge_queue.py"
    assert job["steps"][1]["env"] == {"GH_TOKEN": "${{ github.token }}", "MERGE_QUEUE": "${{ vars.MERGE_QUEUE }}",
                                      "WAIT_RUN": "${{ inputs.wait_run }}"}
    assert "concurrency" not in _wf() and "concurrency" not in job, "races are made safe in the script (spec 7)"
