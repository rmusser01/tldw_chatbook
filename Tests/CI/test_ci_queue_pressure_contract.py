"""Contracts for TASK-22250's bounded pull-request CI fan-out."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = PROJECT_ROOT / ".github" / "workflows"
SCHEDULE_SKIP = "${{ github.event_name != 'schedule' }}"
ALWAYS_SCHEDULE_SKIP = "${{ always() && github.event_name != 'schedule' }}"
ORDINARY_TEST_JOB_CONDITIONS = {
    "core-tests": SCHEDULE_SKIP,
    "artifact-lease-spike": SCHEDULE_SKIP,
    "artifact-lease-shape": SCHEDULE_SKIP,
    "artifact-lease-gate": ALWAYS_SCHEDULE_SKIP,
    "ui-tests": SCHEDULE_SKIP,
    "textual-minimum": SCHEDULE_SKIP,
    "test-summary": ALWAYS_SCHEDULE_SKIP,
}
STANDALONE_WORKFLOWS = (
    "derived-artifacts.yml",
    "css-bundle-guard.yml",
    "perf-guard.yml",
    "backlog-guard.yml",
)
PUSH_ONLY_CANCELLATION = (
    "${{ github.event_name == 'push' && github.ref != 'refs/heads/main' }}"
)
PULL_REQUEST_TYPES = ["opened", "synchronize", "reopened", "ready_for_review"]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOW_ROOT / name).read_text(encoding="utf-8"))


def test_dev_merge_creates_one_heavy_tests_run() -> None:
    """A dev merge is tested on its PR, not duplicated by the dev push."""
    workflow = _workflow("test.yml")
    triggers = workflow.get("on", workflow.get(True))

    assert triggers["pull_request"]["branches"] == ["dev"]
    assert triggers["pull_request"]["types"] == PULL_REQUEST_TYPES
    assert triggers["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in triggers


def test_schedule_skips_the_ordinary_test_suite() -> None:
    """The nightly event runs its deep matrix without duplicating the PR suite."""
    jobs = _workflow("test.yml")["jobs"]

    for job_name, expected_condition in ORDINARY_TEST_JOB_CONDITIONS.items():
        assert jobs[job_name].get("if") == expected_condition
    assert jobs["nightly-deep"]["if"] == (
        "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    )
    assert jobs["all-tests"]["if"] == "github.event_name == 'workflow_dispatch'"


def test_focused_guards_ignore_the_permanent_main_promotion_pr() -> None:
    """Short guards retain dev pushes without following PR #602 into main."""
    for workflow_name in STANDALONE_WORKFLOWS:
        workflow = _workflow(workflow_name)
        triggers = workflow.get("on", workflow.get(True))

        assert triggers["pull_request"]["branches"] == ["dev"]
        assert triggers["push"]["branches"] == ["dev", "main"]
        assert "pull_request.number" not in (WORKFLOW_ROOT / workflow_name).read_text(
            encoding="utf-8"
        )

    assert "pull_request.number" not in (WORKFLOW_ROOT / "test.yml").read_text(
        encoding="utf-8"
    )


def test_ready_for_review_remains_an_explicit_activity() -> None:
    """Activating a dev-targeting draft must request fresh checks."""
    for workflow_name in ("test.yml", *STANDALONE_WORKFLOWS):
        workflow = _workflow(workflow_name)
        triggers = workflow.get("on", workflow.get(True))
        assert triggers["pull_request"]["types"] == PULL_REQUEST_TYPES


def test_pull_request_shards_leave_capacity_for_required_checks() -> None:
    """One Tests run may occupy at most six shard slots at a time."""
    jobs = _workflow("test.yml")["jobs"]
    assert jobs["core-tests"]["strategy"]["max-parallel"] == 3
    assert jobs["ui-tests"]["strategy"]["max-parallel"] == 3


def test_tests_still_cancels_only_superseded_push_runs() -> None:
    """Queue bounding must not reintroduce pull-request cancellation sweeps."""
    for workflow_name in ("test.yml", *STANDALONE_WORKFLOWS):
        concurrency = _workflow(workflow_name)["concurrency"]
        assert concurrency["cancel-in-progress"] == PUSH_ONLY_CANCELLATION
