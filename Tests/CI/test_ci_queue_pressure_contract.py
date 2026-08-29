"""Contracts for TASK-22250's bounded pull-request CI fan-out."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = PROJECT_ROOT / ".github" / "workflows"
PROMOTION_GUARD = (
    "${{ github.event_name != 'pull_request' || "
    "github.event.pull_request.draft == false || github.head_ref != 'dev' || "
    "github.base_ref != 'main' }}"
)
ALWAYS_PROMOTION_GUARD = (
    "${{ always() && (github.event_name != 'pull_request' || "
    "github.event.pull_request.draft == false || github.head_ref != 'dev' || "
    "github.base_ref != 'main') }}"
)
TEST_JOB_GUARDS = {
    "core-tests": PROMOTION_GUARD,
    "artifact-lease-spike": PROMOTION_GUARD,
    "artifact-lease-shape": PROMOTION_GUARD,
    "artifact-lease-gate": ALWAYS_PROMOTION_GUARD,
    "ui-tests": PROMOTION_GUARD,
    "textual-minimum": PROMOTION_GUARD,
    "test-summary": ALWAYS_PROMOTION_GUARD,
}
STANDALONE_JOB_GUARDS = {
    "derived-artifacts.yml": "derived-artifacts",
    "css-bundle-guard.yml": "css-bundle-reproducible",
    "perf-guard.yml": "ui-latency-guardrails",
    "backlog-guard.yml": "duplicate-task-ids",
}
PUSH_ONLY_CANCELLATION = (
    "${{ github.event_name == 'push' && github.ref != 'refs/heads/main' }}"
)
PULL_REQUEST_TYPES = ["opened", "synchronize", "reopened", "ready_for_review"]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOW_ROOT / name).read_text(encoding="utf-8"))


def test_draft_dev_to_main_promotion_does_not_request_runners() -> None:
    """The permanent draft promotion PR must report skips, not consume slots."""
    test_jobs = _workflow("test.yml")["jobs"]
    for job_name, expected_guard in TEST_JOB_GUARDS.items():
        assert test_jobs[job_name].get("if") == expected_guard

    for workflow_name, job_name in STANDALONE_JOB_GUARDS.items():
        assert _workflow(workflow_name)["jobs"][job_name].get("if") == PROMOTION_GUARD


def test_ready_for_review_retriggers_jobs_skipped_while_draft() -> None:
    """Marking the promotion ready must replace its earlier skipped verdicts."""
    for workflow_name in ("test.yml", *STANDALONE_JOB_GUARDS):
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
    for workflow_name in ("test.yml", *STANDALONE_JOB_GUARDS):
        concurrency = _workflow(workflow_name)["concurrency"]
        assert concurrency["cancel-in-progress"] == PUSH_ONLY_CANCELLATION
