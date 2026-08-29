"""TASK-19572: the shape of the one CI check that is meant to be required.

The `Tests` workflow has produced no verdict since 2026-06-26 -- 200-227 minute
runtime against 23-50 merges/day on `dev`, with cancel-in-progress killing every
in-flight run. `derived-artifacts.yml` is the replacement gate: install-free,
~90 s, and safe to mark as a required status check.

These tests pin the properties that make it requireable at all. They are
deliberately shape-only (the workflow cannot be executed here), and every
assertion below corresponds to a way the gate would silently stop gating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "derived-artifacts.yml"
CHECKERS = (
    "tldw_chatbook/css/check_bundle_sync.py",
    "scripts/check_profile_owned_path_inventory.py",
    "scripts/check_persistent_diagnostic_inventory.py",
    "scripts/check_backlog_task_ids.py",
    # TASK-20971. VALID_TABLES['chachanotes'] went stale, was repaired, and
    # went stale again 14.5 hours later; this is its authoring-time half.
    "scripts/check_schema_table_allowlist.py",
    "scripts/check_index_plan_pins.py",
)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _job() -> dict:
    return _workflow()["jobs"]["derived-artifacts"]


def _steps() -> list[dict]:
    return _job()["steps"]


def test_workflow_is_valid_yaml_with_one_job():
    """One job means one required-check name for the owner to enter."""
    assert list(_workflow()["jobs"]) == ["derived-artifacts"]


def test_triggers_are_not_path_filtered():
    """A path-filtered required check never reports and blocks the PR forever.

    GitHub leaves a skipped required check on "Expected - waiting for status to
    be reported", so a docs-only PR would be unmergeable. At ~90 s the job is
    cheap enough to run unconditionally; adding `paths:` is the one edit that
    would brick merges without failing anything.
    """
    triggers = _workflow()[True]  # PyYAML parses the bare `on:` key as True
    assert set(triggers) == {"pull_request", "push"}
    for event, config in triggers.items():
        assert not (config or {}).get("paths"), f"{event} must not be path-filtered"
        assert not (config or {}).get("paths-ignore"), f"{event} must not path-ignore"


def test_every_checker_runs():
    """All four checkers are invoked, so one job covers the whole census."""
    script = "\n".join(step.get("run", "") for step in _steps())
    for checker in CHECKERS:
        assert checker in script, f"{checker} is not run by the required job"


def test_checker_steps_survive_an_earlier_failure():
    """One red checker must not hide the others.

    With the default `success()` condition the first failure skips the rest, so
    a burn-down needs one push per checker. `!cancelled()` reports all of the
    drift in a single run while still failing the job.
    """
    checker_steps = [step for step in _steps() if "python " in step.get("run", "")]
    assert len(checker_steps) == len(CHECKERS)
    for step in checker_steps:
        assert "cancelled()" in str(step.get("if", "")), (
            f"step {step.get('name')!r} would be skipped after an earlier failure"
        )


def test_job_installs_nothing():
    """Install-free is what keeps this at ~90 s; a pip install re-creates the
    runtime that made the Tests workflow unusable."""
    job = _job()
    assert "pip install" not in yaml.safe_dump(job)
    for step in job["steps"]:
        uses = step.get("uses", "")
        assert not uses.startswith("actions/setup-python") or "cache" not in (
            step.get("with") or {}
        ), "no pip cache is needed when nothing is installed"


def test_required_check_name_is_stable():
    """Renaming this silently detaches branch protection from the job."""
    assert _job()["name"] == "Derived artifacts reproduce from their sources"


def test_backlog_guard_delegates_to_the_shared_script():
    """backlog-guard and derived-artifacts must not keep two copies of the
    duplicate-id logic, or the required check and the standalone guard drift."""
    backlog_guard = (
        PROJECT_ROOT / ".github" / "workflows" / "backlog-guard.yml"
    ).read_text(encoding="utf-8")
    assert "scripts/check_backlog_task_ids.py" in backlog_guard
    assert "uniq -d" not in backlog_guard, "inline shell copy was reintroduced"
