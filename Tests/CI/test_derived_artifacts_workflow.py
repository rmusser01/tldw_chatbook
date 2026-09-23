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
    "scripts/check_canvas_mermaid_assets.py",
    "scripts/check_profile_owned_path_inventory.py",
    "scripts/check_persistent_diagnostic_inventory.py",
    "scripts/check_backlog_task_ids.py",
    # TASK-20971. VALID_TABLES['chachanotes'] went stale, was repaired, and
    # went stale again 14.5 hours later; this is its authoring-time half.
    "scripts/check_schema_table_allowlist.py",
    # TASK-21593. Every shipped database index must have an explicit query-plan
    # decision instead of being assumed useful because it exists.
    "scripts/check_index_plan_pins.py",
    # TASK-32800.4. Three of the four P0s in the 2026-09-17 core-runtime review
    # were one defect class with no guard: a synchronous callable handed to
    # run_worker, and a query_one resuming into a removed subtree.
    "scripts/check_textual_worker_contract.py",
    # TASK-32803.1 / ADR-173. Timestamp writers must go through the shared UTC
    # helper: datetime.utcnow() is forbidden and naive datetime.now().isoformat()
    # is ratcheted.
    "scripts/check_timestamp_writers.py",
    # TASK-32908. Tests/UI gated nothing on a PR; a verified-green subset now
    # runs in the ui-fast-lane job, and this checker is what stops that subset
    # from being quietly edited away instead of fixed.
    "scripts/check_ui_pr_gate_census.py",
)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _job() -> dict:
    return _workflow()["jobs"]["derived-artifacts"]


def _steps() -> list[dict]:
    return _job()["steps"]


def test_workflow_has_one_fast_prerequisite_and_one_required_aggregator():
    """The added lanes must not create a second branch-protection context.

    TASK-32908 added `ui-fast-lane` beside `pr-fast-lane`. Both are ordinary
    jobs that report their own square; neither is named under branch
    protection. `derived-artifacts` stays the single required context, and
    `needs` + its two verdict steps are what make a red lane fail it -- see
    `test_required_aggregator_fails_when_either_lane_fails`.
    """
    assert list(_workflow()["jobs"]) == [
        "pr-fast-lane",
        "ui-fast-lane",
        "derived-artifacts",
    ]


def test_required_aggregator_fails_when_either_lane_fails():
    """A lane that goes red must take the REQUIRED check down with it.

    Without a verdict step a failing lane shows its own red square while
    `Derived artifacts reproduce from their sources` stays green -- the
    "guard nobody is gated on" shape this whole workflow exists to replace.
    """
    job = _workflow()["jobs"]["derived-artifacts"]
    assert job.get("needs") == ["pr-fast-lane", "ui-fast-lane"]
    for lane in ("pr-fast-lane", "ui-fast-lane"):
        verdict = next(
            step
            for step in job["steps"]
            if step.get("name") == f"Require successful {'PR' if lane == 'pr-fast-lane' else 'UI'} fast lane"
        )
        assert verdict["if"] == (
            "${{ github.event_name == 'pull_request' && "
            f"needs.{lane}.result != 'success' }}}}"
        )
        assert "exit 1" in verdict["run"]


def test_ui_fast_lane_runs_the_census_serially_on_the_minimal_dep_set():
    """TASK-32908: the run CI performs must be the run the census was verified
    against.

    The census was verified serially, in file order, on the same minimal
    dependency set pr-fast-lane installs. xdist, a different plugin set, or a
    different order would all be untested configurations for a gate whose
    entire value is that it is green -- and a gate that lands red trains
    people to ignore it.
    """
    job = _workflow()["jobs"]["ui-fast-lane"]

    assert job["if"] == "github.event_name == 'pull_request'"
    assert "strategy" not in job, "sharding would change the verified order"

    install = next(
        step for step in job["steps"]
        if step.get("name") == "Install fast-lane dependencies"
    )["run"]
    assert "requirements-test.txt" not in install
    assert "pytest-xdist" not in install
    assert ".[" not in install

    run = next(
        step for step in job["steps"]
        if step.get("name") == "Run the gated Tests/UI slice"
    )["run"]
    assert "scripts/ui_pr_gate_census.txt" in run
    assert "-n auto" not in run and "--dist" not in run
    assert "-p no:randomly" not in run  # not installed; order is collection order


def test_ui_gate_census_is_non_empty_and_every_entry_exists():
    """A census of renamed-away paths collects nothing and still exits 0."""
    census = PROJECT_ROOT / "scripts" / "ui_pr_gate_census.txt"
    entries = [
        line.strip()
        for line in census.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert entries, "the gated Tests/UI census must not be empty"
    assert len(entries) == len(set(entries))
    for entry in entries:
        assert entry.startswith("Tests/UI/")
        assert (PROJECT_ROOT / entry).is_file(), f"censused file is gone: {entry}"


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
    """Every registered checker is invoked, so one job covers the census."""
    script = "\n".join(step.get("run", "") for step in _steps())
    for checker in CHECKERS:
        assert checker in script, f"{checker} is not run by the required job"


def test_local_preflight_runs_the_same_checker_inventory_in_order():
    """Local authoring checks must not silently omit a required CI checker."""
    preflight = (PROJECT_ROOT / "scripts" / "preflight.sh").read_text(encoding="utf-8")

    positions = [preflight.index(checker) for checker in CHECKERS]

    assert positions == sorted(positions)


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


def test_derived_artifact_checkers_use_mermaid_builder_python_pin():
    setup = next(
        step for step in _steps() if step.get("uses") == "actions/setup-python@v5"
    )

    assert setup["with"] == {"python-version": "3.12.11"}


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
