"""Contracts for the bounded fast PR lane and comprehensive CI cadence."""

from __future__ import annotations

import copy
import shlex
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = PROJECT_ROOT / ".github" / "workflows"
PULL_REQUEST_TYPES = ["opened", "synchronize", "reopened", "ready_for_review"]
PUSH_ONLY_CANCELLATION = (
    "${{ github.event_name == 'push' && github.ref != 'refs/heads/main' }}"
)
FAST_LANE_TARGETS = (
    "Tests/CI",
    "Tests/test_smoke.py",
    "Tests/Model_Artifacts/test_operation_leases.py",
    "Tests/Model_Artifacts/test_operation_leases_process.py",
    "Tests/UI/test_mcp_workbench.py",
    "Tests/UI/test_mcp_tools_mode.py",
)
HEAVY_JOB_KEYS = {
    "core-tests",
    "artifact-lease-spike",
    "artifact-lease-shape",
    "artifact-lease-gate",
    "ui-tests",
    "textual-minimum",
    "all-tests",
    "test-summary",
}
STANDALONE_WORKFLOWS = (
    "derived-artifacts.yml",
    "css-bundle-guard.yml",
    "perf-guard.yml",
    "backlog-guard.yml",
)


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOW_ROOT / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def _named_step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def _pytest_targets(run: str) -> tuple[str, ...]:
    tokens = shlex.split(run.replace("\\\n", " "))
    return tuple(
        token
        for token in tokens[1:]
        if token == "Tests" or token.startswith("Tests/")
    )


def _assert_required_aggregation(workflow: dict) -> None:
    fast = workflow["jobs"]["pr-fast-lane"]
    required = workflow["jobs"]["derived-artifacts"]
    assert not fast.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in fast["steps"])
    assert required["name"] == "Derived artifacts reproduce from their sources"
    assert required.get("needs") == ["pr-fast-lane"]
    assert required["if"] == "${{ always() }}"
    assert not required.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in required["steps"])

    verdict = _named_step(required, "Require successful PR fast lane")
    assert not verdict.get("continue-on-error", False)
    assert verdict["if"] == (
        "${{ github.event_name == 'pull_request' && "
        "needs.pr-fast-lane.result != 'success' }}"
    )
    assert "needs.pr-fast-lane.result" in verdict["run"]
    assert "exit 1" in verdict["run"]


def _assert_fast_lane_invocation(workflow: dict) -> None:
    fast = workflow["jobs"]["pr-fast-lane"]
    run = _named_step(fast, "Run fast PR contract")["run"]
    assert tuple(shlex.split(run.replace("\\\n", " "))) == (
        "pytest",
        *FAST_LANE_TARGETS,
        "--timeout=180",
        "--tb=short",
    )


def test_heavy_tests_run_only_on_main_push_or_manual_dispatch() -> None:
    workflow = _workflow("test.yml")
    triggers = _triggers(workflow)

    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"]["branches"] == ["main"]
    assert set(workflow["jobs"]) == HEAVY_JOB_KEYS
    assert workflow["permissions"] == {"contents": "read"}
    assert "createComment" not in (WORKFLOW_ROOT / "test.yml").read_text()


def test_dedicated_nightly_owns_exact_schedule_and_full_tree_matrix() -> None:
    workflow = _workflow("nightly-deep.yml")
    triggers = _triggers(workflow)

    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "30 8 * * *"}]
    assert set(workflow["jobs"]) == {"resolve-dev-sha", "nightly-deep"}

    resolver = workflow["jobs"]["resolve-dev-sha"]
    assert resolver["outputs"] == {"sha": "${{ steps.resolve.outputs.sha }}"}
    resolver_checkout = next(
        step
        for step in resolver["steps"]
        if step.get("uses") == "actions/checkout@v4"
    )
    assert resolver_checkout["with"] == {"ref": "dev"}
    resolve = _named_step(resolver, "Resolve one dev commit for every matrix leg")
    assert resolve["id"] == "resolve"
    assert "git rev-parse HEAD" in resolve["run"]
    assert "GITHUB_OUTPUT" in resolve["run"]

    nightly = workflow["jobs"]["nightly-deep"]
    assert nightly["needs"] == ["resolve-dev-sha"]
    assert nightly["strategy"]["matrix"]["include"] == [
        {"os": "ubuntu-latest", "python-version": "3.11"},
        {"os": "ubuntu-latest", "python-version": "3.12"},
        {"os": "ubuntu-latest", "python-version": "3.13"},
        {"os": "macos-latest", "python-version": "3.12"},
        {"os": "windows-latest", "python-version": "3.12"},
    ]
    checkout = next(
        step
        for step in nightly["steps"]
        if step.get("uses") == "actions/checkout@v4"
    )
    assert checkout["with"] == {
        "ref": "${{ needs.resolve-dev-sha.outputs.sha }}",
        "fetch-depth": 0,
    }
    record = _named_step(nightly, "Record tested dev commit")
    assert "needs.resolve-dev-sha.outputs.sha" in record["run"]
    assert "GITHUB_STEP_SUMMARY" in record["run"]
    run = _named_step(
        nightly, "Run deep suite (serial, thorough, slow tiers, cache-off)"
    )
    assert "pytest ./Tests/" in run["run"]
    assert "--run-slow" in run["run"]
    assert "-n auto" not in run["run"]


def test_fast_lane_is_one_serial_minimal_python_311_job() -> None:
    fast = _workflow("derived-artifacts.yml")["jobs"]["pr-fast-lane"]

    assert fast["name"] == "PR Fast Lane"
    assert fast["if"] == "github.event_name == 'pull_request'"
    assert fast["runs-on"] == "ubuntu-latest"
    assert fast["timeout-minutes"] == 20
    assert "strategy" not in fast
    assert len(fast["steps"]) == 4

    setup = next(
        step
        for step in fast["steps"]
        if step.get("uses") == "actions/setup-python@v5"
    )
    assert setup["with"]["python-version"] == "3.11"

    install = _named_step(fast, "Install fast-lane dependencies")["run"]
    assert shlex.split(install) == [
        "python",
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "pytest",
        "pytest-asyncio",
        "pytest-timeout",
        "packaging",
    ]
    assert "requirements-test.txt" not in install
    assert ".[" not in install
    all_commands = "\n".join(str(step.get("run", "")) for step in fast["steps"])
    assert all_commands.count("pip install") == 1


def test_fast_lane_target_set_is_exact_and_non_overlapping() -> None:
    workflow = _workflow("derived-artifacts.yml")
    fast = workflow["jobs"]["pr-fast-lane"]
    run = _named_step(fast, "Run fast PR contract")["run"]
    targets = _pytest_targets(run)

    _assert_fast_lane_invocation(workflow)
    assert targets == FAST_LANE_TARGETS
    for index, target in enumerate(targets):
        target_path = Path(target)
        for other in targets[index + 1 :]:
            other_path = Path(other)
            assert target_path not in other_path.parents
            assert other_path not in target_path.parents


def test_required_context_fails_closed_and_keeps_artifact_checks_install_free() -> None:
    workflow = _workflow("derived-artifacts.yml")
    _assert_required_aggregation(workflow)

    required = workflow["jobs"]["derived-artifacts"]
    verdict_index = next(
        index
        for index, step in enumerate(required["steps"])
        if step.get("name") == "Require successful PR fast lane"
    )
    checker_steps = required["steps"][verdict_index + 1 :]
    assert checker_steps
    assert all(step.get("if") == "${{ !cancelled() }}" for step in checker_steps)
    assert "pip install" not in "\n".join(str(step) for step in required["steps"])


def test_required_aggregation_contract_rejects_missing_prerequisite() -> None:
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    mutated["jobs"]["derived-artifacts"].pop("needs", None)

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


def test_required_aggregation_contract_rejects_partial_failure_check() -> None:
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    verdict = _named_step(
        mutated["jobs"]["derived-artifacts"], "Require successful PR fast lane"
    )
    verdict["if"] = verdict["if"].replace("!= 'success'", "== 'failure'")

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


@pytest.mark.parametrize(
    ("job_name", "step_name"),
    [
        ("pr-fast-lane", None),
        ("derived-artifacts", None),
        ("derived-artifacts", "Require successful PR fast lane"),
        ("derived-artifacts", "Generated stylesheets reproduce from their sources"),
    ],
)
def test_required_aggregation_contract_rejects_continue_on_error(
    job_name: str, step_name: str | None
) -> None:
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    job = mutated["jobs"][job_name]
    target = job if step_name is None else _named_step(job, step_name)
    target["continue-on-error"] = True

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


@pytest.mark.parametrize(
    "flag",
    ["--collect-only", "-k smoke", "--ignore=Tests/CI", "--deselect=Tests/test_smoke.py"],
)
def test_fast_lane_contract_rejects_selection_suppressing_flags(flag: str) -> None:
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    run_step = _named_step(mutated["jobs"]["pr-fast-lane"], "Run fast PR contract")
    run_step["run"] += f" {flag}"

    with pytest.raises(AssertionError):
        _assert_fast_lane_invocation(mutated)


def test_focused_guards_keep_dev_pr_and_dev_main_push_coverage() -> None:
    for workflow_name in STANDALONE_WORKFLOWS:
        triggers = _triggers(_workflow(workflow_name))

        assert triggers["pull_request"]["branches"] == ["dev"]
        assert triggers["pull_request"]["types"] == PULL_REQUEST_TYPES
        assert triggers["push"]["branches"] == ["dev", "main"]


def test_pull_request_workflows_are_never_cancelled_in_progress() -> None:
    for workflow_name in STANDALONE_WORKFLOWS:
        concurrency = _workflow(workflow_name)["concurrency"]
        assert concurrency["cancel-in-progress"] == PUSH_ONLY_CANCELLATION

    heavy = _workflow("test.yml")
    assert "pull_request" not in _triggers(heavy)
    assert heavy["concurrency"]["cancel-in-progress"] == PUSH_ONLY_CANCELLATION
