"""Contract tests for TASK-19637's bounded cross-platform evidence workflow."""

from __future__ import annotations

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github/workflows/task-19637-platform-evidence.yml"
JUNIT_PATH = "${{ runner.temp }}/task-19637-platform-junit.xml"

EXPECTED_NODES = (
    "Tests/Tools/test_workspace_root_pin.py::test_root_replacement_before_pin_is_refused_by_identity",
    "Tests/Tools/test_workspace_root_pin.py::test_root_replacement_after_pin_never_redirects_relative_io",
    "Tests/Tools/test_workspace_root_pin.py::test_root_replacement_after_pin_never_redirects_relative_write",
    "Tests/Tools/test_workspace_tool_executor.py::test_executor_uses_fixed_private_launch_and_admits_before_stdin",
    "Tests/Tools/test_workspace_tool_executor.py::test_timeout_terminates_the_tree_and_returns_no_in_process_result",
    "Tests/Tools/test_workspace_tool_executor.py::test_crash_and_bounded_stderr_return_only_fixed_metadata",
    "Tests/Tools/test_workspace_tool_executor.py::test_platform_evidence_representative_one_shot_operations",
    "Tests/Tools/test_workspace_tool_executor.py::test_platform_evidence_outer_executor_git_ignores_workspace_path",
    "Tests/Tools/test_workspace_tool_executor.py::test_pinned_git_supports_linked_worktree_without_granting_metadata_fs_access",
    "Tests/Agents/test_local_tool_provider.py::test_each_local_workspace_tool_routes_once_through_injected_executor",
    "Tests/Agents/test_virtual_cli_provider.py::test_provider_constructs_and_injects_real_executor_by_default",
)


def _workflow() -> dict[str, object]:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _steps() -> list[dict[str, object]]:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["platform-evidence"]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    return steps


def _step(name: str) -> dict[str, object]:
    matches = [step for step in _steps() if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def test_workflow_retests_labeled_pr_updates_with_read_only_permissions() -> None:
    workflow = _workflow()

    assert workflow["on"] == {
        "pull_request": {"types": ["labeled", "synchronize"]},
        "workflow_dispatch": "",
    }
    assert workflow["permissions"] == {"contents": "read"}
    serialized = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "secrets." not in serialized
    assert ": write" not in serialized


def test_workflow_pins_exact_matrix_python_head_and_job_bound() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"platform-evidence"}
    job = jobs["platform-evidence"]
    assert isinstance(job, dict)

    assert job["if"] == (
        "github.event_name == 'workflow_dispatch' || "
        "contains(github.event.pull_request.labels.*.name, "
        "'task-19637-platform-evidence')"
    )
    assert job["timeout-minutes"] == "30"
    assert job["strategy"] == {
        "fail-fast": "false",
        "matrix": {
            "os": ["ubuntu-24.04", "windows-2022", "macos-15-intel"]
        },
    }
    assert job["runs-on"] == "${{ matrix.os }}"

    checkout = _step("Check out the exact tested commit")
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"] == {
        "ref": "${{ github.event.pull_request.head.sha || github.sha }}"
    }
    setup = _step("Set up Python 3.12")
    assert setup["uses"] == "actions/setup-python@v5"
    assert setup["with"] == {"python-version": "3.12"}


def test_workflow_runs_each_bounded_named_node_once_and_emits_junit() -> None:
    run_step = _step("Run bounded pinned-workspace evidence")
    command = run_step["run"]
    assert isinstance(command, str)

    assert "python -m pytest" in command
    for node in EXPECTED_NODES:
        assert command.split().count(node) == 1
    assert command.count("Tests/") == len(EXPECTED_NODES)
    assert f'--junitxml="{JUNIT_PATH}"' in command
    assert "$RUNNER_TEMP" not in command
    assert "--tb=short" in command
    assert "continue-on-error" not in run_step
    assert "if" not in run_step


def test_workflow_uploads_junit_on_failure_without_masking_test_failure() -> None:
    serialized = WORKFLOW_PATH.read_text(encoding="utf-8")
    install = _step("Install test dependencies")
    run_step = _step("Run bounded pinned-workspace evidence")
    upload = _step("Upload JUnit evidence")

    assert install["run"] == 'python -m pip install -e ".[dev]"'
    assert "continue-on-error" not in serialized
    assert "|| true" not in serialized
    assert upload == {
        "name": "Upload JUnit evidence",
        "if": "always()",
        "uses": "actions/upload-artifact@v4",
        "with": {
            "name": "task-19637-platform-evidence-${{ matrix.os }}",
            "path": JUNIT_PATH,
            "if-no-files-found": "error",
        },
    }
    assert f'--junitxml="{upload["with"]["path"]}"' in run_step["run"]
    assert "if" not in run_step
