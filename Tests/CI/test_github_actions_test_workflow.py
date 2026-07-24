import shlex
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_LEASE_TEST_TARGETS = (
    "Tests/Model_Artifacts/test_operation_leases.py",
    "Tests/Model_Artifacts/test_operation_leases_process.py",
)


def _workflow_text() -> str:
    return (PROJECT_ROOT / ".github" / "workflows" / "test.yml").read_text()


def _all_tests_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  all-tests:")
    end = workflow.index("  test-summary:", start)
    return workflow[start:end]


def _textual_minimum_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  textual-minimum:")
    end = workflow.index("  all-tests:", start)
    return workflow[start:end]


def _artifact_lease_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  artifact-lease-spike:")
    shape_start = workflow.find("  artifact-lease-shape:", start)
    end = (
        shape_start
        if shape_start != -1
        else workflow.index("  artifact-lease-gate:", start)
    )
    return workflow[start:end]


def _artifact_lease_shape_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  artifact-lease-shape:")
    end = workflow.index("  artifact-lease-gate:", start)
    return workflow[start:end]


def _artifact_lease_gate_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  artifact-lease-gate:")
    end = workflow.index("  integration-tests:", start)
    return workflow[start:end]


def _test_summary_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  test-summary:")
    return workflow[start:]


def _pytest_invocations(block: str) -> list[list[str]]:
    lines = iter(block.splitlines())
    pytest_invocations: list[list[str]] = []

    for raw_line in lines:
        command = raw_line.strip()
        if command != "pytest" and not command.startswith("pytest "):
            continue

        while command.endswith("\\"):
            command = f"{command[:-1].rstrip()} {next(lines).strip()}"
        pytest_invocations.append(shlex.split(command))

    return pytest_invocations


def _assert_artifact_lease_test_targets(block: str) -> None:
    pytest_invocations = _pytest_invocations(block)

    assert len(pytest_invocations) == 1
    test_targets = tuple(
        token.removeprefix("./")
        for token in pytest_invocations[0][1:]
        if token.removeprefix("./") == "Tests"
        or token.removeprefix("./").startswith("Tests/")
    )
    assert test_targets == ARTIFACT_LEASE_TEST_TARGETS


def test_ci_installs_pytest_timeout_for_configured_test_timeouts() -> None:
    requirements = (PROJECT_ROOT / "requirements-test.txt").read_text()

    assert "pytest-timeout" in requirements


def test_pytest_ui_marker_is_registered_for_ci_marker_selection() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()

    assert '"ui: marks tests as UI/Textual tests"' in pyproject


def test_full_suite_job_is_bounded_and_manual_only() -> None:
    all_tests_job = _all_tests_job_block()

    assert "timeout-minutes:" in all_tests_job
    assert "if: github.event_name == 'workflow_dispatch'" in all_tests_job
    assert "pull_request" not in all_tests_job
    assert "name: Full Test Suite (Manual)" in all_tests_job
    assert "pytest ./Tests/" in all_tests_job


def test_ci_exercises_mcp_against_minimum_textual() -> None:
    textual_minimum = _textual_minimum_job_block()
    test_summary = _test_summary_job_block()

    assert 'pip install "textual==8.0.0"' in textual_minimum
    assert "Tests/CI/test_textual_runtime_contract.py" in textual_minimum
    assert "Tests/UI/test_mcp_workbench.py" in textual_minimum
    assert "Tests/UI/test_mcp_tools_mode.py" in textual_minimum
    assert (
        "needs: [unit-tests, integration-tests, ui-tests, textual-minimum, "
        "artifact-lease-gate]" in test_summary
    )


def test_artifact_lease_spike_runs_natively_on_three_operating_systems() -> None:
    block = _artifact_lease_job_block()

    assert "ubuntu-latest" in block
    assert "macos-latest" in block
    assert "windows-latest" in block
    assert 'python-version: ["3.11"]' in block
    assert "pip install -e ." in block
    assert "pip install -r requirements-test.txt" in block
    _assert_artifact_lease_test_targets(block)


def test_artifact_lease_target_check_rejects_unrelated_explicit_test() -> None:
    block = _artifact_lease_job_block()
    mutated = block.replace(
        "Tests/Model_Artifacts/test_operation_leases_process.py -v",
        "Tests/Model_Artifacts/test_operation_leases_process.py \\\n"
        "          Tests/Other/test_unrelated.py -v",
    )

    assert mutated != block
    with pytest.raises(AssertionError):
        _assert_artifact_lease_test_targets(mutated)


def test_ci_shape_regression_runs_in_dedicated_pull_request_job() -> None:
    workflow = _workflow_text()

    assert "  artifact-lease-shape:" in workflow
    shape = _artifact_lease_shape_job_block()
    install_commands = [
        line.strip()
        for line in shape.splitlines()
        if line.strip().startswith(("pip install ", "python -m pip install "))
    ]

    assert "runs-on: ubuntu-latest" in shape
    assert "if:" not in shape
    assert "uses: actions/checkout@v4" in shape
    assert "uses: actions/setup-python@v5" in shape
    assert 'python-version: "3.11"' in shape
    assert install_commands == ["python -m pip install pytest pytest-timeout"]
    assert _pytest_invocations(shape) == [
        [
            "pytest",
            "Tests/CI/test_github_actions_test_workflow.py",
            "--confcutdir=Tests/CI",
        ]
    ]


def test_artifact_lease_gate_exposes_stable_required_context() -> None:
    gate = _artifact_lease_gate_job_block()
    test_summary = _test_summary_job_block()

    assert "name: Artifact Lease Gate" in gate
    assert "runs-on: ubuntu-latest" in gate
    assert "needs: [artifact-lease-spike, artifact-lease-shape]" in gate
    assert "if: always()" in gate
    assert (
        'if [ "${{ needs.artifact-lease-spike.result }}" != "success" ] || '
        '[ "${{ needs.artifact-lease-shape.result }}" != "success" ]; then' in gate
    )
    assert "exit 1" in gate
    assert "artifact-lease-gate" in test_summary
