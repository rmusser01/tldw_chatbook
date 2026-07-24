import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        "artifact-lease-gate]"
        in test_summary
    )


def test_artifact_lease_spike_runs_natively_on_three_operating_systems() -> None:
    block = _artifact_lease_job_block()

    assert "ubuntu-latest" in block
    assert "macos-latest" in block
    assert "windows-latest" in block
    assert 'python-version: ["3.11"]' in block
    assert "pip install -e ." in block
    assert "pip install -r requirements-test.txt" in block
    assert "Tests/Model_Artifacts/test_operation_leases.py" in block
    assert "Tests/Model_Artifacts/test_operation_leases_process.py" in block
    assert (
        re.search(
            r"^\s*(?:pytest\s+)?(?:\./)?Tests/?(?:\s|\\|$)",
            block,
            re.MULTILINE,
        )
        is None
    )


def test_artifact_lease_gate_exposes_stable_required_context() -> None:
    gate = _artifact_lease_gate_job_block()
    test_summary = _test_summary_job_block()

    assert "name: Artifact Lease Gate" in gate
    assert "runs-on: ubuntu-latest" in gate
    assert "needs: [artifact-lease-spike]" in gate
    assert "if: always()" in gate
    assert "needs.artifact-lease-spike.result" in gate
    assert '!= "success"' in gate
    assert "exit 1" in gate
    assert "artifact-lease-gate" in test_summary
