"""Read-only planning must use the real native primary and required settlement."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile


def test_accepted_planning_uses_original_native_primary(tmp_path):
    _run(
        tmp_path,
        "primary",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize("mode", ["new-saved", "new-unsaved"])
def test_original_initial_planning_persistence_and_acceptance(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


def test_planning_never_constructs_optional_tool_services(tmp_path):
    _run(
        tmp_path,
        "factory",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize(
    "mode", ["invalid-json", "budget-zero", "cancel-cleanup", "tools"]
)
def test_native_planning_receiving_boundaries(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize("mode", ["wrapper", "wrapper-changed", "wrapper-unavailable"])
def test_native_planning_future_verifier_metadata(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize(
    "mode", ["instructions", "unused", "repair", "direct-guard", "cleanup-source"]
)
def test_native_planning_original_input_and_receivers(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "required-write",
        "required-return",
        "worker-failure",
        "response-close-failure",
        "late-source-return",
        "prepare-return-error",
    ],
)
def test_native_planning_required_failure_retains_original_custody(tmp_path, mode):
    # This is an uncertainty-retention control, never a successful drain receipt.
    with pytest.raises(AssertionError, match="F_REQUIRED_CUSTODY_RETAINED_" + mode):
        _run(
            tmp_path,
            mode,
            "retained",
            script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
            timeout=60,
        )


def test_planning_retains_root_until_original_provider_worker_returns(tmp_path):
    _run(
        tmp_path,
        "worker-tail",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


def test_shutdown_stop_cut_retains_original_custody_after_worker_return(tmp_path):
    # Current watchdog Stop may cut before terminal usage. Physical return alone
    # is not a known accounting receipt (ADR-285 current-dev clarification).
    with pytest.raises(AssertionError, match="F_REQUIRED_CUSTODY_RETAINED_shutdown-worker-tail"):
        _run(
            tmp_path,
            "shutdown-worker-tail",
            "retained",
            script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
            timeout=60,
        )


@pytest.mark.parametrize(
    "mode", ["system-context", "system-oversize", "repair-instruction"]
)
def test_native_planning_retains_required_selected_and_repair_inputs(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


def test_native_runtime_waits_for_original_maintenance_operation_return(tmp_path):
    _run(
        tmp_path,
        "maintenance-tail",
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "executor-replaced",
        "reader-replaced",
        "executor-hash-replaced",
        "executor-resolver-replaced",
    ],
)
def test_native_planning_refuses_replaced_receivers_without_effects(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_planning_probe.py").read_text(),
        timeout=60,
    )
