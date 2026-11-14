"""The required lanes must execute approval privacy, currentness and real gates."""

import copy
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
UI_TARGETS = {
    "Tests/UI/test_console_projection_reads_off_loop.py",
    "Tests/UI/test_approval_feedback.py::test_feedback_message_compatibility_alias_reaches_current_screen_handler",
    "Tests/UI/test_approval_details.py::test_paged_details_escape_unicode_controls_in_keys_and_values",
    "Tests/UI/test_approval_details.py::test_mounted_details_keep_unicode_separators_literal_across_pages",
    "Tests/UI/test_approval_review_repairs.py",
    "Tests/UI/test_approval_details_privacy.py",
    "Tests/UI/test_approval_details.py::test_page_from_old_revision_is_ignored",
    "Tests/UI/test_approval_interaction.py::test_stale_more_options_cannot_change_replacement",
}
OWNER_TARGETS = {
    "Tests/Canvas/test_canvas_kill_switch.py",
    "Tests/MCP/test_control_plane_permissions.py::test_remember_exact_inputs_renews_same_args_for_changed_definition",
    "Tests/MCP/test_control_plane_permissions.py::test_remember_exact_inputs_valid_rule_is_truthful_noop",
    "Tests/MCP/test_control_plane_permissions.py::test_remember_exact_inputs_requires_requested_profile_exact_confirmation",
    "Tests/MCP/test_control_plane_permissions.py::test_remember_exact_inputs_writer_failure_preserves_error_and_stale_rule",
    "Tests/MCP/test_unified_control_plane_service.py::test_remember_exact_inputs_persistence_failure_keeps_current_call",
    "Tests/Audio/test_dictation_meter_import_contract.py",
    "Tests/RAG/test_state_paths.py",
    "Tests/Chat/test_console_approval_feedback.py::test_feedback_first_access_is_single_and_binds_only_existing_bridge",
    "Tests/Chat/test_console_relay_approval_integration.py",
    "Tests/Web_Server/test_served_relay_publisher.py",
    "Tests/Chat/test_approval_display_controls.py",
    "Tests/Chat/test_console_approval_scope_journeys.py",
    "Tests/Chat/test_console_approval_feedback.py::test_host_final_arbitration_wins_after_controller_receipt",
    "Tests/RAG/test_metadata_import_boundary.py",
}


MOUNTED_OWNER_TARGETS = {
    "Tests/Backup_Recovery/test_canvas_policy_worker.py::test_stock_canvas_policy_poll_declines_busy_config_lock",
    "Tests/Backup_Recovery/test_canvas_policy_worker.py::test_stock_canvas_poll_leaves_the_retired_read_callback_handoff",
    "Tests/Backup_Recovery/test_canvas_policy_worker.py::test_identity_releases_unissued_read_and_accepts_unchanged_install",
}


def _step(workflow, job, name):
    return next(
        step for step in workflow["jobs"][job]["steps"] if step.get("name") == name
    )


def _assert_required_owner_lanes(workflow):
    commands = {}
    for job_name, step_name in (
        ("pr-fast-lane", "Run fast PR contract"),
        ("pr-fast-lane-admission", "Run admission-sensitive suites"),
    ):
        job = workflow["jobs"][job_name]
        step = _step(workflow, job_name, step_name)
        command = shlex.split(step["run"].replace("\\\n", " "), comments=True)
        assert "--continue-on-collection-errors" not in command
        assert not step.get("continue-on-error")
        assert not job.get("continue-on-error")
        commands[job_name] = set(command)
    assert OWNER_TARGETS <= commands["pr-fast-lane"]
    assert MOUNTED_OWNER_TARGETS <= commands["pr-fast-lane-admission"]
    assert not MOUNTED_OWNER_TARGETS & commands["pr-fast-lane"]
    assert {"pr-fast-lane", "pr-fast-lane-admission", "ui-fast-lane"} <= set(
        workflow["jobs"]["derived-artifacts"]["needs"]
    )


def _workflow():
    return yaml.safe_load(
        (ROOT / ".github/workflows/derived-artifacts.yml").read_text(encoding="utf-8")
    )


def test_approval_safety_targets_are_in_required_executable_lanes():
    census = {
        line.strip()
        for line in (ROOT / "scripts/ui_pr_gate_census.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert UI_TARGETS <= census
    _assert_required_owner_lanes(_workflow())


@pytest.mark.parametrize("target", sorted(MOUNTED_OWNER_TARGETS))
def test_mounted_approval_controls_cannot_also_run_in_sandboxed_lane(target):
    workflow = copy.deepcopy(_workflow())
    step = _step(workflow, "pr-fast-lane", "Run fast PR contract")
    step["run"] = step["run"].replace("--timeout=180", target + " --timeout=180", 1)
    with pytest.raises(AssertionError):
        _assert_required_owner_lanes(workflow)


@pytest.mark.parametrize("mounted", [False, True])
def test_approval_owner_membership_cannot_be_satisfied_by_comments(mounted):
    workflow = copy.deepcopy(_workflow())
    job, name, targets = (
        (
            "pr-fast-lane-admission",
            "Run admission-sensitive suites",
            MOUNTED_OWNER_TARGETS,
        )
        if mounted
        else ("pr-fast-lane", "Run fast PR contract", OWNER_TARGETS)
    )
    target = sorted(targets)[0]
    step = _step(workflow, job, name)
    lines = step["run"].splitlines(keepends=True)
    matching = [
        line for line in lines if line.strip() and line.strip().split()[0] == target
    ]
    assert len(matching) == 1
    step["run"] = (
        "".join(line for line in lines if line is not matching[0])
        + "\n"
        + "# "
        + target
        + "\n"
    )
    with pytest.raises(AssertionError):
        _assert_required_owner_lanes(workflow)
