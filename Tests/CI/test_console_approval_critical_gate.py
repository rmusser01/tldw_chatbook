"""The required lanes must execute approval privacy, currentness and real gates."""

import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
UI_TARGETS = {
    "Tests/UI/test_approval_feedback.py::test_feedback_message_compatibility_alias_reaches_current_screen_handler",
    "Tests/UI/test_approval_details.py::test_paged_details_escape_unicode_controls_in_keys_and_values",
    "Tests/UI/test_approval_details.py::test_mounted_details_keep_unicode_separators_literal_across_pages",
    "Tests/UI/test_approval_review_repairs.py",
    "Tests/UI/test_approval_details_privacy.py",
    "Tests/UI/test_approval_details.py::test_page_from_old_revision_is_ignored",
    "Tests/UI/test_approval_interaction.py::test_stale_more_options_cannot_change_replacement",
}
OWNER_TARGETS = {
    "Tests/Audio/test_dictation_meter_import_contract.py",
    "Tests/RAG/test_state_paths.py",
    "Tests/Chat/test_console_approval_feedback.py::test_feedback_first_access_is_single_and_binds_only_existing_bridge",
    "Tests/Chat/test_console_relay_approval_integration.py",
    "Tests/Web_Server/test_served_relay_publisher.py",
    "Tests/Chat/test_approval_display_controls.py",
    "Tests/Chat/test_console_approval_scope_journeys.py",
    "Tests/Chat/test_console_approval_feedback.py::test_host_final_arbitration_wins_after_controller_receipt",
}


def test_approval_safety_targets_are_in_required_executable_lanes():
    census = {
        line.strip()
        for line in (ROOT / "scripts/ui_pr_gate_census.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert UI_TARGETS <= census
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/derived-artifacts.yml").read_text()
    )
    job = workflow["jobs"]["pr-fast-lane"]
    step = next(
        step for step in job["steps"] if step.get("name") == "Run fast PR contract"
    )
    command = shlex.split(step["run"].replace("\\\n", " "))
    assert OWNER_TARGETS <= set(command)
    assert "--continue-on-collection-errors" not in command
    assert not step.get("continue-on-error")
    assert not job.get("continue-on-error")
    assert {"pr-fast-lane", "ui-fast-lane"} <= set(
        workflow["jobs"]["derived-artifacts"]["needs"]
    )
