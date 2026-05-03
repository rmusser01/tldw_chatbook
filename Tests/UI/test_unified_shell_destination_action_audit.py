from pathlib import Path

from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER


PHASE_1_ROOT = Path("Docs/superpowers/qa/unified-shell/phase-1")
AUDIT = PHASE_1_ROOT / "2026-05-03-destination-action-audit.md"
README = PHASE_1_ROOT / "README.md"
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")

REQUIRED_COLUMNS = {
    "Destination",
    "Primary route",
    "Action owner",
    "Usability status",
    "Classification",
    "Evidence",
    "Follow-up",
}

REQUIRED_CLASSIFICATIONS = {
    "working-workflow",
    "honest-blocked",
    "false-affordance",
}

REQUIRED_EVIDENCE = {
    "Tests/UI/test_master_shell_navigation.py",
    "Tests/UI/test_destination_shells.py",
    "Tests/UI/test_unified_shell_destination_action_audit.py",
}


def test_destination_action_audit_exists_and_covers_every_top_level_destination():
    assert AUDIT.exists()
    text = AUDIT.read_text(encoding="utf-8")

    for column in REQUIRED_COLUMNS:
        assert column in text

    for destination in SHELL_DESTINATION_ORDER:
        assert f"| {destination.label} | `{destination.primary_route}` |" in text
        assert destination.destination_id in text


def test_destination_action_audit_distinguishes_workflow_statuses_and_false_affordances():
    text = AUDIT.read_text(encoding="utf-8")

    for classification in REQUIRED_CLASSIFICATIONS:
        assert classification in text

    assert "False affordance" in text
    assert "Honest blocked state" in text
    assert "Working workflow" in text


def test_destination_action_audit_records_running_app_evidence_and_scope_boundary():
    text = AUDIT.read_text(encoding="utf-8")

    for evidence in REQUIRED_EVIDENCE:
        assert evidence in text

    assert "Running-App QA Evidence" in text
    assert "Follow-on Scope Boundary" in text
    assert "PR-sized" in text
    assert "TASK-2.2" in text


def test_phase_one_index_and_roadmap_link_destination_action_audit():
    audit_name = AUDIT.name

    assert audit_name in README.read_text(encoding="utf-8")
    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    assert "TASK-2.2" in roadmap_text
    assert audit_name in roadmap_text
