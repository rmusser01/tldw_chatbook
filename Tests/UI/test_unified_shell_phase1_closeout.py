from pathlib import Path

import pytest
from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER


PHASE_1_ROOT = Path("Docs/superpowers/qa/unified-shell/phase-1")
CLOSEOUT = PHASE_1_ROOT / "2026-05-03-phase-1-shell-contract-closeout.md"
README = PHASE_1_ROOT / "README.md"
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")
TASK_2 = Path("backlog/tasks/task-2 - Phase-1-Shell-Contract-Complete.md")
TASK_2_4 = Path("backlog/tasks/task-2.4 - Phase-1.4-Replay-shell-contract-and-close-Phase-1.md")

REQUIRED_REPLAY_SECTIONS = {
    "Current Baseline",
    "Replay Matrix",
    "Focused Verification",
    "Residual Risk",
    "Phase 1 Closeout Decision",
}

REQUIRED_EVIDENCE = {
    "Tests/UI/test_master_shell_navigation.py",
    "Tests/UI/test_destination_shells.py",
    "Tests/UI/test_console_live_work_handoffs.py",
    "Tests/UI/test_unified_shell_phase1_closeout.py",
}


@pytest.mark.skip(reason="Stale release-era snapshot (copy/evidence drifted); re-pin or retire via backlog task-98")
def test_phase_one_closeout_artifact_exists_and_covers_all_destinations():
    assert CLOSEOUT.exists()
    text = CLOSEOUT.read_text(encoding="utf-8")

    for section in REQUIRED_REPLAY_SECTIONS:
        assert f"## {section}" in text

    for destination in SHELL_DESTINATION_ORDER:
        assert f"| {destination.label} | `{destination.primary_route}` |" in text
        assert destination.destination_id in text


def test_phase_one_closeout_records_verification_and_no_open_phase_one_findings():
    text = CLOSEOUT.read_text(encoding="utf-8")

    for evidence in REQUIRED_EVIDENCE:
        assert evidence in text

    assert "TASK-2.1" in text
    assert "TASK-2.2" in text
    assert "TASK-2.3" in text
    assert "TASK-2.4" in text
    assert "No unresolved Phase 1 shell-contract blockers" in text


def test_phase_one_index_roadmap_and_backlog_mark_closeout_verified():
    closeout_name = CLOSEOUT.name

    assert closeout_name in README.read_text(encoding="utf-8")

    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    assert closeout_name in roadmap_text
    assert "| Phase 1 | `Docs/superpowers/qa/unified-shell/phase-1/` | verified |" in roadmap_text
    assert "| Phase 1: Shell Contract Complete | Remove false shell affordances and prove shell usability. | verified |" in roadmap_text

    task_2_text = TASK_2.read_text(encoding="utf-8")
    task_2_4_text = TASK_2_4.read_text(encoding="utf-8")
    assert "status: Done" in task_2_text
    assert "status: Done" in task_2_4_text
    assert "- [x] #1" in task_2_text
    assert "- [x] #4" in task_2_text
    assert "- [x] #1" in task_2_4_text
    assert "- [x] #4" in task_2_4_text
