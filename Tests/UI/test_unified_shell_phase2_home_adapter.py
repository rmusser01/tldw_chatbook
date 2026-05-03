from pathlib import Path


PHASE_2_ROOT = Path("Docs/superpowers/qa/unified-shell/phase-2")
EVIDENCE = PHASE_2_ROOT / "2026-05-03-home-active-work-adapter-contract.md"
README = PHASE_2_ROOT / "README.md"
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")
TASK_4_1 = Path("backlog/tasks/task-4.1 - Phase-2.1-Add-Home-active-work-adapter-contract.md")


def test_phase_two_home_adapter_evidence_exists_and_records_verification():
    assert EVIDENCE.exists()
    text = EVIDENCE.read_text(encoding="utf-8")

    assert "TASK-4.1" in text
    assert "Home active-work adapter contract" in text
    assert "Tests/Home/test_active_work_adapter.py" in text
    assert "Tests/UI/test_home_screen.py" in text
    assert "15 passed" in text
    assert "UnavailableHomeActiveWorkAdapter" in text


def test_phase_two_home_adapter_is_linked_from_index_roadmap_and_task():
    evidence_name = EVIDENCE.name

    assert evidence_name in README.read_text(encoding="utf-8")

    roadmap_text = ROADMAP.read_text(encoding="utf-8")
    assert "TASK-4.1" in roadmap_text
    assert evidence_name in roadmap_text
    assert "| Phase 2 | `Docs/superpowers/qa/unified-shell/phase-2/` | in-progress |" in roadmap_text

    task_text = TASK_4_1.read_text(encoding="utf-8")
    assert "status: Done" in task_text
    assert "- [x] #1" in task_text
    assert "- [x] #4" in task_text
