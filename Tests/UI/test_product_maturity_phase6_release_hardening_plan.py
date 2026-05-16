"""Phase 6 release hardening and documentation planning regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PLAN = Path("Docs/superpowers/plans/2026-05-16-phase-6-release-hardening.md")
PHASE_6_QA_README = Path("Docs/superpowers/qa/product-maturity/phase-6/README.md")
TASK_13 = Path(
    "backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md"
)

PHASE_6_CHILD_TASKS = {
    "TASK-13.1": Path(
        "backlog/tasks/task-13.1 - Phase-6.1-Release-hardening-planning-and-task-breakdown.md"
    ),
    "TASK-13.2": Path(
        "backlog/tasks/task-13.2 - Phase-6.2-Full-first-time-user-release-replay.md"
    ),
    "TASK-13.3": Path(
        "backlog/tasks/task-13.3 - Phase-6.3-Power-user-workflow-release-replay.md"
    ),
    "TASK-13.4": Path(
        "backlog/tasks/task-13.4 - Phase-6.4-Keyboard-focus-accessibility-and-visual-sweep.md"
    ),
    "TASK-13.5": Path(
        "backlog/tasks/task-13.5 - Phase-6.5-Recovery-setup-and-documentation-alignment.md"
    ),
    "TASK-13.6": Path(
        "backlog/tasks/task-13.6 - Phase-6.6-Packaging-configuration-and-data-safety-validation.md"
    ),
    "TASK-13.7": Path(
        "backlog/tasks/task-13.7 - Phase-6.7-Public-roadmap-release-closeout.md"
    ),
}


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _markdown_table_row(markdown: str, first_cell_text: str) -> list[str]:
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or first_cell_text not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == first_cell_text:
            return cells
    raise AssertionError(f"Missing markdown table row for {first_cell_text!r}")


def test_phase6_release_hardening_plan_tracks_pr_sized_gates() -> None:
    plan = _text(PLAN)
    parent_task = _text(TASK_13)
    tracker = _text(TRACKER)
    qa_readme = _text(PHASE_6_QA_README)

    assert "status: In Progress" in parent_task
    assert PLAN.as_posix() in parent_task
    assert PLAN.as_posix() in tracker
    assert PHASE_6_QA_README.as_posix() in tracker
    assert "Status: TASK-13.1 and TASK-13.2 done; TASK-13.3 through TASK-13.7 not started" in qa_readme

    for required_section in (
        "## Source Of Truth",
        "## Phase 6 Scope Boundary",
        "## Phase 6 Child Task Plan",
        "## Release QA Evidence Requirements",
        "### Task 1: Phase 6.1 Release Hardening Planning And Task Breakdown",
        "### Task 7: Phase 6.7 Public Roadmap Release Closeout",
    ):
        assert required_section in plan

    for required_workflow in (
        "first-time user replay",
        "power-user workflow replay",
        "keyboard/focus/accessibility",
        "visual polish",
        "error/recovery copy",
        "packaging",
        "configuration",
        "migration",
        "public roadmap",
    ):
        assert required_workflow in plan

    for task_id, task_path in PHASE_6_CHILD_TASKS.items():
        task = _text(task_path)
        assert task_id in plan
        assert task_id in parent_task
        assert task_id in tracker
        assert "parent_task_id: TASK-13" in task
        assert "QA walkthrough" in task
        assert "focused regression" in task.lower()
        assert "P0/P1" in task
        if task_id == "TASK-13.1":
            assert "status: Done" in task
            for ac_number in range(1, 5):
                assert f"- [x] #{ac_number}" in task
            assert "## Implementation Notes" in task
        elif task_id == "TASK-13.2":
            assert "status: Done" in task
            for ac_number in range(1, 5):
                assert f"- [x] #{ac_number}" in task
            assert "## Implementation Notes" in task
        else:
            assert "status: To Do" in task
            for ac_number in range(1, 5):
                assert f"- [ ] #{ac_number}" in task

    phase_row = _markdown_table_row(tracker, "Phase 6: Release Hardening And Documentation")
    assert "in-progress; TASK-13.1 and TASK-13.2 done" in phase_row[2]
    for task_id in PHASE_6_CHILD_TASKS:
        assert task_id in phase_row[3]
    assert PLAN.as_posix() in phase_row[4]
    assert PHASE_6_QA_README.as_posix() in phase_row[4]
    assert "first-time" in phase_row[5]
    assert "power-user" in phase_row[5]
    assert "packaging" in phase_row[5]
