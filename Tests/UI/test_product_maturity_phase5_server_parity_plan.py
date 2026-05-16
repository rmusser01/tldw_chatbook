"""Phase 5 server parity and live integrations planning regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT_PARENT_DEPTH = 2
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PLAN = Path("Docs/superpowers/plans/2026-05-16-phase-5-server-parity-live-integrations.md")
PHASE_5_QA_README = Path("Docs/superpowers/qa/product-maturity/phase-5/README.md")
TASK_12 = Path(
    "backlog/tasks/task-12 - Product-Maturity-Phase-5-Server-Parity-And-Live-Integrations.md"
)
TASK_12_1_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-1-current-state-inventory.md"
)
TASK_12_2_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-2-active-server-auth-live-status.md"
)
TASK_12_3_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-3-server-events-notifications-live-feed.md"
)
TASK_12_4_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-4-sync-mirror-dry-run-workflow-surfacing.md"
)
TASK_12_5_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-5-high-value-domain-parity-workflows.md"
)

PHASE_5_CHILD_TASKS = {
    "TASK-12.1": Path(
        "backlog/tasks/task-12.1 - Phase-5.1-Server-parity-current-state-inventory.md"
    ),
    "TASK-12.2": Path(
        "backlog/tasks/task-12.2 - Phase-5.2-Active-server-auth-live-status.md"
    ),
    "TASK-12.3": Path(
        "backlog/tasks/task-12.3 - Phase-5.3-Server-events-and-notifications-live-feed.md"
    ),
    "TASK-12.4": Path(
        "backlog/tasks/task-12.4 - Phase-5.4-Sync-mirror-dry-run-workflow-surfacing.md"
    ),
    "TASK-12.5": Path(
        "backlog/tasks/task-12.5 - Phase-5.5-High-value-domain-parity-workflows.md"
    ),
    "TASK-12.6": Path(
        "backlog/tasks/task-12.6 - Phase-5.6-Server-parity-live-integration-closeout.md"
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


def test_phase5_server_parity_plan_reconciles_current_dev_state() -> None:
    plan = _text(PLAN)
    parent_task = _text(TASK_12)
    tracker = _text(TRACKER)
    qa_readme = _text(PHASE_5_QA_README)
    inventory_evidence = _text(TASK_12_1_QA_EVIDENCE)

    assert "status: In Progress" in parent_task
    assert PLAN.as_posix() in parent_task
    assert PLAN.as_posix() in tracker
    assert PHASE_5_QA_README.as_posix() in tracker
    assert "Status: TASK-12.1, TASK-12.2, TASK-12.3, TASK-12.4, and TASK-12.5 verified; closeout remains open" in qa_readme
    assert TASK_12_1_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_12_2_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_12_3_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_12_4_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_12_5_QA_EVIDENCE.as_posix() in qa_readme

    for required_section in (
        "## Source Of Truth",
        "## Current Dev Inventory",
        "## Phase 5 Child Task Plan",
        "## Risk Controls",
        "### Task 1: Phase 5.1 Server Parity Current-State Inventory",
        "### Task 6: Phase 5.6 Server Parity Live Integration Closeout",
    ):
        assert required_section in plan

    for current_anchor in (
        "RuntimeServerContextProvider",
        "EventStateRepository",
        "Sync_Interop",
        "server_parity_contracts.py",
        "backend-parity-phase-tracker.md",
    ):
        assert current_anchor in inventory_evidence

    for residual in (
        "ACP runtime launch",
        "Schedules and Workflows run-control services",
        "write sync remains deferred",
    ):
        assert residual in inventory_evidence

    for task_id, task_path in PHASE_5_CHILD_TASKS.items():
        task = _text(task_path)
        assert task_id in plan
        assert task_id in parent_task
        assert task_id in tracker
        assert "parent_task_id:" in task
        assert "TASK-12" in task
        assert "QA walkthrough" in task
        assert "focused regression" in task.lower()
        if task_id in {"TASK-12.1", "TASK-12.2", "TASK-12.3", "TASK-12.4", "TASK-12.5"}:
            assert "status: Done" in task
            for ac_number in range(1, 5):
                assert f"- [x] #{ac_number}" in task
            assert "## Implementation Notes" in task
        else:
            assert "status: To Do" in task
            for ac_number in range(1, 5):
                assert f"- [ ] #{ac_number}" in task

    phase_row = _markdown_table_row(tracker, "Phase 5: Server-Parity And Live Integrations")
    assert "in-progress; TASK-12.1, TASK-12.2, TASK-12.3, TASK-12.4, and TASK-12.5 verified" in phase_row[2]
    for task_id in PHASE_5_CHILD_TASKS:
        assert task_id in phase_row[3]
    assert TASK_12_1_QA_EVIDENCE.name in phase_row[4]
    assert TASK_12_2_QA_EVIDENCE.name in phase_row[4]
    assert TASK_12_3_QA_EVIDENCE.name in phase_row[4]
    assert TASK_12_4_QA_EVIDENCE.name in phase_row[4]
    assert TASK_12_5_QA_EVIDENCE.name in phase_row[4]
    assert PLAN.as_posix() in phase_row[4]
    assert "active-server/auth" in phase_row[5]
    assert "server event/feed" in phase_row[5]
    assert "Library/Search/RAG" in phase_row[5]
    assert "write sync" in phase_row[5]
