"""Phase 4 agent configuration and execution planning regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PLAN = Path("Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md")
PHASE_4_QA_README = Path("Docs/superpowers/qa/product-maturity/phase-4/README.md")
TASK_11_2_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-2-personas-runtime-launch.md"
)
TASK_11_2_SCREENSHOT = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/personas-selected-2026-05-12.png"
)
TASK_11_2_APPROVED_SCREENSHOT = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/personas-selected-polish-2026-05-12.png"
)
TASK_11_3_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-3-skills-attach-validation.md"
)
TASK_11_3_SCREENSHOT = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/skills-valid-invalid-2026-05-12.png"
)
TASK_11_4_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-4-mcp-source-scope.md"
)
TASK_11_4_SCREENSHOT = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/mcp-source-scope-final-real-viewport-2026-05-14.png"
)
TASK_11_5_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-5-acp-runtime-session.md"
)
TASK_11_5_SCREENSHOT = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/acp-runtime-session-2026-05-14.png"
)
TASK_11_6_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/2026-05-15-phase-4-6-schedules-workflows-run-control.md"
)
TASK_11_6_SCHEDULES_SCREENSHOT = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/schedules-run-control-2026-05-15-polish.png"
)
TASK_11_6_WORKFLOWS_SCREENSHOT = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/workflows-run-control-2026-05-15-polish.png"
)
TASK_11_7_QA_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-4/2026-05-16-phase-4-7-agent-execution-closeout.md"
)
TASK_11 = Path(
    "backlog/tasks/task-11 - Product-Maturity-Phase-4-Agent-Configuration-And-Execution.md"
)

PHASE_4_CHILD_TASKS = {
    "TASK-11.1": Path(
        "backlog/tasks/task-11.1 - Phase-4.1-Agent-execution-baseline-and-contracts.md"
    ),
    "TASK-11.2": Path(
        "backlog/tasks/task-11.2 - Phase-4.2-Personas-runtime-launch-and-Console-context.md"
    ),
    "TASK-11.3": Path(
        "backlog/tasks/task-11.3 - Phase-4.3-Skills-attach-validation-and-local-execution-contract.md"
    ),
    "TASK-11.4": Path(
        "backlog/tasks/task-11.4 - Phase-4.4-MCP-source-scope-and-action-readiness.md"
    ),
    "TASK-11.5": Path(
        "backlog/tasks/task-11.5 - Phase-4.5-ACP-runtime-session-contract.md"
    ),
    "TASK-11.6": Path(
        "backlog/tasks/task-11.6 - Phase-4.6-Schedules-and-Workflows-run-control.md"
    ),
    "TASK-11.7": Path(
        "backlog/tasks/task-11.7 - Phase-4.7-Agent-execution-QA-closeout.md"
    ),
}
PHASE_4_COMPLETED_TASKS = {
    "TASK-11.1",
    "TASK-11.2",
    "TASK-11.3",
    "TASK-11.4",
    "TASK-11.5",
    "TASK-11.6",
    "TASK-11.7",
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


def test_phase4_agent_execution_plan_splits_parent_into_reviewable_child_tasks() -> None:
    plan = _text(PLAN)
    parent_task = _text(TASK_11)
    tracker = _text(TRACKER)
    qa_readme = _text(PHASE_4_QA_README)
    personas_evidence = _text(TASK_11_2_QA_EVIDENCE)
    skills_evidence = _text(TASK_11_3_QA_EVIDENCE)
    mcp_evidence = _text(TASK_11_4_QA_EVIDENCE)
    acp_evidence = _text(TASK_11_5_QA_EVIDENCE)
    run_control_evidence = _text(TASK_11_6_QA_EVIDENCE)
    closeout_evidence = _text(TASK_11_7_QA_EVIDENCE)

    assert "status: Done" in parent_task
    assert PLAN.as_posix() in parent_task
    assert PLAN.as_posix() in tracker
    assert PHASE_4_QA_README.as_posix() in tracker
    assert "Status: Phase 4 verified; TASK-11.1 through TASK-11.7 complete" in qa_readme
    assert TASK_11_2_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_11_3_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_11_4_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_11_5_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_11_6_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_11_7_QA_EVIDENCE.as_posix() in qa_readme
    assert TASK_11_2_SCREENSHOT.as_posix() in personas_evidence
    assert TASK_11_2_APPROVED_SCREENSHOT.as_posix() in personas_evidence
    assert TASK_11_3_SCREENSHOT.as_posix() in skills_evidence
    assert TASK_11_4_SCREENSHOT.as_posix() in mcp_evidence
    assert TASK_11_5_SCREENSHOT.as_posix() in acp_evidence
    assert TASK_11_6_SCHEDULES_SCREENSHOT.as_posix() in run_control_evidence
    assert TASK_11_6_WORKFLOWS_SCREENSHOT.as_posix() in run_control_evidence
    assert "User approval: approved" in skills_evidence
    assert "User approval: approved" in acp_evidence
    assert "User approval: approved" in run_control_evidence
    assert "valid SKILL.md" in skills_evidence
    assert "invalid SKILL.md" in skills_evidence
    assert "ACP runtime/session state" in acp_evidence
    assert "Schedules failed-run recovery" in run_control_evidence
    assert "Workflows pending-approval recovery" in run_control_evidence
    assert "## Workflow Matrix" in closeout_evidence
    assert "Personas" in closeout_evidence
    assert "Skills" in closeout_evidence
    assert "MCP" in closeout_evidence
    assert "ACP" in closeout_evidence
    assert "Schedules" in closeout_evidence
    assert "Workflows" in closeout_evidence
    assert "P1 accepted residual for Phase 5" in closeout_evidence
    assert (REPO_ROOT / TASK_11_2_SCREENSHOT).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (REPO_ROOT / TASK_11_2_APPROVED_SCREENSHOT).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (REPO_ROOT / TASK_11_3_SCREENSHOT).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (REPO_ROOT / TASK_11_4_SCREENSHOT).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (REPO_ROOT / TASK_11_5_SCREENSHOT).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (REPO_ROOT / TASK_11_6_SCHEDULES_SCREENSHOT).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (REPO_ROOT / TASK_11_6_WORKFLOWS_SCREENSHOT).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    for required_section in (
        "## Source Of Truth",
        "## Scope",
        "## File Structure",
        "## Risk Controls",
        "### Task 1: Phase 4.1 Agent Execution Baseline And Contracts",
        "### Task 7: Phase 4.7 Agent Execution QA Closeout",
    ):
        assert required_section in plan

    for task_id, task_path in PHASE_4_CHILD_TASKS.items():
        task = _text(task_path)
        assert task_id in plan
        assert task_id in parent_task
        assert task_id in tracker
        if task_id in PHASE_4_COMPLETED_TASKS:
            assert "status: Done" in task
            for ac_number in range(1, 5):
                assert f"- [x] #{ac_number}" in task
            assert "## Implementation Notes" in task
        else:
            assert "status: To Do" in task
        assert "parent_task_id:" in task
        assert "TASK-11" in task
        assert "QA walkthrough" in task
        assert "focused regression" in task.lower()

    phase_row = _markdown_table_row(tracker, "Phase 4: Agent Configuration And Execution")
    assert "verified; TASK-11.1 through TASK-11.7 done" in phase_row[2]
    for task_id in PHASE_4_CHILD_TASKS:
        assert task_id in phase_row[3]
    assert "phase-4/" in phase_row[4]
    assert TASK_11_2_QA_EVIDENCE.name in phase_row[4]
    assert TASK_11_3_QA_EVIDENCE.name in phase_row[4]
    assert TASK_11_4_QA_EVIDENCE.name in phase_row[4]
    assert TASK_11_5_QA_EVIDENCE.name in phase_row[4]
    assert TASK_11_6_QA_EVIDENCE.name in phase_row[4]
    assert TASK_11_7_QA_EVIDENCE.name in phase_row[4]
    assert "ACP runtime" in phase_row[5]
    assert "Schedules and Workflows" in phase_row[5]
    assert "server parity" in phase_row[5]
