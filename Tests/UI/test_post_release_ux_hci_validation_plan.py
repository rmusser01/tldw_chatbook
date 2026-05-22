"""Post-release UX/HCI functional validation tracking regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = Path("Docs/superpowers/plans/2026-05-17-post-release-ux-hci-functional-validation.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
QA_README = Path("Docs/superpowers/qa/product-maturity/post-release-ux-hci/README.md")
QA_TEMPLATE = Path("Docs/superpowers/qa/product-maturity/post-release-ux-hci/walkthrough-template.md")
WORKFLOW_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/post-release-ux-hci/2026-05-22-cross-screen-workflow-validation.md"
)
TASK_60 = Path("backlog/tasks/task-60 - Post-release-UX-HCI-and-functionality-validation-tranche.md")
CHILD_TASKS = {
    "TASK-60.1": Path(
        "backlog/tasks/task-60.1 - Post-release-actual-screen-UX-HCI-audit-harness.md"
    ),
    "TASK-60.2": Path(
        "backlog/tasks/task-60.2 - Post-release-top-level-screen-functionality-audit.md"
    ),
    "TASK-60.3": Path(
        "backlog/tasks/task-60.3 - Post-release-cross-screen-workflow-validation.md"
    ),
    "TASK-60.4": Path(
        "backlog/tasks/task-60.4 - Post-release-deferred-feature-tranche-planning.md"
    ),
    "TASK-60.5": Path(
        "backlog/tasks/task-60.5 - Fix-Personas-destination-indefinite-behavior-context-loading-state.md"
    ),
    "TASK-60.6": Path(
        "backlog/tasks/task-60.6 - Fix-Watchlists-destination-indefinite-local-snapshot-loading-state.md"
    ),
}
REQUIRED_SCREENS = (
    "Home",
    "Console",
    "Library",
    "Artifacts",
    "Personas",
    "Watchlists",
    "Schedules",
    "Workflows",
    "MCP",
    "ACP",
    "Skills",
    "Settings",
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _markdown_table_columns(row: str) -> list[str]:
    return [column.strip() for column in row.strip().strip("|").split("|")]


def test_post_release_validation_plan_requires_actual_app_use() -> None:
    plan = _text(PLAN)

    for required in (
        "Re-audit the rendered app as an actual user",
        "textual-web/CDP",
        "actual terminal screenshots",
        "actual-use validation",
        "P0/P1/P2/P3",
        "Do not prioritize deferred feature implementation ahead of unresolved P0/P1",
    ):
        assert required in plan

    for screen in REQUIRED_SCREENS:
        assert screen in plan

    for workflow in (
        "Home active work",
        "Library Search/RAG",
        "Console can accept visible composer input",
        "Artifacts and Chatbooks",
        "Personas, Skills, MCP, ACP, Watchlists, Schedules, and Workflows",
    ):
        assert workflow in plan


def test_post_release_qa_harness_requires_real_screenshots_and_approval() -> None:
    readme = _text(QA_README)
    template = _text(QA_TEMPLATE)

    for required in (
        "Actual screenshots are required",
        "Do not use generated SVGs",
        "textual-web with CDP/browser automation",
        "actual terminal screenshot",
        "A screen that renders but has dead controls",
        "P0/P1 findings require Backlog follow-up tasks",
    ):
        assert required in readme

    table_rows = [_markdown_table_columns(row) for row in readme.splitlines()]
    for screen in REQUIRED_SCREENS:
        matching_rows = [row for row in table_rows if row and row[0] == screen]
        assert len(matching_rows) == 1, (
            f"{screen} must have exactly one row in the post-release QA index"
        )
        columns = matching_rows[0]
        assert len(columns) >= 5, f"{screen} row must include all QA index columns"
        assert columns[2] in {"pending", "approved"}, (
            f"{screen} row must track screenshot approval explicitly"
        )

    for required_field in (
        "Evidence method:",
        "Actual screenshot path:",
        "Screenshot approval:",
        "Nielsen Norman Heuristic Findings",
        "Keyboard And Focus Findings",
        "Cross-Screen Handoff Findings",
        "Power-User Repetition Findings",
        "Severity Decisions",
        "Accepted: no",
    ):
        assert required_field in template


def test_post_release_backlog_tasks_track_screens_workflows_and_deferred_features() -> None:
    parent = _text(TASK_60)

    assert "status: To Do" in parent
    assert "actual rendered screenshot audit" in parent
    assert "Cross-screen workflows are validated end-to-end" in parent
    assert "No screen is marked accepted without actual screenshot approval" in parent

    for task_id, path in CHILD_TASKS.items():
        task = _text(path)
        assert f"id: {task_id}" in task
        assert "parent_task_id: TASK-60" in task
        assert "<!-- AC:BEGIN -->" in task

    expected_statuses = {
        "TASK-60.1": "Done",
        "TASK-60.2": "Done",
        "TASK-60.3": "Done",
        "TASK-60.4": "To Do",
        "TASK-60.5": "Done",
        "TASK-60.6": "Done",
    }
    for task_id, expected_status in expected_statuses.items():
        assert f"status: {expected_status}" in _text(CHILD_TASKS[task_id])

    assert "forbids SVG/code-layout substitutes" in _text(CHILD_TASKS["TASK-60.1"])
    assert "Home, Console, Library, Artifacts, Personas, Watchlists" in _text(
        CHILD_TASKS["TASK-60.2"]
    )
    assert "At least five power-user repeated workflows" in _text(CHILD_TASKS["TASK-60.3"])
    assert "ACP runtime launch, write sync promotion, Workspaces/Library depth" in _text(
        CHILD_TASKS["TASK-60.4"]
    )
    assert "Personas screen leaves loading state deterministically" in _text(
        CHILD_TASKS["TASK-60.5"]
    )
    assert "Watchlists screen leaves loading state deterministically" in _text(
        CHILD_TASKS["TASK-60.6"]
    )


def test_product_maturity_tracker_lists_post_release_validation_tranche() -> None:
    tracker = _text(TRACKER)

    assert "Post-Release UX/HCI Functional Validation" in tracker
    assert "TASK-60" in tracker
    for task_id in CHILD_TASKS:
        assert task_id in tracker
    assert "actual screenshots" in tracker
    assert "actual-use functionality evidence" in tracker
    assert "cross-screen workflow validation" in tracker


def test_post_release_cross_screen_workflow_evidence_records_verification() -> None:
    evidence = _text(WORKFLOW_EVIDENCE)
    readme = _text(QA_README)

    for required in (
        "TASK-60.3",
        "Workflow Matrix",
        "Home primary action opens target route",
        "Library Search/RAG mode and source handoff",
        "Artifacts/Chatbook resume to Console",
        "Personas and Skills attach to Console",
        "Watchlists, Schedules, Workflows run follow",
        "No unresolved P0/P1 findings remain",
        "Result: 5 passed",
        "Result: 8 passed",
    ):
        assert required in evidence

    assert "2026-05-22-cross-screen-workflow-validation.md" in readme
