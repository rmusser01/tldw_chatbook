"""Post-release UX/HCI functional validation tracking regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = Path("Docs/superpowers/plans/2026-05-17-post-release-ux-hci-functional-validation.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
QA_README = Path("Docs/superpowers/qa/product-maturity/post-release-ux-hci/README.md")
QA_TEMPLATE = Path("Docs/superpowers/qa/product-maturity/post-release-ux-hci/walkthrough-template.md")
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

    for screen in REQUIRED_SCREENS:
        assert f"| {screen} | pending | pending | pending | `TASK-60.2` |" in readme

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

    assert "status: Done" in _text(CHILD_TASKS["TASK-60.1"])
    for task_id in ("TASK-60.2", "TASK-60.3", "TASK-60.4"):
        assert "status: To Do" in _text(CHILD_TASKS[task_id])

    assert "forbids SVG/code-layout substitutes" in _text(CHILD_TASKS["TASK-60.1"])
    assert "Home, Console, Library, Artifacts, Personas, Watchlists" in _text(
        CHILD_TASKS["TASK-60.2"]
    )
    assert "At least five power-user repeated workflows" in _text(CHILD_TASKS["TASK-60.3"])
    assert "ACP runtime launch, write sync promotion, Workspaces/Library depth" in _text(
        CHILD_TASKS["TASK-60.4"]
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
