"""Product maturity Phase 1.1 QA harness contract."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = Path("Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
BACKLOG_DOC = Path("backlog/docs/product-maturity-roadmap.md")
QA_ROOT = Path("Docs/superpowers/qa/product-maturity")
PHASE_1_ROOT = QA_ROOT / "phase-1"
PHASE_1_README = PHASE_1_ROOT / "README.md"
PHASE_2_ROOT = QA_ROOT / "phase-2"
PHASE_2_README = PHASE_2_ROOT / "README.md"
PROTOCOL = PHASE_1_ROOT / "walkthrough-protocol.md"
TEMPLATE = PHASE_1_ROOT / "walkthrough-template.md"
SMOKE = PHASE_1_ROOT / "2026-05-05-phase-1-1-harness-smoke.md"
PHASE_2_3_EVIDENCE = PHASE_2_ROOT / "2026-05-05-phase-2-3-saved-chatbook-artifact-reopen-contract.md"
PHASE_2_4_EVIDENCE = PHASE_2_ROOT / "2026-05-05-phase-2-4-home-chatbook-artifact-resume-contract.md"
PHASE_2_5_EVIDENCE = PHASE_2_ROOT / "2026-05-06-phase-2-5-core-loop-closeout-replay.md"
PHASE_1_2_PLAN = Path("Docs/superpowers/plans/2026-05-05-product-maturity-phase-1-2-first-run-walkthrough.md")
BACKLOG_TASKS = Path("backlog/tasks")

TASK_ID_PATTERN = r"TASK-[0-9]+(?:\.[0-9]+)*"
TASK_ID_RE = re.compile(rf"`({TASK_ID_PATTERN})`")
FRONTMATTER_RE = re.compile(r"\A\ufeff?\s*---\n(?P<body>.*?)\n---\n", re.DOTALL)
TASK_FRONTMATTER_ID_RE = re.compile(
    rf"^id:\s*['\"]?({TASK_ID_PATTERN})['\"]?\s*(?:#.*)?$",
    re.MULTILINE,
)

REQUIRED_TEMPLATE_SECTIONS = {
    "Environment",
    "Task Or Phase",
    "Entry Path",
    "Terminal Size",
    "Clean-Run Setup",
    "Steps Attempted",
    "Visual/Focus Notes",
    "Keyboard Path Result",
    "Mouse/Click Path Result",
    "Functional Result",
    "Defects Found",
    "Evidence",
    "Residual Risk",
    "Exit Decision",
    "Product QA Boundary",
}

REQUIRED_SEVERITIES = {
    "blocker",
    "workflow-degradation",
    "recoverability",
    "polish",
}

REQUIRED_PRIORITY_LABELS = {"P0", "P1", "P2", "P3"}


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _task_ids_from_phase_row(row: list[str]) -> set[str]:
    assert len(row) > 3, "phase row should include a task column"
    return set(TASK_ID_RE.findall(row[3]))


def _task_text_by_id(task_id: str) -> str:
    for path in sorted((REPO_ROOT / BACKLOG_TASKS).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        frontmatter_match = FRONTMATTER_RE.match(text)
        if frontmatter_match and task_id in TASK_FRONTMATTER_ID_RE.findall(frontmatter_match.group("body")):
            return text
    raise AssertionError(f"task {task_id!r} not found by YAML frontmatter id")


def _task_ids_by_path() -> dict[str, list[Path]]:
    task_ids: dict[str, list[Path]] = {}

    for path in sorted((REPO_ROOT / BACKLOG_TASKS).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        frontmatter_match = FRONTMATTER_RE.match(text)
        assert frontmatter_match is not None, f"{path.relative_to(REPO_ROOT)} must start with YAML frontmatter"

        parsed_ids = TASK_FRONTMATTER_ID_RE.findall(frontmatter_match.group("body"))
        assert len(parsed_ids) == 1, (
            f"{path.relative_to(REPO_ROOT)} must contain exactly one valid TASK-* id in frontmatter; "
            f"found {parsed_ids!r}"
        )

        task_ids.setdefault(parsed_ids[0], []).append(path.relative_to(REPO_ROOT))

    return task_ids


def _phase_row(markdown: str, phase_title: str) -> list[str]:
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if columns and columns[0] == phase_title:
            return columns
    raise AssertionError(f"{phase_title!r} row not found")


def test_backlog_task_frontmatter_ids_are_unique() -> None:
    task_ids = _task_ids_by_path()
    duplicates = {
        task_id: paths
        for task_id, paths in task_ids.items()
        if len(paths) > 1
    }

    assert duplicates == {}


def test_product_maturity_tracker_links_phase_one_harness_and_tasks() -> None:
    tracker = _text(TRACKER)

    assert str(SPEC) in tracker
    assert str(PROTOCOL) in tracker
    assert str(TEMPLATE) in tracker
    assert str(SMOKE) in tracker
    assert str(PHASE_1_2_PLAN) in tracker
    assert "Phase 1.1" in tracker
    assert "Phase 1.2" in tracker
    assert "<PHASE_" not in tracker

    phase_one_row = _phase_row(tracker, "Phase 1: QA Baseline And Usability Guardrails")
    phase_one_task_ids = _task_ids_from_phase_row(phase_one_row)
    assert len(phase_one_task_ids) >= 2
    phase_one_parent_ids = [task_id for task_id in phase_one_task_ids if "." not in task_id]
    phase_one_child_ids = [task_id for task_id in phase_one_task_ids if "." in task_id]
    assert len(phase_one_parent_ids) == 1
    assert phase_one_child_ids
    phase_one_task_id = phase_one_parent_ids[0]
    phase_one_task = _task_text_by_id(phase_one_task_id)
    phase_one_child_tasks = [_task_text_by_id(task_id) for task_id in phase_one_child_ids]

    assert phase_one_row[2] in {"planned", "in_progress", "verified"}
    assert "Phase 1.1" in phase_one_row[3]
    assert "Phase 1.2" in phase_one_row[3]
    assert "Phase 1.7" in phase_one_row[3]
    assert "TASK-" in phase_one_row[3]
    assert "phase-1/" in phase_one_row[4]

    assert "QA walkthrough verifies the running app is usable" in phase_one_task
    assert any("Product-maturity QA protocol defines clean-run setup" in task for task in phase_one_child_tasks)
    assert any("Harness smoke evidence states" in task for task in phase_one_child_tasks)
    assert any("clean first-run launch" in task.lower() for task in phase_one_child_tasks)
    assert any("Narrow Core Loop Proof" in task for task in phase_one_child_tasks)


