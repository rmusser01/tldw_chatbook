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
PROTOCOL = PHASE_1_ROOT / "walkthrough-protocol.md"
TEMPLATE = PHASE_1_ROOT / "walkthrough-template.md"
SMOKE = PHASE_1_ROOT / "2026-05-05-phase-1-1-harness-smoke.md"
BACKLOG_TASKS = Path("backlog/tasks")

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


def _task_text_containing(title_fragment: str) -> str:
    matches: list[str] = []
    for path in (REPO_ROOT / BACKLOG_TASKS).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if title_fragment in text:
            matches.append(text)
    assert len(matches) == 1, f"expected one task containing {title_fragment!r}, found {len(matches)}"
    return matches[0]


def _phase_row(markdown: str, phase_title: str) -> list[str]:
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if columns and columns[0] == phase_title:
            return columns
    raise AssertionError(f"{phase_title!r} row not found")


def test_product_maturity_phase_one_harness_files_exist() -> None:
    for path in (SPEC, TRACKER, BACKLOG_DOC, QA_ROOT / "README.md", PHASE_1_README, PROTOCOL, TEMPLATE, SMOKE):
        assert (REPO_ROOT / path).exists(), f"{path} should exist"


def test_product_maturity_template_captures_required_fields_and_severity() -> None:
    text = _text(TEMPLATE)

    for section in REQUIRED_TEMPLATE_SECTIONS:
        assert f"## {section}" in text
    for severity in REQUIRED_SEVERITIES:
        assert severity in text
    for priority in REQUIRED_PRIORITY_LABELS:
        assert priority in text
    assert "usable, not merely rendered" in text


def test_product_maturity_protocol_defines_clean_run_and_terminal_matrix() -> None:
    text = _text(PROTOCOL)

    assert "python3 -m tldw_chatbook.app" in text
    assert "Fresh HOME" in text
    assert "XDG_CONFIG_HOME" in text
    assert "XDG_DATA_HOME" in text
    assert "minimum supported compact" in text
    assert "common laptop terminal" in text
    assert "large power-user workspace" in text
    assert "render-only" in text
    assert "click-event-only" in text
    assert "Tests/UI/test_product_maturity_phase1_harness.py" in text


def test_product_maturity_tracker_links_phase_one_harness_and_tasks() -> None:
    tracker = _text(TRACKER)
    phase_one_task = _task_text_containing("Product Maturity Phase 1: QA Baseline And Usability Guardrails")
    phase_one_one_task = _task_text_containing("Product Maturity Phase 1.1: Canonical QA Harness")

    assert str(SPEC) in tracker
    assert str(PROTOCOL) in tracker
    assert str(TEMPLATE) in tracker
    assert str(SMOKE) in tracker
    assert "Phase 1.1" in tracker
    assert "<PHASE_" not in tracker

    phase_one_row = _phase_row(tracker, "Phase 1: QA Baseline And Usability Guardrails")
    assert phase_one_row[2] in {"planned", "in_progress"}
    assert "Phase 1.1" in phase_one_row[3]
    assert "TASK-" in phase_one_row[3]
    assert "phase-1/" in phase_one_row[4]

    assert "QA walkthrough verifies the running app is usable" in phase_one_task
    assert "Product-maturity QA protocol defines clean-run setup" in phase_one_one_task
    assert "Harness smoke evidence states" in phase_one_one_task


def test_phase_one_one_smoke_evidence_records_harness_only_boundary() -> None:
    text = _text(SMOKE)

    assert "Phase 1.1" in text
    assert "Canonical QA Harness" in text
    assert "Product QA Boundary" in text
    assert re.search(r"harness[- ]only", text, re.IGNORECASE)
    assert "does not complete the full product walkthrough" in text
    assert "Tests/UI/test_product_maturity_phase1_harness.py" in text
    assert "<PHASE_" not in text
