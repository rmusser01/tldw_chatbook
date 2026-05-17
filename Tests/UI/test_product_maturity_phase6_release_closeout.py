"""Product maturity Phase 6.7 release closeout and public roadmap alignment."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROADMAP = Path("Docs/Product_Roadmap.md")
RECOVERY_DOC = Path("Docs/Development/release-recovery-setup.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_6_README = Path("Docs/superpowers/qa/product-maturity/phase-6/README.md")
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-7-release-closeout.md"
)
TASK_13 = Path(
    "backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md"
)
TASK_13_7 = Path(
    "backlog/tasks/task-13.7 - Phase-6.7-Public-roadmap-release-closeout.md"
)
CHILD_TASKS = {
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
    "TASK-13.7": TASK_13_7,
}
REQUIRED_PHASE_6_EVIDENCE = (
    "Docs/superpowers/plans/2026-05-16-phase-6-release-hardening.md",
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-2-first-time-user-release-replay.md",
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-3-power-user-workflow-release-replay.md",
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-4-keyboard-focus-accessibility-visual-sweep.md",
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-5-recovery-setup-docs-alignment.md",
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-6-packaging-config-data-safety.md",
    EVIDENCE.as_posix(),
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_7_RELEASE_CLOSEOUT_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_7_RELEASE_CLOSEOUT_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _markdown_table_row(markdown: str, first_cell_text: str) -> list[str]:
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or first_cell_text not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == first_cell_text:
            return cells
    raise AssertionError(f"Missing markdown table row for {first_cell_text!r}")


def _assert_task_done(task: str, task_path: Path) -> None:
    text = _text(task_path)
    assert "status: Done" in text, task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in text, task
    assert "## Implementation Notes" in text, task


def test_phase6_release_closeout_evidence_and_tracking_are_complete() -> None:
    evidence = _text(EVIDENCE)
    phase_6_readme = _text(PHASE_6_README)
    tracker = _text(TRACKER)
    metadata = _metadata(evidence)

    assert metadata["task"] == "TASK-13.7"
    assert metadata["parent_task"] == "TASK-13"
    assert metadata["decision"] == "release_closeout_recorded"
    assert metadata["phase6_status"] == "verified"
    assert metadata["p0_p1_findings"] == []
    assert set(metadata["public_docs_reviewed"]) == {
        PUBLIC_ROADMAP.as_posix(),
        RECOVERY_DOC.as_posix(),
        PHASE_6_README.as_posix(),
    }
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0

    for section in (
        "## Environment",
        "## Evidence Completeness",
        "## Public Roadmap Review",
        "## Release Closeout Decision",
        "## P0/P1 Decision",
        "## Residual Risk",
        "## Verification",
    ):
        assert section in evidence

    for required_path in REQUIRED_PHASE_6_EVIDENCE:
        assert required_path in evidence
        assert required_path in phase_6_readme or required_path.endswith(
            "2026-05-16-phase-6-7-release-closeout.md"
        )
    assert "Status: TASK-13.1 through TASK-13.7 done; Phase 6 verified" in phase_6_readme
    assert EVIDENCE.as_posix() in phase_6_readme

    qa_row = _markdown_table_row(tracker, "Phase 6 QA index")
    assert qa_row[2] == "verified; TASK-13.1 through TASK-13.7 done"
    phase_row = _markdown_table_row(tracker, "Phase 6: Release Hardening And Documentation")
    assert phase_row[2] == "verified; TASK-13.1 through TASK-13.7 done"
    assert EVIDENCE.name in phase_row[4]
    assert "release hardening complete" in phase_row[5].lower()
    assert "Phase 6 verified" in tracker

    _assert_task_done("TASK-13", TASK_13)
    for task_id, task_path in CHILD_TASKS.items():
        _assert_task_done(task_id, task_path)


def test_public_roadmap_matches_release_closeout_without_internal_commitments() -> None:
    roadmap = _text(PUBLIC_ROADMAP)
    recovery = _text(RECOVERY_DOC)

    for required_heading in (
        "## Current Release Baseline",
        "## Current Limits And Recovery",
        "## Now: Reliability And Product Confidence",
        "## Next: Complete Workflow Loops",
        "## Later: Server-Backed And Live Capabilities",
        "## Always: Local-First Control",
    ):
        assert required_heading in roadmap

    for required_concept in (
        "Home is the default status and notification surface",
        "Console is the live agentic control surface",
        "Library is the source, Search/RAG, import/export, and Collections surface",
        "Chatbooks and other durable outputs live under Artifacts",
        "ACP runtime launch and write sync remain future work",
    ):
        assert required_concept in roadmap

    forbidden_patterns = (
        r"\bETA\b",
        r"\bdeadline\b",
        r"\bwill\s+ship\s+on\b",
        r"\bTASK-",
        r"\bPhase\s+\d",
        r"\b2026-\d{2}-\d{2}\b",
    )
    assert not any(re.search(pattern, roadmap, flags=re.IGNORECASE) for pattern in forbidden_patterns)
    assert "release-candidate recovery reference" in recovery
