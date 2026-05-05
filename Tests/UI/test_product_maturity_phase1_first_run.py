"""Product maturity Phase 1.2 first-run walkthrough contract."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-2-first-run-walkthrough.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md")


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_phase_one_two_evidence_records_clean_first_run_walkthrough() -> None:
    evidence = _text(EVIDENCE)

    assert "/Users/" not in evidence
    for required_text in (
        "## Clean-Run Setup",
        "Fresh HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "running Textual app",
        "Home",
        "Console",
        "Library",
        "Settings",
        "usable, not merely rendered",
        "TASK-8.2",
    ):
        assert required_text in evidence


def test_phase_one_two_tracking_and_task_closeout_are_current() -> None:
    tracker = _text(TRACKER)
    readme = _text(PHASE_1_README)
    task = _text(TASK)

    assert "Phase 1.2" in tracker
    assert "TASK-8.2" in tracker
    assert EVIDENCE.name in tracker
    assert EVIDENCE.name in readme
    assert "Phase 1.2 clean first-run status: verified" in readme
    assert "status: Done" in task
    for acceptance_criterion in range(1, 6):
        assert f"- [x] #{acceptance_criterion}" in task
    assert "Implementation Notes" in task
