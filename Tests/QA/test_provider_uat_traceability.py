"""Provider UAT evidence traceability regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACEABILITY_REPORT = Path(
    "Docs/superpowers/qa/provider-cdp-uat/2026-06-16-provider-uat-traceability.md"
)
TASK_123 = Path("backlog/tasks/task-123 - Restore-provider-UAT-evidence-traceability.md")


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_provider_uat_traceability_report_exists_and_discloses_gap() -> None:
    report = REPO_ROOT / TRACEABILITY_REPORT

    assert report.exists()

    text = _text(TRACEABILITY_REPORT)
    assert "TASK-84" in text
    assert "missing from current dev" in text
    assert "not reconstructed" in text
    assert "PR #527" in text


def test_provider_uat_traceability_task_points_to_existing_report() -> None:
    assert (REPO_ROOT / TASK_123).exists()
    task = _text(TASK_123)

    assert TRACEABILITY_REPORT.as_posix() in task
    assert (REPO_ROOT / TRACEABILITY_REPORT).exists()
