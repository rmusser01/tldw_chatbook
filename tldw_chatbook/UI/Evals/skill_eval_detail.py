"""Run-group detail view for skill evals (ResultsGrid stays word-bench-only).

Read-only rendering of one skill-eval run group's stored report (``storage.
load_report``) plus an artifact count over the run's ``eval_results`` rows
(``storage.iter_artifacts``). Every line is a plain ``Static`` with
``markup=False``: report text (finding remediations, warning strings, skill
names) is free-text that must never be parsed as Rich markup -- the same
hazard class ``library_rail``/``evals_screen`` already guard against for
bench names and exception text.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from ...Evals.skill_eval.storage import iter_artifacts, load_report


def _bar(value: float, width: int = 10) -> str:
    """A single-width block bar for one dimension's blended score.

    ``▓``/``░`` (never emoji -- double-width in this app's terminal, a
    repeated past defect), clamped to [0, 1] so a malformed score renders a
    full or empty bar instead of raising.
    """
    filled = round(max(0.0, min(1.0, value)) * width)
    return "▓" * filled + "░" * (width - filled)


class SkillEvalDetail(Widget):
    """Header (composite/grade/confidence/methodology/provenance), one line
    per dimension, findings with remediation, warnings, artifact count."""

    DEFAULT_CSS = """
    SkillEvalDetail { padding: 1; }
    """

    def __init__(self, view_model: Any, run_group_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view_model = view_model
        self._run_group_id = run_group_id

    def compose(self) -> ComposeResult:
        db = getattr(self._view_model, "db", None)
        report = load_report(db, self._run_group_id) if db else None
        with VerticalScroll():
            if report is None:
                yield Static(
                    "No skill eval report found for this run group.",
                    markup=False,
                )
                return
            yield Static(
                f"{report['composite']:.1f}  {report['grade']}  "
                f"({report['confidence']}, {report['depth']}, "
                f"{report['methodology_version']})",
                markup=False,
            )
            prov = report.get("provenance") or {}
            yield Static(
                f"Subject: {prov.get('name')} · trust={prov.get('trust_status')} "
                f"· digest={str(prov.get('digest', ''))[:12]}…",
                markup=False,
            )
            for dim in report.get("dimensions", []):
                yield Static(
                    f"{dim['name']:<24} {_bar(dim['blended'])} "
                    f"{dim['blended']:.2f}  ({'+'.join(dim['available_layers'])})",
                    markup=False,
                )
            for finding in report.get("findings", []):
                yield Static(
                    f"finding: {finding['code']} — {finding['remediation']}",
                    markup=False,
                )
            for warning in report.get("warnings", []):
                yield Static(f"warning: {warning}", markup=False)
            if db:
                run_rows = db.list_runs(run_group_id=self._run_group_id)
                count = sum(
                    len(list(iter_artifacts(db, r["id"]))) for r in run_rows
                )
                yield Static(f"artifacts: {count}", markup=False)
