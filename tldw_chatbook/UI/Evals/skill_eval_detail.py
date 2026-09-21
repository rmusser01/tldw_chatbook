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
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from ...Evals.skill_eval.storage import iter_artifacts, load_report


def _bar(value: float, width: int = 10) -> str:
    """A single-width block bar for one dimension's blended score.

    ``▓``/``░`` (never emoji -- double-width in this app's terminal, a
    repeated past defect), clamped to [0, 1] so a malformed score renders a
    full or empty bar instead of raising.
    """
    filled = round(max(0.0, min(1.0, value)) * width)
    return "▓" * filled + "░" * (width - filled)


def _ci_suffix(ci: Any) -> str:
    """``" (CI lo–hi)"`` for a persisted confidence interval, else ``""``.

    Tolerates the tuple→list shape change the JSON round-trip through
    ``config_overrides`` imposes, and any missing/malformed entry (an
    optional stat never reported should render nothing, not crash the
    whole detail view).
    """
    try:
        lo, hi = ci
        return f" (CI {float(lo):.2f}–{float(hi):.2f})"
    except (TypeError, ValueError):
        return ""


class SkillEvalDetail(Widget):
    """Header (composite/grade/confidence/methodology/provenance), one line
    per dimension, layer statistics, findings with remediation, warnings,
    artifact count."""

    DEFAULT_CSS = """
    SkillEvalDetail { padding: 1; }
    """

    class RunAgainRequested(Message, namespace="skill_eval_detail"):
        """TASK-32889: "Run again" on a completed report -- the panel that
        configured this run was swapped out for the report, so without
        this affordance a rerun meant re-selecting the bench in the rail
        to remount the launch panel. Carries the owning bench's id (the
        run rows' task_id), resolved at press time from the same db the
        report was read from."""

        def __init__(self, bench_id: str) -> None:
            super().__init__()
            self.bench_id = bench_id

    def __init__(self, view_model: Any, run_group_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view_model = view_model
        self._run_group_id = run_group_id

    def _bench_id_for_group(self) -> str:
        """The owning bench id: the first run row's task_id, or ''."""
        db = getattr(self._view_model, "db", None)
        if db is None:
            return ""
        try:
            rows = db.list_runs(run_group_id=self._run_group_id)
        except Exception:
            return ""
        for row in rows:
            task_id = row.get("task_id")
            if task_id:
                return str(task_id)
        return ""

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
            yield from self._compose_layer_statistics(
                report.get("layer_summaries") or {}
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
            # TASK-32889: rerunning after a completed report used to
            # require re-selecting the bench in the rail to remount the
            # launch panel. Only offered when the owning bench resolves.
            if self._bench_id_for_group():
                yield Button("Run again", id="skill-eval-run-again",
                             compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skill-eval-run-again":
            bench_id = self._bench_id_for_group()
            if bench_id:
                self.post_message(self.RunAgainRequested(bench_id))

    def _compose_layer_statistics(self, layer_summaries: dict) -> ComposeResult:
        """The persisted per-layer stats the report snapshot carries.

        Final-review Important 6: the snapshot has always stored judge
        trigger F1/precision/recall and the simulation's activation/
        consistency/failure rates with their CIs, but the detail view
        rendered only an artifact count -- the numbers the confidence label
        rests on were invisible. Each block renders only when the layer was
        usable (``None`` blocks are skipped, matching how ``build_report``
        nulls an unusable layer's summary); individual missing stats (a
        ``None`` trigger F1 or CI) drop their own line rather than the
        whole block.
        """
        judge = layer_summaries.get("judge")
        sim = layer_summaries.get("simulation")
        if not judge and not sim:
            return
        yield Static("Layer statistics", markup=False)
        if judge:
            rubrics = judge.get("rubrics") or {}
            if rubrics:
                yield Static(
                    "  judge rubrics: "
                    + ", ".join(f"{k}={float(v):.2f}"
                                for k, v in sorted(rubrics.items())),
                    markup=False,
                )
            f1 = judge.get("trigger_f1")
            if f1 is not None:
                line = f"  judge trigger: F1={float(f1):.2f}"
                precision = judge.get("trigger_precision")
                if precision is not None:
                    line += f" · precision={float(precision):.2f}"
                recall = judge.get("trigger_recall")
                if recall is not None:
                    line += f" · recall={float(recall):.2f}"
                yield Static(line, markup=False)
        if sim:
            activation = sim.get("activation")
            if activation is not None:
                yield Static(
                    f"  sim activation: {float(activation):.2f}"
                    f"{_ci_suffix(sim.get('activation_ci'))}",
                    markup=False,
                )
            consistency = sim.get("consistency")
            if consistency is not None:
                yield Static(
                    f"  sim consistency: {float(consistency):.2f}"
                    f"{_ci_suffix(sim.get('consistency_ci'))}",
                    markup=False,
                )
            failure_rate = sim.get("failure_rate")
            if failure_rate is not None:
                yield Static(
                    f"  sim failure rate: {float(failure_rate):.2f}"
                    f"{_ci_suffix(sim.get('failure_ci'))}",
                    markup=False,
                )
