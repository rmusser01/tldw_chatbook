"""Readiness inspector: per-target readiness, recovery callouts, and a
call/time estimate for a selected word bench.

Mounted by ``evals_screen.py``'s ``_compose_inspector_pane`` above the
existing ``#evals-primary-action`` button (unchanged from Task 3 -- this
widget only adds content, it does not own the run control). Readiness
renders from a ``preflight`` map (``word_bench.storage.load_run_preflight``,
via ``EvalsViewModel.preflight_for_bench``) resolved ONCE per selection by
``evals_screen.py`` and passed into ``__init__`` -- see this class's own
``__init__`` and ``bench_editor.py``'s identical parameter -- rather than
this widget resolving it itself, so a bench selection does not read the
same run-group snapshot twice. Never calls a provider -- see
``bench_editor.py``'s module docstring for the shared "never imports the
runner" guarantee both widgets carry.

Per the design contract, ``.ds-status-badge`` colour lives in app-tier CSS
(``css/features/_evals.tcss``), never in this widget's own CSS -- there is
no ``DEFAULT_CSS`` here at all, deliberately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ...config import LOCAL_PROVIDERS
from ...Evals.word_bench import analysis
from ...Evals.word_bench.models import CellError, PreflightResult
from ...Evals.word_bench.storage import load_bench
from .evals_state import EvalsViewModel
from .results_grid import render_probe_reading, render_token

if TYPE_CHECKING:
    from .results_grid import ResultsGrid

#: Case-insensitive: `eval_models.provider` values in this codebase's own
#: fixtures and the design spec's examples are lowercase ("llama_cpp")
#: while `LOCAL_PROVIDERS`' keys mix case ("Ollama", "koboldcpp").
_LOCAL_PROVIDER_NAMES = {name.lower() for name in LOCAL_PROVIDERS}

#: There is no real timing data before a bench has ever run (nothing here
#: calls a provider -- see the module docstring). This is a rough,
#: explicitly-labelled placeholder used only to render a human-scale
#: duration, not a measurement.
_ASSUMED_SECONDS_PER_CALL = 0.4

#: State -> (problem, next action). Mirrors the design spec's Preflight
#: table (`2026-07-25-evals-console-rebuild-design.md`); an unlisted state
#: falls back to the target's own `detail` string plus a generic next step.
_BLOCKED_COPY: dict[str, tuple[str, str]] = {
    "unreachable": (
        "endpoint could not be reached.",
        "Check that the target's server is running and reachable, then retry.",
    ),
    "no_logprobs": (
        "responded but did not return logprobs.",
        "This provider cannot report logprobs for a word bench; choose a different target.",
    ),
    "mode_unsupported": (
        "does not support this bench's prompt mode.",
        "Switch the bench to the other prompt mode, or choose a different target.",
    ),
    "no_content_token": (
        "chat template only emitted control tokens within the capture window.",
        "This target cannot be measured in chat mode as configured.",
    ),
}


def _is_local_provider(provider: str) -> bool:
    return provider.strip().lower() in _LOCAL_PROVIDER_NAMES


def _format_estimate_duration(total_seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(round(total_seconds))), 60)
    return f"~{minutes:02d}:{seconds:02d}"


def _status_css_class(result: Optional[PreflightResult]) -> str:
    if result is None:
        return "evals-status-unchecked"
    return {
        "Ready": "evals-status-ready",
        "Unavailable": "evals-status-unavailable",
        "Blocked": "evals-status-blocked",
    }.get(result.status_label, "evals-status-blocked")


def _recovery_callout_text(target_label: str, result: PreflightResult) -> str:
    """Owner/problem/next-action copy for a target whose preflight is not
    a clean pass -- covers the warned-but-Ready case (naming the target
    and what its canary produced) and every Blocked/Unavailable state
    (naming an owner, the problem, and a next action), per the design
    spec's Preflight table.
    """
    if result.is_warned:
        return (
            f"{target_label} preflighted with a degenerate canary: its plain-text "
            "continuation looked out-of-distribution rather than failing outright. "
            "This target is still runnable -- a large divergence in its column may "
            "reflect that, not the prompt."
        )
    problem, next_action = _BLOCKED_COPY.get(
        result.state,
        (result.detail or "could not be confirmed ready.", "Review this target's configuration."),
    )
    return (
        f"Owner: {target_label}'s configured provider.\n"
        f"Problem: {target_label} {problem}\n"
        f"Next: {next_action}"
    )


class EvalsInspector(Vertical):
    """Inspector-pane content for a selected word bench: a Readiness list
    (one row per target, with a recovery callout for anything short of a
    clean Ready) and an Estimate (call count, time, and cost for paid
    targets only)."""

    def __init__(
        self,
        view_model: EvalsViewModel,
        bench_id: str,
        preflight: Optional[dict[str, PreflightResult]] = None,
        **kwargs: Any,
    ) -> None:
        """``preflight`` is resolved ONCE by ``EvalsScreen`` per selection
        and threaded into both this widget and ``BenchEditor`` -- see
        ``BenchEditor.__init__``'s identical parameter for why (I2 in the
        PR 3a fix report: each pane calling
        ``EvalsViewModel.preflight_for_bench`` independently read the
        bench's run-group snapshot twice on one render). ``None`` falls
        back to resolving it locally, for a widget constructed directly.
        """
        super().__init__(**kwargs)
        self._view_model = view_model
        self._bench_id = bench_id
        self._preflight = preflight

    def compose(self) -> ComposeResult:
        db = self._view_model.db
        if db is None:
            return
        try:
            config = load_bench(db, self._bench_id)
        except Exception:
            return

        preflight = (
            self._preflight
            if self._preflight is not None
            else self._view_model.preflight_for_bench(self._bench_id)
        )

        yield Static("Readiness", classes="destination-section evals-pane-title")
        if not config.target_ids:
            yield Static("No targets configured yet.", id="evals-inspector-readiness-empty")

        providers: list[str] = []
        # Index-derived widget ids, not target_id-derived -- same fix, same
        # reasoning as `bench_editor.py`'s identical target table: a
        # `BenchConfig.target_ids` duplicate must not collide two widget
        # ids and fail to compose this whole pane. `target_id` is still
        # used below for every lookup (`db.get_model`, `preflight.get`),
        # just never as part of a widget id.
        for index, target_id in enumerate(config.target_ids):
            model = db.get_model(target_id)
            target_label = model["name"] if model else f"(deleted target {target_id})"
            if model is not None:
                providers.append(str(model.get("provider") or ""))

            result = preflight.get(target_id)
            status_text = result.status_label if result is not None else "Not yet checked"
            yield Static(
                f"{target_label}: {status_text}",
                id=f"evals-inspector-target-{index}",
                classes=f"ds-status-badge {_status_css_class(result)}",
                markup=False,
            )
            # Only a clean pass (Ready, not warned) needs no callout -- see
            # the design spec's Preflight table: every other row states
            # "which endpoint / cannot report logprobs / switch mode /
            # out-of-distribution", never leaves a bare non-Ready badge.
            needs_callout = result is not None and (
                result.is_warned or result.status_label != "Ready"
            )
            if needs_callout:
                yield Static(
                    _recovery_callout_text(target_label, result),
                    id=f"evals-inspector-target-callout-{index}",
                    classes="ds-recovery-callout",
                    markup=False,
                )

        yield Static("Estimate", classes="destination-section evals-pane-title")
        dataset = self._view_model.dataset_by_id(config.dataset_id)
        sample_count = ((dataset or {}).get("metadata") or {}).get("sample_count") or 0
        target_count = len(config.target_ids)
        call_count = sample_count * target_count
        duration = _format_estimate_duration(call_count * _ASSUMED_SECONDS_PER_CALL)
        yield Static(
            f"{call_count} calls · {duration}",
            id="evals-inspector-estimate-calls",
            markup=False,
        )

        unresolved_target_count = len(config.target_ids) - len(providers)
        if unresolved_target_count > 0:
            # A deleted target contributes no provider to `providers` at
            # all (see the loop above) -- a bench whose targets have ALL
            # been deleted used to fall through to the `else` below and
            # claim "local · no cost", which is a claim about money this
            # code has no basis for: the deleted target's provider (paid or
            # local) is simply unknown, not confirmed local. Any
            # unresolvable target makes the whole cost line unknown, not
            # just the deleted target's own row.
            cost_text = (
                "cost unknown -- one or more targets could not be resolved"
            )
        elif providers and any(not _is_local_provider(provider) for provider in providers):
            cost_text = (
                "One or more targets are paid providers; this workbench does not "
                "estimate cost yet."
            )
        else:
            cost_text = "local · no cost"
        yield Static(cost_text, id="evals-inspector-estimate-cost", markup=False)


class EvalsCellInspector(Vertical):
    """Inspector-pane content for a ``"run_group"`` selection: a focused
    grid cell's full top-K and probe table.

    Mounted by ``evals_screen.py``'s ``_compose_inspector_pane`` alongside
    the (disabled, already-completed-run) primary action button. Updated by
    ``EvalsScreen._on_grid_cell_focused`` calling ``show_cell()`` directly
    against this already-mounted widget whenever ``results_grid.ResultsGrid``
    posts a ``CellFocused`` message -- NEVER by a screen-level
    ``refresh(recompose=True)``. Arrow-key movement in the grid fires one of
    these per keystroke; recomposing the screen on every keystroke would
    tear down and rebuild the grid's own ``DataTable``, losing cursor
    position and re-reading the run group from the database each time (see
    ``results_grid.py``'s module docstring).

    Renders no arithmetic of its own -- every value it prints is read
    directly off a ``CellCapture``/``ProbeReading`` the engine already
    computed, mirroring ``results_grid.py``'s own rule.
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Focused cell", classes="destination-section evals-pane-title"
        )
        yield Static(
            "Focus a cell in the grid to see its full top-K and probe "
            "table here.",
            id="evals-cell-inspector-body",
            markup=False,
        )

    def show_cell(self, event: "ResultsGrid.CellFocused") -> None:
        """Renders one focused cell. ``event`` is a
        ``results_grid.ResultsGrid.CellFocused`` message -- typed as a
        string annotation (see the ``TYPE_CHECKING`` import above) so this
        module has no runtime dependency on ``results_grid.py``, only the
        other way around.
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches evals_screen.py's own local import

        try:
            body = self.query_one("#evals-cell-inspector-body", Static)
        except QueryError:
            return

        lines = [f"{event.snippet_text!r} × {event.target_name}"]
        cell = event.cell

        if cell is None:
            lines.append("")
            lines.append("Not yet run.")
        elif isinstance(cell, CellError):
            lines.append("")
            lines.append(f"Failed: {cell.reason}")
            if cell.detail:
                lines.append(cell.detail)
        else:
            lines.append("")
            lines.append(
                f"K requested {cell.k_requested} · K returned {cell.k_returned} · "
                f"canary {cell.canary}"
            )
            lines.append(f"Truncated mass: {cell.truncated_mass * 100:.1f}%")
            lines.append("")
            lines.append("Top-K:")
            for index, tok in enumerate(cell.top_k, start=1):
                lines.append(
                    f"  {index}. {render_token(tok.token)}  "
                    f"{tok.logprob:.2f}  {tok.prob * 100:.1f}%"
                )
            if event.probes:
                lines.append("")
                lines.append("Probes:")
                for probe in event.probes:
                    reading = analysis.resolve_probe(
                        cell, probe, ever_observed=event.ever_observed.get(probe, False)
                    )
                    matched = (
                        next((t for t in cell.top_k if t.token == probe), None)
                        if reading.state == "observed"
                        else None
                    )
                    lines.append(
                        f"  {render_token(probe)}: "
                        f"{render_probe_reading(reading, matched)}"
                    )

        body.update("\n".join(lines))
