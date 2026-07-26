"""Readiness inspector: per-target readiness, recovery callouts, and a
call/time estimate for a selected word bench.

Mounted by ``evals_screen.py``'s ``_compose_inspector_pane`` above the
existing ``#evals-primary-action`` button (unchanged from Task 3 -- this
widget only adds content, it does not own the run control). Readiness
renders from ``EvalsViewModel.preflight_for_bench``, which itself reads a
stored run snapshot (``word_bench.storage.load_grid``) and never calls a
provider -- see ``bench_editor.py``'s module docstring for the shared
"never imports the runner" guarantee both widgets carry.

Per the design contract, ``.ds-status-badge`` colour lives in app-tier CSS
(``css/features/_evals.tcss``), never in this widget's own CSS -- there is
no ``DEFAULT_CSS`` here at all, deliberately.
"""

from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ...config import LOCAL_PROVIDERS
from ...Evals.word_bench.models import PreflightResult
from ...Evals.word_bench.storage import load_bench
from .evals_state import EvalsViewModel

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

    def __init__(self, view_model: EvalsViewModel, bench_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view_model = view_model
        self._bench_id = bench_id

    def compose(self) -> ComposeResult:
        db = self._view_model.db
        if db is None:
            return
        try:
            config = load_bench(db, self._bench_id)
        except Exception:
            return

        preflight = self._view_model.preflight_for_bench(self._bench_id)

        yield Static("Readiness", classes="destination-section evals-pane-title")
        if not config.target_ids:
            yield Static("No targets configured yet.", id="evals-inspector-readiness-empty")

        providers: list[str] = []
        for target_id in config.target_ids:
            model = db.get_model(target_id)
            target_label = model["name"] if model else f"(deleted target {target_id})"
            if model is not None:
                providers.append(str(model.get("provider") or ""))

            result = preflight.get(target_id)
            status_text = result.status_label if result is not None else "Not yet checked"
            yield Static(
                f"{target_label}: {status_text}",
                id=f"evals-inspector-target-{target_id}",
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
                    id=f"evals-inspector-target-callout-{target_id}",
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

        if providers and any(not _is_local_provider(provider) for provider in providers):
            cost_text = (
                "One or more targets are paid providers; this workbench does not "
                "estimate cost yet."
            )
        else:
            cost_text = "local · no cost"
        yield Static(cost_text, id="evals-inspector-estimate-cost", markup=False)
