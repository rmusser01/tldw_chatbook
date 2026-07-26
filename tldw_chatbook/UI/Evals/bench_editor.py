"""Detail-pane content for a selected bench: the word bench editor, and
the classic (non-word-bench) task's read-only detail.

Mounted by ``evals_screen.py``'s ``_compose_detail_pane`` in place of the
inline ``Static`` fields it used to yield directly (Task 3's placeholder
bench/classic branches) -- see that module's own docstring for why no
``Screen`` subclass is mounted anywhere here.

Readiness renders from ``word_bench.storage.load_grid``'s stored
``preflight`` mapping (via ``EvalsViewModel.preflight_for_bench``), never
recomputed here. Neither this module nor ``inspector.py`` imports the HTTP
capture client or the runner that drives it -- a source-scan test in
``Tests/UI/test_evals_bench_editor.py`` pins that neither module can reach
a provider at all, not just that today's ``compose()`` happens not to
call one.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ...Evals.word_bench.storage import load_bench
from .evals_state import EvalsViewModel

#: Verbatim. The design spec's own classic-task copy
#: (`2026-07-25-evals-console-rebuild-design.md`, "Classic tasks" section) --
#: asserted byte-for-byte by
#: ``test_classic_task_detail_shows_run_history_and_deferral_sentence``.
#: Launching a classic task from this workbench is a deliberate scope
#: decision, not an omission still to be wired; do not reword this into a
#: promise of a future date.
CLASSIC_TASK_DEFERRAL_SENTENCE = "Running classic tasks is not available in this slice."


def _target_status_text(preflight: dict[str, Any], target_id: str) -> str:
    result = preflight.get(target_id)
    if result is None:
        # The bench has never run, or this target was added after the
        # last run -- there is no stored verdict to read, and rendering
        # one of Ready/Unavailable/Blocked here would be a claim no
        # preflight ever made.
        return "Not yet checked"
    return result.status_label


class BenchEditor(Vertical):
    """Word bench detail: name, dataset, prompt mode, top-K, probes, and
    the target table (name/provider + readiness, resolved at render time
    from ``eval_models``)."""

    def __init__(self, view_model: EvalsViewModel, bench_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view_model = view_model
        self._bench_id = bench_id

    def compose(self) -> ComposeResult:
        db = self._view_model.db
        if db is None:
            yield Static(
                "The evaluation service is unavailable.",
                id="evals-bench-editor-unavailable",
            )
            return
        try:
            config = load_bench(db, self._bench_id)
        except Exception:
            yield Static(
                "This bench's configuration could not be read.",
                id="evals-bench-editor-error",
            )
            return

        yield Static(
            config.name, id="evals-detail-bench-name", classes="evals-pane-heading"
        )
        if config.description:
            yield Static(config.description, id="evals-detail-bench-description")

        dataset = self._view_model.dataset_by_id(config.dataset_id)
        dataset_name = dataset.get("name") if dataset else "(dataset not found)"
        sample_count = ((dataset or {}).get("metadata") or {}).get("sample_count")
        dataset_text = f"Dataset: {dataset_name}"
        if sample_count is not None:
            dataset_text += f" ({sample_count} snippets)"
        yield Static(dataset_text, id="evals-detail-bench-dataset")

        yield Static(
            f"Prompt mode: {config.prompt_mode}", id="evals-detail-bench-prompt-mode"
        )
        yield Static(f"Top-K: {config.top_k}", id="evals-detail-bench-top-k")
        probes_text = (
            " ".join(f'"{probe}"' for probe in config.probes) if config.probes else "(none)"
        )
        yield Static(f"Probes: {probes_text}", id="evals-detail-bench-probes")

        preflight = self._view_model.preflight_for_bench(self._bench_id)
        yield Static(
            f"Targets ({len(config.target_ids)})",
            classes="destination-section evals-pane-title",
        )
        if not config.target_ids:
            yield Static("No targets configured yet.", id="evals-bench-targets-empty")
            return
        with Vertical(id="evals-bench-target-table"):
            for target_id in config.target_ids:
                model = db.get_model(target_id)
                status_text = _target_status_text(preflight, target_id)
                if model is None:
                    # config_data.target_ids carries no foreign key (see the
                    # design spec's "Run provenance" section) -- a deleted
                    # eval_models row leaves a dangling reference here.
                    label = f"(deleted target {target_id}) — unresolvable"
                else:
                    label = f"{model['name']} ({model['provider']}) — {status_text}"
                yield Static(
                    label,
                    id=f"evals-bench-target-{target_id}",
                    classes="evals-bench-target-row",
                    markup=False,
                )


class ClassicTaskDetail(Vertical):
    """Read-only detail for a pre-existing (non-word-bench) eval_tasks row:
    its run history and the fixed deferral sentence. No run control is
    composed anywhere in this widget -- see ``evals_screen.py``'s
    ``_compose_inspector_pane``, which renders no button at all for a
    classic selection."""

    def __init__(self, view_model: EvalsViewModel, task: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view_model = view_model
        # NOT `self._task` -- Textual's own `MessagePump` uses that name
        # for the widget's message-pump asyncio task and overwrites it
        # once mounting starts, silently clobbering a same-named instance
        # attribute set here in `__init__` (confirmed the hard way: this
        # was `self._task = task` originally, and `compose()` below saw an
        # `asyncio.Task`, not the row dict, by the time it ran).
        self._classic_task = task

    def compose(self) -> ComposeResult:
        task = self._classic_task
        yield Static(
            str(task.get("name") or "Untitled task"),
            id="evals-detail-classic-name",
            classes="evals-pane-heading",
        )
        yield Static(
            f"Task type: {task.get('task_type', 'unknown')}",
            id="evals-detail-classic-type",
        )
        yield Static("Run history", classes="destination-section evals-pane-title")

        runs = self._view_model.runs_for_task(task["id"])
        if not runs:
            yield Static("No runs yet.", id="evals-detail-classic-runs-empty")
        else:
            for index, run in enumerate(runs):
                status = run.get("status", "unknown")
                created_at = run.get("created_at", "")
                model_name = run.get("model_name", "unknown")
                yield Static(
                    f"{created_at}  {model_name}  {status}",
                    id=f"evals-detail-classic-run-{index}",
                    classes="evals-classic-run-row",
                    markup=False,
                )

        yield Static(
            CLASSIC_TASK_DEFERRAL_SENTENCE, id="evals-detail-classic-deferral", markup=False
        )
