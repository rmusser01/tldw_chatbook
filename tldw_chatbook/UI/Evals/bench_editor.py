"""Detail-pane content for a selected bench: the word bench editor, and
the classic (non-word-bench) task's read-only detail.

Mounted by ``evals_screen.py``'s ``_compose_detail_pane`` in place of the
inline ``Static`` fields it used to yield directly (Task 3's placeholder
bench/classic branches) -- see that module's own docstring for why no
``Screen`` subclass is mounted anywhere here.

Readiness renders from ``word_bench.storage.load_run_preflight``'s stored
``preflight`` mapping (via ``EvalsViewModel.preflight_for_bench``), never
recomputed here. That map is resolved ONCE per selection by
``evals_screen.py`` and passed into ``BenchEditor`` as a constructor
argument -- see ``BenchEditor.__init__`` -- rather than this widget calling
``preflight_for_bench`` itself, so a bench selection does not read the same
run-group snapshot twice (once for this pane, once for ``inspector.py``'s).
Neither this module nor ``inspector.py`` imports the HTTP capture client or
the runner that drives it -- a source-scan test in
``Tests/UI/test_evals_bench_editor.py`` pins that neither module can reach
a provider at all, not just that today's ``compose()`` happens not to
call one.

Task 5 (task-1482): ``BenchEditor`` becomes an editable form for name,
description, prompt mode, top-K, and probes. Dataset and targets stay
read-only this task (targets are Task 6's own scope; the dataset is
create-time-only, permanently -- ``save_bench`` has no ``dataset_id``
parameter, see its own docstring). Editing is display-only until Save:
no field posts or reacts to a live ``Changed`` message, so there is no
watcher to accidentally trip on the Select-posts-Changed-on-mount trap
this codebase has hit before. Save reads every widget fresh, builds a
``BenchConfig`` with the CURRENT stored ``target_ids`` (untouched by this
task), and persists via ``save_bench``. On failure (``ValueError`` --
either this module's own top-K parse, ``BenchConfig`` validation, or
``Evals_DB.InputError`` from a blank/control-char name; or
``Evals_DB.ConflictError`` from a name collision) the error renders
in-place in ``#evals-bench-form-error`` and NOTHING recomposes -- every
other field keeps exactly what the user typed. On success this widget
posts ``Saved``; ``evals_screen.py`` handles that by calling its own
``select(kind="bench", ...)``, which recomposes from the freshly
persisted row (picking up anything ``save_bench``'s own cleaning -- e.g.
``_clean_task_name``'s control-character strip -- changed from what was
typed).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Select, Static, TextArea

from ...DB.Evals_DB import ConflictError, EvalsDB
from ...Evals.word_bench.models import BenchConfig, PreflightResult, Target
from ...Evals.word_bench.storage import load_bench, save_bench
from .evals_state import EvalsViewModel
from .snippet_editor import render_snippet_cell

#: Verbatim. The design spec's own classic-task copy
#: (`2026-07-25-evals-console-rebuild-design.md`, "Classic tasks" section) --
#: asserted byte-for-byte by
#: ``test_classic_task_detail_shows_run_history_and_deferral_sentence``.
#: Launching a classic task from this workbench is a deliberate scope
#: decision, not an omission still to be wired; do not reword this into a
#: promise of a future date.
CLASSIC_TASK_DEFERRAL_SENTENCE = "Running classic tasks is not available in this slice."

#: Verbatim. Task 5's own pinned error string for an unparseable or
#: sub-1 top-K value -- asserted exactly by
#: ``test_top_k_parse_failure_renders_the_pinned_callout``.
TOP_K_ERROR_TEXT = "Top-K must be a whole number of 1 or more."


def _target_status_text(preflight: dict[str, Any], target_id: str) -> str:
    result = preflight.get(target_id)
    if result is None:
        # The bench has never run, or this target was added after the
        # last run -- there is no stored verdict to read, and rendering
        # one of Ready/Unavailable/Blocked here would be a claim no
        # preflight ever made.
        return "Not yet checked"
    return result.status_label


def _resolve_bench_targets(db: EvalsDB, target_ids: Sequence[str]) -> list[Target]:
    """Resolves ``target_ids`` to ``Target`` instances via their
    ``eval_models`` rows, for the save-time prompt-mode/target validation
    below -- a target id with no resolvable row (a deleted target, already
    rendered as unresolvable in the target table further down this same
    widget) is skipped rather than raising: it cannot be checked either
    way, and ``save_bench`` itself never rejects a bench for carrying one.

    A thin, LOCAL mirror of ``sample_bench.py``'s own ``_resolve_targets``
    (same ``db.get_model`` lookup, same three fields) rather than an
    import of that private helper: ``sample_bench.py`` imports the runner
    and the HTTP client that drives it, both of which this module's own
    source-scan test pins it must never reach, even transitively through
    an import graph (see the module docstring's own "provider" mention
    above). Never sets ``prefix``/``system_prompt`` -- no
    ``eval_models`` column stores either field today, so every ``Target``
    built here is always valid for both prompt modes in production; the
    ``is_valid_for_mode`` check this feeds is a wired-but-currently-
    unreachable seam, exercised in tests by monkeypatching this function
    to return a hand-built ``Target`` that does carry one.
    """
    targets: list[Target] = []
    for target_id in target_ids:
        model = db.get_model(target_id)
        if model is None:
            continue
        targets.append(
            Target(
                id=model["id"],
                name=model["name"],
                provider=model["provider"],
                model_id=model["model_id"],
            )
        )
    return targets


class BenchEditor(Vertical):
    """Word bench editor: name, description, prompt mode, top-K, and
    probes are editable (Save/Revert); dataset and the target table
    (name/provider + readiness, resolved at render time from
    ``eval_models``) stay read-only this task."""

    class Saved(Message, namespace="bench_editor"):
        """Posted after ``save_bench`` succeeds. Carries the bench's own
        ``eval_tasks`` id (unchanged across a save -- this is always an
        edit, never a create) so the handler can re-select it without
        this widget reaching into ``self.screen`` itself; see
        ``evals_screen.py``'s own handler, which recomposes via its public
        ``select()`` so the reopened form reads back whatever
        ``save_bench``'s own write path actually persisted (e.g.
        ``_clean_task_name``'s control-character strip), not merely what
        was typed.
        """

        def __init__(self, bench_id: str) -> None:
            super().__init__()
            self.bench_id = bench_id

    def __init__(
        self,
        view_model: EvalsViewModel,
        bench_id: str,
        preflight: Optional[dict[str, PreflightResult]] = None,
        **kwargs: Any,
    ) -> None:
        """``preflight`` is the bench's readiness map, resolved ONCE by
        ``EvalsScreen`` per selection (see its ``_preflight_for_selection``)
        and passed in here rather than this widget calling
        ``EvalsViewModel.preflight_for_bench`` itself -- ``EvalsInspector``
        needs the identical map for the same selection, and each pane
        calling it independently read the bench's run-group snapshot twice
        on one render (see I2 in the PR 3a fix report). ``None`` (the
        default) falls back to resolving it locally, so a widget
        constructed directly -- as this module's own tests do -- still
        works without a caller threading the map through.
        """
        super().__init__(**kwargs)
        self._view_model = view_model
        self._bench_id = bench_id
        self._preflight = preflight
        #: The config `compose()` most recently loaded -- read back by the
        #: Save handler for the fields Task 5 does not edit (`dataset_id`,
        #: `target_ids`, `concurrency`), which must round-trip verbatim.
        #: `None` only when `compose()` bailed out before reaching the form
        #: (no db, or an unreadable row) -- in which case no Save/Revert
        #: button exists for a press to ever reach this attribute through.
        self._loaded_config: Optional[BenchConfig] = None

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
        self._loaded_config = config

        yield Static("Name", classes="evals-bench-field-label")
        yield Input(value=config.name, id="evals-bench-name")

        yield Static("Description", classes="evals-bench-field-label")
        yield Input(value=config.description, id="evals-bench-description")

        dataset = self._view_model.dataset_by_id(config.dataset_id)
        dataset_name = dataset.get("name") if dataset else "(dataset not found)"
        sample_count = ((dataset or {}).get("metadata") or {}).get("sample_count")
        dataset_text = f"Dataset: {dataset_name}"
        if sample_count is not None:
            dataset_text += f" ({sample_count} snippets)"
        # markup=False: `dataset_name` is user-authored free text -- a bare
        # `[/]` would raise `MarkupError` the instant this Static lays out.
        # The dataset stays read-only permanently (a create-time-only
        # field -- `save_bench` has no `dataset_id` parameter on its edit
        # path, see that function's own docstring), so the tooltip states
        # that rather than merely omitting an edit control silently.
        dataset_static = Static(dataset_text, id="evals-detail-bench-dataset", markup=False)
        dataset_static.tooltip = (
            "The dataset is set when a bench is created and cannot be changed here."
        )
        yield dataset_static

        yield Static("Prompt mode", classes="evals-bench-field-label")
        yield Select(
            [("raw", "raw"), ("chat", "chat")],
            value=config.prompt_mode,
            id="evals-bench-prompt-mode",
            allow_blank=False,
        )

        yield Static("Top-K", classes="evals-bench-field-label")
        yield Input(value=str(config.top_k), id="evals-bench-top-k")

        yield Static(
            "Probes — one per line; leading and trailing spaces are "
            "significant and shown as ␣",
            classes="evals-bench-field-label",
        )
        if config.probes:
            with Vertical(id="evals-bench-probes-preview"):
                for index, probe in enumerate(config.probes):
                    yield Static(
                        render_snippet_cell(probe),
                        id=f"evals-bench-probe-preview-{index}",
                        classes="evals-bench-probe-preview-row",
                        markup=False,
                    )
        else:
            yield Static(
                "(no probes yet)", id="evals-bench-probes-preview-empty"
            )
        yield TextArea("\n".join(config.probes), id="evals-bench-probes")

        # The `.ds-recovery-callout` CLASS is deliberately withheld here
        # and added only by `_show_form_error` on an actual failure -- not
        # just `display = False`. `EvalsInspector`'s own preflight callout
        # shares this exact class (see `#evals-inspector-bench .ds-
        # recovery-callout` in _evals.tcss), and `test_never_run_bench_
        # renders_unpreflighted_state` asserts a screen-WIDE `not screen.
        # query(".ds-recovery-callout")` for the clean-preflight case; an
        # always-classed-but-hidden Static here would still match that
        # query and fail an invariant this widget has nothing to do with.
        # An always-visible empty `.ds-recovery-callout` would ALSO be a
        # permanent bordered blank box (the class's own padding/border
        # render even with no text) -- a second, independent reason this
        # class is not applied until there is something to say.
        error_widget = Static("", id="evals-bench-form-error", markup=False)
        error_widget.display = False
        yield error_widget

        with Horizontal(id="evals-bench-form-actions", classes="ds-toolbar"):
            yield Button(
                "Save",
                id="evals-bench-save",
                classes="console-action-primary",
                tooltip="Save name, description, prompt mode, top-K, and probes.",
            )
            yield Button(
                "Revert",
                id="evals-bench-revert",
                classes="console-action-secondary",
                tooltip="Discard unsaved changes and reload this bench.",
            )

        preflight = (
            self._preflight
            if self._preflight is not None
            else self._view_model.preflight_for_bench(self._bench_id)
        )
        yield Static(
            f"Targets ({len(config.target_ids)})",
            classes="destination-section evals-pane-title",
        )
        if not config.target_ids:
            yield Static("No targets configured yet.", id="evals-bench-targets-empty")
            return
        with Vertical(id="evals-bench-target-table"):
            # Index-derived widget ids, not target_id-derived: `target_ids`
            # is user-editable data (see `BenchConfig`) with no uniqueness
            # or identifier-safety constraint enforced anywhere on write, so
            # a duplicate target id would otherwise collide and fail to
            # compose the whole pane. `target_id` itself is still used
            # below, just never as (or as part of) a widget id -- see
            # `snippet_editor.py`'s identical `_compose_row` fix for the
            # same principle applied to snippets.
            for index, target_id in enumerate(config.target_ids):
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
                    id=f"evals-bench-target-{index}",
                    classes="evals-bench-target-row",
                    markup=False,
                )

    def _show_form_error(self, message: str) -> None:
        """Renders ``message`` in ``#evals-bench-form-error`` IN PLACE --
        never via ``self.refresh(recompose=True)``. A recompose here would
        rebuild every field from the last-saved ``BenchConfig``, discarding
        whatever the user had just typed -- exactly the state loss a failed
        Save must not cause (see the module docstring). ``add_class`` (not
        a class set at compose time) is what actually makes this callout
        `.ds-recovery-callout`-styled -- see this widget's own compose()
        comment for why that has to wait until there is a real error."""
        error_widget = self.query_one("#evals-bench-form-error", Static)
        error_widget.update(message)
        error_widget.add_class("ds-recovery-callout")
        error_widget.display = True

    @on(Button.Pressed, "#evals-bench-save")
    def _on_save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        db = self._view_model.db
        loaded = self._loaded_config
        if db is None or loaded is None:
            # Defensive only: this button is never composed unless both a
            # db and a successfully loaded config exist (see compose()'s
            # own early returns above).
            return

        name = self.query_one("#evals-bench-name", Input).value
        description = self.query_one("#evals-bench-description", Input).value
        prompt_mode = self.query_one("#evals-bench-prompt-mode", Select).value
        top_k_raw = self.query_one("#evals-bench-top-k", Input).value
        probes_text = self.query_one("#evals-bench-probes", TextArea).text

        try:
            top_k = int(top_k_raw.strip())
            if top_k < 1:
                raise ValueError("top_k below 1")
        except ValueError:
            self._show_form_error(TOP_K_ERROR_TEXT)
            return

        # One probe per line, whitespace preserved exactly -- see the
        # module docstring and `render_snippet_cell`'s own callers below.
        # Splitting on "\n" alone is not enough: a user who presses Enter
        # after the last probe (or leaves a blank line anywhere) produces
        # a genuine zero-length line -- `BenchConfig` accepts it happily,
        # and `analysis.resolve_probe` would then carry a meaningless
        # empty-string probe column all the way through a run. Only a
        # ZERO-LENGTH line is dropped here; a WHITESPACE-ONLY line (e.g. a
        # lone " ") is kept byte-exact -- "whitespace preserved exactly"
        # is a claim about a token's CONTENT, and a single space is a
        # legitimate (if unusual) exact token, not an empty one. Note this
        # is a real distinction from `compose()`'s own `"\n".join(config.
        # probes)`, which never appends a trailing newline of its own --
        # `TextArea.text` reflects exactly what the user TYPED, trailing
        # Enter-press included, and that is a different guarantee.
        probes = tuple(line for line in probes_text.split("\n") if line != "")

        try:
            config = BenchConfig(
                name=name,
                description=description,
                prompt_mode=prompt_mode,
                top_k=top_k,
                dataset_id=loaded.dataset_id,
                target_ids=loaded.target_ids,
                probes=probes,
                concurrency=loaded.concurrency,
            )
        except ValueError as exc:
            self._show_form_error(str(exc))
            return

        # Prompt-mode/target revalidation: see `_resolve_bench_targets`'s
        # own docstring for why this is currently unreachable through a
        # real db-backed target (no `eval_models` column stores `prefix`/
        # `system_prompt` yet) but stays wired and tested.
        resolved_targets = _resolve_bench_targets(db, config.target_ids)
        invalid_target = next(
            (t for t in resolved_targets if not t.is_valid_for_mode(config.prompt_mode)),
            None,
        )
        if invalid_target is not None:
            self._show_form_error(
                f"{invalid_target.name} is not valid for {config.prompt_mode} mode; "
                "change its prefix/system prompt settings before switching modes."
            )
            return

        try:
            save_bench(db, config, self._bench_id)
        except (ValueError, ConflictError) as exc:
            # ValueError: BenchConfig re-validation inside save_bench (a
            # duplicate target_id -- unreachable here since `loaded.
            # target_ids` round-trips verbatim and was itself already
            # valid), or `Evals_DB.InputError` (a ValueError subclass) from
            # `_clean_task_name` rejecting a blank/control-char-only name.
            # ConflictError: `eval_tasks.name` collided with another task's
            # name, live OR soft-deleted (see `save_bench`'s docstring).
            # Mutation check: dropping the `ConflictError` half of this
            # tuple makes a rename-to-a-taken-name Save raise straight out
            # of this handler instead of rendering the callout.
            self._show_form_error(str(exc))
            return

        self.post_message(self.Saved(self._bench_id))

    @on(Button.Pressed, "#evals-bench-revert")
    def _on_revert_pressed(self, event: Button.Pressed) -> None:
        """Discards unsaved edits by re-selecting this same bench -- the
        screen's own `select()` recompose reloads every field from
        storage, which is what "revert" means here (there is no separate
        in-memory draft to roll back; the fields ARE the widgets)."""
        event.stop()
        screen_select = getattr(self.screen, "select", None)
        if callable(screen_select):
            screen_select(kind="bench", id=self._bench_id)


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
        # markup=False: the classic task's name is exactly as user-authored
        # as a word bench's (see `_classic_row_label` in library_rail.py,
        # fixed in the same task-1482 sweep) -- a bare `[/]` would raise
        # `MarkupError` the instant this Static lays out. `task_type` below
        # is not free text (a fixed, internally-controlled vocabulary), so
        # it needs no equivalent guard.
        yield Static(
            str(task.get("name") or "Untitled task"),
            id="evals-detail-classic-name",
            classes="evals-pane-heading",
            markup=False,
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
