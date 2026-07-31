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
description, prompt mode, top-K, and probes. The dataset stays read-only
permanently -- ``save_bench`` has no ``dataset_id`` parameter, see its own
docstring. Editing is display-only until Save: no field posts or reacts
to a live ``Changed`` message, so there is no watcher to accidentally trip
on the Select-posts-Changed-on-mount trap this codebase has hit before.
Save reads every widget fresh, builds a ``BenchConfig``, and persists via
``save_bench``. On failure (``ValueError`` -- either this module's own
top-K parse, ``BenchConfig`` validation, or ``Evals_DB.InputError`` from a
blank/control-char name; or ``Evals_DB.ConflictError`` from a name
collision) the error renders in-place in ``#evals-bench-form-error`` and
NOTHING recomposes -- every other field keeps exactly what the user
typed. On success this widget posts ``Saved``; ``evals_screen.py`` handles
that by calling its own ``select(kind="bench", ...)``, which recomposes
from the freshly persisted row (picking up anything ``save_bench``'s own
cleaning -- e.g. ``_clean_task_name``'s control-character strip -- changed
from what was typed).

Task 6 (task-1482): targets become editable too, but through a SEPARATE
mechanism from every field above -- a staged ``self._staged_target_ids``
list, mutated in place by per-row ``Remove`` buttons and an Add picker
over ``EvalsViewModel.llama_targets()`` (``db.list_models(provider=
"llama_cpp")``), and read back verbatim by ``_on_save_pressed`` in place
of the "CURRENT stored ``target_ids``" this docstring used to describe
before this task. Add/Remove mutate ONLY ``#evals-bench-targets-section``
(``remove_children()`` + ``mount_all()``, built by
``_build_targets_section``) rather than a whole-widget recompose -- a
recompose would discard whatever the user has typed into Name/
Description/Top-K/Probes above, exactly the state loss the "display-only
until Save" paragraph above exists to avoid. A duplicate add (the target
id is already staged) is rejected inline with the exact text ``"Target
already on this bench."``, through the same ``#evals-bench-form-error``
callout Top-K/name failures use. When NO ``llama_cpp`` ``eval_models`` row
exists anywhere in the db, the picker is replaced by
``#evals-bench-create-target``, which posts ``CreateTargetRequested`` for
``evals_screen.py`` to handle -- this module must never import
``sample_bench.resolve_sample_target`` (the function that actually
creates the row) itself: ``sample_bench.py`` imports the capture client
and the runner, both of which the source-scan test mentioned above
(``Tests/UI/test_evals_bench_editor.py``) pins this module can never
reach, even transitively. ``stage_target()`` is the targeted
(non-recompose) call the screen makes once it has created the row.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
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
    """Word bench editor: name, description, prompt mode, top-K, probes,
    and targets (Task 6: Add/Remove via a staged list) are all editable
    (Save/Revert); the dataset (name/sample count, resolved at render time)
    stays read-only permanently -- see the module docstring."""

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

    class CreateTargetRequested(Message, namespace="bench_editor"):
        """Posted when the zero-``llama_cpp``-models ``#evals-bench-
        create-target`` button is pressed. Handled by ``evals_screen.py``,
        never here -- see the module docstring's source-scan pin:
        resolving/creating the row reuses ``sample_bench.
        resolve_sample_target``, which (transitively, via ``capture_
        client``/``runner``) this module must never import, even just to
        call it directly. Carries no payload -- the handler already has
        the view model and app config, and reaches the mounted editor via
        ``self.query_one(BenchEditor)`` (only one is ever mounted at a
        time, for the current selection) to call ``stage_target()`` on it.
        """

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
        #: Save handler for the fields this widget does not stage directly
        #: (`dataset_id`, `concurrency`), which must round-trip verbatim.
        #: `None` only when `compose()` bailed out before reaching the form
        #: (no db, or an unreadable row) -- in which case no Save/Revert
        #: button exists for a press to ever reach this attribute through.
        self._loaded_config: Optional[BenchConfig] = None
        #: Task 6: the staged target id list -- FORM STATE like every
        #: other field on this widget (see the module docstring), mutated
        #: in place by the Add/Remove/`stage_target` handlers below and
        #: read back verbatim by `_on_save_pressed`. Reset to whatever
        #: `compose()` loads whenever the whole widget is rebuilt (a fresh
        #: selection, or Revert's re-select). Populated in `compose()`,
        #: not here, so a widget that never composes (no db, unreadable
        #: row) never has one of these that could go stale against nothing.
        self._staged_target_ids: list[str] = []
        #: The preflight map resolved by `compose()` (either the one
        #: passed in, or a freshly resolved one) -- cached so `_build_
        #: targets_section`'s later, targeted re-renders (Add/Remove/
        #: `stage_target`) never need to re-resolve it themselves; per-
        #: target readiness only ever changes on a fresh bench selection,
        #: never mid-edit.
        self._preflight_map: dict[str, PreflightResult] = {}

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
        self._staged_target_ids = list(config.target_ids)

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
        self._preflight_map = preflight
        yield Vertical(*self._build_targets_section(), id="evals-bench-targets-section")

    def _build_targets_section(self) -> list[Widget]:
        """Builds the whole "Targets (N)" slice -- heading, row table (or
        the empty state), and the Add picker / zero-models create-target
        affordance -- as concrete widget INSTANCES rather than a
        `with Container(): yield child`-composed generator.

        That `with`-block pattern (used by every OTHER section of
        `compose()` above) only works while Textual's own compose
        machinery has an active `app._compose_stacks` frame open -- true
        during a real `compose()` call, NOT true when this same building
        logic needs to run again later from an event handler
        (`_on_add_target_pressed`, `_on_remove_target_pressed`,
        `stage_target`). Building plain widget instances instead (passed
        as `*children` to each container's own constructor -- a fully
        supported `Widget.__init__` form, not a workaround) works
        identically in both places: `compose()` just yields the returned
        list's container, and `_refresh_targets_section` mounts the same
        builder's output directly into the already-live `#evals-bench-
        targets-section` container.
        """
        db = self._view_model.db
        preflight = self._preflight_map
        widgets: list[Widget] = [
            Static(
                f"Targets ({len(self._staged_target_ids)})",
                id="evals-bench-targets-heading",
                classes="destination-section evals-pane-title",
            )
        ]
        if not self._staged_target_ids:
            widgets.append(
                Static("No targets configured yet.", id="evals-bench-targets-empty")
            )
        else:
            # Index-derived widget ids, not target_id-derived: `target_ids`
            # is user-editable data (see `BenchConfig`) with no uniqueness
            # or identifier-safety constraint enforced anywhere on write,
            # so a duplicate target id would otherwise collide and fail to
            # compose the whole pane. `target_id` itself is still used
            # below, just never as (or as part of) a widget id -- see
            # `snippet_editor.py`'s identical `_compose_row` fix for the
            # same principle applied to snippets.
            rows = [
                self._build_target_row(db, preflight, index, target_id)
                for index, target_id in enumerate(self._staged_target_ids)
            ]
            widgets.append(Vertical(*rows, id="evals-bench-target-table"))
        widgets.append(self._build_target_add_control())
        return widgets

    @staticmethod
    def _build_target_row(
        db: Optional[EvalsDB],
        preflight: dict[str, Any],
        index: int,
        target_id: str,
    ) -> Widget:
        """One target row: its readiness ``Static`` (unchanged id/class
        from before Task 6, ``#evals-bench-target-{index}`` /
        ``evals-bench-target-row``) plus a per-row ``Remove`` button
        (``#evals-bench-target-remove-{index}``, following the row's own
        numbering)."""
        model = db.get_model(target_id) if db is not None else None
        status_text = _target_status_text(preflight, target_id)
        if model is None:
            # config_data.target_ids carries no foreign key (see the
            # design spec's "Run provenance" section) -- a deleted
            # eval_models row leaves a dangling reference here. Still
            # removable via the button below, just never resolvable to a
            # real name.
            label = f"(deleted target {target_id}) — unresolvable"
        else:
            label = f"{model['name']} ({model['provider']}) — {status_text}"
        return Horizontal(
            Static(
                label,
                id=f"evals-bench-target-{index}",
                classes="evals-bench-target-row",
                markup=False,
            ),
            Button(
                "Remove",
                id=f"evals-bench-target-remove-{index}",
                classes="evals-bench-target-remove console-action-secondary",
            ),
            classes="evals-bench-target-row-wrap",
        )

    def _build_target_add_control(self) -> Widget:
        """The Add picker (a ``Select`` over ``EvalsViewModel.
        llama_targets()`` plus an ``Add`` button), or -- when no
        ``llama_cpp`` ``eval_models`` row exists anywhere in the db yet --
        the ``#evals-bench-create-target`` button instead.

        ``Select`` raises ``EmptySelectError`` when constructed with zero
        options and ``allow_blank=False`` (see its own docstring) --
        ``llama_targets()`` being empty is exactly this method's own
        create-target branch condition, so the two can never disagree and
        this never risks that error.
        """
        llama_targets = self._view_model.llama_targets()
        if not llama_targets:
            return Button(
                "Create target from configured llama.cpp server",
                id="evals-bench-create-target",
                classes="console-action-secondary",
            )
        # escape_markup: `Select` options parse their label as markup on
        # render (the same `Content.from_markup` hazard this widget's
        # other user-authored strings already guard against, see the
        # module docstring's markup-hazard sweep) -- a model name is free
        # text a user typed, or imported, elsewhere in this app.
        options = [
            (escape_markup(f"{row['name']} ({row['model_id']})"), row["id"])
            for row in llama_targets
        ]
        return Horizontal(
            # compact=True: a bordered Select is 3 rows tall by default
            # (see Textual's own DEFAULT_CSS) against every other row in
            # this section's 1-row convention -- the same fix `sources_
            # pane.py`'s TASK-995 comment applies to a Select inside a
            # height-constrained strip, for the identical reason: without
            # it, this row alone could push the section's own fixed Add
            # control past `#evals-detail-pane`'s clip rectangle at a
            # realistic viewport (confirmed live via `test_every_pane_
            # descendant_stays_within_its_pane`).
            Select(options, allow_blank=False, compact=True, id="evals-bench-add-target"),
            Button(
                "Add",
                id="evals-bench-add-target-button",
                classes="console-action-secondary",
            ),
            id="evals-bench-add-target-row",
        )

    async def _refresh_targets_section(self) -> None:
        """Re-renders just ``#evals-bench-targets-section`` after a staged
        Add/Remove/``stage_target`` mutation -- see ``_build_targets_
        section``'s own docstring for why this is ``remove_children()`` +
        ``mount_all()``, never ``self.refresh(recompose=True)`` on the
        whole editor: a full recompose here would discard whatever the
        user has typed into Name/Description/Top-K/Probes, exactly the
        state loss the module docstring's Task 5 paragraph exists to
        avoid. Also clears any stale ``#evals-bench-form-error`` (e.g. a
        duplicate-add rejection) -- the state it complained about just
        changed.

        ``async`` (and its own callers below too) -- ``remove_children()``
        returns an awaitable that completes once Textual has actually torn
        the old rows down; without awaiting it, ``mount_all()`` runs
        against a DOM that STILL holds the old, same-id widgets (removal
        is scheduled, not immediate), and mounting id-colliding replacements
        into it raises ``DuplicateIds`` -- confirmed the hard way, not
        merely reasoned about, when this was first written as a bare
        `self.refresh`-free but still-synchronous method.
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            section = self.query_one("#evals-bench-targets-section", Vertical)
        except QueryError:
            return
        await section.remove_children()
        await section.mount_all(self._build_targets_section())
        self._clear_form_error()

    async def stage_target(self, model_row: Mapping[str, Any]) -> None:
        """Stages a freshly created ``eval_models`` row as a bench target
        -- called by ``evals_screen.py``'s ``CreateTargetRequested``
        handler after IT creates the row via ``sample_bench.
        resolve_sample_target(..., create=True)`` (see that message's own
        docstring for why this module cannot make that call itself). A
        TARGETED call against the already-mounted editor instance, never a
        recompose -- see ``_build_targets_section``'s own docstring.
        """
        target_id = model_row.get("id") if isinstance(model_row, Mapping) else None
        if not target_id or target_id in self._staged_target_ids:
            # Defensive only: a freshly `_unique_name`d row cannot already
            # be staged, but this mirrors the Add-picker's own duplicate
            # guard rather than assuming the caller never will pass one.
            return
        self._staged_target_ids.append(target_id)
        await self._refresh_targets_section()

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

    def _clear_form_error(self) -> None:
        """Hides the shared inline error callout (``#evals-bench-form-
        error``) -- called by a target mutation's success path so a stale
        duplicate-target (or an earlier Top-K/name) error does not linger
        once the state it complained about has changed. Mirrors
        ``_show_form_error``'s own in-place-update contract: never a
        recompose."""
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            error_widget = self.query_one("#evals-bench-form-error", Static)
        except QueryError:
            return
        error_widget.update("")
        error_widget.remove_class("ds-recovery-callout")
        error_widget.display = False

    @on(Button.Pressed, "#evals-bench-add-target-button")
    async def _on_add_target_pressed(self, event: Button.Pressed) -> None:
        """Stages the picker's currently selected target -- form state
        only (see the module docstring); the actual `eval_tasks.
        config_data.target_ids` write happens at Save, same as every other
        field this widget edits."""
        event.stop()
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            picker = self.query_one("#evals-bench-add-target", Select)
        except QueryError:
            # Defensive only: this button is never composed without the
            # picker beside it (see `_build_target_add_control`) -- the
            # zero-models state renders `#evals-bench-create-target`
            # instead, with no `Add` button at all.
            return
        target_id = picker.value
        if target_id is Select.BLANK or not isinstance(target_id, str):
            # Defensive only: `allow_blank=False` plus at least one option
            # (see `_build_target_add_control`'s own `EmptySelectError`
            # note) means Select auto-selects a real value the instant it
            # mounts.
            return
        if target_id in self._staged_target_ids:
            self._show_form_error("Target already on this bench.")
            return
        self._staged_target_ids.append(target_id)
        await self._refresh_targets_section()

    @on(Button.Pressed, ".evals-bench-target-remove")
    async def _on_remove_target_pressed(self, event: Button.Pressed) -> None:
        """Un-stages one target row by INDEX -- see `_build_target_row`'s
        own comment for why widget ids here are index-, not target_id-,
        derived. Removing the last target is allowed (a draft state --
        the zero-target Run gate and readiness copy both read SAVED state,
        never this widget's staged form state, see the module docstring)."""
        event.stop()
        button_id = event.button.id or ""
        prefix = "evals-bench-target-remove-"
        if not button_id.startswith(prefix):
            return
        try:
            index = int(button_id[len(prefix):])
        except ValueError:
            return
        if not 0 <= index < len(self._staged_target_ids):
            return
        del self._staged_target_ids[index]
        await self._refresh_targets_section()

    @on(Button.Pressed, "#evals-bench-create-target")
    def _on_create_target_pressed(self, event: Button.Pressed) -> None:
        """Posts `CreateTargetRequested` for `evals_screen.py` to handle --
        see that message class's own docstring for why this module cannot
        create the row itself."""
        event.stop()
        self.post_message(self.CreateTargetRequested())

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
                # Task 6: the staged list (Add/Remove'd via this same
                # form, never yet persisted) replaces what used to be
                # `loaded.target_ids`'s verbatim carry-through -- see the
                # module docstring's Task 6 paragraph. `BenchConfig`'s own
                # `strict=True` default still rejects a duplicate here,
                # which the Add picker's own inline rejection (`_on_add_
                # target_pressed`) should already have made unreachable.
                target_ids=tuple(self._staged_target_ids),
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
        except (ValueError, ConflictError, RuntimeError) as exc:
            # ValueError: BenchConfig re-validation inside save_bench (a
            # duplicate target_id -- unreachable here since the Add
            # picker's own inline rejection, `_on_add_target_pressed`,
            # already keeps `self._staged_target_ids` duplicate-free), or
            # `Evals_DB.InputError` (a ValueError subclass) from
            # `_clean_task_name` rejecting a blank/control-char-only name.
            # ConflictError: `eval_tasks.name` collided with another task's
            # name, live OR soft-deleted (see `save_bench`'s docstring).
            # RuntimeError: `save_bench`'s update branch found no matching
            # row -- the bench was deleted (this process or another)
            # between this form loading it and this Save (PR #1138
            # review). Without this branch the exception propagated
            # uncaught out of this handler, crashing the worker, AND the
            # user would otherwise have seen nothing at all -- not even a
            # crash, if some caller ever swallowed it -- instead of the
            # honest "this bench is gone" this callout states.
            # Mutation check: dropping either the `ConflictError` or the
            # `RuntimeError` half of this tuple makes the matching Save
            # failure raise straight out of this handler instead of
            # rendering the callout.
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
