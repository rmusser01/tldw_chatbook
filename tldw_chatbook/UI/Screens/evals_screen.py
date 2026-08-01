"""Evaluations screen implementation.

The evaluation hub used to push a separate Textual ``Screen`` object as a
child inside a plain ``Container`` (``EvalsWindowV3.compose()`` yielding
``EvalNavigationScreen``). That is not a supported way to mount a
``Screen``: it mounts structurally (child widgets are still queryable) but
the compositor never gives it a laid-out region, so it renders with zero
size -- confirmed both by PR 1's before/after screen capture (header and
mode strip render, the body is empty) and by an isolated reproduction here
during Task 3 (a nested ``Screen``'s own descendants report
``region=Region(0, 0, 0, 0)`` despite existing in the DOM).

This screen replaces that architecture with the shared Lab frame
(``lab_frame.LabScreen``): the library rail, detail body and readiness
inspector are the frame's three regions, driven by selection state
(``EvalsSelection``, ``evals_state.py``) instead of a hand-rolled screen
stack. **No ``Screen`` subclass is mounted inside any region here.** Detail
and inspector content is swapped by a screen-level
``refresh(recompose=True)`` on selection change, which tears down and
remounts plain widgets (``Static``/``Button``/``Vertical``) -- never a
``Screen``. ``LabScreen.recompose()`` repopulates the regions and
re-schedules the deferred body afterwards, which is what makes that
selection-driven recompose safe here.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...DB.Evals_DB import ConflictError, EvalsDB
from ...Evals.word_bench.models import PreflightResult
from ...Evals.word_bench.models import Target as WordBenchTarget
from ...Evals.word_bench.runner import CancelToken, CaptureClientLike
from ...Evals.word_bench.storage import _unique_name, duplicate_bench
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ..Evals import sample_bench
from ..Evals.bench_editor import BenchEditor, ClassicTaskDetail
from ..Evals.evals_state import EvalsSelection, EvalsViewModel, SelectionKind
from ..Evals.inspector import EvalsCellInspector, EvalsInspector
from ..Evals.library_rail import RAIL_SECTIONS, LibraryRail
from ..Evals.results_grid import ResultsGrid
from ..Evals.snippet_editor import SnippetEditor
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Lab_Modules.lab_rail_layout import LabRailLayout
from .lab_frame import LabScreen

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvalsScreen(LabScreen):
    """Evals mode: library rail, detail body, readiness inspector -- on the Lab frame."""

    #: Both rails open on a first run. Unlike Models' server list or Speech's
    #: dependency detail, the Evals inspector is where target readiness is
    #: reported -- the reason to look at a bench before running it. Behind a
    #: collapsed handle it is content the user has to know to go find.
    LAB_FIRST_RUN_RAILS = LabRailLayout()

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "evals", **kwargs)
        self._view_model = EvalsViewModel(self._resolve_db(app_instance))
        self._selection = EvalsSelection()
        #: Preflight resolved once per selection, not once per pane.
        #: The frame calls compose_lab_rail/compose_lab_inspector during
        #: _populate_regions and build_lab_body later, from the deferred
        #: mount -- three separate calls where the old single
        #: compose_content() resolved it once and threaded it into both
        #: panes. Without this cache, adopting the frame would silently
        #: reintroduce the duplicate run-group snapshot read that I2
        #: fixed. Cleared wherever the selection changes.
        self._preflight_cache: dict[str, PreflightResult] | None = None
        # Shared with LibraryRail by reference (see its own docstring) so
        # collapsed/expanded rail sections survive a selection-triggered
        # recompose, which constructs a brand-new LibraryRail instance.
        self._rail_open_sections: dict[str, bool] = {
            section_id: True for section_id in RAIL_SECTIONS
        }
        #: DI seam for tests only -- overrides sample_bench.py's default
        #: (real ``WordBenchCaptureClient``) with a fake, mirroring
        #: ``WordBenchRunner``'s own client_factory parameter. ``None`` in
        #: production.
        self._sample_bench_client_factory: Optional[
            Callable[[WordBenchTarget], CaptureClientLike]
        ] = None
        #: True for the duration of one create-and-run flow. Guards against
        #: a second click starting a second worker once a run is genuinely
        #: in flight -- the button is also disabled live (see
        #: ``_set_sample_bench_running_ui``), but a disabled widget not yet
        #: re-rendered, or a message posted directly as this screen's own
        #: tests do, must not be able to race past it. For the tighter
        #: race -- two requests already queued before either dispatches --
        #: it is ``exclusive=True`` on the worker (below), not this flag,
        #: that actually protects: Textual cancels the second worker's Task
        #: before its first step, so the first worker's body (including its
        #: flag-set line) never runs.
        self._sample_bench_running: bool = False
        #: The active run's cooperative cancel token, or ``None`` when no
        #: run is in flight. NOTHING READS THIS TODAY (TASK-861 audited
        #: it): the running-guard above prevents the second-click race
        #: that would otherwise need it, and no Cancel affordance exists
        #: yet in this screen. Kept as a real, threaded seam rather than a
        #: decorative parameter, since ``WordBenchRunner.run`` already
        #: accepts one and a future PR wiring an actual Cancel button (PR
        #: 3c, per this program's own PR numbering) should not need a
        #: second plumbing pass to reach it.
        self._sample_bench_cancel_token: Optional[CancelToken] = None
        #: The selection snapshotted in ``_on_sample_bench_requested`` at
        #: PRESS time, before ``run_worker`` is even called -- same
        #: capture-outside-the-worker rationale as ``_bench_run_task_id``
        #: below (the selection can move before the scheduled worker's
        #: first line actually runs). Unlike a bench-run, a sample bench
        #: does not exist yet when this button is pressed, so there is no
        #: bench id to pin; what a completing worker must not yank the
        #: user away from is wherever they WERE, not a specific bench --
        #: see ``_selection_unmoved_since_launch`` (task-1482 Task 2).
        self._sample_bench_launch_selection: EvalsSelection = EvalsSelection()
        #: True for the duration of one run-existing-bench flow. Same
        #: double-guard rationale as ``_sample_bench_running`` (see that
        #: field's own comment above): this flag stops a second press once
        #: a worker has already set it, while ``exclusive=True`` on the
        #: worker (below) covers the tighter race of two presses already
        #: queued before either dispatches.
        self._bench_run_running: bool = False
        #: The bench (``eval_tasks``) id the in-flight run worker is
        #: running, resolved from the current selection at PRESS time (see
        #: ``_on_primary_action_pressed``) and never re-read from
        #: ``self._selection`` inside the worker -- the selection can move
        #: (another rail click) while the worker is still in flight.
        self._bench_run_task_id: Optional[str] = None
        #: The active run's cooperative cancel token, or ``None`` -- same
        #: no-current-caller status as ``_sample_bench_cancel_token`` above
        #: (no Cancel affordance exists in this screen yet): kept as a
        #: real, threaded seam rather than a decorative parameter.
        self._bench_run_cancel_token: Optional[CancelToken] = None
        #: task-1482 Task 7 fix round 1 (reviewer-found reentrancy): True
        #: from the moment ``_on_delete_bench_pressed`` dispatches
        #: ``_delete_bench_flow`` until ``_apply_bench_deletion`` finishes
        #: (confirmed, cancelled, or erroring out). See that handler's own
        #: docstring for why a synchronous flag -- not ``exclusive=True``
        #: on the worker, unlike the run-bench/sample-bench pattern this
        #: screen uses everywhere else -- is the correct guard here.
        self._bench_delete_pending: bool = False

    def _current_app_config(self) -> dict[str, Any]:
        """The app's loaded settings, read fresh on every recompose (not
        cached in ``__init__``) so a provider configured in Settings after
        this screen first mounted is picked up without a restart."""
        return dict(getattr(self.app_instance, "app_config", None) or {})

    @staticmethod
    def _resolve_db(app_instance: object) -> Optional[EvalsDB]:
        """Find the app's real ``EvalsDB``, or ``None`` if unavailable.

        ``app.py``'s ``_wire_evaluation_services`` already constructs
        ``app_instance.evaluation_orchestrator`` (an ``EvaluationOrchestrator``
        wrapping a real ``EvalsDB`` as ``.db``) at startup; this screen reads
        that existing wiring rather than opening a second database handle.
        ``evaluation_orchestrator`` is ``None`` when that wiring itself
        failed (caught and logged in ``_wire_evaluation_services``), so this
        degrades to ``None`` rather than raising -- ``EvalsViewModel``
        renders an empty (not broken) workbench in that case.
        """
        orchestrator = getattr(app_instance, "evaluation_orchestrator", None)
        return getattr(orchestrator, "db", None)

    def select(self, *, kind: SelectionKind, id: Optional[str] = None) -> None:  # noqa: A002
        """Set the workbench's active selection and refresh dependent panes.

        Public, not just an internal message handler: it is the shell's own
        selection API. ``LibraryRail.EvalsSelectionChanged`` (posted on a
        rail row press) routes here via ``_on_library_selection_changed``
        below, but a caller may also drive selection directly.

        A plain (non-async) method: it only schedules the recompose
        (``refresh(recompose=True)``), it does not await its
        completion -- callers that need the panes settled should
        ``await pilot.pause()`` afterward, mirroring every other
        recompose-driven screen in this app.

        Args:
            kind: The selected object's kind (``SelectionKind`` --
                ``"none"``, ``"bench"``, ``"classic"``, ``"dataset"``, or
                ``"run_group"``).
            id: The selected object's id. Only meaningful for a non-
                ``"none"`` ``kind``; may be ``None`` (e.g. for ``kind=
                "none"``, or a caller clearing the selection).
        """
        self._selection = EvalsSelection(kind=kind, id=id)
        self._preflight_cache = None
        self._register_grid_shortcuts()
        if self.is_mounted:
            self.refresh(recompose=True)

    def _selection_unmoved_since_launch(
        self, launch_selection: EvalsSelection, bench_task_id: Optional[str]
    ) -> bool:
        """True when it is safe for a just-finished background worker
        (``_run_bench_worker``/``_create_sample_bench_worker``) to move the
        screen's selection to the run group it just produced.

        Two cases count as safe, matching what a user would read as "I'm
        still watching this run" rather than "I've moved on":

        1. ``self._selection`` is unchanged from ``launch_selection`` -- the
           selection captured at the moment the run/creation was started
           (``_bench_run_task_id`` for the bench-run worker, ``self.
           _sample_bench_launch_selection`` for the sample-bench worker,
           which has no pre-existing bench to pin against).
        2. The user has since navigated INTO one of ``bench_task_id``'s own
           run groups (e.g. clicked a still-"running" row in the rail while
           the run was in flight, per ``test_rail_run_row_shows_the_
           running_glyph_while_the_run_is_in_flight``) -- moving them to the
           freshly finished run group there is a refresh, not a yank.

        Any other selection means the user navigated somewhere unrelated
        while the worker was running -- once the bench editor holds
        unsaved form state (task-1482 Task 2's own motivation), forcing a
        recompose there would destroy it. The completing worker must
        degrade to a toast-only notification instead of calling
        ``select()`` (task-1482 Task 2).

        A THIRD, independent check overrides both branches above (task-1610):
        if the currently mounted detail pane holds a ``BenchEditor`` whose
        ``is_dirty()`` is ``True``, this returns ``False`` regardless of
        selection identity -- a recompose would destroy that unsaved state
        even when the selection itself never moved. This is deliberately
        NOT limited to ``bench_task_id``'s own editor: the sample-bench
        worker's sharpest case is a user parked on some OTHER bench's
        editor (unrelated to the sample bench just created elsewhere) with
        unsaved edits -- ``self._selection`` reads "unmoved" there (it never
        pointed at the sample bench to begin with), but the mounted editor
        is still real, unsaved, user state a recompose must not touch.
        Queried defensively (``QueryError`` -> not dirty, nothing to
        protect): most selections never mount a ``BenchEditor`` at all.
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            editor = self.query_one("#evals-bench-editor", BenchEditor)
        except QueryError:
            editor = None
        if editor is not None and editor.is_dirty():
            return False

        if self._selection == launch_selection:
            return True
        if bench_task_id and self._selection.kind == "run_group" and self._selection.id:
            group = self._view_model.run_group_by_id(self._selection.id)
            if group is not None and group.get("task_id") == bench_task_id:
                return True
        return False

    def _register_grid_shortcuts(self) -> None:
        """Advertises the results grid's `l`/`b`/`s`/`e` keys (see
        ``results_grid.ResultsGrid.BINDINGS``) through the shared
        ``ShortcutContext`` machinery only while a run group is selected --
        the only selection kind that mounts a ``ResultsGrid`` at all -- so
        the footer never advertises a grid shortcut with no grid on
        screen. `e` (export) is Task 2's addition -- Task 1 deliberately
        left it unbound and unadvertised so this task could claim it
        without a collision.

        Mirrors ``library_screen.py``'s ``_register_footer_shortcuts``: a
        static hint set, re-registered on every selection change rather
        than driven from inside the grid widget itself, since the grid
        does not know when it stops being the active selection (its own
        unmount does not fire a footer-clearing hook).
        """
        if self._selection.kind == "run_group" and self._selection.id:
            self.register_footer_shortcuts(
                source="evals-grid",
                shortcuts=(
                    ("l", "lens"), ("b", "baseline"), ("s", "sort"), ("e", "export"),
                ),
            )
        else:
            self.clear_footer_shortcuts(source="evals-grid")

    @on(LibraryRail.EvalsSelectionChanged)
    def _on_library_selection_changed(
        self, event: LibraryRail.EvalsSelectionChanged
    ) -> None:
        event.stop()
        self.select(kind=event.selection.kind, id=event.selection.id)

    @on(BenchEditor.Saved)
    def _on_bench_editor_saved(self, event: BenchEditor.Saved) -> None:
        """A successful `BenchEditor` Save re-selects the same bench --
        `select()`'s recompose reloads the form from what `save_bench`
        actually persisted (see `BenchEditor.Saved`'s own docstring for why
        that can differ from what was typed, e.g. `_clean_task_name`'s
        control-character strip), and refreshes the rail row and inspector
        alongside it for free, the same way any other selection change
        does."""
        event.stop()
        self.select(kind="bench", id=event.bench_id)

    @on(BenchEditor.CreateTargetRequested)
    async def _on_bench_create_target_requested(
        self, event: BenchEditor.CreateTargetRequested
    ) -> None:
        """Creates a real `eval_models` row for `bench_editor.py`'s
        "+ New target" mini-form -- ALWAYS rendered there (task-1611 T2),
        not only in the zero-`llama_cpp`-models state -- and stages it on
        the mounted `BenchEditor`. See that message class's own docstring
        for why `bench_editor.py` cannot make this call itself (the
        source-scan pin against the provider client/runner imports).

        Calls `EvalsDB.create_model` DIRECTLY (task-1611 T2) rather than
        `sample_bench.resolve_sample_target`, which this handler used
        exclusively before this task: that function reuses an already-
        registered `llama_cpp` row FIRST, before ever minting a new one --
        exactly wrong once this control's whole point is minting an
        ADDITIONAL, possibly differently-steered target even when one (or
        several) already exist. `configured_llama_cpp_url`/
        `configured_llama_cpp_model_id` (the same config-only, no-network
        reads `resolve_sample_target` itself uses internally) resolve the
        endpoint and model id instead.

        A blank/whitespace-only `event.name` auto-names via
        `storage._unique_name(sample_bench.BENCH_EDITOR_TARGET_NAME)` --
        the SAME base name/convention the old zero-models-only flow always
        used for its one auto-created row. A NON-blank name is used
        VERBATIM (never uniqued) so an intentional collision surfaces as
        the `ConflictError` it is, rather than being silently suffixed
        into a different row than the one the user asked to create.

        `event.prefix`/`event.system_prompt` are already mutually
        exclusive and already blank-normalized to `None` by
        `bench_editor.py`'s own `_on_create_target_pressed` (only one
        steering `Input` is ever mounted at a time) -- this handler only
        decides which non-`None` one becomes a `config` key; an empty
        `config` (`{}`) is what an unsteered target's row already gets
        everywhere else in this codebase (`EvalsDB.create_model`'s own
        `config or {}` default).

        A plain DB read/write, not a network call -- run inline, no
        worker, mirroring `BenchEditor._on_save_pressed`'s own synchronous
        `save_bench` call just one pane over.
        """
        event.stop()
        db = self._view_model.db
        if db is None:
            return
        app_config = self._current_app_config()
        if sample_bench.configured_llama_cpp_url(app_config) is None:
            self.app_instance.notify(
                "No llama.cpp server is configured; set one in Settings "
                "first.",
                severity="error",
                markup=False,
            )
            return
        model_id = sample_bench.configured_llama_cpp_model_id(app_config) or "default"
        typed_name = event.name.strip() if event.name else ""
        name = event.name if typed_name else _unique_name(sample_bench.BENCH_EDITOR_TARGET_NAME)
        config: dict[str, str] = {}
        if event.prefix:
            config["prefix"] = event.prefix
        if event.system_prompt:
            config["system_prompt"] = event.system_prompt
        try:
            new_id = db.create_model(
                name=name, provider="llama_cpp", model_id=model_id, config=config
            )
        except ConflictError as exc:
            self.app_instance.notify(str(exc), severity="error", markup=False)
            return
        model_row = db.get_model(new_id)
        if model_row is None:
            # Defensive only: create_model just returned this id.
            return
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            editor = self.query_one(BenchEditor)
        except QueryError:
            # Defensive only: this handler only ever runs from a press on
            # a button the mounted BenchEditor itself composed.
            return
        await editor.stage_target(model_row)

    @on(LibraryRail.SampleBenchRequested)
    def _on_sample_bench_requested(
        self, event: LibraryRail.SampleBenchRequested
    ) -> None:
        """Creates and runs the one-click sample bench (see
        ``sample_bench.py``). Real DB writes plus a real HTTP call (in
        production) -- run as a worker, never inline in a message handler,
        per CLAUDE.md's "Workers for operations >100ms" rule.

        Two guards cover two different race windows. If two requests are
        already queued before either dispatches, both see
        ``_sample_bench_running`` as ``False`` and both reach
        ``run_worker(exclusive=True, ...)``; it is ``exclusive=True`` that
        protects there, cancelling the second worker's Task before it takes
        its first step, so only one worker body (and one flag-set) ever
        runs. Once a worker IS running and has set the flag, THIS check is
        what stops a later request from calling ``run_worker`` again --
        without it, that call would cancel the already-running worker via
        the same ``exclusive`` group after it has done real work, abandoning
        its in-flight DB rows (see
        ``sample_bench._mark_orphaned_runs_cancelled`` for the cleanup that
        path needs).

        A THIRD check, ``_bench_run_running``, closes a different race: PR
        #1113 review (Qodo, seconding whole-branch review Note 6) found the
        sample-bench worker and the bench-run worker (``_run_bench_worker``,
        started from ``_on_primary_action_pressed``) were only ever guarded
        against THEMSELVES -- each lived in its own ``exclusive`` group, so
        neither worker's ``exclusive=True`` cancelled the other, and a press
        of one while the other was genuinely in flight started two REAL,
        overlapping runs (interleaved toasts, last-wins completion
        ``select()``). The recompose-time UI already disables both controls
        while EITHER flag is set (see ``_primary_action_state``'s and
        ``LibraryRail.sample_bench_running``'s own in-flight branches), but
        that alone does not stop a stale-render/queued-press race from
        reaching this handler -- this cross-check is the same belt this
        function's OTHER two guards already provide for the same-worker
        case, just against the other worker.

        The worker is handed as a CALLABLE, not a pre-built coroutine:
        ``exclusive=True`` cancels the superseded worker's Task before its
        first step, and a coroutine object constructed at the call site is
        then never awaited at all (``RuntimeWarning: coroutine ... was
        never awaited``). Textual only calls the callable when the worker
        actually starts, so in the very race this docstring describes no
        orphan coroutine is created.

        ``self._selection`` is also snapshotted into
        ``self._sample_bench_launch_selection`` HERE, before ``run_worker``
        is even called -- not re-read from ``self._selection`` inside the
        worker, mirroring ``_on_primary_action_pressed``'s own
        ``_bench_run_task_id`` capture and for the identical reason: the
        selection can move before the scheduled worker's first line
        actually runs. The completing worker reads this snapshot to decide
        whether it is still safe to move the selection to the new run
        group, or whether the user has navigated elsewhere and a recompose
        there would yank them (see ``_selection_unmoved_since_launch``,
        task-1482 Task 2).
        """
        event.stop()
        if self._sample_bench_running or self._bench_run_running:
            return
        self._sample_bench_launch_selection = self._selection
        self.run_worker(
            self._create_sample_bench_worker,
            exclusive=True,
            group="evals-sample-bench",
        )

    async def _create_sample_bench_worker(self) -> None:
        """Creates and runs the one-click sample bench (see
        ``sample_bench.create_and_run_sample_bench``).

        On success, ``select(run_group)`` ONLY when
        ``_selection_unmoved_since_launch`` says the screen's current
        selection is still ``self._sample_bench_launch_selection`` (the
        selection snapshotted in ``_on_sample_bench_requested`` at press
        time) or has since moved into the freshly created bench's own run
        groups. Otherwise the run/creation is not lost -- it is still in
        the DB and the Runs section -- but a completing background worker
        must not force a recompose that would yank the user from wherever
        they navigated to mid-flight, e.g. into a half-edited bench editor
        form (task-1482 Task 2's own motivation).
        """
        app_config = self._current_app_config()
        cancel_token = CancelToken()
        self._sample_bench_running = True
        self._sample_bench_cancel_token = cancel_token
        self._set_sample_bench_running_ui()
        result = None
        try:
            result = await sample_bench.create_and_run_sample_bench(
                self._view_model,
                app_config,
                client_factory=self._sample_bench_client_factory,
                progress=self._on_sample_bench_progress,
                cancel_token=cancel_token,
            )
        except asyncio.CancelledError:
            # sample_bench.py's own except-and-re-raise already marked any
            # created run rows "cancelled" before this propagated here --
            # log and let it continue propagating; swallowing a
            # CancelledError is its own bug (Textual's worker bookkeeping
            # needs to observe the real cancellation).
            logger.info("Sample bench worker was cancelled.")
            raise
        except Exception as exc:
            # Type only: persistent exception diagnostics can serialize frame
            # locals, which here include app config and user-authored datasets.
            logger.warning(
                "Sample bench creation failed (exception_category={}).",
                type(exc).__name__,
            )
            # markup=False: `exc` can carry user-controlled text (e.g. a
            # dataset name derived from an imported filename stem) and
            # `notify()` defaults to markup=True -- unbalanced markup in
            # that text (a bare `[/]`) raises MarkupError inside the toast
            # renderer and crashes the whole app. See the identical fix on
            # `_run_bench_worker`'s two notify() calls below.
            self.app_instance.notify(
                f"Could not create the sample bench: {exc}",
                severity="error",
                markup=False,
            )
        finally:
            self._sample_bench_running = False
            self._sample_bench_cancel_token = None
            self._reset_sample_bench_running_ui()
        if result is not None:
            if self._selection_unmoved_since_launch(
                self._sample_bench_launch_selection, result.task_id
            ):
                self.app_instance.notify(
                    "Sample bench created and run.",
                    severity="information",
                    markup=False,
                )
                self.select(kind="run_group", id=result.run_group_id)
            else:
                # The user navigated elsewhere while the run was in flight
                # -- see `_selection_unmoved_since_launch`'s own docstring.
                # The bench and run group both still exist; only the
                # auto-navigate is skipped.
                self.app_instance.notify(
                    "Sample bench created and run — see the Runs section.",
                    severity="information",
                    markup=False,
                )

    def _on_sample_bench_progress(self, done: int, total: int) -> None:
        """``sample_bench.ProgressFn`` -- called synchronously from within
        ``WordBenchRunner.run``'s own coroutine (this worker's, not a
        separate OS thread), so mutating widgets directly here is safe,
        the same way ``_on_grid_cell_focused`` mutates the inspector
        directly rather than needing ``call_from_thread``."""
        self._set_sample_bench_running_ui(done=done, total=total)

    def _set_sample_bench_running_ui(self, *, done: int = 0, total: int = 0) -> None:
        """Disables the "Create sample bench" button and gives it a live
        running label for as long as a run is in flight -- see the class
        docstring note above on why a disabled-but-not-yet-rerendered
        button is not by itself a sufficient guard against a second click,
        only a visible signal that one is already running.
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            button = self.query_one("#evals-create-sample-bench", Button)
        except QueryError:
            return
        button.disabled = True
        button.label = (
            f"Running sample bench… ({done}/{total})" if total else "Creating sample bench…"
        )

    def _reset_sample_bench_running_ui(self) -> None:
        """Restores the button after a run ends -- on BOTH the success and
        failure paths.

        TASK-1478 made "Create sample bench" a persistent control (rendered
        at the top of the Benches section regardless of whether any benches
        exist yet -- see ``library_rail.py``'s module docstring, "Creation
        affordances are not empty-only"), so the claim this docstring used
        to make -- that the success path's ``self.select(...)`` recompose
        "replaces this button with the bench's own row" -- is no longer
        true: a fresh ``LibraryRail`` recompose still renders the SAME
        button, at the same id, still needing to be un-disabled and
        re-labelled. The ``QueryError`` guard remains for the case the
        button genuinely isn't in the DOM at all (no configured provider,
        so the rail renders "Open Settings" instead)."""
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            button = self.query_one("#evals-create-sample-bench", Button)
        except QueryError:
            return
        button.disabled = False
        button.label = "Create sample bench"

    @on(ResultsGrid.CellFocused)
    def _on_grid_cell_focused(self, event: ResultsGrid.CellFocused) -> None:
        """Forwards a focused grid cell to the inspector pane's
        ``EvalsCellInspector`` -- a targeted ``show_cell()`` call against
        an already-mounted widget, never a screen recompose (see
        ``results_grid.py``'s module docstring for why that distinction
        matters on every arrow-key press)."""
        event.stop()
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches _footer_status's own local import

        try:
            inspector = self.query_one(EvalsCellInspector)
        except QueryError:
            return
        inspector.show_cell(event)

    @on(Button.Pressed, "#evals-primary-action")
    def _on_primary_action_pressed(self, event: Button.Pressed) -> None:
        """Runs the selected bench (see ``sample_bench.run_existing_bench``).

        Mirrors ``_on_sample_bench_requested``'s guard rationale exactly --
        see that method's own docstring for the full three-part
        explanation, repeated here only in brief. If two presses are
        already queued before either dispatches, both see
        ``_bench_run_running`` as ``False`` and both reach
        ``run_worker(exclusive=True, ...)``; it is ``exclusive=True`` that
        protects there, cancelling the second worker's Task before it takes
        its first step, so only one worker body (and one flag-set) ever
        runs. Once a worker IS running and has set the flag, THIS check is
        what stops a later press from calling ``run_worker`` again --
        without it, that call would cancel the already-running worker via
        the same ``exclusive`` group after it has done real work,
        abandoning its in-flight DB rows (see
        ``sample_bench._mark_orphaned_runs_cancelled`` for the cleanup that
        path needs). A THIRD check, ``_sample_bench_running``, closes the
        cross-worker race PR #1113 review found: this button and
        ``#evals-create-sample-bench`` live in separate ``exclusive``
        groups, so without this cross-check a press here while the SAMPLE
        bench worker is in flight would start a second, genuinely
        overlapping run.

        The selected bench id is resolved and stored on the instance HERE,
        not re-read from ``self._selection`` inside the worker -- selection
        can move (another rail click) while the worker is in flight, and
        the worker must keep running the bench it was actually launched
        against.
        """
        event.stop()
        if self._bench_run_running or self._sample_bench_running:
            return
        selection = self._selection
        if selection.kind != "bench" or not selection.id:
            # Defensive only: `_primary_action_state` keeps the button
            # disabled (so Textual never emits `Pressed` at all) for every
            # selection kind but a found bench.
            return
        self._bench_run_task_id = selection.id
        self.run_worker(
            self._run_bench_worker,
            exclusive=True,
            group="evals-run-bench",
        )

    async def _run_bench_worker(self) -> None:
        """Runs ``self._bench_run_task_id`` via
        ``sample_bench.run_existing_bench``. Mirrors
        ``_create_sample_bench_worker`` structure exactly -- see that
        method's own comments for the parts not re-explained here,
        including the "does not auto-select on completion once the user
        has navigated elsewhere" rule (``_selection_unmoved_since_launch``,
        task-1482 Task 2): here the launch selection to compare against is
        always ``EvalsSelection(kind="bench", id=task_id)``, since
        ``_on_primary_action_pressed`` only ever dispatches this worker
        for a selected bench.
        """
        app_config = self._current_app_config()
        task_id = self._bench_run_task_id
        cancel_token = CancelToken()
        self._bench_run_running = True
        self._bench_run_cancel_token = cancel_token
        self._set_bench_run_running_ui()
        result = None
        try:
            result = await sample_bench.run_existing_bench(
                self._view_model,
                app_config,
                task_id,
                client_factory=self._sample_bench_client_factory,
                progress=self._on_bench_run_progress,
                cancel_token=cancel_token,
            )
        except asyncio.CancelledError:
            # run_existing_bench's own except-and-re-raise already marked
            # any of this bench's still-"running" run rows "cancelled"
            # before this propagated here -- log and let it continue
            # propagating; swallowing a CancelledError is its own bug
            # (Textual's worker bookkeeping needs to observe the real
            # cancellation).
            logger.info("Bench run worker was cancelled.")
            raise
        except Exception as exc:
            # Type only: persistent exception diagnostics can serialize frame
            # locals, including the selected dataset id and current app config.
            logger.warning(
                "Bench run failed (exception_category={}).",
                type(exc).__name__,
            )
            # markup=False: `exc` can carry user-controlled text -- e.g.
            # `sample_bench._load_snippets` raises `RuntimeError(f"Dataset
            # {name!r} has no snippets to run.")`, and an imported dataset's
            # name defaults to the imported filename's stem, so a file named
            # `notes[/].txt` puts live markup straight into this string.
            # `notify()` defaults to markup=True; unbalanced markup (a bare
            # `[/]`) raises MarkupError inside the toast renderer and takes
            # down the whole app -- this path was unreachable before this
            # button was wired up (it was always disabled), so it is new
            # here.
            self.app_instance.notify(
                f"Could not run the bench: {exc}",
                severity="error",
                markup=False,
            )
        finally:
            self._bench_run_running = False
            self._bench_run_cancel_token = None
            self._reset_bench_run_running_ui()
        if result is not None:
            launch_selection = EvalsSelection(kind="bench", id=task_id)
            if self._selection_unmoved_since_launch(launch_selection, task_id):
                # markup=False for uniformity with the error toast above --
                # this string is static today, but pinning it keeps the
                # pair consistent if it ever starts interpolating the
                # bench name.
                self.app_instance.notify(
                    "Bench run finished.", severity="information", markup=False
                )
                self.select(kind="run_group", id=result.run_group_id)
            else:
                # The user navigated elsewhere while the run was in flight
                # -- see `_selection_unmoved_since_launch`'s own docstring.
                # The run group still exists; only the auto-navigate is
                # skipped.
                self.app_instance.notify(
                    "Bench run finished — see the Runs section.",
                    severity="information",
                    markup=False,
                )

    def _on_bench_run_progress(self, done: int, total: int) -> None:
        """``sample_bench.ProgressFn`` -- called synchronously from within
        ``WordBenchRunner.run``'s own coroutine (this worker's, not a
        separate OS thread), so mutating the button directly here is safe,
        mirroring ``_on_sample_bench_progress``."""
        self._set_bench_run_running_ui(done=done, total=total)

    def _set_bench_run_running_ui(self, *, done: int = 0, total: int = 0) -> None:
        """Disables the primary-action button and gives it a live running
        label for as long as a run is in flight -- see
        ``_set_sample_bench_running_ui``'s own note on why a disabled-but-
        not-yet-rerendered button is only a visible signal, not by itself a
        sufficient guard against a second press."""
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            button = self.query_one("#evals-primary-action", Button)
        except QueryError:
            return
        button.disabled = True
        button.label = f"Running… ({done}/{total})" if total else "Running…"

    def _reset_bench_run_running_ui(self) -> None:
        """Restores the primary-action button after a run ends, from
        ``_primary_action_state()`` -- the current selection's own fresh
        label/disabled/tooltip, not a hardcoded constant, since (unlike
        ``_reset_sample_bench_running_ui``'s "Create sample bench") the
        ready-state label here is per-bench (``f"Run {name}"``). A no-op
        (via the same ``QueryError`` guard) on the success path, where
        ``self.select(...)`` immediately recomposes the inspector pane and
        replaces this button entirely -- this only matters on the failure
        path, where the SAME button instance survives and must not be left
        permanently disabled with a stale "Running…" label."""
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            button = self.query_one("#evals-primary-action", Button)
        except QueryError:
            return
        label, disabled, tooltip = self._primary_action_state()
        button.disabled = disabled
        button.label = label
        button.tooltip = tooltip

    def _bench_delete_disabled_reason(self, bench_id: Optional[str]) -> Optional[str]:
        """Why ``#evals-delete-bench`` should be disabled for ``bench_id``,
        or ``None`` when it's safe to delete.

        Gated ONLY on ``_bench_run_running`` for THIS bench -- unlike
        ``_primary_action_state``, which also blocks while the SAMPLE
        bench worker is running. That extra gate exists there because a
        completing sample-bench worker eventually selects a brand-new
        bench the primary action could otherwise race a second run
        against; the sample-bench worker never touches an *existing*
        bench id (it creates its own, not-yet-selected one) until it
        finishes, so it must not block deleting some OTHER, unrelated,
        already-selected bench here.
        """
        if bench_id and self._bench_run_running and self._bench_run_task_id == bench_id:
            return "A run of this bench is in flight."
        return None

    @on(Button.Pressed, "#evals-duplicate-bench")
    def _on_duplicate_bench_pressed(self, event: Button.Pressed) -> None:
        """Duplicates the selected bench via ``storage.duplicate_bench``
        (Task 3) -- a plain ``eval_tasks`` insert, never a network call, so
        this runs in-widget with no worker, mirroring ``library_rail.py``'s
        ``_create_new_bench``/``_create_new_dataset`` (the same "no worker
        for a bare DB write" convention).

        Catches broad ``Exception``, not ``duplicate_bench``'s own
        narrower ``RuntimeError`` (which it raises only for a missing/
        soft-deleted source) -- controller ruling from Task 3's review: a
        CORRUPT legacy bench (task-1132's lenient ``load_bench`` still
        loads it, but ``BenchConfig``/``save_bench`` downstream can raise
        their own native diagnostic exception for a shape ``load_bench``
        never normalised) must still toast here rather than crash this
        screen, matching every other DB-write handler in this file (see
        ``_run_bench_worker``'s own broad catch above).
        """
        event.stop()
        selection = self._selection
        if selection.kind != "bench" or not selection.id:
            # Defensive only: this button is composed only inside the
            # resolved-bench branch of `_compose_inspector_pane`.
            return
        db = self._view_model.db
        if db is None:
            self.app_instance.notify(
                "The evaluation service is unavailable.", severity="error"
            )
            return
        try:
            new_id = duplicate_bench(db, selection.id)
        except Exception as exc:
            logger.opt(exception=True).warning("Could not duplicate bench.")
            # markup=False: `exc` can carry the source bench's own
            # free-text name -- same hazard `_run_bench_worker`'s own
            # error toast documents.
            self.app_instance.notify(
                f"Could not duplicate the bench: {exc}",
                severity="error",
                markup=False,
            )
            return
        new_bench = self._view_model.bench_by_id(new_id)
        new_name = str(new_bench.get("name")) if new_bench else "the new bench"
        self.select(kind="bench", id=new_id)
        self.app_instance.notify(
            f"Duplicated as {new_name}.", severity="information", markup=False
        )

    @on(Button.Pressed, "#evals-delete-bench")
    def _on_delete_bench_pressed(self, event: Button.Pressed) -> None:
        """Starts the confirm-then-delete flow for the selected bench.

        Dispatches a worker: ``push_screen_wait`` raises ``NoActiveWorker``
        outside one (see ``ConsoleShellScreen.confirm_navigation``'s
        identical note in ``chat_screen.py``). The bench id and name are
        resolved here, before the worker's first line runs -- mirrors
        ``_on_primary_action_pressed``'s own capture-outside-the-worker
        rationale (the selection can move while the confirm dialog is
        still up).

        ``_bench_delete_pending`` guards a race review reproduced directly
        (screen stack depth 2 -> 4): two ``Button.Pressed`` messages queued
        with no intervening ``await`` both reach this synchronous handler
        before either's ``run_worker`` call has taken its first step, so
        without a check-and-set flag BOTH calls pass the (unrelated)
        in-flight-run guard above and each starts its own
        ``_delete_bench_flow`` worker -- pushing two ``ConfirmationDialog``s
        onto the screen stack.

        This is deliberately a plain flag, NOT ``exclusive=True`` on the
        worker below, unlike ``_on_primary_action_pressed``'s identical-
        looking double-press race (see that handler's own docstring for the
        contrast). There, ``exclusive=True`` is correct: Textual cancels a
        superseded worker's ``Task`` before its first step, so only one
        worker body -- and the DB write it performs -- ever runs. Here that
        would be actively wrong: ``_delete_bench_flow`` awaits
        ``self.app.push_screen_wait(...)``, which internally awaits
        ``asyncio.shield(future)`` -- shielding the WAIT itself from
        cancellation, not the widget it already pushed. Cancelling this
        worker's Task via an exclusive group after it has already pushed
        its ``ConfirmationDialog`` would tear down the coroutine waiting on
        that dialog's result while leaving the dialog itself mounted on the
        screen stack -- a user's Confirm/Cancel click would land on a
        dialog whose owning code no longer exists to act on it, a silent
        no-op indistinguishable from a hang. A synchronous flag, checked
        and set here before the FIRST worker is ever dispatched, avoids
        needing to cancel anything: the second queued press sees the flag
        already set and returns before calling ``run_worker`` at all, so
        only one worker -- and one dialog -- is ever created. Cleared in a
        ``finally`` inside ``_apply_bench_deletion`` (see that method's own
        docstring) once the flow fully resolves, whichever way it resolves.
        """
        event.stop()
        selection = self._selection
        if selection.kind != "bench" or not selection.id:
            return
        if self._bench_delete_disabled_reason(selection.id):
            # Defensive only: `_compose_inspector_pane` already disables
            # the button for this case, and a disabled Textual `Button`
            # never emits `Pressed`.
            return
        if self._bench_delete_pending:
            return
        self._bench_delete_pending = True
        bench = self._view_model.bench_by_id(selection.id)
        name = str(bench.get("name")) if bench else "Untitled bench"
        self.run_worker(
            self._delete_bench_flow(selection.id, name),
            group="evals-delete-bench",
        )

    async def _delete_bench_flow(self, task_id: str, name: str) -> None:
        """Confirms, then applies (via ``_apply_bench_deletion`` below)
        deleting ``task_id``.

        ``escape_markup(name)``: ``ConfirmationDialog.compose`` renders
        ``message`` through a plain ``Label`` (``markup`` left at its
        Textual-matching default of ``True``), so an unescaped bench name
        here would hit the same bare-``[/]``-crashes-the-app hazard
        ``_primary_action_state``'s own ``name`` computation documents.
        """
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Delete bench?",
                message=f'Delete "{escape_markup(name)}"? This can\'t be undone.',
                confirm_label="Delete bench",
                cancel_label="Cancel",
            )
        )
        self._apply_bench_deletion(bool(confirmed), task_id)

    def _apply_bench_deletion(self, confirmed: bool, task_id: str) -> None:
        """Applies the confirm dialog's own result.

        Public-shaped (a plain ``(confirmed, task_id)`` signature, not
        name-mangled) so tests call this directly with
        ``confirmed=True``/``False``, bypassing the modal (and the worker
        above) entirely -- mirrors ``snippet_editor.py``'s
        ``_handle_import_file_selected`` (the ``FileOpen`` dialog's own
        callback): driving a real modal in a test is expensive, and this
        is the one place the dialog's yes/no decision reaches code.

        The whole body runs inside a ``try/finally`` that clears
        ``_bench_delete_pending`` -- the single-flight guard
        ``_on_delete_bench_pressed`` sets before ever dispatching
        ``_delete_bench_flow`` (see that method's own docstring for the
        race this closes). Every return path here -- cancelled, no DB,
        delete failed, or genuinely completed -- is "the flow is over, a
        fresh press should be allowed again," so the reset lives in
        ``finally`` rather than being duplicated at each ``return``. Tests
        that call this method directly, bypassing ``_on_delete_bench_
        pressed`` entirely, harmlessly reset a flag that was never set.
        """
        try:
            if not confirmed:
                return
            db = self._view_model.db
            if db is None:
                self.app_instance.notify(
                    "The evaluation service is unavailable.", severity="error"
                )
                return
            try:
                db.delete_task(task_id)
            except Exception as exc:
                logger.opt(exception=True).warning("Could not delete bench.")
                self.app_instance.notify(
                    f"Could not delete the bench: {exc}",
                    severity="error",
                    markup=False,
                )
                return
            self.select(kind="none")
            # Provenance rule (task-1482 plan, "Delete vs runs"): deleting a
            # bench does not cascade its run history -- `EvalsDB.delete_task`
            # only soft-deletes the `eval_tasks` row; `list_runs`/`get_run`'s
            # own `JOIN eval_tasks` (unfiltered on `t.deleted_at`) still
            # resolves the runs, and `EvalsViewModel.run_groups()` reads
            # `list_runs()` directly, never `_all_tasks()` (which DOES filter
            # deleted tasks) -- so the Runs section keeps listing them, and
            # opening one still renders the grid. This toast is the only
            # place a user learns that on purpose.
            self.app_instance.notify(
                "Bench deleted. Its runs remain in the Runs section.",
                severity="information",
                markup=False,
            )
        finally:
            self._bench_delete_pending = False

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return the Evals destination header copy.

        Returns:
            Header state. The status is constant because nothing on this
            screen is a whole-destination readiness signal -- per-target
            readiness is the inspector's job, and a badge that never changes
            would only be decoration wearing a status label.
        """
        return WorkbenchHeaderState(
            title="Evals",
            subtitle="Run and review evaluation jobs.",
            status="ready",
        )

    def compose_lab_rail(self) -> ComposeResult:
        """Yield the library rail.

        A fresh ``LibraryRail`` per compose is deliberate: open/collapsed
        section state lives in ``self._rail_open_sections`` and is shared by
        reference, so it survives the instance being rebuilt.
        """
        yield LibraryRail(
            self._view_model,
            selection=self._selection,
            open_sections=self._rail_open_sections,
            app_config=self._current_app_config(),
            # Whole-branch review: gated on EITHER worker, not just the
            # sample-bench one -- "a run is in flight" is the condition
            # that makes starting a SECOND one (via this button) a stale-
            # button trap, regardless of which worker owns the first run.
            # See `_primary_action_state`'s own in-flight branch just
            # below for the identical rationale on the primary action.
            sample_bench_running=self._sample_bench_running or self._bench_run_running,
            id="evals-library-pane",
        )

    def build_lab_body(self) -> Vertical:
        """Build the detail pane.

        Returns:
            A ``Vertical`` holding this selection's detail widgets. Built as
            a factory, not composed inline, because the frame mounts the body
            after first paint -- and a widget instance would not survive a
            ``recompose=True`` while a factory does.
        """
        return Vertical(
            *self._compose_detail_pane(self._preflight_for_selection()),
            id="evals-detail-pane",
        )

    def compose_lab_inspector(self) -> ComposeResult:
        """Yield the readiness inspector for the current selection.

        Wrapped in ``#evals-inspector-pane`` rather than yielded flat into
        the frame's region: that id is the inspector's stable selector and
        keeps the ``ds-inspector`` surface styling. The old
        ``destination-workbench-pane`` class is dropped -- sizing is the
        frame region's job now.
        """
        yield Vertical(
            *self._compose_inspector_pane(self._preflight_for_selection()),
            id="evals-inspector-pane",
            classes="ds-inspector",
        )

    def _preflight_for_selection(self) -> dict[str, PreflightResult]:
        """The current selection's readiness map, resolved once per selection.

        ``{}`` for every selection kind but ``"bench"`` -- no other kind's
        panes read it. Memoised because the body, rail and inspector are now
        composed by three separate frame hooks at three different times; see
        ``_preflight_cache``.

        Returns:
            Target id -> readiness, or an empty mapping.
        """
        if self._preflight_cache is not None:
            return self._preflight_cache
        selection = self._selection
        if selection.kind != "bench" or not selection.id:
            self._preflight_cache = {}
        else:
            self._preflight_cache = self._view_model.preflight_for_bench(selection.id)
        return self._preflight_cache

    def _compose_detail_pane(
        self, preflight: dict[str, PreflightResult]
    ) -> ComposeResult:
        selection = self._selection
        yield Static("Detail", classes="destination-section evals-pane-title")

        if selection.kind == "bench":
            bench = self._view_model.bench_by_id(selection.id) if selection.id else None
            if bench is None:
                yield Static(
                    "This bench could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            yield BenchEditor(
                self._view_model, selection.id, preflight, id="evals-bench-editor"
            )
            return

        if selection.kind == "classic":
            task = (
                self._view_model.classic_task_by_id(selection.id)
                if selection.id
                else None
            )
            if task is None:
                yield Static(
                    "This task could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            yield ClassicTaskDetail(self._view_model, task, id="evals-classic-detail")
            return

        if selection.kind == "dataset":
            dataset = (
                self._view_model.dataset_by_id(selection.id) if selection.id else None
            )
            if dataset is None:
                yield Static(
                    "This dataset could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            yield SnippetEditor(
                self._view_model, selection.id, id="evals-snippet-editor"
            )
            return

        if selection.kind == "run_group":
            group = (
                self._view_model.run_group_by_id(selection.id)
                if selection.id
                else None
            )
            if group is None:
                yield Static(
                    "This run could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            # ResultsGrid renders its own header (bench name, prompt mode,
            # effective K, cell/failure counts) -- see results_grid.py's
            # _render_header -- so no separate name/count Statics are
            # yielded here; that would restate the same facts from a
            # SECOND, unsynchronized source (this pane reads `group` from
            # `EvalsViewModel.run_groups()`'s pivot, the grid reads its own
            # `load_grid` snapshot -- two reads of related but distinct
            # data that must not drift against each other in the UI).
            yield ResultsGrid(
                self._view_model, selection.id, id="evals-results-grid"
            )
            return

        yield Static(
            self._empty_detail_text(),
            id="evals-detail-empty",
            markup=False,
        )

    def _empty_detail_text(self) -> str:
        """Copy for the ``"none"``-selection Detail pane.

        TASK-1076: the old, single wording ("Select a bench, dataset, or
        run in the library rail...") is unactionable at the one moment it
        is guaranteed to show -- a first launch, where the rail has
        nothing to select at all. Distinguishes that genuinely-empty
        library (nothing in any of the three rail sections) from the more
        common "none" case -- a user who deleted their selection, or
        clicked empty rail padding, while real rows still exist -- where
        the original sentence is still the correct instruction.

        The emptiness check itself lives in
        ``EvalsViewModel.library_is_empty()``, not inline here: this
        method reruns on every selection change (``select()`` ->
        ``refresh(recompose=True)``), so a single, minimal-read helper
        matters more here than in a one-shot call site -- see that
        method's docstring for why it costs one task read (not two) and a
        1-row dataset existence check (not a 500-row page).
        """
        if self._view_model.library_is_empty():
            return (
                "Nothing here yet. Create a sample bench in the Catalog "
                "rail to get started — it builds a dataset and a run for "
                "you in one step."
            )
        return (
            "Select a bench, dataset, or run in the Catalog rail to see "
            "its detail here."
        )

    def _compose_inspector_pane(
        self, preflight: dict[str, PreflightResult]
    ) -> ComposeResult:
        yield Static("Inspector", classes="destination-section evals-pane-title")
        selection = self._selection

        if selection.kind == "bench":
            bench = (
                self._view_model.bench_by_id(selection.id) if selection.id else None
            )
            if bench is not None:
                yield EvalsInspector(
                    self._view_model,
                    selection.id,
                    preflight,
                    id="evals-inspector-bench",
                )
                # Duplicate/Delete are composed further down, AFTER
                # `#evals-primary-action` -- see the comment there (task-
                # 1482 Task 7 fix round 1) for why.

        if selection.kind == "classic":
            # Classic tasks are read-only in this workbench (see the design
            # spec's "Classic tasks" section and BenchEditor's
            # `ClassicTaskDetail`, which carries the deferral sentence) --
            # no run control is rendered here at all, not even a disabled
            # one; `_primary_action_state()` is never consulted for this
            # kind.
            return

        if selection.kind == "run_group":
            group = (
                self._view_model.run_group_by_id(selection.id)
                if selection.id
                else None
            )
            if group is not None:
                # Focused-cell detail (full top-K + probe table), updated
                # by `_on_grid_cell_focused` as the grid's cell cursor
                # moves -- see that handler and results_grid.py's module
                # docstring for why this is a targeted `show_cell()` call,
                # never a recompose. The primary action button below still
                # renders (with its existing "already completed" reason,
                # unchanged from Task 3) beneath it.
                yield EvalsCellInspector(id="evals-cell-inspector")

        label, disabled, tooltip = self._primary_action_state()
        if disabled and tooltip:
            # TASK-1076: a disabled Textual `Button` never emits `Pressed`
            # -- a click on it produces no toast, no inline message, no
            # state change, which is exactly the "silent no-op" UAT found.
            # `tooltip=` below is real (screen-reader/mouse-hover users
            # still get it) but it is the ONLY place the reason lived
            # before this, and a hover-only explanation is not reachable
            # from a keyboard-only session. Mirrors `EvalsInspector`'s own
            # readiness convention just above (and reachable through the
            # SAME `.ds-status-badge`/`evals-status-blocked` classes a
            # Blocked target row uses, in `_status_css_class`) rather than
            # inventing a second "why can't I do this" vocabulary: a
            # status badge naming the action, plus a callout stating the
            # reason -- always visible, never conditional on a mouse.
            yield Static(
                f"{label}: Blocked",
                id="evals-primary-action-status",
                classes="ds-status-badge evals-status-blocked",
                markup=False,
            )
            yield Static(
                tooltip,
                id="evals-primary-action-reason",
                classes="ds-recovery-callout",
                markup=False,
            )
        yield Button(
            label,
            id="evals-primary-action",
            disabled=disabled,
            tooltip=tooltip,
        )

        # task-1482 Task 7 fix round 1: composed AFTER `#evals-primary-
        # action`, not before it -- the design spec's inspector mock
        # orders these `[ Run bench ]` then `[ Duplicate ]` then
        # `[ Delete ]`, and the original Task 7 placement (right after
        # `EvalsInspector`, ahead of the primary action) inverted that.
        # Still bench-selection-only, and still gated on a RESOLVED bench
        # (`bench is not None`, set in the `selection.kind == "bench"`
        # branch above): an unresolvable bench id renders no
        # `EvalsInspector` and, per this same guard, neither of these
        # buttons either -- there is nothing here to duplicate or delete.
        if selection.kind == "bench" and bench is not None:
            yield Button("Duplicate", id="evals-duplicate-bench")
            delete_reason = self._bench_delete_disabled_reason(selection.id)
            if delete_reason:
                # Mirrors the primary action's own TASK-1076 convention
                # just above (a status badge plus an always-visible
                # callout, not a hover-only tooltip -- see that block's
                # comment for the accessibility rationale). Not factored
                # into one shared helper: the primary action's version
                # also folds in the bench's own NAME (this button's label
                # never changes).
                yield Static(
                    "Delete: Blocked",
                    id="evals-delete-bench-status",
                    classes="ds-status-badge evals-status-blocked",
                    markup=False,
                )
                yield Static(
                    delete_reason,
                    id="evals-delete-bench-reason",
                    classes="ds-recovery-callout",
                    markup=False,
                )
            yield Button(
                "Delete",
                id="evals-delete-bench",
                disabled=bool(delete_reason),
                tooltip=delete_reason,
            )

    def _primary_action_state(self) -> tuple[str, bool, str]:
        """Label, disabled, and tooltip-reason for the primary action button.

        A bare "Run bench" against an ambiguous or stale selection is how
        the old screen produced dead-end toasts (see the plan's design
        note) -- every branch here names the concrete object the action
        would run, or states a concrete reason it can't.

        The found-bench branch is the only one that ever enables the
        button -- every other branch (an unresolvable bench, a dataset, a
        completed run group, or no selection at all) stays disabled with
        its own stated reason, since none of those names an object this
        action can actually run. The in-flight branch below overrides ALL
        of those, found-bench included, whenever a run is genuinely in
        progress.
        """
        selection = self._selection

        if self._bench_run_running or self._sample_bench_running:
            # Whole-branch review Important finding: this function used to
            # never consult either running-flag at all, so a rail click
            # during an in-flight run -- `EvalsScreen.select()` always
            # schedules `refresh(recompose=True)`, even for a same-bench
            # reselection -- recomposed the inspector into a FRESH,
            # ENABLED "Run <name>" button. A press there hits
            # `_on_primary_action_pressed`'s own `_bench_run_running`
            # guard and silently no-ops: the exact dead-end-toast/silent-
            # no-op anti-pattern this whole function's naming rule exists
            # to avoid, just reopened by a recompose instead of by a
            # missing press handler. Checked first, before every other
            # branch, so it wins regardless of what's currently selected --
            # including the found-bench branch just below, whose own label
            # this still borrows (escaped) so the button keeps naming its
            # object even while blocked.
            bench = (
                self._view_model.bench_by_id(selection.id)
                if selection.kind == "bench" and selection.id
                else None
            )
            name = escape_markup(str(bench.get("name") or "Untitled bench")) if bench else None
            return (
                f"Run {name}" if name else "Run Bench",
                True,
                "A bench run is already in flight.",
            )

        if selection.kind == "bench":
            bench = (
                self._view_model.bench_by_id(selection.id) if selection.id else None
            )
            if bench is None:
                return (
                    "Run Bench",
                    True,
                    "The selected bench no longer exists; choose another "
                    "bench to run.",
                )
            # escape_markup: `name` is free-text and reaches TWO markup-
            # parsed surfaces from here -- this tooltip string (both
            # branches below), AND (via this same return value)
            # `Button(label=...)`'s construction in
            # `_compose_inspector_pane` plus the live `button.label = ...`/
            # `button.tooltip = ...` reassignment in
            # `_reset_bench_run_running_ui`. `Content.from_text`'s
            # markup=True default applies on EVERY assignment to a
            # Button's `.label`, not just construction (Textual's
            # `validate_label` reactive validator), so a bare `[/]` in a
            # bench name would raise `MarkupError` and crash the rail --
            # the same hazard class task-1476 fixed for bench-run toast
            # text, and library_rail.py's `_run_group_row_label` fixed for
            # run rows; this closes the last unescaped instance of it in
            # this file. Computed once here, ahead of the target-count
            # check below, so both the found-but-target-less and the
            # runnable branch can name the bench in their label.
            name = escape_markup(str(bench.get("name") or "Untitled bench"))
            # task-1482 fix round 1: a draft bench created via "+ New
            # bench" has `target_ids=()` until the bench editor (Task 6)
            # wires one on. Read straight from the already-loaded row's
            # `config_data` (no extra DB call -- `list_tasks`/`bench_by_id`
            # already parsed it) rather than `storage.load_bench`, which
            # this function has never otherwise needed. Without this
            # guard, pressing "Run" reached `run_existing_bench` with zero
            # targets, which "completed" an EMPTY run group -- the exact
            # dead-end-toast pattern this function's own naming rule
            # exists to prevent, just reopened one step further downstream
            # ("Bench run finished." followed by "This run could not be
            # found"). Wording matches the readiness panel's own "No
            # targets configured yet." (inspector.py/bench_editor.py) for
            # the same state, so the vocabulary stays consistent across
            # the two surfaces.
            target_ids = (bench.get("config_data") or {}).get("target_ids") or []
            if not target_ids:
                # task-1612: staging a target in the bench editor's Add
                # picker does NOT touch this row's persisted `target_ids`
                # -- only Save does (see `bench_editor.py`'s own module
                # docstring: staged targets are form state until Save
                # writes them via `save_bench`). Without naming Save here,
                # a user who has just staged one reads this tooltip as
                # stale or wrong, since it still says "no targets yet"
                # while one is visibly staged in the editor.
                return (
                    f"Run {name}",
                    True,
                    "This bench has no targets yet; add one in the bench "
                    "editor and Save.",
                )
            return (
                f"Run {name}",
                False,
                f"Runs {name} against its configured targets.",
            )

        # No "classic" branch: `_compose_inspector_pane` never calls this
        # function for a classic-task selection at all -- classic tasks
        # are read-only (see `ClassicTaskDetail`'s deferral sentence) and
        # get no run control, not even a disabled one.

        if selection.kind == "dataset":
            return (
                "Run Bench",
                True,
                # task-1482: names the concrete fix ("+ New bench" in the
                # Catalog rail creates a bench bound to THIS dataset)
                # instead of the old, more general "select a bench that
                # uses this dataset instead" -- which presupposed one
                # already existed, leaving a genuine dead end for a
                # dataset with no bench yet.
                "Datasets are run from within a bench; use + New bench in "
                "the Catalog rail to create one against this dataset.",
            )

        if selection.kind == "run_group":
            return (
                "Run Bench",
                True,
                "This run has already completed; select a bench to start a "
                "new run.",
            )

        return (
            "Run Bench",
            True,
            "Select a bench in the Catalog rail to run it.",
        )

    def save_state(self):
        """Save evals screen state."""
        return super().save_state()

    def restore_state(self, state):
        """Restore evals screen state."""
        super().restore_state(state)
