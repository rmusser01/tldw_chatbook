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

This screen replaces that architecture with the house three-pane
workbench -- library rail, detail pane, readiness inspector -- driven by
selection state (``EvalsSelection``, ``evals_state.py``) instead of a
hand-rolled screen stack. **No ``Screen`` subclass is mounted inside any
workbench container here.** Detail and inspector content is swapped by a
screen-level ``refresh(recompose=True)`` on selection change, which tears
down and remounts plain widgets (``Static``/``Button``/``Vertical``) --
never a ``Screen``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ...DB.Evals_DB import EvalsDB
from ...Evals.word_bench.models import PreflightResult
from ...Evals.word_bench.models import Target as WordBenchTarget
from ...Evals.word_bench.runner import CancelToken, CaptureClientLike
from ..Evals import sample_bench
from ..Evals.bench_editor import BenchEditor, ClassicTaskDetail
from ..Evals.evals_state import EvalsSelection, EvalsViewModel, SelectionKind
from ..Evals.inspector import EvalsCellInspector, EvalsInspector
from ..Evals.library_rail import RAIL_SECTIONS, LibraryRail
from ..Evals.results_grid import ResultsGrid
from ..Evals.snippet_editor import SnippetEditor
from ..Navigation.base_app_screen import BaseAppScreen
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .lab_mode_strip import LabModeStrip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvalsScreen(BaseAppScreen):
    """Evals destination seat: library rail + detail pane + readiness inspector."""

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "evals", **kwargs)
        self._view_model = EvalsViewModel(self._resolve_db(app_instance))
        self._selection = EvalsSelection()
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
        #: True for the duration of one create-and-run flow. The PRIMARY
        #: guard against a second click starting a second worker (the
        #: button is also disabled live -- see ``_set_sample_bench_
        #: running_ui`` -- but a disabled widget not yet re-rendered, or a
        #: message posted directly as this screen's own tests do, must not
        #: be able to race past a disabled button and start two runs).
        self._sample_bench_running: bool = False
        #: The active run's cooperative cancel token, or ``None`` when no
        #: run is in flight. Not currently triggered by anything in this
        #: screen (the running-guard above prevents the second-click race
        #: that would otherwise need it) -- kept as a real, threaded
        #: seam rather than a decorative parameter, since
        #: ``WordBenchRunner.run`` already accepts one and a future cancel
        #: affordance should not need a second plumbing pass.
        self._sample_bench_cancel_token: Optional[CancelToken] = None

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
        (``BaseAppScreen.refresh(recompose=True)``), it does not await its
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
        self._register_grid_shortcuts()
        if self.is_mounted:
            self.refresh(recompose=True)

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

    @on(LibraryRail.SampleBenchRequested)
    def _on_sample_bench_requested(
        self, event: LibraryRail.SampleBenchRequested
    ) -> None:
        """Creates and runs the one-click sample bench (see
        ``sample_bench.py``). Real DB writes plus a real HTTP call (in
        production) -- run as a worker, never inline in a message handler,
        per CLAUDE.md's "Workers for operations >100ms" rule.

        ``_sample_bench_running`` is checked FIRST and unconditionally: the
        button is disabled for the run's duration (``_set_sample_bench_
        running_ui``), but a disabled widget mid-recompose, or a message
        posted directly (as this screen's own tests do to simulate a race),
        must not be able to start a SECOND worker in the same
        ``exclusive=True`` group -- that is exactly what previously let a
        second click hard-cancel the first run via ``asyncio.
        CancelledError`` and abandon its DB rows mid-flight (see
        ``sample_bench._mark_orphaned_runs_cancelled`` for the cleanup this
        guard makes almost always unnecessary in practice, kept anyway as a
        second line of defence for whatever OTHER path might cancel this
        worker, e.g. the screen itself unmounting mid-run).
        """
        event.stop()
        if self._sample_bench_running:
            return
        self.run_worker(
            self._create_sample_bench_worker(),
            exclusive=True,
            group="evals-sample-bench",
        )

    async def _create_sample_bench_worker(self) -> None:
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
            logger.opt(exception=True).warning("Sample bench creation failed.")
            self.app_instance.notify(
                f"Could not create the sample bench: {exc}", severity="error"
            )
        finally:
            self._sample_bench_running = False
            self._sample_bench_cancel_token = None
            self._reset_sample_bench_running_ui()
        if result is not None:
            self.app_instance.notify(
                "Sample bench created and run.", severity="information"
            )
            self.select(kind="run_group", id=result.run_group_id)

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
        """Restores the button after a run ends. A no-op (via the same
        ``QueryError`` guard) on the success path, where ``self.select(...)``
        immediately recomposes the rail and replaces this button with the
        bench's own row anyway -- this only matters on the failure path,
        where the SAME ``LibraryRail`` instance survives and must not be
        left permanently disabled."""
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

    # No `#evals-primary-action` press handler: `_primary_action_state`
    # below keeps the button disabled unconditionally (even for a found,
    # selected bench) until PR 3b wires real bench execution to it. An
    # enabled button whose only handler shows a "not wired yet" toast is
    # itself the dead-end-toast anti-pattern this button's own design note
    # exists to avoid -- see _primary_action_state's docstring.

    def compose_content(self) -> ComposeResult:
        with Vertical(id="evals-shell"):
            yield DestinationHeader(
                WorkbenchHeaderState(
                    title="Evals",
                    subtitle="Run and review evaluation jobs.",
                    status="ready",
                ),
                id="evals-destination-header",
            )
            yield LabModeStrip(active_route="evals", id="lab-mode-strip")
            with Horizontal(
                id="evals-workbench", classes="ds-panel destination-workbench"
            ):
                yield LibraryRail(
                    self._view_model,
                    selection=self._selection,
                    open_sections=self._rail_open_sections,
                    app_config=self._current_app_config(),
                    id="evals-library-pane",
                    classes="destination-workbench-pane",
                )
                # Resolved once per selection/recompose and threaded into
                # BOTH panes below -- BenchEditor and EvalsInspector each
                # independently calling `EvalsViewModel.preflight_for_bench`
                # read the bench's run-group snapshot twice on one render
                # (see I2 in the PR 3a fix report).
                preflight = self._preflight_for_selection()
                with Vertical(
                    id="evals-detail-pane", classes="destination-workbench-pane"
                ):
                    yield from self._compose_detail_pane(preflight)
                with Vertical(
                    id="evals-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                ):
                    yield from self._compose_inspector_pane(preflight)

    def _preflight_for_selection(self) -> dict[str, PreflightResult]:
        """The current selection's readiness map, or ``{}`` for every
        selection kind but ``"bench"`` (no other kind's panes read it)."""
        selection = self._selection
        if selection.kind != "bench" or not selection.id:
            return {}
        return self._view_model.preflight_for_bench(selection.id)

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
            "Select a bench, dataset, or run in the library rail to see its "
            "detail here.",
            id="evals-detail-empty",
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
        yield Button(
            label,
            id="evals-primary-action",
            disabled=disabled,
            tooltip=tooltip,
        )

    def _primary_action_state(self) -> tuple[str, bool, str]:
        """Label, disabled, and tooltip-reason for the primary action button.

        A bare "Run bench" against an ambiguous or stale selection is how
        the old screen produced dead-end toasts (see the plan's design
        note) -- every branch here names the concrete object the action
        would run, or states a concrete reason it can't.

        Every branch is currently disabled, including a found, selected
        bench: this PR (3a) wires selection and the shell, not execution --
        the word bench runner (PR 2) has no button connecting to it yet,
        and that wiring is PR 3b's job (the results grid it runs into). An
        *enabled* button whose only handler pops a "not wired yet" toast
        would be exactly the dead-end-toast pattern this function's naming
        rule exists to avoid, just moved one click later.
        """
        selection = self._selection

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
            name = str(bench.get("name") or "Untitled bench")
            return (
                f"Run {name}",
                True,
                "Running a bench from this workbench isn't wired up yet; "
                "that lands with the results grid in a later PR.",
            )

        # No "classic" branch: `_compose_inspector_pane` never calls this
        # function for a classic-task selection at all -- classic tasks
        # are read-only (see `ClassicTaskDetail`'s deferral sentence) and
        # get no run control, not even a disabled one.

        if selection.kind == "dataset":
            return (
                "Run Bench",
                True,
                "Datasets are run from within a bench; select a bench that "
                "uses this dataset instead.",
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
            "Select a bench in the library rail to run it.",
        )

    def save_state(self):
        """Save evals screen state."""
        return super().save_state()

    def restore_state(self, state):
        """Restore evals screen state."""
        super().restore_state(state)
