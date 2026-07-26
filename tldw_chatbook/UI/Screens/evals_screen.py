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

from typing import TYPE_CHECKING, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ...DB.Evals_DB import EvalsDB
from ..Evals.bench_editor import BenchEditor, ClassicTaskDetail
from ..Evals.evals_state import EvalsSelection, EvalsViewModel, SelectionKind
from ..Evals.inspector import EvalsInspector
from ..Evals.library_rail import RAIL_SECTIONS, LibraryRail
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
        """
        self._selection = EvalsSelection(kind=kind, id=id)
        if self.is_mounted:
            self.refresh(recompose=True)

    @on(LibraryRail.EvalsSelectionChanged)
    def _on_library_selection_changed(
        self, event: LibraryRail.EvalsSelectionChanged
    ) -> None:
        event.stop()
        self.select(kind=event.selection.kind, id=event.selection.id)

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
                    id="evals-library-pane",
                    classes="destination-workbench-pane",
                )
                with Vertical(
                    id="evals-detail-pane", classes="destination-workbench-pane"
                ):
                    yield from self._compose_detail_pane()
                with Vertical(
                    id="evals-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                ):
                    yield from self._compose_inspector_pane()

    def _compose_detail_pane(self) -> ComposeResult:
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
            yield BenchEditor(self._view_model, selection.id, id="evals-bench-editor")
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
            yield Static(
                str(group.get("task_name") or "Untitled run"),
                id="evals-detail-run-name",
                classes="evals-pane-heading",
            )
            yield Static(
                f"Targets run: {group.get('run_count', 0)}",
                id="evals-detail-run-count",
            )
            return

        yield Static(
            "Select a bench, dataset, or run in the library rail to see its "
            "detail here.",
            id="evals-detail-empty",
        )

    def _compose_inspector_pane(self) -> ComposeResult:
        yield Static("Inspector", classes="destination-section evals-pane-title")
        selection = self._selection

        if selection.kind == "bench":
            bench = (
                self._view_model.bench_by_id(selection.id) if selection.id else None
            )
            if bench is not None:
                yield EvalsInspector(
                    self._view_model, selection.id, id="evals-inspector-bench"
                )

        if selection.kind == "classic":
            # Classic tasks are read-only in this workbench (see the design
            # spec's "Classic tasks" section and BenchEditor's
            # `ClassicTaskDetail`, which carries the deferral sentence) --
            # no run control is rendered here at all, not even a disabled
            # one; `_primary_action_state()` is never consulted for this
            # kind.
            return

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
