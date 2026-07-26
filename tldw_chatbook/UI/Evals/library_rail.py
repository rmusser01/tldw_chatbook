"""Evals library rail: Benches / Datasets / Runs.

Three collapsible sections, each a live-count header plus selectable rows
(or an empty-state line). Posts ``EvalsSelectionChanged`` on a row press;
``EvalsScreen`` owns the actual selection state (see
``evals_state.EvalsSelection``) and reacts to the message rather than the
rail mutating shell state itself.

Rows are plain ``Button``s, never Screens -- see ``evals_screen.py``'s
module docstring for why that distinction is the entire point of this PR.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from ...Chat.console_glyphs import GLYPH_COLLAPSED, GLYPH_EXPANDED
from .evals_state import EvalsSelection, EvalsViewModel

EVALS_RAIL_SECTION_TOGGLE_PREFIX = "evals-rail-toggle-"
EVALS_RAIL_ROW_PREFIX = "evals-rail-row-"

#: Section ids in display order; also the default keys of ``open_sections``.
RAIL_SECTIONS: tuple[str, ...] = ("benches", "datasets", "runs")


def _default_open_sections() -> dict[str, bool]:
    return {section_id: True for section_id in RAIL_SECTIONS}


#: The design mockup's own subgroup label
#: (``Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md``,
#: "Classic orchestrator tasks appear in a labelled subgroup under
#: Benches"). Not a Button -- it never carries a selection, it only marks
#: where classic rows start.
CLASSIC_SUBGROUP_LABEL = "─ classic ─"

EVALS_RAIL_CLASSIC_ROW_PREFIX = "evals-rail-row-benches-classic-"


def _bench_row_label(row: dict[str, Any]) -> str:
    return str(row.get("name") or "Untitled bench")


def _classic_row_label(row: dict[str, Any]) -> str:
    return str(row.get("name") or "Untitled task")


def _dataset_row_label(row: dict[str, Any]) -> str:
    return str(row.get("name") or "Untitled dataset")


def _run_group_row_label(row: dict[str, Any]) -> str:
    name = row.get("task_name") or "Untitled run"
    count = row.get("run_count") or 0
    target_word = "target" if count == 1 else "targets"
    return f"{name} ({count} {target_word})"


class LibraryRail(Vertical):
    """Left rail: Benches, Datasets, Runs -- each collapsible, with counts."""

    class EvalsSelectionChanged(Message, namespace="library_rail"):
        """Posted when the user presses a selectable row."""

        def __init__(self, selection: EvalsSelection) -> None:
            super().__init__()
            self.selection = selection

    def __init__(
        self,
        view_model: EvalsViewModel,
        *,
        selection: Optional[EvalsSelection] = None,
        open_sections: Optional[dict[str, bool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.view_model = view_model
        self.selection = selection or EvalsSelection()
        # Shared, mutated in place (never reassigned) rather than copied:
        # EvalsScreen holds this same dict and passes it back in on every
        # recompose, so a section's collapsed/expanded state survives the
        # screen-level `refresh(recompose=True)` that a selection change
        # triggers (which tears down and rebuilds this whole widget).
        self.open_sections = (
            open_sections if open_sections is not None else _default_open_sections()
        )
        self._row_targets: dict[str, EvalsSelection] = {}

    def compose(self) -> ComposeResult:
        self._row_targets = {}
        yield from self._benches_section()
        yield from self._section(
            section_id="datasets",
            title="Datasets",
            rows=self.view_model.datasets(),
            kind="dataset",
            empty_copy="No datasets yet.",
            row_label=_dataset_row_label,
        )
        yield from self._section(
            section_id="runs",
            title="Runs",
            rows=self.view_model.run_groups(),
            kind="run_group",
            empty_copy="No runs yet.",
            row_label=_run_group_row_label,
        )

    def _section(
        self,
        *,
        section_id: str,
        title: str,
        rows: list[dict[str, Any]],
        kind: str,
        empty_copy: str,
        row_label: Callable[[dict[str, Any]], str],
    ) -> ComposeResult:
        open_state = self.open_sections.get(section_id, True)
        yield Horizontal(
            Static(
                f"{title} ({len(rows)})",
                classes="destination-section evals-rail-section-label",
                markup=False,
            ),
            Button(
                GLYPH_EXPANDED if open_state else GLYPH_COLLAPSED,
                id=f"{EVALS_RAIL_SECTION_TOGGLE_PREFIX}{section_id}",
                classes="evals-rail-section-toggle",
                compact=True,
                tooltip=f"{'Collapse' if open_state else 'Expand'} {title}.",
            ),
            classes="evals-rail-section-header",
        )
        yield self._section_body(
            section_id=section_id,
            rows=rows,
            kind=kind,
            empty_copy=empty_copy,
            row_label=row_label,
            open_state=open_state,
        )

    def _row_button(
        self, *, button_id: str, kind: str, row_id: Optional[str], label: str
    ) -> Button:
        """A selectable rail row, registered in ``_row_targets`` so
        ``on_button_pressed`` can resolve which ``EvalsSelection`` it
        posts. Shared by every rail row -- benches, classic tasks (see
        ``_benches_section_body``), datasets, and runs -- so all four kinds
        stay wired through the exact same press -> post_message path."""
        row_selection = EvalsSelection(kind=kind, id=row_id)
        self._row_targets[button_id] = row_selection
        is_selected = self.selection.kind == kind and self.selection.id == row_id
        button = Button(label, id=button_id, classes="evals-rail-row", compact=True)
        button.set_class(is_selected, "is-active")
        return button

    def _section_body(
        self,
        *,
        section_id: str,
        rows: list[dict[str, Any]],
        kind: str,
        empty_copy: str,
        row_label: Callable[[dict[str, Any]], str],
        open_state: bool,
    ) -> Vertical:
        children: list[Any] = []
        if rows:
            for index, row in enumerate(rows):
                button_id = f"{EVALS_RAIL_ROW_PREFIX}{section_id}-{index}"
                children.append(
                    self._row_button(
                        button_id=button_id,
                        kind=kind,
                        row_id=row.get("id"),
                        label=row_label(row),
                    )
                )
        else:
            children.append(
                Static(empty_copy, classes="evals-rail-empty-copy", markup=False)
            )
        body = Vertical(
            *children,
            id=f"evals-rail-section-body-{section_id}",
            classes="evals-rail-section-body",
        )
        if not open_state:
            body.styles.display = "none"
        return body

    def _benches_section(self) -> ComposeResult:
        """The Benches section, with classic (non-word-bench) tasks
        rendered in a labelled subgroup beneath the word benches -- per the
        design spec's "Classic orchestrator tasks appear in a labelled
        subgroup under Benches." Handled separately from ``_section``
        (datasets/runs are single-kind lists) because this section mixes
        two selection kinds and an inert separator row under one header.

        The header count is bench-rows-plus-classic-rows, matching the
        design mockup's own worked example (2 word benches + 2 classic
        tasks -> "BENCHES (4)") -- the section's count is "how many rows
        are under this header," not "how many word benches exist."
        """
        benches = self.view_model.benches()
        classic_tasks = self.view_model.classic_tasks()
        open_state = self.open_sections.get("benches", True)
        yield Horizontal(
            Static(
                f"Benches ({len(benches) + len(classic_tasks)})",
                classes="destination-section evals-rail-section-label",
                markup=False,
            ),
            Button(
                GLYPH_EXPANDED if open_state else GLYPH_COLLAPSED,
                id=f"{EVALS_RAIL_SECTION_TOGGLE_PREFIX}benches",
                classes="evals-rail-section-toggle",
                compact=True,
                tooltip=f"{'Collapse' if open_state else 'Expand'} Benches.",
            ),
            classes="evals-rail-section-header",
        )
        yield self._benches_section_body(benches, classic_tasks, open_state)

    def _benches_section_body(
        self,
        benches: list[dict[str, Any]],
        classic_tasks: list[dict[str, Any]],
        open_state: bool,
    ) -> Vertical:
        children: list[Any] = []
        if benches:
            for index, row in enumerate(benches):
                button_id = f"{EVALS_RAIL_ROW_PREFIX}benches-{index}"
                children.append(
                    self._row_button(
                        button_id=button_id,
                        kind="bench",
                        row_id=row.get("id"),
                        label=_bench_row_label(row),
                    )
                )
        elif not classic_tasks:
            children.append(
                Static("No benches yet.", classes="evals-rail-empty-copy", markup=False)
            )

        if classic_tasks:
            # Inert -- never registered in `_row_targets`, so a press on
            # this row (it is a Static, not a Button, so it cannot receive
            # one anyway) never posts a selection.
            children.append(
                Static(
                    CLASSIC_SUBGROUP_LABEL,
                    classes="evals-rail-classic-separator",
                    markup=False,
                )
            )
            for index, row in enumerate(classic_tasks):
                button_id = f"{EVALS_RAIL_CLASSIC_ROW_PREFIX}{index}"
                children.append(
                    self._row_button(
                        button_id=button_id,
                        kind="classic",
                        row_id=row.get("id"),
                        label=_classic_row_label(row),
                    )
                )

        body = Vertical(
            *children,
            id="evals-rail-section-body-benches",
            classes="evals-rail-section-body",
        )
        if not open_state:
            body.styles.display = "none"
        return body

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith(EVALS_RAIL_SECTION_TOGGLE_PREFIX):
            event.stop()
            section_id = button_id.removeprefix(EVALS_RAIL_SECTION_TOGGLE_PREFIX)
            self.open_sections[section_id] = not self.open_sections.get(
                section_id, True
            )
            self.refresh(recompose=True)
            return
        selection = self._row_targets.get(button_id)
        if selection is None:
            return
        event.stop()
        self.post_message(self.EvalsSelectionChanged(selection))
