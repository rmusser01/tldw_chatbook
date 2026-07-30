"""Evals library rail: Benches / Datasets / Runs.

Three collapsible sections, each a live-count header plus selectable rows
(or an empty-state line). Posts ``EvalsSelectionChanged`` on a row press;
``EvalsScreen`` owns the actual selection state (see
``evals_state.EvalsSelection``) and reacts to the message rather than the
rail mutating shell state itself.

Rows are plain ``Button``s, never Screens -- see ``evals_screen.py``'s
module docstring for why that distinction is the entire point of this PR.

**Empty states** (design spec "Empty states and first run"). A fresh
install's most common condition is zero benches, zero datasets, zero runs,
and possibly zero configured providers:

- No word benches exist -> the Benches section offers either "Create
  sample bench" (``sample_bench.provider_is_configured`` is ``True``) or
  "Open Settings" (it is ``False``) -- **independent of whether classic
  tasks exist**. An earlier version of this gate also required `not
  classic_tasks`, which meant a user with a pre-existing classic task and
  no word benches (exactly this rebuild's upgrading population) saw
  NEITHER offer, whatever providers they had configured -- a real
  regression caught by review, not by this file's own tests. The full
  explanatory copy (``_no_providers_message``/"No benches yet.") is still
  reserved for a FULLY empty section (no classic tasks either), since
  otherwise it would be a redundant wall of text above a real list; the
  actionable button always renders regardless. Scoped to the Benches
  section only, never the whole rail: Datasets/Runs never showed a target
  list or preflight results to begin with, and classic (non-word-bench)
  tasks need no provider at all.
- The "no provider" copy names llama.cpp specifically ("No local
  llama.cpp provider is configured"), not "a provider" in general --
  ``provider_is_configured`` only ever asks whether a ``llama_cpp`` target
  resolves (see ``sample_bench.py``), so a user with e.g. OpenAI
  configured is not missing "a provider" by any honest reading.
- No datasets -> the Datasets section's empty copy gains "+ New dataset"
  and "Import…" side by side, handled locally here (dataset creation and
  import are plain DB/file operations, not provider calls, mirroring
  ``snippet_editor.py``'s own self-contained import flow) rather than
  routed through ``EvalsScreen``.

**Creation affordances are not empty-only** (TASK-1478). A live UAT pass
found the rail became read-only the moment it had one row: "Create sample
bench" and "+ New dataset"/"Import…" used to render only in each
section's *empty* branch, so a single bench or dataset was a one-way
trapdoor out of ever creating another one without going through some
other screen. Both now render unconditionally at the top of their
section's body (``_benches_section_body``'s ``_create_sample_bench_button``
call, ``_dataset_actions``) -- only the *explanatory copy* ("No benches
yet."/"No datasets yet.", the first-run hint, the no-providers message)
stays scoped to the genuinely-empty case; a real list below it needs no
prose. The provider gate itself is unchanged: no benches and no provider
still routes to "Open Settings" rather than a button pointing at nothing,
and that escape hatch is never duplicated once real benches exist -- see
``_benches_section_body``'s own comment on why a failed gate with existing
benches (in practice unreachable, since a word bench's target already
satisfies ``provider_is_configured``) adds no row at all.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from ...Widgets.destination_rail import GLYPH_COLLAPSED, GLYPH_EXPANDED
from ...Constants import TAB_SETTINGS
from ...Third_Party.textual_fspicker import FileOpen, Filters
from ...Utils.path_validation import validate_path_simple
from ..Navigation.main_navigation import NavigateToScreen
from . import sample_bench
from .evals_state import EvalsSelection, EvalsViewModel
from .notify_mixin import NotifyMixin
from .snippet_editor import (
    import_snippets_into_dataset,
    parse_csv_snippets,
    parse_json_snippets,
    parse_plain_text_snippets,
)

EVALS_RAIL_SECTION_TOGGLE_PREFIX = "evals-rail-toggle-"
EVALS_RAIL_ROW_PREFIX = "evals-rail-row-"

#: Section ids in display order; also the default keys of ``open_sections``.
RAIL_SECTIONS: tuple[str, ...] = ("benches", "datasets", "runs")

#: Suffix-dispatched, mirroring snippet_editor.py's own _IMPORT_PARSERS --
#: kept as a SEPARATE mapping here (rather than importing that private dict)
#: since the three parser functions themselves are the public surface.
_RAIL_IMPORT_PARSERS = {
    ".csv": parse_csv_snippets,
    ".json": parse_json_snippets,
}

#: eval_datasets.name is UNIQUE with no deleted_at exemption -- a bare
#: literal default name would collide on a second "+ New dataset" click.
_NEW_DATASET_BASE_NAME = "Untitled dataset"


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


#: Single-cell-width status glyphs for run rows -- NEVER emoji, which
#: render double-width in this app's terminal (a repeated past defect).
#: Status must never be conveyed by colour alone; the glyph itself is the
#: signal. Keyed by ``EvalsViewModel.run_groups()``'s rolled-up
#: ``"status"`` (TASK-1480) -- an unrecognised/missing status (there
#: should never be one; kept only so a stale or malformed row degrades to
#: a glyph rather than a ``KeyError`` crashing the rail) falls back to the
#: "done" glyph.
_RUN_STATUS_GLYPHS: dict[str, str] = {
    "running": "●",  # ● BLACK CIRCLE
    "cancelled": "✗",  # ✗ BALLOT X
    "completed": "✓",  # ✓ CHECK MARK
}


def _run_group_row_time(created_at: Any) -> str:
    """``created_at`` as ``HH:MM``, per the design spec's rail mock.

    ``EvalsDB`` stores ``eval_runs.created_at`` via SQLite's
    ``datetime('now', 'utc')`` (``"YYYY-MM-DD HH:MM:SS"``, no ``T`` or UTC
    offset) -- a format ``datetime.fromisoformat`` happens to accept
    directly on Python 3.11+. It is still a free-text column with no
    format enforcement at the DB layer, so this parses defensively and
    falls back to the raw string on any parse failure rather than crash
    the rail over a timestamp.
    """
    text = "" if created_at is None else str(created_at)
    try:
        return datetime.fromisoformat(text).strftime("%H:%M")
    except (TypeError, ValueError):
        return text


def _run_group_row_label(row: dict[str, Any]) -> str:
    """``● 14:31 · <task_name>`` / ``✓ 14:02 · <task_name>`` / ``✗ 13:55 ·
    <task_name>`` -- the design spec's own rail mock
    (``Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md``).

    TASK-1480: before this, every run row rendered the exact same shape a
    bench row did (``"<name> (N targets)"``), so a live UAT pass could not
    tell a bench apart from one of its own past runs at a glance, and had
    no way to see a run currently in flight. See ``run_groups()``'s own
    docstring for how the leading glyph's status is rolled up from the
    group's per-target runs.

    ``task_name`` reaches this function as a free-text bench name;
    ``Button(label=...)`` parses its argument as Textual markup by
    default (``Content.from_text``'s ``markup=True`` default), so an
    unescaped name containing a bare ``[/]`` would raise ``MarkupError``
    and crash the rail the instant it composes -- the same hazard
    task-1476 fixed for bench-run toast text (``evals_screen.py``'s
    ``_run_bench_worker``/``_create_sample_bench_worker``), left open
    there as a separate, out-of-scope issue in this exact function (see
    that commit's ``_RaisingCaptureClient`` docstring in
    ``Tests/UI/test_evals_screen.py``). Escaping here closes it.
    """
    name = str(row.get("task_name") or "Untitled run")
    glyph = _RUN_STATUS_GLYPHS.get(row.get("status"), _RUN_STATUS_GLYPHS["completed"])
    time_text = _run_group_row_time(row.get("created_at"))
    return f"{glyph} {time_text} · {escape_markup(name)}"


class LibraryRail(NotifyMixin, Vertical):
    """Left rail: Benches, Datasets, Runs -- each collapsible, with counts."""

    class EvalsSelectionChanged(Message, namespace="library_rail"):
        """Posted when the user presses a selectable row."""

        def __init__(self, selection: EvalsSelection) -> None:
            super().__init__()
            self.selection = selection

    class SampleBenchRequested(Message, namespace="library_rail"):
        """Posted when "Create sample bench" is pressed.

        Carries no payload -- creating and running the sample bench needs
        real DB/network work (``sample_bench.create_and_run_sample_bench``
        is a coroutine), so ``EvalsScreen`` runs it as a worker rather than
        this widget doing it inline, mirroring why row selection is a
        message rather than a direct call too.
        """

    def __init__(
        self,
        view_model: EvalsViewModel,
        *,
        selection: Optional[EvalsSelection] = None,
        open_sections: Optional[dict[str, bool]] = None,
        app_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.view_model = view_model
        self.selection = selection or EvalsSelection()
        #: The app's loaded settings (``TldwCli.app_config``), read only for
        #: ``sample_bench.provider_is_configured``'s gate. ``None`` (a fake
        #: app_instance in a test, or a real one composed before settings
        #: load) degrades to ``{}`` -- "no providers configured", never a
        #: crash.
        self.app_config: dict[str, Any] = dict(app_config or {})
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
        # Read once, reused by _benches_section (to decide first-run
        # primacy, TASK-1076) and by the sections below -- rather than each
        # of the three sections independently re-querying the same rows.
        datasets = self.view_model.datasets()
        run_groups = self.view_model.run_groups()
        yield from self._benches_section(
            is_first_run=not datasets and not run_groups
        )
        yield from self._section(
            section_id="datasets",
            title="Datasets",
            rows=datasets,
            kind="dataset",
            empty_copy="No datasets yet.",
            row_label=_dataset_row_label,
            actions=self._dataset_actions,
        )
        yield from self._section(
            section_id="runs",
            title="Runs",
            rows=run_groups,
            kind="run_group",
            empty_copy="No runs yet.",
            row_label=_run_group_row_label,
        )

    def _no_providers_message(self) -> ComposeResult:
        """The two-line explanation for the Benches section's empty copy
        when NO provider is configured -- per requirement 1: no target
        list, no wall of preflight failures.

        Only yielded when the section is otherwise FULLY empty (see
        ``_benches_section_body`` -- with a classic task also present, this
        would just be a redundant wall of text above a real list; the
        actionable ``_open_settings_button`` below still renders either
        way, which is the part that actually matters).

        The copy names llama.cpp specifically, not "a provider" in
        general: ``provider_is_configured`` only ever asks whether a
        ``llama_cpp`` target resolves (see ``sample_bench.py``'s own "Why
        the target resolution is narrow" note) -- a user with, say, OpenAI
        configured is NOT missing "a provider" by any honest reading, and
        the old, broader wording made a claim about their setup that
        wasn't true. This is the same do-not-fabricate principle applied
        to copy instead of data.
        """
        yield Static(
            "No local llama.cpp provider is configured.",
            id="evals-rail-no-providers",
            classes="evals-pane-heading",
            markup=False,
        )
        yield Static(
            "Configure a local llama.cpp server in Settings, then come "
            "back here to build or run a bench.",
            id="evals-rail-no-providers-detail",
            classes="evals-rail-empty-copy",
            markup=False,
        )

    @staticmethod
    def _open_settings_button() -> Button:
        return Button(
            "Open Settings",
            id="evals-rail-open-settings",
            tooltip="No local llama.cpp provider is configured yet.",
        )

    @staticmethod
    def _create_sample_bench_button() -> Button:
        """Shared by both branches of ``_benches_section_body`` (TASK-1478:
        the button is no longer empty-only, so both the "no benches yet"
        and "benches already exist" paths need the identical control, id
        included -- see the module docstring's "Creation affordances are
        not empty-only" note)."""
        return Button(
            "Create sample bench",
            id="evals-create-sample-bench",
            tooltip=(
                "Creates the loaded-nouns sample dataset, wires it to a "
                "configured target, and runs it."
            ),
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
        actions: Optional[Callable[[], ComposeResult]] = None,
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
            actions=actions,
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
        actions: Optional[Callable[[], ComposeResult]] = None,
    ) -> Vertical:
        children: list[Any] = []
        # TASK-1478: rendered unconditionally, at the top -- a creation
        # affordance must survive the section's first row existing, not
        # just precede it. Lives in the section body (not the header) so it
        # still collapses with the rest of the section, same as every row.
        if actions is not None:
            children.extend(actions())
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

    def _benches_section(self, *, is_first_run: bool) -> ComposeResult:
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

        Args:
            is_first_run: Whether Datasets and Runs are ALSO both empty --
                see ``_benches_section_body``'s ``is_first_run`` for what
                this changes.
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
        yield self._benches_section_body(
            benches, classic_tasks, open_state, is_first_run=is_first_run
        )

    def _benches_section_body(
        self,
        benches: list[dict[str, Any]],
        classic_tasks: list[dict[str, Any]],
        open_state: bool,
        *,
        is_first_run: bool = False,
    ) -> Vertical:
        children: list[Any] = []
        # Read once, shared by both the non-empty and empty branches below
        # (TASK-1478 needs it in both -- the sample-bench button is no
        # longer offered only when the section is empty).
        provider_ready = sample_bench.provider_is_configured(
            self.view_model, self.app_config
        )
        if benches:
            # TASK-1478: a word bench already exists (which itself means a
            # llama_cpp target -- an eval_models row -- already satisfies
            # `provider_is_configured`; see sample_bench.py's gate), but
            # the sample-bench button used to disappear the moment this
            # branch was reached at all, making bench creation a one-way
            # trapdoor. Keep it reachable at the top of the list. When the
            # gate fails here regardless (in practice unreachable, per the
            # note above), no row is added -- the "Open Settings" escape
            # hatch below is scoped to the fully-empty branch and must not
            # be duplicated for a rail that already has real benches.
            if provider_ready:
                children.append(self._create_sample_bench_button())
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
        else:
            # No word benches -- offer sample-bench creation (if a
            # provider is configured) or a Settings route, REGARDLESS of
            # whether classic tasks exist. Gating this on `not
            # classic_tasks` too was a real regression (caught by review,
            # not by this file's own tests -- see Tests/UI/test_evals_
            # empty_states.py's test_sample_bench_offer_is_reachable_
            # alongside_a_classic_task): it left a user with a
            # pre-existing classic task and no word benches -- exactly
            # this rebuild's upgrading population -- with NEITHER offer,
            # no matter what providers they had configured.
            if not classic_tasks:
                # Fully empty section -- the full explanatory copy. With a
                # classic task also present, this text would just be a
                # redundant wall above a real list; the actionable button
                # below still renders either way.
                if provider_ready and is_first_run:
                    # TASK-1076: a genuinely first-run rail (no benches, no
                    # classic tasks, no datasets, no runs -- every count is
                    # zero at once) offered three equal-weight affordances
                    # ("Create sample bench" / "+ New dataset" / no action
                    # for Runs) with nothing marking which one is the
                    # intended starting point, or that a bench itself sets
                    # up a dataset. Only this fully-empty condition gets the
                    # callout; a user who already has datasets or runs is
                    # past "first open" and the plain copy below still
                    # applies. Styled distinctly from
                    # `.evals-rail-empty-copy` (bold/primary vs. muted) so
                    # it reads as the recommended path at a glance, not just
                    # in wording -- see `.evals-rail-first-run-hint` in
                    # features/_evals.tcss.
                    children.append(
                        Static(
                            "Start here — no benches yet. The sample "
                            "bench below builds a dataset and a target for "
                            "you, then runs it.",
                            id="evals-rail-first-run-hint",
                            classes="evals-rail-first-run-hint",
                            markup=False,
                        )
                    )
                elif provider_ready:
                    children.append(
                        Static(
                            "No benches yet.",
                            classes="evals-rail-empty-copy",
                            markup=False,
                        )
                    )
                else:
                    children.extend(self._no_providers_message())
            if provider_ready:
                # A real target IS resolvable here -- the button never
                # appears pointing at nothing (see sample_bench.py's "Do
                # not fabricate" note).
                children.append(self._create_sample_bench_button())
            else:
                children.append(self._open_settings_button())

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

    def _dataset_actions(self) -> ComposeResult:
        """Authoring and import, side by side (design spec's "Empty states
        and first run" table) -- both handled locally (plain DB/file work,
        never a provider call), mirroring ``snippet_editor.py``'s own
        self-contained import flow.

        TASK-1478: no longer empty-only -- rendered unconditionally at the
        top of the Datasets section body (see ``_section_body``), so
        dataset creation stays reachable once a dataset already exists
        rather than only before the first one. Renamed from
        ``_dataset_empty_actions`` accordingly.
        """
        yield Horizontal(
            Button("+ New dataset", id="evals-rail-new-dataset", compact=True),
            Button("Import…", id="evals-rail-import-dataset", compact=True),
            classes="evals-rail-empty-actions",
        )

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
        if button_id == "evals-rail-open-settings":
            event.stop()
            self.post_message(NavigateToScreen(TAB_SETTINGS))
            return
        if button_id == "evals-create-sample-bench":
            event.stop()
            self.post_message(self.SampleBenchRequested())
            return
        if button_id == "evals-rail-new-dataset":
            event.stop()
            self._create_new_dataset()
            return
        if button_id == "evals-rail-import-dataset":
            event.stop()
            self._open_dataset_import_dialog()
            return
        selection = self._row_targets.get(button_id)
        if selection is None:
            return
        event.stop()
        self.post_message(self.EvalsSelectionChanged(selection))

    def _create_new_dataset(self) -> None:
        db = self.view_model.db
        if db is None:
            self._notify("The evaluation service is unavailable.", severity="error")
            return
        name = f"{_NEW_DATASET_BASE_NAME} {uuid.uuid4().hex[:8]}"
        try:
            dataset_id = db.create_dataset(
                name=name, format="custom", source_path=f"inline:{name}"
            )
        except Exception as exc:
            self._notify(f"Could not create dataset: {exc}", severity="error")
            return
        self.post_message(
            self.EvalsSelectionChanged(EvalsSelection(kind="dataset", id=dataset_id))
        )

    def _open_dataset_import_dialog(self) -> None:
        filters = Filters(
            ("Text (one snippet per line)", lambda p: p.suffix.lower() == ".txt"),
            ("CSV", lambda p: p.suffix.lower() == ".csv"),
            ("JSON", lambda p: p.suffix.lower() == ".json"),
            ("All files", lambda p: True),
        )
        self.app.push_screen(
            FileOpen(title="Import as a new dataset", filters=filters),
            self._handle_dataset_import_file_selected,
        )

    def _handle_dataset_import_file_selected(self, path: Optional[Any]) -> None:
        """Creates a NEW dataset from an imported file in one step -- there
        is no existing dataset to import INTO yet (that is
        ``snippet_editor.SnippetEditor``'s job, once a dataset exists and is
        selected). Public-shaped so a test can drive it directly with a real
        temp file, bypassing the modal picker -- mirrors
        ``SnippetEditor._handle_import_file_selected``.
        """
        if not path:
            return
        db = self.view_model.db
        if db is None:
            self._notify("The evaluation service is unavailable.", severity="error")
            return
        try:
            file_path = validate_path_simple(path, require_exists=True)
        except ValueError as exc:
            self._notify(f"Could not read {Path(path).name}: {exc}", severity="error")
            return
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self._notify(f"Could not read {file_path.name}: {exc}", severity="error")
            return

        parser = _RAIL_IMPORT_PARSERS.get(
            file_path.suffix.lower(), parse_plain_text_snippets
        )
        try:
            new_snippets, skipped_count = parser(content)
        except ValueError as exc:
            self._notify(f"Import failed: {exc}", severity="error")
            return
        if not new_snippets:
            self._notify("No snippets found to import.", severity="warning")
            return

        dataset_name = f"{file_path.stem or 'Imported dataset'} {uuid.uuid4().hex[:8]}"
        try:
            dataset_id = db.create_dataset(
                name=dataset_name, format="custom", source_path=f"inline:{dataset_name}"
            )
            import_snippets_into_dataset(db, dataset_id, new_snippets)
        except Exception as exc:
            self._notify(f"Import failed: {exc}", severity="error")
            return

        message = f"Imported {len(new_snippets)} snippet(s) into a new dataset"
        if skipped_count:
            entry_word = "entry" if skipped_count == 1 else "entries"
            message += f"; skipped {skipped_count} invalid {entry_word}"
        self._notify(f"{message}.", severity="information")
        self.post_message(
            self.EvalsSelectionChanged(EvalsSelection(kind="dataset", id=dataset_id))
        )
