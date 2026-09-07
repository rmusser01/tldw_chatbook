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
satisfies ``provider_is_configured``) adds no SAMPLE-BENCH row at all.

**"+ New bench" has no provider gate at all** (task-1482, ``_new_bench_
actions``). Unlike the sample bench, creating a draft bench is a plain DB
write -- no network call -- so it renders (enabled whenever at least one
dataset exists) in both branches of ``_benches_section_body`` regardless
of ``provider_is_configured``. This closes the one cell the paragraph
above still left dark: a rail with real benches but a failed provider
gate used to offer no bench-creation affordance whatsoever.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Static

from ...Widgets.destination_rail import GLYPH_COLLAPSED, GLYPH_EXPANDED
from ...Constants import TAB_SETTINGS
from ...Evals.character_probe.probe_format import parse_probe_text
from ...Evals.character_probe.storage import (
    is_character_bench,
    is_probe_set,
    save_probe_set,
)
from ...Evals.word_bench.models import BenchConfig
from ...Evals.word_bench.storage import _unique_name, save_bench
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


async def _read_import_file_off_thread(file_path: Path) -> str:
    """Reads ``file_path``'s full text on a worker thread, never blocking
    the UI event loop.

    Qodo review (task-1691 phase 2 fix wave), platform compliance rule
    497164: a plain ``file_path.read_text()`` call runs synchronously on
    whatever thread calls it, and both rail import handlers below
    (``_handle_dataset_import_file_selected``, ``_handle_probe_import_
    file_selected``) are invoked as ``FileOpen`` dismiss callbacks running
    on Textual's own main/UI thread -- a slow disk or a large file would
    freeze the whole app for the duration of the read. This is the ONE
    seam both handlers share (the read itself is byte-for-byte identical
    between them; only what happens to the text afterward -- snippet
    parsing vs. probe-set parsing -- differs), so the fix lives here once
    rather than being duplicated into each handler and risking the two
    drifting apart again.

    ``asyncio.to_thread`` (not a Textual ``@work`` worker): both call
    sites are themselves plain callables handed to ``push_screen(...,
    callback)``, invoked through Textual's own ``invoke()`` helper
    (``textual._callback``), which already awaits an ``async def``
    callback -- see that helper's own ``isawaitable`` check. A one-off
    background-thread hop is all this single blocking call needs; a full
    worker (with its own cancellation/exclusivity semantics) would be
    strictly more machinery for no behavioural gain here.
    """
    return await asyncio.to_thread(file_path.read_text, encoding="utf-8")


def _default_open_sections() -> dict[str, bool]:
    return {section_id: True for section_id in RAIL_SECTIONS}


#: The design mockup's own subgroup label
#: (``Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md``,
#: "Classic orchestrator tasks appear in a labelled subgroup under
#: Benches"). Not a Button -- it never carries a selection, it only marks
#: where classic rows start.
CLASSIC_SUBGROUP_LABEL = "─ classic ─"

EVALS_RAIL_CLASSIC_ROW_PREFIX = "evals-rail-row-benches-classic-"

#: Prefixes a rail row whose bench or dataset belongs to the character-probe
#: eval, so the two kinds sharing one section are distinguishable at a
#: glance -- a probe set and a snippet dataset otherwise look identical, and
#: selecting a bench row is a guess about which detail pane will appear.
#: Single-width by construction: a double-width glyph would shift every rail
#: row's alignment (the ␣/⏎/✓✗ markers elsewhere follow the same rule; see
#: ``test_the_marker_glyph_is_single_width``).
CHARACTER_PROBE_MARKER = "◆ "


def _bench_row_label(row: dict[str, Any]) -> str:
    # escape_markup: `Button(label=...)` parses its argument as Textual
    # markup by default (`Content.from_text`'s own `markup=True` default),
    # so an unescaped bench name containing a bare `[/]` raises
    # `MarkupError` the instant the rail lays out -- the same hazard
    # `_run_group_row_label` (below) already closed for run rows. Bench
    # names are machine-generated today, but the bench-authoring program
    # makes them user-typed (task-1482).
    #
    # No CHARACTER_PROBE_MARKER check here: every row reaching this
    # function already passed `EvalsViewModel._is_word_bench` (it is only
    # ever called for `benches()` rows -- see `_benches_section_body`
    # below), and `is_character_bench`/`_is_word_bench` are mutually
    # exclusive (`config_data["bench_type"]` is either `"word_bench"` or
    # `"character_probe"`, never both). Marking here would be dead code.
    # A character-probe bench instead reaches `_classic_row_label` below,
    # since it is not a word bench and `classic_tasks()` has no other
    # category for it yet (see that method's own docstring).
    return escape_markup(str(row.get("name") or "Untitled bench"))


def _classic_row_label(row: dict[str, Any]) -> str:
    name = escape_markup(str(row.get("name") or "Untitled task"))
    return f"{CHARACTER_PROBE_MARKER}{name}" if is_character_bench(row) else name


def _dataset_row_label(row: dict[str, Any]) -> str:
    name = escape_markup(str(row.get("name") or "Untitled dataset"))
    return f"{CHARACTER_PROBE_MARKER}{name}" if is_probe_set(row) else name


#: Single-cell-width status glyphs for run rows -- NEVER emoji, which
#: render double-width in this app's terminal (a repeated past defect).
#: Status must never be conveyed by colour alone; the glyph itself is the
#: signal. Keyed by ``EvalsViewModel.run_groups()``'s rolled-up
#: ``"status"`` (TASK-1480) for the two states that don't depend on cell
#: data -- "completed" is deliberately absent here, since its glyph also
#: depends on ``"all_cells_failed"`` (see ``_run_group_row_glyph`` below).
_RUN_STATUS_GLYPHS: dict[str, str] = {
    "running": "●",  # ● BLACK CIRCLE
    "cancelled": "✗",  # ✗ BALLOT X -- also covers eval_runs' run-level
    # "failed" status, folded into "cancelled" by run_groups()'s roll-up.
}

#: A completed group where every captured cell errored -- TWO single-width
#: glyphs (CHECK MARK + BALLOT X), never one double-width character: the
#: run genuinely finished (``✓``), but produced nothing but failures
#: (``✗``). Ordering is deliberate -- "finished, then: all failures" reads
#: left-to-right the way the run itself happened.
_COMPLETED_ALL_FAILED_GLYPH = "✓✗"
#: A completed group with at least one successful cell (including a
#: completed group with zero captured cells at all -- vacuously "nothing
#: failed", per TASK-1480's amendment). Partial failures still render this
#: glyph; the results grid's own callout is what explains a partial
#: failure, not the rail row.
_COMPLETED_GLYPH = "✓"  # ✓ CHECK MARK


def _run_group_row_glyph(row: dict[str, Any]) -> str:
    """The leading status glyph for a run row (TASK-1480 + its amendment).

    An unrecognised or missing ``"status"`` (there should never be one --
    ``EvalsViewModel.run_groups()`` always sets exactly one of "running" /
    "cancelled" / "completed") falls through to the "completed" branch
    rather than raising, so a stale or malformed row degrades to a glyph
    instead of crashing the rail.
    """
    status = row.get("status")
    if status in _RUN_STATUS_GLYPHS:
        return _RUN_STATUS_GLYPHS[status]
    return _COMPLETED_ALL_FAILED_GLYPH if row.get("all_cells_failed") else _COMPLETED_GLYPH


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
    A completed group where every captured cell errored instead renders
    ``✓✗ 14:02 · <task_name>`` (TASK-1480 amendment, user-directed): the
    run genuinely finished, but produced nothing but failures, which is a
    materially different outcome from a normal or partially-failed
    completion (the results grid's own callout explains a partial
    failure; the plain ``✓`` in that case is unchanged).

    TASK-1480: before this, every run row rendered the exact same shape a
    bench row did (``"<name> (N targets)"``), so a live UAT pass could not
    tell a bench apart from one of its own past runs at a glance, and had
    no way to see a run currently in flight. See ``run_groups()``'s own
    docstring for how the leading glyph's status (and, for a completed
    group, ``all_cells_failed``) is rolled up from the group's per-target
    runs and captured cells.

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
    glyph = _run_group_row_glyph(row)
    # escape_markup: `_run_group_row_time`'s parse-failure fallback returns
    # the RAW `created_at` string verbatim (a free-text DB column with no
    # format enforcement -- see that function's own docstring), so it
    # carries the identical markup hazard `name` does. Escaping only `name`
    # and not this left the fallback path unescaped.
    time_text = escape_markup(_run_group_row_time(row.get("created_at")))
    return f"{glyph} {time_text} · {escape_markup(name)}"


class LibraryRail(NotifyMixin, Vertical):
    """Left rail: Benches, Datasets, Runs -- each collapsible, with counts."""

    class EvalsSelectionChanged(Message, namespace="library_rail"):
        """Posted when this rail changes the screen's selection.

        Two very different situations post this: a plain row press (the rows
        on screen are exactly the rows that exist -- only the active marker
        moves) and a rail-initiated MUTATION that selects what it just made
        (dataset import, "+ New bench", "+ New dataset", probe-set import),
        after which the rows on screen are stale.

        ``rail_dirty`` tells them apart so the screen can skip rebuilding the
        rail for the common case (task-15475). It defaults to ``True``, the
        answer that is never wrong -- only ``on_button_pressed``'s row-press
        path opts out. Getting this backwards leaves a rail missing the row
        the user just created, which is exactly what a first draft of this
        change did.
        """

        def __init__(
            self, selection: EvalsSelection, *, rail_dirty: bool = True
        ) -> None:
            super().__init__()
            self.selection = selection
            self.rail_dirty = rail_dirty

    class SampleBenchRequested(Message, namespace="library_rail"):
        """Posted when "Create sample bench" is pressed.

        Carries no payload -- creating and running the sample bench needs
        real DB/network work (``sample_bench.create_and_run_sample_bench``
        is a coroutine), so ``EvalsScreen`` runs it as a worker rather than
        this widget doing it inline, mirroring why row selection is a
        message rather than a direct call too.
        """

    class NewCharacterBenchRequested(Message, namespace="library_rail"):
        """Posted when "+ New character bench" is pressed.

        Unlike ``_create_new_bench``'s plain in-widget DB write (a draft
        word bench needs nothing beyond a dataset id), creating a draft
        character bench also resolves a target via ``sample_bench.
        resolve_unsteered_llama_cpp_target`` -- a sibling of the ``sample_
        bench.resolve_sample_target`` function ``EvalsScreen``'s own
        sample-bench flow calls (whole-branch review Critical 1: NOT the
        same function -- that one reuses ANY existing ``llama_cpp`` row
        with no regard for its steering, which is unsafe for a character
        probe; see the sibling function's own docstring). Reusing target-
        resolution logic (rather than reimplementing it here) is why this
        posts a message for ``EvalsScreen`` to handle instead of doing the
        write here, the way ``_create_new_bench`` does. See
        ``EvalsScreen._on_new_character_bench_requested`` for why a target
        must be resolved at CREATE time at all: the character-bench editor
        (task-1691 phase 2, Task 4) has no Add-target control of its own,
        so bench creation is the only place ``target_ids`` is ever
        populated -- leaving it empty here would ship a bench with no path
        to ever becoming runnable.
        """

    def __init__(
        self,
        view_model: EvalsViewModel,
        *,
        selection: Optional[EvalsSelection] = None,
        open_sections: Optional[dict[str, bool]] = None,
        app_config: Optional[dict[str, Any]] = None,
        sample_bench_running: bool = False,
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
        #: Whole-branch review: whether ANY word-bench run (this rail's own
        #: sample-bench worker, or a bench-run started from the primary
        #: action in the inspector pane) is in flight right now --
        #: `EvalsScreen` passes its own OR of both running-flags in. TASK-
        #: 1478 made "Create sample bench" a PERSISTENT control (no longer
        #: empty-only), which opened a stale-enabled-button seam identical
        #: to `_primary_action_state`'s own (see that function's in-flight
        #: branch): a rail click during an in-flight run recomposes this
        #: whole widget from scratch, and without this flag the fresh
        #: instance would render the button enabled again regardless of
        #: what is actually running. ``False`` in every context that
        #: doesn't pass it explicitly (production callers other than
        #: ``EvalsScreen``, if any, and every pre-existing test).
        self.sample_bench_running = sample_bench_running
        # Shared, mutated in place (never reassigned) rather than copied:
        # EvalsScreen holds this same dict and passes it back in on every
        # recompose, so a section's collapsed/expanded state survives the
        # screen-level `refresh(recompose=True)` that a selection change
        # triggers (which tears down and rebuilds this whole widget).
        self.open_sections = (
            open_sections if open_sections is not None else _default_open_sections()
        )
        self._row_targets: dict[str, EvalsSelection] = {}

    def apply_selection(self, selection: EvalsSelection) -> None:
        """Re-mark the active row in place, without rebuilding the rail.

        task-15475. A rail click cannot change which ROWS exist -- this
        widget is what posted the selection -- only which one is marked
        active, so the screen no longer rebuilds the rail for it. The marking
        rule is the same one ``_row_button`` applies at compose time, read off
        the same ``_row_targets`` map, so a re-marked rail and a
        freshly-composed one agree by construction.

        Callers whose rows genuinely changed (a save, a finished run, a
        delete) must rebuild the rail instead -- see ``EvalsScreen.select``'s
        ``rail_dirty``.

        Args:
            selection: The screen's new selection.

        Returns:
            None.
        """
        self.selection = selection
        for button_id, target in self._row_targets.items():
            try:
                button = self.query_one(f"#{button_id}", Button)
            except NoMatches:
                # A collapsed/removed section's row; nothing to mark.
                continue
            button.set_class(target == selection, "is-active")

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

    def _create_sample_bench_button(self) -> Button:
        """Shared by both branches of ``_benches_section_body`` (TASK-1478:
        the button is no longer empty-only, so both the "no benches yet"
        and "benches already exist" paths need the identical control, id
        included -- see the module docstring's "Creation affordances are
        not empty-only" note).

        Whole-branch review: no longer a ``@staticmethod`` -- it now reads
        ``self.sample_bench_running`` to stay disabled across a mid-run
        recompose (this button is PERSISTENT since TASK-1478, so a rail
        click while a run is in flight would otherwise rebuild a fresh,
        enabled instance; see ``self.sample_bench_running``'s own comment
        in ``__init__``). The LIVE running label
        (``EvalsScreen._set_sample_bench_running_ui``, driven by
        ``query_one`` against the mounted button) is unaffected by this --
        it still mutates the existing widget directly while a run
        progresses; this only matters for a FRESH instance built by a
        recompose landing mid-run.
        """
        return Button(
            "Create sample bench",
            id="evals-create-sample-bench",
            disabled=self.sample_bench_running,
            tooltip=(
                "A bench run is already in flight."
                if self.sample_bench_running
                else "Creates the loaded-nouns sample dataset, wires it to "
                "a configured target, and runs it."
            ),
        )

    def _new_bench_actions(self) -> ComposeResult:
        """"+ New bench" / "+ New character bench": create a draft bench,
        in-widget for the word-bench case (no worker -- a draft bench is a
        plain DB write, exactly like ``_create_new_dataset``), routed
        through ``EvalsScreen`` for the character-bench case (see
        ``NewCharacterBenchRequested``'s own docstring for why). Shared by
        both branches of ``_benches_section_body`` (task-1482 / task-1691
        phase 2), in one ``Horizontal`` row mirroring ``_dataset_actions``'s
        own shape -- ``#lab-rail .evals-rail-empty-actions`` (``_lab.tcss``)
        stacks this row vertically at the rail's narrow width regardless of
        the Python-level ``Horizontal``, the same fix already proven for
        ``_dataset_actions``'s three buttons, so a second button here does
        not reopen the side-by-side clipping ``_lab.tcss``'s own comment
        documents.

        Deliberately NOT gated on ``sample_bench.provider_is_configured``:
        creating a bench writes only ``eval_tasks``/``eval_datasets`` rows,
        no network call -- unlike "Create sample bench", which also RUNS
        the bench against a real target. This closes a latent cell the
        module docstring's "Creation affordances are not empty-only" note
        left unaddressed: a rail with real benches but a failed provider
        gate used to render no bench-creation affordance at all (see
        ``_benches_section_body``'s own updated comment on this).

        Both buttons are disabled -- with an explanatory tooltip AND an
        adjacent one-line hint, never a silent no-op (the fix-batch
        convention) -- when there is nothing yet to bind the new bench to:
        a WORD-BENCH dataset (``EvalsViewModel.word_bench_datasets()``) for
        "+ New bench", a probe set (``EvalsViewModel.probe_sets()``) for
        "+ New character bench" -- a character bench with no probe set has
        no probes to ever run.

        Whole-branch review Important 3 (fix round): ``has_dataset`` reads
        ``word_bench_datasets()``, not ``datasets()`` -- a probe set is a
        dataset row too, but it holds zero snippets a word bench could
        ever measure. Before this fix, a probe-set-only library still
        showed "+ New bench" enabled (see ``word_bench_datasets()``'s own
        docstring for the exact failure chain this closes).
        """
        has_dataset = bool(self.view_model.word_bench_datasets())
        has_probe_set = bool(self.view_model.probe_sets())
        yield Horizontal(
            Button(
                "+ New bench",
                id="evals-rail-new-bench",
                compact=True,
                disabled=not has_dataset,
                tooltip=(
                    "Create or import a dataset first."
                    if not has_dataset
                    else "Creates a draft bench bound to the selected "
                    "dataset (or the newest one, if none is selected)."
                ),
            ),
            Button(
                "+ New character bench",
                id="evals-rail-new-character-bench",
                compact=True,
                disabled=not has_probe_set,
                tooltip=(
                    "Import or create a probe set first."
                    if not has_probe_set
                    else "Creates a draft character-probe bench bound to "
                    "the newest probe set."
                ),
            ),
            classes="evals-rail-empty-actions",
        )
        if not has_dataset:
            # A DEDICATED class, not `.evals-rail-empty-copy` (same visual
            # treatment, different identity): `test_first_run_marks_the_
            # sample_bench_as_the_recommended_first_step` scopes on that
            # exact class, within this exact section, to confirm "No
            # benches yet." was REPLACED by the "Start here" hint, not
            # supplemented -- this hint answers a different question (why
            # "+ New bench" is disabled) and legitimately coexists with
            # either, so it must not be mistaken for a second copy of that
            # wording by a class-scoped query.
            yield Static(
                "Create or import a dataset first.",
                id="evals-rail-new-bench-hint",
                classes="evals-rail-new-bench-hint",
                markup=False,
            )
        if not has_probe_set:
            # A dedicated id (distinct from `#evals-rail-new-bench-hint`)
            # sharing the SAME visual class -- both hints are one-line,
            # muted "why this button is disabled" copy, and nothing here
            # scopes a query on the shared class alone (see the comment
            # just above for the one place that matters, which is id-
            # scoped already).
            yield Static(
                "Import or create a probe set first.",
                id="evals-rail-new-character-bench-hint",
                classes="evals-rail-new-bench-hint",
                markup=False,
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
            # note above), no SAMPLE-BENCH row is added -- the "Open
            # Settings" escape hatch below is scoped to the fully-empty
            # branch and must not be duplicated for a rail that already
            # has real benches. "+ New bench" (task-1482) is a separate
            # affordance with no provider gate at all -- see
            # `_new_bench_actions`'s own docstring -- so it always renders
            # here regardless of `provider_ready`.
            if provider_ready:
                children.append(self._create_sample_bench_button())
            children.extend(self._new_bench_actions())
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
            # "+ New bench" (task-1482): no provider gate, so it renders
            # regardless of which branch just ran above -- see
            # `_new_bench_actions`'s own docstring.
            children.extend(self._new_bench_actions())

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
                        # A character-probe bench renders in this same
                        # subgroup (marked with CHARACTER_PROBE_MARKER --
                        # see _classic_row_label), but selecting one must
                        # route to its OWN detail surface, never
                        # ClassicTaskDetail's read-only one -- see
                        # EvalsScreen._compose_detail_pane's
                        # "character_bench" branch (task-1691 phase 2,
                        # Task 5). A genuinely classic (pre-word-bench)
                        # task keeps kind="classic".
                        kind="character_bench" if is_character_bench(row) else "classic",
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
            Button(
                "Import probes…", id="evals-rail-import-probes", compact=True
            ),
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
        if button_id == "evals-rail-new-bench":
            event.stop()
            self._create_new_bench()
            return
        if button_id == "evals-rail-new-character-bench":
            event.stop()
            self.post_message(self.NewCharacterBenchRequested())
            return
        if button_id == "evals-rail-new-dataset":
            event.stop()
            self._create_new_dataset()
            return
        if button_id == "evals-rail-import-dataset":
            event.stop()
            self._open_dataset_import_dialog()
            return
        if button_id == "evals-rail-import-probes":
            event.stop()
            self._open_probe_import_dialog()
            return
        selection = self._row_targets.get(button_id)
        if selection is None:
            return
        event.stop()
        # The ONE opt-out (task-15475): a row press changes nothing about
        # which rows exist -- this rail composed them -- so the screen
        # re-marks the active row in place instead of rebuilding the rail.
        self.post_message(self.EvalsSelectionChanged(selection, rail_dirty=False))

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

    def _create_new_bench(self) -> None:
        """Creates a draft ``BenchConfig`` bound to a dataset and selects
        it -- the Benches-section mirror of ``_create_new_dataset``.
        In-widget, no worker: a draft bench is a plain DB write (an
        ``eval_tasks`` row with ``target_ids=()``, no targets wired yet),
        never a network call, so there is nothing here that needs a
        background worker the way running a bench does.

        Dataset binding (task-1482, pinned): the currently selected
        dataset if one is selected and still resolves, else the newest
        dataset -- ``view_model.word_bench_datasets()`` is already newest-
        first (``EvalsDB.list_datasets``'s own ``ORDER BY created_at
        DESC``, ``word_bench_datasets()``'s own filter preserves that
        order -- see its own docstring), so ``datasets[0]`` IS "the newest
        one" with no extra sort needed. A stale ``kind="dataset"``
        selection (its id no longer resolves -- e.g. the dataset was
        deleted from under the rail, OR (whole-branch review Important 3,
        fix round) it resolves to a PROBE SET, which this method must
        never bind a word bench to -- see ``word_bench_datasets()``'s own
        docstring) degrades to the same newest-word-bench-dataset fallback
        rather than creating an unbound or probe-set-bound bench.
        """
        db = self.view_model.db
        if db is None:
            self._notify("The evaluation service is unavailable.", severity="error")
            return
        datasets = self.view_model.word_bench_datasets()
        if not datasets:
            # The button is disabled whenever this is true (see
            # `_new_bench_actions`) -- reachable only via a stale render or
            # a direct press bypassing the widget, but the fix-batch
            # convention is a real toast, never a silent no-op.
            self._notify("Create or import a dataset first.", severity="warning")
            return
        dataset = None
        if self.selection.kind == "dataset" and self.selection.id:
            dataset = next(
                (row for row in datasets if row.get("id") == self.selection.id), None
            )
        if dataset is None:
            dataset = datasets[0]
        dataset_name = str(dataset.get("name") or "Untitled dataset")
        config = BenchConfig(
            name=_unique_name("Untitled bench"),
            prompt_mode="raw",
            top_k=20,
            dataset_id=dataset["id"],
            target_ids=(),
        )
        try:
            bench_id = save_bench(db, config)
        except Exception as exc:
            self._notify(f"Could not create bench: {exc}", severity="error")
            return
        self._notify(f"Bench created against {dataset_name}.")
        self.post_message(
            self.EvalsSelectionChanged(EvalsSelection(kind="bench", id=bench_id))
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

    async def _handle_dataset_import_file_selected(self, path: Optional[Any]) -> None:
        """Creates a NEW dataset from an imported file in one step -- there
        is no existing dataset to import INTO yet (that is
        ``snippet_editor.SnippetEditor``'s job, once a dataset exists and is
        selected). Public-shaped so a test can drive it directly with a real
        temp file, bypassing the modal picker -- mirrors
        ``SnippetEditor._handle_import_file_selected``.

        ``async`` (Qodo review, task-1691 phase 2 fix wave): the file read
        below now hops onto a worker thread via ``_read_import_file_off_
        thread`` -- see that function's own docstring for why, and why an
        ``async def`` callback here costs nothing extra (``push_screen``'s
        own ``invoke()`` helper already awaits it).
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
            content = await _read_import_file_off_thread(file_path)
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

    def _open_probe_import_dialog(self) -> None:
        """Mirrors ``_open_dataset_import_dialog``: no filters, since the
        character-probe plain-text format (``---``/``===`` delimited, see
        ``character_probe.probe_format``) has no standard file extension of
        its own the way snippet CSV/JSON does.
        """
        self.app.push_screen(
            FileOpen(title="Import probe set"),
            self._handle_probe_import_file_selected,
        )

    async def _handle_probe_import_file_selected(self, path: Optional[Any]) -> None:
        """Creates a NEW probe-set dataset from an imported plain-text file.

        Public-shaped (not ``_on_...``) so a test can drive it directly with
        a real temp file, bypassing the modal picker -- mirrors
        ``_handle_dataset_import_file_selected``'s own convention for
        snippet imports.

        ``async`` (Qodo review, task-1691 phase 2 fix wave): the file read
        below now hops onto a worker thread via ``_read_import_file_off_
        thread`` -- the SAME helper ``_handle_dataset_import_file_
        selected`` uses, so the two siblings stay identical on this point
        rather than diverging. See that function's own docstring for why,
        and why an ``async def`` callback here costs nothing extra
        (``push_screen``'s own ``invoke()`` helper already awaits it).

        Args:
            path: The chosen file, or ``None``/falsy when the dialog was
                cancelled.
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
            text = await _read_import_file_off_thread(file_path)
        except (OSError, UnicodeDecodeError) as exc:
            self._notify(f"Could not read {file_path.name}: {exc}", severity="error")
            return
        try:
            probe_set = parse_probe_text(text)
        except ValueError as exc:
            self._notify(
                f"That file is not a valid probe set: {exc}", severity="error"
            )
            return

        dataset_name = f"{file_path.stem or 'Imported probes'} {uuid.uuid4().hex[:8]}"
        try:
            dataset_id = save_probe_set(db, dataset_name, probe_set)
        except Exception as exc:
            self._notify(f"Import failed: {exc}", severity="error")
            return

        count = len(probe_set.probes)
        probe_word = "probe" if count == 1 else "probes"
        self._notify(f"Imported {count} {probe_word} into a new probe set.")
        self.post_message(
            self.EvalsSelectionChanged(EvalsSelection(kind="dataset", id=dataset_id))
        )
