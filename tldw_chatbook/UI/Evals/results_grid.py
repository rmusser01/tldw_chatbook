"""Results grid: pivots a word bench run group's cells into a lensed,
focusable table.

Mounted by ``evals_screen.py``'s ``_compose_detail_pane`` for a
``selection.kind == "run_group"`` selection, replacing the placeholder
``Static`` fields that branch used to yield. This is the centrepiece of the
Evals workbench and the most likely place to misrepresent the engine's
numbers -- **every value shown in a cell comes from ``word_bench.analysis``
or a ``CellCapture``/``ProbeReading`` property the engine already computed.
This module adds no arithmetic beyond formatting** (percent/nats rounding,
token-quoting so whitespace is visible) -- see the self-review note at the
bottom of this docstring before adding anything that looks like a
computation here.

Five lenses decide what a cell renders: Top-1, Entropy, Probe, Truncation
(``truncated_mass`` -- labelled "Truncation" rather than the plan's
original "Coverage": the raw quantity is MISSING, unobserved mass, and a
user reading a high percentage next to the word "Coverage" would
reasonably expect "well measured", the opposite of what a high
``truncated_mass`` means. The internal lens key/Select value stays
``"coverage"`` for stability; only the displayed label changed, and the
quantity is still the plan's unobserved mass -- read at the shared
effective K, see 4 below), and Δ baseline. Four of them would
misrepresent the engine if rendered naively, and each is pinned by a test
in ``Tests/UI/test_evals_results_grid.py``:

1. **A bare Top-1 winner on a near-tie.** Two identical requests to the same
   server, seconds apart at the same neutral sampler, returned the top two
   tokens in OPPOSITE rank order, magnitudes stable to ~0.002 nats (see
   ``Tests/Evals/word_bench/test_normalizer.py::
   test_a_near_tie_between_the_top_two_is_visible_in_the_fixture``, which
   pins the same live-captured fixture). A grid that renders a bare winner
   there shows a spurious difference between cells that are statistically
   identical. ``_render_top1`` marks the tie instead by calling
   ``analysis.near_tie`` -- see ``analysis.NEAR_TIE_LOGPROB_GAP_NATS`` for
   the threshold and its rationale (moved into ``analysis.py`` so the
   methodology -- what counts as "too close to call" -- lives with the
   rest of the methodology, not split across the engine and the view).
2. **A "≥" prefix on divergence.** The original design spec claimed
   divergence was a lower bound; PR 2's whole-branch review disproved that
   with a feasible counterexample (0.291 reported against 0.121 true --
   see ``analysis.divergence``'s own docstring and
   ``test_divergence_is_an_estimate_not_a_guaranteed_bound``). The Δ
   baseline lens renders the number plainly and marks high-truncation
   cells with a trailing ``!`` (``analysis.TRUNCATION_WARN_THRESHOLD``),
   never with a leading ``≥``.
3. **Entropy without a shared K.** ``analysis.divergence`` truncates both
   cells to ``min(K)`` so its number reflects behaviour, not settings;
   entropy must do the same. ``_render_table`` computes
   ``analysis.effective_k`` ONCE per render over every successfully
   captured cell and passes it into every ``analysis.entropy`` call, and
   the grid header states that effective K.
4. **Truncation at each cell's own native K.** Same failure as 3, one lens
   over, and it shipped there first: ``CellCapture.truncated_mass`` is
   computed over each cell's FULL native top-K, so a K=20 and a K=5 cell
   holding the same distribution over the first 5 ranks rendered 1% vs 2%
   while the header advertised ``K 5``. The lens now reads
   ``analysis.truncated_mass(cap, k=effective_k)`` -- the same shared K
   the header states and entropy uses.

A fifth misrepresentation is not about a number but about its label: the
Probe lens can only show ONE probe per cell, so the state line NAMES the
active probe (``_lens_description``) and ``#evals-probe-selector`` (mounted
when a bench configures more than one) makes the others reachable.
Unattributable numbers -- a column of ``-1.83  16.0%`` with no way to tell
which probe produced them -- are their own kind of false precision.

Cell states, never conflated (see each lens's own render function):

- **unrun** (no row for this ``(snippet, target)`` in ``load_grid``'s
  ``cells``) -> blank. Never ``0``.
- **failed** (``CellError``) -> ``FAILED_MARK`` ("—"), with the error's
  reason/detail surfaced in the inspector when the cell is focused (see
  ``ResultsGrid.CellFocused`` below). Never ``0``.
- **warned column** (the target's ``PreflightResult.is_warned``, carried
  through the run's stored preflight snapshot) -> the column header carries
  a readable ``" [warned]"`` suffix, so a large divergence in that column is
  never silently read as a finding about content rather than about a
  preflighted-degenerate target.

Interaction: arrow keys move the ``DataTable``'s own cell cursor (built in,
not reimplemented here); ``DataTable.CellHighlighted`` fires on every move,
and ``_on_cell_highlighted`` turns that into a ``ResultsGrid.CellFocused``
message the screen forwards to ``inspector.EvalsCellInspector`` -- a
TARGETED update (``EvalsCellInspector.show_cell()``, a plain method call
against an already-mounted widget), never a screen-level
``refresh(recompose=True)``. Recomposing on every arrow-key press would
tear down and rebuild the ``DataTable``, losing cursor position and
re-reading the run group from the database on every keystroke; see the
PR 3a "widget present but zero size" trap this program has already hit once
for why targeted updates matter here, not just for cursor continuity. `l`
(lens), `b` (baseline), `s` (sort) are ``ResultsGrid.BINDINGS`` for the same
reason -- switching lens/baseline/sort mutates state and calls
``_render_table``/``_render_header`` directly against the mounted
``DataTable``/``Static``s, never a recompose.

Export (`e`, ``action_export``) writes CSV for the ACTIVE lens or JSON for
the whole run group, chosen by the extension of the ``FileSave``-picked
destination. Both formats are built from ``_compute_active_lens_rows``
(CSV) and the raw ``self._grid`` snapshot/cells (JSON) -- never a second,
parallel read of the run group -- so an export can never show numbers the
on-screen grid itself did not.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Literal, Optional

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import DataTable, Select, Static

from ...Evals.word_bench import analysis
from ...Evals.word_bench.models import CellCapture, CellError, PreflightResult
from ...Evals.word_bench.storage import load_grid
from ...Third_Party.textual_fspicker import FileSave, Filters
from ...Utils.path_validation import validate_path_simple
from .evals_state import EvalsViewModel

LensKey = Literal["top1", "entropy", "probe", "coverage", "delta"]
BaselineMode = Literal["column", "row"]
SortMode = Literal["none", "desc", "asc"]

LENS_LABELS: dict[LensKey, str] = {
    "top1": "Top-1",
    "entropy": "Entropy",
    "probe": "Probe",
    # Internal key stays "coverage" (stable Select value / lens-cycle
    # identity), but the raw quantity is `truncated_mass` -- the MISSING,
    # unobserved probability mass, not how much was captured. Labelling
    # that "Coverage" inverts the natural reading: a user seeing a HIGH
    # percentage next to "Coverage" would reasonably expect "well
    # measured", when a high truncated_mass means the opposite -- exactly
    # the misreading-by-word-choice class of defect this PR exists to
    # prevent, even though the NUMBER itself is unchanged and matches the
    # plan's own "Coverage (`truncated_mass`)" mapping. "Truncation" reads
    # correctly in both directions: high = more was missed.
    "coverage": "Truncation",
    "delta": "Δ baseline",
}
#: Cycle order for the `l` key -- also the ``#evals-lens-selector`` option
#: order, so the keyboard shortcut and the dropdown never disagree about
#: "next".
LENS_ORDER: tuple[LensKey, ...] = ("top1", "entropy", "probe", "coverage", "delta")

#: Never "0" -- a failed or unrun cell must not read as "measured and found
#: nothing". Unrun cells render as a plain empty string instead (see
#: ``_render_cell``); this mark is only for a cell that was measured and
#: came back an error.
FAILED_MARK = "—"


@dataclass(frozen=True)
class _DeltaReading:
    """One Δ-baseline cell's rendered text plus the context behind it --
    see ``ResultsGrid._delta_reading``. ``is_real_comparison`` is ``False``
    for a baseline position, an unrun cell, or an unavailable comparison
    (baseline itself failed) -- exactly the cases where ``jsd`` must not
    feed ``analysis.group_means`` (see ``_render_delta``) and where the
    inspector has no divergence to explain."""

    text: str
    is_real_comparison: bool = False
    jsd: Optional[float] = None
    is_bounded: Optional[bool] = None
    combined_truncated_mass: Optional[float] = None


def _safe_cell(value: str) -> Text:
    """Wraps a string as a literal ``rich.text.Text`` before handing it to
    ``DataTable``.

    ``DataTable``'s own ``default_cell_formatter`` (and ``add_column``'s
    label handling) run any plain ``str`` cell/column value through
    ``Text.from_markup`` -- confirmed by an early version of this module's
    own test suite, where a column label of ``"steered [warned]"`` silently
    rendered as ``"steered "`` because Rich's markup parser consumed
    ``"[warned]"`` as an (unknown, self-closing) style tag rather than
    literal text. Every string this module puts into a ``DataTable`` can
    contain a user-authored snippet or target name (free text, may contain
    ``[...]``) or this module's own literal annotations (``" (warned)"``,
    group brackets), so every one of them is wrapped here rather than only
    the ones observed to collide today. ``Text(value)`` -- the plain
    constructor, never ``Text.from_markup`` -- is never re-parsed, mirroring
    every other widget in this package's ``markup=False`` convention for
    the exact same reason.
    """
    return Text(value)


def render_token(token: str) -> str:
    """The canonical token renderer: wraps a token in literal quotes so
    whitespace is visible. ``" a"`` and ``"a"`` are different tokens in a
    grid about token-level behaviour and must not look identical -- a bare
    ``a`` vs. `` a`` in a terminal cell is nearly impossible to tell apart,
    especially with the cell-padding a ``DataTable`` adds.
    """
    return f'"{token}"'


def render_probe_reading(reading: "analysis.ProbeReading") -> str:
    """Format one probe reading, shared between the Probe lens
    (``_render_cell``) and the focused-cell inspector
    (``inspector.EvalsCellInspector``) so the two never disagree about what
    "observed" / "bounded" / "never observed" look like.

    The percentage comes from ``reading.matched`` -- the ``TokenProb``
    ``analysis.resolve_probe`` itself matched inside the cell's ``top_k``.
    Both call sites used to re-derive that with a hand-copied
    ``token == probe`` predicate; reading it off the ``ProbeReading``
    instead means a change to the engine's matching rule (e.g. to the
    bytes-based ``TokenProb.identity()`` the rest of ``analysis`` aligns
    on) reaches every renderer automatically. ``TokenProb.prob`` is the
    engine's own logprob-to-probability conversion, reused for display
    exactly as the engine defines it rather than this module calling
    ``math.exp`` on a bare logprob itself.
    """
    if reading.state == "never_observed":
        return "never observed"
    if reading.state == "bounded":
        if reading.logprob is None:
            return FAILED_MARK
        return f"< {reading.logprob:.2f}"
    # observed
    if reading.matched is not None:
        return f"{reading.logprob:.2f}  {reading.matched.prob * 100:.1f}%"
    return f"{reading.logprob:.2f}"


class ResultsGrid(Vertical):
    """Detail-pane content for a selected run group: the pivoted results
    grid, its lens/baseline controls, and the header stating the effective
    K, cell/failure counts, and which lens/baseline/sort is active."""

    BINDINGS = [
        Binding("l", "cycle_lens", "lens", show=False),
        Binding("b", "cycle_baseline", "baseline", show=False),
        Binding("s", "cycle_sort", "sort", show=False),
        Binding("e", "export", "export", show=False),
    ]

    can_focus = False  # the DataTable child is the focusable element

    class CellFocused(Message):
        """Posted when the grid's cell cursor highlights a new cell.
        Caught by ``evals_screen.py`` and forwarded to
        ``inspector.EvalsCellInspector.show_cell`` -- a targeted update,
        not a recompose (see the module docstring)."""

        def __init__(
            self,
            *,
            snippet_id: str,
            target_id: str,
            snippet_text: str,
            target_name: str,
            cell: CellCapture | CellError | None,
            probes: tuple[str, ...],
            ever_observed: dict[str, bool],
            delta: Optional["_DeltaReading"] = None,
        ) -> None:
            self.snippet_id = snippet_id
            self.target_id = target_id
            self.snippet_text = snippet_text
            self.target_name = target_name
            self.cell = cell
            self.probes = probes
            self.ever_observed = ever_observed
            #: The SAME ``_DeltaReading`` ``_render_cell`` used to draw this
            #: cell's grid text, present only when the Δ lens is active and
            #: this is a real comparison (never the "baseline" literal, an
            #: unrun cell, or an unavailable comparison). Carries the
            #: divergence, whether it is flagged, and the COMBINED
            #: truncated mass that triggered the flag -- the ``!`` marker
            #: is the grid's entire substitute for the leading ``≥`` PR 2's
            #: review disproved, so the inspector must be able to explain
            #: it in the SAME units it was decided in, not recompute a
            #: possibly-different number (see ``analysis.combined_
            #: truncation``'s own docstring for why a naive per-cell sum
            #: would disagree with the real figure at mixed K).
            self.delta = delta
            super().__init__()

    def __init__(
        self, view_model: EvalsViewModel, run_group_id: str, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._view_model = view_model
        self._run_group_id = run_group_id
        #: Loaded once in ``compose()`` -- never re-read from the database
        #: on a lens/baseline/sort change, only on a fresh selection (a new
        #: ``ResultsGrid`` instance). See ``storage.load_grid``'s own
        #: docstring: draining every ``eval_results`` page is real DB work.
        self._grid: Optional[dict[str, Any]] = None
        self._lens: LensKey = "top1"
        self._baseline_mode: BaselineMode = "column"
        self._baseline_index: int = 0
        self._sort_mode: SortMode = "none"
        #: Which of the bench's configured probes the Probe lens reads.
        #: The lens can only show one at a time, so WHICH one must be
        #: switchable (``#evals-probe-selector``, mounted only when there
        #: is more than one) and, always, NAMED in the state line -- a
        #: column of bare ``-1.83  16.0%`` readings a user cannot attribute
        #: to a probe is unattributable data, and the other probes would
        #: otherwise be unreachable from the grid entirely.
        self._probe_index: int = 0

    # -- compose -----------------------------------------------------

    def compose(self) -> ComposeResult:
        db = self._view_model.db
        if db is None:
            yield Static(
                "The evaluation service is unavailable.",
                id="evals-grid-unavailable",
            )
            return
        try:
            self._grid = load_grid(db, self._run_group_id)
        except ValueError:
            yield Static(
                "This run's data could not be loaded; it may have been "
                "deleted.",
                id="evals-grid-error",
            )
            return

        snapshot = self._grid["snapshot"]
        if not snapshot.get("snippets") or not snapshot.get("targets"):
            # Drop the loaded grid: this branch yields NO DataTable, and
            # ``on_mount`` (plus every lens/baseline/sort action and the
            # export path) keys off ``self._grid is None`` to decide there
            # is a table to render. Leaving it populated made the friendly
            # empty state dead on arrival -- ``_render_table``'s
            # ``query_one("#evals-grid-table")`` raised ``NoMatches`` out
            # of ``on_mount``, taking the app down on the one input this
            # branch exists to handle.
            self._grid = None
            yield Static(
                "This run has no snippets or no targets to render.",
                id="evals-grid-empty",
            )
            return

        # markup=False: both Statics carry user-authored text (bench name,
        # snippet text via _baseline_description()) interpolated by
        # _render_header() below -- see _safe_cell's docstring for the
        # exact same defect (Rich markup parsing "[...]") on the DataTable
        # side of this module. A Static with markup enabled runs its
        # `.update(str)` argument through the identical `Text.from_markup`
        # path.
        yield Static("", id="evals-grid-meta", markup=False)
        yield Static("", id="evals-grid-state", markup=False)
        with Horizontal(id="evals-grid-controls"):
            yield Select(
                [(LENS_LABELS[k], k) for k in LENS_ORDER],
                value=self._lens,
                id="evals-lens-selector",
                allow_blank=False,
            )
            yield Select(
                self._baseline_options(),
                value=self._baseline_value(),
                id="evals-baseline-selector",
                allow_blank=False,
            )
            probes = tuple(snapshot.get("probes") or ())
            if len(probes) > 1:
                # Only when there is a choice to make: a one-option Select
                # is noise, and the single probe is already named in the
                # state line either way (see _render_header).
                yield Select(
                    [
                        (_safe_cell(f"Probe · {render_token(p)}"), index)
                        for index, p in enumerate(probes)
                    ],
                    value=min(self._probe_index, len(probes) - 1),
                    id="evals-probe-selector",
                    allow_blank=False,
                )
        yield DataTable(id="evals-grid-table", cursor_type="cell", zebra_stripes=True)

    def on_mount(self) -> None:
        if self._grid is None:
            return
        self._render_table()
        self._render_header()
        # `l`/`b`/`s` are advertised in the footer the instant a run group
        # is selected (see evals_screen.py's _register_grid_shortcuts),
        # but Textual key bindings only resolve against the FOCUSED
        # widget's ancestor chain -- nothing focuses the grid's DataTable
        # by default, so those keys would be dead until the user tabs or
        # clicks in, silently contradicting what the footer just promised.
        self.query_one("#evals-grid-table", DataTable).focus()

    # -- baseline option plumbing -------------------------------------

    def _baseline_options(self) -> list[tuple[Text, tuple[str, str]]]:
        """Option labels are ``rich.text.Text``, not plain ``str`` -- both
        embed user-authored free text (target names, snippet text) that
        can legally contain ``[...]``. Confirmed on this project's pinned
        Textual version: a plain-string option label of
        ``"row · The rioters [loaded] were"`` silently renders with the
        bracketed span stripped, and one containing ``"a[/]b"`` raises
        ``MarkupError`` outright -- the same ``Text.from_markup`` defect
        ``_safe_cell`` documents for ``DataTable``, one widget over. See
        ``test_baseline_selector_options_survive_markup_special_characters_
        in_snippet_text`` in ``Tests/UI/test_evals_results_grid.py``.
        """
        snapshot = self._grid["snapshot"]
        options: list[tuple[Text, tuple[str, str]]] = []
        for target in snapshot["targets"]:
            options.append(
                (_safe_cell(f"Column · {target['name']}"), ("column", target["id"]))
            )
        for snippet in snapshot["snippets"]:
            text = snippet["text"]
            label = text if len(text) <= 28 else f"{text[:27]}…"
            options.append((_safe_cell(f"Row · {label}"), ("row", snippet["id"])))
        return options

    def _baseline_value(self) -> tuple[str, str]:
        ref_id = (
            self._baseline_target_id()
            if self._baseline_mode == "column"
            else self._baseline_snippet_id()
        )
        return (self._baseline_mode, ref_id or "")

    def _baseline_target_id(self) -> Optional[str]:
        targets = self._grid["snapshot"].get("targets") or []
        if not targets:
            return None
        idx = min(self._baseline_index, len(targets) - 1)
        return targets[idx]["id"]

    def _baseline_snippet_id(self) -> Optional[str]:
        snippets = self._grid["snapshot"].get("snippets") or []
        if not snippets:
            return None
        idx = min(self._baseline_index, len(snippets) - 1)
        return snippets[idx]["id"]

    def _index_for_baseline(self, mode: BaselineMode, ref_id: str) -> int:
        rows = self._grid["snapshot"]["targets" if mode == "column" else "snippets"]
        for index, row in enumerate(rows):
            if row["id"] == ref_id:
                return index
        return 0

    def _lens_description(self) -> str:
        """The active lens, with the Probe lens's own probe NAMED.

        The Probe lens can only render one probe per cell. Saying just
        "Lens: Probe" above a column of ``-1.83  16.0%`` leaves the reader
        unable to attribute the number to a probe at all -- and with more
        than one configured, which of them is showing is a real choice the
        reader has to be able to see (and, via ``#evals-probe-selector``,
        change).
        """
        label = LENS_LABELS[self._lens]
        if self._lens != "probe":
            return label
        probes = self._probes()
        if not probes:
            return f"{label} (no probes configured)"
        probe = self._active_probe()
        if len(probes) == 1:
            return f"{label} ({render_token(probe)})"
        position = min(self._probe_index, len(probes) - 1) + 1
        return f"{label} ({render_token(probe)} · {position} of {len(probes)})"

    def _baseline_description(self) -> str:
        if self._baseline_mode == "column":
            targets = self._grid["snapshot"].get("targets") or []
            idx = min(self._baseline_index, max(len(targets) - 1, 0))
            name = targets[idx]["name"] if targets else "?"
            return f"column · {name}"
        snippets = self._grid["snapshot"].get("snippets") or []
        idx = min(self._baseline_index, max(len(snippets) - 1, 0))
        text = snippets[idx]["text"] if snippets else "?"
        return f"row · {text}"

    # -- lens/baseline/sort controls -----------------------------------

    @on(Select.Changed, "#evals-lens-selector")
    def _on_lens_changed(self, event: Select.Changed) -> None:
        event.stop()
        if event.value == self._lens:
            return
        self._lens = event.value
        self._render_table()
        self._render_header()

    @on(Select.Changed, "#evals-baseline-selector")
    def _on_baseline_changed(self, event: Select.Changed) -> None:
        event.stop()
        mode, ref_id = event.value
        if mode == self._baseline_mode and ref_id == (
            self._baseline_target_id()
            if mode == "column"
            else self._baseline_snippet_id()
        ):
            return
        self._baseline_mode = mode
        self._baseline_index = self._index_for_baseline(mode, ref_id)
        self._render_table()
        self._render_header()

    @on(Select.Changed, "#evals-probe-selector")
    def _on_probe_changed(self, event: Select.Changed) -> None:
        event.stop()
        if event.value == self._probe_index:
            return
        self._probe_index = event.value
        self._render_table()
        self._render_header()

    def _probes(self) -> tuple[str, ...]:
        if self._grid is None:
            return ()
        return tuple(self._grid["snapshot"].get("probes") or ())

    def _active_probe(self) -> Optional[str]:
        """The one probe the Probe lens reads, or ``None`` when the bench
        configured none. Clamped, so a stale ``_probe_index`` can never
        index past a shorter probe tuple."""
        probes = self._probes()
        if not probes:
            return None
        return probes[min(self._probe_index, len(probes) - 1)]

    def action_cycle_lens(self) -> None:
        if self._grid is None:
            return
        idx = LENS_ORDER.index(self._lens)
        next_lens = LENS_ORDER[(idx + 1) % len(LENS_ORDER)]
        self.query_one("#evals-lens-selector", Select).value = next_lens

    def action_cycle_baseline(self) -> None:
        """Toggles the baseline MODE (column <-> row), resetting to the
        first row/column in the new mode. Fine-grained selection of WHICH
        column or row is the baseline is the `#evals-baseline-selector`
        dropdown's job -- this keyboard shortcut only does the coarse
        column/row switch the design spec asks for ("switchable between a
        column and a row")."""
        if self._grid is None:
            return
        new_mode: BaselineMode = "row" if self._baseline_mode == "column" else "column"
        rows = self._grid["snapshot"]["targets" if new_mode == "column" else "snippets"]
        if not rows:
            return
        self.query_one("#evals-baseline-selector", Select).value = (
            new_mode,
            rows[0]["id"],
        )

    def action_cycle_sort(self) -> None:
        if self._grid is None:
            return
        order: dict[SortMode, SortMode] = {"none": "desc", "desc": "asc", "asc": "none"}
        self._sort_mode = order[self._sort_mode]
        self._render_table()
        self._render_header()

    # -- export ------------------------------------------------------

    def action_export(self) -> None:
        """Opens a ``FileSave`` dialog offering both formats; the CHOSEN
        destination's extension picks which one gets written (see
        ``_write_export_file``) -- one dialog, one key, per the design
        spec's "Export (`e`) writes the grid as CSV for the active lens, or
        JSON for the whole run group."
        """
        if self._grid is None:
            return
        bench_name = str(self._grid["snapshot"].get("bench_name") or "run")
        safe_name = (
            "".join(c for c in bench_name if c.isalnum() or c in (" ", "-", "_")).strip()
            or "run"
        )
        filters = Filters(
            ("JSON (full run group)", lambda p: p.suffix.lower() == ".json"),
            ("CSV (active lens)", lambda p: p.suffix.lower() == ".csv"),
            ("All files", lambda p: True),
        )
        self.app.push_screen(
            FileSave(
                title="Export results grid",
                default_file=f"{safe_name}.json",
                filters=filters,
            ),
            self._write_export_file,
        )

    def _write_export_file(self, selected_path: Optional[Any]) -> None:
        """Writes the export chosen via ``action_export``'s ``FileSave``
        dialog. ``.csv`` writes the active lens; anything else (``.json``
        or an unrecognized extension) writes the full run-group JSON --
        the more complete, reproducible form is the safer default for an
        ambiguous filename. Public-shaped (not name-mangled) so a test can
        call it directly with a real temp path, bypassing the modal picker
        -- mirrors ``SnippetEditor._handle_import_file_selected``/
        ``library_screen.py``'s own export write-helpers.
        """
        if not selected_path or self._grid is None:
            return
        try:
            validated_path = validate_path_simple(selected_path, require_exists=False)
        except ValueError as exc:
            logger.warning(f"Rejected results-grid export path {selected_path!r}: {exc}")
            self._notify(f"Rejected export path: {exc}", severity="warning")
            return

        try:
            if validated_path.suffix.lower() == ".csv":
                validated_path.write_text(self._export_csv_text(), encoding="utf-8")
            else:
                payload = self._export_json_payload()
                validated_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        except OSError as exc:
            logger.opt(exception=True).warning(
                f"Could not write results-grid export to {validated_path}."
            )
            self._notify(f"Export failed: {exc}", severity="error")
            return
        self._notify(f"Exported to {validated_path}", severity="information")

    def _notify(self, message: str, *, severity: str = "information") -> None:
        """Mirrors ``snippet_editor.py``'s identical helper: routes through
        the screen's ``app_instance`` (what a test harness's fake actually
        observes), falling back to ``self.app.notify``."""
        app_instance = getattr(self.screen, "app_instance", None)
        if app_instance is not None and hasattr(app_instance, "notify"):
            app_instance.notify(message, severity=severity)
        else:
            self.app.notify(message, severity=severity)

    def _export_csv_text(self) -> str:
        """CSV for the ACTIVE lens -- built from the exact same
        ``_compute_active_lens_rows`` the on-screen ``DataTable`` renders
        from, so this can never show a different lens/baseline/sort than
        what is currently visible."""
        column_labels, snippet_rows, group_mean_rows = self._compute_active_lens_rows()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(column_labels)
        for _row_key, cells in snippet_rows:
            writer.writerow(cells)
        for _row_key, cells in group_mean_rows:
            writer.writerow(cells)
        return buffer.getvalue()

    def _export_json_payload(self) -> dict[str, Any]:
        """The whole run group: snapshot, every cell's top-K, and the
        resolved probe readings -- "what makes a run reproducible outside
        the app" per the design spec. Unlike the CSV export (one lens),
        this is lens-independent: it reads ``self._grid`` directly, the
        same snapshot/cells ``compose()`` loaded once from ``load_grid``,
        never re-querying the database.
        """
        snapshot = self._grid["snapshot"]
        cells: dict[tuple[str, str], CellCapture | CellError] = self._grid["cells"]
        probes = tuple(snapshot.get("probes") or ())
        targets = snapshot.get("targets") or []

        ever_observed_by_target = {
            target["id"]: self._ever_observed_all_probes(target["id"])
            for target in targets
        }

        cell_payloads: dict[str, dict[str, Any]] = {}
        for (sid, tid), cap_or_err in cells.items():
            key = f"{sid}|{tid}"
            if isinstance(cap_or_err, CellError):
                cell_payloads[key] = {
                    "snippet_id": sid,
                    "target_id": tid,
                    "status": "failed",
                    "reason": cap_or_err.reason,
                    "detail": cap_or_err.detail,
                }
                continue
            cap = cap_or_err
            probe_readings: dict[str, dict[str, Any]] = {}
            for probe in probes:
                reading = analysis.resolve_probe(
                    cap, probe, ever_observed=ever_observed_by_target.get(tid, {}).get(probe, False)
                )
                probe_readings[probe] = {
                    "state": reading.state,
                    "logprob": reading.logprob,
                }
            cell_payloads[key] = {
                "snippet_id": sid,
                "target_id": tid,
                "status": "captured",
                "prompt_mode": cap.prompt_mode,
                "k_requested": cap.k_requested,
                "k_returned": cap.k_returned,
                "content_offset": cap.content_offset,
                "canary": cap.canary,
                "captured_at": cap.captured_at,
                "top_k": [
                    {
                        "token": tok.token,
                        "logprob": tok.logprob,
                        "bytes": list(tok.bytes_),
                        "token_id": tok.token_id,
                    }
                    for tok in cap.top_k
                ],
                "probes": probe_readings,
            }

        return {
            "run_group_id": self._run_group_id,
            "snapshot": snapshot,
            "cells": cell_payloads,
        }

    # -- header ----------------------------------------------------------

    def _render_header(self) -> None:
        if self._grid is None:
            return
        snapshot = self._grid["snapshot"]
        cells = self._grid["cells"]
        caps = [c for c in cells.values() if isinstance(c, CellCapture)]
        failed = sum(1 for c in cells.values() if isinstance(c, CellError))
        effective_k = analysis.effective_k(caps)

        meta = (
            f"{snapshot.get('bench_name') or 'Untitled bench'} · "
            f"{snapshot.get('prompt_mode') or '?'} · K {effective_k} · "
            f"{len(caps)} cells · {failed} failed"
        )
        self.query_one("#evals-grid-meta", Static).update(meta)

        sort_label = {"none": "dataset order", "desc": "spread ▼", "asc": "spread ▲"}[
            self._sort_mode
        ]
        state = (
            f"Lens: {self._lens_description()}   "
            f"Baseline: {self._baseline_description()}   "
            f"Sort: {sort_label}"
        )
        self.query_one("#evals-grid-state", Static).update(state)

    # -- table rendering ---------------------------------------------

    def _render_table(self) -> None:
        if self._grid is None:
            return
        table = self.query_one("#evals-grid-table", DataTable)
        table.clear(columns=True)

        targets = self._grid["snapshot"].get("targets") or []
        column_labels, snippet_rows, group_mean_rows = self._compute_active_lens_rows()

        table.add_column(_safe_cell(column_labels[0]), key="__snippet__")
        for index, target in enumerate(targets):
            table.add_column(_safe_cell(column_labels[1 + index]), key=target["id"])
        if self._lens == "delta":
            table.add_column(_safe_cell(column_labels[-1]), key="__spread__")

        for row_key, cell_texts in snippet_rows:
            table.add_row(*(_safe_cell(c) for c in cell_texts), key=row_key)
        for row_key, cell_texts in group_mean_rows:
            table.add_row(*(_safe_cell(c) for c in cell_texts), key=row_key)

    def _compute_active_lens_rows(
        self,
    ) -> tuple[list[str], list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
        """The single source of truth for "what the active lens/baseline/
        sort renders" -- shared by the on-screen ``DataTable``
        (``_render_table``, above) and CSV export (``_export_csv_rows``,
        below) so the two can never disagree about what "the active lens"
        means. Every value here is plain text (no ``rich.text.Text``
        wrapping) -- ``_render_table`` wraps it via ``_safe_cell`` for the
        ``DataTable``; the CSV writer uses it as-is.

        Returns:
            ``(column_labels, snippet_rows, group_mean_rows)`` --
            ``column_labels`` is ``["Snippet", <target label>, ...]``, with
            a trailing ``"Spread"`` only for the Δ lens. ``snippet_rows``
            and ``group_mean_rows`` are ``[(row_key, [cell_text, ...]),
            ...]`` in display (post-sort) order; ``group_mean_rows`` is
            empty outside the Δ lens or when no snippet in this run carries
            a ``group``.
        """
        snapshot = self._grid["snapshot"]
        cells: dict[tuple[str, str], CellCapture | CellError] = self._grid["cells"]
        preflight: dict[str, PreflightResult] = self._grid.get("preflight") or {}
        targets = snapshot.get("targets") or []
        snippets = snapshot.get("snippets") or []

        show_delta_extras = self._lens == "delta"
        baseline_target_id = (
            self._baseline_target_id() if self._baseline_mode == "column" else None
        )

        column_labels: list[str] = ["Snippet"]
        for target in targets:
            label = target["name"]
            result = preflight.get(target["id"])
            if result is not None and result.is_warned:
                label = f"{label} [warned]"
            if show_delta_extras and target["id"] == baseline_target_id:
                label = f"{label} · baseline"
            column_labels.append(label)
        if show_delta_extras:
            column_labels.append("Spread")

        caps_by_capture = {
            key: cap for key, cap in cells.items() if isinstance(cap, CellCapture)
        }
        effective_k = analysis.effective_k(list(caps_by_capture.values()))
        active_probe = self._active_probe()
        ever_observed_active_probe = self._ever_observed_active_probe(
            targets, snippets, cells, active_probe
        )

        row_order = list(snippets)
        if self._sort_mode != "none":
            row_order = self._sorted_rows(row_order, cells, targets)

        #: {target_id: [(group, divergence_value), ...]} -- only populated
        #: for the Δ lens, and only with rows that produced a real number
        #: (never the literal "baseline" cells or an unavailable
        #: comparison), so ``analysis.group_means`` below sees exactly the
        #: same values the grid itself rendered.
        column_group_rows: dict[str, list[tuple[Optional[str], float]]] = {
            target["id"]: [] for target in targets
        }

        snippet_rows: list[tuple[str, list[str]]] = []
        for snippet in row_order:
            sid = snippet["id"]
            label = snippet["text"]
            if snippet.get("group"):
                label = f"{label} [{snippet['group']}]"
            row: list[str] = [label]
            for target in targets:
                tid = target["id"]
                cap_or_err = cells.get((sid, tid))
                text, divergence_value = self._render_cell(
                    lens=self._lens,
                    sid=sid,
                    tid=tid,
                    cap_or_err=cap_or_err,
                    cells=cells,
                    effective_k=effective_k,
                    probe=active_probe,
                    ever_observed_active_probe=ever_observed_active_probe,
                )
                row.append(text)
                if show_delta_extras and divergence_value is not None:
                    column_group_rows[tid].append((snippet.get("group"), divergence_value))
            if show_delta_extras:
                row_caps = [
                    caps_by_capture.get((sid, target["id"])) for target in targets
                ]
                valid = [c for c in row_caps if c is not None]
                row.append(f"{analysis.spread(valid):.2f}" if len(valid) >= 2 else "")
            snippet_rows.append((sid, row))

        group_mean_rows: list[tuple[str, list[str]]] = []
        if show_delta_extras:
            group_mean_rows = self._group_mean_rows(targets, column_group_rows)

        return column_labels, snippet_rows, group_mean_rows

    def _render_cell(
        self,
        *,
        lens: LensKey,
        sid: str,
        tid: str,
        cap_or_err: CellCapture | CellError | None,
        cells: dict[tuple[str, str], CellCapture | CellError],
        effective_k: int,
        probe: Optional[str],
        ever_observed_active_probe: dict[str, bool],
    ) -> tuple[str, Optional[float]]:
        """Renders one cell for the active lens.

        Returns ``(display_text, divergence_value)`` -- ``divergence_value``
        is only ever non-``None`` for the Δ lens's REAL comparison cells
        (never a "baseline" literal or an unavailable comparison), so the
        caller can feed exactly those into ``analysis.group_means`` without
        re-deriving which cells were real comparisons.
        """
        if lens == "delta":
            return self._render_delta(sid=sid, tid=tid, cap_or_err=cap_or_err, cells=cells)

        if cap_or_err is None:
            return "", None  # unrun -- never "0"
        if isinstance(cap_or_err, CellError):
            return FAILED_MARK, None  # failed -- never "0"
        cap = cap_or_err

        if lens == "top1":
            return self._render_top1(cap), None
        if lens == "entropy":
            return f"{analysis.entropy(cap, k=effective_k):.2f}", None
        if lens == "coverage":
            # analysis.truncated_mass at the SHARED effective K, not
            # CellCapture.truncated_mass's own native-K figure -- the
            # header states this K, and two cells with identical
            # behaviour at it must not read as a 2x difference purely
            # from their requested K (see analysis.truncated_mass).
            return (
                f"{analysis.truncated_mass(cap, k=effective_k) * 100:.0f}%",
                None,
            )
        if lens == "probe":
            if probe is None:
                return "n/a", None
            # `reading.matched` carries the TokenProb resolve_probe itself
            # matched -- never re-derived here (see render_probe_reading).
            reading = analysis.resolve_probe(
                cap, probe, ever_observed=ever_observed_active_probe.get(tid, False)
            )
            return render_probe_reading(reading), None
        return "", None  # pragma: no cover -- exhaustive over LensKey

    def _render_top1(self, cap: CellCapture) -> str:
        top = cap.top_k
        if not top:
            return FAILED_MARK
        first = top[0]
        # The near-tie DECISION is the engine's methodology, not the
        # view's -- ``analysis.near_tie`` owns the threshold
        # (``analysis.NEAR_TIE_LOGPROB_GAP_NATS``) so it lives in one place
        # alongside ``TRUNCATION_WARN_THRESHOLD`` and ``divergence``'s own
        # ``is_bounded``, rather than this module recomputing a raw logprob
        # gap and re-deciding "too close to call" on its own.
        if len(top) > 1 and analysis.near_tie(cap):
            second = top[1]
            return (
                f"{render_token(first.token)}≈{render_token(second.token)}  "
                f"{first.prob * 100:.0f}/{second.prob * 100:.0f}%"
            )
        return f"{render_token(first.token)}  {first.prob * 100:.0f}%"

    def _delta_reading(
        self,
        *,
        sid: str,
        tid: str,
        cap_or_err: CellCapture | CellError | None,
        cells: dict[tuple[str, str], CellCapture | CellError],
    ) -> "_DeltaReading":
        """The single source of truth for one Δ-baseline cell -- both its
        grid text AND the extra context the inspector needs to explain a
        ``!`` marker (the *combined* truncated mass that triggered it, not
        just this cell's own). ``_render_cell`` and ``_on_cell_highlighted``
        both call this rather than each computing (and risking disagreeing
        about) the comparison independently.
        """
        if self._baseline_mode == "column":
            baseline_id = self._baseline_target_id()
            is_baseline_position = tid == baseline_id
            baseline_cell = cells.get((sid, baseline_id)) if baseline_id else None
        else:
            baseline_id = self._baseline_snippet_id()
            is_baseline_position = sid == baseline_id
            baseline_cell = cells.get((baseline_id, tid)) if baseline_id else None

        if is_baseline_position:
            # The design spec's own mockup renders the baseline's own
            # position as the literal word "baseline", never a number --
            # comparing a cell to itself is not a finding.
            if cap_or_err is None:
                return _DeltaReading(text="")
            if isinstance(cap_or_err, CellError):
                return _DeltaReading(text=FAILED_MARK)
            return _DeltaReading(text="baseline")

        # "When the baseline cell itself failed, the whole comparison is
        # unavailable for that row or column and renders as such, never as
        # zero" (design spec). Unrun vs. failed baseline get the SAME
        # unrun/failed treatment every other cell gets, for the same
        # reason: neither reads as "measured and found nothing".
        if baseline_cell is None:
            return _DeltaReading(text="")
        if isinstance(baseline_cell, CellError):
            return _DeltaReading(text=FAILED_MARK)

        if cap_or_err is None:
            return _DeltaReading(text="")
        if isinstance(cap_or_err, CellError):
            return _DeltaReading(text=FAILED_MARK)

        jsd, is_bounded = analysis.divergence(cap_or_err, baseline_cell)
        combined = analysis.combined_truncation(cap_or_err, baseline_cell)
        text = f"{jsd:.2f}"
        if is_bounded:
            text += " !"
        return _DeltaReading(
            text=text,
            is_real_comparison=True,
            jsd=jsd,
            is_bounded=is_bounded,
            combined_truncated_mass=combined,
        )

    def _render_delta(
        self,
        *,
        sid: str,
        tid: str,
        cap_or_err: CellCapture | CellError | None,
        cells: dict[tuple[str, str], CellCapture | CellError],
    ) -> tuple[str, Optional[float]]:
        reading = self._delta_reading(sid=sid, tid=tid, cap_or_err=cap_or_err, cells=cells)
        return reading.text, (reading.jsd if reading.is_real_comparison else None)

    def _group_mean_rows(
        self,
        targets: list[dict[str, Any]],
        column_group_rows: dict[str, list[tuple[Optional[str], float]]],
    ) -> list[tuple[str, list[str]]]:
        groups_in_order: list[str] = []
        seen: set[str] = set()
        for rows in column_group_rows.values():
            for group, _ in rows:
                if group is not None and group not in seen:
                    seen.add(group)
                    groups_in_order.append(group)
        if not groups_in_order:
            return []

        means_by_target = {
            tid: analysis.group_means(rows) for tid, rows in column_group_rows.items()
        }
        result: list[tuple[str, list[str]]] = []
        for group in groups_in_order:
            row: list[str] = [f"group mean [{group}]"]
            for target in targets:
                means = means_by_target.get(target["id"], {})
                value = means.get(group)
                row.append(f"{value:.2f}" if value is not None else "")
            row.append("")  # Spread column: not meaningful for a summary row.
            result.append((f"__group_mean__{group}", row))
        return result

    def _sorted_rows(
        self,
        snippets: list[dict[str, Any]],
        cells: dict[tuple[str, str], CellCapture | CellError],
        targets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sorts by ``analysis.spread`` -- "where targets disagree most" --
        independent of the active lens/baseline, so `s` never needs an
        invented per-lens aggregate (averaging Top-1 tokens or Probe
        readings across a row is not a number the engine computes, and this
        module adds no arithmetic beyond formatting). Rows with fewer than
        two successfully captured cells have no defined spread and sort
        last in descending order.
        """

        def sort_key(snippet: dict[str, Any]) -> float:
            sid = snippet["id"]
            caps = [
                cells.get((sid, target["id"]))
                for target in targets
            ]
            valid = [c for c in caps if isinstance(c, CellCapture)]
            if len(valid) < 2:
                return -1.0
            return analysis.spread(valid)

        return sorted(snippets, key=sort_key, reverse=(self._sort_mode == "desc"))

    def _ever_observed_active_probe(
        self,
        targets: list[dict[str, Any]],
        snippets: list[dict[str, Any]],
        cells: dict[tuple[str, str], CellCapture | CellError],
        probe: Optional[str],
    ) -> dict[str, bool]:
        """``{target_id: bool}`` for whether the Probe lens's ACTIVE probe
        (``_active_probe``) was EVER observed in that target's top-K across
        the whole run -- computed once per table render (not once per cell)
        and threaded into every ``analysis.resolve_probe`` call for the
        Probe lens, per its own ``ever_observed`` contract."""
        if probe is None:
            return {}
        result: dict[str, bool] = {}
        for target in targets:
            tid = target["id"]
            observed = False
            for snippet in snippets:
                cap = cells.get((snippet["id"], tid))
                if isinstance(cap, CellCapture) and any(
                    tok.token == probe for tok in cap.top_k
                ):
                    observed = True
                    break
            result[tid] = observed
        return result

    def _ever_observed_all_probes(self, target_id: str) -> dict[str, bool]:
        """Same computation as ``_ever_observed_first_probe``, but for
        EVERY configured probe and one target -- used only when a cell is
        focused (``_on_cell_highlighted``), for the inspector's full probe
        table. Kept separate from the render-time helper above so a table
        render (which only needs the lens's single active probe) does not
        pay for every probe on every render."""
        snapshot = self._grid["snapshot"] if self._grid else {}
        probes = tuple(snapshot.get("probes") or ())
        snippets = snapshot.get("snippets") or []
        cells = self._grid["cells"] if self._grid else {}
        result: dict[str, bool] = {}
        for probe in probes:
            observed = False
            for snippet in snippets:
                cap = cells.get((snippet["id"], target_id))
                if isinstance(cap, CellCapture) and any(
                    tok.token == probe for tok in cap.top_k
                ):
                    observed = True
                    break
            result[probe] = observed
        return result

    # -- focus -> inspector ----------------------------------------------

    @on(DataTable.CellHighlighted)
    def _on_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        event.stop()
        if self._grid is None:
            return
        row_key = event.cell_key.row_key.value
        column_key = event.cell_key.column_key.value
        if not isinstance(row_key, str) or not isinstance(column_key, str):
            return
        if row_key.startswith("__group_mean__") or column_key in (
            "__snippet__",
            "__spread__",
        ):
            return  # summary row / non-data column: no cell detail to show

        snapshot = self._grid["snapshot"]
        snippet = next(
            (s for s in snapshot["snippets"] if s["id"] == row_key), None
        )
        target = next(
            (t for t in snapshot["targets"] if t["id"] == column_key), None
        )
        if snippet is None or target is None:
            return

        cells = self._grid["cells"]
        cell = cells.get((row_key, column_key))
        delta_reading: Optional[_DeltaReading] = None
        if self._lens == "delta":
            reading = self._delta_reading(
                sid=row_key, tid=column_key, cap_or_err=cell, cells=cells
            )
            if reading.is_real_comparison:
                delta_reading = reading

        self.post_message(
            self.CellFocused(
                snippet_id=row_key,
                target_id=column_key,
                snippet_text=snippet["text"],
                target_name=target["name"],
                cell=cell,
                probes=tuple(snapshot.get("probes") or ()),
                ever_observed=self._ever_observed_all_probes(column_key),
                delta=delta_reading,
            )
        )
