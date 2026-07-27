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

Five lenses decide what a cell renders: Top-1, Entropy, Probe, Coverage
(``truncated_mass``), and Δ baseline. Three of them would misrepresent the
engine if rendered naively, and each is pinned by a test in
``Tests/UI/test_evals_results_grid.py``:

1. **A bare Top-1 winner on a near-tie.** Two identical requests to the same
   server, seconds apart at the same neutral sampler, returned the top two
   tokens in OPPOSITE rank order, magnitudes stable to ~0.002 nats (see
   ``Tests/Evals/word_bench/test_normalizer.py::
   test_a_near_tie_between_the_top_two_is_visible_in_the_fixture``, which
   pins the same live-captured fixture). A grid that renders a bare winner
   there shows a spurious difference between cells that are statistically
   identical. ``_render_top1`` marks the tie instead -- see
   ``NEAR_TIE_LOGPROB_GAP_NATS`` below for the threshold and its rationale.
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

Export (`e`) is Task 2's job. The key is deliberately left unbound here so
it stays free.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import DataTable, Select, Static

from ...Evals.word_bench import analysis
from ...Evals.word_bench.models import CellCapture, CellError, PreflightResult, TokenProb
from ...Evals.word_bench.storage import load_grid
from .evals_state import EvalsViewModel

LensKey = Literal["top1", "entropy", "probe", "coverage", "delta"]
BaselineMode = Literal["column", "row"]
SortMode = Literal["none", "desc", "asc"]

LENS_LABELS: dict[LensKey, str] = {
    "top1": "Top-1",
    "entropy": "Entropy",
    "probe": "Probe",
    "coverage": "Coverage",
    "delta": "Δ baseline",
}
#: Cycle order for the `l` key -- also the ``#evals-lens-selector`` option
#: order, so the keyboard shortcut and the dropdown never disagree about
#: "next".
LENS_ORDER: tuple[LensKey, ...] = ("top1", "entropy", "probe", "coverage", "delta")

#: Rank 1 and rank 2 are flagged as a near-tie when their logprob GAP is
#: below this many nats. Chosen from an observed instability, not derived:
#: two identical requests to the same server, seconds apart, at the same
#: neutral sampler settings, returned the top two tokens in OPPOSITE rank
#: order while each token's own logprob held stable to ~0.002 nats
#: (-0.698/-0.794 one call, -0.697/-0.792 the next). The committed fixture
#: this was captured from carries a ~0.095-0.096 nat gap between those two
#: tokens, and this codebase already has one considered judgment call about
#: where "near-tie" starts for that exact fixture:
#: ``Tests/Evals/word_bench/test_normalizer.py::
#: test_a_near_tie_between_the_top_two_is_visible_in_the_fixture`` asserts
#: ``abs(gap) < 0.15`` as the boundary for calling it a near-tie. 0.15 nats
#: is reused here for the same phenomenon: it comfortably covers the
#: observed ~0.095-0.096 nat gap that already produced a rank flip, while
#: sitting roughly two orders of magnitude above the ~0.002 nat run-to-run
#: noise floor, so it will not fire on ordinary sampling jitter far from a
#: real tie.
NEAR_TIE_LOGPROB_GAP_NATS = 0.15

#: Never "0" -- a failed or unrun cell must not read as "measured and found
#: nothing". Unrun cells render as a plain empty string instead (see
#: ``_render_cell``); this mark is only for a cell that was measured and
#: came back an error.
FAILED_MARK = "—"


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


def render_probe_reading(
    reading: "analysis.ProbeReading", matched: Optional[TokenProb]
) -> str:
    """Format one probe reading, shared between the Probe lens
    (``_render_cell``) and the focused-cell inspector
    (``inspector.EvalsCellInspector``) so the two never disagree about what
    "observed" / "bounded" / "never observed" look like.

    ``matched`` is the ``TokenProb`` ``resolve_probe`` matched inside the
    cell's own ``top_k`` (``None`` unless ``reading.state == "observed"``),
    passed in rather than re-derived here so this function needs no access
    to the cell itself. Its ``.prob`` property is the engine's own
    logprob-to-probability conversion (``models.TokenProb.prob``) -- reused
    for display exactly as the engine defines it, rather than this module
    calling ``math.exp`` on a bare logprob itself.
    """
    if reading.state == "never_observed":
        return "never observed"
    if reading.state == "bounded":
        if reading.logprob is None:
            return FAILED_MARK
        return f"< {reading.logprob:.2f}"
    # observed
    if matched is not None:
        return f"{reading.logprob:.2f}  {matched.prob * 100:.1f}%"
    return f"{reading.logprob:.2f}"


class ResultsGrid(Vertical):
    """Detail-pane content for a selected run group: the pivoted results
    grid, its lens/baseline controls, and the header stating the effective
    K, cell/failure counts, and which lens/baseline/sort is active."""

    BINDINGS = [
        Binding("l", "cycle_lens", "lens", show=False),
        Binding("b", "cycle_baseline", "baseline", show=False),
        Binding("s", "cycle_sort", "sort", show=False),
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
        ) -> None:
            self.snippet_id = snippet_id
            self.target_id = target_id
            self.snippet_text = snippet_text
            self.target_name = target_name
            self.cell = cell
            self.probes = probes
            self.ever_observed = ever_observed
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
            yield Static(
                "This run has no snippets or no targets to render.",
                id="evals-grid-empty",
            )
            return

        yield Static("", id="evals-grid-meta")
        yield Static("", id="evals-grid-state")
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
        yield DataTable(id="evals-grid-table", cursor_type="cell", zebra_stripes=True)

    def on_mount(self) -> None:
        if self._grid is None:
            return
        self._render_table()
        self._render_header()

    # -- baseline option plumbing -------------------------------------

    def _baseline_options(self) -> list[tuple[str, tuple[str, str]]]:
        snapshot = self._grid["snapshot"]
        options: list[tuple[str, tuple[str, str]]] = []
        for target in snapshot["targets"]:
            options.append((f"Column · {target['name']}", ("column", target["id"])))
        for snippet in snapshot["snippets"]:
            text = snippet["text"]
            label = text if len(text) <= 28 else f"{text[:27]}…"
            options.append((f"Row · {label}", ("row", snippet["id"])))
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

    def action_cycle_lens(self) -> None:
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
        new_mode: BaselineMode = "row" if self._baseline_mode == "column" else "column"
        rows = self._grid["snapshot"]["targets" if new_mode == "column" else "snippets"]
        if not rows:
            return
        self.query_one("#evals-baseline-selector", Select).value = (
            new_mode,
            rows[0]["id"],
        )

    def action_cycle_sort(self) -> None:
        order: dict[SortMode, SortMode] = {"none": "desc", "desc": "asc", "asc": "none"}
        self._sort_mode = order[self._sort_mode]
        self._render_table()
        self._render_header()

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
            f"Lens: {LENS_LABELS[self._lens]}   "
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

        snapshot = self._grid["snapshot"]
        cells: dict[tuple[str, str], CellCapture | CellError] = self._grid["cells"]
        preflight: dict[str, PreflightResult] = self._grid.get("preflight") or {}
        targets = snapshot.get("targets") or []
        snippets = snapshot.get("snippets") or []
        probes = tuple(snapshot.get("probes") or ())

        show_delta_extras = self._lens == "delta"
        baseline_target_id = (
            self._baseline_target_id() if self._baseline_mode == "column" else None
        )

        table.add_column(_safe_cell("Snippet"), key="__snippet__")
        for target in targets:
            label = target["name"]
            result = preflight.get(target["id"])
            if result is not None and result.is_warned:
                label = f"{label} [warned]"
            if show_delta_extras and target["id"] == baseline_target_id:
                label = f"{label} · baseline"
            table.add_column(_safe_cell(label), key=target["id"])
        if show_delta_extras:
            table.add_column(_safe_cell("Spread"), key="__spread__")

        caps_by_capture = {
            key: cap for key, cap in cells.items() if isinstance(cap, CellCapture)
        }
        effective_k = analysis.effective_k(list(caps_by_capture.values()))
        ever_observed_first_probe = self._ever_observed_first_probe(
            targets, snippets, cells, probes
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
                    probes=probes,
                    ever_observed_first_probe=ever_observed_first_probe,
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
            table.add_row(*(_safe_cell(cell) for cell in row), key=sid)

        if show_delta_extras:
            self._add_group_mean_rows(table, targets, column_group_rows)

    def _render_cell(
        self,
        *,
        lens: LensKey,
        sid: str,
        tid: str,
        cap_or_err: CellCapture | CellError | None,
        cells: dict[tuple[str, str], CellCapture | CellError],
        effective_k: int,
        probes: tuple[str, ...],
        ever_observed_first_probe: dict[str, bool],
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
            return f"{cap.truncated_mass * 100:.0f}%", None
        if lens == "probe":
            if not probes:
                return "n/a", None
            probe = probes[0]
            reading = analysis.resolve_probe(
                cap, probe, ever_observed=ever_observed_first_probe.get(tid, False)
            )
            matched = (
                next((t for t in cap.top_k if t.token == probe), None)
                if reading.state == "observed"
                else None
            )
            return render_probe_reading(reading, matched), None
        return "", None  # pragma: no cover -- exhaustive over LensKey

    def _render_top1(self, cap: CellCapture) -> str:
        top = cap.top_k
        if not top:
            return FAILED_MARK
        first = top[0]
        if len(top) > 1:
            second = top[1]
            gap = first.logprob - second.logprob
            if abs(gap) < NEAR_TIE_LOGPROB_GAP_NATS:
                return (
                    f"{render_token(first.token)}≈{render_token(second.token)}  "
                    f"{first.prob * 100:.0f}/{second.prob * 100:.0f}%"
                )
        return f"{render_token(first.token)}  {first.prob * 100:.0f}%"

    def _render_delta(
        self,
        *,
        sid: str,
        tid: str,
        cap_or_err: CellCapture | CellError | None,
        cells: dict[tuple[str, str], CellCapture | CellError],
    ) -> tuple[str, Optional[float]]:
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
                return "", None
            if isinstance(cap_or_err, CellError):
                return FAILED_MARK, None
            return "baseline", None

        # "When the baseline cell itself failed, the whole comparison is
        # unavailable for that row or column and renders as such, never as
        # zero" (design spec). Unrun vs. failed baseline get the SAME
        # unrun/failed treatment every other cell gets, for the same
        # reason: neither reads as "measured and found nothing".
        if baseline_cell is None:
            return "", None
        if isinstance(baseline_cell, CellError):
            return FAILED_MARK, None

        if cap_or_err is None:
            return "", None
        if isinstance(cap_or_err, CellError):
            return FAILED_MARK, None

        jsd, is_bounded = analysis.divergence(cap_or_err, baseline_cell)
        text = f"{jsd:.2f}"
        if is_bounded:
            text += " !"
        return text, jsd

    def _add_group_mean_rows(
        self,
        table: DataTable,
        targets: list[dict[str, Any]],
        column_group_rows: dict[str, list[tuple[Optional[str], float]]],
    ) -> None:
        groups_in_order: list[str] = []
        seen: set[str] = set()
        for rows in column_group_rows.values():
            for group, _ in rows:
                if group is not None and group not in seen:
                    seen.add(group)
                    groups_in_order.append(group)
        if not groups_in_order:
            return

        means_by_target = {
            tid: analysis.group_means(rows) for tid, rows in column_group_rows.items()
        }
        for group in groups_in_order:
            row: list[str] = [f"group mean [{group}]"]
            for target in targets:
                means = means_by_target.get(target["id"], {})
                value = means.get(group)
                row.append(f"{value:.2f}" if value is not None else "")
            row.append("")  # Spread column: not meaningful for a summary row.
            table.add_row(*(_safe_cell(cell) for cell in row), key=f"__group_mean__{group}")

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

    def _ever_observed_first_probe(
        self,
        targets: list[dict[str, Any]],
        snippets: list[dict[str, Any]],
        cells: dict[tuple[str, str], CellCapture | CellError],
        probes: tuple[str, ...],
    ) -> dict[str, bool]:
        """``{target_id: bool}`` for whether ``probes[0]`` was EVER observed
        in that target's top-K across the whole run -- computed once per
        table render (not once per cell) and threaded into every
        ``analysis.resolve_probe`` call for the Probe lens, per its own
        ``ever_observed`` contract."""
        if not probes:
            return {}
        probe = probes[0]
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

        self.post_message(
            self.CellFocused(
                snippet_id=row_key,
                target_id=column_key,
                snippet_text=snippet["text"],
                target_name=target["name"],
                cell=self._grid["cells"].get((row_key, column_key)),
                probes=tuple(snapshot.get("probes") or ()),
                ever_observed=self._ever_observed_all_probes(column_key),
            )
        )
