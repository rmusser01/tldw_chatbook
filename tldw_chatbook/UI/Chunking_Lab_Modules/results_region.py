"""Precision Workbench results: captured evidence, bounded native inspection.

Operate mode; inherit Chatbook theme, focus and native text selection. A/B tables
share equal space at wide sizes; narrow Compare shows the inspected candidate.
The signature interaction links only verified source spans, explaining missing
alignment directly beside the inspector. The coordinator owns persistence and
current/Previous choice; this region never executes recipes or loads DB records.
"""

from __future__ import annotations

import asyncio
import json

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.document._document import Selection
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, DataTable, Select, Static, TextArea

from tldw_chatbook.Chunking.lab_comparison import (
    chunk_mapping,
    comparison_deltas,
    comparison_reason,
    diff_configs,
    linked_chunks,
    summarize_result,
)
from tldw_chatbook.Chunking.lab_models import RunResult

PAGE_SIZE = 100
TEXT_PAGE_SIZE = 8192
DETAILS = (
    ("Chunk text", "chunk"),
    ("Source / transformed", "source"),
    ("Statistics", "statistics"),
    ("Effective config diff", "effective"),
    ("Authored config diff", "authored"),
    ("Execution / metadata", "execution"),
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _prepare(
    a: RunResult | None, b: RunResult | None, comparison_block: str | None = None
) -> dict:
    summaries = {
        side: summarize_result(result)
        for side, result in (("A", a), ("B", b))
        if result
    }
    reason = comparison_block or (
        comparison_reason(a, b)
        if a and b
        else "Comparison requires two successful results. Run both to compare."
    )
    stats = []
    for side, summary in summaries.items():
        if summary["status"] != "completed":
            stats.append(
                f"{side} · {summary['status']}; no output statistics available."
            )
            continue
        lines = [
            f"{side} · {summary['chunk_count']:,} chunks",
            "             Minimum  Median  p95  Maximum  Total",
        ]
        for key, label in (("characters", "Characters"), ("words", "Words")):
            values = summary[key]
            line = "  ".join(
                "unavailable" if value is None else str(value)
                for value in values.values()
            )
            lines.append(f"{label}: {line}")
        budget = summary["budget"]
        lines.extend(
            [
                f"Method budget: {budget['limit']} {budget['unit'] or '(unit unavailable)'} · {budget['method']}",
                f"Oversized chunks: {budget['oversized_chunks'] if budget['oversized_chunks'] is not None else 'unavailable'}",
                "Tokens unavailable: no local measurement tokenizer supplied.",
                f"Expansion (emitted/source characters): {summary['expansion_ratio'] if summary['expansion_ratio'] is not None else 'unavailable'}",
                f"Verified source overlap (characters): {summary['overlap_characters'] if summary['overlap_characters'] is not None else 'unavailable'}",
                f"Elapsed: {summary['elapsed_ms_observation']:g} ms · one observation, not a benchmark ranking.",
            ]
        )
        stats.append("\n".join(lines))
    stats.append(
        "Characters count Unicode code points, len(text). Words use len(text.split()).\n"
        "p95 uses nearest rank, ceil(0.95 × count). Expansion is not measured overlap.\n"
        "These measurements describe output shape, not retrieval quality."
    )
    if reason is None:
        stats.append(
            "B minus A common counts\n"
            + _json(comparison_deltas(summaries["A"], summaries["B"]))
        )
    else:
        stats.append(reason)
    documents = {"statistics": "\n\n".join(stats)}
    runtime_text = []
    for side, result in (("A", a), ("B", b)):
        if result:
            runtime = result.request.recipe.runtime
            runtime_text.append(
                f"{side} runtime: {runtime.backend} · engine {runtime.engine_version} · execution {runtime.execution_version}\n"
                + "Local assets: "
                + _json(runtime.assets)
            )
    for authored, name in ((False, "effective"), (True, "authored")):
        if a and b:
            diffs = diff_configs(a, b, authored=authored)
            documents[name] = (
                "Captured snapshots; operation arrays compared by position.\n"
                + "\n\n".join(runtime_text)
                + "\n\n"
                + (
                    "\n\n".join(
                        f"{d['kind']} {d['path']}\nA: {_json(d['A'])}\nB: {_json(d['B'])}"
                        for d in diffs
                    )
                    if diffs
                    else "No configuration differences."
                )
            )
        else:
            result = a or b
            documents[name] = (
                (
                    result.request.recipe.authored_json
                    if authored
                    else result.request.recipe.effective_json
                )
                if result
                else "Run a preview to inspect captured configuration."
            )
    return {"summaries": summaries, "reason": reason, "documents": documents}


class ResultsRegion(Widget):
    """Paged captured output with selection/rerun messages for the coordinator."""

    BUNDLED_CSS = """
    ResultsRegion { height: 1fr; min-height: 18; background: $background; }
    ResultsRegion .results-toolbar { height: 1; }
    ResultsRegion Button { min-width: 6; width: auto; margin-right: 1; }
    ResultsRegion .results-status { height: 2; padding: 0 1; color: $text; }
    ResultsRegion #comparison-status { height: 2; padding: 0 1; }
    ResultsRegion #result-tables { height: 1fr; min-height: 9; }
    ResultsRegion .result-column { width: 1fr; height: 1fr; min-width: 0; }
    ResultsRegion .result-column:focus-within { background: $panel; }
    ResultsRegion DataTable { height: 1fr; min-height: 1; }
    ResultsRegion .page-controls { height: 1; }
    ResultsRegion .page-label { width: 1fr; height: 1; content-align: center middle; }
    ResultsRegion #detail-controls { height: 3; }
    ResultsRegion #detail-slot { width: 1fr; height: 3; min-width: 0; }
    ResultsRegion #detail-kind { width: 100%; }
    ResultsRegion #mapping-status { height: 2; padding: 0 1; color: $text; }
    ResultsRegion #chunk-inspector { height: 1fr; min-height: 3; border: round $surface-lighten-1; }
    ResultsRegion #chunk-inspector:focus { border: round $accent; }
    ResultsRegion #text-page { width: 18; height: 3; content-align: center middle; }
    """

    class SelectionChanged(Message):
        """Small results view patch; merge under session.view['results']."""

        def __init__(
            self,
            candidate_id: str | None,
            chunk_index: int,
            active_view: str,
            view: dict,
        ):
            super().__init__()
            self.candidate_id = candidate_id
            self.chunk_index = chunk_index
            self.active_view = active_view
            self.view = view

    class RerunRequested(Message):
        """User requests that the coordinator capture and run both candidates."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results = {"A": None, "B": None}
        self._prepared = None
        self._stale_ids = frozenset()
        self._previous_ids = frozenset()
        self._previous_sides = None
        self._comparison_block = None
        self._view = {"active_view": "B", "selections": {}, "detail": "chunk"}
        self._pages = {"A": 0, "B": 0}
        self._linked = {"A": (), "B": ()}
        self._generation = 0
        self._inspection = 0
        self._document = "Run preview to inspect captured chunks."
        self._text_page = 0
        self._highlight = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="results-toolbar"):
            yield Button("A", id="view-a")
            yield Button("B", id="view-b")
            yield Button("Compare", id="view-compare")
            yield Button("Run both", id="rerun-both")
        yield Static(
            "Run preview to inspect captured chunks.",
            id="comparison-status",
            markup=False,
        )
        with Horizontal(id="result-tables"):
            for side in ("a", "b"):
                with Vertical(id=f"column-{side}", classes="result-column"):
                    yield Static(
                        "No result",
                        id=f"status-{side}",
                        classes="results-status",
                        markup=False,
                    )
                    yield DataTable(
                        id=f"chunks-{side}", cursor_type="row", zebra_stripes=True
                    )
                    with Horizontal(classes="page-controls"):
                        yield Button("First", id=f"first-{side}")
                        yield Button("Back", id=f"back-{side}")
                        yield Static(
                            "", id=f"page-{side}", classes="page-label", markup=False
                        )
                        yield Button("Next", id=f"next-{side}")
                        yield Button("Last", id=f"last-{side}")
        with Horizontal(id="detail-controls"):
            with Vertical(id="detail-slot"):
                yield Select(
                    DETAILS,
                    value=self._view["detail"],
                    allow_blank=False,
                    id="detail-kind",
                )
            yield Button("Text back", id="text-back")
            yield Static("", id="text-page", markup=False)
            yield Button("Text next", id="text-next")
        yield Static("", id="mapping-status", markup=False)
        yield TextArea(read_only=True, id="chunk-inspector", show_line_numbers=True)

    def on_mount(self) -> None:
        for table in self.query(DataTable):
            table.add_columns("Link", "#", "Chars", "Words", "Preview")
        self._refresh_results()
        self._load()

    def configure_view(
        self,
        view: dict,
        *,
        previous_ids: frozenset[str] = frozenset(),
        previous_sides: frozenset[str] | None = None,
    ) -> None:
        """Restore session.view['results'] without emitting a user edit.

        Results keys: active_view (A/B/Compare), selections ({candidate_id: index}),
        inspected_candidate (optional ID), detail (one of DETAILS). Run IDs in
        previous_ids describe explicit retained-result choices, not staleness.
        """
        saved = view.get("results", {})
        saved = saved if isinstance(saved, dict) else {}
        selections = saved.get("selections", {})
        selections = selections if isinstance(selections, dict) else {}
        self._view = {
            **saved,
            "active_view": saved.get("active_view")
            if saved.get("active_view") in ("A", "B", "Compare")
            else "B",
            "selections": {
                k: v
                for k, v in selections.items()
                if isinstance(k, str) and type(v) is int and v >= 0
            },
            "detail": saved.get("detail")
            if saved.get("detail") in tuple(value for _, value in DETAILS)
            else "chunk",
            "inspected_candidate": saved.get("inspected_candidate"),
        }
        self._previous_ids = previous_ids
        self._previous_sides = previous_sides
        if self.is_mounted:
            self.query_one("#detail-kind", Select).value = self._view["detail"]
            self._refresh_results()
            self._inspect()

    def show_results(
        self,
        a: RunResult | None,
        b: RunResult | None,
        *,
        stale_ids: frozenset[str],
        comparison_block: str | None = None,
    ) -> None:
        """Show caller-selected results. None never substitutes previous output.

        Pass reused immutable RunResult instances, converted off-loop by the
        coordinator. Stale-only updates reuse measurements and diff documents.
        """
        changed = (
            a is not self._results["A"]
            or b is not self._results["B"]
            or comparison_block != self._comparison_block
        )
        self._stale_ids = stale_ids
        self._comparison_block = comparison_block
        if changed:
            self._results = {"A": a, "B": b}
            self._prepared = None
            self._generation += 1
            self._inspection += 1
            self._linked = {"A": (), "B": ()}
        if self.is_mounted:
            self._refresh_results()
            if changed:
                self._document = "Preparing captured evidence…"
                self._highlight = None
                self._text_page = 0
                self.query_one("#mapping-status", Static).update("", layout=False)
                self._paint_text()
                self._load()

    def _load(self) -> None:
        generation = self._generation
        a, b = self._results.values()
        comparison_block = self._comparison_block

        async def prepare() -> None:
            if not self.is_mounted or not self.query("#comparison-status"):
                return
            prepared = await asyncio.to_thread(_prepare, a, b, comparison_block)
            if (
                generation != self._generation
                or not self.is_mounted
                or not self.query("#comparison-status")
            ):
                return
            self._prepared = prepared
            self._refresh_results()
            self._inspect()

        self.run_worker(prepare, group="result-statistics", exclusive=True)

    def _side(self) -> str:
        active = self._view["active_view"]
        if active in {"A", "B"}:
            return active
        candidate_id = self._view.get("inspected_candidate")
        for side, result in self._results.items():
            if result and result.request.candidate_id == candidate_id:
                return side
        return "B" if self._results["B"] else "A"

    def _index(self, side: str) -> int:
        result = self._results[side]
        if not result or not result.report or not result.report.chunks:
            return 0
        return min(
            self._view["selections"].get(result.request.candidate_id, 0),
            len(result.report.chunks) - 1,
        )

    def on_resize(self) -> None:
        if self.is_mounted:
            self._visibility()

    def _visibility(self) -> None:
        active = self._view["active_view"]
        for side in ("A", "B"):
            self.query_one(f"#column-{side.lower()}").display = (
                active == side
                or active == "Compare"
                and (self.size.width >= 120 or self._side() == side)
            )
            self.query_one(f"#view-{side.lower()}", Button).variant = (
                "primary" if active == side else "default"
            )
        self.query_one("#view-compare", Button).variant = (
            "primary" if active == "Compare" else "default"
        )

    def _refresh_results(self) -> None:
        self._visibility()
        reason = (
            self._prepared["reason"]
            if self._prepared
            else "Preparing captured evidence…"
        )
        self.query_one("#comparison-status", Static).update(
            reason
            or "Comparable captured results · common counts only · * linked source span",
            layout=False,
        )
        for side in ("A", "B"):
            self._render_table(side)

    def _render_table(self, side: str) -> None:
        lower = side.lower()
        result = self._results[side]
        chunks = result.report.chunks if result and result.report else ()
        index = self._index(side)
        page = index // PAGE_SIZE
        self._pages[side] = page
        status = f"{side} · No current result. Run preview."
        if result:
            labels = [
                side,
                result.status,
                f"{len(chunks):,} chunks",
                result.request.recipe.runtime.backend,
            ]
            if result.request.run_id in self._previous_ids and (
                self._previous_sides is None or side in self._previous_sides
            ):
                labels.append("Previous")
            if result.request.run_id in self._stale_ids:
                labels.append("Newer draft")
            status = " · ".join(labels)
        self.query_one(f"#status-{lower}", Static).update(status, layout=False)
        table = self.query_one(f"#chunks-{lower}", DataTable)
        # Suppress delayed clear/move echoes; only actual user intent updates view.
        with table.prevent(DataTable.RowHighlighted, DataTable.RowSelected):
            table.clear()
            start = page * PAGE_SIZE
            summary = self._prepared["summaries"].get(side) if self._prepared else None
            for i in range(start, min(start + PAGE_SIZE, len(chunks))):
                text = chunks[i]["text"]
                char_count = summary["character_sizes"][i] if summary else "…"
                word_count = summary["word_sizes"][i] if summary else "…"
                table.add_row(
                    Text("*" if i in self._linked[side] else ""),
                    str(i + 1),
                    str(char_count),
                    str(word_count),
                    Text(text[:64].replace("\n", " ")),
                    key=f"{result.request.run_id}:{i}",
                )
            if chunks:
                table.move_cursor(row=index - start, column=0)
        total_pages = max(1, (len(chunks) + PAGE_SIZE - 1) // PAGE_SIZE)
        self.query_one(f"#page-{lower}", Static).update(
            f"Page {page + 1}/{total_pages}", layout=False
        )
        for action in ("first", "back"):
            self.query_one(f"#{action}-{lower}", Button).disabled = page == 0
        for action in ("next", "last"):
            self.query_one(f"#{action}-{lower}", Button).disabled = (
                page == total_pages - 1
            )

    def _emit_selection(self) -> None:
        side = self._side()
        result = self._results[side]
        view = {**self._view, "selections": dict(self._view["selections"])}
        self.post_message(
            self.SelectionChanged(
                result.request.candidate_id if result else None,
                self._index(side),
                self._view["active_view"],
                view,
            )
        )

    def _select(self, side: str, index: int) -> None:
        result = self._results[side]
        if not result or not result.report or not result.report.chunks:
            return
        self._view["selections"][result.request.candidate_id] = index
        self._view["inspected_candidate"] = result.request.candidate_id
        self._render_table(side)
        self._inspect()
        self._emit_selection()

    @on(DataTable.RowHighlighted)
    @on(DataTable.RowSelected)
    def _row(self, event: DataTable.RowHighlighted | DataTable.RowSelected) -> None:
        side = "A" if event.data_table.id == "chunks-a" else "B"
        result = self._results[side]
        row = event.cursor_row
        if not result or row != event.data_table.cursor_row:
            return
        index = self._pages[side] * PAGE_SIZE + row
        if event.row_key.value != f"{result.request.run_id}:{index}":
            return
        if index == self._index(side) and self._side() == side:
            return
        self._select(side, index)

    @on(Select.Changed, "#detail-kind")
    def _detail_changed(self, event: Select.Changed) -> None:
        if event.value == self._view["detail"]:
            return
        self._view["detail"] = str(event.value)
        self._inspect()
        self._emit_selection()

    @on(Button.Pressed)
    def _button(self, event: Button.Pressed) -> None:
        name = event.button.id or ""
        if name == "rerun-both":
            self.post_message(self.RerunRequested())
        elif name.startswith("view-"):
            self._view["active_view"] = (
                name[5:].capitalize() if name == "view-compare" else name[-1].upper()
            )
            self._refresh_results()
            self._inspect()
            self._emit_selection()
        elif name in {"text-back", "text-next"}:
            self._text_page += -1 if name == "text-back" else 1
            self._paint_text()
        elif name.rsplit("-", 1)[0] in {"first", "back", "next", "last"}:
            action, lower = name.rsplit("-", 1)
            side = lower.upper()
            result = self._results[side]
            if not result or not result.report or not result.report.chunks:
                return
            last = (len(result.report.chunks) - 1) // PAGE_SIZE
            page = {
                "first": 0,
                "last": last,
                "back": self._pages[side] - 1,
                "next": self._pages[side] + 1,
            }[action]
            self._select(side, max(0, min(page, last)) * PAGE_SIZE)

    def _inspect(self) -> None:
        self._inspection += 1
        inspection = self._inspection
        side = self._side()
        result = self._results[side]
        other_side = "B" if side == "A" else "A"
        other = self._results[other_side]
        index = self._index(side)
        detail = self._view["detail"]
        prepared = self._prepared

        def inspect() -> tuple:
            if not result:
                return "No current result. Run preview to inspect chunks.", "", None, ()
            if detail in {"statistics", "effective", "authored"}:
                return (
                    (
                        prepared["documents"][detail]
                        if prepared
                        else "Preparing captured evidence…"
                    ),
                    "Captured result snapshots",
                    None,
                    (),
                )
            if detail == "execution":
                runtime = result.request.recipe.runtime
                chunk = (
                    result.report.chunks[index]
                    if result.report and result.report.chunks
                    else None
                )
                return (
                    _json(
                        {
                            "run_id": result.request.run_id,
                            "status": result.status,
                            "backend": runtime.backend,
                            "engine_version": runtime.engine_version,
                            "execution_version": runtime.execution_version,
                            "assets": runtime.assets,
                            "elapsed_ms_observation": result.elapsed_ms,
                            "started_at": result.started_at,
                            "finished_at": result.finished_at,
                            "sample_hash": result.request.sample.sample_hash,
                            "recipe_hash": result.request.recipe.recipe_hash,
                            "metadata": chunk["metadata"] if chunk else None,
                            "provenance": chunk["provenance"] if chunk else None,
                            "diagnostics": result.report.diagnostics
                            if result.report
                            else (),
                            "error": result.error,
                        }
                    ),
                    f"{side} · Captured execution details",
                    None,
                    (),
                )
            if not result.report:
                return (
                    _json(result.error),
                    f"{result.status}; this run has no comparison output.",
                    None,
                    (),
                )
            if not result.report.chunks:
                return (
                    "This successful recipe emitted zero chunks.",
                    "No chunk selected; alignment and quantiles unavailable.",
                    None,
                    (),
                )
            chunk = result.report.chunks[index]
            mapping = chunk_mapping(result, index)
            linked = (
                linked_chunks(result, index, other)
                if other and other.report and not self._comparison_block
                else ()
            )
            space = mapping["coordinate_space"]
            label = (
                f"{side} chunk {index + 1} · {space} [{mapping['start']}:{mapping['end']}]"
                if space
                else f"{side} chunk {index + 1} · Alignment unavailable"
            )
            if mapping["reason"]:
                label += " · " + mapping["reason"]
            if detail == "source" and space:
                text = (
                    result.request.sample.text
                    if space == "source"
                    else result.report.transformed_text
                )
                return text, label, (mapping["start"], mapping["end"]), linked
            return chunk["text"], label, None, linked

        async def update() -> None:
            if not self.is_mounted or not self.query("#mapping-status"):
                return
            document, label, highlight, linked = await asyncio.to_thread(inspect)
            if (
                inspection != self._inspection
                or not self.is_mounted
                or not self.query("#mapping-status")
            ):
                return
            self._document, self._highlight = document, highlight
            self._text_page = highlight[0] // TEXT_PAGE_SIZE if highlight else 0
            self.query_one("#mapping-status", Static).update(label, layout=False)
            self._linked = {side: (), other_side: linked}
            self._render_table(other_side)
            self._paint_text()

        self.run_worker(update, group="result-inspection", exclusive=True)

    def _paint_text(self) -> None:
        pages = max(1, (len(self._document) + TEXT_PAGE_SIZE - 1) // TEXT_PAGE_SIZE)
        self._text_page = max(0, min(self._text_page, pages - 1))
        start = self._text_page * TEXT_PAGE_SIZE
        text = self._document[start : start + TEXT_PAGE_SIZE]
        area = self.query_one(TextArea)
        area.load_text(text)
        if self._highlight:
            left, right = self._highlight
            if left < start + len(text) and right > start:

                def location(offset: int) -> tuple[int, int]:
                    prefix = text[:offset]
                    return prefix.count("\n"), len(prefix.rsplit("\n", 1)[-1])

                area.selection = Selection(
                    location(max(0, left - start)),
                    location(min(len(text), right - start)),
                )
        self.query_one("#text-page", Static).update(
            f"Text {self._text_page + 1}/{pages}", layout=False
        )
        self.query_one("#text-back", Button).disabled = self._text_page == 0
        self.query_one("#text-next", Button).disabled = self._text_page == pages - 1
