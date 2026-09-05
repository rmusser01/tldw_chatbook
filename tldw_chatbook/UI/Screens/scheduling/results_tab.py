"""Automation-results view for the Schedules workbench (schedules-handoff
PR-6 Task 3).

The module keeps its `results_tab` name (redesign PR-4 task 5's own
judgment: renaming the file buys nothing but churn), but the TAB it was
built for is retired -- `ResultsTab` is mounted only inside a pushed
`ResultsHostScreen` now (task 2), reached from the rail's `Results (N)`
button or a definition pane's unread row.


Minimal-honest inbox (plan ruling 1): a `DataTable` of `automation_results`
rows spanning every owner (Task 1's `list_automation_results(owner_id=None)`)
plus a read-only detail pane. Row mutations (read/dismiss/mark-solved/mark
all read) are keybindings owned by the SCREEN, not by this widget -- this
widget is a pure renderer. They began as `SchedulesWorkbench` bindings; task
2 factored the orchestration into the module-level helpers below and task 5
retired the workbench's copies, leaving `ResultsHostScreen` (bottom of this
module) as their owner. Unlike `ConflictsTab`
(whose "Use server"/"Use local" buttons resolve locally, synchronously),
the actions here (`SchedulingService.review_automation_result`/
`resolve_definition`) are async server-aware calls, so they live on a screen
with `run_worker` plumbing rather than in the renderer.

redesign PR-4, task 2 (Results relocation) adds two things on top of the
above, unchanged: (1) `ResultsTab` gains the same `initial_*`-self-paints-
on-mount seam `ConflictsTab.initial_conflicts` established in task 1, plus
an optional `heading` override -- a caller (`SchedulesWorkbench._push_
results_overlay`) can hand a pushed instance an already-scoped (optionally
definition-filtered) results/total pair, and the SAME "showing newest N of
TOTAL" cap-line math renders under whatever heading text it is given,
already-escaped by the caller (this widget never markup-escapes the
heading itself -- it is rendered through the same `Static.update(str)` ->
`Content.from_markup` parser `escape_markup`'s own docstring warns about).
(2) `review_selected_result`/`mark_selected_result_solved`/`mark_results_
read` are the read/dismiss/mark-solved/mark-all-read orchestration,
factored out of `SchedulesWorkbench`'s own tab-routed actions so the tab
and the pushed view could not drift apart. Task 5 then retired the tab
and its copies of those actions, leaving `ResultsHostScreen` (a pushed
`Screen` never receives a screen-underneath's `BINDINGS`, so it must own
them) as the only caller of the first two. The
synchronous "nothing selected"/eligibility gates stay in each CALLER,
not in these helpers (`SchedulesWorkbench.action_mark_result_solved`'s
existing tests pin those refusals firing WITHOUT a worker round-trip);
these helpers only need to be async because the mutation itself is.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Static

# `index_definitions_by_id`/`definition_for_result` moved to
# `unified_rows.py` (redesign PR-2 Task 1 -- that pure module resolves
# results across the same dual local/server id space and cannot import a
# Textual-heavy module without dragging Textual in as a side effect);
# imported back here unchanged so every existing call site keeps working.
from .unified_rows import (
    definition_for_result,
    index_definitions_by_id,  # noqa: F401  (re-export: schedules_workbench.py imports this from here)
)
from .workbench_host_screen import WorkbenchHostScreen

_KIND_GLYPHS = {"finding": "●", "failure": "✕"}  # ● / ✕

#: The tab's heading when every stored result fits in the listing.
RESULTS_HEADING = "Automation results"


def escape_markup(value: str) -> str:
    """Escape EVERY ``[`` so the detail pane renders content literally.

    Deliberately NOT `rich.markup.escape` (nor `textual.markup.escape`,
    which shares its regex): both only escape tags matching
    ``\\[[a-z#/@]...]``, but the parser this pane actually renders
    through -- `Static.update(str)` -> `Content.from_markup` -- consumes
    ANY ``[...]`` token, uppercase included. Live verification (task 6)
    lost a literal ``[PR-6]`` out of a real result answer while the
    existing escaping test passed, because that test only used a
    lowercase ``[bold]`` token, which rich DOES escape. Escaping every
    bracket is the only escape that matches this parser.

    Args:
        value: Text to render literally. Coerced via `str()`, so a
            non-string stored-JSON value is safe to pass straight in.

    Returns:
        The same text with every ``[`` backslash-escaped.
    """
    return str(value).replace("[", "\\[")


#: Failure rows get the same red-toned Rich style `status_badge_text`
#: (task_detail.py) uses for BLOCKED/CONFLICT -- there is no Rich-usable
#: `$error` CSS token, so this reuses that established literal-style idiom
#: rather than inventing a second one.
_FAILURE_KIND_STYLE = "bold white on red"


def _result_kind_cell(kind: str) -> Text:
    """Render the Kind column: a glyph, red-styled for a failure row."""
    glyph = _KIND_GLYPHS.get(kind, "?")
    if kind == "failure":
        return Text(f" {glyph} ", style=_FAILURE_KIND_STYLE)
    return Text(f" {glyph} ")


def _result_owner_suffix(result: dict[str, Any]) -> str:
    """Row-title owner suffix (ruling 4's `_transfer_row_suffix` idiom).

    `""` for a local row -- only a server-scoped owner gets a visible
    suffix, matching `_transfer_row_suffix`'s own "nothing to say" case.
    """
    from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

    owner_id = result.get("owner_id")
    if not is_server_scoped_owner(owner_id):
        return ""
    owner_id = str(owner_id)
    label = owner_id.split(":", 1)[1] if ":" in owner_id else owner_id
    return f" (server: {label})"


def _parse_created_at(created_at: str | None) -> datetime | None:
    """Parse a result's `created_at` into an aware UTC `datetime`, or
    `None` when missing/unparseable.

    `datetime.fromisoformat` (3.11+) accepts both a `Z` suffix (server-
    mirrored rows) and a `+00:00` offset (locally-written rows) -- the
    real comparison this repo's DB layer takes pains to get right via
    `strftime` ordering (`ScheduledTasksDB.list_automation_results`'s own
    docstring: raw text comparison mis-orders those two forms). Naive
    timestamps are treated as UTC, matching `_format_relative`'s
    convention.
    """
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(str(created_at))
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created


def _result_sort_key(result: dict[str, Any]) -> datetime:
    """Newest-first sort key for MERGING two already-DB-sorted result
    lists (redesign PR-4 task 2's definition-filtered query, which reads
    the local- and server-id spaces as two separate queries -- see
    `SchedulesWorkbench._definition_results_query`). An unparseable/
    missing timestamp sorts last (oldest), never raises.

    ``ponytail:`` this compares REAL instants (via `_parse_created_at`),
    correctly ordering the `Z`-vs-`+00:00` mix the DB's own docstring
    warns a raw string sort gets wrong -- but does not replicate that
    same docstring's millisecond-then-raw-text-then-id tie-break for two
    results stamped in the same microsecond. Only used to merge a single
    definition's (at most two id-spaces') results, a low-volume case
    where an exact tie is unlikely; upgrade to a real SQL merge if a
    definition-scoped tie ever turns out to matter.
    """
    return _parse_created_at(result.get("created_at")) or datetime.min.replace(
        tzinfo=UTC
    )


def _format_result_created(
    created_at: str | None, *, now: datetime | None = None
) -> str:
    """Relative "created" timestamp for a results-tab row.

    Same distance-math shape as `task_detail._format_relative`, but a
    result is a past event, not a future run -- "overdue" would be a
    category error, so this uses "X ago" instead.
    """
    created = _parse_created_at(created_at)
    if created is None:
        return "-" if not created_at else str(created_at)
    reference = now if now is not None else datetime.now(UTC)
    seconds = (reference - created).total_seconds()
    future = seconds < 0
    seconds = abs(seconds)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        amount = f"{int(seconds // 60)}m"
    elif seconds < 2 * 86400:
        amount = f"{int(seconds // 3600)}h"
    else:
        amount = f"{int(seconds // 86400)}d"
    return f"in {amount}" if future else f"{amount} ago"


def _degraded(value: Any) -> str:
    """Dim, escaped fallback for a stored-JSON field whose shape didn't
    parse as expected (guard-every-imported-read rule -- these fields
    round-trip through `upsert_automation_results_from_server`, an
    untrusted-payload boundary, and a malformed value must render
    degraded, never raise)."""
    return f"[dim](unparsed — {escape_markup(str(value))})[/dim]"


def _review_state_cell(review_state: str) -> Text:
    """Render the State column: unread is bold with a leading dot."""
    if review_state == "unread":
        return Text(f"● {review_state}", style="bold")
    return Text(review_state or "-")


def solved_eligibility(
    result: dict[str, Any], definitions_by_id: dict[str, dict[str, Any]]
) -> tuple[bool, str | None]:
    """Whether `result` can currently drive a mark-solved action.

    Plan ruling 1's gate: only a `kind="finding"` result can resolve its
    definition, and only while that definition is still unresolved --
    resolution is definition-level (plan ruling 2), so a definition
    already `"solved"` reads as a confusing no-op if offered again here,
    even though the server itself treats a repeat mark-solved as an
    idempotent no-op (Task 2's report). Server-connectivity/transfer-lock
    refusals are NOT checked here: those only surface once the action
    actually runs, from `ResolveOutcome.reason` (UX-073) -- this is a
    client-side "does this even make sense" gate, not a connectivity
    probe.

    Args:
        result: The ``automation_results`` row under the cursor.
        definitions_by_id: The index built by `index_definitions_by_id`.

    Returns:
        ``(True, None)`` when mark-solved applies, else ``(False,
        reason)`` with user-facing copy explaining the refusal.
    """
    if result.get("kind") != "finding":
        return False, "Only findings can be marked solved."
    definition = definition_for_result(result, definitions_by_id)
    if definition is None:
        return False, "This result's automation definition could not be found."
    if definition.get("resolution_state") == "solved":
        return False, "This automation is already marked solved."
    return True, None


class ResultsTab(Vertical):
    """DataTable + detail pane for the `automation_results` inbox."""

    BUNDLED_CSS = """
    ResultsTab {
        height: 1fr;
    }
    #scheduling-results-table {
        height: 1fr;
    }
    #scheduling-results-empty {
        color: $text-muted;
        padding: 2 1;
        display: none;
    }
    #scheduling-results-detail {
        height: auto;
        max-height: 14;
        padding: 0 1;
        color: $text;
        border-top: solid $surface-lighten-2;
    }
    .results-detail-muted {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        initial_results: list[dict[str, Any]] | None = None,
        initial_definitions_by_id: dict[str, dict[str, Any]] | None = None,
        initial_total: int | None = None,
        heading: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the results tab.

        Args:
            initial_results: When given, `populate()`s the table with
                these on mount -- `ConflictsTab.initial_conflicts`'s own
                idiom (task 1). A pushed instance (redesign PR-4, task
                2's rail/definition-pane overlays, via `ResultsHostScreen`)
                has no external `.populate()` driver the way the retired
                mounted tab instance did, so it self-populates. Still
                optional: `populate()` remains callable from outside, and
                `ResultsHostScreen` re-calls it after every mutation.
            initial_definitions_by_id: Paired with `initial_results` --
                the mark-solved eligibility index (`solved_eligibility`).
            initial_total: Paired with `initial_results` -- the honest
                "of TOTAL" denominator for the cap line.
            heading: Overrides `RESULTS_HEADING` (default `None` keeps
                it). MUST already be markup-safe (`escape_markup`): this
                is rendered through the same `Static.update(str)` ->
                `Content.from_markup` parser the detail pane's own
                escaping discipline guards against, and, unlike the
                heading's own hardcoded text, a caller-built heading may
                now carry a user-authored definition name. Lets a
                definition-filtered pushed view say what it's scoped to
                while reusing the SAME "showing newest N of TOTAL" cap
                math unchanged.
            **kwargs: Forwarded verbatim to `Vertical` (id, classes, ...).
        """
        super().__init__(**kwargs)
        self._results_by_id: dict[str, dict[str, Any]] = {}
        self._definitions_by_id: dict[str, dict[str, Any]] = {}
        self._selected_result_id: str | None = None
        self._heading = heading or RESULTS_HEADING
        self._initial_results = initial_results
        self._initial_definitions_by_id = initial_definitions_by_id
        self._initial_total = initial_total

    def compose(self) -> ComposeResult:
        """Build the tab layout."""
        yield Static(self._heading, id="scheduling-results-heading")
        table = DataTable(id="scheduling-results-table")
        table.add_columns("Kind", "Result", "Created", "State")
        yield table
        yield Static(
            "Select a result to see details.",
            id="scheduling-results-detail",
            classes="results-detail-muted",
        )
        yield Static("No results yet.", id="scheduling-results-empty")

    def on_mount(self) -> None:
        """Configure the table cursor, and self-populate if constructed
        with data (see `initial_results`)."""
        table = self.query_one("#scheduling-results-table", DataTable)
        table.cursor_type = "row"
        if self._initial_results is not None:
            self.populate(
                self._initial_results,
                self._initial_definitions_by_id,
                total=self._initial_total,
            )

    @property
    def definitions_by_id(self) -> dict[str, dict[str, Any]]:
        """The definition rows the last `populate()` call was given, by id."""
        return self._definitions_by_id

    def results(self) -> list[dict[str, Any]]:
        """Every result row currently loaded in the tab."""
        return list(self._results_by_id.values())

    def populate(
        self,
        results: list[dict[str, Any]],
        definitions_by_id: dict[str, dict[str, Any]] | None = None,
        total: int | None = None,
    ) -> None:
        """Rebuild the table from a fresh `list_automation_results` page.

        The listing is a capped newest-window (see
        `schedules_workbench.RESULTS_INBOX_LIMIT`), so when `total`
        exceeds the rows given the heading says so outright rather than
        letting the table imply it holds everything -- the unread badge
        counts EVERY result, and a silently truncated table next to it is
        the disagreement this reports honestly. There is deliberately no
        pagination: saying what is hidden is the whole fix.

        Args:
            results: Rows from `ScheduledTasksDB.list_automation_results`.
            definitions_by_id: The owning definitions, by id, for the
                mark-solved eligibility gate (`solved_eligibility`).
            total: How many results exist in total
                (`count_automation_results`). ``None`` means "not
                measured" and never renders a count line.
        """
        self._definitions_by_id = dict(definitions_by_id or {})
        previous_selection = self._selected_result_id
        self._results_by_id = {result["id"]: result for result in results}
        self.query_one("#scheduling-results-heading", Static).update(
            f"{self._heading} — showing newest {len(results)} of {total}"
            if total is not None and total > len(results)
            else self._heading
        )

        table = self.query_one("#scheduling-results-table", DataTable)
        table.clear()
        for result in results:
            table.add_row(
                _result_kind_cell(result.get("kind", "")),
                # `Text`, not `str` (task 6 round 2, D8): `DataTable` runs
                # string cells through `rich.text.Text.from_markup` -- a
                # DIFFERENT parser from the detail pane's, with its own
                # lowercase-tag regex. An LLM-written title carrying
                # `[bold]` would be eaten here even though the detail pane
                # escapes it correctly (round 1 filed exactly this as a
                # known follow-up). A `Text` is passed through unparsed,
                # so no escape is needed at all.
                Text(
                    str(
                        result.get("title")
                        or result.get("definition_id")
                        or "Untitled"
                    )
                    + _result_owner_suffix(result)
                ),
                _format_result_created(result.get("created_at")),
                _review_state_cell(result.get("review_state", "")),
                key=result["id"],
            )

        has_rows = bool(results)
        table.display = has_rows
        detail = self.query_one("#scheduling-results-detail", Static)
        empty_state = self.query_one("#scheduling-results-empty", Static)
        empty_state.display = "none" if has_rows else "block"
        if not has_rows:
            detail.display = False
            self._selected_result_id = None
            return
        detail.display = True

        row_keys = [result["id"] for result in results]
        if previous_selection in row_keys:
            # Restoring the cursor fires RowHighlighted, which re-records
            # the same id -- belt and braces, set both explicitly (the
            # same by-id reconciliation every table rebuild here uses).
            table.cursor_coordinate = (row_keys.index(previous_selection), 0)
            self._selected_result_id = previous_selection
            self._show_detail(self._results_by_id[previous_selection])
        else:
            self._selected_result_id = None
            detail.update("Select a result to see details.")

    @on(DataTable.RowHighlighted, "#scheduling-results-table")
    def _on_result_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show the highlighted result's detail."""
        result_id, result = self._selected_result()
        if result_id is None or result is None:
            return
        self._selected_result_id = result_id
        self._show_detail(result)

    def _selected_result(self) -> tuple[str, dict[str, Any]] | tuple[None, None]:
        """Return the (id, result) pair at the current cursor row."""
        table = self.query_one("#scheduling-results-table", DataTable)
        if table.cursor_row is None or not table.ordered_rows:
            return None, None
        row = table.ordered_rows[table.cursor_row]
        result_id = row.key.value
        return result_id, self._results_by_id.get(result_id)

    def selected_result(self) -> dict[str, Any] | None:
        """Return the result row under the tab's cursor, if any."""
        _, result = self._selected_result()
        return result

    def selected_result_id(self) -> str | None:
        """Return the id of the result row under the tab's cursor, if any."""
        result_id, _ = self._selected_result()
        return result_id

    def _show_detail(self, result: dict[str, Any]) -> None:
        """Render answer / evidence / source_refs / review metadata (spec).

        Every server/LLM-derived string is markup-escaped before
        interpolation (the Library-redesign `escape_markup` lesson --
        this pane renders LLM-generated answers/evidence, so a bracket
        token in real content is likely, not hypothetical), and every
        stored-JSON field (`answer`/`source_refs`/each ref item) is
        shape-checked before use -- a malformed value degrades to an
        `_degraded()` note instead of raising.
        """
        lines: list[str] = []
        title = result.get("title") or "(untitled)"
        lines.append(f"{escape_markup(str(title))}{_result_owner_suffix(result)}")
        lines.append(
            f"Kind: {escape_markup(str(result.get('kind', '?')))}  ·  "
            f"Definition: {escape_markup(str(result.get('definition_id', '?')))}"
        )

        answer = result.get("answer")
        if not answer:
            lines.append("Answer: (none)")
        elif isinstance(answer, str):
            lines.append(f"Answer: {escape_markup(answer)}")
        else:
            lines.append(f"Answer: {_degraded(answer)}")

        source_refs = result.get("source_refs")
        if not source_refs:
            lines.append("Evidence: (none)")
        elif not isinstance(source_refs, list):
            lines.append(f"Evidence: {_degraded(source_refs)}")
        else:
            lines.append("Evidence:")
            for ref in source_refs:
                if isinstance(ref, dict):
                    source_type = escape_markup(str(ref.get("source_type", "?")))
                    source_id = escape_markup(str(ref.get("source_id", "?")))
                    lines.append(f"  - {source_type}: {source_id}")
                elif isinstance(ref, str):
                    lines.append(f"  - {escape_markup(ref)}")
                else:
                    lines.append(f"  - {_degraded(ref)}")

        review_state = result.get("review_state", "unread")
        review_line = f"Review: {review_state}"
        if result.get("reviewed_at"):
            reviewed_by = escape_markup(str(result.get("reviewed_by") or "?"))
            review_line += f" ({reviewed_by} at {result['reviewed_at']})"
        if result.get("review_note"):
            review_line += f" -- {escape_markup(str(result['review_note']))}"
        lines.append(review_line)

        if result.get("kind") == "finding":
            eligible, reason = solved_eligibility(result, self._definitions_by_id)
            lines.append(
                "Solve: eligible"
                if eligible
                else f"Solve: not eligible — {reason}"
            )

        self.query_one("#scheduling-results-detail", Static).update("\n".join(lines))


# -- Shared read/dismiss/mark-solved/mark-all-read orchestration -----------
#
# redesign PR-4, task 2: factored out of `SchedulesWorkbench` so the
# Results tab and `ResultsHostScreen` below could not drift apart; task 5
# retired the tab, so the host screen is now the only caller of the first
# two (the workbench still calls `mark_results_read` for its rail-level
# `Mark all read`). `notify` is `Callable[[message, severity], None]` --
# the caller's own notification sink (`app_instance.notify`/`app.notify`,
# same call shape, different attribute name depending on whether the
# caller is a `BaseAppScreen` subclass or a plain `Screen`).


async def review_selected_result(
    service: Any,
    results_tab: ResultsTab,
    review_state: str,
    notify: Callable[[str, str], None],
) -> None:
    """Read/dismiss the result under `results_tab`'s cursor.

    `SchedulingService.review_automation_result` writes the local row
    and, for a server mirror, queues the sync pushback mutation in the
    same DB transaction -- nothing extra to do here for that half.

    Args:
        service: The `SchedulingService` whose `review_automation_result`
            performs the write.
        results_tab: The `ResultsTab` instance whose cursor selects the
            result to review.
        review_state: `"read"` or `"dismissed"` (the two states the `r`/
            `d` actions drive).
        notify: The caller's notification sink, `(message, severity)`.
    """
    result = results_tab.selected_result()
    if result is None:
        notify("Select a result first.", "warning")
        return
    updated = await service.review_automation_result(result["id"], review_state)
    if not updated:
        notify("Could not update this result — see the log.", "error")


async def mark_selected_result_solved(
    service: Any,
    result: dict[str, Any],
    definitions_by_id: dict[str, dict[str, Any]],
    notify: Callable[[str, str], None],
) -> None:
    """Mark `result`'s definition solved -- the async half of the `o`
    action. Eligibility (`solved_eligibility`) is checked by the CALLER,
    synchronously, before spawning the worker this runs in (existing
    tests pin that refusal firing without a worker round-trip).

    Args:
        service: The `SchedulingService` whose `resolve_definition`
            performs the write.
        result: The `automation_results` row under the cursor.
        definitions_by_id: The index built by `index_definitions_by_id`,
            used to resolve `result`'s owning definition.
        notify: The caller's notification sink, `(message, severity)`.
    """
    definition = definition_for_result(result, definitions_by_id)
    local_definition_id = str((definition or {}).get("id") or "")
    outcome = await service.resolve_definition(
        local_definition_id, solved=True, result_id=result["id"]
    )
    if outcome.status == "saved":
        notify("Marked solved.", "information")
    else:
        notify(outcome.reason or "Could not mark this result solved.", "warning")


async def mark_results_read(
    service: Any,
    result_ids: list[str],
    notify: Callable[[str, str], None],
) -> None:
    """Per-row `review_automation_result` fan-out for a batch of result
    ids -- there is no bulk DB primitive for this (documented fan-out,
    mirroring `SchedulesWorkbench._on_bulk_delete_confirmed`'s loop-and-
    count shape).

    Args:
        service: The `SchedulingService` whose `review_automation_result`
            performs each write.
        result_ids: Every result id to mark `"read"` (the `a` action's
            scope -- global or one definition's, per the caller).
        notify: The caller's notification sink, `(message, severity)`;
            called once at the end with the read/failed count.
    """
    errors = 0
    for result_id in result_ids:
        if not await service.review_automation_result(result_id, "read"):
            errors += 1
    count = len(result_ids) - errors
    notify(
        f"Marked {count} result{'s' if count != 1 else ''} read"
        + (f" ({errors} failed)" if errors else "")
        + ".",
        "information" if not errors else "warning",
    )


class ResultsHostScreen(WorkbenchHostScreen):
    """`WorkbenchHostScreen` + the Results-specific r/d/o/a bindings
    (redesign PR-4, task 2).

    `SchedulesWorkbench` has no r/d/o/a bindings at all any more (task 5
    retired its tab-gated copies along with the Results tab), and it could
    not receive those keys here regardless -- Textual routes a key through
    the CURRENTLY ACTIVE screen's own binding chain only, and a screen
    underneath is not part of that chain. This pushed view therefore owns
    the four actions outright, over the module-level service orchestration
    above (`review_selected_result`/`mark_selected_result_solved`/`mark_
    results_read`), which is where the tab-era logic was factored to
    before the tab went away.

    `query`/`unread_ids` are closures the constructing `SchedulesWorkbench`
    supplies, scoped to whatever this push is (the global inbox, or one
    definition's results across both its id spaces) -- this class has no
    opinion on that scope, it only re-runs whichever closure it was given
    after each mutation to repaint ITS OWN `ResultsTab` instance in place.
    The rail/Queue-tab/badge refresh on the workbench BEHIND this screen
    happens once, on pop, via the base class's `dismissed` hook -- not on
    every keystroke here (brief: "refresh the rail + unified rows on
    dismissed").
    """

    BINDINGS: ClassVar = [
        *WorkbenchHostScreen.BINDINGS,
        Binding("r", "review_read", "Read"),
        Binding("d", "review_dismiss", "Dismiss"),
        Binding("o", "mark_solved", "Mark solved"),
        Binding("a", "mark_all_read", "Mark all read"),
    ]

    def __init__(
        self,
        widget_factory: Callable[[], ResultsTab],
        *,
        title: str,
        service: Any,
        query: Callable[[], tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int]],
        unread_ids: Callable[[], list[str]],
        dismissed: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the host.

        Args:
            widget_factory: Builds the fresh `ResultsTab` instance (see
                `WorkbenchHostScreen`).
            title: Shown in the `Header`.
            service: The `SchedulingService` the r/d/o/a actions call.
            query: Re-runs the scope's own results/definitions/total read
                (global or definition-filtered) -- called once after
                every mutation to repaint this screen's `ResultsTab`.
            unread_ids: Every unread result id in THIS scope, uncapped by
                `RESULTS_INBOX_LIMIT` (the `a` action's own Qodo-HIGH
                fix: `SchedulesWorkbench._unread_result_ids`'s docstring).
            dismissed: Runs once on pop (`Esc`), same as the base class.
        """
        super().__init__(widget_factory, title=title, dismissed=dismissed)
        self._service = service
        self._query = query
        self._unread_ids = unread_ids

    def _results_tab(self) -> ResultsTab:
        return self.query_one(ResultsTab)

    def _notify(self, message: str, severity: str) -> None:
        self.app.notify(message, severity=severity)

    def _repaint(self) -> None:
        results, definitions_by_id, total = self._query()
        self._results_tab().populate(results, definitions_by_id, total=total)

    def action_review_read(self) -> None:
        """`r`: mark the result under the cursor read."""
        self._run_review("read")

    def action_review_dismiss(self) -> None:
        """`d`: dismiss the result under the cursor."""
        self._run_review("dismissed")

    def _run_review(self, review_state: str) -> None:
        results_tab = self._results_tab()

        async def _do() -> None:
            await review_selected_result(
                self._service, results_tab, review_state, self._notify
            )
            self._repaint()

        self.run_worker(_do, exclusive=True, group="schedules-results-host")

    def action_mark_solved(self) -> None:
        """`o`: mark the result under the cursor's definition solved.

        The synchronous eligibility gate (`solved_eligibility`) runs here,
        before the worker -- a refusal notifies without a worker round-
        trip.
        """
        results_tab = self._results_tab()
        result = results_tab.selected_result()
        if result is None:
            self._notify("Select a result first.", "warning")
            return
        eligible, reason = solved_eligibility(result, results_tab.definitions_by_id)
        if not eligible:
            self._notify(reason or "This result cannot be marked solved.", "warning")
            return

        async def _do() -> None:
            await mark_selected_result_solved(
                self._service, result, results_tab.definitions_by_id, self._notify
            )
            self._repaint()

        self.run_worker(_do, exclusive=True, group="schedules-results-host")

    def action_mark_all_read(self) -> None:
        """`a`: mark every unread result in this screen's scope read.

        Scope is whatever `unread_ids` (constructor arg) was built to
        return -- the global inbox, or one definition's results.
        """
        unread_ids = self._unread_ids()
        if not unread_ids:
            self._notify("Nothing unread.", "information")
            return

        async def _do() -> None:
            await mark_results_read(self._service, unread_ids, self._notify)
            self._repaint()

        self.run_worker(_do, exclusive=True, group="schedules-results-host")
