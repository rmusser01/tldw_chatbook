"""Runs pane for the watchlists screen."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.text import Text
from textual import work
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Static
from textual.worker import get_current_worker

from ...Subscriptions.html_text import strip_control_characters
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .humane_time import humane_timestamp
from .table_selection import highlight_is_user_driven


class RunSelected(Message):
    """Posted when the user selects a run in the runs table."""

    def __init__(self, run: dict[str, Any] | None) -> None:
        self.run = run
        super().__init__()


class CancelRunRequested(Message):
    """Posted when the user requests cancellation of a run."""

    def __init__(self, run_id: Any) -> None:
        self.run_id = run_id
        super().__init__()


class RefreshRunsRequested(Message):
    """Posted when the user requests a fresh run list."""


class RunProgressTick(Message):
    """Posted once a second by `RunsPane.run_poll` while a run is running.

    Distinct from `RunSelected` (Qodo, PR #1348). The poll used to re-post
    `RunSelected` on every tick, and the screen's handler cannot tell a tick
    from a click -- so a selected running run scheduled a full run-detail
    load, worker and item query included, every second with no user action.
    `RunSelected` now means "the user picked a different run"; this means
    "the run you are looking at may have moved on", and its handler refreshes
    only what actually changed.
    """

    def __init__(self, run_id: Any) -> None:
        self.run_id = run_id
        super().__init__()


class RerunRunRequested(Message):
    """Posted when the user requests re-running a source/job."""

    def __init__(self, runtime_backend: str, target_id: Any, name: str) -> None:
        self.runtime_backend = runtime_backend
        self.target_id = target_id
        self.name = name
        super().__init__()


class RunsPane(RecomposeCaptureGuard, Vertical):
    """Run list and run inspector for watchlists."""

    #: task-876: same Rich terminal-agnostic "current item" idiom as
    #: `SourcesPane._SELECTED_ROW_STYLE` -- see that attribute's docstring.
    _SELECTED_ROW_STYLE = "reverse bold"

    runs = reactive[list[dict[str, Any]]](list, recompose=True)
    selected_run = reactive[dict[str, Any] | None](None)
    #: task-2306. Deliberately NOT `recompose=True`, unlike `runs`: both are
    #: rewritten on every run selection, and a pane recompose rebuilds
    #: `#runs-table` -- the very table the user just clicked -- discarding its
    #: cursor and remounting it unfocused, which `highlight_is_user_driven`
    #: would then read as a non-user highlight. They are pushed into the live
    #: detail widgets instead, the same in-place discipline
    #: `_update_selection_highlight` already uses for the table itself.
    run_items = reactive[list[dict[str, Any]]](list)
    run_logs = reactive("")
    #: Why the Items table looks the way it does, whenever the rows alone
    #: would mislead (review wave, Important 1 / Minor 2). An empty items
    #: table is produced by four unrelated situations -- a run whose item rows
    #: a later check re-claimed, a genuinely empty check, a server-backend run
    #: whose items cannot be listed at all, and a failed query -- and all four
    #: render identically, directly beneath a stats block that may well say
    #: `Found: 3`. A truncated table is the same self-contradiction in
    #: reverse. The screen names the situation; this pane only shows what it
    #: was told.
    run_items_note = reactive("")
    runtime_backend = reactive("local")
    #: The canonical operation identity selected by the screen. RunsPane only
    #: presents membership in this value; the screen owns its construction.
    selected_operation_key = reactive[str | None](None)
    #: Shared Check-now/Re-run operations currently in flight on the screen.
    busy_operation_keys = reactive[frozenset[str]](frozenset())
    #: The subset of busy operations that originated from this pane's Re-run.
    rerun_operation_keys = reactive[frozenset[str]](frozenset())

    # Plain attribute, not a reactive: mirrors SourcesPane's
    # `_highlighted_source_key` for the identical reason -- see that
    # attribute's docstring.
    _highlighted_run_key: str | None = None

    def compose(self):
        rerun_target, _ = self._rerun_target_and_name(
            self.selected_run, self.runtime_backend
        )
        operation_key = self.selected_operation_key
        rerun_busy = operation_key is not None and (
            operation_key in self.busy_operation_keys
            or operation_key in self.rerun_operation_keys
        )
        rerun_origin = operation_key is not None and operation_key in self.rerun_operation_keys
        rerun_label = (
            "Re-running..."
            if rerun_origin
            else "Checking..."
            if rerun_busy
            else "Re-run source"
        )
        with Horizontal(id="runs-toolbar", classes="destination-filter-strip"):
            yield Button("Refresh", id="runs-refresh-button", variant="primary")
            yield Button("Cancel run", id="runs-cancel-button", disabled=True)
            yield Button(
                rerun_label,
                id="runs-rerun-button",
                disabled=not self._has_rerun_target(rerun_target) or rerun_busy,
            )

        selected_key = str(self.selected_run.get("id")) if self.selected_run else None
        table = DataTable(id="runs-table")
        table.add_columns(
            "Source / Job", "Status", "Started", "Duration", "Found", "Processed", "Filtered", "Errors"
        )
        for run in self.runs:
            row_key = str(run.get("id") or id(run))
            table.add_row(
                *self._run_row_cells(run, row_key == selected_key),
                key=row_key,
            )
        # See `SourcesPane.compose()`'s identical assignment for why this is
        # authoritative going forward.
        self._highlighted_run_key = selected_key
        yield table
        # TASK-2313, AC#4: a bare empty table with zero guidance, next to
        # Overview's own multi-paragraph first-run walkthrough, read as
        # broken rather than merely empty. One line, not Overview's full
        # guidance -- this pane is reached only once a watchlist already
        # exists, so it only has to explain the ONE remaining step.
        if not self.runs:
            yield Static(
                "No runs yet. Press Check now under Sources, or wait for "
                "the next scheduled check.",
                id="runs-empty-state",
                classes="watchlists-hint-line",
            )

        selected_run = self.selected_run
        with Vertical(id="runs-detail-pane"):
            yield Static("Run detail", classes="pane-title")
            # `Text`, not the bare string: the detail block names the run's
            # source and watchlist (user-typed) and, on a failure, quotes the
            # remote error verbatim -- a `Static` given a `str` renders it as
            # console markup.
            yield Static(
                Text(self._stats_text(selected_run)),
                id="runs-detail-stats",
            )
            yield Static("Items", classes="pane-title")
            items_table = DataTable(id="runs-detail-items")
            items_table.add_columns("Title", "Status", "Alerts")
            for item in self.run_items:
                items_table.add_row(*self._run_item_row_cells(item))
            yield items_table
            note = Static(
                Text(self.run_items_note),
                id="runs-detail-items-note",
                classes="runs-detail-note",
            )
            note.display = bool(self.run_items_note)
            yield note
            yield Static("Logs", classes="pane-title")
            yield Static(Text(self.run_logs), id="runs-detail-logs")

    @staticmethod
    def _run_item_row_cells(item: dict[str, Any]) -> tuple[Text, ...]:
        """One run-detail item row, inert.

        `DataTable`'s `default_cell_formatter` runs `Text.from_markup` over any
        plain `str` cell, and an item title is remote content (a feed entry's
        own `<title>`), so these must arrive as `Text` already.

        Batch-4 review, I1: `Text(...)` protects against Rich markup, not a
        raw control byte -- the title is stripped for the same reason
        `sources_pane._source_row_cells` strips a source's name.
        """
        return (
            Text(strip_control_characters(item.get("title") or "Untitled")),
            Text(str(item.get("status") or "-")),
            Text(str(item.get("alert_count") or "0")),
        )

    @staticmethod
    def _run_row_cells(run: dict[str, Any], highlighted: bool) -> tuple[Text, ...]:
        """One row's cell values, styled if `highlighted` (task-876).

        Shared between `compose()` and `_update_selection_highlight` so both
        draw an identical row -- see `SourcesPane._source_row_cells`.
        """
        style = RunsPane._SELECTED_ROW_STYLE if highlighted else ""
        return (
            Text(RunsPane._run_identity(run), style=style),
            Text(str(run.get("status") or "-"), style=style),
            # TASK-2308: local, human-scale. The stored value is a UTC ISO
            # string with microseconds, and it was the widest column in a
            # table with eight of them.
            Text(humane_timestamp(run.get("started_at")), style=style),
            Text(str(run.get("duration") or "-"), style=style),
            Text(str(run.get("found_count") or "0"), style=style),
            Text(str(run.get("processed_count") or "0"), style=style),
            Text(str(run.get("filtered_count") or "0"), style=style),
            Text(str(run.get("error_count") or "0"), style=style),
        )

    @staticmethod
    def _run_identity(run: dict[str, Any]) -> str:
        """What the "Source / Job" column says for `run` (task-2305).

        The source's name, plus the watchlist it sits in when it sits in one
        -- a run history that names only sources is ambiguous the moment the
        same feed is watched from two watchlists. Only the FIRST watchlist is
        spelled out, with a `+N` for the rest: `DataTable` sizes a column to
        its widest cell, so an unbounded join would push the eight accounting
        columns off the side of the pane.

        Args:
            run: A normalized run record.

        Returns:
            e.g. `"Hacker News · Morning read"`, `"Hacker News · Morning read
            +2"`, `"Hacker News"`, or `"Untitled"` for a run whose source can
            no longer be resolved.
        """
        # No `job_name` fallback: no normalizer emits that key (review wave,
        # Minor 4), and an unreachable fallback reads as "some backend
        # supplies this" to the next person here.
        #
        # Batch-4 review, I1: `source_title`/`watchlist_names` are stripped
        # of control characters -- `_run_row_cells` wraps this string in a
        # `Text(...)`, which stops Rich markup but not a raw control byte,
        # and `source_title` is remote-derived the same way an item's title
        # is (see `_run_item_row_cells`).
        source = strip_control_characters(run.get("source_title") or "").strip()
        if not source:
            source = "Untitled"
        names = [
            strip_control_characters(name).strip()
            for name in (run.get("watchlist_names") or [])
            if str(name).strip()
        ]
        if not names:
            return source
        suffix = names[0] if len(names) == 1 else f"{names[0]} +{len(names) - 1}"
        return f"{source} · {suffix}"

    @staticmethod
    def _stats_text(run: dict[str, Any] | None) -> str:
        if not run:
            return "No run selected."
        # task-2305: the detail block names the run's source outright, and
        # lists EVERY watchlist it belongs to -- the row abbreviates for width,
        # the detail block has no such constraint and is where the full answer
        # belongs.
        identity = f"Source: {run.get('source_title') or 'Untitled'}\n"
        watchlists = [
            str(name).strip()
            for name in (run.get("watchlist_names") or [])
            if str(name).strip()
        ]
        if watchlists:
            identity += f"Watchlists: {', '.join(watchlists)}\n"
        base = (
            identity
            + f"Status: {run.get('status', '-')}\n"
            f"Started: {humane_timestamp(run.get('started_at'))}\n"
            f"Duration: {run.get('duration', '-')}\n"
            f"Found: {run.get('found_count', 0)} | "
            f"Processed: {run.get('processed_count', 0)} | "
            f"Filtered: {run.get('filtered_count', 0)} | "
            f"Errors: {run.get('error_count', 0)}"
        )
        # TASK-1362 Task 7 (spec §4): a url-family run's check dispositions,
        # so a silent run finally says WHY it was silent (unchanged? withheld
        # under threshold? re-baselined?) instead of just "Found: 0". Absent
        # entirely for feed/API runs, which have no dispositions at all (see
        # `normalize_watchlist_run`) -- `dispositions` is only ever `{}` or
        # missing for those, so no empty "Checks:" line is added.
        dispositions = run.get("dispositions") or {}
        if dispositions:
            # Whole-branch review, Critical 1. `baseline` and `rebaselined` are
            # rendered separately because they mean opposite things: a first
            # check discarded nothing, while a settings-change re-baseline
            # threw away a real diff window in which a change could have been
            # lost. Spec §3 accepts that lost window only on the strength of
            # this line saying so -- one `baseline` count could not, which left
            # the disposition's `reason` with no consumer anywhere in the
            # product.
            withheld = dispositions.get("withheld", 0)
            withheld_text = f"{withheld} withheld"
            max_withheld = run.get("max_withheld_pct")
            if withheld and isinstance(max_withheld, (int, float)):
                # Spec §1: say what is being withheld, not merely that
                # something was. Without the number the user cannot tell a
                # threshold that is slightly too high from one that is
                # swallowing everything.
                withheld_text += f" (largest {float(max_withheld):.1f}%)"
            base += (
                f"\nChecks: {dispositions.get('changed', 0)} changed | "
                f"{dispositions.get('unchanged', 0)} unchanged | "
                f"{withheld_text} | "
                f"{dispositions.get('baseline', 0)} baseline | "
                f"{dispositions.get('rebaselined', 0)} re-baselined "
                "(settings changed) | "
                # task-1394: a URL that raised (timeout, SSRF block, HTTP
                # error) instead of completing `check_url` -- rendered
                # unconditionally, same as `changed`/`baseline`/etc. above,
                # so a partially-failed run says so rather than reading like
                # a clean one that merely found nothing.
                f"{dispositions.get('error', 0)} error"
            )
            # task-16838: URLs this run never checked because another check
            # of the same source was already running (a scheduled check
            # overlapping a Check Now). Conditional, unlike `error`: an
            # omitted segment always means a true zero -- the counts are
            # zero-filled at write time and `.get(..., 0)` covers rows from
            # before the counter existed -- so absence is not ambiguous, and
            # a rare event does not widen every normal run's line.
            skipped = dispositions.get("skipped", 0)
            if skipped:
                base += f" | {skipped} skipped (check already running)"
        return base

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.data_table.id != "runs-table":
            return
        self.select_run_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        if event.data_table.id != "runs-table":
            return
        self.select_run_by_id(str(event.cell_key.row_key.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a mouse click produces.

        TASK-1105, matching `SourcesPane`. Scoped to `#runs-table`: this pane
        also owns `#runs-detail-items`, whose rows are the *content* of the
        selected run, not runs -- highlighting one of those must not try to
        re-select a run by an item's key (and would resolve to `None`,
        clearing the very selection that produced the detail table).
        """
        event.stop()
        if event.data_table.id != "runs-table":
            return
        if not highlight_is_user_driven(event):
            return
        if event.row_key is not None and event.row_key.value is not None:
            self.select_run_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped."""
        event.stop()
        if event.data_table.id != "runs-table":
            return
        if not highlight_is_user_driven(event):
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is not None and row_key.value is not None:
            self.select_run_by_id(str(row_key.value))

    def select_run_by_id(self, run_id: str) -> None:
        """Select the run with the given id and notify listeners."""
        run = None
        for candidate in self.runs:
            if str(candidate.get("id") or "") == run_id:
                run = candidate
                break
        self.selected_run = run

    def watch_selected_run(self, run: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(RunSelected(run))
        self._update_action_buttons()
        self._update_selection_highlight(run)
        # task-2306. THE defect this task exists for: `selected_run` is not
        # `recompose=True` (and must not become one -- see `run_items`), so
        # `#runs-detail-stats` was written exactly once, by the `compose()`
        # that ran before anything was selected. Every later selection moved
        # the row highlight and armed the buttons while the detail block sat
        # on "No run selected." forever.
        self._update_detail_stats(run)
        # The previous run's items and log belong to the previous run. Cleared
        # here rather than left standing until the screen's loader answers, so
        # a slow (or failing) load can never attribute one run's items to
        # another. The screen re-fills both -- see
        # `WatchlistsCollectionsScreen._load_run_detail`.
        self.run_items = []
        self.run_logs = ""
        self.run_items_note = ""
        if run and str(run.get("status", "")).lower() == "running":
            self._start_run_poll(run)

    def watch_run_items(self, items: list[dict[str, Any]]) -> None:
        """Repopulate `#runs-detail-items` in place (task-2306).

        Args:
            items: The selected run's item rows, newest first. An empty list
                clears the table; `run_items_note` is what explains why.
        """
        try:
            table = self.query_one("#runs-detail-items", DataTable)
        except Exception:
            # Not composed yet; `compose()` seeds the table from the same
            # reactive, so nothing is lost.
            return
        try:
            table.clear()
            for item in items:
                table.add_row(*self._run_item_row_cells(item))
        except Exception:
            pass

    def watch_run_items_note(self, note: str) -> None:
        """Repaint the Items empty/truncation note in place (review wave, I1).

        Hidden rather than left as an empty line when there is nothing to say,
        so the note never puts a blank gap between the table and `Logs`.

        Args:
            note: Why the table looks the way it does, or `""` when the rows
                speak for themselves (which hides the widget).
        """
        try:
            widget = self.query_one("#runs-detail-items-note", Static)
        except Exception:
            return
        try:
            widget.update(Text(str(note)))
            widget.display = bool(note)
        except Exception:
            return

    def watch_run_logs(self, logs: str) -> None:
        """Repaint `#runs-detail-logs` in place (task-2306).

        Args:
            logs: The selected run's log text, rendered inert -- a failed run
                quotes the remote error verbatim.
        """
        try:
            self.query_one("#runs-detail-logs", Static).update(Text(str(logs)))
        except Exception:
            return

    def _update_detail_stats(self, run: dict[str, Any] | None) -> None:
        """Repaint the run-detail stats block for `run`."""
        try:
            self.query_one("#runs-detail-stats", Static).update(
                Text(self._stats_text(run))
            )
        except Exception:
            return

    def _update_selection_highlight(self, run: dict[str, Any] | None) -> None:
        """Move the table's selected-row highlight without rebuilding it.

        Mirrors `SourcesPane._update_selection_highlight` -- see that
        method's docstring; `selected_run` is not `recompose=True` for the
        same reason `selected_source` is not.
        """
        new_key = str(run.get("id")) if run else None
        old_key = self._highlighted_run_key
        if new_key == old_key:
            return
        try:
            table = self.query_one("#runs-table", DataTable)
        except Exception:
            self._highlighted_run_key = new_key
            return
        try:
            column_keys = list(table.columns.keys())
        except Exception:
            column_keys = []
        for row_key, highlighted in ((old_key, False), (new_key, True)):
            if row_key is None:
                continue
            candidate = next(
                (r for r in self.runs if str(r.get("id") or "") == row_key), None
            )
            if candidate is None:
                continue
            cells = self._run_row_cells(candidate, highlighted)
            for column_key, value in zip(column_keys, cells):
                try:
                    table.update_cell(row_key, column_key, value, update_width=False)
                except Exception:
                    pass
        self._highlighted_run_key = new_key

    def _update_action_buttons(self) -> None:
        try:
            cancel_button = self.query_one("#runs-cancel-button", Button)
            rerun_button = self.query_one("#runs-rerun-button", Button)
        except Exception:
            return
        run = self.selected_run
        can_cancel = run is not None and str(run.get("status", "")).lower() == "running"
        rerun_target, _ = self._rerun_target_and_name(run, self.runtime_backend)
        can_rerun = self._has_rerun_target(rerun_target)
        operation_key = self.selected_operation_key
        rerun_busy = operation_key is not None and (
            operation_key in self.busy_operation_keys
            or operation_key in self.rerun_operation_keys
        )
        rerun_origin = operation_key is not None and operation_key in self.rerun_operation_keys
        cancel_button.disabled = not can_cancel
        rerun_button.disabled = not can_rerun or rerun_busy
        rerun_button.label = (
            "Re-running..."
            if rerun_origin
            else "Checking..."
            if rerun_busy
            else "Re-run source"
        )

    def watch_selected_operation_key(self, _value: str | None) -> None:
        """Repaint action buttons without rebuilding the table."""
        self._update_action_buttons()

    def watch_busy_operation_keys(self, _value: frozenset[str]) -> None:
        """Repaint shared Check-now busy state in place."""
        self._update_action_buttons()

    def watch_rerun_operation_keys(self, _value: frozenset[str]) -> None:
        """Repaint Re-run-origin busy state in place."""
        self._update_action_buttons()

    def watch_runtime_backend(self, _value: str) -> None:
        """Re-evaluate backend-specific Re-run eligibility in place."""
        self._update_action_buttons()

    @staticmethod
    def _has_rerun_target(target_id: Any) -> bool:
        """Return whether a backend-specific launch id is present."""
        return target_id is not None and bool(str(target_id).strip())

    @classmethod
    def _rerun_target_and_name(
        cls, run: dict[str, Any] | None, runtime_backend: str
    ) -> tuple[Any, str]:
        """Choose the launch id and inert display name for a selected run."""
        if not run:
            return None, ""
        backend = str(runtime_backend).lower()
        if backend == "server":
            target_id = run.get("job_id")
            fallback = f"Job {target_id}" if cls._has_rerun_target(target_id) else ""
            name = strip_control_characters(str(run.get("source_title") or "")).strip()
            name = name or fallback
        else:
            target_id = run.get("source_id")
            name = cls._run_identity(run)
        return target_id, name

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        run = self.selected_run
        if button_id == "runs-cancel-button" and run:
            self.post_message(CancelRunRequested(run.get("id")))
        elif button_id == "runs-rerun-button" and run:
            target_id, name = self._rerun_target_and_name(run, self.runtime_backend)
            if self._has_rerun_target(target_id):
                self.post_message(
                    RerunRunRequested(self.runtime_backend, target_id, name)
                )
        elif button_id == "runs-refresh-button":
            self.post_message(RefreshRunsRequested())
        event.stop()

    def _start_run_poll(self, run: dict[str, Any]) -> None:
        self.run_poll(run)

    @work(exclusive=True, group="watchlists-runs-poll")
    async def run_poll(self, run: dict[str, Any]) -> None:
        """Poll the selected run while it is running.

        Posts `RunProgressTick`, not `RunSelected` (Qodo, PR #1348): a tick is
        not a selection, and the screen's `RunSelected` handler schedules a
        full detail load. The tick's own handler re-reads the run record and
        does nothing further unless it actually changed.

        Args:
            run: The run to watch. The poll stops as soon as the selection
                moves off it or it leaves the `running` state.
        """
        worker = get_current_worker()
        run_id = run.get("id")
        for _ in range(60):
            if worker.is_cancelled:
                return
            await asyncio.sleep(1)
            current = self.selected_run
            if current is None or str(current.get("id")) != str(run_id):
                return
            if str(current.get("status", "")).lower() != "running":
                return
            self.post_message(RunProgressTick(run_id))

    def apply_run_progress(self, run: dict[str, Any]) -> None:
        """Fold a re-read run record into the table and the detail stats.

        The targeted half of the tick (Qodo, PR #1348). Deliberately does NOT
        assign the `selected_run` reactive normally: this is the SAME run
        progressing, not a new selection, and `watch_selected_run` would post
        `RunSelected`, wipe the detail and restart the poll. `set_reactive`
        updates the value with the watcher suppressed, and the two things that
        genuinely change -- the row's cells and the stats block -- are
        repainted directly.

        Args:
            run: The freshly-read record for a run already in `runs`.
        """
        key = str(run.get("id") or "")
        for index, candidate in enumerate(self.runs):
            if str(candidate.get("id") or "") == key:
                # Mutating the list rather than reassigning the reactive: a
                # reassignment recomposes the pane, which rebuilds the table
                # the user's cursor is sitting in.
                self.runs[index] = run
                break
        self._refresh_run_row(run)
        selected = self.selected_run
        if selected is not None and str(selected.get("id") or "") == key:
            self.set_reactive(RunsPane.selected_run, run)
            self._update_detail_stats(run)

    def _refresh_run_row(self, run: dict[str, Any]) -> None:
        """Repaint one run's row cells in place, keeping its highlight."""
        key = str(run.get("id") or "")
        try:
            table = self.query_one("#runs-table", DataTable)
            column_keys = list(table.columns.keys())
        except Exception:
            return
        cells = self._run_row_cells(run, key == self._highlighted_run_key)
        for column_key, value in zip(column_keys, cells):
            try:
                table.update_cell(key, column_key, value, update_width=False)
            except Exception:
                pass
