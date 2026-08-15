"""Console trajectory (trace) screen: ledger, inspector, search, pagination.

Modal over the Console (task-4 of the trajectory view; spec:
``Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md``,
"UI — Trajectory screen"). Read-only: it renders a ``TrajectorySnapshot``
from the pure projection (``Chat/trajectory.py``) and never queries the
DB itself.

Layout (vertical): title line, search ``Input``, ``DataTable`` ledger
(``cursor_type="row"``), inspector ``Static`` (hidden until toggled),
footer hints line.

Ledger semantics pinned by the projection's contracts:

- ``TrajectoryRecord.seq`` is the 1-based LEDGER RENDER POSITION -- it is
  used as the DataTable row key and nothing else (never a DB seq).
- Tool records (``depth == 1``) render indented under their owning
  assistant step; their ``payload`` carries name/args/result (result is
  the full untruncated output -- the inspector shows it verbatim).
- Variant contents over-attach to ALL assistant records of a turn, so
  the inspector labels that list "superseded variants (turn-level)".
- Timing fields may be ``None``: blanks are rendered, and durations are
  only ever computed between two PROVIDED endpoints (display-only
  elapsed, never a fabricated fact).

Pagination mirrors dsh: the newest ``PAGE_SIZE`` records mount first; a
"load earlier" row sits above them while older records remain. Rendering
moves off the UI thread (``run_worker``) once the conversation exceeds
``WORKER_THRESHOLD`` records.

Keybindings follow ADR-031: single-letter htop-style actions, no
terminal-convention chords, safe ``escape`` (blur the search box first,
dismiss second), and the footer hints line stays 1:1 with ``BINDINGS``
(enforced by ``Tests/UI/test_trajectory_screen.py``).
"""

from __future__ import annotations

import json
from datetime import datetime

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Static

from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
)

__all__ = [
    "LOAD_EARLIER_ROW_KEY",
    "PAGE_SIZE",
    "WORKER_THRESHOLD",
    "TrajectoryScreen",
]

#: Ledger page size: newest records mounted first (dsh-style pagination).
PAGE_SIZE = 500

#: Above this many records the initial ledger render moves to a worker.
WORKER_THRESHOLD = 5000

#: DataTable row key of the "load earlier" control row.
LOAD_EARLIER_ROW_KEY = "__load_earlier__"

_COLUMNS = (
    ("#", 5),
    ("Kind", 11),
    ("Content", 56),
    ("In", 6),
    ("Cache", 7),
    ("Out", 6),
    ("Start", 9),
    ("Done", 9),
)


def _fmt_clock(unix: float | None) -> str:
    """Format a unix timestamp as local ``HH:MM:SS``; em dash when ``None``."""
    if unix is None:
        return "—"
    try:
        return datetime.fromtimestamp(unix).strftime("%H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "—"


def _fmt_span(start: float | None, end: float | None) -> str | None:
    """Elapsed seconds between two PROVIDED facts; ``None`` when either is missing.

    Display-only derivation from stored endpoints -- never a fabricated
    duration for partially known records.
    """
    if start is None or end is None:
        return None
    return f"{end - start:.2f}s"


class TrajectoryScreen(ModalScreen[None]):
    """Read-only trajectory ledger + inspector for one Console conversation."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("enter", "inspect_cursor_row", "Inspect"),
        Binding("t", "toggle_turn", "Collapse"),
        Binding("i", "toggle_inspector", "Inspector"),
        Binding("e", "load_earlier", "Earlier"),
        Binding("/", "focus_search", "Search"),
    ]

    #: Footer hints, 1:1 with the non-escape BINDINGS (ADR-031; tested).
    TRAJECTORY_SHORTCUTS = (
        ("enter", "inspect"),
        ("t", "collapse"),
        ("i", "inspector"),
        ("/", "search"),
        ("e", "earlier"),
    )

    DEFAULT_CSS = """
    TrajectoryScreen {
        align: left top;
        background: $background;
    }
    #trajectory-screen {
        width: 100%;
        height: 100%;
        layout: vertical;
        padding: 0 1;
    }
    #trajectory-title {
        height: 1;
        color: $text;
        background: $panel;
    }
    #trajectory-search {
        height: 3;
    }
    #trajectory-table {
        height: 1fr;
        min-height: 3;
    }
    #trajectory-inspector {
        height: auto;
        max-height: 40%;
        border-top: solid $primary;
        padding: 0 1;
    }
    #trajectory-hints {
        height: 1;
        color: $text-disabled;
    }
    """

    def __init__(
        self,
        snapshot: TrajectorySnapshot,
        *,
        screen_title: str = "",
        conversation_id: str | None = None,
    ) -> None:
        """Store the projection output; all rendering happens on mount.

        Args:
            snapshot: The ``derive_trajectory`` output to render.
            screen_title: Display title (typically the conversation name).
            conversation_id: Conversation id, shown in the title line.
        """
        super().__init__()
        self._snapshot = snapshot
        self._screen_title = screen_title
        self._conversation_id = conversation_id
        self._turns: tuple[TrajectoryTurn, ...] = snapshot.turns
        self._turn_numbers: dict[str, int] = {
            turn.turn_id: index + 1 for index, turn in enumerate(self._turns)
        }
        self._collapsed: set[str] = set()
        self._query = ""
        #: Number of newest records mounted (grows one page per `e` press).
        self._visible_count = min(self._total_records, PAGE_SIZE)
        #: Row key -> record (None for header/control rows), in row order.
        self._row_records: dict[str, TrajectoryRecord | None] = {}
        self._row_turn_ids: dict[str, str] = {}
        self._visible_keys: list[str] = []
        #: Bumped by every ledger render; a worker-built page carries the
        #: generation it was built at and is DROPPED if a newer render
        #: landed first (typing a search or pressing t/e mid-build must
        #: never be overwritten by a stale off-thread snapshot).
        self._render_generation = 0
        #: False once unmounted: a worker-built ledger arriving late is
        #: dropped instead of poking a dead screen. (``Widget.is_mounted``
        #: is still False DURING ``on_mount``, so the flag is explicit.)
        self._alive = True

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="trajectory-screen"):
            yield Static(self._title_text(), id="trajectory-title", markup=False)
            yield Input(placeholder="Search records…", id="trajectory-search")
            yield DataTable(id="trajectory-table", cursor_type="row")
            inspector = Static("", id="trajectory-inspector", markup=False)
            inspector.display = False  # hidden until toggled (spec)
            yield inspector
            yield Static("", id="trajectory-hints", markup=False)

    def on_mount(self) -> None:
        table = self.query_one("#trajectory-table", DataTable)
        for label, width in _COLUMNS:
            table.add_column(label, width=width)
        self._refresh_hints()
        if self._total_records > WORKER_THRESHOLD:
            # Large conversation: build the row specs off the UI thread.
            self.run_worker(self._render_worker, thread=True, group="trajectory-ledger")
        else:
            self._render_ledger()
        table.focus()

    def on_unmount(self) -> None:
        self._alive = False

    def _title_text(self) -> str:
        parts = ["Trajectory"]
        if self._screen_title:
            parts.append(self._screen_title)
        if self._conversation_id:
            parts.append(f"conv {self._conversation_id}")
        parts.append(f"{self._total_records} records")
        return " · ".join(parts)

    # -- ledger rendering ------------------------------------------------------

    @property
    def _total_records(self) -> int:
        return sum(len(turn.records) for turn in self._turns)

    @property
    def _hidden_earlier(self) -> int:
        return max(0, self._total_records - self._visible_count)

    def _render_ledger(self) -> None:
        self._render_generation += 1
        self._apply_row_specs(self._build_row_specs())

    def _render_worker(self) -> None:
        """Worker-thread half of the >WORKER_THRESHOLD path (video-player pattern)."""
        try:
            generation = self._render_generation
            specs = self._build_row_specs()
            self.app.call_from_thread(self._apply_row_specs, specs, generation)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            logger.warning(
                "Trajectory ledger render failed: component=trajectory_screen "
                "error_type={}",
                type(exc).__name__,
            )

    def _flat_slice(self) -> list[tuple[TrajectoryTurn, TrajectoryRecord]]:
        """(turn, record) pairs for the mounted window (newest ``visible_count``)."""
        flat = [(turn, rec) for turn in self._turns for rec in turn.records]
        start = max(0, len(flat) - self._visible_count)
        return flat[start:]

    def _build_row_specs(self) -> list[tuple[str, tuple[Text, ...]]]:
        """Row specs (key, cells) for the current window/filter/collapse state."""
        specs: list[tuple[str, tuple[Text, ...]]] = []
        if self._hidden_earlier:
            specs.append(
                (
                    LOAD_EARLIER_ROW_KEY,
                    (
                        Text(""),
                        Text(""),
                        Text(
                            f"… load earlier ({self._hidden_earlier} older records) — press e",
                        ),
                        Text(""),
                        Text(""),
                        Text(""),
                        Text(""),
                        Text(""),
                    ),
                )
            )

        query = self._query.lower()
        open_turn: TrajectoryTurn | None = None
        turn_records: list[TrajectoryRecord] = []
        for turn, record in self._flat_slice():
            if open_turn is not None and turn.turn_id != open_turn.turn_id:
                specs.extend(self._turn_row_specs(open_turn, turn_records, query))
                turn_records = []
            open_turn = turn
            turn_records.append(record)
        if open_turn is not None:
            specs.extend(self._turn_row_specs(open_turn, turn_records, query))
        return specs

    def _turn_row_specs(
        self, turn: TrajectoryTurn, records: list[TrajectoryRecord], query: str
    ) -> list[tuple[str, tuple[Text, ...]]]:
        """Header row + child rows for one turn under the current filter.

        Search semantics (spec): child rows match on their own text; the
        turn header survives iff any child matches. A search overrides
        collapse (searching reveals), otherwise collapsed turns show the
        header only.
        """
        matching = [rec for rec in records if self._record_matches(rec, query)]
        if query and not matching:
            return []  # nothing in this turn matches: header included, hidden
        header_key = f"turn:{turn.turn_id}"
        collapsed = turn.turn_id in self._collapsed
        number = self._turn_numbers.get(turn.turn_id, 0)
        marker = "▸" if collapsed else "▾"
        label = Text(f"{marker} Turn {number} · {len(records)} records", style="bold")
        specs = [
            (
                header_key,
                (
                    Text(""),
                    Text("turn"),
                    label,
                    Text(""),
                    Text(""),
                    Text(""),
                    Text(""),
                    Text(""),
                ),
            )
        ]
        if collapsed and not query:
            return specs
        for rec in matching:
            specs.append((str(rec.seq), self._record_cells(rec)))
        return specs

    def _record_cells(self, rec: TrajectoryRecord) -> tuple[Text, ...]:
        usage = rec.usage
        if usage is None:
            tokens = ("", "", "")
        else:
            tokens = (
                str(usage.uncached_input),
                str(usage.cache_read + usage.cache_write),
                str(usage.output),
            )
        indent = "    ↳ " if rec.depth else ""
        content = Text(f"{indent}{rec.content_preview}")
        return (
            Text(str(rec.seq)),
            Text(rec.kind),
            content,
            Text(tokens[0]),
            Text(tokens[1]),
            Text(tokens[2]),
            Text(_fmt_clock(rec.step_started_at)),
            Text(_fmt_clock(rec.completed_at)),
        )

    def _record_matches(self, rec: TrajectoryRecord, query: str) -> bool:
        """Case-insensitive match over the record's searchable text.

        Covers everything the user can see (preview, kind, model/provider)
        plus the tool payload's name/args/result -- tool output lives only
        here, and the 120-char preview would otherwise hide it from search.
        """
        if not query:
            return True
        if query in rec.content_preview.lower():
            return True
        if query in rec.kind:
            return True
        if rec.model and query in rec.model.lower():
            return True
        if rec.provider and query in rec.provider.lower():
            return True
        payload = rec.payload
        if payload:
            name = str(payload.get("name") or "")
            if query in name.lower():
                return True
            args = payload.get("args")
            if args is not None:
                try:
                    args_text = json.dumps(args, sort_keys=True)
                except (TypeError, ValueError):
                    args_text = str(args)
                if query in args_text.lower():
                    return True
            result = payload.get("result")
            if result is not None and query in str(result).lower():
                return True
        return False

    def _apply_row_specs(
        self,
        specs: list[tuple[str, tuple[Text, ...]]],
        generation: int | None = None,
    ) -> None:
        """Rebuild the DataTable from row specs, preserving the cursor by key.

        ``generation`` is the render generation the specs were BUILT at; a
        mismatch against the current one means a newer render already
        landed (the user typed a search or pressed t/e while a worker was
        building) and the stale specs are dropped. ``None`` is the current
        synchronous render, which always applies.
        """
        if not self._alive:
            return
        if generation is not None and generation != self._render_generation:
            return
        table = self.query_one("#trajectory-table", DataTable)
        previous_key: str | None = None
        if self._visible_keys:
            row = table.cursor_row
            if 0 <= row < len(self._visible_keys):
                previous_key = self._visible_keys[row]
        table.clear(columns=False)
        self._row_records = {}
        self._row_turn_ids = {}
        self._visible_keys = []
        for key, cells in specs:
            table.add_row(*cells, key=key)
            self._visible_keys.append(key)
            if key.startswith("turn:"):
                self._row_turn_ids[key] = key.removeprefix("turn:")
            elif key != LOAD_EARLIER_ROW_KEY:
                self._row_records[key] = None  # resolved below
        for turn in self._turns:
            for rec in turn.records:
                if str(rec.seq) in self._row_records:
                    self._row_records[str(rec.seq)] = rec
        if self._visible_keys:
            index = 0
            if previous_key is not None:
                try:
                    index = self._visible_keys.index(previous_key)
                except ValueError:
                    index = 0
            try:
                table.move_cursor(row=index, animate=False)
            except Exception as exc:  # noqa: BLE001 - cursor clamp is best-effort
                logger.debug(
                    "Trajectory cursor restore skipped: {}", type(exc).__name__
                )
        self._refresh_hints()
        if self.query_one("#trajectory-inspector", Static).display:
            self._refresh_inspector()

    # -- footer hints ------------------------------------------------------

    def _refresh_hints(self) -> None:
        """Render the hints line: 1:1 with BINDINGS minus what has no target.

        The ``e earlier`` hint drops while no older records remain (the
        ADR-031 task-1340 refinement: advertised == working in the active
        context; pressing ``e`` then answers with guidance instead).
        """
        pairs = [
            (key, label)
            for key, label in self.TRAJECTORY_SHORTCUTS
            if key != "e" or self._hidden_earlier > 0
        ]
        text = " · ".join(f"{key} {label}" for key, label in pairs)
        try:
            self.query_one("#trajectory-hints", Static).update(text)
        except Exception:  # noqa: BLE001 - pre-mount refresh
            pass

    # -- inspector -----------------------------------------------------------

    def _inspector_text_for_record(self, rec: TrajectoryRecord) -> str:
        lines = [f"#{rec.seq} {rec.kind} · turn {rec.turn_id}"]
        model = rec.model or "—"
        provider = rec.provider or "—"
        lines.append(f"model {model} · provider {provider}")
        usage = rec.usage
        if usage is None:
            lines.append("tokens —")
        else:
            lines.append(
                f"tokens uncached input {usage.uncached_input} · "
                f"cache read {usage.cache_read} · "
                f"cache write {usage.cache_write} · "
                f"output {usage.output}"
            )
        ttft = _fmt_span(rec.step_started_at, rec.first_token_at)
        elapsed = _fmt_span(rec.step_started_at, rec.completed_at)
        first = _fmt_clock(rec.first_token_at)
        if ttft is not None:
            first = f"{first} (+{ttft})"
        completed = _fmt_clock(rec.completed_at)
        if elapsed is not None:
            completed = f"{completed} (elapsed {elapsed})"
        lines.append(
            f"timing start {_fmt_clock(rec.step_started_at)} · "
            f"first token {first} · completed {completed}"
        )
        if rec.content_preview:
            lines.append(f"content {rec.content_preview}")
        payload = rec.payload
        if payload:
            name = str(payload.get("name") or "—")
            lines.append(f"tool {name}")
            args = payload.get("args")
            if args is not None:
                try:
                    lines.append(f"args {json.dumps(args, sort_keys=True)}")
                except (TypeError, ValueError):
                    lines.append(f"args {args!r}")
            result = payload.get("result")
            if result is not None:
                lines.append(f"result {result}")  # full, untruncated
            if payload.get("truncated"):
                lines.append("truncated yes (stored payload hit the 256 KiB cap)")
        if rec.variants:
            # Variant-set contents attach at TURN level to every assistant
            # record of that turn -- the label says so explicitly.
            lines.append("superseded variants (turn-level)")
            lines.extend(
                f"  {i}. {content}" for i, content in enumerate(rec.variants, start=1)
            )
        return "\n".join(lines)

    def _inspector_text_for_turn(self, turn_id: str) -> str:
        number = self._turn_numbers.get(turn_id, 0)
        state = "collapsed" if turn_id in self._collapsed else "expanded"
        count = next(
            (len(turn.records) for turn in self._turns if turn.turn_id == turn_id), 0
        )
        return f"Turn {number} · {count} records · {state} · id {turn_id}"

    def _cursor_key(self) -> str | None:
        if not self._visible_keys:
            return None
        table = self.query_one("#trajectory-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._visible_keys):
            return self._visible_keys[row]
        return None

    def _inspect_cursor(self) -> None:
        key = self._cursor_key()
        if key is None:
            return
        if key == LOAD_EARLIER_ROW_KEY:
            self.action_load_earlier()
            return
        if key in self._row_turn_ids:
            text = self._inspector_text_for_turn(self._row_turn_ids[key])
        else:
            record = self._row_records.get(key)
            if record is None:
                return
            text = self._inspector_text_for_record(record)
        self._show_inspector(text)

    def _show_inspector(self, text: str) -> None:
        inspector = self.query_one("#trajectory-inspector", Static)
        inspector.update(text)
        inspector.display = True

    def _refresh_inspector(self) -> None:
        key = self._cursor_key()
        if key is None:
            return
        if key == LOAD_EARLIER_ROW_KEY:
            self._show_inspector(
                f"{self._hidden_earlier} older records not loaded — press e"
            )
        elif key in self._row_turn_ids:
            self._show_inspector(self._inspector_text_for_turn(self._row_turn_ids[key]))
        else:
            record = self._row_records.get(key)
            if record is not None:
                self._show_inspector(self._inspector_text_for_record(record))

    # -- events -----------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        """Live search: every keystroke re-filters the ledger."""
        if event.input.id != "trajectory-search":
            return
        self._query = event.value.strip()
        self._render_ledger()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box applies the filter and returns to the table."""
        if event.input.id != "trajectory-search":
            return
        self.query_one("#trajectory-table", DataTable).focus()

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a cursor row opens the inspector (the table consumes enter)."""
        event.stop()
        self._inspect_cursor()

    @on(DataTable.RowHighlighted)
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Cursor moves refresh an OPEN inspector (live follow)."""
        event.stop()
        if self.query_one("#trajectory-inspector", Static).display:
            self._refresh_inspector()

    # -- actions (ADR-031: single-letter htop-style) ----------------------------

    def action_dismiss(self) -> None:
        """Safe escape: blur the search box first, dismiss the modal second."""
        search = self.query_one("#trajectory-search", Input)
        if search.has_focus:
            self.query_one("#trajectory-table", DataTable).focus()
            return
        self.dismiss(None)

    def action_toggle_turn(self) -> None:
        """`t`: collapse/expand the turn under (or owning) the cursor row."""
        key = self._cursor_key()
        if key is None or key == LOAD_EARLIER_ROW_KEY:
            return
        turn_id = self._row_turn_ids.get(key)
        if turn_id is None:
            record = self._row_records.get(key)
            if record is None:
                return
            turn_id = record.turn_id
        if turn_id in self._collapsed:
            self._collapsed.discard(turn_id)
        else:
            self._collapsed.add(turn_id)
        self._render_ledger()

    def action_toggle_inspector(self) -> None:
        """`i`: show/hide the inspector pane."""
        inspector = self.query_one("#trajectory-inspector", Static)
        if inspector.display:
            inspector.display = False
        else:
            self._refresh_inspector()
            inspector.display = True

    def action_load_earlier(self) -> None:
        """`e`: mount one more page of older records (guidance when exhausted)."""
        if self._hidden_earlier <= 0:
            self.app.notify("All records are already loaded.", severity="information")
            return
        self._visible_count += PAGE_SIZE
        self._render_ledger()

    def action_focus_search(self) -> None:
        """`/`: focus the search box."""
        self.query_one("#trajectory-search", Input).focus()

    def action_inspect_cursor_row(self) -> None:
        """`enter`: open the inspector on the cursor row.

        The focused DataTable consumes enter itself (its own binding posts
        ``RowSelected``, handled above); this screen-level binding covers
        the case where focus sits elsewhere on the screen.
        """
        self._inspect_cursor()
