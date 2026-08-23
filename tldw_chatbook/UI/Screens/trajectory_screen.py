"""Console trajectory (trace) screen: ledger, inspector, search, pagination.

Modal over the Console (task-4 of the trajectory view; spec:
``Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md``,
"UI — Trajectory screen"). Read-only: it renders a ``TrajectorySnapshot``
from the pure projection (``Chat/trajectory.py``) and never queries the
DB itself.

Layout (vertical): title line, search ``Input``, brushable timeline strip
(:class:`TrajectoryTimeline`, task-16315 -- always mounted so its zoom
state survives; the widget itself collapses to a ``no timing data``
placeholder when the snapshot has no timing), ``DataTable`` ledger
(``cursor_type="row"``), inspector ``VerticalScroll`` (hidden until toggled),
footer hints line.

Timeline <-> ledger integration:

- The strip is fed the same snapshot as the ledger, live refreshes
  included (``_apply_live_snapshot``).
- A brush range filters the ledger to records ACTIVE in the range (the
  widget model's ``records_in_range`` semantics), composed with the
  search query (AND); brush=None clears only the time filter. The
  strip's caption is the brush status note (range + active count) --
  deliberately not duplicated elsewhere.
- Bar click moves the ledger cursor to that record (growing the mounted
  window if the record is only paginated out); the ledger cursor
  highlights the bar via ``set_selected``. The search filter is never
  touched by the timeline.

Ledger semantics pinned by the projection's contracts:

- ``TrajectoryRecord.seq`` is the 1-based display position only. Stable
  ``event_id`` owns selection; legacy records use a deterministic fallback.
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

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Static

from tldw_chatbook.Chat.trajectory import (
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    KIND_USER_FEEDBACK,
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
)
from tldw_chatbook.Chat.trajectory_import import (
    TrajectoryImportError,
    load_trajectory_snapshot,
)
from tldw_chatbook.UI.Widgets.trajectory_timeline import TrajectoryTimeline

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

_NARROW_COLUMNS = (
    ("#", 6),
    ("Event", 14),
    ("Summary", 26),
    ("State", 10),
)
_COMPACT_COLUMNS = _NARROW_COLUMNS + (("Tokens", 10), ("Duration", 10))
_WIDE_COLUMNS = (
    ("#", 4),
    ("Event", 14),
    ("Summary", 26),
    ("State", 10),
    ("In", 6),
    ("Cache", 7),
    ("Out", 6),
    ("Duration", 10),
    ("Start", 9),
    ("Done", 9),
)

_STATUS_LABELS = {
    "accepted": "Accepted",
    "cancelled": "Cancelled",
    "completed": "Done",
    "failed": "Failed",
    "in_progress": "Running",
    "pending": "Pending",
    "rejected": "Rejected",
    "running": "Running",
    "succeeded": "Done",
}

_INCOMPLETE_FIELD_STATES = frozenset(
    {"capture_failed", "legacy_missing", "missing", "source_unavailable"}
)


class _TraceInspector(VerticalScroll):
    """Native focusable detail viewport that keeps its fold cue current."""

    def __init__(self, *, on_scroll_changed: Callable[[], None]) -> None:
        super().__init__(
            Static("", id="trajectory-inspector-content", markup=False),
            id="trajectory-inspector",
            can_focus=True,
        )
        self._on_scroll_changed = on_scroll_changed

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if old_value != new_value:
            self._on_scroll_changed()


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
        Binding("f", "resume_follow", "Follow"),
        Binding("d", "toggle_detail_full", "Full detail"),
        Binding("r", "retry", "Retry"),
        Binding("x", "clear_filters", "Clear filters"),
        Binding("o", "open_trace", "Open trace"),
    ]

    #: Footer hints, 1:1 with the non-escape BINDINGS (ADR-031; tested).
    #: ``f follow`` is only RENDERED on live screens (``_refresh_hints``).
    TRAJECTORY_SHORTCUTS = (
        ("enter", "inspect"),
        ("t", "collapse"),
        ("i", "inspector"),
        ("/", "search"),
        ("e", "earlier"),
        ("f", "follow"),
        ("d", "full detail"),
        ("r", "retry"),
        ("x", "clear filters"),
        ("o", "open trace"),
    )

    BUNDLED_CSS = """
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
    #trajectory-state {
        height: 2;
        color: $text-muted;
    }
    #trajectory-search {
        height: 3;
    }
    #trajectory-table {
        height: 1fr;
        min-height: 3;
    }
    #trajectory-inspector {
        height: 40%;
        max-height: 40%;
        border-top: solid $primary;
        padding: 0 1;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    #trajectory-inspector:focus {
        border-top: heavy $accent;
    }
    #trajectory-inspector-content {
        width: 100%;
        height: auto;
    }
    #trajectory-inspector-overflow {
        height: 1;
        color: $text-muted;
    }
    TrajectoryScreen.trace-detail-full #trajectory-search,
    TrajectoryScreen.trace-detail-full #trajectory-timeline,
    TrajectoryScreen.trace-detail-full #trajectory-table {
        display: none;
    }
    TrajectoryScreen.trace-inspector-open #trajectory-timeline {
        display: none;
    }
    TrajectoryScreen.trace-detail-full #trajectory-inspector {
        height: 1fr;
        max-height: 100%;
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
        revision_provider: Callable[[], int] | None = None,
        snapshot_builder: Callable[[], TrajectorySnapshot] | None = None,
        shared_trace: bool = False,
    ) -> None:
        """Store the projection output; all rendering happens on mount.

        Args:
            snapshot: The ``derive_trajectory`` output to render.
            screen_title: Display title (typically the conversation name).
            conversation_id: Conversation id, shown in inspector metadata.
            revision_provider: Live-mode callable returning the store's
                payload revision for this conversation (task-5 tail-follow).
                When provided together with ``snapshot_builder`` the screen
                polls it every 0.5s and rebuilds on change.
            snapshot_builder: Live-mode callable rebuilding the snapshot
                from current persisted state; run in a worker thread.
            shared_trace: Whether this snapshot came from an imported shared
                trace and must be visibly presented as read-only.
        """
        super().__init__()
        self._snapshot = snapshot
        self._screen_title = screen_title
        self._conversation_id = conversation_id
        self._revision_provider = revision_provider
        self._snapshot_builder = snapshot_builder
        self._shared_trace = shared_trace
        self._last_revision: int | None = None
        #: Tail-follow: True while the reader is at the bottom; scrolling
        #: up suspends it, ``f`` re-enables.
        self._follow = True
        #: Explicit resume grace: geometry-polling must not cancel an
        #: in-flight ``f`` before its deferred ``scroll_end`` lands.
        self._follow_grace_until = 0.0
        self._turns: tuple[TrajectoryTurn, ...] = snapshot.turns
        self._turn_numbers: dict[str, int] = {
            turn.turn_id: index + 1 for index, turn in enumerate(self._turns)
        }
        self._collapsed: set[str] = set()
        self._query = ""
        #: Active brush time range from the timeline strip (None = no
        #: time filter); mirrors the widget's brush state.
        self._brush_range: tuple[float, float] | None = None
        self._width_tier: str | None = None
        self._detail_full = False
        self._inspector_target_key: str | None = None
        self._loading = False
        self._failure: str | None = None
        self._retry_target: str | None = None
        #: The always-mounted timeline strip (created here so filters,
        #: pagination growth and selection sync can reach it pre-mount).
        self._timeline = TrajectoryTimeline(id="trajectory-timeline")
        #: Number of newest records mounted (grows one page per `e` press).
        self._visible_count = min(self._total_records, PAGE_SIZE)
        #: Row key -> record (None for header/control rows), in row order.
        self._row_records: dict[str, TrajectoryRecord | None] = {}
        self._row_turn_ids: dict[str, str] = {}
        self._visible_keys: list[str] = []
        self._ledger_rendered = False
        self._record_keys: dict[int, str] = {}
        self._rebuild_record_keys()
        #: Bumped by every ledger render; a worker-built page carries the
        #: generation it was built at and is DROPPED if a newer render
        #: landed first (typing a search or pressing t/e mid-build must
        #: never be overwritten by a stale off-thread snapshot).
        self._render_generation = 0
        self._pending_restore_key: str | None = None
        #: False once unmounted: a worker-built ledger arriving late is
        #: dropped instead of poking a dead screen. (``Widget.is_mounted``
        #: is still False DURING ``on_mount``, so the flag is explicit.)
        self._alive = True

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="trajectory-screen"):
            yield Static(self._title_text(), id="trajectory-title", markup=False)
            yield Static("", id="trajectory-state", markup=False)
            yield Input(placeholder="Search trace events…", id="trajectory-search")
            yield self._timeline  # always mounted: zoom state survives
            table = DataTable(id="trajectory-table", cursor_type="row")
            table.cell_padding = 0
            yield table
            inspector = _TraceInspector(on_scroll_changed=self._refresh_inspector_cue)
            inspector.display = False  # hidden until toggled (spec)
            yield inspector
            cue = Static("", id="trajectory-inspector-overflow", markup=False)
            cue.display = False
            yield cue
            yield Static("", id="trajectory-hints", markup=False)

    def on_mount(self) -> None:
        table = self.query_one("#trajectory-table", DataTable)
        self._configure_columns(force=True)
        self._timeline.set_snapshot(self._snapshot)
        self._refresh_state()
        self._refresh_hints()
        if self._total_records > WORKER_THRESHOLD:
            # Large conversation: build the row specs off the UI thread.
            self._loading = True
            self._refresh_state()
            self.run_worker(self._render_worker, thread=True, group="trajectory-ledger")
        else:
            self._render_ledger()
        table.focus()
        if self._revision_provider is not None and self._snapshot_builder is not None:
            try:
                self._last_revision = self._revision_provider()
            except Exception:  # noqa: BLE001 - provider is external state
                self._last_revision = None
            self.set_interval(0.5, self._poll_revision)

    def on_unmount(self) -> None:
        self._alive = False

    def on_resize(self) -> None:
        """Rebuild columns only when the screen crosses a responsive tier."""

        self._configure_columns()
        try:
            self.query_one("#trajectory-inspector", VerticalScroll)
            self._schedule_inspector_cue()
        except Exception:  # noqa: BLE001 - resize may precede composition
            pass

    @staticmethod
    def _tier_for_width(width: int) -> str:
        if width >= 120:
            return "wide"
        if width >= 100:
            return "compact"
        return "narrow"

    @staticmethod
    def _columns_for_tier(tier: str) -> tuple[tuple[str, int], ...]:
        if tier == "wide":
            return _WIDE_COLUMNS
        if tier == "compact":
            return _COMPACT_COLUMNS
        return _NARROW_COLUMNS

    def _configure_columns(self, *, force: bool = False) -> None:
        """Install the current tier's columns without churning within a tier."""

        tier = self._tier_for_width(self.size.width)
        if not force and tier == self._width_tier:
            return
        try:
            table = self.query_one("#trajectory-table", DataTable)
        except Exception:  # noqa: BLE001 - resize may arrive before mount
            self._width_tier = tier
            return
        previous_key = self._cursor_key() if self._visible_keys else None
        table.clear(columns=True)
        for label, width in self._columns_for_tier(tier):
            table.add_column(label, width=width)
        self._width_tier = tier
        if self._visible_keys:
            self._pending_restore_key = previous_key
            self._render_ledger()

    def _rebuild_record_keys(self) -> None:
        """Build collision-safe deterministic row identities for this snapshot."""

        keys: dict[int, str] = {}
        occurrences: dict[str, int] = {}
        for turn in self._turns:
            for record in turn.records:
                if record.event_id:
                    base = record.event_id
                else:
                    identity = asdict(record)
                    identity.pop("seq", None)
                    identity.pop("event_id", None)
                    material = json.dumps(
                        identity,
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    )
                    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
                    base = f"legacy:{digest}"
                occurrence = occurrences.get(base, 0)
                occurrences[base] = occurrence + 1
                key = base if occurrence == 0 else f"{base}:collision:{occurrence + 1}"
                keys[id(record)] = key
        self._record_keys = keys

    def _record_key(self, record: TrajectoryRecord) -> str:
        return self._record_keys[id(record)]

    @property
    def _snapshot_is_incomplete(self) -> bool:
        for turn in self._turns:
            for record in turn.records:
                if record.kind == "capture_failed":
                    return True
                if any(
                    state in _INCOMPLETE_FIELD_STATES
                    for state in record.field_states.values()
                ):
                    return True
        return False

    @property
    def _snapshot_has_timing(self) -> bool:
        return any(
            record.step_started_at is not None
            for turn in self._turns
            for record in turn.records
        )

    @property
    def _visible_record_count(self) -> int:
        return sum(record is not None for record in self._row_records.values())

    def _refresh_state(self) -> None:
        """Expose orthogonal mode, completeness, filter, timing, and recovery."""

        total = self._total_records
        filtering = bool(self._query) or self._brush_range is not None
        visible = self._visible_record_count if self._ledger_rendered else total
        parts: list[str] = []
        if self._shared_trace:
            parts.append("READ-ONLY SHARED TRACE")
        if self._snapshot_builder is not None:
            parts.extend(("LIVE", "FOLLOWING" if self._follow else "PAUSED"))
            if not self._follow:
                parts.append("f resume")
        if self._failure is not None:
            parts.extend(("FAILED", self._failure, "r retry"))
        elif self._loading:
            parts.extend(("LOADING", "Building trace ledger…"))
        elif not parts and total:
            parts.append("READY")
        if total == 0:
            parts.extend(("EMPTY", "No trace events yet", "o open trace"))
        if self._snapshot_is_incomplete:
            parts.append("INCOMPLETE")
        if filtering:
            if visible:
                parts.extend(("FILTERED", f"{visible}/{total} events"))
            else:
                parts.extend(("NO MATCHES", f"0/{total} events", "x clear filters"))
        elif total:
            parts.append(f"{total} events")
        if total and not self._snapshot_has_timing:
            parts.extend(("NO TIMING", "Duration unavailable"))
        try:
            self.query_one("#trajectory-state", Static).update(" · ".join(parts))
        except Exception:  # noqa: BLE001 - state can update before mount
            pass

    # -- live tail-follow (task-5) --------------------------------------------

    def _poll_revision(self) -> None:
        """Interval tick: rebuild the snapshot when the store revision moves."""
        if not self._alive or self._snapshot_builder is None:
            return
        if self._revision_provider is None:
            return
        self._sync_follow_from_scroll()
        try:
            revision = self._revision_provider()
        except Exception:  # noqa: BLE001 - provider is external state
            return
        if revision == self._last_revision:
            return
        self._last_revision = revision
        # exclusive: a slow rebuild from an older revision cannot be
        # overtaken and land after a newer one started; the revision guard
        # below additionally drops results built for a stale revision.
        self.run_worker(
            lambda: self._live_rebuild_worker(revision),
            thread=True,
            group="trajectory-live",
            exclusive=True,
        )

    def _live_rebuild_worker(self, revision: int | None) -> None:
        """Worker-thread half of the live rebuild (video-player pattern)."""
        builder = self._snapshot_builder
        if builder is None:
            return
        try:
            snapshot = builder()
        except Exception as exc:  # noqa: BLE001 - worker boundary
            logger.warning(
                "Trajectory live rebuild failed: component=trajectory_screen "
                "error_type={}",
                type(exc).__name__,
            )
            try:
                self.app.call_from_thread(
                    self._set_failure,
                    "Live refresh unavailable.",
                    "live",
                )
            except Exception:  # noqa: BLE001 - screen may have closed
                pass
            return
        try:
            self.app.call_from_thread(self._apply_live_snapshot, snapshot, revision)
        except Exception:  # noqa: BLE001 - worker boundary
            return

    def _apply_live_snapshot(
        self, snapshot: TrajectorySnapshot, revision: int | None = None
    ) -> None:
        """Swap in a rebuilt snapshot, preserving reader state; follow tail.

        ``revision`` is the store revision the snapshot was built for; a
        result arriving after a NEWER revision was observed is dropped so
        out-of-order workers can never regress the ledger.
        """
        if not self._alive:
            return
        if revision is not None and revision != self._last_revision:
            return
        selected_key = self._cursor_key()
        self._snapshot = snapshot
        self._turns = snapshot.turns
        self._rebuild_record_keys()
        if self._retry_target == "live":
            self._loading = False
            self._failure = None
            self._retry_target = None
        self._turn_numbers = {
            turn.turn_id: index + 1 for index, turn in enumerate(self._turns)
        }
        # Feed the strip the same data the ledger renders. set_snapshot
        # resets the widget's brush/selection WITHOUT posting, so keep an
        # active brush alive across the swap (a 0.5s revision tick must
        # not destroy it): re-apply it iff it still intersects the new
        # domain (appends only grow it), else clear both sides so the
        # ledger's time filter can never outlive the visual brush.
        self._timeline.set_snapshot(snapshot)
        if self._brush_range is not None:
            lo, hi = self._brush_range
            domain = self._timeline.model.domain
            if domain is not None and lo <= domain[1] and hi >= domain[0]:
                self._timeline.apply_brush(self._brush_range)
            else:
                self._brush_range = None
        # Keep collapsed turns and the search query; never shrink the window.
        self._visible_count = max(
            self._visible_count, min(self._total_records, PAGE_SIZE)
        )
        if selected_key is not None and not self._follow:
            flat = [record for turn in self._turns for record in turn.records]
            for index, record in enumerate(flat):
                if self._record_key(record) == selected_key:
                    self._visible_count = max(self._visible_count, len(flat) - index)
                    self._pending_restore_key = selected_key
                    break
        self._render_ledger()
        try:
            self.query_one("#trajectory-title", Static).update(self._title_text())
        except Exception:  # noqa: BLE001 - pre-mount refresh
            pass
        self._refresh_hints()
        self._refresh_state()
        if self._follow:
            try:
                table = self.query_one("#trajectory-table", DataTable)
                # The re-render resets scroll geometry; scrolling must land
                # after the table re-lays out, not at the stale range.
                table.call_after_refresh(table.scroll_end, animate=False)
            except Exception:  # noqa: BLE001 - pre-mount refresh
                pass

    def _sync_follow_from_scroll(self) -> None:
        """Suspend follow while the reader is off the bottom edge.

        Pull-based (checked on every poll tick) rather than event-based:
        DataTable scroll geometry is authoritative and needs no message
        plumbing, and a table that cannot scroll at all keeps following.
        """
        try:
            table = self.query_one("#trajectory-table", DataTable)
        except Exception:  # noqa: BLE001 - pre-mount refresh
            return
        if table.max_scroll_y <= 0:
            return
        if time.monotonic() < self._follow_grace_until:
            return
        following = table.scroll_y >= table.max_scroll_y - 1
        if following != self._follow:
            self._follow = following
            self._refresh_state()

    def action_resume_follow(self) -> None:
        """Re-enable tail-follow and jump to the newest records (``f``)."""
        self._follow = True
        self._follow_grace_until = time.monotonic() + 1.0
        try:
            table = self.query_one("#trajectory-table", DataTable)
            table.call_after_refresh(table.scroll_end, animate=False)
        except Exception:  # noqa: BLE001 - pre-mount refresh
            pass
        self._refresh_state()

    def _title_text(self) -> str:
        parts = ["Trace"]
        if self._screen_title:
            parts.append(self._screen_title)
        parts.append(f"{self._total_records} events")
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
            try:
                self.app.call_from_thread(
                    self._set_failure,
                    "Trace ledger could not be rendered.",
                    "render",
                )
            except Exception:  # noqa: BLE001 - screen may have closed
                pass

    def _set_failure(self, message: str, retry_target: str) -> None:
        """Show a payload-safe failure with one working recovery action."""

        if not self._alive:
            return
        self._loading = False
        self._failure = message
        self._retry_target = retry_target
        self._refresh_state()
        self._refresh_hints()

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
                    self._primary_cells(
                        "",
                        "Earlier events",
                        f"{self._hidden_earlier} older events — press e",
                        "Available",
                    ),
                )
            )

        query = self._query.lower()
        brush_seqs = self._brush_active_seqs()
        open_turn: TrajectoryTurn | None = None
        turn_records: list[TrajectoryRecord] = []
        for turn, record in self._flat_slice():
            if open_turn is not None and turn.turn_id != open_turn.turn_id:
                specs.extend(
                    self._turn_row_specs(open_turn, turn_records, query, brush_seqs)
                )
                turn_records = []
            open_turn = turn
            turn_records.append(record)
        if open_turn is not None:
            specs.extend(
                self._turn_row_specs(open_turn, turn_records, query, brush_seqs)
            )
        return specs

    def _brush_active_seqs(self) -> frozenset[int] | None:
        """Seqs of records ACTIVE in the brush; ``None`` when no brush.

        Reuses the timeline model's ``records_in_range`` semantics (a
        record is active when its interval intersects the range; untimed
        records are never in the model, so a brush hides them) instead
        of reimplementing the interval math here.
        """
        if self._brush_range is None:
            return None
        lo, hi = self._brush_range
        return frozenset(
            record.seq for record in self._timeline.model.records_in_range(lo, hi)
        )

    def _turn_row_specs(
        self,
        turn: TrajectoryTurn,
        records: list[TrajectoryRecord],
        query: str,
        brush_seqs: frozenset[int] | None = None,
    ) -> list[tuple[str, tuple[Text, ...]]]:
        """Header row + child rows for one turn under the current filter.

        Search semantics (spec): child rows match on their own text; the
        turn header survives iff any child matches. A search overrides
        collapse (searching reveals), otherwise collapsed turns show the
        header only. A brush composes with the search (AND) and follows
        the same header-survival and reveal rules.
        """
        matching = [
            rec
            for rec in records
            if self._record_matches(rec, query)
            and (brush_seqs is None or rec.seq in brush_seqs)
        ]
        filtering = bool(query) or brush_seqs is not None
        if filtering and not matching:
            return []  # nothing in this turn is visible: header included, hidden
        header_key = f"turn:{turn.turn_id}"
        collapsed = turn.turn_id in self._collapsed
        number = self._turn_numbers.get(turn.turn_id, 0)
        marker = "▸" if collapsed else "▾"
        label = Text(f"{marker} Turn {number} · {len(records)} records", style="bold")
        specs = [
            (
                header_key,
                self._primary_cells(
                    "", "Turn", label, "Collapsed" if collapsed else "Expanded"
                ),
            )
        ]
        if collapsed and not filtering:
            return specs
        for rec in matching:
            specs.append((self._record_key(rec), self._record_cells(rec)))
        return specs

    def _primary_cells(
        self,
        identity: str | Text,
        event: str | Text,
        summary: str | Text,
        state: str | Text,
    ) -> tuple[Text, ...]:
        """Build a tier-sized row with primary facts and blank metrics."""

        primary = tuple(
            value if isinstance(value, Text) else Text(value)
            for value in (identity, event, summary, state)
        )
        tier = self._width_tier or self._tier_for_width(self.size.width)
        extra = len(self._columns_for_tier(tier)) - len(primary)
        return primary + tuple(Text("") for _ in range(extra))

    @staticmethod
    def _record_state(record: TrajectoryRecord) -> str:
        if record.status:
            return _STATUS_LABELS.get(
                record.status,
                record.status.replace("_", " ").capitalize(),
            )
        if record.kind == "capture_failed":
            return "Failed"
        if record.completed_at is not None:
            return "Complete"
        if record.step_started_at is not None:
            return "Observed"
        return "Recorded"

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
        primary = (
            Text(str(rec.seq)),
            Text(
                rec.label or rec.kind.replace("_", " ").strip().capitalize() or "Event"
            ),
            content,
            Text(self._record_state(rec)),
        )
        tier = self._width_tier or self._tier_for_width(self.size.width)
        duration = _fmt_span(rec.step_started_at, rec.completed_at) or "—"
        if tier == "narrow":
            return primary
        if tier == "compact":
            total = "—" if rec.usage is None else str(rec.usage.total_tokens)
            return primary + (Text(total), Text(duration))
        return primary + (
            Text(tokens[0] or "—"),
            Text(tokens[1] or "—"),
            Text(tokens[2] or "—"),
            Text(duration),
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
        previous_key = self._pending_restore_key
        self._pending_restore_key = None
        if previous_key is None and self._visible_keys:
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
        self._ledger_rendered = True
        for turn in self._turns:
            for rec in turn.records:
                record_key = self._record_key(rec)
                if record_key in self._row_records:
                    self._row_records[record_key] = rec
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
        self._sync_timeline_selection()
        if generation is not None:
            self._loading = False
            if self._retry_target == "render":
                self._failure = None
                self._retry_target = None
        self._refresh_state()
        if self.query_one("#trajectory-inspector", VerticalScroll).display:
            self._refresh_inspector()

    # -- footer hints ------------------------------------------------------

    def _refresh_hints(self) -> None:
        """Render the hints line: 1:1 with BINDINGS minus what has no target.

        The ``e earlier`` hint drops while no older records remain (the
        ADR-031 task-1340 refinement: advertised == working in the active
        context; pressing ``e`` then answers with guidance instead).
        """
        try:
            inspector_open = self.query_one(
                "#trajectory-inspector", VerticalScroll
            ).display
        except Exception:  # noqa: BLE001 - pre-mount refresh
            inspector_open = False
        has_cursor_target = bool(self._visible_keys)
        has_turn_target = any(key != LOAD_EARLIER_ROW_KEY for key in self._visible_keys)
        filtering = bool(self._query) or self._brush_range is not None
        if self._detail_full:
            pairs = [("i", "close"), ("d", "split view")]
            if self._failure is not None:
                pairs.append(("r", "retry"))
            if filtering:
                pairs.append(("x", "clear filters"))
        else:
            pairs = [
                (key, "close" if key == "i" and inspector_open else label)
                for key, label in self.TRAJECTORY_SHORTCUTS
                if (key not in {"enter", "i"} or has_cursor_target)
                and (key != "t" or has_turn_target)
                and (key != "e" or self._hidden_earlier > 0)
                and (key != "f" or self._snapshot_builder is not None)
                and (key != "d" or inspector_open)
                and (key != "r" or self._failure is not None)
                and (key != "x" or filtering)
            ]
        text = " · ".join(f"{key} {label}" for key, label in pairs)
        try:
            self.query_one("#trajectory-hints", Static).update(text)
        except Exception:  # noqa: BLE001 - pre-mount refresh
            pass

    # -- inspector -----------------------------------------------------------

    def _inspector_text_for_record(self, rec: TrajectoryRecord) -> str:
        lines = [
            f"#{rec.seq} "
            f"{rec.label or rec.kind.replace('_', ' ').strip().capitalize() or 'Event'} "
            f"· turn {rec.turn_id}"
        ]
        event_id = self._record_key(rec)
        lines.append(f"event id {event_id} · raw kind {rec.kind}")
        conversation_id = self._conversation_id or rec.conversation_id
        if conversation_id:
            lines.append(f"conversation {conversation_id}")
        if rec.message_id:
            lines.append(f"message {rec.message_id}")
        if rec.source_seq is not None:
            lines.append(f"source sequence {rec.source_seq}")
        if rec.status:
            lines.append(f"status {rec.status}")
        if rec.actor_kind or rec.actor_id:
            lines.append(
                "actor "
                + " ".join(part for part in (rec.actor_kind, rec.actor_id) if part)
            )
        if rec.run_id:
            lines.append(f"run {rec.run_id}")
        if rec.parent_event_id:
            lines.append(f"parent event {rec.parent_event_id}")
        if rec.source_event_id:
            lines.append(f"source event {rec.source_event_id}")
        if rec.replacement_event_id:
            lines.append(f"replacement event {rec.replacement_event_id}")
        if rec.observed_at is not None:
            lines.append(f"observed {_fmt_clock(rec.observed_at)}")
        if rec.field_states:
            lines.append(
                "field states "
                + json.dumps(rec.field_states, sort_keys=True, ensure_ascii=False)
            )
        if rec.sensitivity:
            lines.append(f"sensitivity {rec.sensitivity}")
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
        if payload and rec.kind == KIND_USER_FEEDBACK:
            # task-17169: feedback payloads are action/quote/comment, none of
            # the tool keys below. Falling through would print `tool —` and
            # drop the record's entire content.
            lines.append(f"feedback {payload.get('action') or '—'}")
            quote = payload.get("quote")
            if quote:
                lines.append(f"quote {quote}")  # full, untruncated
            comment = payload.get("comment")
            if comment:
                lines.append(f"comment {comment}")
        elif payload and rec.kind in {KIND_TOOL_CALL, KIND_TOOL_RESULT}:
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
            if payload.get("redacted"):
                lines.append(
                    "result redacted (shared trace keeps payload previews only)"
                )
        elif payload:
            try:
                serialized = json.dumps(
                    payload, sort_keys=True, ensure_ascii=False, default=str
                )
            except (TypeError, ValueError):
                serialized = repr(payload)
            lines.append(f"payload {serialized}")
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
        text = f"Turn {number} · {count} events · {state} · id {turn_id}"
        if self._conversation_id:
            text += f"\nconversation {self._conversation_id}"
        return text

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
        self._show_inspector(text, focus=True, target_key=key)

    def _show_inspector(
        self, text: str, *, focus: bool = False, target_key: str | None = None
    ) -> None:
        inspector = self.query_one("#trajectory-inspector", VerticalScroll)
        content = self.query_one("#trajectory-inspector-content", Static)
        target_changed = target_key != self._inspector_target_key
        content.update(text)
        self._inspector_target_key = target_key
        inspector.display = True
        self.add_class("trace-inspector-open")
        self.query_one("#trajectory-inspector-overflow", Static).display = True
        if target_changed:
            inspector.scroll_home(animate=False)
        if focus:
            inspector.focus()
        self._schedule_inspector_cue()
        self._refresh_hints()

    def _refresh_inspector(self) -> None:
        key = self._cursor_key()
        if key is None:
            return
        if key == LOAD_EARLIER_ROW_KEY:
            self._show_inspector(
                f"{self._hidden_earlier} older records not loaded — press e",
                target_key=key,
            )
        elif key in self._row_turn_ids:
            self._show_inspector(
                self._inspector_text_for_turn(self._row_turn_ids[key]), target_key=key
            )
        else:
            record = self._row_records.get(key)
            if record is not None:
                self._show_inspector(
                    self._inspector_text_for_record(record), target_key=key
                )

    def _refresh_inspector_cue(self) -> None:
        """Show the fold cue only while inspector content remains below."""

        try:
            inspector = self.query_one("#trajectory-inspector", VerticalScroll)
            cue = self.query_one("#trajectory-inspector-overflow", Static)
        except Exception:  # noqa: BLE001 - layout may be tearing down
            return
        if not inspector.display:
            cue.display = False
            cue.update("")
            return
        cue.display = True
        has_more = (
            inspector.max_scroll_y > 0 and inspector.scroll_y < inspector.max_scroll_y
        )
        cue.update("▼ more — scroll…" if has_more else "")

    def _schedule_inspector_cue(self) -> None:
        """Reconcile fold geometry after the class-driven layout has settled."""

        self.call_after_refresh(self._refresh_inspector_cue)
        # Opening detail also hides the timeline, which may settle in the next
        # compositor cycle at a responsive boundary. This bounded second pass
        # prevents a one-frame missing cue without polling or repaint loops.
        self.set_timer(0.01, self._refresh_inspector_cue)

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
        table = self.query_one("#trajectory-table", DataTable)
        if table.display:
            table.focus()

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a cursor row opens the inspector (the table consumes enter)."""
        event.stop()
        self._inspect_cursor()

    @on(DataTable.RowHighlighted)
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Cursor moves refresh an OPEN inspector (live follow)."""
        event.stop()
        self._sync_timeline_selection()
        if self.query_one("#trajectory-inspector", VerticalScroll).display:
            self._refresh_inspector()

    # -- timeline integration (task-16315) -----------------------------------

    def _sync_timeline_selection(self) -> None:
        """Highlight the bar for the record under the ledger cursor.

        Header/control rows (and a missing record) clear the highlight;
        ``set_selected`` is pull-only so this cannot echo a bar-click
        event back into the ledger.
        """
        key = self._cursor_key()
        record = self._row_records.get(key) if key is not None else None
        self._timeline.set_selected(record.seq if record is not None else None)

    @on(TrajectoryTimeline.TrajectoryBrushChanged)
    def _on_brush_changed(
        self, event: TrajectoryTimeline.TrajectoryBrushChanged
    ) -> None:
        """Brush range filters the ledger (AND with the search query)."""
        self._brush_range = event.brush_range
        self._render_ledger()

    @on(TrajectoryTimeline.TrajectoryBarSelected)
    def _on_bar_selected(self, event: TrajectoryTimeline.TrajectoryBarSelected) -> None:
        """Bar click: move the ledger cursor to that record's row.

        The widget clears its own brush before posting, so mirror that
        here (the time filter drops; the SEARCH stays). If the record is
        only paginated out, grow the mounted window to cover it; if the
        search still hides it, that is a no-op on the cursor.
        """
        if self._brush_range is not None:
            self._brush_range = None
            self._render_ledger()
        flat = [record for turn in self._turns for record in turn.records]
        try:
            selected_record = next(
                record for record in flat if record.seq == event.record_key
            )
        except StopIteration:
            return  # unknown record (stale bar from a live snapshot): no-op
        key = self._record_key(selected_record)
        if key in self._visible_keys:
            self._move_cursor_to_key(key)
            return
        try:
            flat_index = next(
                i for i, record in enumerate(flat) if record.seq == event.record_key
            )
        except StopIteration:
            return  # unknown record (stale bar from a live snapshot): no-op
        if flat_index < len(flat) - self._visible_count:
            self._visible_count = len(flat) - flat_index
            self._render_ledger()
        if key in self._visible_keys:
            self._move_cursor_to_key(key)

    def _move_cursor_to_key(self, key: str) -> None:
        """Best-effort cursor move to the row with ``key``."""
        try:
            index = self._visible_keys.index(key)
        except ValueError:
            return
        table = self.query_one("#trajectory-table", DataTable)
        try:
            table.move_cursor(row=index, animate=False)
        except Exception as exc:  # noqa: BLE001 - cursor clamp is best-effort
            logger.debug("Trajectory bar-select cursor move skipped: {}", type(exc))

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
        inspector = self.query_one("#trajectory-inspector", VerticalScroll)
        cue = self.query_one("#trajectory-inspector-overflow", Static)
        if inspector.display:
            inspector.display = False
            cue.display = False
            cue.update("")
            self._detail_full = False
            self.remove_class("trace-detail-full")
            self.remove_class("trace-inspector-open")
            self.query_one("#trajectory-table", DataTable).focus()
        else:
            if self._cursor_key() is None:
                return
            self._refresh_inspector()
            inspector.display = True
            cue.display = True
            inspector.focus()
            self._schedule_inspector_cue()
        self._refresh_hints()

    def action_toggle_detail_full(self) -> None:
        """`d`: expand an open inspector to a reversible full-detail pane."""

        inspector = self.query_one("#trajectory-inspector", VerticalScroll)
        if not inspector.display:
            self._inspect_cursor()
            inspector = self.query_one("#trajectory-inspector", VerticalScroll)
        if not inspector.display:
            return
        self._detail_full = not self._detail_full
        self.set_class(self._detail_full, "trace-detail-full")
        inspector.focus()
        self._schedule_inspector_cue()
        self._refresh_hints()

    def action_load_earlier(self) -> None:
        """`e`: mount one more page of older records (guidance when exhausted)."""
        if self._hidden_earlier <= 0:
            self.app.notify("All records are already loaded.", severity="information")
            return
        self._visible_count += PAGE_SIZE
        self._render_ledger()

    def action_focus_search(self) -> None:
        """`/`: focus the search box."""
        search = self.query_one("#trajectory-search", Input)
        if search.display:
            search.focus()

    def action_clear_filters(self) -> None:
        """`x`: clear search and timeline brush as one recovery action."""

        if not self._query and self._brush_range is None:
            return
        self._query = ""
        self.query_one("#trajectory-search", Input).value = ""
        self._brush_range = None
        self._timeline.apply_brush(None)
        self._render_ledger()

    def action_retry(self) -> None:
        """`r`: retry only the failed render or live-refresh operation."""

        target = self._retry_target
        if target is None:
            return
        self._loading = True
        self._refresh_state()
        self._refresh_hints()
        if target == "live" and self._snapshot_builder is not None:
            revision = self._last_revision
            if revision is None and self._revision_provider is not None:
                try:
                    revision = self._revision_provider()
                except Exception:  # noqa: BLE001 - external revision seam
                    revision = None
            self.run_worker(
                lambda: self._live_rebuild_worker(revision),
                thread=True,
                group="trajectory-live",
                exclusive=True,
            )
            return
        self._render_generation += 1
        self.run_worker(self._render_worker, thread=True, group="trajectory-ledger")

    # -- import (task-16320: open shared traces read-only) ----------------------

    async def action_open_trace(self) -> None:
        """`o`: open a shared trajectory trace file as a read-only view.

        The picked file is loaded through the pure import seam (never the
        app DB) and pushed as a NEW ``TrajectoryScreen`` with no
        ``conversation_id`` / live providers -- the imported view is itself
        the trajectory surface, fully read-only (no revision polling).
        Import failures surface as an error notification carrying the
        actionable message from the shared validator.
        """
        path = await self._pick_trace_file()
        if path is None:
            return  # picker dismissed: no-op, stay on the current screen
        try:
            snapshot = load_trajectory_snapshot(path)
        except TrajectoryImportError as exc:
            self.app.notify(str(exc), title="Import failed", severity="error")
            return
        self.app.push_screen(
            TrajectoryScreen(
                snapshot,
                screen_title=f"Shared trace — {path.stem}",
                shared_trace=True,
            )
        )

    async def _pick_trace_file(self) -> Path | None:
        """Open the repo's file picker; ``None`` when dismissed.

        Separate seam so tests can stub the picker (the fspicker modal is
        not pilot-friendly) while the binding/load/push path stays real.
        """
        from tldw_chatbook.Third_Party.textual_fspicker import Filters
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

        picker = EnhancedFileOpen(
            title="Open shared trace",
            filters=Filters(
                ("Trace files", lambda p: p.name.lower().endswith(".json")),
                ("All Files", lambda p: True),
            ),
            context="trajectory_import",
        )
        selected = await self.app.push_screen_wait(picker)
        return None if selected is None else Path(str(selected))

    def action_inspect_cursor_row(self) -> None:
        """`enter`: open the inspector on the cursor row.

        The focused DataTable consumes enter itself (its own binding posts
        ``RowSelected``, handled above); this screen-level binding covers
        the case where focus sits elsewhere on the screen.
        """
        self._inspect_cursor()
