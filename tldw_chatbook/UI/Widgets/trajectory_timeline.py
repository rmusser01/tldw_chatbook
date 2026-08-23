"""Semantic, brushable time-domain strip over a ``TrajectorySnapshot``.

The compact widget renders every timed trajectory record at its observed
interval in one of four named lanes: Input, Model, Tools, and Agents.
Distinct monochrome glyphs identify the lanes; turn and child-agent marks
show grouping without claiming serial causality. Mouse and keyboard both
support event selection, range selection, zoom, and pan. Geometry lives in
:class:`TimelineModel`, a pure class that never touches Textual, while the
widget is a thin event/render shell.

Records with NULL timing are skipped (blank, never fabricated -- dsh
precedent); a snapshot with no usable timing renders a centered
``no timing data`` placeholder row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import Mapping, Sequence

from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.events import MouseEvent
from textual.message import Message
from textual.widget import Widget

from tldw_chatbook.Chat.trajectory import (
    KIND_ASSISTANT,
    KIND_COMPACTION,
    KIND_SYSTEM,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    KIND_USER_FEEDBACK,
    KIND_USER,
    TrajectoryRecord,
    TrajectorySnapshot,
)

__all__ = [
    "LANE_COUNT",
    "STRIP_HEIGHT",
    "TimelineModel",
    "TrajectoryTimeline",
]

#: Lanes stacked top (0) to bottom in the strip.
LANE_COUNT = 4
LANE_NAMES = ("Input", "Model", "Tools", "Agents")
LANE_LABEL_WIDTH = max(map(len, LANE_NAMES)) + 1

#: Total rendered lines: LANE_COUNT lane rows + an axis row + the caption.
STRIP_HEIGHT = LANE_COUNT + 2

#: Domain padding on each side, as a fraction of the data time span.
DOMAIN_PADDING = 0.05

#: Wheel/keyboard zoom factor (>1 zooms in).
ZOOM_FACTOR = 1.25

#: Keyboard pan step, as a fraction of the current window span.
PAN_FRACTION = 0.25

#: Lane glyphs are the primary monochrome event encoding; color is secondary.
LANE_GLYPHS = ("◆", "━", "▶", "●")
TURN_BOUNDARY_CHAR = "│"
AGENT_BOUNDARY_CHAR = "┆"

_PLACEHOLDER = "No timing data — events remain in the ledger"

#: Secondary semantic styles per ledger kind; lane glyphs remain the primary
#: differentiation and work without color.
KIND_STYLES: dict[str, Style] = {
    KIND_USER: Style(color="cyan"),
    KIND_ASSISTANT: Style(color="green"),
    KIND_TOOL_CALL: Style(color="yellow"),
    KIND_TOOL_RESULT: Style(color="magenta"),
    KIND_COMPACTION: Style(color="red"),
    KIND_USER_FEEDBACK: Style(color="bright_blue"),
}
_FALLBACK_STYLE = Style(color="white")

#: Overlay for the selected record's bar (host-screen ledger cursor).
_SELECTED_STYLE = Style(reverse=True)

# Geometry-free focus cue used because the app-wide outline paints over the
# widget's perimeter cells (the Input row and first lane-label column).
_FOCUS_LABEL_STYLE = Style(reverse=True, bold=True)

_CAPTION_STYLE = Style(color="grey62")


@dataclass(frozen=True)
class TimelineBoundary:
    """A visual grouping marker, never a causal ordering edge."""

    kind: str
    record_key: str
    time: float


def _fmt_clock(t: float) -> str:
    """Format a unix timestamp as local ``HH:MM:SS``.

    Local time on purpose: the trajectory ledger's Start/Done columns
    format local time, so the strip's axis and brush caption must match
    the rows they select.
    """
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


class TimelineModel:
    """Pure time-domain geometry for the trajectory strip.

    No Textual dependency: everything here is arithmetic over the
    snapshot's timed records, so correctness (domain padding, mapping,
    lane packing, active-in-range, zoom/pan clamping) is unit-testable
    without a widget. Never mutates the records and never derives a
    timestamp -- NULL timing simply drops the record from the strip.
    """

    def __init__(
        self,
        records: Sequence[TrajectoryRecord] = (),
        *,
        record_keys: Mapping[int, str] | None = None,
    ) -> None:
        self._timed: tuple[TrajectoryRecord, ...] = tuple(
            r for r in records if r.step_started_at is not None
        )
        provided_keys = record_keys or {}
        self._record_keys = {
            id(record): provided_keys.get(id(record))
            or record.event_id
            or f"legacy-object:{id(record)}"
            for record in self._timed
        }
        if self._timed:
            lo = min(r.step_started_at for r in self._timed)
            hi = max(self.interval(r)[1] for r in self._timed)
            span = hi - lo
            pad = (span if span > 0 else 1.0) * DOMAIN_PADDING
            self._domain: tuple[float, float] | None = (lo - pad, hi + pad)
        else:
            self._domain = None
        self._lanes = self._assign_lanes()
        self._boundaries = self._build_boundaries()

    # -- data ---------------------------------------------------------------

    @property
    def timed_records(self) -> tuple[TrajectoryRecord, ...]:
        """Records with a usable start stamp, in ledger order."""
        return self._timed

    @property
    def domain(self) -> tuple[float, float] | None:
        """Padded time domain ``[min start, max end]``; ``None`` if empty."""
        return self._domain

    @property
    def has_data(self) -> bool:
        """Whether any record has timing to draw."""
        return self._domain is not None

    @property
    def lanes(self) -> tuple[int, ...]:
        """Lane index per timed record, parallel to ``timed_records``."""
        return self._lanes

    @property
    def lane_names(self) -> tuple[str, ...]:
        return LANE_NAMES

    @property
    def boundaries(self) -> tuple[TimelineBoundary, ...]:
        return self._boundaries

    def record_key(self, record: TrajectoryRecord) -> str:
        """Stable selection identity supplied by projection/screen ownership."""

        return self._record_keys[id(record)]

    @staticmethod
    def lane_for(record: TrajectoryRecord) -> int:
        """Assign every timed event to one named semantic lane."""

        kind = record.kind.lower()
        actor = (record.actor_kind or "").lower()
        if kind.startswith(("tool_", "approval_")) or kind in {
            KIND_TOOL_CALL,
            KIND_TOOL_RESULT,
        }:
            return 2
        if actor in {"agent", "subagent", "child_agent"} or kind.startswith(
            ("agent_", "subagent_")
        ):
            return 3
        if kind in {KIND_USER, KIND_SYSTEM, KIND_USER_FEEDBACK} or kind.startswith(
            ("user_", "feedback", "branch_", "edit_", "regenerate")
        ):
            return 0
        return 1

    def glyph_for(self, record: TrajectoryRecord) -> str:
        """Return a monochrome marker for the record's event family."""

        kind = record.kind.lower()
        status = (record.status or "").lower()
        lane = self.lane_for(record)
        if lane == 0:
            return (
                "◇"
                if kind == KIND_USER_FEEDBACK or "feedback" in kind
                else LANE_GLYPHS[lane]
            )
        if lane == 1:
            if "error" in kind or status in {
                "error",
                "failed",
                "rejected",
                "timed_out",
            }:
                return "!"
            return LANE_GLYPHS[lane]
        if lane == 2:
            return (
                "◀"
                if kind == KIND_TOOL_RESULT or kind.endswith("_result")
                else LANE_GLYPHS[lane]
            )
        return LANE_GLYPHS[lane] if kind in {"agent_run", "subagent_run"} else "○"

    @staticmethod
    def interval(record: TrajectoryRecord) -> tuple[float, float]:
        """Render interval ``[start, end]``; end falls back to start.

        A NULL ``completed_at`` (or a stored end before the start) yields
        a zero-width interval -- the widget draws it as one cell rather
        than fabricating a duration.
        """
        start = float(record.step_started_at)  # type: ignore[arg-type]
        end = record.completed_at
        if end is None or end < start:
            end = start
        return start, float(end)

    def _assign_lanes(self) -> tuple[int, ...]:
        return tuple(self.lane_for(record) for record in self._timed)

    def _build_boundaries(self) -> tuple[TimelineBoundary, ...]:
        boundaries: list[TimelineBoundary] = []
        previous_turn: str | None = None
        for record in self._timed:
            start, _ = self.interval(record)
            key = self.record_key(record)
            if record.turn_id != previous_turn:
                boundaries.append(TimelineBoundary("turn", key, start))
                previous_turn = record.turn_id
            if self.lane_for(record) == 3 and record.parent_event_id:
                boundaries.append(TimelineBoundary("agent", key, start))
        return tuple(boundaries)

    # -- mapping ------------------------------------------------------------

    @staticmethod
    def fraction(time: float, window: tuple[float, float]) -> float:
        """Map a time to a [0, 1] fraction of ``window`` (clamped)."""
        start, end = window
        span = end - start
        if span <= 0:
            return 0.0
        return min(1.0, max(0.0, (time - start) / span))

    @staticmethod
    def time_at(fraction: float, window: tuple[float, float]) -> float:
        """Map a [0, 1] fraction of ``window`` back to a time (clamped)."""
        start, end = window
        f = min(1.0, max(0.0, fraction))
        return start + f * (end - start)

    def records_in_range(self, lo: float, hi: float) -> tuple[TrajectoryRecord, ...]:
        """Timed records whose interval intersects ``[lo, hi]``."""
        a, b = (lo, hi) if lo <= hi else (hi, lo)
        return tuple(
            r
            for r in self._timed
            if self.interval(r)[0] <= b and self.interval(r)[1] >= a
        )

    # -- viewport -----------------------------------------------------------

    def zoom(
        self,
        window: tuple[float, float],
        factor: float,
        focal_fraction: float = 0.5,
    ) -> tuple[float, float]:
        """Zoom ``window`` by ``factor`` keeping ``focal_fraction`` fixed.

        ``factor > 1`` zooms in (narrower span). The result is clamped
        inside the domain; a zoomed-out span covering the whole domain is
        the domain itself (identity). Returns ``window`` unchanged when
        there is no data.
        """
        if self._domain is None:
            return window
        d0, d1 = self._domain
        w0, w1 = window
        span = w1 - w0
        new_span = span / factor
        if new_span >= (d1 - d0):
            return (d0, d1)
        focal_fraction = min(1.0, max(0.0, focal_fraction))
        focal = w0 + focal_fraction * span
        start = focal - focal_fraction * new_span
        start = min(max(start, d0), d1 - new_span)
        return (start, start + new_span)

    def pan(
        self, window: tuple[float, float], fraction_of_span: float
    ) -> tuple[float, float]:
        """Shift ``window`` by ``fraction_of_span`` of its own width.

        Clamped to the domain; a no-op (identity) when the window already
        is the full domain. Returns ``window`` unchanged when there is no
        data.
        """
        if self._domain is None:
            return window
        d0, d1 = self._domain
        w0, w1 = window
        span = w1 - w0
        if w0 <= d0 and w1 >= d1:
            return window  # already the full domain: nothing to pan to
        start = min(max(w0 + fraction_of_span * span, d0), d1 - span)
        return (start, start + span)


class TrajectoryTimeline(Widget):
    """Compact brushable time-domain strip over a trajectory snapshot.

    Interactions:

    - drag (left button, horizontal): brush range selection -- posts
      :class:`TrajectoryBrushChanged` with the time range (or ``None``
      when cleared).
    - click on a bar: posts :class:`TrajectoryBarSelected` with the
      record's stable row key and clears any brush.
    - click on empty space: clears the brush.
    - wheel / ``[`` ``]``: zoom out/in (wheel centers on the mouse x,
      keys on the strip center); ```,`' ``/`` .`` pan left/right. Every
      viewport change posts :class:`TrajectoryViewportChanged`.

    The host screen drives bar highlighting via :meth:`set_selected`
    (pull-only: it posts no messages, so a ledger cursor move can
    highlight a bar without echoing a selection event back).

    All keys are single-character (ADR-031 legal).
    """

    can_focus = True

    COMPONENT_CLASSES = {"timeline--brush"}

    BUNDLED_CSS = """
    TrajectoryTimeline {
        height: 6;
        width: 1fr;

        &>.timeline--brush {
            background: $primary 35%;
            text-style: bold;
        }
    }
    """

    class TrajectoryBrushChanged(Message):
        """Brush range changed; ``brush_range`` is a (lo, hi) time tuple or None."""

        def __init__(self, brush_range: tuple[float, float] | None) -> None:
            super().__init__()
            self.brush_range = brush_range

    class TrajectoryBarSelected(Message):
        """A record bar was selected by its stable screen row key."""

        def __init__(self, record_key: str) -> None:
            super().__init__()
            self.record_key = record_key

    class TrajectoryViewportChanged(Message):
        """Zoom/pan moved the time window; ``domain_window`` is (lo, hi)."""

        def __init__(self, domain_window: tuple[float, float]) -> None:
            super().__init__()
            self.domain_window = domain_window

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        # The global focus outline is painted over content in Textual. This
        # widget uses reversed lane labels as its non-obscuring focus cue.
        self.styles.outline = ("", "transparent")
        self._model = TimelineModel()
        self._window: tuple[float, float] | None = None
        self._brush: tuple[float, float] | None = None
        self._selected: str | None = None
        self._range_anchor: str | None = None
        self._drag_x: int | None = None
        self._drag_moved = False

    # -- state --------------------------------------------------------------

    @property
    def model(self) -> TimelineModel:
        """The pure geometry model (read-only view for callers/tests)."""
        return self._model

    @property
    def viewport(self) -> tuple[float, float] | None:
        """Current time window ``(lo, hi)``; ``None`` when no data."""
        return self._window

    @property
    def brush(self) -> tuple[float, float] | None:
        """Current brush range ``(lo, hi)`` in time, or ``None``."""
        return self._brush

    @property
    def selected(self) -> str | None:
        """Stable key of the highlighted bar, or ``None``."""
        return self._selected

    @property
    def range_anchor(self) -> str | None:
        return self._range_anchor

    def set_selected(self, record_key: str | None) -> None:
        """Highlight the bar for ``record_key`` (``None`` clears it).

        Dumb, pull-only selection driven by the host screen's ledger
        cursor: no message is posted, so highlighting cannot echo back
        into a ledger cursor move (the bar-click path owns that arc).
        """
        if self._selected == record_key:
            return
        self._selected = record_key
        self.refresh()

    def set_snapshot(
        self,
        snapshot: TrajectorySnapshot,
        *,
        record_keys: Mapping[int, str] | None = None,
    ) -> None:
        """Load a snapshot while retaining valid viewport/selection/range state."""

        old_window = self._window
        old_brush = self._brush
        old_selected = self._selected
        old_anchor = self._range_anchor
        records = [record for turn in snapshot.turns for record in turn.records]
        self._model = TimelineModel(records, record_keys=record_keys)
        domain = self._model.domain
        keys = {self._model.record_key(record) for record in self._model.timed_records}
        if domain is None:
            self._window = None
        elif (
            old_window is None or old_window[1] < domain[0] or old_window[0] > domain[1]
        ):
            self._window = domain
        else:
            span = min(old_window[1] - old_window[0], domain[1] - domain[0])
            start = min(max(old_window[0], domain[0]), domain[1] - span)
            self._window = (start, start + span)
        self._brush = (
            old_brush
            if old_brush is not None
            and domain is not None
            and old_brush[0] <= domain[1]
            and old_brush[1] >= domain[0]
            else None
        )
        self._selected = old_selected if old_selected in keys else None
        self._range_anchor = old_anchor if old_anchor in keys else None
        self._drag_x = None
        self.styles.height = STRIP_HEIGHT if self._model.has_data else 1
        self.refresh()

    # -- rendering ----------------------------------------------------------

    def render(self) -> Text:
        model = self._model
        width = max(self.size.width, 1)
        if not model.has_data or self._window is None:
            return Text(_PLACEHOLDER, style="dim", no_wrap=True, overflow="ellipsis")
        window = self._window
        plot_width = max(width - LANE_LABEL_WIDTH, 1)
        lines = [
            self._lane_line(lane, plot_width, window) for lane in range(LANE_COUNT)
        ]
        left = _fmt_clock(window[0])
        right = _fmt_clock(window[1])
        middle = " " * max(plot_width - len(left) - len(right), 0)
        lines.append(
            Text(" " * LANE_LABEL_WIDTH + left + middle + right, style=_CAPTION_STYLE)
        )
        lines.append(self._caption_line(width))
        return Text("\n").join(lines)

    def _lane_line(self, lane: int, width: int, window: tuple[float, float]) -> Text:
        """One lane row: bars for its records, brush background overlaid."""
        model = self._model
        cells: list[tuple[str, Style | None]] = [(" ", None)] * width
        for boundary in model.boundaries:
            if boundary.kind != "turn":
                continue
            col = int(TimelineModel.fraction(boundary.time, window) * width)
            cells[min(max(col, 0), width - 1)] = (TURN_BOUNDARY_CHAR, _CAPTION_STYLE)
        for record, record_lane in zip(model.timed_records, model.lanes):
            if record_lane != lane:
                continue
            cols = self._record_columns(record, width, window)
            if cols is None:
                continue
            style = KIND_STYLES.get(record.kind, _FALLBACK_STYLE)
            if model.record_key(record) == self._selected:
                style = style + _SELECTED_STYLE
            for col in range(cols[0], cols[1] + 1):
                cells[col] = (model.glyph_for(record), style)
        for boundary in model.boundaries:
            if boundary.kind != "agent" or lane != 3:
                continue
            col = int(TimelineModel.fraction(boundary.time, window) * width)
            col = min(max(col, 0), width - 1)
            record = next(
                (
                    item
                    for item in model.timed_records
                    if model.record_key(item) == boundary.record_key
                ),
                None,
            )
            if (
                record is not None
                and model.interval(record)[0] == model.interval(record)[1]
            ):
                # A boundary and instantaneous event share one cell. Preserve
                # both meanings instead of overwriting the event marker.
                _, style = cells[col]
                cells[col] = ("◉", style or _CAPTION_STYLE)
            else:
                cells[col] = (AGENT_BOUNDARY_CHAR, _CAPTION_STYLE)
        if self._brush is not None:
            brush_style = self.get_component_rich_style("timeline--brush", partial=True)
            b0, b1 = self._brush_columns(width, window)
            for col in range(b0, b1 + 1):
                char, style = cells[col]
                cells[col] = (char, (style or Style()) + brush_style)
        label = f"{LANE_NAMES[lane]:<{LANE_LABEL_WIDTH}}"
        label_style = (
            _CAPTION_STYLE + _FOCUS_LABEL_STYLE if self.has_focus else _CAPTION_STYLE
        )
        return Text.assemble(
            (label, label_style),
            *self._runs(cells),
            no_wrap=True,
            overflow="ignore",
        )

    @staticmethod
    def _runs(cells: list[tuple[str, Style | None]]) -> list[tuple[str, Style | None]]:
        """Collapse per-cell (char, style) into run segments for Text.assemble."""
        return [
            ("".join(char for char, _ in run), style)
            for style, run in groupby(cells, key=lambda cell: cell[1])
        ]

    def _record_columns(
        self, record: TrajectoryRecord, width: int, window: tuple[float, float]
    ) -> tuple[int, int] | None:
        """Column span [c0, c1] for a record; None when fully off-screen.

        Zero-width (instantaneous) intervals get c1 == c0 so they render
        as exactly one cell.
        """
        start, end = TimelineModel.interval(record)
        c0 = int(TimelineModel.fraction(start, window) * width)
        c1 = max(c0, int(TimelineModel.fraction(end, window) * width))
        if c1 < 0 or c0 >= width:
            return None
        return max(c0, 0), min(c1, width - 1)

    def _brush_columns(
        self, width: int, window: tuple[float, float]
    ) -> tuple[int, int]:
        """Inclusive column span of the brush (whole strip when unset)."""
        assert self._brush is not None
        lo, hi = self._brush
        b0 = int(TimelineModel.fraction(lo, window) * width)
        b1 = int(TimelineModel.fraction(hi, window) * width)
        return max(min(b0, b1), 0), min(max(b0, b1), width - 1)

    def _caption_line(self, width: int) -> Text:
        """Right-aligned brush window + active count (or ``no brush``)."""
        if self._brush is not None:
            lo, hi = self._brush
            active = len(self._model.records_in_range(lo, hi))
            caption = f"{_fmt_clock(lo)}–{_fmt_clock(hi)} · {active} active"
        else:
            caption = "no brush · ◇ feedback ! error ▶ call ◀ result ○ step"
        pad = " " * max(width - len(caption), 0)
        return Text(pad + caption, style=_CAPTION_STYLE)

    # -- geometry helpers (shared by mouse + tests) --------------------------

    def _fraction_from_column(self, column: int, width: int) -> float:
        return min(1.0, max(0.0, (column + 0.5) / width))

    def _plot_width(self) -> int:
        return max(self.size.width - LANE_LABEL_WIDTH, 1)

    @staticmethod
    def _plot_column(column: int) -> int:
        return max(column - LANE_LABEL_WIDTH, 0)

    def brush_columns(self, x1: int, x2: int) -> None:
        """Set the brush from two strip columns (the mouse-drag seam)."""
        if self._model.domain is None or self._window is None:
            return
        width = self._plot_width()
        t1 = TimelineModel.time_at(
            self._fraction_from_column(self._plot_column(x1), width), self._window
        )
        t2 = TimelineModel.time_at(
            self._fraction_from_column(self._plot_column(x2), width), self._window
        )
        self._set_brush((min(t1, t2), max(t1, t2)))

    def _set_brush(self, brush: tuple[float, float] | None) -> None:
        if self._brush == brush:
            return
        self._brush = brush
        self.refresh()
        self.post_message(self.TrajectoryBrushChanged(brush))

    def clear_range(self) -> bool:
        """Clear an active brush/keyboard anchor; return whether state changed."""

        changed = self._brush is not None or self._range_anchor is not None
        self._range_anchor = None
        self._set_brush(None)
        if changed:
            self.refresh()
        return changed

    def apply_brush(self, brush_range: tuple[float, float] | None) -> None:
        """Public re-brush seam for hosts (``None`` clears).

        ``set_snapshot`` resets the brush without posting; a host that
        swaps in a new snapshot (e.g. a live-refreshed trajectory
        screen) uses this to re-apply a brush that is still relevant,
        keeping its own filters in sync via the posted
        :class:`TrajectoryBrushChanged`.
        """
        self._set_brush(brush_range)

    def record_at(self, x: int, y: int) -> TrajectoryRecord | None:
        """The record whose rendered bar covers column ``x`` on row ``y``."""
        if x < LANE_LABEL_WIDTH or y < 0 or y >= LANE_COUNT or self._window is None:
            return None
        width = self._plot_width()
        x = self._plot_column(x)
        for record, lane in reversed(
            tuple(zip(self._model.timed_records, self._model.lanes))
        ):
            if lane != y:
                continue
            cols = self._record_columns(record, width, self._window)
            if cols is not None and cols[0] <= x <= cols[1]:
                return record
        return None

    # -- zoom / pan ----------------------------------------------------------

    def zoom_at(self, factor: float, focal_fraction: float = 0.5) -> None:
        """Zoom by ``factor`` around ``focal_fraction`` of the strip."""
        if self._window is None:
            return
        new_window = self._model.zoom(self._window, factor, focal_fraction)
        if new_window != self._window:
            self._window = new_window
            self.refresh()
            self.post_message(self.TrajectoryViewportChanged(new_window))

    def pan_by(self, fraction_of_span: float) -> None:
        """Pan by ``fraction_of_span`` of the window width (clamped)."""
        if self._window is None:
            return
        new_window = self._model.pan(self._window, fraction_of_span)
        if new_window != self._window:
            self._window = new_window
            self.refresh()
            self.post_message(self.TrajectoryViewportChanged(new_window))

    # -- mouse ----------------------------------------------------------------

    def on_mouse_down(self, event: MouseEvent) -> None:
        if event.button == 1 and event.x >= LANE_LABEL_WIDTH:
            # Capture for the gesture: a drag that leaves the 6-line
            # strip must keep feeding us moves, not strand the brush.
            self.capture_mouse()
            self._drag_x = event.x
            self._drag_moved = False

    def on_mouse_move(self, event: MouseEvent) -> None:
        if self._drag_x is not None and event.button == 1:
            if event.x != self._drag_x:
                self._drag_moved = True
                self.brush_columns(self._drag_x, event.x)

    def on_mouse_up(self, event: MouseEvent) -> None:
        if event.button != 1 or self._drag_x is None:
            return
        self.release_mouse()
        start_x, self._drag_x = self._drag_x, None
        if not self._drag_moved:
            # Plain click: bar select (and clear brush) or clear brush.
            record = self.record_at(event.x, event.y)
            if record is not None:
                self.clear_range()
                self._selected = self._model.record_key(record)
                self.refresh()
                self.post_message(self.TrajectoryBarSelected(self._selected))
            else:
                self.clear_range()
        else:
            self.brush_columns(start_x, event.x)

    def on_mouse_scroll_up(self, event: MouseEvent) -> None:
        width = self._plot_width()
        self.zoom_at(
            ZOOM_FACTOR,
            self._fraction_from_column(self._plot_column(event.x), width),
        )

    def on_mouse_scroll_down(self, event: MouseEvent) -> None:
        width = self._plot_width()
        self.zoom_at(
            1 / ZOOM_FACTOR,
            self._fraction_from_column(self._plot_column(event.x), width),
        )

    # -- keys -----------------------------------------------------------------

    BINDINGS = [
        Binding("k", "previous_event", "Previous event"),
        Binding("j", "next_event", "Next event"),
        Binding("enter", "select_event", "Select event"),
        Binding("b", "toggle_range", "Range"),
        Binding("left_square_bracket", "zoom_out", "Zoom out"),
        Binding("right_square_bracket", "zoom_in", "Zoom in"),
        Binding("comma", "pan_left", "Pan left"),
        Binding("full_stop", "pan_right", "Pan right"),
    ]

    def action_zoom_out(self) -> None:
        self.zoom_at(1 / ZOOM_FACTOR, 0.5)

    def action_zoom_in(self) -> None:
        self.zoom_at(ZOOM_FACTOR, 0.5)

    def action_pan_left(self) -> None:
        self.pan_by(-PAN_FRACTION)

    def action_pan_right(self) -> None:
        self.pan_by(PAN_FRACTION)

    def _select_relative(self, direction: int) -> None:
        records = self._model.timed_records
        if not records:
            return
        keys = [self._model.record_key(record) for record in records]
        if self._selected not in keys:
            index = 0 if direction > 0 else len(keys) - 1
        else:
            index = (keys.index(self._selected) + direction) % len(keys)
        self._selected = keys[index]
        self.refresh()

    def action_previous_event(self) -> None:
        self._select_relative(-1)

    def action_next_event(self) -> None:
        self._select_relative(1)

    def action_select_event(self) -> None:
        if self._selected is not None:
            self.post_message(self.TrajectoryBarSelected(self._selected))

    def action_toggle_range(self) -> None:
        if self._selected is None:
            self._select_relative(1)
        if self._selected is None:
            return
        if self._range_anchor is None:
            self._range_anchor = self._selected
            self.refresh()
            return
        by_key = {
            self._model.record_key(record): record
            for record in self._model.timed_records
        }
        anchor = by_key.get(self._range_anchor)
        selected = by_key.get(self._selected)
        self._range_anchor = None
        if anchor is None or selected is None:
            self.refresh()
            return
        anchor_start, anchor_end = self._model.interval(anchor)
        selected_start, selected_end = self._model.interval(selected)
        self._set_brush(
            (
                min(anchor_start, selected_start),
                max(anchor_end, selected_end),
            )
        )
