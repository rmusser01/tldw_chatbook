"""Brushable time-domain strip over a ``TrajectorySnapshot`` (task-16315).

A compact 6-line widget that renders every *timed* trajectory record as a
horizontal bar (``█``) positioned by its ``[step_started_at,
completed_at]`` interval, on one of 4 greedily-packed lanes. The strip
supports zoom (wheel / ``[`` ``]``), pan (```,`' ``/`` .``), drag-brush
range selection, and click-to-select on a bar. Geometry lives in
:class:`TimelineModel`, a pure class that never touches Textual so the
math (domain padding, mapping, lanes, zoom/pan clamping) is unit-testable
in isolation; the widget is a thin event/render shell.

Records with NULL timing are skipped (blank, never fabricated -- dsh
precedent); a snapshot with no usable timing renders a centered
``no timing data`` placeholder.

Part 1 of the trajectory timeline follow-up: standalone widget only, no
integration with ``TrajectoryScreen`` yet.
"""

from __future__ import annotations

import math
from datetime import datetime
from itertools import groupby
from typing import Sequence

from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.events import MouseEvent
from textual.message import Message
from textual.widget import Widget

from tldw_chatbook.Chat.trajectory import (
    KIND_ASSISTANT,
    KIND_COMPACTION,
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

#: Total rendered lines: LANE_COUNT lane rows + an axis row + the caption.
STRIP_HEIGHT = LANE_COUNT + 2

#: Domain padding on each side, as a fraction of the data time span.
DOMAIN_PADDING = 0.05

#: Wheel/keyboard zoom factor (>1 zooms in).
ZOOM_FACTOR = 1.25

#: Keyboard pan step, as a fraction of the current window span.
PAN_FRACTION = 0.25

#: Bar glyph; instantaneous records render as a single cell.
BAR_CHAR = "█"

_PLACEHOLDER = "no timing data"

#: Distinct bar styles per ledger kind. The trajectory ledger screen has
#: no kind->color scheme (plain per-cell Text), so the strip defines one.
KIND_STYLES: dict[str, Style] = {
    KIND_USER: Style(color="cyan"),
    KIND_ASSISTANT: Style(color="green"),
    KIND_TOOL_CALL: Style(color="yellow"),
    KIND_TOOL_RESULT: Style(color="magenta"),
    KIND_COMPACTION: Style(color="red"),
    KIND_USER_FEEDBACK: Style(color="bright_blue"),
}
_FALLBACK_STYLE = Style(color="white")

#: Background applied to every cell inside the brushed region.
BRUSH_STYLE = Style(bgcolor="#264f78")

#: Overlay for the selected record's bar (host-screen ledger cursor).
_SELECTED_STYLE = Style(reverse=True)

_CAPTION_STYLE = Style(color="grey62")


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

    def __init__(self, records: Sequence[TrajectoryRecord] = ()) -> None:
        self._timed: tuple[TrajectoryRecord, ...] = tuple(
            r for r in records if r.step_started_at is not None
        )
        if self._timed:
            lo = min(r.step_started_at for r in self._timed)
            hi = max(self.interval(r)[1] for r in self._timed)
            span = hi - lo
            pad = (span if span > 0 else 1.0) * DOMAIN_PADDING
            self._domain: tuple[float, float] | None = (lo - pad, hi + pad)
        else:
            self._domain = None
        self._lanes = self._assign_lanes()

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
        """Greedy lane packing: lowest lane free at each record's start.

        Stable in ledger order; a record overlapping all ``LANE_COUNT``
        lanes piles onto the last lane (capped, never grows the strip).
        """
        last_end = [-math.inf] * LANE_COUNT
        lanes: list[int] = []
        for record in self._timed:
            start, end = self.interval(record)
            lane = next(
                (i for i in range(LANE_COUNT) if last_end[i] <= start),
                LANE_COUNT - 1,
            )
            last_end[lane] = max(last_end[lane], end)
            lanes.append(lane)
        return tuple(lanes)

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
      record's ledger ``seq`` and clears any brush.
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

    DEFAULT_CSS = """
    TrajectoryTimeline {
        height: 6;
        width: 1fr;
    }
    """

    class TrajectoryBrushChanged(Message):
        """Brush range changed; ``brush_range`` is a (lo, hi) time tuple or None."""

        def __init__(self, brush_range: tuple[float, float] | None) -> None:
            super().__init__()
            self.brush_range = brush_range

    class TrajectoryBarSelected(Message):
        """A record bar was clicked; ``record_key`` is its ledger seq."""

        def __init__(self, record_key: int) -> None:
            super().__init__()
            self.record_key = record_key

    class TrajectoryViewportChanged(Message):
        """Zoom/pan moved the time window; ``domain_window`` is (lo, hi)."""

        def __init__(self, domain_window: tuple[float, float]) -> None:
            super().__init__()
            self.domain_window = domain_window

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._model = TimelineModel()
        self._window: tuple[float, float] | None = None
        self._brush: tuple[float, float] | None = None
        self._selected: int | None = None
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
    def selected(self) -> int | None:
        """Ledger seq of the highlighted bar, or ``None``."""
        return self._selected

    def set_selected(self, record_key: int | None) -> None:
        """Highlight the bar for ``record_key`` (``None`` clears it).

        Dumb, pull-only selection driven by the host screen's ledger
        cursor: no message is posted, so highlighting cannot echo back
        into a ledger cursor move (the bar-click path owns that arc).
        """
        if self._selected == record_key:
            return
        self._selected = record_key
        self.refresh()

    def set_snapshot(self, snapshot: TrajectorySnapshot) -> None:
        """Load a snapshot: flatten turns, reset viewport to the domain."""
        records = [record for turn in snapshot.turns for record in turn.records]
        self._model = TimelineModel(records)
        self._window = self._model.domain
        self._brush = None
        self._selected = None
        self._drag_x = None
        self.refresh()

    # -- rendering ----------------------------------------------------------

    def render(self) -> Text:
        model = self._model
        width = max(self.size.width, 1)
        if not model.has_data or self._window is None:
            placeholder = Text(_PLACEHOLDER, style="dim")
            lines = [Text("")] * (STRIP_HEIGHT - 1) + [placeholder]
            return Text("\n").join(lines)
        window = self._window
        lines = [self._lane_line(lane, width, window) for lane in range(LANE_COUNT)]
        left = _fmt_clock(window[0])
        right = _fmt_clock(window[1])
        middle = " " * max(width - len(left) - len(right), 0)
        lines.append(Text(left + middle + right, style=_CAPTION_STYLE))
        lines.append(self._caption_line(width))
        return Text("\n").join(lines)

    def _lane_line(self, lane: int, width: int, window: tuple[float, float]) -> Text:
        """One lane row: bars for its records, brush background overlaid."""
        model = self._model
        cells: list[tuple[str, Style | None]] = [(" ", None)] * width
        for record, record_lane in zip(model.timed_records, model.lanes):
            if record_lane != lane:
                continue
            cols = self._record_columns(record, width, window)
            if cols is None:
                continue
            style = KIND_STYLES.get(record.kind, _FALLBACK_STYLE)
            if record.seq == self._selected:
                style = style + _SELECTED_STYLE
            for col in range(cols[0], cols[1] + 1):
                cells[col] = (BAR_CHAR, style)
        if self._brush is not None:
            b0, b1 = self._brush_columns(width, window)
            for col in range(b0, b1 + 1):
                char, style = cells[col]
                cells[col] = (char, (style or Style()) + BRUSH_STYLE)
        return Text.assemble(*self._runs(cells), no_wrap=True, overflow="ignore")

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
            caption = "no brush"
        pad = " " * max(width - len(caption), 0)
        return Text(pad + caption, style=_CAPTION_STYLE)

    # -- geometry helpers (shared by mouse + tests) --------------------------

    def _fraction_from_column(self, column: int, width: int) -> float:
        return min(1.0, max(0.0, (column + 0.5) / width))

    def brush_columns(self, x1: int, x2: int) -> None:
        """Set the brush from two strip columns (the mouse-drag seam)."""
        if self._model.domain is None or self._window is None:
            return
        width = max(self.size.width, 1)
        t1 = TimelineModel.time_at(self._fraction_from_column(x1, width), self._window)
        t2 = TimelineModel.time_at(self._fraction_from_column(x2, width), self._window)
        self._set_brush((min(t1, t2), max(t1, t2)))

    def _set_brush(self, brush: tuple[float, float] | None) -> None:
        if self._brush == brush:
            return
        self._brush = brush
        self.refresh()
        self.post_message(self.TrajectoryBrushChanged(brush))

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
        if y < 0 or y >= LANE_COUNT or self._window is None:
            return None
        width = max(self.size.width, 1)
        for record, lane in zip(self._model.timed_records, self._model.lanes):
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
        if event.button == 1:
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
                self._set_brush(None)
                self.post_message(self.TrajectoryBarSelected(record.seq))
            else:
                self._set_brush(None)
        else:
            self.brush_columns(start_x, event.x)

    def on_mouse_scroll_up(self, event: MouseEvent) -> None:
        width = max(self.size.width, 1)
        self.zoom_at(ZOOM_FACTOR, self._fraction_from_column(event.x, width))

    def on_mouse_scroll_down(self, event: MouseEvent) -> None:
        width = max(self.size.width, 1)
        self.zoom_at(1 / ZOOM_FACTOR, self._fraction_from_column(event.x, width))

    # -- keys -----------------------------------------------------------------

    BINDINGS = [
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
