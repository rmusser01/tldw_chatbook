"""B21: trajectory timeline renders from precomputed lanes; drag repaints coalesce.

1. After ``set_snapshot``, ``render()`` must consume only the precomputed
   per-lane record lists and the key->record map -- no per-lane scan over
   ``model.timed_records`` and no per-boundary linear ``next(...)`` lookup.
2. A 60-event mouse-move drag burst coalesces repaints to at most one per
   ~33 ms (<= 3 full refreshes), instead of one full render per event.

Render-output equality against the pre-change implementation is verified
separately with a golden capture (5 state configurations, plain text +
style spans) captured before the change and compared after.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.app import ComposeResult

import tldw_chatbook.app  # noqa: F401,E402
from tldw_chatbook.Chat.trajectory import (
    KIND_ASSISTANT,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    KIND_USER,
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
)
from tldw_chatbook.UI.Widgets.trajectory_timeline import (
    LANE_COUNT,
    TimelineModel,
    TrajectoryTimeline,
)
from Tests.UI.consolidated_css import ConsolidatedCSSApp

_T0 = 1_755_165_600.0

pytestmark = pytest.mark.bootstrap_profile


class _TimelineHost(ConsolidatedCSSApp):
    def __init__(self, timeline: TrajectoryTimeline) -> None:
        super().__init__()
        self._timeline = timeline

    def compose(self) -> ComposeResult:
        yield self._timeline


def rec(seq, kind, *, start, end=None, turn_id="t1", **extra) -> TrajectoryRecord:
    return TrajectoryRecord(
        seq=seq,
        kind=kind,
        turn_id=turn_id,
        message_id=f"m{seq}",
        content_preview="",
        usage=None,
        step_started_at=None if start is None else _T0 + start,
        first_token_at=None,
        completed_at=None if end is None else _T0 + end,
        model=None,
        provider=None,
        payload=None,
        variants=(),
        depth=0,
        event_id=f"event:{seq}",
        **extra,
    )


def _fixture_snapshot() -> TrajectorySnapshot:
    records = [
        rec(1, KIND_USER, start=0.0, end=4.0, turn_id="t1"),
        rec(2, KIND_ASSISTANT, start=1.0, end=9.0, turn_id="t1"),
        rec(3, KIND_TOOL_CALL, start=2.0, end=6.0, turn_id="t1"),
        rec(4, KIND_TOOL_RESULT, start=3.0, end=8.0, turn_id="t1"),
        rec(
            5,
            "agent_run",
            start=2.5,
            end=7.5,
            turn_id="t1",
            actor_kind="agent",
            run_id="child-1",
            parent_event_id="agent-run:parent",
        ),
        rec(
            6,
            "agent_step",
            start=3.5,
            turn_id="t1",
            actor_kind="agent",
            run_id="child-1",
            parent_event_id="agent-run:parent",
        ),
        rec(7, KIND_USER, start=10.0, end=12.0, turn_id="t2"),
        rec(8, KIND_ASSISTANT, start=10.5, turn_id="t2"),
        rec(9, KIND_TOOL_CALL, start=11.0, end=14.0, turn_id="t2"),
        rec(
            10,
            "subagent_run",
            start=11.5,
            end=13.5,
            turn_id="t2",
            actor_kind="subagent",
            run_id="child-2",
            parent_event_id="agent-run:parent",
        ),
    ]
    return TrajectorySnapshot(
        turns=[
            TrajectoryTurn(turn_id="t1", records=tuple(records[:6])),
            TrajectoryTurn(turn_id="t2", records=tuple(records[6:])),
        ]
    )


def _move_event(x: int, y: int = 5) -> events.MouseMove:
    return events.MouseMove(
        widget=None,
        x=x,
        y=y,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=x,
        screen_y=y,
    )


@pytest.mark.asyncio
async def test_render_consumes_only_precomputed_lane_structures(monkeypatch):
    widget = TrajectoryTimeline()
    async with _TimelineHost(widget).run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        widget.set_snapshot(_fixture_snapshot())
        await pilot.pause()

        timed_accesses: list[int] = []
        real_timed = TimelineModel.timed_records

        @property
        def spy_timed(self):
            timed_accesses.append(1)
            return real_timed.fget(self)

        monkeypatch.setattr(TimelineModel, "timed_records", spy_timed)

        column_calls: list[int] = []
        real_columns = TrajectoryTimeline._record_columns

        def spy_columns(self, record, width, window):
            column_calls.append(1)
            return real_columns(self, record, width, window)

        monkeypatch.setattr(TrajectoryTimeline, "_record_columns", spy_columns)

        text = widget.render()
        monkeypatch.undo()

        # The strip must have rendered (fixture covers all four lanes).
        assert "Input" in text.plain and "Agents" in text.plain

        # Old code touched model.timed_records once per lane plus once per
        # agent boundary (linear next(...) scans); the precomputed lane
        # lists and key map remove every access.
        assert timed_accesses == [], (
            f"render() re-scanned model.timed_records {len(timed_accesses)}x"
        )

        # Record-column work is bounded: one call per rendered record,
        # never lanes x total.
        timed_count = len(widget.model.timed_records)
        per_lane = [0] * LANE_COUNT
        for record, lane in zip(widget.model.timed_records, widget.model.lanes):
            per_lane[lane] += 1
        assert len(column_calls) == timed_count
        assert len(column_calls) <= LANE_COUNT * max(per_lane)

        # The precomputed structures exist and agree with the model: each
        # lane list is exactly that lane's records in timed order.
        assert len(widget._record_by_key) == timed_count
        for lane in range(LANE_COUNT):
            expected = [
                record
                for record, record_lane in zip(
                    widget.model.timed_records, widget.model.lanes
                )
                if record_lane == lane
            ]
            assert list(widget._lane_records[lane]) == expected


@pytest.mark.asyncio
async def test_drag_burst_coalesces_repaints():
    widget = TrajectoryTimeline()
    async with _TimelineHost(widget).run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        widget.set_snapshot(_fixture_snapshot())
        await pilot.pause()

        assert widget.is_mounted and widget.size.width > 0

        refreshes: list[int] = []
        real_refresh = widget.refresh

        def spy_refresh(*args, **kwargs):
            refreshes.append(1)
            return real_refresh(*args, **kwargs)

        monkey_refresh = widget.refresh
        widget.refresh = spy_refresh  # type: ignore[method-assign]

        try:
            # Gesture bookkeeping identical to on_mouse_down.
            widget._drag_x = 20
            widget._drag_moved = False

            for x in range(21, 81):  # 60 move events, distinct columns
                widget.on_mouse_move(_move_event(x))
        finally:
            widget.refresh = monkey_refresh  # type: ignore[method-assign]

        # Mid-gesture the brush tracks the mouse (state), but the repaint
        # is coalesced: at most ~1 per 33 ms window, far below one per event.
        assert widget._brush is not None, "brush must still follow the drag"
        assert len(refreshes) <= 3, (
            f"60-event drag burst triggered {len(refreshes)} repaints"
        )


@pytest.mark.asyncio
async def test_short_drag_paints_final_brush_after_mouse_up():
    widget = TrajectoryTimeline()
    app = _TimelineHost(widget)
    async with app.run_test(size=(100, 40)) as pilot:
        widget.set_snapshot(_fixture_snapshot())
        await pilot.pause()
        assert widget.is_mounted and widget.size.width > 0
        # Settle the first hover/selection initialization before the gesture.
        widget._forward_event(_move_event(20))
        widget._forward_event(events.MouseDown.from_event(widget, _move_event(20)))
        widget._forward_event(events.MouseUp.from_event(widget, _move_event(20)))
        await pilot.pause()
        assert "no brush" in "\n".join(
            strip.text for strip in app.screen._compositor.render_strips()
        )

        widget._forward_event(events.MouseDown.from_event(widget, _move_event(20)))
        widget._forward_event(_move_event(40))
        widget._forward_event(events.MouseUp.from_event(widget, _move_event(40)))
        await pilot.pause(0.12)

        assert widget.brush is not None
        painted = "\n".join(
            strip.text for strip in app.screen._compositor.render_strips()
        )
        assert "active" in painted
        assert "no brush" not in painted
