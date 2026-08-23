"""Tests for the brushable trajectory timeline (task-16315, part 1).

Pure ``TimelineModel`` unit tests cover where correctness lives (domain
padding, mapping, lane packing, active-in-range, zoom/pan math). Widget
tests are pilot-driven per the repo's Textual patterns
(``test_trajectory_screen.py``): snapshots come from the REAL projection
(``derive_trajectory``) fed duck-typed row mirrors, and mouse-drag brush
logic goes through the same ``brush_columns`` seam the mouse handlers
delegate to (the pilot has no drag primitive).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from textual import events
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.trajectory import (
    KIND_ASSISTANT,
    KIND_TOOL_CALL,
    KIND_USER,
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
    derive_trajectory,
)
from tldw_chatbook.UI.Widgets.trajectory_timeline import (
    LANE_COUNT,
    LANE_LABEL_WIDTH,
    ZOOM_FACTOR,
    TimelineModel,
    TrajectoryTimeline,
)

# ---------------------------------------------------------------------------
# Pure-model fixtures
# ---------------------------------------------------------------------------

_T0 = 1_755_165_600.0  # arbitrary unix epoch base


def rec(
    seq: int,
    kind: str = KIND_USER,
    *,
    start: float | None,
    end: float | None = None,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        seq=seq,
        kind=kind,
        turn_id="t1",
        message_id=f"m{seq}",
        content_preview="",
        usage=None,
        step_started_at=start,
        first_token_at=None,
        completed_at=end,
        model=None,
        provider=None,
        payload=None,
        variants=(),
        depth=0,
        event_id=f"event:{seq}",
    )


# ---------------------------------------------------------------------------
# TimelineModel unit tests
# ---------------------------------------------------------------------------


class TestTimelineModel:
    def test_semantic_lanes_replace_anonymous_greedy_packing(self) -> None:
        records = [
            rec(1, KIND_USER, start=0.0, end=10.0),
            rec(2, KIND_ASSISTANT, start=1.0, end=9.0),
            rec(3, KIND_TOOL_CALL, start=2.0, end=8.0),
            replace(
                rec(4, "agent_step", start=3.0, end=7.0),
                actor_kind="agent",
                run_id="child-1",
                parent_event_id="agent-run:parent",
            ),
        ]
        model = TimelineModel(records)

        assert model.lane_names == ("Input", "Model", "Tools", "Agents")
        assert model.lanes == (0, 1, 2, 3)

    def test_semantic_lane_assignment_never_depends_on_overlap(self) -> None:
        records = [
            rec(1, KIND_USER, start=0.0, end=100.0),
            rec(2, KIND_USER, start=1.0, end=99.0),
            rec(3, KIND_ASSISTANT, start=2.0, end=98.0),
            rec(4, KIND_TOOL_CALL, start=3.0, end=97.0),
        ]
        assert TimelineModel(records).lanes == (0, 0, 1, 2)

    def test_record_identity_is_stable_event_id_not_display_seq(self) -> None:
        original = replace(rec(1, start=0.0, end=1.0), event_id="message:stable")
        renumbered = replace(original, seq=999)

        assert TimelineModel([original]).record_key(original) == "message:stable"
        assert TimelineModel([renumbered]).record_key(renumbered) == "message:stable"

    def test_legacy_record_identity_accepts_screen_row_key_contract(self) -> None:
        legacy = replace(rec(1, start=0.0, end=1.0), event_id="")
        model = TimelineModel([legacy], record_keys={id(legacy): "legacy:screen-key"})

        assert model.record_key(legacy) == "legacy:screen-key"

    def test_non_color_glyphs_distinguish_each_semantic_lane(self) -> None:
        records = [
            rec(1, KIND_USER, start=0.0, end=1.0),
            rec(2, KIND_ASSISTANT, start=2.0, end=3.0),
            rec(3, KIND_TOOL_CALL, start=4.0, end=5.0),
            replace(rec(4, "agent_step", start=6.0, end=7.0), actor_kind="agent"),
        ]
        model = TimelineModel(records)

        assert len({model.glyph_for(record) for record in records}) == 4

    def test_turn_and_child_agent_boundaries_are_explicit_not_order_edges(self) -> None:
        child = replace(
            rec(3, "agent_step", start=4.0, end=5.0),
            turn_id="t2",
            actor_kind="agent",
            run_id="child",
            parent_event_id="agent-run:parent",
        )
        records = [
            replace(rec(1, start=0.0, end=1.0), turn_id="t1"),
            replace(rec(2, KIND_ASSISTANT, start=2.0, end=3.0), turn_id="t2"),
            child,
        ]

        boundaries = TimelineModel(records).boundaries
        assert [(item.kind, item.record_key) for item in boundaries] == [
            ("turn", "event:1"),
            ("turn", "event:2"),
            ("agent", "event:3"),
        ]

    def test_domain_padding(self) -> None:
        model = TimelineModel(
            [rec(1, start=10.0, end=20.0), rec(2, start=15.0, end=30.0)]
        )
        assert model.domain == (9.0, 31.0)  # 5% of the 20s span on each side

    def test_null_timing_skipped(self) -> None:
        model = TimelineModel([rec(1, start=None), rec(2, start=10.0, end=12.0)])
        assert [r.seq for r in model.timed_records] == [2]

    def test_all_null_has_no_data(self) -> None:
        model = TimelineModel([rec(1, start=None), rec(2, start=None, end=5.0)])
        assert model.has_data is False
        assert model.domain is None

    def test_instantaneous_span_uses_unit_pad(self) -> None:
        model = TimelineModel([rec(1, start=10.0, end=10.0)])
        assert model.domain == (10.0 - 0.05, 10.0 + 0.05)

    def test_fraction_mapping_and_clamp(self) -> None:
        window = (0.0, 10.0)
        assert TimelineModel.fraction(0.0, window) == 0.0
        assert TimelineModel.fraction(5.0, window) == pytest.approx(0.5)
        assert TimelineModel.fraction(10.0, window) == 1.0
        assert TimelineModel.fraction(-3.0, window) == 0.0
        assert TimelineModel.fraction(99.0, window) == 1.0
        assert TimelineModel.time_at(0.25, window) == pytest.approx(2.5)

    def test_lane_assignment_overlap_and_reuse(self) -> None:
        model = TimelineModel(
            [
                rec(1, start=0.0, end=10.0),
                rec(2, KIND_ASSISTANT, start=2.0, end=8.0),  # overlaps rec 1
                rec(3, KIND_TOOL_CALL, start=3.0, end=4.0),  # overlaps both
                rec(4, start=11.0, end=12.0),  # after everything: lane 0 free
            ]
        )
        assert model.lanes == (0, 1, 2, 0)

    def test_same_semantic_family_stays_in_one_lane_under_heavy_overlap(self) -> None:
        records = [rec(i + 1, start=float(i), end=100.0) for i in range(LANE_COUNT + 2)]
        model = TimelineModel(records)
        assert model.lanes == (0, 0, 0, 0, 0, 0)

    def test_records_in_range(self) -> None:
        model = TimelineModel(
            [
                rec(1, start=0.0, end=10.0),
                rec(2, KIND_ASSISTANT, start=12.0, end=18.0),
                rec(3, KIND_TOOL_CALL, start=13.0, end=16.0),
                rec(4, start=30.0, end=31.0),
            ]
        )
        assert [r.seq for r in model.records_in_range(5.0, 11.0)] == [1]
        assert [r.seq for r in model.records_in_range(14.0, 14.5)] == [2, 3]
        # touching endpoints count as intersecting
        assert [r.seq for r in model.records_in_range(10.0, 12.0)] == [1, 2]
        assert [r.seq for r in model.records_in_range(100.0, 200.0)] == []

    def test_zoom_in_preserves_focal_point(self) -> None:
        model = TimelineModel([rec(1, start=0.0, end=100.0)])
        domain = model.domain
        assert domain is not None
        window = model.zoom(domain, ZOOM_FACTOR, focal_fraction=0.5)
        span = domain[1] - domain[0]
        focal = domain[0] + 0.5 * span
        assert window[1] - window[0] == pytest.approx(span / ZOOM_FACTOR)
        assert (window[0] + window[1]) / 2 == pytest.approx(focal)
        assert domain[0] <= window[0] and window[1] <= domain[1]

    def test_zoom_out_at_domain_is_identity(self) -> None:
        model = TimelineModel([rec(1, start=0.0, end=100.0)])
        domain = model.domain
        assert domain is not None
        zoomed = model.zoom(domain, ZOOM_FACTOR, 0.5)
        assert model.zoom(zoomed, 1 / ZOOM_FACTOR, 0.5) == domain

    def test_zoom_clamps_inside_domain(self) -> None:
        model = TimelineModel([rec(1, start=0.0, end=100.0)])
        domain = model.domain
        assert domain is not None
        window = model.zoom(domain, 4.0, focal_fraction=0.0)
        assert window[0] == domain[0]  # pinned to the left edge, not past it
        window = model.zoom(domain, 4.0, focal_fraction=1.0)
        assert window[1] == domain[1]

    def test_zoom_clamps_focal_fraction_once(self) -> None:
        """Out-of-range focal fractions clamp consistently everywhere."""
        model = TimelineModel([rec(1, start=0.0, end=100.0)])
        domain = model.domain
        assert domain is not None
        assert model.zoom(domain, 4.0, focal_fraction=2.0) == model.zoom(
            domain, 4.0, focal_fraction=1.0
        )
        assert model.zoom(domain, 4.0, focal_fraction=-1.0) == model.zoom(
            domain, 4.0, focal_fraction=0.0
        )

    def test_pan_noop_at_full_domain(self) -> None:
        model = TimelineModel([rec(1, start=0.0, end=100.0)])
        domain = model.domain
        assert domain is not None
        assert model.pan(domain, -0.25) == domain
        assert model.pan(domain, 0.9) == domain

    def test_pan_clamps_at_edges(self) -> None:
        model = TimelineModel([rec(1, start=0.0, end=100.0)])
        domain = model.domain
        assert domain is not None
        window = model.zoom(domain, 2.0, 0.5)
        span = window[1] - window[0]
        left = model.pan(window, -10.0)
        assert left[0] == domain[0] and left[1] - left[0] == pytest.approx(span)
        right = model.pan(window, 10.0)
        assert right[1] == domain[1]
        small = model.pan(window, 0.25)
        assert small[0] - window[0] == pytest.approx(0.25 * span)


# ---------------------------------------------------------------------------
# Duck-typed projection inputs (same mirrors as the trajectory screen tests)
# ---------------------------------------------------------------------------


def msg(
    mid: str, sender: str, *, content: str, ts: float, parent: str | None = None
) -> dict:
    return {
        "id": mid,
        "sender": sender,
        "content": content,
        "timestamp": ts,
        "parent_message_id": parent,
        "deleted": False,
    }


@dataclass(frozen=True)
class TrajRow:
    message_id: str
    conversation_id: str = "conv-1"
    turn_id: str = "t1"
    seq: int = 0
    event_kind: str = "assistant"
    step_started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None
    model: str | None = None
    provider: str | None = None
    payload_json: str | None = None


def timed_snapshot():
    """u1 [0,10] lane0; a1 [12,18] lane0; tool [13,16] lane1 (overlaps a1)."""
    messages = [
        msg("u1", "user", content="hello", ts=_T0, parent=None),
        msg("a1", "assistant", content="working", ts=_T0 + 1.0, parent="u1"),
    ]
    rows = [
        TrajRow(
            "u1",
            turn_id="t1",
            seq=1,
            event_kind="user",
            step_started_at=_T0,
            completed_at=_T0 + 10.0,
        ),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=2,
            event_kind="assistant",
            step_started_at=_T0 + 12.0,
            completed_at=_T0 + 18.0,
        ),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=3,
            event_kind="tool_call",
            step_started_at=_T0 + 13.0,
            completed_at=_T0 + 16.0,
            payload_json='{"name": "fs_read", "args": {}, "result": "ok"}',
        ),
    ]
    return derive_trajectory(messages, {}, rows, [], [])


def untimed_snapshot():
    """Same shape but every sidecar row lacks timing: NULL-only."""
    messages = [
        msg("u1", "user", content="hello", ts=_T0, parent=None),
        msg("a1", "assistant", content="working", ts=_T0 + 1.0, parent="u1"),
    ]
    rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
    ]
    return derive_trajectory(messages, {}, rows, [], [])


class TimelineApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.captured: list[object] = []

    def compose(self) -> ComposeResult:
        yield TrajectoryTimeline()

    def on_trajectory_timeline_trajectory_brush_changed(self, event) -> None:
        self.captured.append(event)

    def on_trajectory_timeline_trajectory_bar_selected(self, event) -> None:
        self.captured.append(event)

    def on_trajectory_timeline_trajectory_viewport_changed(self, event) -> None:
        self.captured.append(event)


def _brush_events(app: TimelineApp):
    return [
        e
        for e in app.captured
        if isinstance(e, TrajectoryTimeline.TrajectoryBrushChanged)
    ]


def _bar_events(app: TimelineApp):
    return [
        e
        for e in app.captured
        if isinstance(e, TrajectoryTimeline.TrajectoryBarSelected)
    ]


def _viewport_events(app: TimelineApp):
    return [
        e
        for e in app.captured
        if isinstance(e, TrajectoryTimeline.TrajectoryViewportChanged)
    ]


# ---------------------------------------------------------------------------
# Widget pilot tests
# ---------------------------------------------------------------------------


class TestTrajectoryTimelineWidget:
    async def test_render_paints_semantic_labels_glyphs_and_boundaries(self) -> None:
        records = (
            replace(rec(1, KIND_USER, start=0.0, end=1.0), turn_id="t1"),
            replace(rec(2, KIND_ASSISTANT, start=2.0, end=3.0), turn_id="t2"),
            replace(rec(3, KIND_TOOL_CALL, start=4.0, end=5.0), turn_id="t2"),
            replace(
                rec(4, "agent_step", start=6.0, end=7.0),
                turn_id="t2",
                actor_kind="agent",
                run_id="child",
                parent_event_id="agent-run:parent",
            ),
        )
        snapshot = TrajectorySnapshot((TrajectoryTurn("all", records),))
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(snapshot)
            await pilot.pause()

            painted = str(tl.render())
            for label in ("Input", "Model", "Tools", "Agents"):
                assert label in painted
            for glyph in ("◆", "━", "■", "●"):
                assert glyph in painted
            assert "│" in painted  # turn boundary, not a serial-causality arrow
            assert "┆" in painted  # parent/child agent boundary

    async def test_timeline_bindings_cover_keyboard_selection_range_zoom_and_pan(
        self,
    ) -> None:
        actions = {binding.action for binding in TrajectoryTimeline.BINDINGS}
        assert {
            "previous_event",
            "next_event",
            "select_event",
            "toggle_range",
            "zoom_out",
            "zoom_in",
            "pan_left",
            "pan_right",
        } <= actions

    async def test_no_timing_snapshot_collapses_the_actual_widget_to_one_row(
        self,
    ) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(untimed_snapshot())
            await pilot.pause()

            assert tl.size.height == 1
            assert (
                str(tl.render()).strip()
                == "No timing data — events remain in the ledger"
            )

    async def test_keyboard_selection_and_range_use_stable_record_keys(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            tl.focus()

            await pilot.press("j")
            assert tl.selected == "message:u1"
            await pilot.press("b")
            assert tl.range_anchor == "message:u1"
            await pilot.press("j")
            await pilot.press("b")
            assert tl.brush is not None
            assert tl.range_anchor is None

            app.captured.clear()
            await pilot.press("enter")
            bars = _bar_events(app)
            assert bars[-1].record_key == tl.selected

    async def test_keyboard_range_includes_both_intervals_when_selected_backward(
        self,
    ) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            tl.focus()
            tl.set_selected("message:a1")

            await pilot.press("b")
            await pilot.press("k")
            await pilot.press("b")

            assert tl.brush == (_T0, _T0 + 18.0)

    async def test_clicking_semantic_label_column_is_inert(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            tl.brush_columns(LANE_LABEL_WIDTH + 1, LANE_LABEL_WIDTH + 4)
            await pilot.pause()
            brush = tl.brush
            app.captured.clear()

            clicked = await pilot.click(tl, offset=(1, 0))
            assert clicked
            await pilot.pause()
            assert _bar_events(app) == []
            assert _brush_events(app) == []
            assert tl.brush == brush

    async def test_set_snapshot_initializes_viewport_to_domain(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            assert tl.model.has_data
            assert tl.viewport == tl.model.domain
            assert tl.brush is None

    async def test_untimed_snapshot_renders_placeholder(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(untimed_snapshot())
            await pilot.pause()
            assert not tl.model.has_data
            assert "no timing data" in str(tl.render()).lower()

    async def test_render_draws_bars_and_caption(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            lines = str(tl.render()).splitlines()
            assert len(lines) == 6  # 4 lanes + axis + caption
            assert "Input" in lines[0] and "◆" in lines[0]
            assert "no brush" in lines[-1]

    async def test_click_on_bar_selects_and_clears_brush(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            # Pre-existing brush must be cleared by a bar click.
            tl.brush_columns(4, 8)
            await pilot.pause()
            app.captured.clear()

            # u1 spans columns ~3..44 on lane 0 at width 80.
            clicked = await pilot.click(tl, offset=(10, 0))
            assert clicked
            await pilot.pause()
            bars = _bar_events(app)
            assert len(bars) == 1
            assert bars[0].record_key == "message:u1"
            cleared = _brush_events(app)
            assert cleared[-1].brush_range is None
            assert tl.brush is None

    async def test_click_empty_space_clears_brush(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            tl.brush_columns(4, 8)
            await pilot.pause()
            assert tl.brush is not None
            app.captured.clear()

            # Column 48, lane 0 sits in the gap between u1 and a1.
            await pilot.click(tl, offset=(48, 0))
            await pilot.pause()
            assert _bar_events(app) == []
            assert _brush_events(app)[-1].brush_range is None
            assert tl.brush is None

    async def test_drag_brush_posts_range_and_counts_active(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()

            tl.brush_columns(5, 30)  # same seam the mouse-drag handlers use
            await pilot.pause()
            events = _brush_events(app)
            assert events, "brush drag should post TrajectoryBrushChanged"
            lo, hi = events[-1].brush_range
            assert lo < hi
            assert [r.seq for r in tl.model.records_in_range(lo, hi)] == [1]
            assert tl.brush == (lo, hi)

            # The caption now reports the brush instead of "no brush".
            caption = str(tl.render()).splitlines()[-1]
            assert "no brush" not in caption
            assert "1 active" in caption

    async def test_zoom_keys_change_window_and_post_viewport(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            tl.focus()
            domain = tl.model.domain
            assert domain is not None

            await pilot.press("]")  # zoom in at center
            await pilot.pause()
            events = _viewport_events(app)
            assert len(events) == 1
            assert events[0].domain_window == tl.viewport
            assert tl.viewport[1] - tl.viewport[0] < domain[1] - domain[0]

            await pilot.press("[")  # zoom back out: returns to the domain
            await pilot.pause()
            assert tl.viewport == domain
            assert _viewport_events(app)[-1].domain_window == domain

    async def test_pan_keys_clamp_at_domain_edges(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            tl.focus()
            domain = tl.model.domain
            assert domain is not None

            await pilot.press(",")  # pan left at full domain: no-op
            await pilot.pause()
            assert tl.viewport == domain
            assert _viewport_events(app) == []

            await pilot.press("]")  # zoom in, then pan left hits the edge
            await pilot.pause()
            app.captured.clear()
            await pilot.press(",")
            await pilot.pause()
            assert tl.viewport[0] == domain[0]
            panned = tl.viewport
            await pilot.press(",")  # further pans stay clamped
            await pilot.pause()
            assert tl.viewport == panned

            await pilot.press(".")  # pan right moves the window
            await pilot.pause()
            assert tl.viewport[0] > domain[0]

    async def test_zoom_at_mouse_focal(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            domain = tl.model.domain
            assert domain is not None
            before = tl.viewport
            tl.zoom_at(ZOOM_FACTOR, focal_fraction=0.75)  # wheel-up seam
            await pilot.pause()
            after = tl.viewport
            assert after != before
            span = domain[1] - domain[0]
            focal = domain[0] + 0.75 * span
            # 75% point of the old window stays at 75% of the new window.
            assert after[0] + 0.75 * (after[1] - after[0]) == pytest.approx(focal)
            assert _viewport_events(app)[-1].domain_window == after


def _mouse(
    tl: TrajectoryTimeline,
    event_cls: type,
    x: float,
    y: float,
    *,
    button: int = 0,
) -> object:
    """Build a widget-relative MouseEvent of the given class for posting."""
    return event_cls(
        widget=tl,
        x=x,
        y=y,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
    )


class TestTimelineMouseGlue:
    """Hand-posted real mouse events through the handler glue.

    The pilot has no drag/scroll primitives, so these drive
    ``MouseDown``/``MouseMove``/``MouseUp``/``MouseScroll*`` events
    directly at the mounted widget (widget-relative coordinates, the
    same shape Textual's own dispatch delivers).
    """

    async def test_drag_brush_via_posted_events_captures_mouse(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()

            tl.post_message(_mouse(tl, events.MouseDown, 10, 0, button=1))
            await pilot.pause()
            assert app.mouse_captured is tl  # gesture keeps the moves

            tl.post_message(_mouse(tl, events.MouseMove, 30, 0, button=1))
            await pilot.pause()
            tl.post_message(_mouse(tl, events.MouseUp, 30, 0, button=1))
            await pilot.pause()

            assert app.mouse_captured is None  # released on mouse up
            assert _bar_events(app) == []
            events_ = _brush_events(app)
            assert events_, "down+move+up must post brush changes"
            lo, hi = events_[-1].brush_range
            assert lo < hi
            assert tl.brush == (lo, hi)

    async def test_move_without_prior_down_is_ignored(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()

            tl.post_message(_mouse(tl, events.MouseMove, 30, 0, button=1))
            await pilot.pause()
            assert tl.brush is None
            assert _brush_events(app) == []
            assert app.mouse_captured is None

    async def test_scroll_events_zoom_at_mouse_x(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()
            domain = tl.model.domain
            assert domain is not None
            span = domain[1] - domain[0]

            tl.post_message(_mouse(tl, events.MouseScrollUp, 60, 0))
            await pilot.pause()
            after = tl.viewport
            assert after is not None
            assert after != domain
            # Column 60 (fraction (60+0.5)/80) stays fixed by the zoom.
            f = (60 + 0.5) / 80
            assert after[0] + f * (after[1] - after[0]) == pytest.approx(
                domain[0] + f * span
            )
            assert _viewport_events(app)[-1].domain_window == after

            tl.post_message(_mouse(tl, events.MouseScrollDown, 60, 0))
            await pilot.pause()
            assert tl.viewport == domain  # zoom back out: full domain

    async def test_click_vs_drag_discrimination(self) -> None:
        app = TimelineApp()
        async with app.run_test(size=(80, 24)) as pilot:
            tl = app.query_one(TrajectoryTimeline)
            tl.set_snapshot(timed_snapshot())
            await pilot.pause()

            # Down + up at the same spot on u1's bar: a click, not a drag.
            tl.post_message(_mouse(tl, events.MouseDown, 10, 0, button=1))
            await pilot.pause()
            tl.post_message(_mouse(tl, events.MouseUp, 10, 0, button=1))
            await pilot.pause()
            bars = _bar_events(app)
            assert len(bars) == 1 and bars[0].record_key == "message:u1"
            assert tl.brush is None

            # Down + move + up: a drag, never a bar selection.
            app.captured.clear()
            tl.post_message(_mouse(tl, events.MouseDown, 10, 0, button=1))
            tl.post_message(_mouse(tl, events.MouseMove, 30, 0, button=1))
            tl.post_message(_mouse(tl, events.MouseUp, 30, 0, button=1))
            await pilot.pause()
            assert _bar_events(app) == []
            assert _brush_events(app)
            assert tl.brush is not None


def test_feedback_records_get_their_own_timeline_style():
    """task-17169: user_feedback falls back to plain white otherwise, which
    makes review events indistinguishable from an unknown/unhandled kind in
    the one view meant to surface them."""
    from tldw_chatbook.Chat.trajectory import KIND_USER_FEEDBACK
    from tldw_chatbook.UI.Widgets.trajectory_timeline import (
        _FALLBACK_STYLE,
        KIND_STYLES,
    )

    assert KIND_USER_FEEDBACK in KIND_STYLES
    assert KIND_STYLES[KIND_USER_FEEDBACK] != _FALLBACK_STYLE
