"""Integration tests: brushable timeline strip inside TrajectoryScreen (task-16315).

Covers the screen <-> :class:`TrajectoryTimeline` seams only (geometry and
standalone widget behaviour live in ``test_trajectory_timeline.py``):

- the strip mounts between the search box and the ledger, always;
- a brush filters the ledger to ACTIVE-in-range records, composing with
  the search query (AND), and clearing restores;
- bar click moves the ledger cursor (paging in older records when
  needed) and clears only the brush -- the search stays;
- ledger cursor moves highlight the record's bar;
- live refreshes feed the strip the new snapshot.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from textual.widgets import DataTable, Input, Static

from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.UI.Widgets.trajectory_timeline import (
    LANE_LABEL_WIDTH,
    TrajectoryTimeline,
)
from tldw_chatbook.UI.Widgets.trace_filter_bar import TraceFilterState

# Same duck-typed projection stand-ins as the screen tests (Tests is a
# package, so the sibling module is importable).
from Tests.UI.test_trajectory_screen import (  # noqa: I001
    _T0,
    _mounted,
    _record_key_for_seq,
    base_snapshot,
    many_records_snapshot,
    msg,
    TrajRow,
)


def untimed_snapshot():
    """Two records with NULL timing: the strip must show its placeholder."""
    messages = [
        msg("u1", "user", content="hello", ts=_T0),
        msg("a1", "assistant", content="hi", ts=_T0 + 1.0, parent="u1"),
    ]
    rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
        TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
    ]
    return derive_trajectory(messages, {}, rows, [], [])


def disjoint_snapshot():
    """All timing far beyond base_snapshot's domain (a full snapshot swap)."""
    base = _T0 + 100_000.0
    messages = [
        msg("u1", "user", content="later hello", ts=base),
        msg("a1", "assistant", content="later answer", ts=base + 1.0, parent="u1"),
    ]
    rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user", step_started_at=base),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=2,
            event_kind="assistant",
            step_started_at=base,
            completed_at=base + 2.0,
        ),
    ]
    return derive_trajectory(messages, {}, rows, [], [])


def grown_snapshot():
    """base_snapshot plus a third, later turn (all timed)."""
    messages = [
        msg("u1", "user", content="hello trajectory world", ts=_T0),
        msg(
            "a1",
            "assistant",
            content="checking that for you",
            ts=_T0 + 2.0,
            parent="u1",
        ),
        msg(
            "u2",
            "user",
            content="second question about zebras",
            ts=_T0 + 60.0,
            parent="a1",
        ),
        msg(
            "a2", "assistant", content="zebras have stripes", ts=_T0 + 65.0, parent="u2"
        ),
        msg("u3", "user", content="third question", ts=_T0 + 120.0, parent="a2"),
        msg("a3", "assistant", content="third answer", ts=_T0 + 122.0, parent="u3"),
    ]
    rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user", step_started_at=_T0),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=2,
            event_kind="assistant",
            step_started_at=_T0,
            completed_at=_T0 + 5.0,
        ),
        TrajRow(
            "u2", turn_id="t2", seq=3, event_kind="user", step_started_at=_T0 + 60.0
        ),
        TrajRow(
            "a2",
            turn_id="t2",
            seq=4,
            event_kind="assistant",
            step_started_at=_T0 + 65.0,
            completed_at=_T0 + 70.0,
        ),
        TrajRow(
            "u3", turn_id="t3", seq=5, event_kind="user", step_started_at=_T0 + 120.0
        ),
        TrajRow(
            "a3",
            turn_id="t3",
            seq=6,
            event_kind="assistant",
            step_started_at=_T0 + 120.0,
            completed_at=_T0 + 123.0,
        ),
    ]
    return derive_trajectory(messages, {}, rows, [], [])


async def _brush(pilot, timeline: TrajectoryTimeline, lo: float, hi: float) -> None:
    timeline.post_message(TrajectoryTimeline.TrajectoryBrushChanged((lo, hi)))
    await pilot.pause()


# ---------------------------------------------------------------------------
# Layout + data feed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeline_mounts_between_search_and_table() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        ids = [widget.id for widget in screen.query_one("#trajectory-screen").children]
        assert ids.index("trajectory-search") < ids.index("trajectory-timeline")
        assert ids.index("trajectory-timeline") < ids.index("trajectory-table")


@pytest.mark.asyncio
async def test_timeline_renders_bars_for_timed_snapshot() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        assert timeline.size.height == 6  # the 6-line strip
        assert "Input" in str(timeline.render())
        assert "◆" in str(timeline.render())


@pytest.mark.asyncio
async def test_timeline_focus_exposes_truthful_keyboard_equivalents_for_mouse_actions() -> (
    None
):
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        timeline.focus()
        await pilot.pause()

        hints = str(screen.query_one("#trajectory-hints", Static).render())
        assert "j/k event" in hints
        assert "enter select" in hints
        assert "b range" in hints
        assert "[/] zoom" in hints
        assert ",/. pan" in hints


@pytest.mark.asyncio
async def test_timeline_shows_placeholder_without_timing() -> None:
    async with _mounted(untimed_snapshot()) as (app, pilot, screen):
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        assert "no timing data" in str(timeline.render()).lower()


@pytest.mark.asyncio
async def test_escape_clears_active_range_before_dismiss_even_from_ledger() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        domain = timeline.model.domain
        assert domain is not None
        timeline.apply_brush(domain)
        screen.query_one("#trajectory-table", DataTable).focus()
        await pilot.pause()

        await pilot.press("escape")

        assert app.screen is screen
        assert timeline.brush is None
        assert screen._filter_bar.state.time_range is None


@pytest.mark.asyncio
async def test_filter_owner_time_clear_also_clears_the_visual_brush() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        domain = timeline.model.domain
        assert domain is not None
        timeline.apply_brush(domain)
        await pilot.pause()
        assert screen._filter_bar.state.time_range == domain

        screen._filter_bar.clear()
        await pilot.pause()

        assert timeline.brush is None
        assert screen._filter_bar.state.time_range is None


# ---------------------------------------------------------------------------
# Brush filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brush_filters_ledger_to_active_records() -> None:
    """A range over turn 1 keeps seq 1-4; u2 (outside) and a2 (untimed) drop."""
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        assert table.row_count == 8
        await _brush(pilot, timeline, _T0 - 1.0, _T0 + 6.0)
        # Turn 1 header + its 4 records; turn 2 header hidden (no child
        # visible) exactly like the search rule.
        assert table.row_count == 5
        assert table.get_row_index("turn:t1") == 0
        for seq in (1, 2, 3, 4):
            assert table.get_row_index(_record_key_for_seq(screen, seq)) is not None
        for seq in (5, 6):
            with pytest.raises(Exception):
                table.get_row_index(_record_key_for_seq(screen, seq))


@pytest.mark.asyncio
async def test_brush_composes_with_search_as_and() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        search = screen.query_one("#trajectory-search", Input)
        search.value = "zebras"  # only seq 5/6 match
        await pilot.pause()
        assert table.row_count == 3
        await _brush(pilot, timeline, _T0 - 1.0, _T0 + 6.0)  # only seq 1-4 active
        assert table.row_count == 0  # intersection is empty: headers hidden too


@pytest.mark.asyncio
async def test_brush_clear_restores_all_rows() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        await _brush(pilot, timeline, _T0 - 1.0, _T0 + 6.0)
        assert table.row_count == 5
        timeline.post_message(TrajectoryTimeline.TrajectoryBrushChanged(None))
        await pilot.pause()
        assert table.row_count == 8


@pytest.mark.asyncio
async def test_widget_drag_seam_posts_brush_that_filters() -> None:
    """brush_columns (the mouse-drag seam) reaches the ledger too."""
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        timeline.brush_columns(
            LANE_LABEL_WIDTH, LANE_LABEL_WIDTH + 10
        )  # left edge of the plot: turn 1 only
        await pilot.pause()
        assert screen._filter_bar.state.time_range == timeline.brush
        assert table.row_count == 5
        # The strip's caption doubles as the brush status note.
        assert "active" in str(timeline.render())


@pytest.mark.asyncio
async def test_brush_reveals_collapsed_turn_with_active_children() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        await pilot.press("t")  # collapse turn 1 (cursor starts on its header)
        await pilot.pause()
        assert table.row_count == 4
        await _brush(pilot, timeline, _T0 - 1.0, _T0 + 6.0)
        # Same reveal rule as search: filtering shows the active children.
        assert table.row_count == 5


# ---------------------------------------------------------------------------
# Selection sync (both ways)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bar_select_moves_ledger_cursor_and_highlights() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        timeline.post_message(
            TrajectoryTimeline.TrajectoryBarSelected(_record_key_for_seq(screen, 3))
        )
        await pilot.pause()
        assert table.cursor_row == table.get_row_index(_record_key_for_seq(screen, 3))
        assert timeline.selected == _record_key_for_seq(screen, 3)


@pytest.mark.asyncio
async def test_click_outside_active_brush_atomically_clears_time_and_selects(
    monkeypatch,
) -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        await _brush(pilot, timeline, _T0 - 1.0, _T0 + 6.0)
        target = next(record for record in screen._all_records() if record.seq == 5)
        cols = timeline._record_columns(
            target, timeline._plot_width(), timeline.viewport
        )
        assert cols is not None
        notices: list[str] = []
        monkeypatch.setattr(
            app, "notify", lambda message, **_kwargs: notices.append(str(message))
        )
        render_count = 0
        render_ledger = screen._render_ledger

        def counted_render() -> None:
            nonlocal render_count
            render_count += 1
            render_ledger()

        monkeypatch.setattr(screen, "_render_ledger", counted_render)

        clicked = await pilot.click(
            timeline,
            offset=(LANE_LABEL_WIDTH + cols[0], timeline.model.lane_for(target)),
        )
        assert clicked
        await pilot.pause()

        target_key = _record_key_for_seq(screen, 5)
        assert screen._filter_bar.state.time_range is None
        assert timeline.brush is None
        assert screen._cursor_key() == target_key
        assert timeline.selected == target_key
        assert render_count == 1
        assert not any("hidden" in notice.lower() for notice in notices)


@pytest.mark.asyncio
async def test_keyboard_enter_outside_active_brush_uses_same_selection_transaction(
    monkeypatch,
) -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        await _brush(pilot, timeline, _T0 - 1.0, _T0 + 6.0)
        timeline.focus()
        timeline.set_selected(_record_key_for_seq(screen, 4))
        notices: list[str] = []
        monkeypatch.setattr(
            app, "notify", lambda message, **_kwargs: notices.append(str(message))
        )
        render_count = 0
        render_ledger = screen._render_ledger

        def counted_render() -> None:
            nonlocal render_count
            render_count += 1
            render_ledger()

        monkeypatch.setattr(screen, "_render_ledger", counted_render)

        await pilot.press("j", "enter")
        await pilot.pause()

        target_key = _record_key_for_seq(screen, 5)
        assert screen._filter_bar.state.time_range is None
        assert timeline.brush is None
        assert screen._cursor_key() == target_key
        assert timeline.selected == target_key
        assert render_count == 1
        assert not any("hidden" in notice.lower() for notice in notices)


@pytest.mark.asyncio
async def test_bar_message_hidden_by_search_restores_shared_ledger_selection(
    monkeypatch,
) -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        search = screen.query_one("#trajectory-search", Input)
        search.value = "zebras"
        await pilot.pause()
        visible_key = _record_key_for_seq(screen, 6)
        hidden_key = _record_key_for_seq(screen, 2)
        table.move_cursor(row=table.get_row_index(visible_key), animate=False)
        await pilot.pause()
        notices: list[str] = []
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, **_kwargs: notices.append(str(message)),
        )

        timeline.set_selected(hidden_key)
        timeline.post_message(TrajectoryTimeline.TrajectoryBarSelected(hidden_key))
        await pilot.pause()

        assert search.value == "zebras"
        assert screen._cursor_key() == visible_key
        assert timeline.selected == visible_key
        assert notices and "hidden" in notices[-1].lower()


@pytest.mark.asyncio
async def test_timeline_enter_hidden_by_provider_restores_shared_selection(
    monkeypatch,
) -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        screen._filter_bar.set_state(TraceFilterState(provider="test-provider"))
        await pilot.pause()
        visible_key = _record_key_for_seq(screen, 2)
        hidden_key = _record_key_for_seq(screen, 3)
        table.move_cursor(row=table.get_row_index(visible_key), animate=False)
        await pilot.pause()
        notices: list[str] = []
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, **_kwargs: notices.append(str(message)),
        )
        timeline.focus()
        timeline.set_selected(hidden_key)

        await pilot.press("enter")
        await pilot.pause()

        assert screen._filter_bar.state.provider == "test-provider"
        assert screen._cursor_key() == visible_key
        assert timeline.selected == visible_key
        assert notices and "hidden" in notices[-1].lower()


@pytest.mark.asyncio
async def test_bar_select_pages_in_older_record() -> None:
    snapshot = many_records_snapshot(record_count=600)
    async with _mounted(snapshot) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        with pytest.raises(Exception):
            table.get_row_index(_record_key_for_seq(screen, 1))
        timeline.post_message(
            TrajectoryTimeline.TrajectoryBarSelected(_record_key_for_seq(screen, 1))
        )
        await pilot.pause()
        key = _record_key_for_seq(screen, 1)
        assert table.get_row_index(key) is not None
        assert table.cursor_row == table.get_row_index(key)


@pytest.mark.asyncio
async def test_ledger_cursor_move_highlights_timeline_bar() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        table.move_cursor(row=table.get_row_index(_record_key_for_seq(screen, 6)))
        await pilot.pause()
        assert timeline.selected == _record_key_for_seq(screen, 6)
        table.move_cursor(row=0)  # back to the turn header
        await pilot.pause()
        assert timeline.selected is None


# ---------------------------------------------------------------------------
# Live refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_refresh_feeds_timeline_and_preserves_brush() -> None:
    """A revision tick must not destroy an active brush.

    The grown snapshot extends the domain (appends); a brush still
    intersecting it survives on BOTH sides and keeps filtering the
    rebuilt ledger.
    """
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        await _brush(pilot, timeline, _T0 - 1.0, _T0 + 6.0)
        assert table.row_count == 5
        screen._apply_live_snapshot(grown_snapshot())
        await pilot.pause()
        # The strip renders the new snapshot's records...
        assert len(timeline.model.timed_records) == 6
        # ...and the brush survives: widget, screen state and ledger
        # filter all still agree (grown seqs 1/2 are the turn-1 records).
        assert timeline.brush == (_T0 - 1.0, _T0 + 6.0)
        assert screen._filter_bar.state.time_range == (_T0 - 1.0, _T0 + 6.0)
        assert table.row_count == 3  # turn:t1 header + records 1, 2


@pytest.mark.asyncio
async def test_live_refresh_clears_brush_outside_new_domain() -> None:
    """A brush disjoint from the swapped-in domain clears cleanly."""
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        timeline = screen.query_one("#trajectory-timeline", TrajectoryTimeline)
        await _brush(pilot, timeline, _T0 - 1.0, _T0 + 6.0)
        assert table.row_count == 5
        screen._apply_live_snapshot(disjoint_snapshot())
        await pilot.pause()
        assert screen._filter_bar.state.time_range is None
        assert timeline.brush is None
        assert table.row_count == 3  # unfiltered: header + 2 records


# ---------------------------------------------------------------------------
# Brush caption clock matches the ledger's local-time columns
# ---------------------------------------------------------------------------


def test_widget_clock_formats_local_time_like_ledger() -> None:
    from tldw_chatbook.UI.Widgets.trajectory_timeline import _fmt_clock as strip_clock
    from tldw_chatbook.UI.Screens.trajectory_screen import _fmt_clock as ledger_clock

    assert strip_clock(_T0) == ledger_clock(_T0)
    assert strip_clock(_T0) == datetime.fromtimestamp(_T0).strftime("%H:%M:%S")
