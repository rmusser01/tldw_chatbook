"""Structured Trace filters and stable anomaly navigation contracts."""

from __future__ import annotations

import contextlib
import importlib
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Select, Static

from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
)
from tldw_chatbook.css import build_css
from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen


def _filters_module():
    try:
        return importlib.import_module("tldw_chatbook.UI.Widgets.trace_filter_bar")
    except ModuleNotFoundError:
        pytest.fail("TraceFilterState/TraceFilterBar production module is missing")


def _record(
    seq: int,
    kind: str,
    *,
    status: str = "completed",
    actor_kind: str | None = None,
    actor_id: str | None = None,
    provider: str | None = None,
    start: float | None = None,
    parent_event_id: str | None = None,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        seq=seq,
        kind=kind,
        turn_id="turn-1",
        message_id=f"message-{seq}",
        content_preview=f"{kind} event {seq}",
        usage=None,
        step_started_at=start,
        first_token_at=None,
        completed_at=None if start is None else start + 0.5,
        model=None,
        provider=provider,
        payload=None,
        variants=(),
        depth=0,
        event_id=f"event:{seq}",
        status=status,
        actor_kind=actor_kind,
        actor_id=actor_id,
        run_id=actor_id if actor_kind == "agent" else None,
        parent_event_id=parent_event_id,
    )


def filter_snapshot() -> TrajectorySnapshot:
    records = (
        _record(1, "user", actor_kind="user", start=10.0),
        _record(2, "assistant", actor_kind="model", provider="openai", start=20.0),
        _record(3, "provider_error", status="failed", provider="openai", start=30.0),
        _record(4, "tool_call", status="pending", start=40.0),
        _record(5, "user_feedback", actor_kind="user", start=50.0),
        _record(
            6,
            "agent_step",
            actor_kind="agent",
            actor_id="child-1",
            start=60.0,
            parent_event_id="agent-run:parent",
        ),
        _record(7, "assistant", actor_kind="model", provider="anthropic", start=None),
    )
    return TrajectorySnapshot((TrajectoryTurn("turn-1", records),))


_CSS_DIR = Path(build_css.__file__).parent
_SCOPED_CSS, _SELF_CSS = build_css.screen_css_paths(_CSS_DIR)


class _TraceHost(App[None]):
    CSS_PATH = [
        str(_SCOPED_CSS),
        str(_CSS_DIR / "tldw_cli_modular.tcss"),
        str(_SELF_CSS),
    ]

    def compose(self) -> ComposeResult:
        yield Static("Console")


@contextlib.asynccontextmanager
async def _mounted(*, size: tuple[int, int] = (100, 30), snapshot=None):
    app = _TraceHost()
    async with app.run_test(size=size) as pilot:
        screen = TrajectoryScreen(snapshot or filter_snapshot())
        await app.push_screen(screen)
        await pilot.pause()
        yield app, pilot, screen


def _cursor_record(screen: TrajectoryScreen) -> TrajectoryRecord | None:
    key = screen._cursor_key()
    return screen._row_records.get(key) if key else None


def test_trace_filter_state_matches_every_dimension_with_and_semantics() -> None:
    module = _filters_module()
    state = module.TraceFilterState(
        kind="assistant",
        status="completed",
        agent="model",
        provider="openai",
        time_range=(15.0, 25.0),
    )

    assert state.matches(filter_snapshot().turns[0].records[1])
    for record in (
        filter_snapshot().turns[0].records[0],
        filter_snapshot().turns[0].records[2],
        filter_snapshot().turns[0].records[6],
    ):
        assert not state.matches(record)


def test_trace_filter_state_derives_options_and_active_summary() -> None:
    module = _filters_module()
    records = filter_snapshot().turns[0].records
    options = module.TraceFilterState.options_from(records)

    assert options.kinds == tuple(sorted({record.kind for record in records}))
    assert options.statuses == ("completed", "failed", "pending")
    assert options.agents == ("agent:child-1", "model", "user")
    assert options.providers == ("anthropic", "openai")
    state = module.TraceFilterState(kind="tool_call", status="pending")
    assert state.active_count == 2
    assert "Tool call" in state.summary
    assert "Pending" in state.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("size,compact", [((80, 24), True), ((120, 35), False)])
async def test_filter_bar_uses_one_state_owner_in_compact_and_wide_layouts(
    size: tuple[int, int], compact: bool
) -> None:
    module = _filters_module()
    async with _mounted(size=size) as (app, pilot, screen):
        bars = list(screen.query(module.TraceFilterBar))
        assert len(bars) == 1
        bar = bars[0]
        assert bar.compact is compact
        assert "7/7" in str(bar.render())
        visible_selects = [
            select for select in bar.query(Select) if select.region.width
        ]
        assert len(visible_selects) == (0 if compact else 4)
        assert screen.query_one("#trajectory-table", DataTable).max_scroll_x == 0


@pytest.mark.asyncio
async def test_wide_filter_bar_paints_visible_and_total_counts() -> None:
    async with _mounted(size=(100, 30)) as (app, pilot, screen):
        count_widgets = list(screen.query("#trace-filter-counts"))

        assert len(count_widgets) == 1
        counts = count_widgets[0]
        assert counts.region.width > 0
        assert "Filters 7/7" in str(counts.render())
        assert "7/7" in app.export_screenshot(simplify=True)


@pytest.mark.asyncio
async def test_compact_filter_action_opens_keyboard_dismissible_dialog() -> None:
    _filters_module()
    async with _mounted(size=(60, 18)) as (app, pilot, screen):
        assert hasattr(screen, "action_open_filters")
        await screen.action_open_filters()
        assert type(app.screen).__name__ == "TraceFiltersDialog"
        await pilot.press("escape")
        assert app.screen is screen


@pytest.mark.asyncio
async def test_structured_dimensions_filter_the_ledger_and_report_counts() -> None:
    module = _filters_module()
    async with _mounted() as (app, pilot, screen):
        bar = screen.query_one(module.TraceFilterBar)
        bar.set_state(module.TraceFilterState(provider="openai"))
        await pilot.pause()

        visible = tuple(record for record in screen._row_records.values() if record)
        assert {record.event_id for record in visible} == {"event:2", "event:3"}
        assert bar.visible_count == 2
        assert bar.total_count == 7
        assert "2/7" in bar.summary_text


@pytest.mark.asyncio
async def test_search_filter_and_timeline_time_range_compose_with_and_semantics() -> (
    None
):
    module = _filters_module()
    async with _mounted() as (app, pilot, screen):
        bar = screen.query_one(module.TraceFilterBar)
        screen.query_one("#trajectory-search").value = "assistant"
        bar.set_state(
            module.TraceFilterState(provider="openai", time_range=(15.0, 25.0))
        )
        await pilot.pause()

        visible = tuple(record for record in screen._row_records.values() if record)
        assert [record.event_id for record in visible] == ["event:2"]
        assert bar.visible_count == 1


@pytest.mark.asyncio
async def test_clear_all_clears_search_structured_filters_and_brush_truth() -> None:
    module = _filters_module()
    async with _mounted() as (app, pilot, screen):
        bar = screen.query_one(module.TraceFilterBar)
        screen.query_one("#trajectory-search").value = "assistant"
        bar.set_state(module.TraceFilterState(status="failed", time_range=(20.0, 40.0)))
        screen._timeline.apply_brush((20.0, 40.0))
        await pilot.pause()

        screen.action_clear_filters()
        await pilot.pause()
        assert screen.query_one("#trajectory-search").value == ""
        assert not bar.state.is_active
        assert screen._timeline.brush is None
        assert bar.visible_count == bar.total_count == 7


@pytest.mark.asyncio
async def test_navigation_bindings_cover_both_directions_for_every_family() -> None:
    async with _mounted() as (app, pilot, screen):
        actions = {binding.action for binding in screen.BINDINGS}
        assert {
            "next_match",
            "previous_match",
            "next_error",
            "previous_error",
            "next_tool",
            "previous_tool",
            "next_feedback",
            "previous_feedback",
            "next_child_agent",
            "previous_child_agent",
        } <= actions


@pytest.mark.asyncio
async def test_compact_contextual_hints_fit_without_duplicate_navigation_copy() -> None:
    async with _mounted(size=(60, 18)) as (app, pilot, screen):
        hints_widget = screen.query_one("#trajectory-hints", Static)
        lines = str(hints_widget.render()).splitlines()

        assert hints_widget.size.height == 2
        assert len(lines) == 2
        assert all(len(line) <= hints_widget.size.width for line in lines)
        assert "n/p match" in lines[0]
        assert "n next match" not in "\n".join(lines)
        assert "g filters" in lines[1]


@pytest.mark.asyncio
async def test_next_previous_match_use_filtered_order_and_wrap_explicitly() -> None:
    async with _mounted() as (app, pilot, screen):
        screen.query_one("#trajectory-search").value = "assistant"
        await pilot.pause()

        screen.action_next_match()
        assert _cursor_record(screen).event_id == "event:2"
        screen.action_next_match()
        assert _cursor_record(screen).event_id == "event:7"
        screen.action_next_match()
        assert _cursor_record(screen).event_id == "event:2"
        screen.action_previous_match()
        assert _cursor_record(screen).event_id == "event:7"


@pytest.mark.asyncio
async def test_error_navigation_preserves_order_and_wraps_across_multiple_members() -> (
    None
):
    base = filter_snapshot().turns[0].records
    second_error = _record(
        8, "agent_failed", status="failed", actor_kind="agent", start=70.0
    )
    snapshot = TrajectorySnapshot((TrajectoryTurn("turn-1", base + (second_error,)),))
    async with _mounted(snapshot=snapshot) as (app, pilot, screen):
        screen.action_next_error()
        assert _cursor_record(screen).event_id == "event:3"
        screen.action_next_error()
        assert _cursor_record(screen).event_id == "event:8"
        screen.action_next_error()
        assert _cursor_record(screen).event_id == "event:3"
        screen.action_previous_error()
        assert _cursor_record(screen).event_id == "event:8"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forward,backward,event_id",
    [
        ("action_next_error", "action_previous_error", "event:3"),
        ("action_next_tool", "action_previous_tool", "event:4"),
        ("action_next_feedback", "action_previous_feedback", "event:5"),
        ("action_next_child_agent", "action_previous_child_agent", "event:6"),
    ],
)
async def test_anomaly_navigation_wraps_both_directions_on_filtered_records(
    forward: str, backward: str, event_id: str
) -> None:
    async with _mounted() as (app, pilot, screen):
        assert hasattr(screen, forward) and hasattr(screen, backward)
        getattr(screen, forward)()
        assert _cursor_record(screen).event_id == event_id
        getattr(screen, forward)()  # one member => explicit wrap to itself
        assert _cursor_record(screen).event_id == event_id
        getattr(screen, backward)()
        assert _cursor_record(screen).event_id == event_id


@pytest.mark.asyncio
async def test_live_insertion_before_selection_preserves_event_id_filters_and_viewport() -> (
    None
):
    module = _filters_module()
    async with _mounted() as (app, pilot, screen):
        bar = screen.query_one(module.TraceFilterBar)
        bar.set_state(module.TraceFilterState(provider="openai"))
        await pilot.pause()
        screen._move_cursor_to_key("event:3")
        screen._timeline.zoom_at(2.0)
        viewport = screen._timeline.viewport

        before = filter_snapshot().turns[0].records
        inserted = _record(99, "system", actor_kind="system", start=5.0)
        renumbered = tuple(
            record if record.event_id == "event:3" else record for record in before
        )
        refreshed = TrajectorySnapshot(
            (TrajectoryTurn("turn-1", (inserted,) + renumbered),)
        )
        screen._follow = False
        screen._apply_live_snapshot(refreshed)
        await pilot.pause()

        assert _cursor_record(screen).event_id == "event:3"
        assert bar.state.provider == "openai"
        assert screen._timeline.viewport == viewport
