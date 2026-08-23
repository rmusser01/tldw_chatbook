"""Structured Trace filters and stable anomaly navigation contracts."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from html import unescape
import importlib
from pathlib import Path
import re

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Static

from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
)
from tldw_chatbook.css import build_css
from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen
from tldw_chatbook.UI.Widgets.trajectory_timeline import TrajectoryTimeline
from Tests.UI.test_trajectory_screen import many_records_snapshot


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
    run_id: str | None = None,
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
        run_id=run_id
        or (
            actor_id
            if actor_kind in {"agent", "primary", "subagent", "child_agent"}
            else None
        ),
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
            actor_kind="subagent",
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


def _inside(child, parent) -> bool:
    return (
        child.region.x >= parent.content_region.x
        and child.region.y >= parent.content_region.y
        and child.region.right <= parent.content_region.right
        and child.region.bottom <= parent.content_region.bottom
    )


def _painted_text(app: App) -> str:
    svg = app.export_screenshot(simplify=True)
    return unescape(re.sub(r"<[^>]+>", " ", svg)).replace("\N{NO-BREAK SPACE}", " ")


def test_trace_filter_state_matches_every_dimension_with_and_semantics() -> None:
    module = _filters_module()
    matching = _record(
        99,
        "assistant",
        actor_kind="subagent",
        actor_id="child-1",
        provider="openai",
        start=20.0,
    )
    state = module.TraceFilterState(
        kind="assistant",
        status="completed",
        agent="child-1",
        provider="openai",
        time_range=(15.0, 25.0),
    )

    assert state.matches(matching)
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
    assert options.agents == ("child-1",)
    assert options.providers == ("anthropic", "openai")
    agent_state = module.TraceFilterState(agent="child-1")
    assert agent_state.matches(records[5])
    assert not agent_state.matches(records[0])
    assert not agent_state.matches(records[1])
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
        assert "7 shown" in str(bar.render())
        assert "7 matches" in str(bar.render())
        assert "7 total" in str(bar.render())
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
        assert str(counts.render()).splitlines() == [
            "Shown 7",
            "Matches 7",
            "Total 7",
        ]
        painted = _painted_text(app)
        assert "Shown 7" in painted
        assert "Matches 7" in painted
        assert "Total 7" in painted


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (120, 35)])
async def test_wide_filter_controls_are_contained_and_every_tab_target_is_visible(
    size: tuple[int, int],
) -> None:
    async with _mounted(size=size) as (app, pilot, screen):
        wide = screen.query_one("#trace-filter-wide")
        selects = list(wide.query(Select))

        assert len(selects) == 4
        assert all(_inside(select, wide) for select in selects)
        assert all(select in app.screen.focus_chain for select in selects)
        assert all(select.region.width > 0 for select in selects)
        assert screen.query_one("#trace-filter-bar") not in app.screen.focus_chain
        app.export_screenshot(simplify=True)


@pytest.mark.asyncio
async def test_wide_tab_chain_has_no_inert_filter_bar_stop() -> None:
    async with _mounted(size=(100, 30)) as (app, pilot, screen):
        relevant = [
            widget.id
            for widget in app.screen.focus_chain
            if isinstance(widget, (Input, Select, TrajectoryTimeline, DataTable))
        ]

        assert relevant[:7] == [
            "trajectory-search",
            "trace-filter-kind",
            "trace-filter-status",
            "trace-filter-agent",
            "trace-filter-provider",
            "trajectory-timeline",
            "trajectory-table",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 18), (80, 24), (100, 30), (120, 35)])
async def test_focused_timeline_outline_does_not_clip_semantic_content(
    size: tuple[int, int],
) -> None:
    async with _mounted(size=size) as (app, pilot, screen):
        timeline = screen.query_one(TrajectoryTimeline)
        timeline.focus()
        timeline.apply_brush(timeline.model.domain)
        await pilot.pause()

        assert timeline.content_region.size == timeline.region.size
        painted = _painted_text(app)
        for label in ("Input", "Model", "Tools", "Agents"):
            assert label in painted


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
async def test_compact_filter_bar_is_actionable_with_enter_and_paints_cue() -> None:
    async with _mounted(size=(60, 18)) as (app, pilot, screen):
        bar = screen.query_one("#trace-filter-bar")
        assert bar in app.screen.focus_chain
        bar.focus()
        await pilot.pause()
        assert "g filters" in _painted_text(app).lower()

        await pilot.press("enter")
        assert type(app.screen).__name__ == "TraceFiltersDialog"
        await pilot.press("escape")
        assert app.screen is screen


@pytest.mark.asyncio
async def test_compact_filter_dialog_contains_all_controls_and_actions_at_60x18() -> (
    None
):
    async with _mounted(size=(60, 18)) as (app, pilot, screen):
        await screen.action_open_filters()
        dialog_screen = app.screen
        dialog = dialog_screen.query_one("#trace-filter-dialog")
        selects = list(dialog.query(Select))
        buttons = list(dialog.query(Button))
        controls = [*selects, *buttons]

        assert _inside(dialog, dialog_screen)
        assert len(selects) == 4
        assert {button.label.plain for button in buttons} == {
            "Clear",
            "Cancel",
            "Apply",
        }
        assert all(_inside(control, dialog) for control in controls)
        assert all(control in dialog_screen.focus_chain for control in controls)
        painted = _painted_text(app)
        for label in (
            "Event kind",
            "State",
            "Agent",
            "Provider",
            "Clear",
            "Cancel",
            "Apply",
        ):
            assert label in painted

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
        assert "2 shown" in bar.summary_text
        assert "2 matches" in bar.summary_text
        assert "7 total" in bar.summary_text


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
        assert "w export" in lines[1]


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
async def test_anomaly_navigation_expands_collapsed_owning_turn_before_cursor_move() -> (
    None
):
    first = _record(1, "assistant", start=10.0)
    error = replace(
        _record(2, "provider_error", status="failed", start=20.0),
        turn_id="turn-2",
    )
    snapshot = TrajectorySnapshot(
        (
            TrajectoryTurn("turn-1", (first,)),
            TrajectoryTurn("turn-2", (error,)),
        )
    )
    async with _mounted(snapshot=snapshot) as (app, pilot, screen):
        screen._collapsed.add("turn-2")
        screen._render_ledger()
        await pilot.pause()
        assert "event:2" not in screen._visible_keys

        screen.action_next_error()
        await pilot.pause()

        assert "turn-2" not in screen._collapsed
        assert _cursor_record(screen).event_id == "event:2"


@pytest.mark.asyncio
async def test_child_navigation_uses_agent_run_lineage_and_excludes_primary_steps() -> (
    None
):
    records = (
        _record(1, "agent_run", actor_kind="primary", run_id="primary", start=10.0),
        _record(
            2,
            "agent_step",
            actor_kind="agent",
            run_id="primary",
            parent_event_id="agent-run:primary",
            start=11.0,
        ),
        _record(
            3,
            "agent_run",
            actor_kind="subagent",
            run_id="child",
            parent_event_id="agent-step:primary:1",
            start=12.0,
        ),
        _record(
            4,
            "agent_step",
            actor_kind="agent",
            run_id="child",
            parent_event_id="agent-run:child",
            start=13.0,
        ),
    )
    snapshot = TrajectorySnapshot((TrajectoryTurn("turn-1", records),))
    async with _mounted(snapshot=snapshot) as (app, pilot, screen):
        screen.action_next_child_agent()
        assert _cursor_record(screen).event_id == "event:3"
        screen.action_next_child_agent()
        assert _cursor_record(screen).event_id == "event:4"
        screen.action_previous_child_agent()
        assert _cursor_record(screen).event_id == "event:3"


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


@pytest.mark.asyncio
async def test_live_refresh_preserves_disappearing_filter_options_and_select_values() -> (
    None
):
    module = _filters_module()
    async with _mounted() as (app, pilot, screen):
        bar = screen.query_one(module.TraceFilterBar)
        agent = next(value for value in bar.options.agents if "child-1" in value)
        state = module.TraceFilterState(
            kind="agent_step",
            status="completed",
            agent=agent,
            provider="openai",
        )
        bar.set_state(state)
        await pilot.pause()

        replacement = _record(
            100,
            "system",
            status="new",
            actor_kind="system",
            provider="other",
            start=80.0,
        )
        screen._apply_live_snapshot(
            TrajectorySnapshot((TrajectoryTurn("turn-new", (replacement,)),))
        )
        await pilot.pause()

        assert bar.state == state
        assert state.kind in bar.options.kinds
        assert state.status in bar.options.statuses
        assert state.agent in bar.options.agents
        assert state.provider in bar.options.providers
        for widget_id, expected in (
            ("#trace-filter-kind", state.kind),
            ("#trace-filter-status", state.status),
            ("#trace-filter-agent", state.agent),
            ("#trace-filter-provider", state.provider),
        ):
            assert bar.query_one(widget_id, Select).value == expected
        assert bar.matching_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 18), (100, 30)])
async def test_filter_counts_distinguish_mounted_matches_from_total_matches(
    size: tuple[int, int],
) -> None:
    module = _filters_module()
    async with _mounted(size=size, snapshot=many_records_snapshot(600)) as (
        app,
        pilot,
        screen,
    ):
        bar = screen.query_one(module.TraceFilterBar)
        bar.set_state(module.TraceFilterState(kind="assistant"))
        await pilot.pause()

        assert bar.shown_count == 250
        assert bar.matching_count == 300
        assert bar.total_count == 600
        assert "250 shown" in bar.summary_text
        assert "300 matches" in bar.summary_text
        assert "600 total" in bar.summary_text
        presentation = (
            bar.query_one("#trace-filter-compact", Static)
            if size[0] < 100
            else bar.query_one("#trace-filter-counts", Static)
        )
        assert max(map(len, str(presentation.render()).splitlines())) <= (
            presentation.size.width
        )
        painted = _painted_text(app)
        if size[0] < 100:
            assert "250 shown" in painted
            assert "300 matches" in painted
            assert "600 total" in painted
            assert "1 active" in painted
            assert "g filters" in painted
        else:
            assert "Shown 250" in painted
            assert "Matches 300" in painted
            assert "Total 600" in painted
        assert screen.query_one("#trajectory-table", DataTable).max_scroll_x == 0
