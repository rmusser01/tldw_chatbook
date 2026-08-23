"""Responsive, compositor, and explicit-state contracts for Console Trace."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from html import unescape
from pathlib import Path
import re
import threading

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Static

from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
)
from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen
import tldw_chatbook.UI.Screens.trajectory_screen as trajectory_screen_module

from .test_trajectory_screen import base_snapshot


VIEWPORTS = ((60, 18), (80, 24), (100, 30), (120, 35))


class _TraceHost(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("Console")


@contextlib.asynccontextmanager
async def _mounted(
    snapshot: TrajectorySnapshot,
    *,
    size: tuple[int, int] = (80, 24),
    **screen_kwargs,
):
    app = _TraceHost()
    async with app.run_test(size=size) as pilot:
        screen = TrajectoryScreen(snapshot, **screen_kwargs)
        await app.push_screen(screen)
        await pilot.pause()
        yield app, pilot, screen


def _painted_text(app: App) -> str:
    svg = app.export_screenshot(simplify=True)
    return unescape(re.sub(r"<[^>]+>", " ", svg))


def _column_labels(table: DataTable) -> tuple[str, ...]:
    return tuple(column.label.plain for column in table.columns.values())


def _flat(snapshot: TrajectorySnapshot) -> tuple[TrajectoryRecord, ...]:
    return tuple(record for turn in snapshot.turns for record in turn.records)


def _long_detail_snapshot() -> TrajectorySnapshot:
    snapshot = base_snapshot()
    turns: list[TrajectoryTurn] = []
    for turn in snapshot.turns:
        records = []
        for record in turn.records:
            if record.kind == "tool_call":
                payload = dict(record.payload or {})
                payload["result"] = "\n".join(
                    [f"detail line {index:02d}" for index in range(80)]
                    + ["TRACE_BOTTOM_SENTINEL"]
                )
                record = replace(record, payload=payload)
            records.append(record)
        turns.append(replace(turn, records=tuple(records)))
    return replace(snapshot, turns=tuple(turns))


def _untimed_snapshot() -> TrajectorySnapshot:
    record = TrajectoryRecord(
        seq=1,
        kind="assistant",
        turn_id="turn-1",
        message_id="message-1",
        content_preview="Finished without captured clocks",
        usage=None,
        step_started_at=None,
        first_token_at=None,
        completed_at=None,
        model=None,
        provider=None,
        payload=None,
        variants=(),
        depth=0,
        event_id="message:message-1",
    )
    return TrajectorySnapshot((TrajectoryTurn("turn-1", (record,)),))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", VIEWPORTS)
async def test_trace_ledger_uses_in_bounds_responsive_columns(size) -> None:
    async with _mounted(base_snapshot(), size=size) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        labels = _column_labels(table)

        assert labels[:4] == ("#", "Event", "Summary", "State")
        if size[0] < 100:
            assert labels == ("#", "Event", "Summary", "State")
        elif size[0] < 120:
            assert labels == ("#", "Event", "Summary", "State", "Tokens", "Duration")
        else:
            assert labels == (
                "#",
                "Event",
                "Summary",
                "State",
                "In",
                "Cache",
                "Out",
                "Duration",
                "Start",
                "Done",
            )
        assert table.max_scroll_x == 0
        assert table.virtual_size.width <= table.content_region.width

        painted = _painted_text(app)
        assert "Trace" in painted
        assert "Event" in painted
        assert "Summary" in painted
        assert "State" in painted


@pytest.mark.asyncio
async def test_responsive_rebuild_preserves_stable_event_selection_and_filters() -> (
    None
):
    snapshot = base_snapshot()
    assistant = next(record for record in _flat(snapshot) if record.kind == "assistant")
    async with _mounted(snapshot, size=(80, 24)) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        selected_key = assistant.event_id
        table.move_cursor(row=table.get_row_index(selected_key), animate=False)
        screen._collapsed.add("t2")
        screen._query = "checking"
        screen._render_ledger()
        await pilot.resize_terminal(120, 35)
        await pilot.pause()

        assert screen._cursor_key() == selected_key
        assert screen._collapsed == {"t2"}
        assert screen._query == "checking"
        assert _column_labels(table)[-3:] == ("Duration", "Start", "Done")
        assert table.max_scroll_x == 0


@pytest.mark.asyncio
async def test_long_inspector_is_focusable_scrolls_to_painted_bottom_and_full_pane() -> (
    None
):
    snapshot = _long_detail_snapshot()
    async with _mounted(
        snapshot,
        size=(60, 18),
        screen_title="Long tool run",
        conversation_id="conv-private",
    ) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        tool_key = next(
            key
            for key, record in screen._row_records.items()
            if record is not None and record.kind == "tool_call"
        )
        table.move_cursor(row=table.get_row_index(tool_key), animate=False)
        await pilot.press("enter")
        await pilot.pause()

        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        content = screen.query_one("#trajectory-inspector-content", Static)
        cue = screen.query_one("#trajectory-inspector-overflow", Static)
        assert inspector.display
        assert inspector.has_focus
        assert content._render_markup is False
        assert inspector.max_scroll_y > 0
        assert "▼ more — scroll" in str(cue.render())
        assert "TRACE_BOTTOM_SENTINEL" not in _painted_text(app)

        await pilot.press("end")
        await pilot.pause()
        assert inspector.scroll_y == inspector.max_scroll_y
        assert "TRACE_BOTTOM_SENTINEL" in _painted_text(app)
        assert str(cue.render()) == ""

        normal_height = inspector.region.height
        await pilot.press("d")
        await pilot.pause()
        assert screen.has_class("trace-detail-full")
        assert inspector.region.height > normal_height
        assert table.display is False
        await pilot.press("d")
        await pilot.pause()
        assert not screen.has_class("trace-detail-full")
        assert table.display is True


@pytest.mark.asyncio
async def test_overflow_cue_reconciles_into_the_100x30_compositor_frame() -> None:
    """The tier/layout swap must not leave the cue one refresh behind."""

    async with _mounted(_long_detail_snapshot(), size=(100, 30)) as (
        app,
        pilot,
        screen,
    ):
        table = screen.query_one("#trajectory-table", DataTable)
        tool_key = next(
            key
            for key, record in screen._row_records.items()
            if record is not None and record.kind == "tool_call"
        )
        table.move_cursor(row=table.get_row_index(tool_key), animate=False)
        await pilot.press("enter")
        await pilot.pause()

        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        cue = screen.query_one("#trajectory-inspector-overflow", Static)
        assert inspector.max_scroll_y > 0
        assert "▼ more — scroll" in str(cue.render())
        assert "more" in _painted_text(app)


@pytest.mark.asyncio
async def test_title_uses_trace_and_moves_conversation_id_into_inspector_metadata() -> (
    None
):
    async with _mounted(
        base_snapshot(), screen_title="My Conversation", conversation_id="conv-42"
    ) as (app, pilot, screen):
        title = str(screen.query_one("#trajectory-title", Static).render())
        assert title.startswith("Trace · My Conversation")
        assert "conv-42" not in title

        table = screen.query_one("#trajectory-table", DataTable)
        record_key = next(key for key, record in screen._row_records.items() if record)
        table.move_cursor(row=table.get_row_index(record_key), animate=False)
        await pilot.press("enter")
        await pilot.pause()
        detail = str(screen.query_one("#trajectory-inspector-content", Static).render())
        assert "conversation conv-42" in detail


@pytest.mark.asyncio
async def test_record_kinds_are_humanized_in_ledger_but_raw_kind_remains_in_detail() -> (
    None
):
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        tool_key = next(
            key
            for key, record in screen._row_records.items()
            if record is not None and record.kind == "tool_call"
        )
        row = table.get_row(tool_key)
        assert str(row[1]) == "Tool call"
        assert "tool_call" not in str(row[1])

        table.move_cursor(row=table.get_row_index(tool_key), animate=False)
        await pilot.press("enter")
        detail = str(screen.query_one("#trajectory-inspector-content", Static).render())
        assert "raw kind tool_call" in detail


@pytest.mark.asyncio
async def test_explicit_empty_filtered_imported_incomplete_and_no_timing_states() -> (
    None
):
    async with _mounted(TrajectorySnapshot(())) as (app, pilot, screen):
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert "EMPTY" in state
        assert "No trace events yet" in state

    async with _mounted(base_snapshot()) as (app, pilot, screen):
        search = screen.query_one("#trajectory-search")
        search.value = "zebras"
        await pilot.pause()
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert "FILTERED" in state
        assert "2/6 events" in state

        search.value = "no-event-can-match-this"
        await pilot.pause()
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert "NO MATCHES" in state
        assert "0/6 events" in state
        assert "clear search" in state.lower()

    async with _mounted(base_snapshot(), shared_trace=True) as (app, pilot, screen):
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert "READ-ONLY SHARED TRACE" in state

    incomplete = replace(
        base_snapshot(),
        turns=(
            replace(
                base_snapshot().turns[0],
                records=(
                    replace(
                        base_snapshot().turns[0].records[0],
                        field_states={"payload": "capture_failed"},
                    ),
                ),
            ),
        ),
    )
    async with _mounted(incomplete) as (app, pilot, screen):
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert "INCOMPLETE" in state

    async with _mounted(_untimed_snapshot()) as (app, pilot, screen):
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert "NO TIMING" in state
        assert "Duration unavailable" in state


@pytest.mark.asyncio
async def test_loading_state_is_visible_until_real_render_worker_lands(
    monkeypatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class _BlockingRenderScreen(TrajectoryScreen):
        def _build_row_specs(self):
            started.set()
            release.wait(timeout=5)
            return super()._build_row_specs()

    monkeypatch.setattr(trajectory_screen_module, "WORKER_THRESHOLD", 0)
    app = _TraceHost()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = _BlockingRenderScreen(base_snapshot())
        await app.push_screen(screen)
        for _ in range(100):
            if started.is_set():
                break
            await pilot.pause(0.01)
        assert started.is_set()
        assert "LOADING" in str(screen.query_one("#trajectory-state", Static).render())
        release.set()
        table = screen.query_one("#trajectory-table", DataTable)
        for _ in range(100):
            if table.row_count:
                break
            await pilot.pause(0.01)
        assert table.row_count
        assert "LOADING" not in str(
            screen.query_one("#trajectory-state", Static).render()
        )


@pytest.mark.asyncio
async def test_render_failure_is_payload_safe_and_r_retries_real_worker(
    monkeypatch,
) -> None:
    class _FailOnceScreen(TrajectoryScreen):
        fail_once = True

        def _build_row_specs(self):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("SECRET_PAYLOAD_MUST_NOT_RENDER")
            return super()._build_row_specs()

    monkeypatch.setattr(trajectory_screen_module, "WORKER_THRESHOLD", 0)
    app = _TraceHost()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = _FailOnceScreen(base_snapshot())
        await app.push_screen(screen)
        state_widget = screen.query_one("#trajectory-state", Static)
        for _ in range(100):
            if "FAILED" in str(state_widget.render()):
                break
            await pilot.pause(0.01)
        failed = str(state_widget.render())
        assert "FAILED" in failed
        assert "r retry" in failed
        assert "SECRET_PAYLOAD_MUST_NOT_RENDER" not in failed

        await pilot.press("r")
        table = screen.query_one("#trajectory-table", DataTable)
        for _ in range(100):
            if table.row_count:
                break
            await pilot.pause(0.01)
        assert table.row_count
        assert "FAILED" not in str(state_widget.render())


@pytest.mark.asyncio
async def test_live_failure_is_visible_and_retry_uses_snapshot_builder() -> None:
    state = {"revision": 1, "fail": True, "calls": 0}
    snapshot = base_snapshot()

    def build():
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("SECRET_LIVE_PAYLOAD")
        return snapshot

    async with _mounted(
        snapshot,
        revision_provider=lambda: state["revision"],
        snapshot_builder=build,
    ) as (app, pilot, screen):
        state["revision"] = 2
        screen._poll_revision()
        state_widget = screen.query_one("#trajectory-state", Static)
        for _ in range(100):
            if "FAILED" in str(state_widget.render()):
                break
            await pilot.pause(0.01)
        failed = str(state_widget.render())
        assert "FAILED" in failed
        assert "SECRET_LIVE_PAYLOAD" not in failed

        state["fail"] = False
        await pilot.press("r")
        for _ in range(100):
            if "FAILED" not in str(state_widget.render()):
                break
            await pilot.pause(0.01)
        assert state["calls"] >= 2
        assert "LIVE · FOLLOWING" in str(state_widget.render())


@pytest.mark.asyncio
async def test_open_import_marks_the_pushed_production_screen_read_only_shared(
    monkeypatch,
) -> None:
    class _PickerScreen(TrajectoryScreen):
        async def _pick_trace_file(self):
            return Path("shared-trace.json")

    monkeypatch.setattr(
        trajectory_screen_module,
        "load_trajectory_snapshot",
        lambda _path: base_snapshot(),
    )
    app = _TraceHost()
    async with app.run_test(size=(80, 24)) as pilot:
        source = _PickerScreen(base_snapshot())
        await app.push_screen(source)
        await pilot.press("o")
        for _ in range(100):
            if app.screen is not source:
                break
            await pilot.pause(0.01)
        imported = app.screen
        assert isinstance(imported, TrajectoryScreen)
        assert imported._shared_trace is True
        assert "READ-ONLY SHARED TRACE" in str(
            imported.query_one("#trajectory-state", Static).render()
        )


@pytest.mark.asyncio
async def test_live_following_and_paused_states_are_visible_and_actionable() -> None:
    snapshot = base_snapshot()
    async with _mounted(
        snapshot,
        revision_provider=lambda: 1,
        snapshot_builder=lambda: snapshot,
    ) as (app, pilot, screen):
        assert "LIVE · FOLLOWING" in str(
            screen.query_one("#trajectory-state", Static).render()
        )
        screen._follow = False
        screen._refresh_state()
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert "LIVE · PAUSED" in state
        assert "f resume" in state
        await pilot.press("f")
        assert "LIVE · FOLLOWING" in str(
            screen.query_one("#trajectory-state", Static).render()
        )


@pytest.mark.asyncio
async def test_empty_ledgers_do_not_advertise_row_only_actions() -> None:
    async with _mounted(TrajectorySnapshot(())) as (app, pilot, screen):
        hints = str(screen.query_one("#trajectory-hints", Static).render())
        assert "inspect" not in hints
        assert "collapse" not in hints
        assert "inspector" not in hints
        assert "open" in hints
