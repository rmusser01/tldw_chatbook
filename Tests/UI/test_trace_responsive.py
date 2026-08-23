"""Responsive, compositor, and explicit-state contracts for Console Trace."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from html import unescape
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Input, Static

from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
    derive_trajectory,
)
from tldw_chatbook.UI.Screens.trajectory_screen import PAGE_SIZE, TrajectoryScreen
import tldw_chatbook.UI.Screens.trajectory_screen as trajectory_screen_module

from .consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from .test_trajectory_screen import base_snapshot


VIEWPORTS = ((60, 18), (80, 24), (100, 30), (120, 35))
class _TraceHost(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET

    def compose(self) -> ComposeResult:
        yield Static("Console")


def test_trace_harness_uses_latest_dev_consolidated_production_css() -> None:
    assert Path(_TraceHost.CSS_PATH).name == "tldw_cli_modular.tcss"


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
    return unescape(re.sub(r"<[^>]+>", " ", svg)).replace("\N{NO-BREAK SPACE}", " ")


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


def _snapshot_with_records(records: list[TrajectoryRecord]) -> TrajectorySnapshot:
    return TrajectorySnapshot((TrajectoryTurn("turn-1", tuple(records)),))


def _numbered_records(count: int) -> list[TrajectoryRecord]:
    return [
        TrajectoryRecord(
            seq=index,
            kind="assistant",
            turn_id="turn-1",
            message_id=f"message-{index}",
            content_preview=f"event {index}",
            usage=None,
            step_started_at=float(index),
            first_token_at=None,
            completed_at=float(index) + 0.1,
            model="model",
            provider="provider",
            payload=None,
            variants=(),
            depth=0,
            event_id=f"event-{index}",
        )
        for index in range(1, count + 1)
    ]


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
        assert "x clear filters" in state.lower()

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
    async with app.run_test(size=(60, 18)) as pilot:
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
    async with app.run_test(size=(60, 18)) as pilot:
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
        assert "r retry" in _painted_text(app)

        search = screen.query_one("#trajectory-search", Input)
        search.value = "checking"
        await pilot.pause()
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert "FAILED" in str(state_widget.render())
        assert screen._retry_target == "render"

        await pilot.press("r")
        table = screen.query_one("#trajectory-table", DataTable)
        for _ in range(100):
            if table.row_count:
                break
            await pilot.pause(0.01)
        assert table.row_count
        assert "FAILED" not in str(state_widget.render())


@pytest.mark.asyncio
async def test_render_retry_is_visible_single_flight_and_recovers_after_failure(
    monkeypatch,
) -> None:
    retry_started = threading.Event()
    release_retry = threading.Event()
    early_third_started = threading.Event()

    class _BlockedRetryScreen(TrajectoryScreen):
        attempts = 0

        def _build_row_specs(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("initial render failed")
            if self.attempts == 2:
                retry_started.set()
                release_retry.wait(timeout=5)
                raise RuntimeError("retry render failed")
            early_third_started.set()
            return super()._build_row_specs()

    monkeypatch.setattr(trajectory_screen_module, "WORKER_THRESHOLD", 0)
    app = _TraceHost()
    async with app.run_test(size=(60, 18)) as pilot:
        screen = _BlockedRetryScreen(base_snapshot())
        await app.push_screen(screen)
        state_widget = screen.query_one("#trajectory-state", Static)
        for _ in range(100):
            if "FAILED" in str(state_widget.render()):
                break
            await pilot.pause(0.01)

        await pilot.press("r")
        for _ in range(100):
            if retry_started.is_set():
                break
            await pilot.pause(0.01)
        assert retry_started.is_set()
        try:
            assert "RETRYING" in str(state_widget.render())
            assert "RETRYING" in _painted_text(app)
            hints = str(screen.query_one("#trajectory-hints", Static).render())
            assert "r retry" not in hints

            await pilot.press("r")
            for _ in range(50):
                if early_third_started.is_set():
                    break
                await pilot.pause(0.01)
            assert not early_third_started.is_set()
            assert screen.attempts == 2
        finally:
            release_retry.set()

        for _ in range(100):
            if "FAILED" in str(state_widget.render()) and not screen._loading:
                break
            await pilot.pause(0.01)
        assert "FAILED" in str(state_widget.render())
        assert "r retry" in str(screen.query_one("#trajectory-hints", Static).render())

        await pilot.press("r")
        table = screen.query_one("#trajectory-table", DataTable)
        for _ in range(100):
            if table.row_count:
                break
            await pilot.pause(0.01)
        assert table.row_count
        assert screen.attempts == 3
        assert screen._failure is None
        assert screen._loading is False
        assert screen._retry_target is None


@pytest.mark.asyncio
async def test_successful_render_retry_clears_failure_when_filter_makes_specs_stale(
    monkeypatch,
) -> None:
    retry_started = threading.Event()
    release_retry = threading.Event()

    class _StaleRetryScreen(TrajectoryScreen):
        attempts = 0

        def _build_row_specs(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("initial render failed")
            if self.attempts == 2:
                retry_started.set()
                release_retry.wait(timeout=5)
            return super()._build_row_specs()

    monkeypatch.setattr(trajectory_screen_module, "WORKER_THRESHOLD", 0)
    app = _TraceHost()
    async with app.run_test(size=(60, 18)) as pilot:
        screen = _StaleRetryScreen(base_snapshot())
        await app.push_screen(screen)
        state_widget = screen.query_one("#trajectory-state", Static)
        for _ in range(100):
            if "FAILED" in str(state_widget.render()):
                break
            await pilot.pause(0.01)

        await pilot.press("r")
        for _ in range(100):
            if retry_started.is_set():
                break
            await pilot.pause(0.01)
        assert retry_started.is_set()
        try:
            screen.query_one("#trajectory-search", Input).value = "checking"
            await pilot.pause()
        finally:
            release_retry.set()

        for _ in range(100):
            if not screen._retry_in_flight:
                break
            await pilot.pause(0.01)
        assert screen._retry_in_flight is False
        assert screen._loading is False
        assert screen._failure is None
        assert screen._retry_target is None


@pytest.mark.asyncio
async def test_live_failure_is_visible_and_retry_uses_snapshot_builder() -> None:
    retry_started = threading.Event()
    release_retry = threading.Event()
    state = {"revision": 1, "fail": True, "block": False, "calls": 0}
    snapshot = base_snapshot()

    def build():
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("SECRET_LIVE_PAYLOAD")
        if state["block"]:
            retry_started.set()
            release_retry.wait(timeout=5)
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

        search = screen.query_one("#trajectory-search", Input)
        search.value = "zebras"
        await pilot.pause()
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert "FAILED" in str(state_widget.render())
        assert screen._retry_target == "live"

        state["fail"] = False
        state["block"] = True
        await pilot.press("r")
        for _ in range(100):
            if retry_started.is_set():
                break
            await pilot.pause(0.01)
        assert retry_started.is_set()
        try:
            assert "RETRYING" in str(state_widget.render())
            assert "r retry" not in str(
                screen.query_one("#trajectory-hints", Static).render()
            )
            await pilot.press("r")
            for _ in range(50):
                if state["calls"] > 2:
                    break
                await pilot.pause(0.01)
            assert state["calls"] == 2
        finally:
            state["block"] = False
            release_retry.set()
        for _ in range(100):
            if "FAILED" not in str(state_widget.render()):
                break
            await pilot.pause(0.01)
        assert state["calls"] >= 2
        assert "LIVE · FOLLOWING" in str(state_widget.render())


@pytest.mark.asyncio
async def test_stale_live_failure_cannot_replace_newer_revision_success() -> None:
    stale_started = threading.Event()
    release_stale = threading.Event()
    stale_finished = threading.Event()
    state = {"revision": 1, "calls": 0}
    initial = _snapshot_with_records(_numbered_records(1))
    latest = _snapshot_with_records(_numbered_records(3))

    def build() -> TrajectorySnapshot:
        state["calls"] += 1
        if state["calls"] == 1:
            stale_started.set()
            release_stale.wait(timeout=5)
            stale_finished.set()
            raise RuntimeError("STALE_REVISION_SECRET")
        return latest

    async with _mounted(
        initial,
        revision_provider=lambda: state["revision"],
        snapshot_builder=build,
    ) as (app, pilot, screen):
        state["revision"] = 2
        screen._poll_revision()
        for _ in range(100):
            if stale_started.is_set():
                break
            await pilot.pause(0.01)
        assert stale_started.is_set()

        state["revision"] = 3
        screen._poll_revision()
        for _ in range(100):
            if screen._total_records == 3:
                break
            await pilot.pause(0.01)
        assert screen._total_records == 3
        assert screen._last_revision == 3

        release_stale.set()
        for _ in range(100):
            if stale_finished.is_set():
                break
            await pilot.pause(0.01)
        assert stale_finished.is_set()
        await pilot.pause()

        assert screen._total_records == 3
        assert screen._failure is None
        assert screen._retry_target is None
        assert "FAILED" not in str(
            screen.query_one("#trajectory-state", Static).render()
        )


@pytest.mark.asyncio
async def test_open_import_marks_the_pushed_production_screen_read_only_shared(
    monkeypatch,
) -> None:
    class _PickerScreen(TrajectoryScreen):
        async def _pick_trace_file(self):
            return Path("shared-trace.json")

    monkeypatch.setattr(
        trajectory_screen_module,
        "load_imported_trace",
        lambda _path: SimpleNamespace(
            snapshot=base_snapshot(),
            operation_event=replace(
                base_snapshot().turns[0].records[-1],
                seq=3,
                kind="trace_import",
                turn_id="trace",
                event_id="trace-import-test",
            ),
            manifest={
                "format_version": 2,
                "profile": "redacted_diagnostic",
            },
            integrity={"verified": True},
            privacy_inventory={"redacted": 1, "omitted": 0, "truncated": 0},
        ),
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
        assert "o import trace" in hints


@pytest.mark.asyncio
async def test_projected_generic_retrieval_payload_is_preserved_as_safe_json() -> None:
    snapshot = derive_trajectory(
        messages=[],
        usage_by_id={},
        traj_rows=[],
        variant_sets=[],
        compaction_records=[],
        retrieval_runs=[
            {
                "run_id": "rag-7",
                "conversation_id": "conv-1",
                "turn_id": "turn-1",
                "run_ordinal": 2,
                "stage": "hybrid_search",
                "status": "complete",
                "started_at": 10.0,
                "ended_at": 12.0,
            }
        ],
    )
    record = _flat(snapshot)[0]
    assert record.kind == "retrieval_run"
    assert record.payload == {"stage": "hybrid_search"}

    async with _mounted(snapshot, size=(60, 18)) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        table.move_cursor(row=table.get_row_index(record.event_id), animate=False)
        await pilot.press("enter")
        await pilot.pause()

        content = screen.query_one("#trajectory-inspector-content", Static)
        detail = str(content.render())
        assert content._render_markup is False
        assert 'payload {"stage": "hybrid_search"}' in detail
        assert "tool —" not in detail


@pytest.mark.asyncio
async def test_full_detail_keeps_visible_inspector_focus_and_compact_hints() -> None:
    async with _mounted(_long_detail_snapshot(), size=(60, 18)) as (
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
        await pilot.press("d")
        await pilot.pause()

        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        search = screen.query_one("#trajectory-search", Input)
        hints = str(screen.query_one("#trajectory-hints", Static).render())
        painted = _painted_text(app)
        assert inspector.has_focus
        assert search.display is False
        assert "d split view" in hints
        assert "d split view" in painted
        assert "search" not in hints
        assert "inspect" not in hints
        assert "collapse" not in hints

        await pilot.press("/")
        await pilot.pause()
        assert inspector.has_focus
        assert not search.has_focus


@pytest.mark.asyncio
async def test_x_clears_search_and_timeline_brush_together() -> None:
    async with _mounted(base_snapshot(), size=(60, 18)) as (app, pilot, screen):
        search = screen.query_one("#trajectory-search", Input)
        domain = screen._timeline.model.domain
        assert domain is not None
        screen._timeline.apply_brush((domain[1] + 100, domain[1] + 200))
        await pilot.pause()

        state = str(screen.query_one("#trajectory-state", Static).render())
        hints = str(screen.query_one("#trajectory-hints", Static).render())
        assert search.value == ""
        assert "NO MATCHES" in state
        assert "x clear filters" in state.lower()
        assert "x clear filters" in hints

        search.value = "no-event-can-match-this"
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        assert search.value == ""
        assert screen._query == ""
        assert screen._filter_bar.state.time_range is None
        assert screen._timeline._brush is None
        assert "x clear filters" not in str(
            screen.query_one("#trajectory-hints", Static).render()
        )


@pytest.mark.asyncio
async def test_search_focus_keeps_x_as_text_until_escape_enables_recovery() -> None:
    async with _mounted(base_snapshot(), size=(60, 18)) as (app, pilot, screen):
        await pilot.press("/")
        search = screen.query_one("#trajectory-search", Input)
        table = screen.query_one("#trajectory-table", DataTable)
        assert search.has_focus
        await pilot.press("n", "o", "p", "e")
        domain = screen._timeline.model.domain
        assert domain is not None
        screen._timeline.apply_brush((domain[1] + 100, domain[1] + 200))
        await pilot.pause()

        state_widget = screen.query_one("#trajectory-state", Static)
        assert "Esc then x clear filters" in str(state_widget.render())
        assert "x clear filters" not in str(
            screen.query_one("#trajectory-hints", Static).render()
        )

        await pilot.press("x")
        await pilot.pause()
        assert search.value == "nopex"
        assert screen._filter_bar.state.time_range is not None

        await pilot.press("escape")
        await pilot.pause()
        assert table.has_focus
        assert "Esc then" not in str(state_widget.render())
        assert "x clear filters" in str(state_widget.render())

        await pilot.press("x")
        await pilot.pause()
        assert search.value == ""
        assert screen._query == ""
        assert screen._filter_bar.state.time_range is None


@pytest.mark.asyncio
async def test_focus_only_transitions_refresh_filter_recovery_truth() -> None:
    async with _mounted(base_snapshot(), size=(60, 18)) as (app, pilot, screen):
        search = screen.query_one("#trajectory-search", Input)
        table = screen.query_one("#trajectory-table", DataTable)
        state = screen.query_one("#trajectory-state", Static)
        hints = screen.query_one("#trajectory-hints", Static)
        domain = screen._timeline.model.domain
        assert domain is not None
        screen._timeline.apply_brush((domain[1] + 100, domain[1] + 200))
        await pilot.pause()

        assert "x clear filters" in str(state.render())
        assert "x clear filters" in str(hints.render())

        await pilot.press("/")
        assert search.has_focus
        assert "Esc then x clear filters" in str(state.render())
        assert "x clear filters" not in str(hints.render())

        await pilot.press("enter")
        assert table.has_focus
        assert "Esc then" not in str(state.render())
        assert "x clear filters" in str(state.render())
        assert "x clear filters" in str(hints.render())


@pytest.mark.asyncio
async def test_inspector_scroll_survives_same_event_refresh_and_resets_on_change() -> (
    None
):
    snapshot = _long_detail_snapshot()
    async with _mounted(snapshot, size=(60, 18)) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        tool_key = next(
            key
            for key, record in screen._row_records.items()
            if record is not None and record.kind == "tool_call"
        )
        table.move_cursor(row=table.get_row_index(tool_key), animate=False)
        await pilot.press("enter")
        await pilot.press("end")
        await pilot.pause()
        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        before = inspector.scroll_y
        assert before > 0

        screen._render_ledger()
        await pilot.pause()
        assert inspector.scroll_y == before

        screen._follow = False
        screen._apply_live_snapshot(snapshot)
        await pilot.pause()
        assert screen._cursor_key() == tool_key
        assert inspector.scroll_y == before

        different_key = next(
            key
            for key, record in screen._row_records.items()
            if record is not None and key != tool_key
        )
        table.move_cursor(row=table.get_row_index(different_key), animate=False)
        await pilot.pause()
        assert screen._cursor_key() == different_key
        assert inspector.scroll_y == 0


@pytest.mark.asyncio
async def test_empty_modes_and_compound_state_truth_are_painted_at_60_columns() -> None:
    empty = TrajectorySnapshot(())
    async with _mounted(empty, size=(60, 18), shared_trace=True) as (
        app,
        pilot,
        screen,
    ):
        painted = _painted_text(app)
        assert "READ-ONLY SHARED TRACE" in painted
        assert "EMPTY" in painted
        assert "o import trace" in painted

    async with _mounted(
        empty,
        size=(60, 18),
        revision_provider=lambda: 1,
        snapshot_builder=lambda: empty,
    ) as (app, pilot, screen):
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert "LIVE" in state
        assert "FOLLOWING" in state
        assert "EMPTY" in state
        assert "Waiting for first event" in state
        assert "o import trace" not in state
        await pilot.press("i", "enter")
        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        assert not inspector.display
        assert not inspector.has_focus

    record = replace(
        _untimed_snapshot().turns[0].records[0],
        field_states={"payload": "capture_failed"},
    )
    async with _mounted(
        _snapshot_with_records([record]), size=(60, 18), shared_trace=True
    ) as (app, pilot, screen):
        screen.query_one("#trajectory-search", Input).value = "no-match"
        await pilot.pause()
        painted = _painted_text(app)
        for expected in (
            "READ-ONLY SHARED TRACE",
            "INCOMPLETE",
            "NO MATCHES",
            "NO TIMING",
            "x clear filters",
        ):
            assert expected in painted

    live_snapshot = _snapshot_with_records([record])
    async with _mounted(
        live_snapshot,
        size=(60, 18),
        revision_provider=lambda: 1,
        snapshot_builder=lambda: live_snapshot,
    ) as (app, pilot, screen):
        screen.query_one("#trajectory-search", Input).value = "no-match"
        await pilot.pause()
        painted = _painted_text(app)
        for expected in (
            "LIVE",
            "FOLLOWING",
            "INCOMPLETE",
            "NO MATCHES",
            "NO TIMING",
            "x clear filters",
        ):
            assert expected in painted


@pytest.mark.asyncio
async def test_compact_state_prioritizes_filter_recovery_before_secondary_facts() -> (
    None
):
    record = replace(
        _untimed_snapshot().turns[0].records[0],
        field_states={"payload": "capture_failed"},
    )
    async with _mounted(
        _snapshot_with_records([record]), size=(60, 18), shared_trace=True
    ) as (app, pilot, screen):
        screen.query_one("#trajectory-search", Input).value = "no-match"
        await pilot.pause()
        state = str(screen.query_one("#trajectory-state", Static).render())
        assert state.index("READ-ONLY SHARED TRACE") < state.index("x clear filters")
        assert state.index("x clear filters") < state.index("INCOMPLETE")
        assert state.index("INCOMPLETE") < state.index("NO TIMING")
        assert "x clear filters" in _painted_text(app)


@pytest.mark.asyncio
async def test_paused_live_append_retains_selected_event_outside_newest_page() -> None:
    records = _numbered_records(PAGE_SIZE + 1)
    snapshot = _snapshot_with_records(records)
    async with _mounted(snapshot, size=(80, 24)) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        table.move_cursor(row=table.get_row_index("event-2"), animate=False)
        screen._follow = False

        appended = _snapshot_with_records(
            records + _numbered_records(PAGE_SIZE + 2)[-1:]
        )
        screen._apply_live_snapshot(appended)
        await pilot.pause()

        assert screen._cursor_key() == "event-2"
        assert "event-2" in screen._visible_keys
        assert screen._visible_count >= PAGE_SIZE + 1


def _legacy_record(seq: int, provider: str) -> TrajectoryRecord:
    return TrajectoryRecord(
        seq=seq,
        kind="assistant",
        turn_id="turn-1",
        message_id="shared-message",
        content_preview="same preview",
        usage=None,
        step_started_at=10.0,
        first_token_at=11.0,
        completed_at=12.0,
        model="same-model",
        provider=provider,
        payload={"attempt": 1},
        variants=(),
        depth=0,
        event_id="",
        status="completed",
    )


@pytest.mark.asyncio
async def test_legacy_identity_survives_insertion_when_provider_distinguishes_rows() -> (
    None
):
    before_records = [_legacy_record(1, "provider-a"), _legacy_record(2, "provider-b")]
    async with _mounted(_snapshot_with_records(before_records)) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        provider_b_key = screen._record_key(before_records[1])
        table.move_cursor(row=table.get_row_index(provider_b_key), animate=False)
        screen._follow = False

        after_records = [
            _legacy_record(1, "provider-new"),
            _legacy_record(2, "provider-a"),
            _legacy_record(3, "provider-b"),
        ]
        screen._apply_live_snapshot(_snapshot_with_records(after_records))
        await pilot.pause()

        assert screen._record_key(after_records[2]) == provider_b_key
        assert screen._cursor_key() == provider_b_key
        selected = screen._row_records[screen._cursor_key()]
        assert selected is not None
        assert selected.provider == "provider-b"


@pytest.mark.asyncio
async def test_legacy_identity_survives_pending_to_completed_mutation() -> None:
    pending = replace(
        _legacy_record(1, "provider-a"),
        status="pending",
        content_preview="working",
        completed_at=None,
        payload={"phase": "pending"},
        field_states={"payload": "observed"},
    )
    async with _mounted(_snapshot_with_records([pending])) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        pending_key = screen._record_key(pending)
        table.move_cursor(row=table.get_row_index(pending_key), animate=False)
        screen._follow = False

        completed = replace(
            pending,
            status="completed",
            content_preview="finished",
            completed_at=15.0,
            payload={"phase": "completed", "result": "done"},
            field_states={"payload": "redacted"},
        )
        screen._apply_live_snapshot(_snapshot_with_records([completed]))
        await pilot.pause()

        assert screen._record_key(completed) == pending_key
        assert screen._cursor_key() == pending_key
        selected = screen._row_records[pending_key]
        assert selected is not None
        assert selected.status == "completed"


@pytest.mark.asyncio
async def test_legacy_collision_keys_survive_distinct_collider_inserted_ahead() -> None:
    collider_a = replace(
        _legacy_record(1, "provider-a"),
        content_preview="collider A",
        payload={"collider": "A"},
    )
    collider_b = replace(
        _legacy_record(2, "provider-a"),
        content_preview="collider B",
        payload={"collider": "B"},
    )
    async with _mounted(_snapshot_with_records([collider_a, collider_b])) as (
        app,
        pilot,
        screen,
    ):
        table = screen.query_one("#trajectory-table", DataTable)
        collider_b_key = screen._record_key(collider_b)
        table.move_cursor(row=table.get_row_index(collider_b_key), animate=False)
        screen._follow = False

        collider_c = replace(
            _legacy_record(1, "provider-a"),
            content_preview="collider C",
            payload={"collider": "C"},
        )
        moved_a = replace(collider_a, seq=2)
        moved_b = replace(collider_b, seq=3)
        screen._apply_live_snapshot(
            _snapshot_with_records([collider_c, moved_a, moved_b])
        )
        await pilot.pause()

        assert screen._record_key(moved_b) == collider_b_key
        assert screen._cursor_key() == collider_b_key
        selected = screen._row_records[collider_b_key]
        assert selected is not None
        assert selected.content_preview == "collider B"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", VIEWPORTS)
async def test_six_digit_record_identity_is_fully_painted_without_horizontal_scroll(
    size,
) -> None:
    record = replace(_numbered_records(1)[0], seq=123456)
    async with _mounted(_snapshot_with_records([record]), size=size) as (
        app,
        pilot,
        screen,
    ):
        table = screen.query_one("#trajectory-table", DataTable)
        assert "123456" in _painted_text(app)
        assert table.max_scroll_x == 0


@pytest.mark.asyncio
async def test_record_label_prefers_projection_label_then_generic_sentence_case() -> (
    None
):
    records = [
        replace(
            _numbered_records(1)[0],
            kind="future_provider_step",
            label="Provider-specific step",
        ),
        replace(
            _numbered_records(2)[1],
            kind="future_custom_kind",
            label="",
        ),
    ]
    async with _mounted(_snapshot_with_records(records)) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        assert str(table.get_row("event-1")[1]) == "Provider-specific step"
        assert str(table.get_row("event-2")[1]) == "Future custom kind"
