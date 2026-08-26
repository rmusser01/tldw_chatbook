"""UI tests for the Console trajectory screen (task-4 of the trajectory view).

Pilot-driven per the repo's Textual test patterns; snapshots come from the
REAL projection (``derive_trajectory``) fed the same duck-typed row
stand-ins as ``Tests/Chat/test_trajectory_projection.py``, so the screen
is tested against the exact shape Task 3 produces -- never a
fixture-invented one. Keybinding/footer assertions follow the ADR-031
governance suite pattern (``test_schedules_ux_fixes.py``).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Input, Static

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.trajectory import (
    TrajectoryRecord,
    TrajectorySnapshot,
    TrajectoryTurn,
    derive_trajectory,
)
from tldw_chatbook.UI.Screens.trajectory_screen import (
    PAGE_SIZE,
    WORKER_THRESHOLD,
    TrajectoryScreen,
)

# ---------------------------------------------------------------------------
# Duck-typed projection inputs (same mirrors as the projection unit tests)
# ---------------------------------------------------------------------------

LONG_TOOL_RESULT = "R" * 300  # longer than the 120-char preview cap

_T0 = 1_755_165_600.0  # arbitrary unix epoch base


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


@dataclass(frozen=True)
class VariantSetLike:
    turn_id: str
    variants: tuple[str, ...]
    selected_index: int = 0


def base_snapshot():
    """Two turns; turn 1 has tool rows, timing and usage; turn 2 has variants.

    Ledger order (seq): 1 u1, 2 a1, 3 tool_call, 4 tool_result, 5 u2, 6 a2.
    """
    messages = [
        msg("u1", "user", content="hello trajectory world", ts=_T0, parent=None),
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
    ]
    tool_payload = (
        '{"name": "fs_read", "args": {"path": "/tmp/report.txt"}, '
        f'"result": "{LONG_TOOL_RESULT}"}}'
    )
    traj_rows = [
        TrajRow("u1", turn_id="t1", seq=1, event_kind="user", step_started_at=_T0),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=2,
            event_kind="assistant",
            step_started_at=_T0,
            first_token_at=_T0 + 2.0,
            completed_at=_T0 + 5.0,
            model="test-model",
            provider="test-provider",
        ),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=3,
            event_kind="tool_call",
            step_started_at=_T0 + 2.5,
            completed_at=_T0 + 3.0,
            payload_json=tool_payload,
        ),
        TrajRow(
            "a1",
            turn_id="t1",
            seq=4,
            event_kind="tool_result",
            step_started_at=_T0 + 3.0,
            completed_at=_T0 + 3.5,
            payload_json=tool_payload,
        ),
        TrajRow(
            "u2", turn_id="t2", seq=5, event_kind="user", step_started_at=_T0 + 60.0
        ),
        TrajRow(
            "a2",
            turn_id="t2",
            seq=6,
            event_kind="assistant",
            model="test-model",
            provider="test-provider",
        ),
    ]
    usage = {
        "a1": ProviderUsage(
            uncached_input=10,
            cache_read=5,
            cache_write=2,
            output=7,
            provider="test-provider",
            model="test-model",
        ),
    }
    variant_sets = [
        VariantSetLike(
            "t2", ("old zebra draft one", "old zebra draft two"), selected_index=1
        ),
    ]
    return derive_trajectory(messages, usage, traj_rows, variant_sets, [])


def many_records_snapshot(record_count: int):
    """``record_count`` records across user/assistant turns, no sidecar rows."""
    half = record_count // 2
    messages = []
    for i in range(half):
        messages.append(
            msg(f"mu{i}", "user", content=f"user message {i}", ts=_T0 + i * 10.0)
        )
        messages.append(
            msg(
                f"ma{i}",
                "assistant",
                content=f"assistant message {i}",
                ts=_T0 + i * 10.0 + 1.0,
                parent=f"mu{i}",
            )
        )
    return derive_trajectory(messages, {}, [], [], [])


class _Harness(App[None]):
    """Minimal host so the screen can be pushed like the Console would."""

    def compose(self) -> ComposeResult:
        yield Static("base")


@contextlib.asynccontextmanager
async def _mounted(snapshot, **kwargs):
    """Push a TrajectoryScreen on a harness app, keeping the app running.

    Everything (assertions included) must happen INSIDE this context:
    leaving ``run_test()`` tears the app down, so a plain returning helper
    would hand back a dead screen.
    """
    app = _Harness()
    async with app.run_test() as pilot:
        screen = TrajectoryScreen(snapshot, **kwargs)
        await app.push_screen(screen)
        await pilot.pause()
        yield app, pilot, screen


async def _wait_for_rows(pilot, table: DataTable, minimum: int) -> None:
    """Wait for a (worker-rendered) ledger to land, bounded."""
    for _ in range(200):
        if table.row_count >= minimum:
            return
        await pilot.pause(delay=0.02)
    raise AssertionError(f"ledger never reached {minimum} rows (has {table.row_count})")


def _record_key_for_seq(screen: TrajectoryScreen, seq: int) -> str:
    record = next(
        record for turn in screen._turns for record in turn.records if record.seq == seq
    )
    return screen._record_key(record)


def _inspector_content(screen: TrajectoryScreen) -> Static:
    return screen.query_one("#trajectory-inspector-content", Static)


# ---------------------------------------------------------------------------
# Ledger rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mount_renders_one_row_per_record_plus_turn_headers() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        # 6 records + 2 turn-header rows.
        assert table.row_count == 8
        # Stable event identity is the row key; seq stays display-only.
        for seq in range(1, 7):
            assert table.get_row_index(_record_key_for_seq(screen, seq)) is not None
        # Tool rows are present and nested under the assistant step.
        tool_row = table.get_row(_record_key_for_seq(screen, 3))
        assert "Tool call" in str(tool_row[1])


@pytest.mark.asyncio
async def test_title_bar_shows_trace_and_screen_title_without_raw_id() -> None:
    async with _mounted(
        base_snapshot(), screen_title="My Conversation", conversation_id="conv-42"
    ) as (app, pilot, screen):
        title = screen.query_one("#trajectory-title", Static)
        text = str(title.render())
        assert text.startswith("Trace")
        assert "My Conversation" in text
        assert "conv-42" not in text


@pytest.mark.asyncio
async def test_t_toggles_collapse_of_focused_turn() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        assert table.row_count == 8
        # Cursor starts on the first turn header; t collapses it (4 records hidden).
        await pilot.press("t")
        await pilot.pause()
        assert table.row_count == 8 - 4
        # The turn's record rows are gone; the later turn's rows are intact.
        with pytest.raises(Exception):
            table.get_row_index(_record_key_for_seq(screen, 1))
        assert table.get_row_index(_record_key_for_seq(screen, 6)) is not None
        # t again expands it.
        await pilot.press("t")
        await pilot.pause()
        assert table.row_count == 8


@pytest.mark.asyncio
async def test_t_on_record_row_collapses_its_turn() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        row = table.get_row_index(_record_key_for_seq(screen, 6))  # a2, second turn
        table.move_cursor(row=row)
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert table.row_count == 8 - 2  # only turn 2's records hidden
        assert table.get_row_index(_record_key_for_seq(screen, 1)) is not None


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspector_hidden_until_toggled() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        assert inspector.display is False
        await pilot.press("i")
        await pilot.pause()
        assert inspector.display is True
        await pilot.press("i")
        await pilot.pause()
        assert inspector.display is False


@pytest.mark.asyncio
async def test_enter_shows_inspector_with_usage_timing_model() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        row = table.get_row_index(_record_key_for_seq(screen, 2))
        table.move_cursor(row=row)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        assert inspector.display is True
        text = str(_inspector_content(screen).render())
        # Usage breakdown: uncached input / cache read / cache write / output.
        assert "uncached input 10" in text
        assert "cache read 5" in text
        assert "cache write 2" in text
        assert "output 7" in text
        # Model + provider.
        assert "test-model" in text
        assert "test-provider" in text
        # Timing: start -> first token -> completed, with elapsed between facts.
        assert "first token" in text
        assert "2.00s" in text  # start -> first token
        assert "elapsed" in text
        assert "5.00s" in text  # start -> completed
        # The inspected record is identified by its ledger position.
        assert "#2" in text


@pytest.mark.asyncio
async def test_inspector_shows_full_tool_payload() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        row = table.get_row_index(_record_key_for_seq(screen, 3))
        table.move_cursor(row=row)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        text = str(_inspector_content(screen).render())
        assert inspector.display is True
        assert "fs_read" in text
        assert "/tmp/report.txt" in text
        # FULL untruncated tool output, not the 120-char preview.
        assert LONG_TOOL_RESULT in text


@pytest.mark.asyncio
async def test_inspector_timing_blank_when_null() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        row = table.get_row_index(_record_key_for_seq(screen, 1))
        table.move_cursor(row=row)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        inspector = screen.query_one("#trajectory-inspector", VerticalScroll)
        text = str(_inspector_content(screen).render())
        assert inspector.display is True
        # Blanks for the missing timing facts; no fabricated durations.
        assert "—" in text
        assert "elapsed" not in text


@pytest.mark.asyncio
async def test_inspector_lists_turn_level_superseded_variants() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        row = table.get_row_index(_record_key_for_seq(screen, 6))
        table.move_cursor(row=row)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        text = str(_inspector_content(screen).render())
        # Variant contents attach at TURN level; the label must say so.
        assert "superseded variants (turn-level)" in text
        assert "old zebra draft one" in text


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_focuses_search_and_filters_rows() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        await pilot.press("/")
        await pilot.pause()
        search = screen.query_one("#trajectory-search", Input)
        assert search.has_focus
        search.value = "zebras"
        await pilot.pause()
        # Only turn 2 survives: header + its 2 matching records.
        assert table.row_count == 3
        assert table.get_row_index(_record_key_for_seq(screen, 5)) is not None
        assert table.get_row_index(_record_key_for_seq(screen, 6)) is not None
        with pytest.raises(Exception):
            table.get_row_index(_record_key_for_seq(screen, 1))
        # Turn header row survives only because a child matched (it leads
        # the filtered ledger).
        assert table.get_row_index("turn:t2") == 0


@pytest.mark.asyncio
async def test_search_cleared_restores_all_rows() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        search = screen.query_one("#trajectory-search", Input)
        search.value = "zebras"
        await pilot.pause()
        assert table.row_count == 3
        search.value = ""
        await pilot.pause()
        assert table.row_count == 8


@pytest.mark.asyncio
async def test_search_matches_tool_args_and_result_payload() -> None:
    """Tool records are searchable by their payload, not just the preview.

    The preview caps at 120 chars, but tool results are a primary reason to
    search a trajectory -- the query must reach the full args/result.
    """
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        search = screen.query_one("#trajectory-search", Input)
        # Args text (beyond the preview's "fs_read -> RRR..." content).
        search.value = "report.txt"
        await pilot.pause()
        assert table.get_row_index(_record_key_for_seq(screen, 3)) is not None
        assert table.get_row_index(_record_key_for_seq(screen, 4)) is not None
        # Result text far past the 120-char preview cap.
        search.value = "R" * 200
        await pilot.pause()
        assert table.get_row_index(_record_key_for_seq(screen, 3)) is not None


@pytest.mark.asyncio
async def test_escape_from_search_refocuses_table_not_dismiss() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        await pilot.press("/")
        await pilot.pause()
        search = screen.query_one("#trajectory-search", Input)
        assert search.has_focus
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen  # still mounted
        assert screen.query_one("#trajectory-table", DataTable).has_focus


@pytest.mark.asyncio
async def test_escape_dismisses_screen_from_table() -> None:
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        assert screen.query_one("#trajectory-table", DataTable).has_focus
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not screen


# ---------------------------------------------------------------------------
# Pagination + worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_large_ledger_mounts_newest_page_with_load_earlier_row() -> None:
    snapshot = many_records_snapshot(record_count=600)
    async with _mounted(snapshot) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        # Newest 500 records + their 250 turn headers + the load-earlier row.
        assert table.row_count == 1 + 250 + 500
        assert table.get_row_index("__load_earlier__") == 0
        # e loads one more page: all 600 records + 300 headers, no control row.
        await pilot.press("e")
        await pilot.pause()
        assert table.row_count == 300 + 600
        with pytest.raises(Exception):
            table.get_row_index("__load_earlier__")


@pytest.mark.asyncio
async def test_earlier_hint_advertised_only_while_records_remain() -> None:
    snapshot = many_records_snapshot(record_count=PAGE_SIZE + 10)
    async with _mounted(snapshot) as (app, pilot, screen):
        hints = screen.query_one("#trajectory-hints", Static)
        assert "earlier" in str(hints.render())
        await pilot.press("e")
        await pilot.pause()
        assert "earlier" not in str(hints.render())


@pytest.mark.asyncio
async def test_oversize_ledger_renders_via_worker() -> None:
    snapshot = many_records_snapshot(record_count=WORKER_THRESHOLD + 2)
    async with _mounted(snapshot) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        await _wait_for_rows(pilot, table, 1 + 250 + 500)
        assert table.get_row_index("__load_earlier__") == 0


@pytest.mark.asyncio
async def test_stale_worker_render_is_dropped() -> None:
    """A worker-built ledger arriving after a newer render must not land.

    The race: the >WORKER_THRESHOLD worker snapshots the render generation,
    builds its specs off-thread, and dispatches; if the user typed a search
    (or pressed e/t) in between, the sync render already superseded those
    specs -- applying them would silently un-filter the ledger. The guard
    is exercised at the seam because the pilot cannot interleave keystrokes
    inside a worker dispatch deterministically.
    """
    async with _mounted(base_snapshot()) as (app, pilot, screen):
        table = screen.query_one("#trajectory-table", DataTable)
        search = screen.query_one("#trajectory-search", Input)
        search.value = "zebras"
        await pilot.pause()
        assert table.row_count == 3  # filtered
        # Specs "built at generation 0" (before the search) arriving late.
        search.value = ""
        await pilot.pause()  # generation moves on again; unfiltered is current
        assert table.row_count == 8
        stale = [
            (key, cells)
            for key, cells in screen._build_row_specs()
            if key.startswith("turn:t1") or key in {"1", "2", "3", "4"}
        ]
        screen._apply_row_specs(stale, generation=0)
        assert table.row_count == 8  # stale render dropped, ledger unchanged


# ---------------------------------------------------------------------------
# ADR-031 governance (pattern: Tests/UI/test_schedules_ux_fixes.py)
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYS = {
    "ctrl+c",
    "ctrl+v",
    "ctrl+x",
    "ctrl+s",
    "ctrl+d",
    "ctrl+z",
    "ctrl+a",
    "ctrl+r",
    "ctrl+w",
    "ctrl+p",
    "ctrl+q",
}


def test_bindings_avoid_terminal_conventions() -> None:
    bound = {binding.key for binding in TrajectoryScreen.BINDINGS}
    assert bound.isdisjoint(_FORBIDDEN_KEYS), (
        f"TrajectoryScreen binds terminal-convention keys: {bound & _FORBIDDEN_KEYS}"
    )


def test_bindings_use_single_letter_htop_style() -> None:
    bound = {binding.key for binding in TrajectoryScreen.BINDINGS}
    assert {"t", "i", "e", "/", "x"} <= bound


def test_every_binding_has_an_implemented_action() -> None:
    for binding in TrajectoryScreen.BINDINGS:
        action = binding.action
        assert hasattr(TrajectoryScreen, f"action_{action}"), (
            f"Binding '{binding.key}' advertises unimplemented action '{action}'"
        )


def test_footer_hints_match_bindings_exactly() -> None:
    binding_keys = {
        binding.key for binding in TrajectoryScreen.BINDINGS if binding.key != "escape"
    }
    hint_keys = {key for key, _label in TrajectoryScreen.TRAJECTORY_SHORTCUTS}
    assert hint_keys == binding_keys, (
        f"Footer hints {hint_keys} are not 1:1 with BINDINGS {binding_keys}"
    )


def test_inspector_renders_feedback_payload_not_a_phantom_tool() -> None:
    """task-17169: the payload branch is tool-shaped (`tool {name}` / args /
    result). A user_feedback record has none of those keys, so it would
    render a bogus `tool —` line and hide the action, quote and comment that
    are the entire content of the record."""
    record = TrajectoryRecord(
        seq=3,
        kind="user_feedback",
        turn_id="t1",
        message_id="a1",
        content_preview="Request changes: tighten error paths",
        usage=None,
        step_started_at=None,
        first_token_at=None,
        completed_at=None,
        model=None,
        provider=None,
        payload={
            "action": "request-changes",
            "quote": "the retry loop",
            "comment": "tighten error paths",
        },
        variants=(),
        depth=1,
    )

    screen = TrajectoryScreen(TrajectorySnapshot((TrajectoryTurn("t1", (record,)),)))
    text = screen._inspector_text_for_record(record)

    assert "tool" not in text
    assert "feedback request-changes" in text
    assert "quote the retry loop" in text
    assert "comment tighten error paths" in text


def test_inspector_exposes_causal_privacy_and_source_metadata() -> None:
    record = TrajectoryRecord(
        seq=9,
        kind="subagent_steer",
        turn_id="turn-9",
        message_id="message-9",
        content_preview="Steer the reviewer",
        usage=None,
        step_started_at=None,
        first_token_at=None,
        completed_at=None,
        model=None,
        provider=None,
        payload=None,
        variants=(),
        depth=1,
        event_id="agent-step:run-9:2",
        conversation_id="conversation-9",
        source_seq=2,
        label="Agent steered",
        status="accepted",
        actor_kind="subagent",
        actor_id="agent-9",
        run_id="run-9",
        parent_event_id="spawn-9",
        source_event_id="source-9",
        replacement_event_id="replacement-9",
        observed_at=1_755_165_650.0,
        field_states={"payload": "redacted"},
        sensitivity="restricted",
    )

    screen = TrajectoryScreen(
        TrajectorySnapshot((TrajectoryTurn("turn-9", (record,)),))
    )
    text = screen._inspector_text_for_record(record)

    for expected in (
        "source sequence 2",
        "status accepted",
        "actor subagent agent-9",
        "run run-9",
        "parent event spawn-9",
        "source event source-9",
        "replacement event replacement-9",
        "observed",
        'field states {"payload": "redacted"}',
        "sensitivity restricted",
    ):
        assert expected in text
