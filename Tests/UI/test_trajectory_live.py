"""Task 5: Console trajectory launch binding + live tail-follow.

Three seams, per the task brief:

1. **Store revision bus** (unit, no app): the Task 2 trajectory write path
   bumps the payload-revision counter so a polling screen sees changes, and
   the new public getter reads it by conversation id.
2. **Live tail-follow** (pilot-driven, seam-level): a minimal app hosts
   TrajectoryScreen with fake ``revision_provider``/``snapshot_builder``
   callables -- the exact callables the Console launcher passes -- and the
   poll/worker/follow state machine is driven deterministically.
3. **Launch binding registration** (unit, non-async): ChatScreen binds a
   single-letter ADR-299-legal key that pushes the trajectory screen.

Full-app construction is deliberately avoided (the Console workbench pilot
is far heavier than this feature needs); the seam callables are the real
integration surface.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Static

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.citation_trace_repository import ActiveCitationTraceState
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_repository import (
    AuxiliaryAttemptStart,
    ConsoleContextRepository,
)
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, TrajectoryRowWrite
from tldw_chatbook.UI.Console_Modules.review_selection import (
    ConsoleReviewSelectionController,
    _build_trajectory_snapshot,
)
from tldw_chatbook.UI.Console_Modules.wiring import _present_console_trajectory
from tldw_chatbook.UI.Console_Modules.composer_run_controls import TRANSCRIPT_KEYS_FOOTER
from tldw_chatbook.UI.Screens.chat_screen import CONSOLE_WORKBENCH_SHORTCUTS, ChatScreen
from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen

# ---------------------------------------------------------------------------
# Duck-typed projection inputs (same mirrors as the trajectory screen suite)
# ---------------------------------------------------------------------------

_T0 = 1_755_165_600.0


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


def snapshot_with_turns(turn_count: int):
    """A linear conversation: ``turn_count`` user+assistant record pairs."""
    messages = []
    rows = []
    seq = 0
    parent = None
    for index in range(turn_count):
        user_id = f"u{index}"
        assistant_id = f"a{index}"
        turn_id = f"t{index}"
        messages.append(
            msg(
                user_id,
                "user",
                content=f"question {index}",
                ts=_T0 + index * 10.0,
                parent=parent,
            )
        )
        seq += 1
        rows.append(
            TrajRow(
                user_id,
                turn_id=turn_id,
                seq=seq,
                event_kind="user",
                step_started_at=_T0 + index * 10.0,
            )
        )
        messages.append(
            msg(
                assistant_id,
                "assistant",
                content=f"answer {index}",
                ts=_T0 + index * 10.0 + 1.0,
                parent=user_id,
            )
        )
        seq += 1
        rows.append(
            TrajRow(
                assistant_id,
                turn_id=turn_id,
                seq=seq,
                event_kind="assistant",
                model="test-model",
                provider="test-provider",
            )
        )
        parent = assistant_id
    return derive_trajectory(messages, {}, rows, [], [])


# ---------------------------------------------------------------------------
# 1. Store revision bus
# ---------------------------------------------------------------------------


class _FakePersistence:
    """Duck-typed persistence adapter: only the trajectory write method."""

    def __init__(self) -> None:
        self.written: list[TrajectoryRowWrite] = []

    def write_trajectory_rows(self, rows):
        self.written.extend(rows)
        return True


def test_trajectory_write_bumps_payload_revision_for_conversation():
    store = ConsoleChatStore(persistence=_FakePersistence())  # type: ignore[arg-type]
    session = store.create_session(title="s")
    session.persisted_conversation_id = "conv-1"

    before = store.get_payload_revision("conv-1")
    row = TrajectoryRowWrite(
        message_id="m1",
        conversation_id="conv-1",
        turn_id="t1",
        seq=None,
        event_kind="user",
    )
    assert store.write_trajectory_rows([row]) is True
    after = store.get_payload_revision("conv-1")
    assert after > before


def test_get_payload_revision_defaults_to_zero_for_unknown_conversation():
    store = ConsoleChatStore()
    assert store.get_payload_revision("nope") == 0


# ---------------------------------------------------------------------------
# 2. Live tail-follow (seam pilot)
# ---------------------------------------------------------------------------


class TrajectoryHostApp(App[None]):
    """Hosts TrajectoryScreen with pluggable revision/snapshot callables.

    The screen is PUSHED (like the Console does) rather than composed --
    only an active screen's BINDINGS take part in key routing.
    """

    def __init__(self, snapshot, revision_provider, snapshot_builder) -> None:
        super().__init__()
        self._snapshot = snapshot
        self._revision_provider = revision_provider
        self._snapshot_builder = snapshot_builder
        self.screen_instance: TrajectoryScreen | None = None

    def compose(self) -> ComposeResult:

        yield Static("base")

    def on_mount(self) -> None:
        self.screen_instance = TrajectoryScreen(
            self._snapshot,
            screen_title="live",
            conversation_id="conv-1",
            revision_provider=self._revision_provider,
            snapshot_builder=self._snapshot_builder,
        )
        self.push_screen(self.screen_instance)


async def test_live_revision_change_appends_rows_and_follows_tail():
    state = {"revision": 7, "turns": 3}
    builder = lambda: snapshot_with_turns(state["turns"])  # noqa: E731
    app = TrajectoryHostApp(snapshot_with_turns(3), lambda: state["revision"], builder)
    async with app.run_test() as pilot:
        screen = app.screen_instance
        table = screen.query_one("#trajectory-table", DataTable)
        rows_before = table.row_count
        assert rows_before == 6 + 3  # 6 records + 3 turn headers

        # A new turn lands (revision moves); the poll tick rebuilds.
        state["turns"] = 4
        state["revision"] += 1
        screen._poll_revision()
        for _ in range(40):
            if table.row_count >= rows_before + 3:
                break
            await pilot.pause(0.05)
        # new turn header + user + assistant rows
        assert table.row_count == 8 + 4
        # follow was never suspended: the ledger sits at the tail
        assert screen._follow is True
        for _ in range(500):
            if table.scroll_y == table.max_scroll_y:
                break
            await pilot.pause(0.01)
        assert table.scroll_y == table.max_scroll_y
        # live screen advertises the follow key
        assert "follow" in str(screen.query_one("#trajectory-hints").render())


async def test_a_live_trace_opens_on_its_newest_events_and_keeps_following(
    wait_for_condition,
):
    """A live Trace that overflows opens at the tail with follow engaged.

    It used to open at the top while its state line read FOLLOWING, then
    pause itself on the first poll tick (TASK-42414).
    """

    async def until(condition):
        await wait_for_condition(condition, timeout=10.0, interval=0.01)

    state = {"revision": 3, "turns": 60}
    builder = lambda: snapshot_with_turns(state["turns"])  # noqa: E731
    app = TrajectoryHostApp(snapshot_with_turns(60), lambda: state["revision"], builder)
    async with app.run_test():
        screen = app.screen_instance
        table = screen.query_one("#trajectory-table", DataTable)
        await until(lambda: table.max_scroll_y > 0)
        # It reaches the tail, or a poll tick pauses it at the top.
        await until(lambda: table.scroll_y == table.max_scroll_y or not screen._follow)
        assert screen._follow is True
        assert table.scroll_y == table.max_scroll_y
        # The cursor opens on the newest row too, so the first key press
        # moves from where the reader is instead of jumping to the top.
        assert table.cursor_row == table.row_count - 1

        await until(lambda: not screen._scrolls_pending)
        screen._sync_follow_from_scroll()  # what the next poll tick does
        assert screen._follow is True
        state_line = str(screen.query_one("#trajectory-state", Static).render())
        assert "LIVE · FOLLOWING" in state_line


async def test_scrolling_up_suspends_follow_until_f_resumes(wait_for_condition):
    # 60 turns overflow any pilot viewport, so scrolling is real. A scroll
    # lands a refresh or more after it is asked for, and a loaded runner can
    # take several, so each step waits on the table's own geometry with a
    # deadline, not on a count of pauses (TASK-42414).
    async def until(condition):
        await wait_for_condition(condition, timeout=10.0, interval=0.01)

    state = {"revision": 3, "turns": 60}
    builder = lambda: snapshot_with_turns(state["turns"])  # noqa: E731
    app = TrajectoryHostApp(snapshot_with_turns(60), lambda: state["revision"], builder)
    async with app.run_test() as pilot:
        screen = app.screen_instance
        table = screen.query_one("#trajectory-table", DataTable)
        # The Trace opens by scrolling itself to the tail; let that land first.
        await until(lambda: table.max_scroll_y > 0 and not screen._scrolls_pending)
        # Immediate: a DataTable applies scroll_to at once, so a deferred
        # scroll_end could land after the reader's scroll below.
        table.scroll_end(animate=False, immediate=True)
        await until(lambda: table.scroll_y == table.max_scroll_y)
        screen._sync_follow_from_scroll()
        assert screen._follow is True

        # The reader scrolls up to inspect history.
        table.scroll_to(y=0, animate=False)
        await until(lambda: table.scroll_y == 0)
        screen._sync_follow_from_scroll()
        assert screen._follow is False

        # New records arrive while reading: no tail jump, even once every
        # queued ledger scroll has had its chance to land.
        state["turns"] = 61
        state["revision"] += 1
        screen._poll_revision()
        await until(lambda: table.row_count == 61 * 3 and not screen._scrolls_pending)
        assert screen._follow is False
        assert table.scroll_y == 0  # reading position preserved

        # f re-enables follow and jumps to the tail.
        await pilot.press("f")
        assert screen._follow is True
        await until(lambda: table.scroll_y == table.max_scroll_y)


async def test_a_reader_scroll_during_a_live_rebuild_is_not_snapped_back(
    monkeypatch, wait_for_condition
):
    """The rebuild a tick started at the tail lands after the reader scrolled up.

    It must leave the reader where they are and suspend follow, not scroll
    them back to the tail (TASK-42414).
    """

    async def until(condition):
        await wait_for_condition(condition, timeout=10.0, interval=0.01)

    # The test runs the poll tick itself, so no interval tick re-reads the
    # scroll position while the rebuild is held.
    poll = TrajectoryScreen._poll_revision
    monkeypatch.setattr(TrajectoryScreen, "_poll_revision", lambda self: None)
    release = threading.Event()
    state = {"revision": 3, "turns": 60}

    def builder():
        release.wait(10)
        return snapshot_with_turns(state["turns"])

    app = TrajectoryHostApp(snapshot_with_turns(60), lambda: state["revision"], builder)
    async with app.run_test():
        screen = app.screen_instance
        table = screen.query_one("#trajectory-table", DataTable)
        try:
            await until(lambda: table.max_scroll_y > 0 and not screen._scrolls_pending)
            table.scroll_end(animate=False, immediate=True)
            await until(lambda: table.scroll_y == table.max_scroll_y)

            # A new turn lands; the tick sees the reader at the tail and
            # starts the rebuild, which is held.
            state["turns"] = 61
            state["revision"] += 1
            poll(screen)
            assert screen._follow is True

            # The reader scrolls up while the rebuild is in flight.
            table.scroll_to(y=0, animate=False)
            await until(lambda: table.scroll_y == 0)
        finally:
            release.set()

        await until(lambda: table.row_count == 61 * 3 and not screen._scrolls_pending)
        assert table.scroll_y == 0, "the rebuild snapped the reader back to the tail"
        assert screen._follow is False


async def test_back_to_back_live_rebuilds_keep_following_the_tail(wait_for_condition):
    """A rebuild landing before the last one's tail scroll keeps following.

    The first re-render reset the scroll to the top; the second must not
    read that as the reader scrolling up (TASK-42414).
    """

    async def until(condition):
        await wait_for_condition(condition, timeout=10.0, interval=0.01)

    state = {"revision": 3, "turns": 60}
    builder = lambda: snapshot_with_turns(state["turns"])  # noqa: E731
    app = TrajectoryHostApp(snapshot_with_turns(60), lambda: state["revision"], builder)
    async with app.run_test():
        screen = app.screen_instance
        table = screen.query_one("#trajectory-table", DataTable)
        await until(lambda: table.max_scroll_y > 0 and not screen._scrolls_pending)
        table.scroll_end(animate=False, immediate=True)
        await until(lambda: table.scroll_y == table.max_scroll_y)
        screen._sync_follow_from_scroll()
        assert screen._follow is True

        screen._apply_live_snapshot(snapshot_with_turns(61))
        screen._apply_live_snapshot(snapshot_with_turns(62))
        assert screen._follow is True

        await until(lambda: table.row_count == 62 * 3 and not screen._scrolls_pending)
        assert screen._follow is True
        assert table.scroll_y == table.max_scroll_y


@pytest.mark.parametrize("turns", [60, 260], ids=["appended", "newest-page-slides"])
async def test_a_paused_reader_keeps_their_place_across_a_live_rebuild(
    turns, wait_for_condition
):
    """A rebuild while paused leaves the reader on the rows they were reading.

    The re-render's ``DataTable.clear()`` resets the scroll and only the
    cursor row came back, so a reader who had scrolled away from the cursor
    was thrown back to it (TASK-42414). Past the newest page (500 records)
    a new turn also drops rows off the top, so the place is a row, not an
    offset.
    """

    async def until(condition):
        await wait_for_condition(condition, timeout=10.0, interval=0.01)

    state = {"revision": 3, "turns": turns}
    builder = lambda: snapshot_with_turns(state["turns"])  # noqa: E731
    app = TrajectoryHostApp(
        snapshot_with_turns(turns), lambda: state["revision"], builder
    )
    async with app.run_test():
        screen = app.screen_instance
        table = screen.query_one("#trajectory-table", DataTable)
        await until(lambda: table.max_scroll_y > 0 and not screen._scrolls_pending)

        # The reader scrolls to the middle without moving the cursor.
        table.scroll_to(y=40, animate=False)
        await until(lambda: table.scroll_y == 40)
        screen._sync_follow_from_scroll()
        assert screen._follow is False
        reading = screen._visible_keys[40]
        assert table.cursor_row != 40

        state["turns"] = turns + 1
        state["revision"] += 1
        screen._poll_revision()
        await until(
            lambda: (
                screen._total_records == (turns + 1) * 2 and not screen._scrolls_pending
            )
        )
        top = int(table.scroll_y)
        assert screen._visible_keys[top] == reading, (
            f"the reader's top row {reading} now shows {screen._visible_keys[top]}"
        )
        assert screen._follow is False


async def _paused_at(pilot, until, turns, row):
    """Open a live Trace of ``turns`` turns and pause it with ``row`` at the top."""
    screen = pilot.app.screen_instance
    table = screen.query_one("#trajectory-table", DataTable)
    await until(lambda: table.max_scroll_y > 0 and not screen._scrolls_pending)
    table.scroll_to(y=row, animate=False)
    await until(lambda: table.scroll_y == row)
    screen._sync_follow_from_scroll()
    assert screen._follow is False
    return screen, table


@pytest.mark.parametrize("first", ["paused-rebuild", "re-render"])
async def test_a_second_rebuild_in_the_same_frame_keeps_the_readers_row(
    first, wait_for_condition
):
    """A rebuild that lands while a re-render's reset is still in flight.

    The first re-render reset the scroll to the top at once; the second
    must not read that reset as the reader's place (TASK-42414 review).
    """

    async def until(condition):
        await wait_for_condition(condition, timeout=10.0, interval=0.01)

    app = TrajectoryHostApp(
        snapshot_with_turns(260), lambda: 3, lambda: snapshot_with_turns(260)
    )
    async with app.run_test() as pilot:
        screen, table = await _paused_at(pilot, until, 260, 40)
        reading = screen._visible_keys[40]
        assert reading == "turn:t23"

        # Both in one frame: nothing lands between them.
        if first == "paused-rebuild":
            screen._apply_live_snapshot(snapshot_with_turns(261))
        else:
            screen._render_ledger()  # what a search key, t, e or a filter runs
        screen._apply_live_snapshot(snapshot_with_turns(262))

        await until(lambda: not screen._scrolls_pending)
        top = int(table.scroll_y)
        assert screen._visible_keys[top] == reading, (
            f"the reader's top row {reading} now shows {screen._visible_keys[top]}"
        )


@pytest.mark.parametrize("row", [0, 2], ids=["earlier-row-at-top", "oldest-record-at-top"])
async def test_a_paused_reader_on_the_oldest_rows_keeps_them(row, wait_for_condition):
    """Past 500 records each new turn drops the oldest rows off the window.

    A paused reader reading those rows keeps them: the window grows to hold
    them, as it does for the cursor row (TASK-42414 review).
    """

    async def until(condition):
        await wait_for_condition(condition, timeout=10.0, interval=0.01)

    app = TrajectoryHostApp(
        snapshot_with_turns(260), lambda: 3, lambda: snapshot_with_turns(260)
    )
    async with app.run_test() as pilot:
        screen, table = await _paused_at(pilot, until, 260, row)
        seen = screen._visible_keys[row : row + 6]
        for turns in (261, 262, 263):
            screen._apply_live_snapshot(snapshot_with_turns(turns))
            await until(lambda: not screen._scrolls_pending)
            top = int(table.scroll_y)
            assert screen._visible_keys[top : top + 6] == seen, (
                f"after {turns} turns the reader sees "
                f"{screen._visible_keys[top : top + 6]}, not {seen}"
            )


async def test_navigating_while_a_live_trace_loads_keeps_the_cursor_in_view(
    monkeypatch, wait_for_condition
):
    """A key press that renders the ledger before the worker's render lands.

    That render is the first one, but the reader picked a row: the Trace
    must show it, not open at the tail with the cursor off screen, and the
    render ends the load (TASK-42414 review).
    """

    async def until(condition):
        await wait_for_condition(condition, timeout=10.0, interval=0.01)

    release = threading.Event()
    worker_done = threading.Event()
    build = TrajectoryScreen._build_row_specs
    render_worker = TrajectoryScreen._render_worker

    def held_off_thread(self):
        if threading.current_thread() is not threading.main_thread():
            release.wait(10)
        return build(self)

    def render_worker_then_flag(self):
        try:
            render_worker(self)  # returns once its render was applied or dropped
        finally:
            worker_done.set()

    monkeypatch.setattr(TrajectoryScreen, "_build_row_specs", held_off_thread)
    monkeypatch.setattr(TrajectoryScreen, "_render_worker", render_worker_then_flag)
    turns = 600  # 1200 records: past WORKER_THRESHOLD, so a worker renders it
    app = TrajectoryHostApp(
        snapshot_with_turns(turns), lambda: 3, lambda: snapshot_with_turns(turns)
    )
    async with app.run_test() as pilot:
        screen = app.screen_instance
        table = screen.query_one("#trajectory-table", DataTable)
        try:
            await until(lambda: screen._loading)
            await pilot.press("n")
            await until(lambda: table.row_count > 0)
        finally:
            release.set()
        # The worker's build lands last (and is dropped as stale), so by then
        # the load must be over: its LOADING label once stayed up for good.
        await until(lambda: worker_done.is_set() and not screen._scrolls_pending)
        state_line = str(screen.query_one("#trajectory-state", Static).render())
        assert "LOADING" not in state_line, state_line
        assert screen._loading is False

        top = int(table.scroll_y)
        height = table.scrollable_content_region.height
        assert top <= table.cursor_row < top + height, (
            f"cursor row {table.cursor_row} is off screen ({top}..{top + height})"
        )
        screen._sync_follow_from_scroll()  # what the next poll tick does
        assert screen._follow is False


async def test_unchanged_revision_does_not_rebuild():
    state = {"revision": 5, "turns": 3}
    builder_calls = []

    def builder():
        builder_calls.append(1)
        return snapshot_with_turns(state["turns"])

    app = TrajectoryHostApp(snapshot_with_turns(3), lambda: state["revision"], builder)
    async with app.run_test() as pilot:
        screen = app.screen_instance
        screen._poll_revision()
        for _ in range(10):
            await pilot.pause(0.05)
        assert builder_calls == []  # revision static: no recompute scheduled


def test_follow_binding_registered_and_hints_stay_one_to_one():
    """`f` is bound with an implemented action; class-level hints stay 1:1.

    The rendered hint drops on non-live screens (``_refresh_hints`` filter);
    class-level equality is what the ADR-299 governance suite asserts.
    """
    binding_keys = {
        binding.key for binding in TrajectoryScreen.BINDINGS if binding.key != "escape"
    }
    hint_keys = {key for key, _label in TrajectoryScreen.TRAJECTORY_SHORTCUTS}
    assert "f" in binding_keys
    assert hasattr(TrajectoryScreen, "action_resume_follow")
    assert hint_keys == binding_keys


# ---------------------------------------------------------------------------
# 3. Launch binding registration (ADR-299 governance, non-async)
# ---------------------------------------------------------------------------


def test_console_binds_single_letter_trajectory_launch():
    bindings = {b.key: b.action for b in ChatScreen.BINDINGS}
    assert bindings.get("y") == "open_trajectory_view"
    # TASK-33622.3: 'y' acts only from the transcript (the announced list
    # mode), so only the transcript's footer variant advertises it.
    assert ("Y", "trace") in TRANSCRIPT_KEYS_FOOTER
    assert ("Y", "trace") not in CONSOLE_WORKBENCH_SHORTCUTS
    assert hasattr(ChatScreen, "action_open_trajectory_view")
    # 'j' stays owned by the focused transcript (next-message selection in
    # console_transcript.on_key); the launch key must not collide with it.
    assert "j" not in bindings
    # ADR-299: single-letter htop-style key, no terminal-convention chord.
    assert len("y") == 1


def test_build_trajectory_snapshot_renders_compaction_and_variants(tmp_path):
    """Real-seam integration: no getattr-probed phantom sources.

    Drives ``_build_trajectory_snapshot`` against a real temp DB + store:
    message/tool rows land via the Task 2 write path, a compaction attempt
    via ``ConsoleContextRepository``, and a variant set in-memory -- then
    asserts compaction and superseded variants actually reach the snapshot.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hello",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="first draft",
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_list → (3 files)",
            tool_output_full="file-a\nfile-b\nfile-c",
        )
        # In-memory variant: 'first draft' becomes superseded content.
        store.add_variant(assistant.id, "second draft wins")

        # Durable compaction attempt through the real repository seam.
        ConsoleContextRepository(db).start_auxiliary_attempt(
            AuxiliaryAttemptStart(
                operation_id="op-compaction-1",
                conversation_id=conversation_id,
                purpose="conversation_compaction",
                provider="test-provider",
                model="test-model",
                requested_output_cap=100,
                estimated_input_tokens=50,
                started_at="2026-08-14T00:00:00Z",
            )
        )

        snapshot = _build_trajectory_snapshot(store, conversation_id)

        kinds = [r.kind for turn in snapshot.turns for r in turn.records]
        assert "user" in kinds and "assistant" in kinds
        assert "tool_call" in kinds and "tool_result" in kinds
        assert "compaction" in kinds

        superseded = [
            variant
            for turn in snapshot.turns
            for record in turn.records
            for variant in record.variants
        ]
        assert "first draft" in superseded
        assert "second draft wins" not in superseded  # selected == current
    finally:
        db.close()


def test_build_trajectory_snapshot_threads_agent_and_retrieval_owners():
    """The off-thread builder joins public durable owner read seams."""
    requested_agent_kinds: list[str | None] = []

    class _DB:
        def get_messages_for_conversation(self, *_args, **_kwargs):
            return [
                {
                    "id": "a1",
                    "sender": "assistant",
                    "content": "answer",
                    "timestamp": 1.0,
                    "parent_message_id": None,
                    "deleted": False,
                    "usage_json": None,
                }
            ]

        def get_trajectory_rows(self, _conversation_id):
            return [TrajRow("a1", turn_id="t1", seq=1, event_kind="assistant")]

        def get_conversation_active_leaf(self, _conversation_id):
            return "a1"

    evidence_run = SimpleNamespace(
        model_dump=lambda mode="python": {
            "run_id": "rag-1",
            "run_ordinal": 1,
            "stage": "search",
            "started_at": "2026-08-22T12:00:00Z",
            "ended_at": "2026-08-22T12:00:01Z",
        }
    )
    active_result = SimpleNamespace(
        state=ActiveCitationTraceState.ACTIVE,
        summary=SimpleNamespace(trace=SimpleNamespace(evidence_runs=(evidence_run,))),
    )

    class _CitationRepository:
        def active_owner_candidate_message_ids(self, message_ids):
            assert message_ids == ["a1"]
            return {"a1"}

        def get_active_trace_for_current_message(self, message_id, current_body):
            assert (message_id, current_body) == ("a1", "answer")
            return active_result

        def verify_active_trace_result(self, result):
            return result is active_result

    class _RunsDB:
        def list_runs(self, conversation_id, *, agent_kind=None):
            assert conversation_id == "conv-1"
            requested_agent_kinds.append(agent_kind)
            if agent_kind is None:
                raise AssertionError(
                    "broad query would hydrate poison local-command steps"
                )
            if agent_kind == "subagent":
                return []
            assert agent_kind == "primary"
            return [
                {
                    "id": "run-1",
                    "conversation_id": conversation_id,
                    "agent_kind": "primary",
                    "status": "done",
                    "created_at": "2026-08-22T12:00:02Z",
                    "assistant_message_id": "a1",
                    "steps": [
                        {
                            "index": 0,
                            "kind": "model",
                            "summary": "answered",
                            "created_at": "2026-08-22T12:00:03Z",
                        }
                    ],
                }
            ]

    persistence = SimpleNamespace(
        db=_DB(),
        context_repository=None,
        citation_repository=_CitationRepository(),
    )
    store = SimpleNamespace(
        persistence=persistence,
        variant_sets_for_conversation=lambda _conversation_id: (),
    )

    snapshot = _build_trajectory_snapshot(
        store,
        "conv-1",
        agent_runs_db=_RunsDB(),
    )
    event_ids = {record.event_id for turn in snapshot.turns for record in turn.records}

    assert "agent-run:run-1" in event_ids
    assert "agent-step:run-1:0" in event_ids
    assert "retrieval-run:rag-1" in event_ids
    ordered_ids = [
        record.event_id for turn in snapshot.turns for record in turn.records
    ]
    assert ordered_ids.index("retrieval-run:rag-1") < ordered_ids.index("message:a1")
    retrieval = next(
        record
        for turn in snapshot.turns
        for record in turn.records
        if record.event_id == "retrieval-run:rag-1"
    )
    assert retrieval.parent_event_id is None
    agent_run = next(
        record
        for turn in snapshot.turns
        for record in turn.records
        if record.event_id == "agent-run:run-1"
    )
    assert agent_run.turn_id == "t1"
    assert requested_agent_kinds == ["primary", "subagent"]



def test_snapshot_builder_puts_sub_agent_runs_in_the_spawning_turn():
    """TASK-41650: a sub-agent run has no reply of its own, so it inherits
    its parent run's turn whatever order list_runs returns (newest first
    here, with a grandchild read before its parent); a run whose parent is
    missing falls back to the conversation as before."""

    class _DB:
        def get_messages_for_conversation(self, *_args, **_kwargs):
            return [
                msg("u1", "user", content="q", ts=1.0),
                msg("a1", "assistant", content="a", ts=2.0, parent="u1"),
            ]

        def get_trajectory_rows(self, _conversation_id):
            return [
                TrajRow("u1", turn_id="t1", seq=1, event_kind="user"),
                TrajRow("a1", turn_id="t1", seq=2, event_kind="assistant"),
            ]

        def get_conversation_active_leaf(self, _conversation_id):
            return "a1"

    def run(run_id, kind, created, **extra):
        return {
            "id": run_id,
            "conversation_id": "conv-1",
            "agent_kind": kind,
            "status": "done",
            "created_at": created,
            **extra,
        }

    class _Runs:
        def list_runs(self, _conversation_id, *, agent_kind=None):
            if agent_kind == "primary":
                return [run("p1", "primary", "2026-08-22T12:00:00Z", assistant_message_id="a1")]
            return [  # newest first, as AgentRunsDB orders them
                run("orphan", "subagent", "2026-08-22T12:00:03Z", parent_run_id="gone"),
                run("g1", "subagent", "2026-08-22T12:00:02Z", parent_run_id="c1"),
                run("c1", "subagent", "2026-08-22T12:00:01Z", parent_run_id="p1"),
            ]

    store = SimpleNamespace(
        persistence=SimpleNamespace(db=_DB()),
        variant_sets_for_conversation=lambda _conversation_id: (),
    )

    snapshot = _build_trajectory_snapshot(store, "conv-1", agent_runs_db=_Runs())
    turn_of = {
        record.run_id: record.turn_id
        for turn in snapshot.turns
        for record in turn.records
        if record.kind == "agent_run"
    }

    assert turn_of == {"p1": "t1", "c1": "t1", "g1": "t1", "orphan": "conv-1"}

def test_snapshot_builder_prefilters_citations_and_pages_context_failures():
    class _DB:
        def get_messages_for_conversation(self, *_args, **_kwargs):
            return [msg(f"a{i}", "assistant", content="x", ts=i) for i in range(20)]

        def get_trajectory_rows(self, _conversation_id):
            return []

        def get_conversation_active_leaf(self, _conversation_id):
            raise RuntimeError("leaf unavailable")

    class _Context:
        def __init__(self):
            self.offsets = []

        def list_auxiliary_attempts(self, _conversation_id, *, limit, offset):
            self.offsets.append(offset)
            if offset:
                raise RuntimeError("late context failure")
            return [{"operation_id": f"op-{i}"} for i in range(500)]

    class _Runs:
        def list_runs(self, _conversation_id, *, agent_kind=None):
            if agent_kind is None:
                raise AssertionError(
                    "broad query would hydrate poison local-command steps"
                )
            if agent_kind == "subagent":
                return
            assert agent_kind == "primary"
            yield {"id": "retained-run", "conversation_id": "conv-1"}
            raise RuntimeError("late agent failure")

    class _Citations:
        candidate_calls = 0
        detail_calls = 0

        def active_owner_candidate_message_ids(self, message_ids):
            self.candidate_calls += 1
            assert len(message_ids) == 20
            return set()

        def get_active_trace_for_current_message(self, *_args):
            self.detail_calls += 1
            raise AssertionError("prefilter should skip detail reads")

    context = _Context()
    citations = _Citations()
    store = SimpleNamespace(
        persistence=SimpleNamespace(
            db=_DB(), context_repository=context, citation_repository=citations
        ),
        variant_sets_for_conversation=lambda _conversation_id: (),
    )

    snapshot = _build_trajectory_snapshot(store, "conv-1", agent_runs_db=_Runs())
    records = [record for turn in snapshot.turns for record in turn.records]

    assert context.offsets == [0, 500]
    assert citations.candidate_calls == 1
    assert citations.detail_calls == 0
    assert sum(record.kind == "compaction" for record in records) == 500
    assert any(record.event_id == "agent-run:retained-run" for record in records)
    failure = next(record for record in records if record.kind == "capture_failed")
    assert failure.status == "capture_failed"
    assert failure.field_states == {"source": "capture_failed"}
    assert failure.sensitivity == "diagnostic"


def test_retrieval_failures_are_distinct_and_message_owned():
    messages = [
        msg("a1", "assistant", content="x", ts=1),
        msg("a2", "assistant", content="x", ts=2),
    ]

    class _DB:
        def get_messages_for_conversation(self, *_args, **_kwargs):
            return messages

        def get_trajectory_rows(self, _conversation_id):
            return []

        def get_conversation_active_leaf(self, _conversation_id):
            return None

    class _Citations:
        def active_owner_candidate_message_ids(self, message_ids):
            return set(message_ids)

        def get_active_trace_for_current_message(self, *_args):
            raise RuntimeError("SECRET")

    store = SimpleNamespace(
        persistence=SimpleNamespace(
            db=_DB(), context_repository=None, citation_repository=_Citations()
        ),
        variant_sets_for_conversation=lambda _conversation_id: (),
    )
    records = [
        r
        for turn in _build_trajectory_snapshot(store, "conv-1").turns
        for r in turn.records
    ]
    failures = [r for r in records if r.kind == "capture_failed"]

    assert {record.message_id for record in failures} == {"a1", "a2"}
    assert len({record.event_id for record in failures}) == 2
    assert all("SECRET" not in record.content_preview for record in failures)
    assert all(record.field_states["message_id"] == "observed" for record in failures)


async def test_trajectory_launch_action_presents_screen():
    """The `y` action builds off-thread and pushes a real TrajectoryScreen.

    Regression shape matters here (task-16847): the first version of this
    test monkeypatched ``instance.call_from_thread`` and
    ``instance.push_screen`` directly onto the ChatScreen instance --
    doubles for attributes that do not exist on ``Screen`` at all (both are
    App-only in Textual 8). The test passed while pressing ``y`` in the
    real app raised ``AttributeError`` inside the thread worker and never
    presented anything. This version exercises the real seams instead: the
    real ``run_worker(thread=True)``, the real ``App.call_from_thread``
    marshal from the worker thread, and the real ``App.push_screen`` -- so
    a bare-``self.`` regression on any of them fails loudly.
    """

    class _Store:
        active_session_id = "sess-1"

        class _Session:  # noqa: N801 - namespace stand-in
            id = "sess-1"
            title = "active conv"
            persisted_conversation_id = "conv-1"

        _sessions = {"sess-1": _Session}
        _messages_by_session = {"sess-1": []}

        def get_payload_revision(self, conversation_id):
            return 1

    class _HostApp(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("base")

    app = _HostApp()
    async with app.run_test() as pilot:
        instance = ChatScreen.__new__(ChatScreen)  # bypass heavy __init__
        # Real DOMNode/MessagePump plumbing without ChatScreen's heavy
        # __init__, then graft onto the running app so `self.app` (the
        # attribute the production code must reach through) resolves.
        Screen.__init__(instance)
        instance._parent = app
        # Latest dev owns the store through the app-scoped ConsoleRuntime.
        # Seed that public ownership seam directly: assigning the property
        # would try to attach this intentionally bare test screen as a full
        # Console view and require every decomposed controller.
        instance._console_runtime_ref = SimpleNamespace(chat_store=_Store())
        instance.notify = lambda *args, **kwargs: None  # not under test
        instance._review_selection = ConsoleReviewSelectionController(
            store_accessor=lambda: instance._console_runtime_ref.chat_store,
            agent_conversation_id_accessor=lambda: None,
            change_review_provider_accessor=lambda _conversation_id: None,
            run_active_accessor=lambda: False,
            run_active_for_root=lambda _root: False,
            workspace_roots_accessor=lambda: None,
            agent_runs_db_accessor=lambda: None,
            capture_policy_bindings_accessor=lambda _session_id, _conv_id: None,
            native_messages_accessor=lambda: [],
            run_worker=lambda *args, **kwargs: instance.run_worker(*args, **kwargs),
            show_feedback_comment=lambda _action, _quote: None,  # type: ignore[arg-type]
            dispatch_prompt=lambda _text: None,  # type: ignore[arg-type]
            marshal_to_ui=lambda callback, *args: app.call_from_thread(callback, *args),
            present_trajectory=lambda launch: _present_console_trajectory(
                instance, launch
            ),
            notify=instance.notify,
        )

        ChatScreen.action_open_trajectory_view(instance)
        # build() runs on a real worker thread; present() is marshaled back
        # via the real App.call_from_thread and pushes onto the real stack.
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(app.screen, TrajectoryScreen)
        assert app.screen._conversation_id == "conv-1"
