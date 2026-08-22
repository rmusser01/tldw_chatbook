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
   single-letter ADR-031-legal key that pushes the trajectory screen.

Full-app construction is deliberately avoided (the Console workbench pilot
is far heavier than this feature needs); the seam callables are the real
integration surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen, _build_trajectory_snapshot
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


async def test_scrolling_up_suspends_follow_until_f_resumes():
    # 60 turns overflow any pilot viewport, so scrolling is real.
    state = {"revision": 3, "turns": 60}
    builder = lambda: snapshot_with_turns(state["turns"])  # noqa: E731
    app = TrajectoryHostApp(snapshot_with_turns(60), lambda: state["revision"], builder)
    async with app.run_test() as pilot:
        screen = app.screen_instance
        table = screen.query_one("#trajectory-table", DataTable)
        table.scroll_end(animate=False)
        await pilot.pause()
        assert screen._follow is True

        # The reader scrolls up to inspect history.
        table.scroll_to(y=0, animate=False)
        await pilot.pause()
        screen._sync_follow_from_scroll()
        assert screen._follow is False

        # New records arrive while reading: no tail jump.
        state["turns"] = 61
        state["revision"] += 1
        screen._poll_revision()
        for _ in range(40):
            if table.row_count > 123:
                break
            await pilot.pause(0.05)
        assert screen._follow is False
        assert table.scroll_y == 0  # reading position preserved

        # f re-enables follow and jumps to the tail.
        await pilot.press("f")
        assert screen._follow is True
        assert table.scroll_y == table.max_scroll_y


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
    class-level equality is what the ADR-031 governance suite asserts.
    """
    binding_keys = {
        binding.key for binding in TrajectoryScreen.BINDINGS if binding.key != "escape"
    }
    hint_keys = {key for key, _label in TrajectoryScreen.TRAJECTORY_SHORTCUTS}
    assert "f" in binding_keys
    assert hasattr(TrajectoryScreen, "action_resume_follow")
    assert hint_keys == binding_keys


# ---------------------------------------------------------------------------
# 3. Launch binding registration (ADR-031 governance, non-async)
# ---------------------------------------------------------------------------


def test_console_binds_single_letter_trajectory_launch():
    bindings = {b.key: b.action for b in ChatScreen.BINDINGS}
    assert bindings.get("y") == "open_trajectory_view"
    assert hasattr(ChatScreen, "action_open_trajectory_view")
    # 'j' stays owned by the focused transcript (next-message selection in
    # console_transcript.on_key); the launch key must not collide with it.
    assert "j" not in bindings
    # ADR-031: single-letter htop-style key, no terminal-convention chord.
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
        def get_active_trace_for_current_message(self, message_id, current_body):
            assert (message_id, current_body) == ("a1", "answer")
            return active_result

        def verify_active_trace_result(self, result):
            return result is active_result

    class _RunsDB:
        def list_runs(self, conversation_id):
            assert conversation_id == "conv-1"
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
    agent_run = next(
        record
        for turn in snapshot.turns
        for record in turn.records
        if record.event_id == "agent-run:run-1"
    )
    assert agent_run.turn_id == "t1"


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
        instance._console_chat_store = _Store()
        instance.notify = lambda *args, **kwargs: None  # not under test

        ChatScreen.action_open_trajectory_view(instance)
        # build() runs on a real worker thread; present() is marshaled back
        # via the real App.call_from_thread and pushes onto the real stack.
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(app.screen, TrajectoryScreen)
        assert app.screen._conversation_id == "conv-1"
