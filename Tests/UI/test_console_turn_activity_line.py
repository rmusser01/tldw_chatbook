"""The Console's live turn-activity line.

An agent turn's in-flight assistant row used to show nothing at all while a
tool ran -- measured, not assumed: with a tool call held in flight the store
reports ``status='pending'``, ``content=''`` for that row, and the
``CONSOLE_GENERATING_PLACEHOLDER`` branch in ``_message_body`` is gated on
``status == 'streaming'``, so it never fired. The user watched a bare
``Assistant`` row for the whole multi-round turn.

Every assertion here reads the MOUNTED ROW WIDGET's own renderables
(``_rendered_row_text``), never ``ConsoleTranscript._messages`` -- twice in
this programme a "rendered" assertion turned out to only prove data reached
the model.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from textual.app import ComposeResult
from textual.widgets import Markdown, Static

from Tests.Chat.test_console_agent_bridge import _bridge, _fence, _test_resolution
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    STEP_MODEL,
    STEP_TOOL_CALL,
    STEP_TOOL_RESULT,
)
from tldw_chatbook.Chat.console_agent_bridge import AgentLiveSnapshot, AgentLiveStep
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.UI.Console_Modules.agent import (
    CONSOLE_TURN_ACTIVITY_ABANDON_ACTION,
    CONSOLE_TURN_ACTIVITY_ABANDON_AFTER_SECONDS,
    CONSOLE_TURN_ACTIVITY_SEPARATOR,
    CONSOLE_TURN_ACTIVITY_THINKING,
    ConsoleAgentController,
    console_turn_activity_abandon_action,
    console_turn_activity_text,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    CONSOLE_GENERATING_PLACEHOLDER,
    CONSOLE_TURN_ACTIVITY_ABANDON_COPY,
    ConsoleMarkdownMessage,
    ConsoleTranscript,
)


class _ActivityHarness(ConsolidatedCSSApp):
    """A bare mounted transcript -- the same surface the Console mounts."""

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _rendered_row_text(transcript: ConsoleTranscript, message_id: str) -> str:
    """Visible text of one mounted message's presentation owner.

    TASK-19426 moved an Assistant answer's header onto its stable turn shell,
    while the nested message widget owns only the answer body. Resolve that
    shell when present and read its mounted ``Static`` renderables plus the
    nested markdown source, so nothing here can pass by reading the
    transcript's message model.
    """
    row = transcript.query_one(f"#console-message-{message_id}")
    turn_shells = list(transcript.query(f"#console-assistant-turn-{message_id}"))
    presentation_owner = turn_shells[0] if turn_shells else row
    parts = [str(static.renderable) for static in presentation_owner.query(Static)]
    if isinstance(row, ConsoleMarkdownMessage):
        parts.append(row.query_one(Markdown).source)
    if not parts:
        parts = [str(row.renderable)]
    return "\n".join(parts)


def _in_flight_assistant(content: str = "", status: str = "pending"):
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        id="a1",
        status=status,
    )


def _user():
    return ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi", id="u1")


async def _paint(
    transcript: ConsoleTranscript, messages, activity: str, row_id: str = "a1"
) -> str:
    """Push one poll tick's worth of state and return the rendered row text."""
    transcript.set_messages(messages)
    transcript.apply_turn_activity(activity)
    await transcript.refresh_messages()
    return _rendered_row_text(transcript, row_id)


def _snapshot(*steps, status: str = "running") -> AgentLiveSnapshot:
    return AgentLiveSnapshot(status=status, step=len(steps), steps=tuple(steps))


# --------------------------------------------------------------------------
# The red: a real agent turn, a real tool held in flight, a rendered row.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_held_tool_call_names_the_tool_in_the_rendered_assistant_row(
    tmp_path, monkeypatch
):
    """A tool held in flight must name itself in the in-flight row, and tick.

    Production path end to end: the real ``ConsoleAgentBridge`` runs a real
    ``agent_runtime`` turn whose scripted fence calls the real
    ``CalculatorTool``; that tool blocks on an Event, so the run is parked
    exactly where a slow tool parks it. The tool name reaches the row through
    ``bridge.live_snapshot`` -> ``console_turn_activity_text`` ->
    ``ConsoleTranscript.apply_turn_activity`` -- no test double in between.
    """
    from tldw_chatbook.Tools.tool_executor import CalculatorTool

    entered = threading.Event()
    release = threading.Event()

    async def _blocking_execute(self, expression):
        entered.set()
        release.wait(timeout=20)
        return {"value": 42}

    monkeypatch.setattr(CalculatorTool, "execute", _blocking_execute)
    bridge, _db, store, session, assistant_id = _bridge(
        tmp_path,
        [[_fence("calculator", {"expression": "6*7"})], ["It is 42."]],
    )

    def _drive() -> None:
        bridge.run_reply(
            conversation_id="conv-1",
            session_id=session.id,
            resolution=_test_resolution(),
            assistant_message_id=assistant_id,
            model="test-model",
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "hi"}],
            should_cancel=lambda: False,
        )

    worker = threading.Thread(target=_drive, daemon=True)
    worker.start()
    try:
        assert entered.wait(20), "the scripted tool call never reached the tool"
        observed_at = time.monotonic()
        snapshot = bridge.live_snapshot("conv-1")
        messages = store.messages_for_session(session.id)
        in_flight = [
            message
            for message in messages
            if message.role is ConsoleMessageRole.ASSISTANT
        ]
        assert in_flight, "the turn must own an in-flight assistant row"
        assert in_flight[-1].content == "", "the row is empty while the tool runs"

        # The bridge stamped a real reading off this process's monotonic
        # clock, taken as the tool was dispatched -- not a constant, and not
        # a poll-quantised approximation.
        started_at = snapshot.steps[-1].started_at
        assert started_at is not None
        assert 0.0 <= observed_at - started_at < 20.0, (observed_at, started_at)

        app = _ActivityHarness()
        async with app.run_test(size=(80, 24)) as pilot:
            transcript = app.query_one(ConsoleTranscript)

            # Both ticks are driven off that production reading, so the
            # assertions below cannot flake on how long this test's own
            # setup happened to take.
            row_id = in_flight[-1].id
            first = console_turn_activity_text(snapshot, now=started_at + 0.2)
            early = await _paint(transcript, messages, first, row_id)
            await pilot.pause()

            assert "calculator" in early, early
            assert CONSOLE_GENERATING_PLACEHOLDER not in early, early
            assert "<1s" in early, early

            # A later poll tick, same held call: only the elapsed moves.
            later_text = console_turn_activity_text(snapshot, now=started_at + 5.0)
            later = await _paint(transcript, messages, later_text, row_id)
            await pilot.pause()

            assert "calculator" in later, later
            assert "5s" in later, later
            assert "<1s" not in later, later
    finally:
        release.set()
        worker.join(20)


# --------------------------------------------------------------------------
# The states
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_between_tool_calls_the_row_says_thinking_not_a_stale_tool_name():
    """After a tool returns, the row reports the model turn, not the tool."""
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        snapshot = _snapshot(
            AgentLiveStep(STEP_TOOL_CALL, "read_file", AGENT_KIND_PRIMARY, 100.0),
            AgentLiveStep(STEP_TOOL_RESULT, "read_file → ok", AGENT_KIND_PRIMARY, 104.0),
        )
        text = await _paint(
            transcript,
            [_user(), _in_flight_assistant()],
            console_turn_activity_text(snapshot, now=110.0),
        )
        await pilot.pause()
        assert CONSOLE_TURN_ACTIVITY_THINKING in text, text
        assert "6s" in text, text
        assert "read_file" not in text, text


@pytest.mark.asyncio
async def test_before_the_first_step_the_row_keeps_the_generating_copy():
    """A running turn with no step yet is pre-first-token: today's copy."""
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        text = await _paint(
            transcript,
            [_user(), _in_flight_assistant()],
            console_turn_activity_text(_snapshot(), now=1.0),
        )
        await pilot.pause()
        assert CONSOLE_GENERATING_PLACEHOLDER in text, text
        # No elapsed: there is no per-step base to time from, and inventing
        # one would be a lie (`_format_fleet_elapsed`'s own rule). The
        # separator is the only thing an elapsed segment can arrive behind.
        assert CONSOLE_TURN_ACTIVITY_SEPARATOR not in text, text


@pytest.mark.asyncio
async def test_a_finished_turn_drops_the_activity_line_from_the_row():
    """When the turn ends the line disappears -- the reply renders alone."""
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        running = _snapshot(
            AgentLiveStep(STEP_TOOL_CALL, "read_file", AGENT_KIND_PRIMARY, 100.0)
        )
        during = await _paint(
            transcript,
            [_user(), _in_flight_assistant()],
            console_turn_activity_text(running, now=101.0),
        )
        await pilot.pause()
        assert "read_file" in during, during

        done = _snapshot(
            AgentLiveStep(STEP_TOOL_RESULT, "read_file → ok", AGENT_KIND_PRIMARY, 104.0),
            status="done",
        )
        assert console_turn_activity_text(done, now=110.0) == ""
        after = await _paint(
            transcript,
            [
                _user(),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="It is 42.",
                    id="a1",
                    status="complete",
                ),
            ],
            console_turn_activity_text(done, now=110.0),
        )
        await pilot.pause()
        assert "read_file" not in after, after
        assert CONSOLE_TURN_ACTIVITY_THINKING not in after, after
        assert CONSOLE_GENERATING_PLACEHOLDER not in after, after
        assert "It is 42." in after, after


@pytest.mark.asyncio
async def test_only_the_elapsed_changing_repaints_the_default_markdown_row():
    """The markdown row's header must tick on its OWN signature input.

    Found by mutation. A markdown row carries the line in its HEADER, which
    `_message_row_signature` never renders -- it renders the PLAIN row. With
    `live_activity` absent from that signature the elapsed still advanced,
    but only because the plain renderer happened to embed the same text; the
    two renderers were silently coupled. This paints two ticks that differ
    in NOTHING but the elapsed and checks the row updates in place.
    """
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        messages = [_user(), _in_flight_assistant()]
        # TASK-19426 groups an Assistant answer and its activity markers under
        # one stable turn shell; that composite owns the top-level render key.
        row_key = "assistant-turn:a1"

        first = await _paint(transcript, messages, "⚙ read_file · 4s")
        await pilot.pause()
        assert isinstance(transcript.query_one("#console-message-a1"), ConsoleMarkdownMessage)
        assert "4s" in first, first
        builds = transcript.row_build_counts().get(row_key, 0)

        second = await _paint(transcript, messages, "⚙ read_file · 5s")
        await pilot.pause()
        assert "5s" in second, second
        assert "4s" not in second, second
        assert transcript.row_build_counts().get(row_key, 0) == builds, (
            "the row must update in place, not be rebuilt"
        )


@pytest.mark.asyncio
async def test_a_row_still_in_flight_loses_its_line_when_the_run_goes_quiet():
    """An empty line CLEARS the last one -- it never leaves it hanging.

    Found by mutation. The row that stays ``pending`` after its run dies
    without a terminal publish is exactly the row a stale line would sit on
    forever, frozen at whatever elapsed it last painted -- the "looks
    frozen" defect this feature exists to remove, in a new costume. The
    caller's ``""`` must therefore be applied, not skipped.
    """
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        messages = [_user(), _in_flight_assistant()]
        during = await _paint(transcript, messages, "⚙ read_file · 4s")
        await pilot.pause()
        assert "read_file" in during, during

        after = await _paint(transcript, messages, "")
        await pilot.pause()
        assert "read_file" not in after, after


@pytest.mark.asyncio
async def test_a_ticking_line_repaints_only_its_own_row():
    """The line is scoped to ONE row -- every other row must be untouched.

    Found by mutation, and invisible to every display assertion: stamping
    `live_activity` on every message instead of just the in-flight one
    changes nothing a reader can SEE (a row with content never renders the
    line, and only assistant rows can), but it puts `live_activity` into
    every row's signature -- so the whole transcript re-derives and re-syncs
    once a second for the entire turn. This measures the blast radius
    instead of the pixels.
    """
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = [
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER, content="earlier", id="u0"
            ),
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT, content="a reply", id="a0"
            ),
            _user(),
            _in_flight_assistant(),
        ]
        await _paint(transcript, history, "⚙ read_file · 4s")
        await pilot.pause()
        before_signatures = transcript.row_render_signatures()
        before_computes = transcript.message_signature_compute_counts()

        await _paint(transcript, history, "⚙ read_file · 5s")
        await pilot.pause()
        after_signatures = transcript.row_render_signatures()
        after_computes = transcript.message_signature_compute_counts()

        moved = {
            key
            for key in before_signatures
            if before_signatures[key] != after_signatures.get(key)
        }
        assert moved == {"assistant-turn:a1"}, moved
        for message_id in ("u0", "a0", "u1"):
            assert after_computes[message_id] == before_computes[message_id], message_id
        assert after_computes["a1"] > before_computes["a1"]


@pytest.mark.asyncio
async def test_a_partially_streamed_reply_is_never_replaced_by_the_line():
    """Once real text exists it owns the row -- the line is for empty rows."""
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        messages = [_user(), _in_flight_assistant(content="Once upon", status="streaming")]
        assert transcript.apply_turn_activity("⚙ read_file · 4s") == ""
        text = await _paint(transcript, messages, "⚙ read_file · 4s")
        await pilot.pause()
        assert "Once upon" in text, text
        assert "read_file" not in text, text


# --------------------------------------------------------------------------
# Sub-agent isolation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_subagent_step_never_hijacks_the_primary_rows_activity_line():
    """A child's steps belong to the rail, never to the primary's row.

    Not theoretical: ``ConsoleAgentBridge.on_step`` routes a sub-agent step
    with an EMPTY run id into the primary's own live feed (the documented
    "no run attributed" fallback), so the primary snapshot really can carry
    a child's step as its last entry.
    """
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        snapshot = _snapshot(
            AgentLiveStep(STEP_TOOL_CALL, "read_file", AGENT_KIND_PRIMARY, 100.0),
            AgentLiveStep(STEP_TOOL_CALL, "web_fetch", AGENT_KIND_SUBAGENT, 103.0),
        )
        text = await _paint(
            transcript,
            [_user(), _in_flight_assistant()],
            console_turn_activity_text(snapshot, now=105.0),
        )
        await pilot.pause()
        assert "web_fetch" not in text, text
        assert "read_file" in text, text
        # Elapsed is the PRIMARY step's, not the child's (105 - 100).
        assert "5s" in text, text


@pytest.mark.asyncio
async def test_a_turn_with_only_subagent_steps_falls_back_to_generating():
    """No primary step yet means pre-first-token, whatever a child is doing."""
    snapshot = _snapshot(
        AgentLiveStep(STEP_TOOL_CALL, "web_fetch", AGENT_KIND_SUBAGENT, 100.0)
    )
    assert (
        console_turn_activity_text(snapshot, now=105.0)
        == CONSOLE_GENERATING_PLACEHOLDER
    )


# --------------------------------------------------------------------------
# Quiet tools, markup, and the model-step summary
# --------------------------------------------------------------------------


def test_quiet_catalog_tools_still_appear_in_the_activity_line():
    """``find_tools``/``load_tools`` are quiet in MARKERS, not in the line.

    The quiet rule keeps catalog plumbing out of the PERMANENT transcript.
    The activity line is ephemeral and exists precisely so a working turn
    never looks frozen -- suppressing these would reinstate the silent gap
    for the whole discovery round.
    """
    for name in ("find_tools", "load_tools"):
        snapshot = _snapshot(
            AgentLiveStep(STEP_TOOL_CALL, name, AGENT_KIND_PRIMARY, 100.0)
        )
        assert name in console_turn_activity_text(snapshot, now=102.0)


def test_a_model_step_never_leaks_its_summary_into_the_line():
    """``STEP_MODEL``'s summary is the raw turn text (a fence, mid-turn)."""
    snapshot = _snapshot(
        AgentLiveStep(STEP_MODEL, "```tool_call\n{...}", AGENT_KIND_PRIMARY, 100.0)
    )
    text = console_turn_activity_text(snapshot, now=103.0)
    assert text.startswith(CONSOLE_TURN_ACTIVITY_THINKING)
    assert "tool_call" not in text


@pytest.mark.asyncio
async def test_a_bracketed_tool_name_renders_literally_not_escaped():
    """Rows render markup-off, so the line is raw -- no backslash residue."""
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        snapshot = _snapshot(
            AgentLiveStep(STEP_TOOL_CALL, "fetch [docs]", AGENT_KIND_PRIMARY, 100.0)
        )
        text = await _paint(
            transcript,
            [_user(), _in_flight_assistant()],
            console_turn_activity_text(snapshot, now=101.0),
        )
        await pilot.pause()
        assert "fetch [docs]" in text, text
        assert "\\[" not in text, text


@pytest.mark.asyncio
async def test_the_plain_renderer_shows_the_activity_line_too():
    """The span renderer (``assistant_markdown = false``) is not left behind."""
    app = _ActivityHarness()
    app.app_config = {"chat_defaults": {"assistant_markdown": False}}
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        snapshot = _snapshot(
            AgentLiveStep(STEP_TOOL_CALL, "read_file", AGENT_KIND_PRIMARY, 100.0)
        )
        text = await _paint(
            transcript,
            [_user(), _in_flight_assistant()],
            console_turn_activity_text(snapshot, now=104.0),
        )
        await pilot.pause()
        assert "read_file" in text, text
        assert "4s" in text, text
        assert "Streaming…" not in text, text


@pytest.mark.asyncio
async def test_a_streaming_empty_row_shows_the_line_dim_and_without_a_status_line():
    """The line must not double up with "Streaming…", and must read as dim.

    Found by mutation: with the placeholder predicate blind to the activity
    line, a plain row that is STREAMING with empty content -- reachable on a
    fence-gated tool turn, where `reset_stream_buffer` discards leaked prose
    and leaves the row streaming-and-empty -- rendered the line as ordinary
    assistant content with a "Streaming…" status line stacked under it.
    """
    app = _ActivityHarness()
    app.app_config = {"chat_defaults": {"assistant_markdown": False}}
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        messages = [_user(), _in_flight_assistant(status="streaming")]
        text = await _paint(transcript, messages, "⚙ read_file · 4s")
        await pilot.pause()
        assert "⚙ read_file · 4s" in text, text
        assert "Streaming…" not in text, text

        body = transcript.query_one("#console-message-a1").query_one(
            ".console-transcript-message-body", Static
        )
        content = body.renderable
        dimmed = "".join(
            content.plain[span.start : span.end]
            for span in content.spans
            if "dim" in str(span.style)
        )
        assert "⚙ read_file · 4s" in dimmed, (content.plain, content.spans)


# --------------------------------------------------------------------------
# Geometry and the no-idle-repaint contract
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_activity_line_never_pushes_the_row_past_the_screen():
    """Rendered geometry, not DOM presence: neighbours stay on screen."""
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        snapshot = _snapshot(
            AgentLiveStep(
                STEP_TOOL_CALL, "read_file_with_a_very_long_name", AGENT_KIND_PRIMARY, 1.0
            )
        )
        await _paint(
            transcript,
            [_user(), _in_flight_assistant()],
            console_turn_activity_text(snapshot, now=61.0),
        )
        await pilot.pause()
        screen_width = app.screen.size.width
        screen_height = app.screen.size.height
        transcript_right = transcript.content_region.right
        for row_id in ("u1", "a1"):
            row = transcript.query_one(f"#console-message-{row_id}")
            assert row.region.x + row.region.width <= screen_width, (
                row_id,
                row.region,
            )
            assert row.region.x + row.region.width <= transcript_right, (
                row_id,
                row.region,
            )
            for child in row.query(Static):
                assert child.region.x + child.region.width <= screen_width, (
                    row_id,
                    child.region,
                )
        # The other half of the hazard: a `1fr` element in a laid-out row
        # pushes its NEIGHBOURS off, so the earlier row must still be on
        # screen and the in-flight row must stay a couple of lines tall.
        user_row = transcript.query_one("#console-message-u1")
        assert 0 <= user_row.region.y < screen_height, user_row.region
        assert transcript.query_one("#console-message-a1").region.height <= 4


@pytest.mark.asyncio
async def test_no_eligible_row_means_no_effective_activity():
    """Nothing in flight -> the line is inert, so no poll can repaint for it.

    ``apply_turn_activity`` returns the EFFECTIVE value, which is what the
    screen folds into its transcript refresh key -- so a stale ``running``
    snapshot on an idle transcript cannot tick that key once per second
    (task-15664 AC#2: no repaint on a timer when nothing is live).
    """
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(
            [
                _user(),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="It is 42.",
                    id="a1",
                    status="complete",
                ),
            ]
        )
        assert transcript.apply_turn_activity("⚙ read_file · 4s") == ""
        await transcript.refresh_messages()
        await pilot.pause()
        assert "read_file" not in _rendered_row_text(transcript, "a1")


@pytest.mark.asyncio
async def test_an_eligible_row_reports_the_effective_activity():
    """The mirror of the pin above: an in-flight row does take the line."""
    app = _ActivityHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([_user(), _in_flight_assistant()])
        assert transcript.apply_turn_activity("⚙ read_file · 4s") == "⚙ read_file · 4s"


# --------------------------------------------------------------------------
# The controller's gate: only the VIEWED session's live turn owns the line
# --------------------------------------------------------------------------


class _GateController(ConsoleAgentController):
    """Exercises ``console_turn_activity``'s gate over its documented seams.

    The three members it reads (``_console_chat_controller``,
    ``_console_agent_bridge``, ``_current_console_rail_conversation_id``)
    are class-level ``@property`` pass-throughs to the screen, so they are
    overridden here rather than assigned -- the method body under test is
    the real one.
    """

    def __init__(self, *, run_status, bridge, conversation_id="conv-1") -> None:
        self._run_status = run_status
        self._bridge = bridge
        self._conversation_id = conversation_id

    @property
    def _console_chat_controller(self):
        if self._run_status is None:
            return None
        return SimpleNamespace(run_state=SimpleNamespace(status=self._run_status))

    @property
    def _console_agent_bridge(self):
        return self._bridge

    @property
    def _current_console_rail_conversation_id(self):
        return lambda: self._conversation_id


class _SnapshotBridge:
    def __init__(self, snapshot) -> None:
        self._snapshot = snapshot
        self.conversation_ids: list[str] = []

    def live_snapshot(self, conversation_id):
        self.conversation_ids.append(conversation_id)
        return self._snapshot


def _running_tool_snapshot():
    return _snapshot(
        AgentLiveStep(STEP_TOOL_CALL, "read_file", AGENT_KIND_PRIMARY, time.monotonic())
    )


def test_an_active_viewed_run_yields_the_line_for_its_own_conversation():
    bridge = _SnapshotBridge(_running_tool_snapshot())
    controller = _GateController(
        run_status=ConsoleRunStatus.STREAMING, bridge=bridge, conversation_id="conv-7"
    )
    assert "read_file" in controller.console_turn_activity()
    assert bridge.conversation_ids == ["conv-7"]


@pytest.mark.parametrize(
    "run_status", [ConsoleRunStatus.IDLE, ConsoleRunStatus.COMPLETED, None]
)
def test_an_inactive_viewed_run_never_yields_a_line(run_status):
    """The first gate: a stale ``running`` snapshot must not outlive its turn."""
    bridge = _SnapshotBridge(_running_tool_snapshot())
    controller = _GateController(run_status=run_status, bridge=bridge)
    assert controller.console_turn_activity() == ""
    assert bridge.conversation_ids == [], "the bridge is not even consulted"


def test_no_bridge_and_a_partial_bridge_double_both_yield_nothing():
    for bridge in (None, SimpleNamespace()):
        controller = _GateController(
            run_status=ConsoleRunStatus.STREAMING, bridge=bridge
        )
        assert controller.console_turn_activity() == ""


# --------------------------------------------------------------------------
# Screen wiring: the 0.2s poll's transcript sync is where the line is fed in
# --------------------------------------------------------------------------


def _sync_stub(activity: str, effective: str | None = None):
    """A ``ChatScreen``-shaped stand-in for ``_sync_native_console_transcript``.

    ``MagicMock`` answers everything the method merely CALLS; every member it
    iterates, sorts or awaits is overridden explicitly, so a future
    production addition of the first kind cannot silently no-op this test
    while one of the second kind fails loudly and by name.
    """
    transcript = SimpleNamespace(
        pending_selection_id=None,
        set_presentation_context=Mock(),
        set_change_review_provider_factory=Mock(),
        set_messages=Mock(),
        apply_turn_activity=Mock(
            return_value=activity if effective is None else effective
        ),
        set_citation_counts=Mock(),
        set_original_attempt_previews=Mock(),
        set_annotation_previews=Mock(),
        set_summary_boundary=Mock(),
        sync_jump_indicator=Mock(),
        set_image_specs=Mock(),
        set_generation_card_specs=Mock(),
        set_video_card_specs=Mock(),
        refresh_messages=AsyncMock(),
    )
    screen = MagicMock()
    screen.query_one = Mock(return_value=transcript)
    screen._console_transcript_region_or_none = Mock(return_value=None)
    screen._native_console_messages = Mock(return_value=[])
    screen._console_original_attempt_previews = {}
    screen._console_citation_counts = {}
    screen._console_speech_states = {}
    screen._console_image_preparing = set()
    screen._pending_console_swipe_selection = None
    screen._console_chat_controller = None
    screen._last_native_transcript_refresh_key = None
    screen._native_console_transcript_fingerprint = Mock(return_value=("stable",))
    screen._ensure_console_chat_store = Mock(
        return_value=SimpleNamespace(active_session_id=None)
    )
    screen._current_console_run_status_value = Mock(return_value="streaming")
    screen._image._build_console_image_specs = Mock(return_value={})
    screen._image._build_generation_card_specs = Mock(return_value={})
    screen._image._pending_console_generation_card_images = Mock(return_value=())
    screen._video._build_video_card_specs = Mock(return_value={})
    screen._ensure_console_image_view = Mock(
        return_value=(None, SimpleNamespace(pending_ids=lambda _ids: ()))
    )
    screen._message._recent_console_image_messages = Mock(return_value=())
    screen._agent.console_turn_activity = Mock(return_value=activity)
    return screen, transcript


@pytest.mark.asyncio
async def test_the_transcript_sync_hands_the_controllers_line_to_the_transcript():
    """The 0.2s poll's transcript sync is the one feed for this line."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen, transcript = _sync_stub("⚙ read_file · 4s")
    await ChatScreen._sync_native_console_transcript(screen)

    screen._agent.console_turn_activity.assert_called_once_with()
    assert transcript.apply_turn_activity.call_args.args == ("⚙ read_file · 4s",)
    assert transcript.refresh_messages.await_count == 1


@pytest.mark.asyncio
async def test_a_ticking_elapsed_alone_repaints_the_transcript():
    """The refresh key must move when only the elapsed figure changed."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen, transcript = _sync_stub("⚙ read_file · 4s")
    await ChatScreen._sync_native_console_transcript(screen)
    transcript.apply_turn_activity.return_value = "⚙ read_file · 5s"
    screen._agent.console_turn_activity.return_value = "⚙ read_file · 5s"
    await ChatScreen._sync_native_console_transcript(screen)

    assert transcript.refresh_messages.await_count == 2


@pytest.mark.asyncio
async def test_a_real_console_screen_carries_the_line_to_its_mounted_row():
    """The stubs above meet reality: a REAL screen, a REAL mounted transcript.

    Closes the gap the `MagicMock` pins leave open -- that `self._agent`
    exists on a live `ChatScreen`, that `console_turn_activity()` resolves
    through the controller's real property chain (returning "" on an idle
    screen), and that the real `_sync_native_console_transcript` carries a
    line all the way onto the real row widget.
    """
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        # An idle screen derives nothing -- and the read itself must work.
        assert console._agent.console_turn_activity() == ""

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id or store.ensure_session().id
        store.append_message(
            session_id, role=ConsoleMessageRole.USER, content="run a tool"
        )
        pending = store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        console._agent.console_turn_activity = lambda: "⚙ read_file · 4s"

        console._last_native_transcript_refresh_key = None
        await console._sync_native_console_transcript()
        await pilot.pause()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        text = _rendered_row_text(transcript, pending.id)
        assert "⚙ read_file · 4s" in text, text
        assert CONSOLE_GENERATING_PLACEHOLDER not in text, text


@pytest.mark.asyncio
async def test_an_ineffective_activity_never_repaints_an_idle_transcript():
    """task-15664 AC#2: no repaint on a timer when nothing is live.

    The screen folds the EFFECTIVE value into its refresh key, so even a
    stale ``running`` snapshot whose derived line keeps ticking cannot move
    that key while no row is in flight.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen, transcript = _sync_stub("⚙ read_file · 4s", effective="")
    await ChatScreen._sync_native_console_transcript(screen)
    screen._agent.console_turn_activity.return_value = "⚙ read_file · 5s"
    await ChatScreen._sync_native_console_transcript(screen)

    assert transcript.refresh_messages.await_count == 1


# --------------------------------------------------------------------------
# task-31386: fleet turns read as their children, and a long tool call
# offers "abandon call".
# --------------------------------------------------------------------------



def _child(*steps):
    return AgentLiveSnapshot(status="running", step=len(steps), steps=tuple(steps))


def test_a_fleet_turn_names_the_children_and_their_longest_running_tool():
    primary = _snapshot(
        AgentLiveStep(kind=STEP_TOOL_RESULT, text="spawn", agent_kind=AGENT_KIND_PRIMARY, started_at=100.0),
    )
    quick = _child(AgentLiveStep(kind=STEP_TOOL_CALL, text="read_file", agent_kind=AGENT_KIND_SUBAGENT, started_at=108.0))
    slow = _child(AgentLiveStep(kind=STEP_TOOL_CALL, text="grep_files", agent_kind=AGENT_KIND_SUBAGENT, started_at=98.0))
    thinking = _child(AgentLiveStep(kind=STEP_MODEL, text="…", agent_kind=AGENT_KIND_SUBAGENT, started_at=105.0))
    text = console_turn_activity_text(primary, now=110.0, children=[quick, slow, thinking])
    assert text == f"3 sub-agents{CONSOLE_TURN_ACTIVITY_SEPARATOR}⚙ grep_files{CONSOLE_TURN_ACTIVITY_SEPARATOR}12s"
    # No child inside a tool call: the count still replaces "Thinking…".
    assert console_turn_activity_text(primary, now=110.0, children=[thinking]) == "1 sub-agent working"
    # A child whose run has not attached (no live feed yet) still counts.
    assert console_turn_activity_text(primary, now=110.0, children=[None, slow]).startswith("2 sub-agents")
    assert console_turn_activity_text(primary, now=110.0, children=[None]) == "1 sub-agent working"


def test_a_primary_tool_call_still_wins_over_the_fleet_and_single_agent_turns_are_unchanged():
    running_tool = _snapshot(
        AgentLiveStep(kind=STEP_TOOL_CALL, text="fs_write", agent_kind=AGENT_KIND_PRIMARY, started_at=100.0),
    )
    child = _child(AgentLiveStep(kind=STEP_TOOL_CALL, text="grep_files", agent_kind=AGENT_KIND_SUBAGENT, started_at=90.0))
    assert console_turn_activity_text(running_tool, now=103.0, children=[child]).startswith("⚙ fs_write")
    between = _snapshot(
        AgentLiveStep(kind=STEP_TOOL_RESULT, text="x", agent_kind=AGENT_KIND_PRIMARY, started_at=100.0),
    )
    assert console_turn_activity_text(between, now=103.0).startswith(CONSOLE_TURN_ACTIVITY_THINKING)
    assert console_turn_activity_text(between, now=103.0, children=[]) == console_turn_activity_text(between, now=103.0)


def test_the_abandon_action_appears_only_for_a_primary_tool_call_that_has_run_long_enough():
    step = AgentLiveStep(kind=STEP_TOOL_CALL, text="slow", agent_kind=AGENT_KIND_PRIMARY, started_at=100.0)
    snapshot = _snapshot(step)
    before = 100.0 + CONSOLE_TURN_ACTIVITY_ABANDON_AFTER_SECONDS - 0.1
    after = 100.0 + CONSOLE_TURN_ACTIVITY_ABANDON_AFTER_SECONDS
    assert console_turn_activity_abandon_action(snapshot, now=before) == ""
    assert console_turn_activity_abandon_action(snapshot, now=after) == CONSOLE_TURN_ACTIVITY_ABANDON_ACTION
    child_only = _snapshot(AgentLiveStep(kind=STEP_TOOL_CALL, text="c", agent_kind=AGENT_KIND_SUBAGENT, started_at=1.0))
    assert console_turn_activity_abandon_action(child_only, now=after) == ""
    thinking = _snapshot(AgentLiveStep(kind=STEP_MODEL, text="m", agent_kind=AGENT_KIND_PRIMARY, started_at=1.0))
    assert console_turn_activity_abandon_action(thinking, now=after) == ""
    assert console_turn_activity_abandon_action(_snapshot(step, status="done"), now=after) == ""


@pytest.mark.asyncio
async def test_the_row_offers_abandon_call_only_while_the_action_is_set():
    app = _ActivityHarness()
    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.pause()
        transcript = app.query_one(ConsoleTranscript)
        row = _in_flight_assistant()
        transcript.set_messages([_user(), row])
        transcript.apply_turn_activity("⚙ slow · 6s", action=CONSOLE_TURN_ACTIVITY_ABANDON_ACTION)
        await transcript.refresh_messages()
        await pilot.pause()
        assert CONSOLE_TURN_ACTIVITY_ABANDON_COPY in _rendered_row_text(transcript, row.id)
        transcript.apply_turn_activity("⚙ slow · 7s")
        await transcript.refresh_messages()
        await pilot.pause()
        text = _rendered_row_text(transcript, row.id)
        assert "⚙ slow · 7s" in text and CONSOLE_TURN_ACTIVITY_ABANDON_COPY not in text
