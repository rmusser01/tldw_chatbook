"""The run's unproven-cleanup explanation carries its Release action (TASK-40716)."""

from types import SimpleNamespace

from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController
from tldw_chatbook.Widgets.Console.console_cleanup_release import RELEASED_COPY
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def _release_rows(transcript):
    return [
        row for row in transcript._flat_transcript_rows() if row.kind == "cleanup-release"
    ]


def test_release_row_follows_its_explanation_until_released():
    explanation = ConsoleChatMessage(
        id="explanation",
        role=ConsoleMessageRole.SYSTEM,
        content="Agent stopped: the cleanup of fs_read could not be proven.",
    )
    other = ConsoleChatMessage(
        id="other", role=ConsoleMessageRole.SYSTEM, content="Unrelated note."
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([explanation, other])
    assert _release_rows(transcript) == []

    transcript.set_unproven_cleanup_rows({"explanation": False})
    (row,) = _release_rows(transcript)
    rows = transcript._flat_transcript_rows()
    explanation_index = next(
        i for i, r in enumerate(rows) if r.kind == "message" and r.message is explanation
    )
    assert rows[explanation_index + 1].key == row.key == "cleanup-release:explanation"
    button = transcript._build_row_widget(row, track=False)
    assert isinstance(button, Button)
    assert str(button.label) == "Release tool slot"
    assert button.native_message_id == "explanation"
    assert button.has_class("console-transcript-cleanup-release")

    transcript.set_unproven_cleanup_rows({"explanation": True})
    (row,) = _release_rows(transcript)
    released = transcript._build_row_widget(row, track=False)
    assert isinstance(released, Static) and not isinstance(released, Button)
    assert str(released.render()) == RELEASED_COPY


def test_release_rows_follow_retained_runs_whatever_their_outcome():
    """A row is keyed by the run's retained cleanup, not by how the run ended."""
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    owner.bind_run("timed-out-run")
    retention = owner.reserve_tool().retain_unproven("fs_read")
    controller = SimpleNamespace(
        run_terminal_rows={"done-card": "timed-out-run", "other": "clean-run"},
        _agent_bridge=SimpleNamespace(_runtime_capacity=capacity),
    )
    pushed = []
    transcript = SimpleNamespace(set_unproven_cleanup_rows=pushed.append)
    agent = object.__new__(ConsoleAgentController)
    agent._chat_controller_accessor = lambda: controller
    messages = [_card("done-card"), _card("other")]

    agent.sync_unproven_cleanup(transcript, messages)
    assert not agent.release_unproven_tool_slot("other")
    assert agent.release_unproven_tool_slot("done-card")
    assert not agent.release_unproven_tool_slot("done-card")
    agent.sync_unproven_cleanup(transcript, messages)
    assert pushed == [{"done-card": False}, {"done-card": True}]
    assert retention.abandoned and capacity.snapshot().tool_workers == 0
    # A late proof drops the row: nothing is held any more.
    retention.prove()
    agent.sync_unproven_cleanup(transcript, messages)
    assert pushed[-1] == {}


def _card(message_id, persisted=None):
    return ConsoleChatMessage(
        id=message_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Agent stopped.",
        persisted_message_id=persisted,
    )


def test_sync_skips_lookups_before_any_run_has_finished():
    pushed = []
    agent = object.__new__(ConsoleAgentController)
    agent._chat_controller_accessor = lambda: SimpleNamespace(run_terminal_rows={})
    agent.sync_unproven_cleanup(SimpleNamespace(set_unproven_cleanup_rows=pushed.append))
    assert pushed == [{}]


async def test_pressing_release_returns_the_slot_once_and_says_what_stays_held():
    from textual.app import App
    from textual.screen import Screen

    from tldw_chatbook.Widgets.Console.console_cleanup_release import (
        cleanup_release_widget,
    )

    released = []

    class Agent:
        def release_unproven_tool_slot(self, message_id):
            import threading

            assert threading.current_thread() is not threading.main_thread()
            released.append(message_id)
            return True

    class Host(Screen):
        _agent = Agent()

        def compose(self):
            yield cleanup_release_widget("explanation", False)

    class Harness(App):
        def on_mount(self):
            self.push_screen(Host())

    async with Harness().run_test() as pilot:
        await pilot.pause()
        button = pilot.app.screen.query_one(Button)
        await pilot.click(f"#{button.id}")
        await pilot.pause()
        for _ in range(50):
            if str(button.label) == RELEASED_COPY:
                break
            await pilot.pause(0.02)
        # Released off the UI thread; the button then says what stays held.
        assert released == ["explanation"]
        assert str(button.label) == RELEASED_COPY and button.disabled


def test_release_row_renders_under_a_failed_assistant_card():
    """The run's terminal card is usually its (failed or completed) reply."""
    user = ConsoleChatMessage(id="ask", role=ConsoleMessageRole.USER, content="go")
    card = ConsoleChatMessage(
        id="card", role=ConsoleMessageRole.ASSISTANT, content="", status="failed"
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([user, card])
    transcript.set_unproven_cleanup_rows({"card": False})
    keys = [row.key for row in transcript._flat_transcript_rows()]
    assert "cleanup-release:card" in keys



def test_release_and_released_rows_never_move_the_selection():
    assert {
        "console-transcript-cleanup-release",
        "console-transcript-cleanup-released",
    } <= ConsoleTranscript.PROTECTED_CLICK_CLASSES


def test_run_rows_are_forgotten_once_nothing_can_need_release():
    """The row map stays bounded by live runs plus retained records."""
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Chat.console_unproven_cleanup import unproven_cleanup_rows

    capacity = RuntimeCapacity()

    def run(run_id):
        owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
        owner.bind_run(run_id)
        return owner, owner.reserve_tool()

    clean_owner, clean_op = run("clean")
    held_owner, held_op = run("held")
    held = held_op.retain_unproven("fs_read")
    controller = SimpleNamespace(
        run_terminal_rows={"c": "clean", "h": "held"},
        _agent_bridge=SimpleNamespace(_runtime_capacity=capacity),
    )
    messages = [_card("c"), _card("h")]

    # A run still live keeps its row; only a retained run shows one.
    assert unproven_cleanup_rows(controller, messages) == {"h": False}
    assert controller.run_terminal_rows == {"c": "clean", "h": "held"}

    # A run that drained without retention needs no Release row.
    clean_op.finish()
    clean_owner.finish_root()
    unproven_cleanup_rows(controller, messages)
    assert controller.run_terminal_rows == {"h": "held"}

    # A released record stays listed while it is retained...
    assert held.abandon()
    held_owner.finish_root()
    assert unproven_cleanup_rows(controller, messages) == {"h": True}
    # ...and a late proof forgets it.
    held.prove()
    assert unproven_cleanup_rows(controller, messages) == {}
    assert controller.run_terminal_rows == {}


def test_a_closed_and_reopened_card_keeps_its_release_row():
    """Closing the conversation must not strand a held slot without a button."""
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Chat.console_unproven_cleanup import (
        release_unproven_tool_slot,
        unproven_cleanup_rows,
    )

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    owner.bind_run("held")
    owner.reserve_tool().retain_unproven("fs_read")
    owner.finish_root()
    # The card is recorded by its persisted id once it has one.
    controller = SimpleNamespace(
        run_terminal_rows={"db-card": "held"},
        _agent_bridge=SimpleNamespace(_runtime_capacity=capacity),
    )
    assert unproven_cleanup_rows(controller, [_card("live-card", "db-card")]) == {
        "live-card": False
    }
    # Closed: no open message shows it, yet nothing forgets it.
    assert unproven_cleanup_rows(controller, []) == {}
    # Reopened: the same persisted card has a new in-memory id.
    reopened = [_card("reopened-card", "db-card")]
    assert unproven_cleanup_rows(controller, reopened) == {"reopened-card": False}
    assert release_unproven_tool_slot(controller, "reopened-card")
    assert capacity.snapshot().tool_workers == 0
    assert unproven_cleanup_rows(controller, reopened) == {"reopened-card": True}
    # Each reopen replaces the run's alias instead of adding another.
    for index in range(3):
        unproven_cleanup_rows(controller, [_card(f"reopen-{index}", "db-card")])
    assert controller.run_terminal_rows == {"db-card": "held", "reopen-2": "held"}


def test_release_returns_every_slot_a_run_retains():
    """Two retentions on one run are one row, and Release frees both slots."""
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Chat.console_unproven_cleanup import (
        release_unproven_tool_slot,
        unproven_cleanup_rows,
    )

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    owner.bind_run("held")
    first, second = owner.reserve_tool(), owner.reserve_tool()
    first.retain_unproven("fs_read")
    second.retain_unproven("native Git")
    controller = SimpleNamespace(
        run_terminal_rows={"card": "held"},
        _agent_bridge=SimpleNamespace(_runtime_capacity=capacity),
    )
    assert capacity.snapshot().tool_workers == 2
    assert unproven_cleanup_rows(controller, [_card("card")]) == {"card": False}
    assert release_unproven_tool_slot(controller, "card")
    assert capacity.snapshot().tool_workers == 0
    assert unproven_cleanup_rows(controller, [_card("card")]) == {"card": True}
    assert not release_unproven_tool_slot(controller, "card")
