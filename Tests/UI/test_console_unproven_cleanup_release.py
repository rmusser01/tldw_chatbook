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


def test_screen_sync_pushes_rows_only_once_a_run_has_one():
    pushed = []
    transcript = SimpleNamespace(set_unproven_cleanup_rows=pushed.append)
    agent = object.__new__(ConsoleAgentController)
    controller = SimpleNamespace(unproven_cleanup_runs={}, _agent_bridge=None)
    agent._chat_controller_accessor = lambda: controller

    agent.sync_unproven_cleanup(transcript)
    controller.unproven_cleanup_runs["explanation"] = "run-1"
    agent.sync_unproven_cleanup(transcript)
    # No retained record (a late proof, or another capacity) reads as released.
    assert pushed == [{}, {"explanation": True}]
    assert not agent.release_unproven_tool_slot("explanation")


async def test_pressing_release_returns_the_slot_once_and_says_what_stays_held():
    from textual.app import App
    from textual.screen import Screen

    from tldw_chatbook.Widgets.Console.console_cleanup_release import (
        cleanup_release_widget,
    )

    released = []

    class Agent:
        def release_unproven_tool_slot(self, message_id):
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
        assert released == ["explanation"]
        assert str(button.label) == RELEASED_COPY and button.disabled
