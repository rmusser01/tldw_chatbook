"""TASK-1860: the full tool output must be reachable from the transcript."""
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def _row_text(app, message_id: str) -> str:
    """Renderer-agnostic visible text of one transcript row.

    The row became a `Vertical` in 065970aa4 (roleplay speaker theming), so
    `row.render()` now returns `Blank` and asserting against it is vacuous.
    Read the mounted child Statics instead -- deliberately NOT the row's
    `renderable` compatibility projection, which is derived from the row's
    in-memory message and would pass even if the mounted children never
    repainted, i.e. exactly the defect these tests exist to catch.
    """
    row = app.query_one(f"#console-message-{message_id}")
    parts = [str(static.renderable) for static in row.query(Static)]
    return "\n".join(parts) if parts else str(row.render())

def _marker(full: str | None, content: str = "⚙ read_file → data… (+900 chars)"):
    return ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL, content=content, tool_output_full=full
    )


@pytest.mark.unit
def test_a_truncated_tool_marker_offers_a_full_output_action():
    """AC#1: the route exists, and only where there is something to show."""
    actions = {
        a.action_id: a
        for a in ConsoleMessageActionService().available_actions(
            _marker("x" * 2000)
        )
    }
    assert "tool-output" in actions, (
        f"no route to the full result: {sorted(actions)}"
    )
    assert actions["tool-output"].enabled


@pytest.mark.unit
def test_a_marker_showing_everything_offers_no_full_output_action():
    """The affordance must not promise more when there is no more.

    A short result is rendered whole in `content`, so an expand action would
    open an identical view -- a dead control, which is the same defect as
    the `Review tool call` entry TASK-1843 removed.
    """
    short = "⚙ read_file → ok"
    actions = [
        a.action_id
        for a in ConsoleMessageActionService().available_actions(
            _marker(None, content=short)
        )
    ]
    assert "tool-output" not in actions, actions


@pytest.mark.unit
def test_non_tool_messages_never_offer_it():
    for role in (ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT):
        actions = [
            a.action_id
            for a in ConsoleMessageActionService().available_actions(
                ConsoleChatMessage(role=role, content="hello")
            )
        ]
        assert "tool-output" not in actions, (role, actions)


_FULL = "line one\n" + "\n".join(f"result row {n}" for n in range(2, 60))


class _TranscriptHarness(App):
    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="what is in the file?",
                    id="u1",
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.TOOL,
                    content="⚙ read_file → line one… (+900 chars)",
                    id="tool-1",
                    tool_output_full=_FULL,
                ),
            ]
        )
        transcript.selected_message_id = "tool-1"
        yield transcript


@pytest.mark.asyncio
async def test_full_tool_output_is_reachable_from_the_mounted_transcript():
    """AC#1/#6: drive the real widget, and read what is actually on screen.

    Asserting on a helper would prove nothing here -- the whole defect this
    closes is that the full result existed in memory while the transcript
    showed a preview. So this expands the selected marker the way the
    keyboard does and reads the mounted row.
    """
    app = _TranscriptHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        assert "result row 59" not in _row_text(app, "tool-1"), (
            "precondition: the collapsed row shows a preview, not everything"
        )

        transcript.toggle_tool_output("tool-1")
        await pilot.pause()
        await pilot.pause()

        expanded = _row_text(app, "tool-1")
        assert "result row 59" in expanded, (
            f"the full result is still unreachable on screen: {expanded[:200]!r}"
        )
        assert "line one" in expanded

        transcript.toggle_tool_output("tool-1")
        await pilot.pause()
        await pilot.pause()
        assert "result row 59" not in _row_text(app, "tool-1"), "expanding must be reversible"


@pytest.mark.asyncio
async def test_pressing_o_expands_the_selected_marker():
    """AC#1 in full: 'by keyboard', through the binding and the button.

    The sibling test calls `toggle_tool_output` directly, which proves the
    rendering but bypasses BOTH the binding and the action button it presses.
    This drives the key, so a missing binding, a missing action, or an
    unrouted button press all fail here.
    """
    app = _TranscriptHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
        await pilot.pause()

        expanded = _row_text(app, "tool-1")
        assert "result row 59" in expanded, (
            f"pressing 'o' did not reveal the full result: {expanded[:160]!r}"
        )


@pytest.mark.unit
def test_a_failed_step_still_exposes_what_it_produced():
    """AC#3: 'even if it failed' was the reported ask.

    A failing call is exactly when the user wants the output -- the question
    is how far it got, not just that it stopped. An error step's summary IS
    the produced text, so it becomes the expandable body.
    """
    from tldw_chatbook.Chat.console_agent_bridge import (
        STEP_ERROR,
        STEP_TOOL_RESULT,
        full_step_output,
    )

    long_error = "Traceback:\n" + "\n".join(f"  frame {n}" for n in range(40))
    assert full_step_output(
        STEP_ERROR, summary=long_error, marker_text="⚠ Traceback:… (+400 chars)"
    ) == long_error

    failed_result = "ERROR: permission denied\n" + "x" * 500
    assert full_step_output(
        STEP_TOOL_RESULT,
        result=failed_result,
        marker_text="⚙ read_file → ERROR:… (+500 chars)",
    ) == failed_result


@pytest.mark.unit
def test_a_marker_that_already_shows_everything_carries_no_duplicate():
    """No expand control that opens an identical view (see TASK-1843)."""
    from tldw_chatbook.Chat.console_agent_bridge import (
        STEP_TOOL_RESULT,
        full_step_output,
    )

    assert (
        full_step_output(
            STEP_TOOL_RESULT, result="ok", marker_text="⚙ read_file → ok"
        )
        is None
    )


@pytest.mark.asyncio
async def test_two_calls_in_one_turn_expand_independently():
    """AC#4: expansion is per row, not a single global toggle."""
    class _TwoCalls(App):
        def compose(self) -> ComposeResult:
            transcript = ConsoleTranscript(id="console-native-transcript")
            transcript.set_messages([
                ConsoleChatMessage(
                    role=ConsoleMessageRole.TOOL, id="t-a",
                    content="⚙ read_file → alpha… (+9 chars)",
                    tool_output_full="alpha UNIQUE-A",
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.TOOL, id="t-b",
                    content="⚙ read_file → beta… (+9 chars)",
                    tool_output_full="beta UNIQUE-B",
                ),
            ])
            yield transcript

    app = _TwoCalls()
    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.toggle_tool_output("t-a")
        await pilot.pause()
        await pilot.pause()

        assert "UNIQUE-A" in _row_text(app, "t-a")
        assert "UNIQUE-B" not in _row_text(app, "t-b"), (
            "expanding one call revealed another -- the toggle is not per row"
        )


@pytest.mark.unit
def test_expanded_state_is_pruned_when_messages_leave_the_transcript():
    """Expansion is per message id, so it must not outlive the message.

    The state comment claimed it was "dropped when the transcript is rebuilt"
    -- it was not; `set_messages` pruned the signature cache and left this
    set growing for the life of the widget. A recycled id would also come
    back already expanded.
    """
    transcript = ConsoleTranscript()
    kept = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL, id="keep", content="⚙ a → x…",
        tool_output_full="xxxx",
    )
    gone = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL, id="gone", content="⚙ b → y…",
        tool_output_full="yyyy",
    )
    transcript.set_messages([kept, gone])
    transcript._expanded_tool_output_ids.update({"keep", "gone"})

    transcript.set_messages([kept])

    assert transcript._expanded_tool_output_ids == {"keep"}, (
        "expansion state outlived the message it belonged to: "
        f"{transcript._expanded_tool_output_ids}"
    )


@pytest.mark.unit
def test_toggle_ignores_unknown_and_nonexpandable_messages():
    """The shared disclosure seam must not create dead expansion state."""
    transcript = ConsoleTranscript()
    transcript.set_messages(
        [
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER,
                id="user",
                content="ordinary content",
            ),
            ConsoleChatMessage(
                role=ConsoleMessageRole.TOOL,
                id="empty-tool",
                content="",
            ),
        ]
    )

    transcript.toggle_tool_output("unknown")
    transcript.toggle_tool_output("user")
    transcript.toggle_tool_output("empty-tool")

    assert transcript._expanded_tool_output_ids == set()
