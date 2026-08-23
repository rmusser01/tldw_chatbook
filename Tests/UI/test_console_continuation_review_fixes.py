"""Mounted Task 16 review ratchets for bounded continuation presentation."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Markdown, Static

from Tests.UI.test_console_provider_continuation_recovery import (
    _RecoveryApp,
    _message,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    ProviderContinuationRecoveryCallout,
    provider_continuation_recovery_state,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleMessageHeader,
    ConsoleTranscript,
)


class _TranscriptApp(App):
    CSS = "ConsoleTranscript { height: 24; }"

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _mounted_body(transcript: ConsoleTranscript, message_id: str) -> str:
    row = transcript.query_one(f"#console-message-{message_id}")
    if isinstance(row, ConsoleMarkdownMessage):
        return row.query_one(Markdown).source
    bodies = row.query(".console-transcript-message-body")
    assert len(bodies) == 1
    return str(bodies.first(Static).renderable)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected"),
    (
        (
            "accepted",
            "Response accepted on another device; waiting for dispatch.",
        ),
        (
            "dispatch_started",
            "Response delivery status is unknown on the source device.",
        ),
        ("complete", "No response was generated."),
        ("failed", "Response failed."),
        ("discarded", "Response discarded."),
    ),
)
async def test_hydrated_empty_states_render_shared_bounded_copy_in_mounted_rows(
    state: str,
    expected: str,
) -> None:
    """Kills projecting hydrated state while leaving the mounted body blank."""
    tree = {
        "root_threads": [
            {
                "id": f"user-{state}",
                "sender": "user",
                "role": "user",
                "content": "request",
                "children": [
                    {
                        "id": f"assistant-{state}",
                        "sender": "assistant",
                        "role": "assistant",
                        "content": "",
                        "assistant_generation_state": state,
                        "children": [],
                    }
                ],
            }
        ]
    }
    messages = console_messages_from_conversation_tree(tree)
    assert messages[-1].assistant_generation_state == state
    mounted_message_id = messages[-1].id

    app = _TranscriptApp()
    async with app.run_test(size=(90, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        await pilot.pause()

        assert _mounted_body(transcript, mounted_message_id) == expected


@pytest.mark.asyncio
async def test_live_inflight_copy_precedes_closed_state_renderer() -> None:
    """Kills replacing a healthy live activity line with imported-state copy."""
    message = ConsoleChatMessage(
        id="assistant-live",
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        status="pending",
        assistant_generation_state="accepted",
        live_activity="Calling calculator · 2s",
    )
    app = _TranscriptApp()
    async with app.run_test(size=(80, 20)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([message])
        await transcript.refresh_messages()
        await pilot.pause()

        turn = transcript.query_one("#console-assistant-turn-assistant-live")
        header = turn.query_one(".console-markdown-header", ConsoleMessageHeader)
        label = header.query_one(".console-transcript-speaker-label", Static)
        assert "Calling calculator · 2s" in label.renderable.plain
        assert "another device" not in label.renderable.plain
        assert _mounted_body(transcript, "assistant-live") == ""


@pytest.mark.asyncio
async def test_ambiguous_owner_honors_disabled_actions_in_mounted_callout() -> None:
    """Kills the ambiguous branch silently re-enabling Discard."""
    message = _message(call_state="executing")
    message.provider_continuation_actions_enabled = False
    projected = provider_continuation_recovery_state(
        message,
        replay_available=True,
    )
    assert projected is not None
    assert projected.mode == "ambiguous"
    assert projected.actions_enabled is False
    app = _RecoveryApp(message)

    async with app.run_test(size=(48, 18)) as pilot:
        await pilot.pause()
        callout = app.screen.query_one(ProviderContinuationRecoveryCallout)
        discard = callout.query_one("#console-continuation-discard", Button)
        assert discard.display
        assert discard.disabled
        discard.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.actions == []
