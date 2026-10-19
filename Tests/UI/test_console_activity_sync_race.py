"""TASK-34100.3 review R2-2: an activity mount that outlives its turn row.

A Kimi pilot panicked with ``WorkerFailed(IndexError('list index out of
range'))`` inside ``ConsoleTranscript._sync_activity_widgets``: right after
``await activity_stack.mount(disclosure)`` the stack was empty (the
traceback's locals show it), because the assistant-turn row had been pruned
while the mount awaited: the app was tearing the screen down. The sync then
indexed ``children[index]`` of the emptied stack. A pruned row has nothing
left to reconcile, so the sync must stop instead of crashing its worker.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_assistant_turn import StyledTranscriptHarness
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_assistant_turn import (
    ConsoleAssistantTurnWidget,
)


def _tool(call_id: str) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content=f"⚙ fs_read {call_id}",
        id=f"tool-{call_id}",
        activity_presentation=ConsoleActivityPresentation(
            "tool", "fs_read", "running", call_id=call_id
        ),
    )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_activity_sync_stops_when_its_turn_is_pruned_mid_mount() -> None:
    app = StyledTranscriptHarness()
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="", id="race-answer"
    )
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.transcript
        transcript.set_messages([assistant, _tool("one")], session_id="race")
        await transcript.refresh_messages()
        await pilot.pause()
        turn = transcript.query_one(ConsoleAssistantTurnWidget)
        stack = turn.activity_stack
        assert len(stack.children) == 1
        original_mount = stack.mount
        pruned: list[bool] = []

        async def mount_then_tear_down(*widgets, **kwargs):
            await original_mount(*widgets, **kwargs)
            # What a screen teardown does to the transcript (and so to every
            # row in it) while this mount awaits. Rows themselves are only
            # removed under the transcript's refresh lock, which this sync holds.
            await transcript.remove()
            pruned.append(True)

        stack.mount = mount_then_tear_down

        transcript.set_messages(
            [assistant, _tool("one"), _tool("two")], session_id="race"
        )
        await transcript.refresh_messages()

        assert pruned == [True]
        assert not turn.is_attached and not transcript.is_attached
