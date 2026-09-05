"""Read-back must use the same trusted lifecycle as manual Console Speak."""

from unittest.mock import Mock

import pytest

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSMessageSpeechRequestEvent,
)
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_state", ["stopped", "failed"])
@pytest.mark.parametrize("stale_context", [False, True])
async def test_readback_settles_console_speaking_with_owned_playback(
    terminal_state, stale_context
):
    app, host = _ready_host()
    buddy = PersonaBuddyController()
    app.persona_buddy_controller = buddy
    app.post_message = Mock()
    async with host.run_test(size=(140, 42)) as pilot:
        screen = await _mounted_console(host, pilot)
        store = screen._ensure_console_chat_store()
        message = store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="The blue notebook is ready.",
        )
        await screen._console_read_last_response_back()
        events = [call.args[0] for call in app.post_message.call_args_list]
        requests = [
            event for event in events if isinstance(event, TTSMessageSpeechRequestEvent)
        ]
        assert len(requests) == 1, "Read-back bypassed trusted message speech"
        request = requests[0]
        assert request.validator(request.snapshot) == message.content
        assert request.snapshot.message_id == message.id
        assert screen._console_speaking_message_id == message.id
        assert buddy.snapshot().state == "idle"
        assert request.playback_lifecycle.report("playing")
        assert buddy.snapshot().state == "speaking"
        if stale_context:
            store.switch_session(store.create_session().id)
            sink = app.console_runtime.persona_buddy_sink
            sink.voice_state("other-session", 42, "listening")
            assert not request.playback_lifecycle.is_current()
            assert request.playback_lifecycle.report_terminal(terminal_state)
            assert buddy.snapshot().state == "listening"
            sink.release_voice("other-session", 42)
        else:
            assert request.playback_lifecycle.report(terminal_state)
            assert screen._console_speaking_message_id is None
            assert screen._message._console_speech_states[message.id] == terminal_state
        assert buddy.snapshot().state == "idle"
