"""Shared fixtures for Tests/Chat (parallel-agents spec).

``controller_with_two_sessions`` originated in Task 1's
``test_console_run_state_per_session.py`` and is reused by Task 7's
``test_console_run_markers.py`` -- moved here per the Task 7 brief's note
so both files (and any future Chat test module) can consume it without
duplicating the fixture or importing across test modules.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class StreamingGateway:
    """Minimal provider gateway stub -- copied from test_console_chat_controller.py's
    idiom (no network I/O, `ready=True` resolution) since fixture consumers
    typically drive run-state bookkeeping directly rather than actually
    running a send/stream."""

    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
            },
        )()

    async def stream_chat(self, resolution, messages):
        for chunk in ("hel", "lo"):
            yield chunk


@pytest.fixture
def controller_with_two_sessions():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    # `store.new_session` does not exist (verified by grep) -- the real
    # session-creation surface is `store.ensure_session`/`store.create_session`
    # and `controller.new_session`. `controller.new_session()` also activates
    # the session it creates (`ConsoleChatStore.create_session` sets
    # `active_session_id`), matching how `test_controller_creates_and_
    # switches_sessions` in test_console_chat_controller.py builds two
    # sessions. Net effect: session_b (not session_a) is the ACTIVE session
    # once this fixture returns.
    session_a = store.ensure_session(title="Session A")
    session_b = controller.new_session(title="Session B")
    return controller, session_a.id, session_b.id
