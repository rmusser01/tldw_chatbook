"""TASK-33620.1.12: the window between the Console's send gate and acceptance.

The Console checks the emergency stop before a Send; the Send then awaits its
preparation and its prompt hooks before it is accepted. The controller reads
the stop again in that window, so a stop that lands there refuses the turn
before any prompt hook or provider call -- with no prompt hook configured
too, the common case (the hooked case is in
Tests/Agents/test_run_hooks_emergency_stop.py).

Both awaits in the window (that stop re-read, and the hook-v2 prompt input)
run before acceptance, so a cancel landing in either one must release the
optimistic USER echo, as the legacy prompt-hook await does: an echo left
``complete`` enters the next send's provider context.

The stop is the per-test file from Tests/conftest.py's redirect.
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.Chat.test_console_chat_controller import (
    ConsoleChatController,
    ConsoleChatStore,
    RecordingStreamingGateway,
)
from tldw_chatbook import emergency_stop
from tldw_chatbook.Chat import console_automatic_submission
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

# The controller's hook admission reads the profile's hook permissions.
pytestmark = pytest.mark.bootstrap_profile

HELD = "Emergency stop is active"
CANCELLED_COPY = "Send cancelled before acceptance."


def _complete_user_texts(store, session_id):
    """USER rows a later send would put in its provider context."""
    return [
        row.content
        for row in store.messages_for_session(session_id)
        if row.role is ConsoleMessageRole.USER and row.status == "complete"
    ]


async def test_a_stop_landing_after_the_gate_refuses_a_send_with_no_prompt_hook():
    """No hook is configured, so the controller's own re-read is what refuses."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    # The send passed the Console's gate; the stop lands before acceptance.
    emergency_stop.set_emergency_stop(
        emergency_stop.default_emergency_stop_path(), reason="landed after the gate"
    )

    result = await controller.submit_draft("hello")

    assert gateway.messages_seen is None, "the prompt reached the provider"
    assert result.accepted is False and result.should_clear_draft is False
    assert result.visible_copy.startswith(HELD)
    assert _complete_user_texts(store, store.active_session_id) == []


@pytest.mark.emergency_stop_unavailable
async def test_an_unlocatable_stop_refuses_a_send_without_offering_clear(
    monkeypatch,
):
    def refuse():
        raise PermissionError("data dir refused")

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", refuse)
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=gateway
    )

    result = await controller.submit_draft("hello")

    assert gateway.messages_seen is None
    assert result.accepted is False
    assert result.visible_copy == emergency_stop.STATE_UNAVAILABLE_COPY


@pytest.mark.emergency_stop_unavailable
async def test_a_stop_read_that_fails_holds_the_send(monkeypatch):
    """The re-read runs on a worker thread; if that fails, doubt holds."""

    def no_worker():
        raise RuntimeError("cannot schedule new futures after shutdown")

    # to_thread re-raises what its worker raised.
    monkeypatch.setattr(emergency_stop, "current_emergency_stop", no_worker)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    result = await controller.submit_draft("hello")

    assert gateway.messages_seen is None
    assert result.accepted is False
    assert result.visible_copy == emergency_stop.STATE_UNAVAILABLE_COPY
    assert _complete_user_texts(store, store.active_session_id) == []


@pytest.mark.parametrize("site", ["stop_read", "hook_input"])
# A plain draft's echo belongs to its preparation, and abandoning that removes
# it; a /-prefixed draft (no Library retrieval) has no preparation, so only
# the echo's own release covers it.
@pytest.mark.parametrize("draft", ["cancel me", "/not-a-command cancel me"])
async def test_a_cancel_before_acceptance_releases_the_echo(
    monkeypatch, site, draft
):
    """A cancel in either await leaves no ``complete`` echo behind."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    reached = asyncio.Event()

    async def parked(*_args, **_kwargs):
        reached.set()
        await asyncio.Event().wait()

    if site == "stop_read":
        monkeypatch.setattr(
            console_automatic_submission, "emergency_stop_turn_refusal", parked
        )
    else:
        monkeypatch.setattr(controller, "_submission_hook_input", parked)

    send = asyncio.create_task(controller.submit_draft(draft))
    await asyncio.wait_for(reached.wait(), timeout=5)
    send.cancel()
    with pytest.raises(asyncio.CancelledError):
        await send

    session_id = store.active_session_id
    assert _complete_user_texts(store, session_id) == [], (
        "the cancelled echo stayed complete"
    )
    assert controller.run_state_for(session_id).visible_copy == CANCELLED_COPY
    assert gateway.messages_seen is None

    # The next send's provider context does not carry the cancelled text.
    monkeypatch.undo()
    result = await controller.submit_draft("next")
    assert result.accepted is True
    assert [m["content"] for m in gateway.messages_seen if m["role"] == "user"] == [
        "next"
    ]
