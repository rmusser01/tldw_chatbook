"""ADR-220: action-time dependencies through actual controller boundaries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import pytest

pytestmark = pytest.mark.bootstrap_profile
from Tests.Chat.test_console_context_compaction import (
    _controller_preflight_fixture,
    _resolution,
)
from tldw_chatbook.Chat import console_chat_controller as ccc
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
from tldw_chatbook.Chat.console_interrupt_rounds import KIND_SETTER_ATTRS


@pytest.mark.parametrize("kind", tuple(KIND_SETTER_ATTRS))
def test_interrupt_remount_reads_replaced_sink_and_controller_state(kind):
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    session = controller.store.ensure_session()
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
    host = controller._interrupt_host
    payload = {"round_id": "live", "session_id": session.id}
    host.park_round_payload(kind, "live", payload)
    calls = []
    setattr(controller, KIND_SETTER_ATTRS[kind], calls.append)
    host.remount_head(kind, session.id)
    assert calls == [payload]
    setattr(controller, KIND_SETTER_ATTRS[kind], None)
    host.remount_head(kind, session.id)
    assert calls == [payload]
    controller._pending_round_kinds = {}
    replacement = controller._pending_round_kinds
    controller.add_pending_round(session.id, "late", kind)
    assert controller._pending_round_kinds is replacement
    assert controller.pending_round_count(session.id, kind=kind) == 1
    controller.discard_pending_round(session.id, "late")
    assert controller.pending_round_count(session.id, kind=kind) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("preview", [None, 42, "raises", "callable"])
async def test_compaction_reads_replaced_services_globals_and_accounting(
    monkeypatch, preview
):
    controller, _, session, assistant, gateway, messages = (
        _controller_preflight_fixture(ContextCompactionMode.OFF)
    )
    old_accounting = controller._context_accounting_by_session
    controller._context_accounting_by_session = {}
    accounting = controller._context_accounting_by_session
    calls = []
    original = ccc.decide_compaction

    def decide(*args, **kwargs):
        calls.append("decision")
        return original(*args, **kwargs)

    monkeypatch.setattr(ccc, "decide_compaction", decide)
    prepare = gateway.prepare_chat_request

    def prepared(*args, **kwargs):
        calls.append(("prepare", kwargs.get("tools")))
        return prepare(*args, **kwargs)

    def preview_call():
        calls.append("preview")
        if preview == "raises":
            raise ValueError("controlled optional preview failure")
        return []

    controller.provider_gateway = SimpleNamespace(prepare_chat_request=prepared)
    controller._agent_bridge = SimpleNamespace(
        preview_tool_schemas=preview_call if isinstance(preview, str) else preview
    )
    assessment = []
    result = await controller._apply_conversation_memory_preflight(
        session_id=session.id,
        resolution=_resolution(),
        provider_messages=messages,
        assistant_message_id=assistant.id,
        agent_tools_enabled=True,
        assessment_sink=lambda *args: assessment.append(args),
    )
    assert result[1] is None
    assert assessment
    assert controller._context_accounting_by_session is accounting
    assert session.id in accounting and session.id not in old_accounting
    assert calls[-1] == "decision"
    assert ("preview" in calls) is isinstance(preview, str)
    assert calls[0] == ("preview" if isinstance(preview, str) else ("prepare", []))
    for optional in (None, 42):
        controller.provider_gateway = SimpleNamespace(prepare_chat_request=optional)
        assert await controller._apply_conversation_memory_preflight(
            session_id=session.id,
            resolution=_resolution(),
            provider_messages=messages,
            assistant_message_id=assistant.id,
            agent_tools_enabled=True,
        ) == (messages, None)


@pytest.mark.asyncio
async def test_compact_now_enters_and_releases_one_maintenance_fence(monkeypatch):
    controller, _, session, _, _, _ = _controller_preflight_fixture(
        ContextCompactionMode.OFF
    )
    events = []

    class Calls(dict):
        def __setitem__(self, key, value):
            events.append(("enter", value))
            super().__setitem__(key, value)

        def pop(self, key, default=None):
            events.append(("exit", self[key]))
            return super().pop(key, default)

    controller._maintenance_calls = Calls()
    assert (await controller.compact_context_now(session.id))[0] is True
    assert events == [("enter", 1), ("exit", 1)]
    assert not controller._maintenance_calls
    controller._maintenance_paused = True
    assert (await controller.compact_context_now(session.id))[0] is False
    assert events == [("enter", 1), ("exit", 1)]
    controller._maintenance_paused = False

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(controller, "_resolve_for_send_bounded", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await controller.compact_context_now(session.id)
    assert events == [("enter", 1), ("exit", 1)] * 2
    assert not controller._maintenance_calls
