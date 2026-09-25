"""TASK-32954 Task 4: ``ConsoleChatController._character_wiring``."""

from __future__ import annotations

import json

from Tests.Chat.test_console_skill_script_confirm import _FakeApp
from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Tools.character_tool_service import SERVER_REFUSAL


def _make_controller():
    store = persisted_console_store()
    controller = ConsoleChatController(store=store, provider_gateway=object())
    controller.app = _FakeApp()
    return controller, store


def test_no_session_id_returns_empty():
    controller, _ = _make_controller()
    assert controller._character_wiring(None) == {}


def test_no_app_returns_empty():
    controller, store = _make_controller()
    session = store.create_session(runtime_backend="local")
    controller.app = None
    assert controller._character_wiring(session.id) == {}


def test_local_session_gets_a_service_sharing_one_guard_across_calls():
    controller, store = _make_controller()
    session = store.create_session(runtime_backend="local")
    first = controller._character_wiring(session.id)
    assert "character_service" in first
    second = controller._character_wiring(session.id)
    # Same per-SESSION guard across turns (never a new one per turn).
    assert first["character_service"]._guard is second["character_service"]._guard


def test_server_backend_session_refuses_via_the_service():
    controller, store = _make_controller()
    session = store.create_session(runtime_backend="server")
    wiring = controller._character_wiring(session.id)
    service = wiring["character_service"]
    result = json.loads(service.search({}))
    assert result["status"] == "unsupported"
    assert result["message"] == SERVER_REFUSAL
