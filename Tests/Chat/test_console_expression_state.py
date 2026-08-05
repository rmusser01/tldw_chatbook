import pytest
from tldw_chatbook.Chat.console_expression_state import (
    EXPRESSION_STATES,
    EXPRESSION_IMAGE_STATES,
    resolve_console_expression_state,
)
from tldw_chatbook.Chat.console_image_view import resolve_react_character_expressions


class _Msg:
    def __init__(self, role, status):
        self.role = role
        self.status = status


class _FakeRole:
    ASSISTANT = object()
    USER = object()


class _FakeStore:
    """Minimal stand-in exposing messages_for_session, matching the real signature."""
    def __init__(self, messages_by_session):
        self._m = messages_by_session

    def messages_for_session(self, session_id):
        if session_id not in self._m:
            raise KeyError(session_id)
        return list(self._m[session_id])


@pytest.fixture(autouse=True)
def _patch_role(monkeypatch):
    # Point the resolver at the fake role sentinel so _Msg.role comparisons match.
    import tldw_chatbook.Chat.console_expression_state as mod
    monkeypatch.setattr(mod, "ConsoleMessageRole", _FakeRole)


def _state(messages, *, react=True, sid="s1"):
    store = _FakeStore({sid: messages})
    return resolve_console_expression_state(store, sid, react_enabled=react)


def test_no_session_is_idle():
    store = _FakeStore({})
    assert resolve_console_expression_state(store, None, react_enabled=True) == "idle"


def test_missing_session_is_idle():
    store = _FakeStore({})
    assert resolve_console_expression_state(store, "nope", react_enabled=True) == "idle"


def test_no_assistant_message_is_idle():
    assert _state([_Msg(_FakeRole.USER, "complete")]) == "idle"


def test_pending_assistant_is_thinking():
    assert _state([_Msg(_FakeRole.USER, "complete"), _Msg(_FakeRole.ASSISTANT, "pending")]) == "thinking"


def test_streaming_assistant_is_speaking():
    assert _state([_Msg(_FakeRole.ASSISTANT, "streaming")]) == "speaking"


def test_complete_assistant_is_idle():
    assert _state([_Msg(_FakeRole.ASSISTANT, "complete")]) == "idle"


def test_stopped_assistant_is_idle():
    assert _state([_Msg(_FakeRole.ASSISTANT, "stopped")]) == "idle"


def test_failed_assistant_is_error():
    assert _state([_Msg(_FakeRole.ASSISTANT, "failed")]) == "error"


def test_last_assistant_wins():
    # A completed turn followed by a new pending turn -> thinking.
    msgs = [_Msg(_FakeRole.ASSISTANT, "complete"), _Msg(_FakeRole.ASSISTANT, "pending")]
    assert _state(msgs) == "thinking"


def test_react_disabled_pins_idle():
    assert _state([_Msg(_FakeRole.ASSISTANT, "streaming")], react=False) == "idle"


def test_constants():
    assert EXPRESSION_STATES == ("idle", "thinking", "speaking", "error")
    assert EXPRESSION_IMAGE_STATES == ("thinking", "speaking", "error")


def test_react_config_helper_defaults_true():
    assert resolve_react_character_expressions({}) is True
    cfg = {"chat": {"images": {"react_character_expressions": False}}}
    assert resolve_react_character_expressions(cfg) is False
