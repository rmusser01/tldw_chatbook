import pytest
from tldw_chatbook.Chat.console_expression_state import (
    CharacterEmoteHistoryIdentity,
    EXPRESSION_STATES,
    EXPRESSION_IMAGE_STATES,
    resolve_console_expression_selection,
    resolve_console_expression_state,
)
from tldw_chatbook.Chat.message_metadata import (
    CharacterEmoteMetadata,
    MessageMetadata,
)
from tldw_chatbook.Chat.console_image_view import resolve_react_character_expressions


class _Msg:
    def __init__(self, role, status, *, message_id="assistant", metadata=None):
        self.role = role
        self.status = status
        self.id = message_id
        self.metadata = metadata


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


def test_selection_docstring_documents_public_contract() -> None:
    docstring = resolve_console_expression_selection.__doc__ or ""

    assert "\n    Args:" in docstring
    assert "\n    Returns:" in docstring


def test_missing_session_is_idle():
    store = _FakeStore({})
    assert resolve_console_expression_state(store, "nope", react_enabled=True) == "idle"


def test_no_assistant_message_is_idle():
    assert _state([_Msg(_FakeRole.USER, "complete")]) == "idle"


def test_pending_assistant_is_thinking():
    assert (
        _state([_Msg(_FakeRole.USER, "complete"), _Msg(_FakeRole.ASSISTANT, "pending")])
        == "thinking"
    )


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


def test_streaming_explicit_state_wins_after_first_matching_live_event():
    store = _FakeStore({"s1": [_Msg(_FakeRole.ASSISTANT, "streaming")]})

    selection = resolve_console_expression_selection(
        store,
        "s1",
        react_enabled=True,
        explicit_message_id="assistant",
        explicit_state="custom-smug",
    )

    assert selection.state == "custom-smug"
    assert selection.source == "explicit"
    assert selection.message_id == "assistant"
    assert selection.history_identity is None


def test_new_pending_message_ignores_prior_message_explicit_state():
    store = _FakeStore(
        {
            "s1": [
                _Msg(_FakeRole.ASSISTANT, "complete", message_id="old"),
                _Msg(_FakeRole.ASSISTANT, "pending", message_id="new"),
            ]
        }
    )

    selection = resolve_console_expression_selection(
        store,
        "s1",
        react_enabled=True,
        explicit_message_id="old",
        explicit_state="happy",
    )

    assert selection.state == "thinking"
    assert selection.source == "operational"


def test_complete_message_returns_final_immutable_history_identity_only():
    metadata = MessageMetadata(
        character_emote=CharacterEmoteMetadata(
            sanitized_utf16_length=5,
            mood_label="smug",
            actor_kind="character",
            actor_id=7,
            pack_id=11,
            pack_version_id=13,
            expression_key="custom:smug",
            asset_id=17,
        )
    )
    store = _FakeStore(
        {"s1": [_Msg(_FakeRole.ASSISTANT, "complete", metadata=metadata)]}
    )

    selection = resolve_console_expression_selection(
        store,
        "s1",
        react_enabled=True,
        explicit_message_id="assistant",
        explicit_state="older-beat",
    )

    assert selection.state == "smug"
    assert selection.source == "historical"
    assert selection.history_identity == CharacterEmoteHistoryIdentity(
        actor_id=7,
        pack_id=11,
        pack_version_id=13,
        expression_key="custom:smug",
        expression_id=None,
        asset_id=17,
    )


def test_react_config_helper_defaults_true():
    assert resolve_react_character_expressions({}) is True
    cfg = {"chat": {"images": {"react_character_expressions": False}}}
    assert resolve_react_character_expressions(cfg) is False
