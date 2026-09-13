"""Late-binding contracts after retiring private screen forwarding methods."""

from types import SimpleNamespace

from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def test_submission_conversation_callback_resolves_replaced_session_owner():
    screen = ChatScreen(_build_test_app())
    callback = screen._submission._current_console_conversation_id
    calls = []
    for label in ("first", "replacement"):
        screen._session = SimpleNamespace(
            _current_console_conversation_id=lambda: calls.append(label) or label
        )
        assert callback() == label
        assert calls[-1] == label
    assert calls == ["first", "replacement"]


def test_submission_acceptance_is_not_owned_by_view_hooks():
    screen = ChatScreen(_build_test_app())
    hooks = screen.console_view_hooks()
    assert "on_submission_accepted" not in hooks
    assert "prompt_history" not in hooks


def test_settings_callback_observes_replaced_method_and_owner():
    screen = ChatScreen(_build_test_app())
    callback = screen._settings_navigation._console_context_control_state_for_session
    session_id = object()
    option = object()
    result = object()
    calls = []

    def replacement(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    screen._context_cost._console_context_control_state_for_session = replacement
    assert callback(session_id, option=option) is result
    assert calls == [((session_id,), {"option": option})]

    replacement_result = object()
    screen._context_cost = SimpleNamespace(
        _console_context_control_state_for_session=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or replacement_result
        )
    )
    assert callback(session_id, option=option) is replacement_result
    assert calls == [((session_id,), {"option": option})] * 2
