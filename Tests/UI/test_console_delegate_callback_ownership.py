"""Late-binding contracts after retiring private screen forwarding methods."""

from types import SimpleNamespace

from Tests.UI.console_controller_stubs import (
    stub_fleet_controller,
    stub_library_activity_controller,
)
from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def test_submission_hook_resolves_owner_only_when_invoked():
    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace()
    stub_fleet_controller(screen)
    stub_library_activity_controller(screen)
    screen._console_chat_store = None

    hooks = screen.console_view_hooks()
    assert "_submission" not in screen.__dict__

    calls = []
    for label in ("first", "replacement"):
        screen._submission = SimpleNamespace(
            _on_console_submission_accepted=lambda: calls.append(label)
        )
        assert hooks["on_submission_accepted"]() is None
        assert calls[-1] == label
    assert calls == ["first", "replacement"]


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
