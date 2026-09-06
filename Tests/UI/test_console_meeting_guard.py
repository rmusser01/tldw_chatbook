"""Task 9: Console voice entry points refuse while a meeting is active."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Console_Modules.dictation import ConsoleDictationController
from tldw_chatbook.UI.Console_Modules.hands_free import ConsoleHandsFreeController

pytestmark = pytest.mark.unit


class _Bare:
    """Deliberately has ONLY the attributes the guard may touch: if the
    method proceeds past the guard it raises AttributeError."""

    def __init__(self, active: bool):
        self.notices: list[tuple[str, str]] = []
        self.app_instance = SimpleNamespace(
            meeting_session_owner=SimpleNamespace(is_active=active),
            notify=lambda message, severity="information": self.notices.append((message, severity)),
        )
        self._console_dictation_state = "idle"
        self._console_hands_free = None
        self._console_realtime = None


def test_dictation_start_refuses_during_meeting():
    host = _Bare(active=True)
    ConsoleDictationController._request_console_dictation_start(host)
    assert host.notices == [("Meeting in progress: stop it in Meetings before using Console dictation.", "warning")]
    assert host._console_dictation_state == "idle"


def test_dictation_start_proceeds_without_meeting():
    host = _Bare(active=False)
    with pytest.raises(AttributeError):   # past the guard: reaches real state handling
        ConsoleDictationController._request_console_dictation_start(host)
    assert host.notices == []


def test_hands_free_toggle_refuses_during_meeting():
    host = _Bare(active=True)
    ConsoleHandsFreeController.action_toggle_console_hands_free(host)
    assert host.notices == [("Meeting in progress: stop it in Meetings before using hands-free.", "warning")]


def test_hands_free_toggle_proceeds_without_meeting():
    host = _Bare(active=False)
    with pytest.raises(AttributeError):
        ConsoleHandsFreeController.action_toggle_console_hands_free(host)
    assert host.notices == []


def test_guards_tolerate_apps_without_an_owner():
    host = _Bare(active=False)
    del host.app_instance.meeting_session_owner
    with pytest.raises(AttributeError):
        ConsoleDictationController._request_console_dictation_start(host)
    assert host.notices == []
