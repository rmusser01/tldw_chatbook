"""Established chat ownership must not trigger unused registry reads."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("resume_pending", [False, True])
def test_established_store_does_not_read_unused_workspace_context(
    monkeypatch, resume_pending
):
    screen = ChatScreen.__new__(ChatScreen)
    store = SimpleNamespace(active_session_id="established-session")
    screen._console_runtime_ref = SimpleNamespace(
        chat_store=store, canvas_controller=None
    )

    def unused():
        raise AssertionError(
            "Established sessions own their context; registry read is unused"
        )

    screen._workspace = SimpleNamespace(_current_console_workspace_context=unused)
    monkeypatch.setattr(
        ChatScreen, "_console_ordered_resume_pending", lambda self: resume_pending
    )
    assert screen._ensure_console_chat_store() is store


def test_empty_store_still_aligns_with_current_workspace(monkeypatch):
    screen = ChatScreen.__new__(ChatScreen)
    landed = []
    context = object()
    store = SimpleNamespace(active_session_id=None, set_workspace_context=landed.append)
    screen._console_runtime_ref = SimpleNamespace(
        chat_store=store, canvas_controller=None
    )
    screen._workspace = SimpleNamespace(
        _current_console_workspace_context=lambda: context
    )
    monkeypatch.setattr(
        ChatScreen, "_console_ordered_resume_pending", lambda self: False
    )
    assert screen._ensure_console_chat_store() is store
    assert landed == [context]
