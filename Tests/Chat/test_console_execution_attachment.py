"""Selected parent visit revocation does not depend on channel success."""

import pytest


def test_selected_resize_waits_for_the_current_visit_publication(monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import execution_state

    client = SimpleNamespace(_visit=4, _readiness_deadline=10.0)
    screen = SimpleNamespace()
    monkeypatch.setattr(execution_state, "selected_client", lambda _screen: client)
    monkeypatch.setattr(execution_state, "time", SimpleNamespace(monotonic=lambda: 10.0))
    refreshes = []
    monkeypatch.setattr(execution_state, "refresh_selected_when_due", refreshes.append)
    assert execution_state.defer_selected_resize(screen, "first")
    assert execution_state.defer_selected_resize(screen, "latest")
    assert screen._console_pending_process_resize == (client, 4, "latest")
    assert refreshes == [screen, screen]
    client._readiness_deadline = 10.5
    assert not execution_state.defer_selected_resize(screen, "latest")
    assert screen._console_pending_process_resize is None


def test_detach_revokes_visit_before_missing_projection_row():
    from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
    from tldw_chatbook.Chat.console_execution_projection_view import ConsoleProjectionView

    client = ConsoleExecutionClient(None)
    old_view = object()
    client._view, client._visit = old_view, 7
    client.projection_view = ConsoleProjectionView(0, (), None, (), ())
    with pytest.raises(LookupError):
        client.detach_view(old_view, 7)
    assert client._view is None
    assert client._visit == 8
    assert client.detach_view(old_view, 7) is False
    assert not client._view_checkpoints


def test_projection_freshness_uses_the_actual_capture_in_this_request():
    from tldw_chatbook.Chat.console_execution_process import (
        ConsoleChannelLost,
        _projection_readiness_deadline,
    )

    # A long queue precedes the checked capture. Transport ages it normally.
    assert _projection_readiness_deadline(10.0, 22.0, 0.75, 22.5) == 22.75
    with pytest.raises(ConsoleChannelLost, match="expired"):
        _projection_readiness_deadline(10.0, 22.0, 0.75, 22.75)
    for capture in (9.0, 23.0):
        with pytest.raises(ConsoleChannelLost, match="outside"):
            _projection_readiness_deadline(10.0, capture, 0.75, 22.5)
