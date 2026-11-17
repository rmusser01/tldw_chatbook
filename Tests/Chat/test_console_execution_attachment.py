"""Selected parent visit revocation does not depend on channel success."""

import pytest


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
