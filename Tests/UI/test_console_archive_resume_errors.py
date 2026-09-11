"""Recovery read failures retain the original conversation for a safe retry."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.UI.Console_Modules.archive import request_conversation_resume
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)


@pytest.mark.asyncio
async def test_workspace_read_failure_notifies_without_navigation_and_allows_retry():
    workspace_read = Mock(side_effect=RuntimeError("storage unavailable"))
    app = SimpleNamespace(
        local_chat_conversation_service=SimpleNamespace(
            get_conversation_metadata=lambda _: {
                "id": "original",
                "workspace_id": "workspace",
                "archived": False,
            },
        ),
        workspace_registry_service=SimpleNamespace(get_workspace=workspace_read),
        pending_handoffs=PendingHandoffStore(),
        notify=Mock(),
        push_screen=Mock(),
        post_message=Mock(),
    )

    await request_conversation_resume(app, "original")

    app.notify.assert_called_once_with(
        "Could not read this workspace. Refresh Library and try Resume again.",
        severity="error",
    )
    app.push_screen.assert_not_called()
    app.post_message.assert_not_called()
    assert (
        app.pending_handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME) is None
    )

    workspace_read.side_effect = None
    workspace_read.return_value = SimpleNamespace(archived=False)
    await request_conversation_resume(app, "original")

    app.post_message.assert_called_once()
    claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_CONVERSATION_RESUME)
    assert claim.value.conversation_id == "original"
