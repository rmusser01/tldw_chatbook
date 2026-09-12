"""Focused coverage for Console session-close loss accounting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleLifecycleImpact,
    ConsoleMessageRole,
)
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController


@pytest.mark.parametrize(
    ("persisted_message_ids", "expected_unsaved_count"),
    [
        (("saved-1", "saved-2"), 0),
        ((None, None), 2),
        (("saved-1", None, "saved-2"), 1),
    ],
)
def test_session_close_impact_counts_only_unsaved_conversation_tree_nodes(
    persisted_message_ids: tuple[str | None, ...],
    expected_unsaved_count: int,
) -> None:
    messages = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER,
            content=f"message-{index}",
            persisted_message_id=persisted_message_id,
        )
        for index, persisted_message_id in enumerate(persisted_message_ids)
    ]
    store = SimpleNamespace(all_messages_for_session=lambda _session_id: messages)
    lifecycle = ConsoleLifecycleImpact(
        revision=7,
        live_run_count=0,
        queued_session_count=0,
        unsent_prompt_count=0,
    )
    runtime = SimpleNamespace(
        lifecycle_impact=lambda *, session_id: (
            lifecycle if session_id == "session-1" else None
        )
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._chat_store_accessor = lambda: store
    controller._ensure_console_chat_controller_fn = lambda: runtime

    impact = controller._session_close_impact("session-1")

    assert impact is not None
    assert impact.transcript_message_count == expected_unsaved_count
    assert impact.lifecycle is lifecycle
