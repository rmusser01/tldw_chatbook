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

pytestmark = pytest.mark.bootstrap_profile


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
    store = SimpleNamespace(
        all_messages_for_session=lambda _session_id: messages,
        sessions=lambda: (SimpleNamespace(id="session-1", pending_attachments=[]),),
        session_draft=lambda _session_id: "",
    )
    lifecycle = ConsoleLifecycleImpact(
        revision=7,
        live_run_count=0,
        queued_session_count=0,
        unsent_prompt_count=0,
    )
    runtime = SimpleNamespace(
        lifecycle_impact=lambda *, session_id: (
            lifecycle if session_id == "session-1" else None
        ),
        pending_round_kinds=lambda _session_id: frozenset(),
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._chat_store_accessor = lambda: store
    controller._composer_accessor = lambda: None
    controller._ensure_console_chat_controller_fn = lambda: runtime
    controller._console_runtime_accessor = lambda: SimpleNamespace(
        recoveries_for_session=lambda _session_id: ()
    )

    impact = controller._session_close_impact("session-1")

    assert impact is not None
    assert impact.transcript_message_count == expected_unsaved_count
    assert impact.lifecycle is lifecycle
    assert impact.unsent_turn_count == 0


def _idle_lifecycle() -> ConsoleLifecycleImpact:
    return ConsoleLifecycleImpact(
        revision=3, live_run_count=0, queued_session_count=0, unsent_prompt_count=0
    )


def test_session_close_impact_counts_parked_unsent_turns_as_loss() -> None:
    """TASK-33628.1 AC#3: a parked recovery alone is a reason to ask."""
    store = SimpleNamespace(
        all_messages_for_session=lambda _session_id: [],
        sessions=lambda: (SimpleNamespace(id="session-1", pending_attachments=[]),),
        session_draft=lambda _session_id: "",
    )
    chat_controller = SimpleNamespace(
        lifecycle_impact=lambda *, session_id: _idle_lifecycle(),
        pending_round_kinds=lambda _session_id: frozenset(),
    )
    parked = {"session-1": ("turn-a", "turn-b"), "session-2": ("turn-c",)}
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._chat_store_accessor = lambda: store
    controller._composer_accessor = lambda: None
    controller._ensure_console_chat_controller_fn = lambda: chat_controller
    controller._console_runtime_accessor = lambda: SimpleNamespace(
        recoveries_for_session=lambda session_id: parked.get(session_id, ())
    )

    impact = controller._session_close_impact("session-1")

    assert impact is not None
    assert impact.unsent_turn_count == 2
    assert impact.has_loss_risk is True


def test_close_dialog_lists_only_non_zero_rows_including_unsent_turns() -> None:
    """The close copy names parked turns and leaves every zero row out."""
    from tldw_chatbook.UI.Console_Modules.loss_inventory import close_dialog_message
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionCloseImpact

    message = close_dialog_message(
        ConsoleSessionCloseImpact(
            session_id="session-1",
            transcript_message_count=0,
            lifecycle=_idle_lifecycle(),
            unsent_turn_count=2,
        )
    )

    assert message.splitlines() == [
        "Saved history stays in Library.",
        "Closing will discard or cancel:",
        "Unsent turns needing attention: 2",
    ]
