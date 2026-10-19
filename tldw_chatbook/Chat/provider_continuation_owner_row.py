"""Create a continuation's owner row exactly once (TASK-34100.3 review G3V2-02).

``ConsoleChatStore.persist_provider_continuation_event`` creates the assistant
row that owns a provider continuation when the reply has no durable row yet.
The stream's deferred flush inserts that same pending row under the store's
``_pending_row_insert_lock`` and skips it when the row already exists. The
continuation path used to check "not yet persisted" without that lock and then
create the row, so a flush landing in between stored the reply twice, as two
sibling rows (Continue in a saved Kimi chat, then "Assistant (1/2)" after a
relaunch). The create now runs under the same lock and re-checks there; when
the flush won, the caller updates that row instead of forking a sibling.

``console_chat_store.py`` is pinned at its module-size budget, so the create
lives here; the store imports it lazily, only on this path.
"""

from __future__ import annotations

from typing import Any, Callable


def create_continuation_owner_row(
    store: Any,
    message: Any,
    creator: Callable[..., object],
    conversation_id: str,
    content: str,
    provider_continuation_json: str,
) -> int | None:
    """Insert the owner row with its continuation unless a flush already did.

    Args:
        store: The ``ConsoleChatStore`` that owns ``message``.
        message: The live assistant message that owns the continuation.
        creator: The database's ``create_assistant_with_continuation``.
        conversation_id: The saved conversation the row belongs to.
        content: The reply text stored with the continuation.
        provider_continuation_json: The private continuation payload.

    Returns:
        1, the new row's version, when this call created the row; None when the
        stream's deferred flush created it first and the caller must update it.
    """
    parent_message_id = store._previous_persisted_message_id(message)
    with store._pending_row_insert_lock:
        if message.persisted_message_id is not None:
            return None
        creator(
            message_id=message.id,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            content=content,
            provider_continuation_json=provider_continuation_json,
        )
        message.persisted_message_id = message.id
        message.parent_message_id = parent_message_id
        store._pending_persistence_message_ids.discard(message.id)
        return 1
