"""First-use token preparation before a saved Console session is revealed."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...Chat.console_chat_store import ConsoleChatStore
    from ..Screens.chat_screen import ChatScreen


async def prepare_conversation_tokens(
    screen: ChatScreen,
    store: ConsoleChatStore,
    session_id: str,
) -> None:
    """Warm the real thread-safe estimator cache without switching sessions.

    Args:
        screen: Console owning the target's provider selection.
        store: Store retaining the inactive target runtime.
        session_id: Exact session about to be activated.

    Raises:
        ValueError: Target, settings, provider selection, or payload changed
            while the immutable token snapshot was prepared off-loop.
    """
    from ...Chat.console_session_settings import _estimate_tokens_locally

    session = next(row for row in store.sessions() if row.id == session_id)
    conversation_id = session.persisted_conversation_id
    settings_revision = store.session_settings_revision(session_id)
    payload_revision = store.payload_revision(session_id)
    selection = screen._build_console_provider_selection(session_id)
    rows = tuple(
        (message.role.value, message.content)
        for message in store.messages_for_session(session_id)
    )
    await asyncio.to_thread(
        _estimate_tokens_locally,
        [{"role": role, "content": content} for role, content in rows],
        selection.explicit_model or selection.configured_model or "",
        selection.provider,
    )
    if (
        not any(row is session for row in store.sessions())
        or session.persisted_conversation_id != conversation_id
        or store.session_settings_revision(session_id) != settings_revision
        or store.payload_revision(session_id) != payload_revision
        or screen._build_console_provider_selection(session_id) != selection
    ):
        raise ValueError("Conversation changed during token preparation")
