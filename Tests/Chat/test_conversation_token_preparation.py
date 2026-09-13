"""Token warmup retains inactive-session ownership and post-await fences."""

import asyncio
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import Literal

import pytest

from tldw_chatbook.Chat import console_session_settings
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Console_Modules.conversation_token_preparation import (
    prepare_conversation_tokens,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("none", "provider_owner", "payload"))
async def test_token_preparation_uses_current_owner_and_rechecks_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    change: Literal["none", "provider_owner", "payload"],
) -> None:
    """Accept an unchanged token snapshot and reject an intervening owner/edit.

    Args:
        monkeypatch: Replaces token estimation with the controlled worker seam.
        change: Mutation applied while estimation waits, or none for acceptance.
    """
    store = ConsoleChatStore()
    session = store.create_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Saved text")
    selection = ConsoleProviderSelection(provider="openai", explicit_model="test-model")
    requested = []

    def select(session_id):
        requested.append(session_id)
        return selection

    screen = SimpleNamespace(
        _provider_selection=SimpleNamespace(_build_console_provider_selection=select)
    )
    started, release = Event(), Event()
    snapshots = []

    def estimate(rows, model, provider):
        snapshots.append((rows, model, provider))
        started.set()
        assert release.wait(5), "test did not release token estimation"

    monkeypatch.setattr(console_session_settings, "_estimate_tokens_locally", estimate)
    task = asyncio.create_task(prepare_conversation_tokens(screen, store, session.id))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        if change == "provider_owner":
            screen._provider_selection = SimpleNamespace(
                _build_console_provider_selection=lambda _id: replace(
                    selection, explicit_model="changed-model"
                )
            )
        elif change == "payload":
            store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="New text"
            )
    finally:
        release.set()
        if change == "none":
            await task
        else:
            with pytest.raises(ValueError, match="Conversation changed"):
                await task

    assert requested[0] == session.id
    assert snapshots == [
        ([{"role": "user", "content": "Saved text"}], "test-model", "openai")
    ]
    if change == "none":
        assert requested == [session.id, session.id]
