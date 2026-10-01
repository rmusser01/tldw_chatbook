"""Compatibility seams retained by the Console size repair."""

from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat import console_chat_controller as owner
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController


@pytest.mark.asyncio
async def test_lazy_submission_reads_the_live_controller_result_seam(monkeypatch):
    expected = object()
    calls = []

    def result(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(owner, "ConsoleSubmitResult", result)
    controller = SimpleNamespace(store=SimpleNamespace(active_session_id=None))
    actual = await ConsoleChatController._submit_draft_body(
        controller, "hello", preserve_composer=True
    )
    assert actual is expected
    assert calls == [((False, False, "Choose an explicit conversation."), {})]


def test_persistence_protocol_keeps_its_export_and_resolvable_annotations():
    from tldw_chatbook.Chat.console_chat_persistence import (
        ConsoleChatPersistence as PersistenceOwner,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatPersistence

    assert ConsoleChatPersistence is PersistenceOwner
    for method in vars(ConsoleChatPersistence).values():
        if (
            callable(method)
            and getattr(method, "__module__", None) == PersistenceOwner.__module__
        ):
            get_type_hints(method)


@pytest.mark.asyncio
@private_profile_test
async def test_fresh_persistence_contract_annotations_are_resolvable(request):
    """An earlier explicit protocol export must not mask a missing global."""
    from tldw_chatbook.Chat import console_chat_store as store_module

    hints = get_type_hints(store_module.require_thinking_persistence_support)
    from tldw_chatbook.Chat.console_chat_persistence import ConsoleChatPersistence

    assert hints["persistence"] == ConsoleChatPersistence | None
    root = Path(__file__).resolve().parents[2]
    assert Path(store_module.__file__).resolve().is_relative_to(root)
