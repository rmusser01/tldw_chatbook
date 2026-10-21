"""Provider-bound Console evidence keeps workspace grounding available."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    ConsoleChatStore,
    _PolicyCoordinator,
    _RagService,
    _StreamingFence,
    _row,
)
from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _restored_store,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve

pytestmark = pytest.mark.bootstrap_profile


def _assert_mixed_grounding(messages) -> None:
    context = "\n".join(row["content"] for row in messages)
    assert "workspace file evidence" in context
    assert "[S#] labels only for the supplied Library references" in context
    assert "needle body" in context
    assert (
        "Use only available evidence references below to ground factual claims."
        not in context
    )


@pytest.mark.asyncio
async def test_automatic_evidence_provider_payload_allows_workspace_grounding():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    gateway = _StreamingFence()
    service = _RagService(
        {"runtime_backend": "local", "results": [_row("needle body")]}
    )
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    result = await controller.submit_draft("Explain this project")

    assert result.accepted
    assert gateway.provider_calls == 1
    _assert_mixed_grounding(gateway.messages)


@pytest.mark.asyncio
async def test_dispatch_retry_provider_payload_allows_workspace_grounding(tmp_path):
    db, conversation_id, repository = _database(tmp_path / "mixed-retry.sqlite")
    try:
        db.get_connection().execute(
            "UPDATE console_conversation_library_policy SET auto_retrieve_on_send = 1 WHERE conversation_id = ?",
            (conversation_id,),
        )
        db.get_connection().commit()
        initial_store, initial_session_id = _restored_store(db, conversation_id)
        await initial_store.hydrate_session_library_policy(initial_session_id)
        initial_controller = ConsoleChatController(
            store=initial_store,
            provider_gateway=_StreamingFence(),
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            agent_runtime_enabled=False,
        )
        configuration = initial_controller.resolve_turn_configuration_snapshot(
            initial_session_id
        )
        authority = await initial_controller._capture_turn_library_authority(
            initial_session_id, configuration
        )
        assert authority.policy.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
        acceptance = replace(
            _acceptance(conversation_id),
            frozen_authority=authority,
            attempt_id=authority.attempt_id,
        )
        _insert(db, repository, acceptance)
        store, session_id = _restored_store(db, conversation_id)
        await store.hydrate_session_library_policy(session_id)
        gateway = _StreamingFence()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            agent_runtime_enabled=False,
        )
        controller.app = SimpleNamespace(
            library_rag_search_service=_RagService(
                {"runtime_backend": "local", "results": [_row("needle body")]},
            )
        )

        result = await controller.retry_dispatch_recovery(session_id)

        assert result.accepted, result.visible_copy
        assert gateway.provider_calls == 1
        _assert_mixed_grounding(gateway.messages)
    finally:
        db.close()
