"""Task 14: controller publication and provider-entry fences after commit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.Chat.test_console_durable_turn_acceptance import _ready_store


_POSTCOMMIT_EFFECTS = (
    "identity_publication",
    "durable_owner_publication",
    "staged_input_clearing",
    "workspace_projection",
    "queue_acknowledgement",
    "accepted_hook",
    "prompt_history",
    "preparation_publication",
    "checkpoint_transition",
    "provider_entry",
)


class _CheckpointObservingGateway:
    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db
        self.calls = 0
        self.states_seen: list[str] = []

    async def resolve_for_send(self, _selection: object) -> object:
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
                "resolved_destination": ConsoleResolvedDestination(
                    provider="llama_cpp",
                    model="test-model",
                    endpoint_identity="http://127.0.0.1:9099",
                    egress_class=ConsoleEgressClass.ON_DEVICE,
                ),
            },
        )()

    async def stream_chat(
        self, _resolution: object, _messages: list[dict[str, Any]], **_kwargs: Any
    ):
        self.calls += 1
        assert self.db.get_connection().in_transaction is False
        row = (
            self.db.get_connection()
            .execute(
                "SELECT state FROM console_dispatch_checkpoints "
                "ORDER BY created_at DESC LIMIT 1"
            )
            .fetchone()
        )
        self.states_seen.append(row["state"] if row is not None else "missing")
        yield "done"


def _controller(
    tmp_path: Path,
) -> tuple[
    CharactersRAGDB,
    ConsoleChatStore,
    ConsoleChatController,
    _CheckpointObservingGateway,
]:
    db = CharactersRAGDB(tmp_path / "controller.sqlite", client_id="task14-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    store.create_session(session_id="session-1", title="Chat 1")
    gateway = _CheckpointObservingGateway(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
    )
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    return db, store, controller, gateway


@pytest.mark.asyncio
async def test_real_durable_adapter_without_atomic_method_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    persistence = store.persistence
    assert persistence is not None
    monkeypatch.setattr(persistence, "commit_durable_turn", None)

    result = await controller.submit_draft(
        "must not use the legacy path", session_id="session-1"
    )

    assert result.accepted is False
    assert "durable turn acceptance is unavailable" in result.visible_copy.lower()
    assert gateway.calls == 0
    assert store.sessions()[0].persisted_conversation_id is None
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    )


@pytest.mark.asyncio
async def test_first_durable_send_commits_owner_then_cas_before_provider_entry(
    tmp_path: Path,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    accepted_hooks = 0

    def accepted() -> None:
        nonlocal accepted_hooks
        accepted_hooks += 1

    controller.on_submission_accepted = accepted

    result = await controller.submit_draft(
        "first durable prompt", session_id="session-1"
    )

    assert result.accepted is True
    assert gateway.calls == 1
    assert gateway.states_seen == [
        ConsoleDispatchCheckpointState.DISPATCH_STARTED.value
    ]
    session = store.sessions()[0]
    assert session.persisted_conversation_id is not None
    rows = db.get_messages_for_conversation(session.persisted_conversation_id, limit=20)
    assert [(row["sender"], row["content"]) for row in rows] == [
        ("user", "first durable prompt"),
        ("assistant", "done"),
    ]
    checkpoint = (
        db.get_connection()
        .execute(
            "SELECT state, user_message_id, assistant_message_id "
            "FROM console_dispatch_checkpoints"
        )
        .fetchone()
    )
    assert checkpoint is not None
    assert checkpoint["state"] == "dispatch_started"
    assert checkpoint["user_message_id"] == result.user_message_id
    assert checkpoint["assistant_message_id"] == result.assistant_message_id
    assert accepted_hooks == 1
    assert controller.prompt_history.size == 1
    assert store.durable_content_retention_count() == 0
    assert store.durable_tombstone_count() == 1


@pytest.mark.asyncio
async def test_precommit_failure_keeps_input_and_never_calls_provider(
    tmp_path: Path,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    session = store.sessions()[0]
    session.draft = "first durable prompt"
    db.get_connection().execute(
        "CREATE TEMP TRIGGER task14_fail_checkpoint "
        "BEFORE INSERT ON console_dispatch_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'task14 injected failure'); END"
    )

    result = await controller.submit_draft(
        "first durable prompt", session_id="session-1"
    )

    assert result.accepted is False
    assert result.should_clear_draft is False
    assert "couldn't save" in result.visible_copy.lower()
    assert gateway.calls == 0
    assert session.persisted_conversation_id is None
    assert session.title == "Chat 1"
    assert session.draft == "first durable prompt"
    preparation = store.preparation_for_session(session.id)
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.PAUSED
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        == 0
    )
    assert (
        db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("effect_name", _POSTCOMMIT_EFFECTS)
async def test_postcommit_effect_failure_is_reentered_once_by_preparation_id(
    tmp_path: Path,
    effect_name: str,
) -> None:
    _db, _service, store, _preparation, acceptance = _ready_store(tmp_path)
    store.commit_durable_turn(acceptance)
    fingerprint = store.durable_acceptance_fingerprint_for("preparation-1")
    assert fingerprint is not None
    controller = ConsoleChatController(store=store, provider_gateway=object())
    calls = 0

    async def flaky_effect() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("task14 injected postcommit failure")

    with pytest.raises(RuntimeError, match="injected postcommit"):
        await controller._run_durable_postcommit_effect(
            "preparation-1", effect_name, flaky_effect, fingerprint=fingerprint
        )
    failed = store.durable_postcommit_effects_for(
        "preparation-1", fingerprint=fingerprint
    )
    assert failed is not None
    assert effect_name not in failed.completed

    await controller._run_durable_postcommit_effect(
        "preparation-1", effect_name, flaky_effect, fingerprint=fingerprint
    )
    await controller._run_durable_postcommit_effect(
        "preparation-1", effect_name, flaky_effect, fingerprint=fingerprint
    )

    completed = store.durable_postcommit_effects_for(
        "preparation-1", fingerprint=fingerprint
    )
    assert completed is not None
    assert effect_name in completed.completed
    assert calls == 2


@pytest.mark.asyncio
async def test_provider_entry_failure_keeps_same_dispatch_started_durable_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    original_stream = gateway.stream_chat
    attempts = 0

    async def fail_once(*args: Any, **kwargs: Any):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("task14 injected provider entry failure")
        async for chunk in original_stream(*args, **kwargs):
            yield chunk

    monkeypatch.setattr(gateway, "stream_chat", fail_once)

    first = await controller.submit_draft(
        "first durable prompt", session_id="session-1"
    )

    assert first.accepted is True
    assert first.provider_started is True
    checkpoint = (
        db.get_connection()
        .execute(
            "SELECT preparation_id, assistant_message_id, state "
            "FROM console_dispatch_checkpoints"
        )
        .fetchone()
    )
    assert checkpoint is not None
    assert checkpoint["state"] == "dispatch_started"
    row_counts = tuple(
        db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "conversations",
            "messages",
            "console_dispatch_checkpoints",
        )
    )

    second = await controller.resume_durable_postcommit(checkpoint["preparation_id"])

    assert second.accepted is False
    assert "unavailable" in second.visible_copy.lower()
    assert attempts == 1
    assert (
        tuple(
            db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "conversations",
                "messages",
                "console_dispatch_checkpoints",
            )
        )
        == row_counts
    )
    assert controller.run_state_for("session-1").status is ConsoleRunStatus.FAILED
