"""Task 14: controller publication and provider-entry fences after commit."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleSettingsComponent,
)
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    hydrate_console_generation_settings,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_generation_settings_metadata import (
    ConsoleGenerationSettingsReadStatus,
    ConsoleGenerationSettingsWriteResult,
    ConsoleGenerationSettingsWriteStatus,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsDraftState,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
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
    *,
    initial_settings: ConsoleSessionSettings | None = None,
) -> tuple[
    CharactersRAGDB,
    ConsoleChatStore,
    ConsoleChatController,
    _CheckpointObservingGateway,
]:
    db = CharactersRAGDB(tmp_path / "controller.sqlite", client_id="task14-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    store.create_session(
        session_id="session-1",
        title="Chat 1",
        settings=initial_settings,
        canonical_settings_baseline=initial_settings,
    )
    gateway = _CheckpointObservingGateway(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
    )
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    return db, store, controller, gateway


async def _stage_first_send_settings(
    store: ConsoleChatStore,
    *,
    submission_id: str = "first-send-settings",
    model: str = "first-send-model",
    temperature: float = 0.61,
    compaction_mode: ContextCompactionMode = ContextCompactionMode.OFF,
    expected_staged: bool = True,
) -> None:
    submission = ConsoleSettingsSubmission(
        submission_id=submission_id,
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        surface=ConsoleSettingsSurface.FULL_SETTINGS,
        origin=store.capture_console_settings_origin("session-1"),
        draft=ConsoleSettingsDraftState(
            settings=ConsoleSessionSettings(
                provider="openai",
                model=model,
                temperature=temperature,
                streaming=False,
            ),
            context_policy_overrides=ConsoleContextPolicyOverrides(
                compaction_mode=compaction_mode,
            ),
            field_drafts=(),
            model_drafts=(),
            endpoint_draft=None,
        ),
        user_display_name_override=None,
        default_field_mask=frozenset(),
    )
    commit = store.commit_console_settings_live(submission)
    outcome = await store.persist_console_settings_commit_serialized(commit)
    assert outcome.staged is expected_staged


@pytest.mark.asyncio
async def test_first_send_reconciles_interleaved_apply_and_records_exact_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = ConsoleSessionSettings(
        provider="openai",
        model="first-send-model",
        temperature=0.61,
        streaming=False,
        source="global_default",
    )
    _db, store, controller, _gateway = _controller(
        tmp_path,
        initial_settings=initial,
    )
    persistence = store.persistence
    assert isinstance(persistence, ChatPersistenceService)
    entered = Event()
    release = Event()
    original_commit = persistence.commit_durable_turn

    def blocked_commit(**kwargs: Any):
        entered.set()
        assert release.wait(timeout=5)
        return original_commit(**kwargs)

    monkeypatch.setattr(persistence, "commit_durable_turn", blocked_commit)
    submit = asyncio.create_task(
        controller.submit_draft("race the first send", session_id="session-1")
    )
    assert await asyncio.to_thread(entered.wait, 5)
    await _stage_first_send_settings(
        store,
        submission_id="newer-settings",
        model="newer-model",
        temperature=0.27,
        compaction_mode=ContextCompactionMode.AUTOMATIC,
    )
    original_generation_write = persistence.update_conversation_generation_settings
    monkeypatch.setattr(
        persistence,
        "update_conversation_generation_settings",
        lambda **_kwargs: ConsoleGenerationSettingsWriteResult(
            ConsoleGenerationSettingsWriteStatus.MISSING
        ),
    )
    release.set()

    result = await submit

    assert result.accepted is True
    session = store.sessions()[0]
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    durable_generation = persistence.get_conversation_generation_settings(
        conversation_id
    )
    durable_context = persistence.get_conversation_context_policy(conversation_id)
    assert durable_generation.snapshot is not None
    assert durable_generation.snapshot.model == "first-send-model"
    assert durable_context.overrides.compaction_mode is ContextCompactionMode.AUTOMATIC
    failure = session.settings_persistence_failures[
        ConsoleSettingsComponent.GENERATION_SETTINGS
    ]
    assert failure.revision == session.generation_settings_revision
    assert failure.generation_snapshot is not None
    assert failure.generation_snapshot.model == "newer-model"
    assert failure.persisted_conversation_id == conversation_id
    assert ConsoleSettingsComponent.CONTEXT_POLICY not in (
        session.settings_persistence_failures
    )
    assert session.staged_context_policy_failure_label is None
    assert session.staged_context_policy_failure_revision is None

    monkeypatch.setattr(
        persistence,
        "update_conversation_generation_settings",
        original_generation_write,
    )
    assert await store.retry_console_settings_persistence(
        session_id=session.id,
        component=ConsoleSettingsComponent.GENERATION_SETTINGS,
        revision=failure.revision,
    )
    retried = persistence.get_conversation_generation_settings(conversation_id)
    assert retried.snapshot is not None
    assert retried.snapshot.model == "newer-model"
    assert session.settings_persistence_failures == {}


@pytest.mark.asyncio
async def test_normal_first_send_atomically_persists_staged_settings_and_reopens(
    tmp_path: Path,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)
    await _stage_first_send_settings(store)

    result = await controller.submit_draft(
        "persist my staged settings", session_id="session-1"
    )

    assert result.accepted is True
    session = store.sessions()[0]
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    persistence = store.persistence
    assert isinstance(persistence, ChatPersistenceService)
    generation = persistence.get_conversation_generation_settings(conversation_id)
    context = persistence.get_conversation_context_policy(conversation_id)
    assert generation.status is ConsoleGenerationSettingsReadStatus.VALID
    assert generation.snapshot is not None
    assert (
        generation.snapshot.provider,
        generation.snapshot.model,
        generation.snapshot.temperature,
        generation.snapshot.streaming,
    ) == ("openai", "first-send-model", pytest.approx(0.61), False)
    assert context.overrides.compaction_mode is ContextCompactionMode.OFF
    assert context.revision == 1
    assert session.generation_durable_snapshot == generation.snapshot
    assert session.context_policy_durable_revision == context.revision
    assert session.staged_context_policy_failure_label is None
    assert session.staged_context_policy_failure_revision is None
    assert session.settings_persistence_failures == {}

    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation is not None
    hydration = hydrate_console_generation_settings({}, conversation)
    reopened_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    reopened = reopened_store.restore_persisted_session(
        title=str(conversation["title"]),
        workspace_id=conversation.get("workspace_id"),
        persisted_conversation_id=conversation_id,
        all_nodes=(),
        settings=hydration.settings,
        generation_durable_snapshot=hydration.durable_snapshot,
        generation_metadata_status=hydration.metadata_status,
    )
    assert reopened.settings is not None
    assert (
        reopened.settings.provider,
        reopened.settings.model,
        reopened.settings.temperature,
        reopened.settings.streaming,
    ) == ("openai", "first-send-model", pytest.approx(0.61), False)
    assert reopened.context_policy_overrides.compaction_mode is ContextCompactionMode.OFF


@pytest.mark.asyncio
async def test_first_send_persists_revision_zero_new_chat_default(
    tmp_path: Path,
) -> None:
    initial = ConsoleSessionSettings(
        provider="anthropic",
        model="saved-global-model",
        temperature=0.42,
        streaming=False,
        source="global_default",
    )
    _db, store, controller, _gateway = _controller(
        tmp_path,
        initial_settings=initial,
    )
    session = store.sessions()[0]
    assert session.generation_settings_revision == 0

    result = await controller.submit_draft("use my default", session_id="session-1")

    assert result.accepted is True
    persistence = store.persistence
    assert isinstance(persistence, ChatPersistenceService)
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    persisted = persistence.get_conversation_generation_settings(
        conversation_id
    )
    assert persisted.status is ConsoleGenerationSettingsReadStatus.VALID
    assert persisted.snapshot is not None
    assert (
        persisted.snapshot.provider,
        persisted.snapshot.model,
        persisted.snapshot.temperature,
        persisted.snapshot.streaming,
    ) == ("anthropic", "saved-global-model", pytest.approx(0.42), False)
    assert session.generation_durable_snapshot == persisted.snapshot


@pytest.mark.asyncio
async def test_later_send_does_not_republish_first_persist_settings_bases(
    tmp_path: Path,
) -> None:
    _db, store, controller, _gateway = _controller(tmp_path)
    await _stage_first_send_settings(store)
    first = await controller.submit_draft("first", session_id="session-1")
    assert first.accepted is True
    await _stage_first_send_settings(
        store,
        submission_id="persisted-settings",
        model="persisted-model",
        compaction_mode=ContextCompactionMode.AUTOMATIC,
        expected_staged=False,
    )
    session = store.sessions()[0]
    assert session.context_policy_durable_revision == 2

    second = await controller.submit_draft("second", session_id="session-1")

    assert second.accepted is True
    assert session.context_policy_durable_revision == 2
    persistence = store.persistence
    assert isinstance(persistence, ChatPersistenceService)
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    persisted = persistence.get_conversation_context_policy(conversation_id)
    assert persisted.revision == 2
    assert persisted.overrides.compaction_mode is ContextCompactionMode.AUTOMATIC


@pytest.mark.asyncio
async def test_first_send_context_write_failure_rolls_back_the_whole_turn(
    tmp_path: Path,
) -> None:
    db, store, controller, gateway = _controller(tmp_path)
    await _stage_first_send_settings(store)
    db.get_connection().execute(
        "CREATE TRIGGER fail_first_context_policy "
        "BEFORE INSERT ON console_conversation_context_policy "
        "BEGIN SELECT RAISE(ABORT, 'injected context failure'); END"
    )

    result = await controller.submit_draft(
        "must remain atomic", session_id="session-1"
    )

    assert result.accepted is False
    assert gateway.calls == 0
    session = store.sessions()[0]
    assert session.persisted_conversation_id is None
    assert session.settings is not None
    assert session.settings.model == "first-send-model"
    assert session.context_policy_overrides.compaction_mode is ContextCompactionMode.OFF
    for table in (
        "conversations",
        "console_conversation_library_policy",
        "console_conversation_context_policy",
        "messages",
        "console_dispatch_checkpoints",
    ):
        assert db.get_connection().execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0] == 0


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
    # TASK-22030: the refusal is right; its old shape (a bare result, no run
    # state, no row, no toast) was not. Assert the user-visible surface, not
    # just the return value.
    assert "not sent" in result.visible_copy.lower()
    assert result.should_clear_draft is False
    run_state = controller.run_state_for("session-1")
    assert run_state.status is ConsoleRunStatus.BLOCKED
    assert run_state.visible_copy == result.visible_copy
    rows = store.messages_for_session("session-1")
    assert [row.role for row in rows] == [ConsoleMessageRole.SYSTEM]
    assert rows[0].content == result.visible_copy
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
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )
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
    # TASK-22205: a permanent (not TEMP) trigger — the durable commit now
    # runs on a worker thread with its own thread-local connection, and a
    # TEMP trigger is per-connection so it would never fire there.
    db.get_connection().execute(
        "CREATE TRIGGER task14_fail_checkpoint "
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
async def test_provider_entry_failure_atomically_settles_durable_owner(
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
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
        == 0
    )
    row_counts = tuple(
        db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "conversations",
            "messages",
            "console_dispatch_checkpoints",
        )
    )

    second = await controller.resume_durable_postcommit(first.preparation_id or "")

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
