"""Task 14: controller publication and provider-entry fences after commit."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatSession,
    ConsoleChatStore,
    ConsoleSettingsComponent,
    ConsoleSettingsPersistenceOutcome,
)
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    hydrate_console_generation_settings,
    hydrate_console_session,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    encode_project_context_json,
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
from Tests.private_profile import private_profile_test


# Real controller sends use the configuration admitted at collection time.
pytestmark = pytest.mark.bootstrap_profile


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
) -> ConsoleSettingsPersistenceOutcome:
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
    return outcome


async def _apply_full_settings_display_name(
    store: ConsoleChatStore,
    *,
    submission_id: str,
    display_name: str,
) -> ConsoleChatSession:
    submission = ConsoleSettingsSubmission(
        submission_id=submission_id,
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        surface=ConsoleSettingsSurface.FULL_SETTINGS,
        origin=store.capture_console_settings_origin("session-1"),
        draft=ConsoleSettingsDraftState(
            settings=ConsoleSessionSettings(
                provider="openai",
                model="display-name-model",
                streaming=False,
            ),
            context_policy_overrides=ConsoleContextPolicyOverrides(),
            field_drafts=(),
            model_drafts=(),
            endpoint_draft=None,
        ),
        user_display_name_override=display_name,
        default_field_mask=frozenset(),
    )
    commit = store.commit_console_settings_live(submission)
    outcome = await store.persist_console_settings_commit_serialized(commit)
    assert outcome.staged is (commit.persisted_conversation_id is None)
    session, roleplay_plan = (
        store.prepare_session_user_display_name_override_for_commit(
            commit,
            submission.user_display_name_override,
            global_default="User",
        )
    )
    assert session is not None
    assert roleplay_plan is not None
    roleplay_result = await store.persist_roleplay_projection_plan_serialized(
        roleplay_plan
    )
    assert roleplay_result is not None
    assert store.accept_roleplay_projection_persistence_result(roleplay_result)
    return session


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
    assert (
        reopened.context_policy_overrides.compaction_mode is ContextCompactionMode.OFF
    )


@pytest.mark.asyncio
async def test_identity_publication_retry_preserves_newer_settings_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)
    await _stage_first_send_settings(store)
    persistence = store.persistence
    assert isinstance(persistence, ChatPersistenceService)
    original_publish = store.publish_durable_turn_identity
    publication_attempts = 0

    def publish_then_fail_once(*args: Any, **kwargs: Any) -> None:
        nonlocal publication_attempts
        publication_attempts += 1
        original_publish(*args, **kwargs)
        if publication_attempts == 1:
            raise RuntimeError("identity callback failed after publication")

    monkeypatch.setattr(
        store,
        "publish_durable_turn_identity",
        publish_then_fail_once,
    )
    first = await controller.submit_draft(
        "retain the first accepted turn",
        session_id="session-1",
    )
    assert first.accepted is True
    assert first.provider_started is False
    assert first.preparation_id is not None

    original_generation_write = persistence.update_conversation_generation_settings
    monkeypatch.setattr(
        persistence,
        "update_conversation_generation_settings",
        lambda **_kwargs: ConsoleGenerationSettingsWriteResult(
            ConsoleGenerationSettingsWriteStatus.MISSING
        ),
    )
    newer = await _stage_first_send_settings(
        store,
        submission_id="intervening-settings",
        model="intervening-model",
        temperature=0.27,
        compaction_mode=ContextCompactionMode.AUTOMATIC,
        expected_staged=False,
    )
    assert newer.failed_components == frozenset(
        {ConsoleSettingsComponent.GENERATION_SETTINGS}
    )
    assert newer.written_components == frozenset(
        {ConsoleSettingsComponent.CONTEXT_POLICY}
    )
    session = store.sessions()[0]
    newer_failure = session.settings_persistence_failures[
        ConsoleSettingsComponent.GENERATION_SETTINGS
    ]
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    context_before_retry = persistence.get_conversation_context_policy(conversation_id)

    resumed = await controller.resume_durable_postcommit(first.preparation_id)

    assert resumed.accepted is True
    assert publication_attempts == 2
    assert (
        session.settings_persistence_failures[
            ConsoleSettingsComponent.GENERATION_SETTINGS
        ]
        == newer_failure
    )
    assert ConsoleSettingsComponent.CONTEXT_POLICY not in (
        session.settings_persistence_failures
    )
    durable_context = persistence.get_conversation_context_policy(conversation_id)
    assert durable_context.revision == context_before_retry.revision
    assert durable_context.overrides.compaction_mode is ContextCompactionMode.AUTOMATIC

    monkeypatch.setattr(
        persistence,
        "update_conversation_generation_settings",
        original_generation_write,
    )
    assert await store.retry_console_settings_persistence(
        session_id=session.id,
        component=ConsoleSettingsComponent.GENERATION_SETTINGS,
        revision=newer_failure.revision,
    )
    final = await _stage_first_send_settings(
        store,
        submission_id="subsequent-settings",
        model="subsequent-model",
        temperature=0.11,
        compaction_mode=ContextCompactionMode.OFF,
        expected_staged=False,
    )

    assert final.written_components == frozenset(ConsoleSettingsComponent)
    assert final.failed_components == frozenset()
    assert session.settings_persistence_failures == {}
    durable_generation = persistence.get_conversation_generation_settings(
        conversation_id
    )
    durable_context = persistence.get_conversation_context_policy(conversation_id)
    assert durable_generation.snapshot is not None
    assert durable_generation.snapshot.model == "subsequent-model"
    assert durable_context.overrides.compaction_mode is ContextCompactionMode.OFF


@pytest.mark.asyncio
async def test_first_send_atomically_persists_unsaved_display_name_and_reopens(
    tmp_path: Path,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)
    session = await _apply_full_settings_display_name(
        store,
        submission_id="display-name-settings",
        display_name="Alice",
    )

    sent = await controller.submit_draft(
        "remember my display name",
        session_id="session-1",
    )

    assert sent.accepted is True
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation is not None
    metadata = json.loads(conversation["metadata"])
    assert metadata["console_roleplay_context"] == {
        "version": 2,
        "user_name_override": "Alice",
    }
    assert "api_key" not in metadata
    assert "base_url" not in metadata
    assert "endpoint" not in metadata
    hydration = hydrate_console_generation_settings({}, conversation)
    reopened_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    reopened = await hydrate_console_session(
        app=SimpleNamespace(chachanotes_db=db),
        store=reopened_store,
        conversation_id=conversation_id,
        tree={"conversation": conversation, "root_threads": []},
        settings=hydration.settings,
        generation_durable_snapshot=hydration.durable_snapshot,
        generation_metadata_status=hydration.metadata_status,
    )

    assert reopened.user_display_name_override == "Alice"


@pytest.mark.asyncio
async def test_display_name_applied_during_first_commit_has_retryable_postcommit_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)
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
        controller.submit_draft(
            "race my display name",
            session_id="session-1",
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    session = await _apply_full_settings_display_name(
        store,
        submission_id="racing-display-name",
        display_name="Bob",
    )
    assert session.persisted_conversation_id is None
    original_roleplay_write = persistence.update_conversation_roleplay_context
    monkeypatch.setattr(
        persistence,
        "update_conversation_roleplay_context",
        lambda **_kwargs: False,
    )
    release.set()

    first = await submit

    assert first.accepted is True
    assert first.provider_started is False
    assert "retained for recovery" in first.visible_copy.lower()
    assert first.preparation_id is not None
    monkeypatch.setattr(
        persistence,
        "update_conversation_roleplay_context",
        original_roleplay_write,
    )
    resumed = await controller.resume_durable_postcommit(first.preparation_id)
    assert resumed.accepted is True
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation is not None
    hydration = hydrate_console_generation_settings({}, conversation)
    reopened = await hydrate_console_session(
        app=SimpleNamespace(chachanotes_db=db),
        store=ConsoleChatStore(persistence=ChatPersistenceService(db)),
        conversation_id=conversation_id,
        tree={"conversation": conversation, "root_threads": []},
        settings=hydration.settings,
        generation_durable_snapshot=hydration.durable_snapshot,
        generation_metadata_status=hydration.metadata_status,
    )

    assert reopened.user_display_name_override == "Bob"


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
    persisted = persistence.get_conversation_generation_settings(conversation_id)
    assert persisted.status is ConsoleGenerationSettingsReadStatus.VALID
    assert persisted.snapshot is not None
    assert (
        persisted.snapshot.provider,
        persisted.snapshot.model,
        persisted.snapshot.temperature,
        persisted.snapshot.streaming,
    ) == ("anthropic", "saved-global-model", pytest.approx(0.42), False)
    assert session.generation_durable_snapshot == persisted.snapshot


@private_profile_test
@pytest.mark.asyncio
async def test_first_send_persists_a_project_folder_chosen_before_the_chat_was_saved(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """TASK-33621.13 review: a folder chosen for project instructions in a new,
    unsaved chat lived only in memory -- the write is skipped while there is
    no conversation id -- and the first send's durable commit never wrote it
    either, so the reopened chat showed 'Off'. It must survive a restart.

    A private profile: the send reads model capabilities from config, which
    trips ``RecoveryRequired`` under the per-test sandbox otherwise."""
    db, store, controller, _gateway = _controller(tmp_path)
    chosen = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-7",
        working_folder_locator_fingerprint="f" * 64,
        project_instruction_notice_key="notice-key",
    )
    store.set_session_project_instruction_state("session-1", chosen)
    session = store.sessions()[0]
    assert session.persisted_conversation_id is None

    result = await controller.submit_draft("keep my folder", session_id="session-1")

    assert result.accepted is True
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    assert db.get_conversation_console_project_context(conversation_id) == (
        encode_project_context_json(chosen)
    )
    conversation = db.get_conversation_by_id(conversation_id)
    reopened = ConsoleChatStore(
        persistence=ChatPersistenceService(db)
    ).restore_persisted_session(
        title=str(conversation["title"]),
        workspace_id=conversation.get("workspace_id"),
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )
    assert reopened.project_instruction_state == chosen


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop_after_commit", ["trace_provenance_pause", "identity_publication_failure"]
)
async def test_project_folder_survives_a_first_send_that_stops_after_its_commit(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_after_commit: str,
) -> None:
    """TASK-33621.13 live re-verification: the first send in a new named
    workspace committed the chat and then stopped as 'Blocked' ~5 ms later,
    before identity publication ever ran. The folder was written only at
    publication, so the saved chat's column stayed NULL and after a restart
    the chat read 'Off · Project'. Both ways a send can stop after its
    durable commit -- a paused turn that never reaches publication, and a
    publication that fails -- must leave the folder saved with that chat, as
    must a folder chosen again while the turn is still stopped."""
    db, store, controller, gateway = _controller(tmp_path)
    if stop_after_commit == "trace_provenance_pause":

        def refuse_trace_request(**_kwargs: Any) -> None:
            raise RuntimeError("trace provenance unavailable")

        monkeypatch.setattr(
            controller, "_build_durable_trace_request", refuse_trace_request
        )
    else:

        def fail_publication(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("identity publication failed")

        monkeypatch.setattr(store, "publish_durable_turn_identity", fail_publication)
    chosen = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-7",
        working_folder_locator_fingerprint="f" * 64,
        project_instruction_notice_key="notice-key",
    )
    store.set_session_project_instruction_state("session-1", chosen)

    result = await controller.submit_draft("keep my folder", session_id="session-1")

    # The send stopped after its commit: the chat exists, identity was never
    # published to the session, and the provider was never called.
    assert result.accepted is True
    assert result.provider_started is False
    assert gateway.calls == 0
    assert store.sessions()[0].persisted_conversation_id is None
    rows = (
        db.get_connection()
        .execute("SELECT id FROM conversations WHERE deleted = 0")
        .fetchall()
    )
    assert len(rows) == 1
    conversation_id = rows[0]["id"]
    assert db.get_conversation_console_project_context(conversation_id) == (
        encode_project_context_json(chosen)
    )

    rechosen = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-8",
        working_folder_locator_fingerprint="e" * 64,
        project_instruction_notice_key="notice-key",
    )
    store.set_session_project_instruction_state("session-1", rechosen)
    assert db.get_conversation_console_project_context(conversation_id) == (
        encode_project_context_json(rechosen)
    )

    # A restart: a fresh connection and a fresh store over the same file.
    reopened_db = CharactersRAGDB(tmp_path / "controller.sqlite", client_id="restart")
    conversation = reopened_db.get_conversation_by_id(conversation_id)
    reopened = ConsoleChatStore(
        persistence=ChatPersistenceService(reopened_db)
    ).restore_persisted_session(
        title=str(conversation["title"]),
        workspace_id=conversation.get("workspace_id"),
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )
    assert reopened.project_instruction_state == rechosen


_FOLDER_7 = ProjectInstructionControlState(
    project_instructions_enabled=True,
    working_folder_binding_id="binding-7",
    working_folder_locator_fingerprint="f" * 64,
    project_instruction_notice_key="notice-key",
)
_FOLDER_8 = ProjectInstructionControlState(
    project_instructions_enabled=True,
    working_folder_binding_id="binding-8",
    working_folder_locator_fingerprint="e" * 64,
    project_instruction_notice_key="notice-key",
)


def _restored_project_state(
    tmp_path: Path,
) -> tuple[str, ProjectInstructionControlState]:
    """Reopen the only saved chat the way a restart does: new connection, new
    store. Returns its id and the project controls it comes back with."""
    reopened_db = CharactersRAGDB(tmp_path / "controller.sqlite", client_id="restart")
    rows = (
        reopened_db.get_connection()
        .execute("SELECT id FROM conversations WHERE deleted = 0")
        .fetchall()
    )
    assert len(rows) == 1, rows
    conversation = reopened_db.get_conversation_by_id(rows[0]["id"])
    reopened = ConsoleChatStore(
        persistence=ChatPersistenceService(reopened_db)
    ).restore_persisted_session(
        title=str(conversation["title"]),
        workspace_id=conversation.get("workspace_id"),
        persisted_conversation_id=rows[0]["id"],
        all_nodes=(),
    )
    return rows[0]["id"], reopened.project_instruction_state


class _WriteLockHolder:
    """A second connection holding the database write lock, as a busy
    background writer does, so a durable commit waits inside its thread."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._acquired, self._release = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        connection = sqlite3.connect(self._db_path, timeout=5)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._acquired.set()
            self._release.wait(timeout=10)
            connection.commit()
        finally:
            connection.close()

    def __enter__(self) -> _WriteLockHolder:
        self._thread.start()
        assert self._acquired.wait(timeout=5)
        return self

    def release(self) -> None:
        self._release.set()
        self._thread.join(timeout=5)

    def __exit__(self, *_exc: object) -> None:
        self.release()


async def _until(predicate, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        await asyncio.sleep(0.01)
    return bool(predicate())


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chosen", [_FOLDER_7, None], ids=["folder-chosen", "no-folder-chosen"]
)
async def test_first_send_saves_project_controls_inside_its_off_loop_commit(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chosen: ProjectInstructionControlState | None,
) -> None:
    """TASK-33621.13 review: the first send saved the project controls with a
    second write on the event loop once its commit had returned -- an on-loop
    stall under write-lock contention (TASK-22205 moved the commit itself off
    the loop for exactly that). They now commit in the durable transaction,
    on its thread. A chat with no folder chosen reopens as it was (enabled,
    no folder), the way a promoted chat already does -- not as 'Off'."""
    db, store, controller, _gateway = _controller(tmp_path)
    writes: list[tuple[bool, bool]] = []
    original_write = db.set_conversation_console_project_context

    def recording_write(conversation_id: str, project_context_json: str | None) -> None:
        writes.append(
            (
                db.get_connection().in_transaction,
                threading.current_thread() is threading.main_thread(),
            )
        )
        original_write(conversation_id, project_context_json)

    monkeypatch.setattr(db, "set_conversation_console_project_context", recording_write)
    if chosen is not None:
        store.set_session_project_instruction_state("session-1", chosen)
    expected = chosen or ProjectInstructionControlState.new_session()

    result = await controller.submit_draft("keep my controls", session_id="session-1")

    assert result.accepted is True
    # One write, inside the commit's transaction, never on the event loop.
    assert writes == [(True, False)]
    conversation_id, restored = _restored_project_state(tmp_path)
    assert conversation_id == store.sessions()[0].persisted_conversation_id
    assert restored == expected


@private_profile_test
@pytest.mark.asyncio
async def test_project_folder_survives_a_first_send_cancelled_during_its_commit(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """Cancelled first saves retain both their native work and chosen folder."""
    db, store, controller, gateway = _controller(tmp_path)
    store.set_session_project_instruction_state("session-1", _FOLDER_7)
    task = None
    try:
        with _WriteLockHolder(tmp_path / "controller.sqlite"):
            task = asyncio.create_task(
                controller.submit_draft("cancelled mid-commit", session_id="session-1")
            )
            assert await _until(
                lambda: bool(store._durable_commit_in_flight)
            ), "the durable commit never started; the cancel would land too early"
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0.05)
                assert not task.done(), "first-save ownership ended before SQLite"
                assert store._durable_commit_in_flight
        result = await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert result.accepted is True
        assert result.provider_started is False
    finally:
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        assert await _until(lambda: not store._durable_commit_in_flight)

    assert gateway.calls == 0
    _conversation_id, restored = _restored_project_state(tmp_path)
    assert restored == _FOLDER_7


@private_profile_test
@pytest.mark.asyncio
async def test_a_folder_rechosen_while_the_first_commit_runs_is_the_one_saved(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """The commit stores the controls it snapshotted when it began. A folder
    chosen again while that commit waits cannot be written yet -- there is no
    saved chat to write it to -- so the commit must store the newer choice
    once it lands, or the chat reopens with the folder the user replaced."""
    _db, store, controller, _gateway = _controller(tmp_path)
    store.set_session_project_instruction_state("session-1", _FOLDER_7)

    with _WriteLockHolder(tmp_path / "controller.sqlite") as lock:
        task = asyncio.create_task(
            controller.submit_draft("rechosen mid-commit", session_id="session-1")
        )
        assert await _until(lambda: bool(store._durable_commit_in_flight))
        store.set_session_project_instruction_state("session-1", _FOLDER_8)
        lock.release()
        result = await asyncio.wait_for(task, timeout=15)

    assert result.accepted is True
    _conversation_id, restored = _restored_project_state(tmp_path)
    assert restored == _FOLDER_8


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

    result = await controller.submit_draft("must remain atomic", session_id="session-1")

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
        assert (
            db.get_connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            == 0
        )


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
