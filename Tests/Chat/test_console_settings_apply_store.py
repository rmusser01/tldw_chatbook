from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatSession,
    ConsoleChatStore,
    ConsoleSettingsComponent,
    ConsoleSettingsPersistenceFailure,
)
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_context_repository import (
    ContextPolicyReadResult,
    ContextPolicyWriteResult,
    ContextPolicyWriteStatus,
)
from tldw_chatbook.Chat.console_generation_settings_metadata import (
    ConsoleGenerationSettingsReadStatus,
    ConsoleGenerationSettingsWriteResult,
    ConsoleGenerationSettingsWriteStatus,
    parse_console_generation_settings,
    snapshot_from_session_settings,
)
from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicySnapshot
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsDraftState,
    ConsoleSettingsSubmission,
)
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _settings(model: str, *, system_prompt: str | None = None) -> ConsoleSessionSettings:
    return ConsoleSessionSettings(
        provider="openai",
        model=model,
        temperature=0.2 if model == "model-a" else 0.8,
        system_prompt=system_prompt,
    )


def _submission(
    store: ConsoleChatStore,
    session_id: str,
    *,
    submission_id: str,
    model: str,
    compaction: ContextCompactionMode = ContextCompactionMode.ASK,
) -> ConsoleSettingsSubmission:
    return ConsoleSettingsSubmission(
        submission_id=submission_id,
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        origin=store.capture_console_settings_origin(session_id),
        draft=ConsoleSettingsDraftState(
            settings=_settings(model, system_prompt="draft must not replace owner"),
            context_policy_overrides=ConsoleContextPolicyOverrides(
                compaction_mode=compaction
            ),
            field_drafts=(),
            model_drafts=(),
            endpoint_draft=None,
        ),
        user_display_name_override=None,
        default_field_mask=frozenset(),
    )


class _SettingsPersistence:
    def __init__(self) -> None:
        self.db = None
        self.generation_snapshot = None
        self.context_policy = ConsoleContextPolicyOverrides()
        self.context_revision = None
        self.generation_calls = []
        self.context_calls = []
        self.created_conversations = []
        self.promotion_kwargs = None
        self.fail_generation = False
        self.fail_context = False
        self.roleplay_calls = []
        self.pinned_prefill_calls = []

    def update_conversation_generation_settings(self, **kwargs):
        self.generation_calls.append(kwargs)
        if self.fail_generation:
            raise RuntimeError("generation unavailable")
        if kwargs["expected_snapshot"] != self.generation_snapshot:
            return ConsoleGenerationSettingsWriteResult(
                ConsoleGenerationSettingsWriteStatus.SUPERSEDED,
                self.generation_snapshot,
            )
        self.generation_snapshot = kwargs["snapshot"]
        return ConsoleGenerationSettingsWriteResult(
            ConsoleGenerationSettingsWriteStatus.WRITTEN,
            self.generation_snapshot,
        )

    def update_conversation_context_policy(self, **kwargs):
        self.context_calls.append(kwargs)
        if self.fail_context:
            raise RuntimeError("context unavailable")
        expected = kwargs.get("expected_revision", self.context_revision)
        if expected != self.context_revision:
            return ContextPolicyWriteResult(
                ContextPolicyWriteStatus.CONFLICT,
                self.context_revision,
            )
        self.context_policy = kwargs["overrides"]
        self.context_revision = (
            None
            if self.context_policy.is_empty
            else 1 if self.context_revision is None else self.context_revision + 1
        )
        return ContextPolicyWriteResult(
            ContextPolicyWriteStatus.WRITTEN,
            self.context_revision,
        )

    def get_conversation_context_policy(self, _conversation_id):
        return ContextPolicyReadResult(self.context_policy, self.context_revision)

    def create_conversation(self, **kwargs):
        self.created_conversations.append(kwargs)
        return "created-conversation"

    def promote_console_conversation_bundle(self, **kwargs):
        self.promotion_kwargs = kwargs
        candidate = kwargs["policy_candidate"]
        return ConsoleLibraryPolicySnapshot(
            auto_retrieve=candidate.auto_retrieve,
            assistant_access=candidate.assistant_access,
            policy_revision=1,
            source="durable",
        )

    def update_conversation_roleplay_context(self, **kwargs):
        self.roleplay_calls.append(kwargs)
        return True

    def update_conversation_pinned_prefill(self, **kwargs):
        self.pinned_prefill_calls.append(kwargs)
        return True


class _AtomicSettingsPersistence(_SettingsPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.first_persist_kwargs = None

    def persist_console_conversation_with_policy(self, **kwargs):
        self.first_persist_kwargs = kwargs
        self.created_conversations.append(dict(kwargs["conversation_kwargs"]))
        candidate = kwargs["policy_candidate"]
        return ConsoleLibraryPolicySnapshot(
            auto_retrieve=candidate.auto_retrieve,
            assistant_access=candidate.assistant_access,
            policy_revision=1,
            source="durable",
        )


def test_live_commit_updates_exact_origin_preserves_prompt_and_deduplicates() -> None:
    store = ConsoleChatStore()
    origin = store.create_session(settings=_settings("old", system_prompt="owned"))
    other = store.create_session(settings=_settings("other"))
    submission = _submission(
        store,
        origin.id,
        submission_id="submission-1",
        model="model-a",
    )

    commit = store.commit_console_settings_live(submission)

    assert store.active_session_id == other.id
    assert store.session_settings(origin.id).model == "model-a"
    assert store.session_settings(origin.id).source == "user"
    assert store.session_settings(origin.id).system_prompt == "owned"
    assert store.session_settings(other.id).model == "other"
    assert commit.generation_revision == 1
    assert commit.context_policy_revision == 1
    with pytest.raises(ValueError, match="already applied"):
        store.commit_console_settings_live(submission)
    assert origin.generation_settings_revision == 1
    assert origin.context_policy_revision == 1


def test_missing_or_rebound_origin_is_rejected_without_mutation() -> None:
    store = ConsoleChatStore()
    session = store.create_session(settings=_settings("old"))
    missing = _submission(
        store,
        session.id,
        submission_id="missing",
        model="model-a",
    )
    store.close_session(session.id)

    with pytest.raises(ValueError, match="closed"):
        store.commit_console_settings_live(missing)

    rebound = store.create_session(settings=_settings("old"))
    stale = _submission(
        store,
        rebound.id,
        submission_id="rebound",
        model="model-b",
    )
    store.rebind_persisted_conversation(rebound.id, "different-conversation")
    before = rebound.settings

    with pytest.raises(ValueError, match="closed"):
        store.commit_console_settings_live(stale)

    assert rebound.settings is before
    assert rebound.generation_settings_revision == 0
    assert rebound.context_policy_revision == 0


def test_first_persist_binding_preserves_revision_but_explicit_rebind_advances() -> None:
    store = ConsoleChatStore()
    session = store.create_session(settings=_settings("old"))
    origin = store.capture_console_settings_origin(session.id)

    store.publish_first_persisted_conversation(session.id, "conversation-a")

    assert session.conversation_binding_revision == origin.conversation_binding_revision
    accepted = replace(
        _submission(
            store,
            session.id,
            submission_id="first-bind",
            model="model-a",
        ),
        origin=origin,
    )
    store.commit_console_settings_live(accepted)
    store.rebind_persisted_conversation(session.id, "conversation-b")
    assert session.conversation_binding_revision == origin.conversation_binding_revision + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_component", list(ConsoleSettingsComponent))
async def test_live_values_survive_component_failure_and_retry_is_exact(
    failed_component: ConsoleSettingsComponent,
) -> None:
    persistence = _SettingsPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(settings=_settings("old", system_prompt="owned"))
    store.publish_first_persisted_conversation(session.id, "conversation-a")
    setattr(
        persistence,
        "fail_generation"
        if failed_component is ConsoleSettingsComponent.GENERATION_SETTINGS
        else "fail_context",
        True,
    )
    commit = store.commit_console_settings_live(
        _submission(
            store,
            session.id,
            submission_id="failure",
            model="model-b",
            compaction=ContextCompactionMode.OFF,
        )
    )

    await store.persist_console_settings_commit_serialized(commit)

    assert session.settings.model == "model-b"
    assert session.context_policy_overrides.compaction_mode is ContextCompactionMode.OFF
    assert set(session.settings_persistence_failures) == {failed_component}
    failure = session.settings_persistence_failures[failed_component]
    assert failure.revision == (
        session.generation_settings_revision
        if failed_component is ConsoleSettingsComponent.GENERATION_SETTINGS
        else session.context_policy_revision
    )
    assert not hasattr(failure, "base_url")
    setattr(
        persistence,
        "fail_generation"
        if failed_component is ConsoleSettingsComponent.GENERATION_SETTINGS
        else "fail_context",
        False,
    )

    assert await store.retry_console_settings_persistence(
        session_id=session.id,
        component=failed_component,
        revision=failure.revision,
    )
    assert session.settings_persistence_failures == {}


@pytest.mark.asyncio
async def test_newer_apply_supersedes_failure_and_stale_retry() -> None:
    persistence = _SettingsPersistence()
    persistence.fail_generation = True
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(settings=_settings("old"))
    store.publish_first_persisted_conversation(session.id, "conversation-a")
    first = store.commit_console_settings_live(
        _submission(store, session.id, submission_id="first", model="model-a")
    )
    await store.persist_console_settings_commit_serialized(first)
    failure = session.settings_persistence_failures[
        ConsoleSettingsComponent.GENERATION_SETTINGS
    ]

    store.commit_console_settings_live(
        _submission(store, session.id, submission_id="second", model="model-b")
    )

    assert ConsoleSettingsComponent.GENERATION_SETTINGS not in (
        session.settings_persistence_failures
    )
    persistence.fail_generation = False
    assert not await store.retry_console_settings_persistence(
        session_id=session.id,
        component=ConsoleSettingsComponent.GENERATION_SETTINGS,
        revision=failure.revision,
    )


@pytest.mark.asyncio
async def test_rapid_applies_serialize_and_finish_with_newest_durable_values() -> None:
    persistence = _SettingsPersistence()
    started = threading.Event()
    release = threading.Event()
    original_write = persistence.update_conversation_generation_settings

    def gated_write(**kwargs):
        if not persistence.generation_calls:
            started.set()
            release.wait(timeout=2)
        return original_write(**kwargs)

    persistence.update_conversation_generation_settings = gated_write
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(settings=_settings("old"))
    store.publish_first_persisted_conversation(session.id, "conversation-a")
    first = store.commit_console_settings_live(
        _submission(store, session.id, submission_id="first", model="model-a")
    )
    first_task = asyncio.create_task(
        store.persist_console_settings_commit_serialized(first)
    )
    assert await asyncio.to_thread(started.wait, 1)
    second = store.commit_console_settings_live(
        _submission(
            store,
            session.id,
            submission_id="second",
            model="model-b",
            compaction=ContextCompactionMode.OFF,
        )
    )
    second_task = asyncio.create_task(
        store.persist_console_settings_commit_serialized(second)
    )
    release.set()

    await asyncio.gather(first_task, second_task)

    assert [call["snapshot"].model for call in persistence.generation_calls] == [
        "model-a",
        "model-b",
    ]
    assert [
        call["overrides"].compaction_mode for call in persistence.context_calls
    ] == [ContextCompactionMode.OFF]
    assert persistence.generation_snapshot.model == "model-b"
    assert persistence.context_policy.compaction_mode is ContextCompactionMode.OFF
    assert session.generation_durable_snapshot.model == "model-b"
    assert session.context_policy_durable_revision == persistence.context_revision


@pytest.mark.asyncio
async def test_inflight_apply_drains_to_latest_without_scheduling_newer_commits() -> None:
    persistence = _SettingsPersistence()
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    release_second = threading.Event()
    original_write = persistence.update_conversation_generation_settings

    def gated_write(**kwargs):
        model = kwargs["snapshot"].model
        if model == "model-a":
            first_started.set()
            release_first.wait(timeout=2)
        elif model == "model-b":
            second_started.set()
            release_second.wait(timeout=2)
        return original_write(**kwargs)

    persistence.update_conversation_generation_settings = gated_write
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(settings=_settings("old"))
    store.publish_first_persisted_conversation(session.id, "conversation-a")
    first = store.commit_console_settings_live(
        _submission(store, session.id, submission_id="first", model="model-a")
    )
    drain = asyncio.create_task(store.persist_console_settings_commit_serialized(first))
    assert await asyncio.to_thread(first_started.wait, 1)

    store.commit_console_settings_live(
        _submission(
            store,
            session.id,
            submission_id="second",
            model="model-b",
            compaction=ContextCompactionMode.OFF,
        )
    )
    release_first.set()
    assert await asyncio.to_thread(second_started.wait, 1)
    assert persistence.generation_snapshot.model == "model-a"
    assert session.generation_durable_snapshot is None

    store.commit_console_settings_live(
        _submission(
            store,
            session.id,
            submission_id="third",
            model="model-c",
            compaction=ContextCompactionMode.AUTOMATIC,
        )
    )
    release_second.set()
    outcome = await drain

    assert [call["snapshot"].model for call in persistence.generation_calls] == [
        "model-a",
        "model-b",
        "model-c",
    ]
    assert [
        call["overrides"].compaction_mode for call in persistence.context_calls
    ] == [ContextCompactionMode.AUTOMATIC]
    assert persistence.generation_snapshot.model == "model-c"
    assert persistence.context_policy.compaction_mode is ContextCompactionMode.AUTOMATIC
    assert session.generation_durable_snapshot.model == "model-c"
    assert session.context_policy_durable_revision == persistence.context_revision
    assert outcome.written_components == frozenset(ConsoleSettingsComponent)
    assert outcome.failed_components == frozenset()
    assert outcome.stale_components == frozenset()


def test_restore_state_fences_same_session_id_and_resets_old_apply_state() -> None:
    store = ConsoleChatStore()
    session = store.create_session(session_id="stable", settings=_settings("old"))
    stale = _submission(
        store,
        session.id,
        submission_id="before-restore",
        model="model-a",
    )
    safe_base = snapshot_from_session_settings(_settings("durable"))
    session.applied_settings_submission_ids.append("old-activation")
    session.settings_persistence_failures[
        ConsoleSettingsComponent.GENERATION_SETTINGS
    ] = ConsoleSettingsPersistenceFailure(
        component=ConsoleSettingsComponent.GENERATION_SETTINGS,
        revision=0,
        persisted_conversation_id="conversation-a",
        conversation_binding_revision=0,
        generation_snapshot=safe_base,
    )
    restored = ConsoleChatSession(
        id=session.id,
        persisted_conversation_id="conversation-a",
        settings=_settings("restored"),
        generation_durable_snapshot=safe_base,
        context_policy_durable_revision=9,
        settings_persistence_failures=session.settings_persistence_failures,
        applied_settings_submission_ids=session.applied_settings_submission_ids,
    )

    store.restore_state(sessions=[restored], active_session_id=session.id)
    replacement = store.sessions()[0]

    assert replacement.conversation_binding_revision == 1
    assert replacement.generation_durable_snapshot == safe_base
    assert replacement.context_policy_durable_revision == 9
    assert replacement.settings_persistence_failures == {}
    assert tuple(replacement.applied_settings_submission_ids) == ()
    with pytest.raises(ValueError, match="closed"):
        store.commit_console_settings_live(stale)
    assert replacement.settings.model == "restored"


@pytest.mark.asyncio
async def test_external_generation_change_is_refused_not_overwritten(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "generation-owned-base.db", "task-22515")
    try:
        service = ChatPersistenceService(db)
        conversation_id = service.create_conversation(conversation_title="External")
        store = ConsoleChatStore(persistence=service)
        session = store.create_session(settings=_settings("old"))
        store.publish_first_persisted_conversation(session.id, conversation_id)
        commit = store.commit_console_settings_live(
            _submission(store, session.id, submission_id="local", model="model-a")
        )
        external = replace(commit.settings, model="external")
        service.update_conversation_generation_settings(
            conversation_id=conversation_id,
            snapshot=snapshot_from_session_settings(external),
            expected_snapshot=None,
        )

        await store.persist_console_settings_commit_serialized(commit)

        assert service.get_conversation_generation_settings(
            conversation_id
        ).snapshot.model == "external"
        assert ConsoleSettingsComponent.GENERATION_SETTINGS in (
            session.settings_persistence_failures
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_generation_failure_captures_only_exact_safe_snapshot() -> None:
    persistence = _SettingsPersistence()
    persistence.fail_generation = True
    store = ConsoleChatStore(persistence=persistence)
    secret_endpoint = "https://user:secret@example.invalid/api?token=private"
    session = store.create_session(
        settings=replace(_settings("old"), base_url=secret_endpoint)
    )
    store.publish_first_persisted_conversation(session.id, "conversation-a")
    submission = _submission(
        store,
        session.id,
        submission_id="safe-failure",
        model="model-b",
    )
    submission = replace(
        submission,
        draft=replace(
            submission.draft,
            settings=replace(submission.draft.settings, base_url=secret_endpoint),
        ),
    )
    commit = store.commit_console_settings_live(submission)

    await store.persist_console_settings_commit_serialized(commit)

    failure = session.settings_persistence_failures[
        ConsoleSettingsComponent.GENERATION_SETTINGS
    ]
    assert failure.generation_snapshot == snapshot_from_session_settings(commit.settings)
    assert not hasattr(failure.generation_snapshot, "base_url")
    assert secret_endpoint not in repr(failure)


def test_unsaved_apply_first_persist_preserves_existing_owner_paths() -> None:
    persistence = _AtomicSettingsPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        settings=replace(
            _settings("old", system_prompt="Owned system"),
            pinned_prefill="Owned prefill",
        )
    )
    session.speech_preferences = ConsoleSpeechPreferences(auto_speak=True)
    session.user_display_name_override = "Ari"
    session.character_system_template = "Hello {user_name}"
    commit = store.commit_console_settings_live(
        _submission(
            store,
            session.id,
            submission_id="staged",
            model="model-b",
            compaction=ContextCompactionMode.AUTOMATIC,
        )
    )

    asyncio.run(store.persist_console_settings_commit_serialized(commit))

    assert persistence.created_conversations == []
    conversation_id = store.persist_session_if_needed(session.id)
    metadata = persistence.first_persist_kwargs["conversation_kwargs"]["metadata"]
    assert conversation_id == persistence.first_persist_kwargs["conversation_id"]
    assert parse_console_generation_settings(metadata).snapshot.model == "model-b"
    assert session.generation_durable_snapshot.model == "model-b"
    assert persistence.context_policy.compaction_mode is ContextCompactionMode.AUTOMATIC
    assert persistence.first_persist_kwargs["conversation_kwargs"][
        "speech_preferences"
    ] == ConsoleSpeechPreferences(auto_speak=True)
    assert persistence.first_persist_kwargs["policy_candidate"].auto_retrieve == (
        session.library_policy_holder.snapshot.auto_retrieve
    )
    assert persistence.roleplay_calls == [
        {
            "conversation_id": conversation_id,
            "user_name_override": "Ari",
            "character_system_template": "Hello {user_name}",
        }
    ]
    assert persistence.pinned_prefill_calls == [
        {
            "conversation_id": conversation_id,
            "pinned_prefill": "Owned prefill",
        }
    ]
    assert persistence.generation_calls == []


def test_temporary_apply_does_not_write_and_promotion_uses_staged_settings() -> None:
    persistence = _SettingsPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(settings=_settings("old"), ephemeral=True)
    commit = store.commit_console_settings_live(
        _submission(store, session.id, submission_id="temporary", model="model-b")
    )

    asyncio.run(store.persist_console_settings_commit_serialized(commit))

    assert persistence.generation_calls == []
    assert persistence.context_calls == []
    store.promote_ephemeral_session(session.id)
    metadata = persistence.promotion_kwargs["conversation_kwargs"]["metadata"]
    assert parse_console_generation_settings(metadata).snapshot.model == "model-b"
    assert session.generation_durable_snapshot.model == "model-b"


def test_promotion_context_failure_enters_ledger_without_rolling_back() -> None:
    persistence = _SettingsPersistence()
    persistence.fail_context = True
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(settings=_settings("old"), ephemeral=True)
    store.commit_console_settings_live(
        _submission(
            store,
            session.id,
            submission_id="promotion-context-failure",
            model="model-b",
            compaction=ContextCompactionMode.AUTOMATIC,
        )
    )

    conversation_id = store.promote_ephemeral_session(session.id)

    assert conversation_id is not None
    assert session.ephemeral is False
    assert session.persisted_conversation_id == conversation_id
    failure = session.settings_persistence_failures[
        ConsoleSettingsComponent.CONTEXT_POLICY
    ]
    assert failure.context_policy_overrides == session.context_policy_overrides
    assert failure.revision == session.context_policy_revision


@pytest.mark.asyncio
async def test_resumed_absent_policy_apply_empty_publishes_none_without_failure() -> None:
    persistence = _SettingsPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.restore_persisted_session(
        title="Restored empty",
        workspace_id=None,
        persisted_conversation_id="conversation-empty",
        all_nodes=[],
        settings=_settings("old"),
    )
    submission = _submission(
        store,
        session.id,
        submission_id="empty-policy",
        model="model-a",
    )
    submission = replace(
        submission,
        draft=replace(
            submission.draft,
            context_policy_overrides=ConsoleContextPolicyOverrides(),
        ),
    )
    commit = store.commit_console_settings_live(submission)

    await store.persist_console_settings_commit_serialized(commit)

    assert persistence.context_calls[-1]["expected_revision"] is None
    assert persistence.context_revision is None
    assert session.context_policy_durable_revision is None
    assert ConsoleSettingsComponent.CONTEXT_POLICY not in (
        session.settings_persistence_failures
    )


def test_restore_seeds_context_durable_revision_and_generation_metadata_state() -> None:
    persistence = _SettingsPersistence()
    persistence.context_policy = ConsoleContextPolicyOverrides(
        compaction_mode=ContextCompactionMode.AUTOMATIC
    )
    persistence.context_revision = 7
    store = ConsoleChatStore(persistence=persistence)

    session = store.restore_persisted_session(
        title="Restored",
        workspace_id=None,
        persisted_conversation_id="conversation-a",
        all_nodes=[],
        settings=_settings("model-a"),
        generation_metadata_status=ConsoleGenerationSettingsReadStatus.ABSENT,
    )
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF),
    )

    assert session.context_policy_durable_revision == 8
    assert persistence.context_calls[-1]["expected_revision"] == 7
    assert session.context_policy_overrides.compaction_mode is ContextCompactionMode.OFF


def test_session_failure_state_does_not_capture_endpoint_or_secrets() -> None:
    session_store = ConsoleChatStore()
    session = session_store.create_session(
        settings=replace(_settings("model-a"), base_url="https://secret.example/key")
    )

    assert session.settings_persistence_failures == {}
    assert session.applied_settings_submission_ids.maxlen == 32
    assert session.generation_metadata_status is ConsoleGenerationSettingsReadStatus.ABSENT
    assert session.generation_metadata_warning_shown is False
    assert session.new_chat_default_generation == 0
