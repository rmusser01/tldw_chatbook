"""Exact-origin mix-in selection, capture and detached turn custody."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prompt_mixins import (
    MixinAttemptSnapshot,
    MixinPreferenceConflictError,
    MixinPreferences,
    MixinSelection,
    MixinSnapshot,
    MixinSnapshotUnavailableError,
    MixinSourceChangedError,
    MixinSourceRef,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    LiteralMixinDefinition,
)

pytestmark = pytest.mark.bootstrap_profile
A = MixinSourceRef("profile-test", "local", "00000000-0000-4000-8000-000000000001")
B = replace(A, record_uuid="00000000-0000-4000-8000-000000000002")


def test_placement_edit_moves_entry_after_existing_after_entries():
    from tldw_chatbook.Chat.console_prompt_mixins import reconcile_mixin_placements

    prefs = MixinPreferences(
        entries=(
            MixinSelection(A, True, "A", "before"),
            MixinSelection(B, False, "B", "after"),
        )
    )
    updated = MixinSnapshot(A, 2, "A", LiteralMixinDefinition("A body", "after"))
    reconciled = reconcile_mixin_placements(prefs, (updated,))
    assert tuple(row.source for row in reconciled.entries) == (B, A)
    assert reconciled.entries[0].enabled is False
    assert reconciled.entries[1].last_used_version == 2


class Scope:
    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.source, self.database = object(), object()
        self.calls = []
        self.records = {
            ref.record_uuid: {
                "uuid": ref.record_uuid,
                "version": 1,
                "name": ref.record_uuid,
                "artifact_type": "mixin",
                "prompt_format": "structured",
                "prompt_schema_version": 3,
                "prompt_definition": {
                    "kind": "text_mixin",
                    "schema_version": 3,
                    "body": "body-" + ref.record_uuid,
                    "placement": "before",
                },
            }
            for ref in (A, B)
        }

    def mixin_source_identity(self, *, mode):
        return ("profile-test", self.source, self.database)

    async def read_mixin_records(self, *, mode, record_uuids):
        self.calls.append(record_uuids)
        self.entered.set()
        await self.release.wait()
        return tuple(self.records[key] for key in record_uuids)


@pytest.fixture
def case():
    from tldw_chatbook.Chat.console_prompt_mixin_service import (
        ConsolePromptMixinService,
    )

    store = ConsoleChatStore()
    a = store.create_session(session_id="chat-a", ephemeral=True)
    b = store.create_session(session_id="chat-b", ephemeral=True)
    scope = Scope()
    service = ConsolePromptMixinService(
        store=store, prompt_scope_service=scope, persistence=None
    )
    service.replace_preferences(
        a.id,
        expected_revision=0,
        value=MixinPreferences(
            entries=(
                MixinSelection(A, True, "A", "before"),
                MixinSelection(B, False, "B", "after"),
            )
        ),
    )
    try:
        yield SimpleNamespace(store=store, a=a, b=b, scope=scope, service=service)
    finally:
        for session in store.sessions():
            store.close_session(session.id)
        store.end_app_runtime()


@pytest.mark.asyncio
async def test_held_capture_uses_origin_and_disabled_missing_does_not_block(case):
    task = asyncio.create_task(case.service.capture(case.a.id, expected_revision=1))
    await asyncio.wait_for(case.scope.entered.wait(), 5)
    case.store.switch_session(case.b.id)
    case.service.replace_preferences(
        case.b.id,
        expected_revision=0,
        value=MixinPreferences(entries=(MixinSelection(B, True, "B", "after"),)),
    )
    del case.scope.records[B.record_uuid]
    case.scope.release.set()
    snapshot = await task
    assert tuple(item.source for item in snapshot.items) == (A,)
    assert case.scope.calls == [(A.record_uuid,)]
    assert case.service.preferences(case.b.id).entries[0].source == B


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["preferences", "source", "database", "incarnation"])
async def test_held_capture_refuses_stale_source_and_preserves_draft(case, change):
    case.store.set_session_draft(case.a.id, "keep this draft")
    task = asyncio.create_task(case.service.capture(case.a.id, expected_revision=1))
    await asyncio.wait_for(case.scope.entered.wait(), 5)
    if change == "preferences":
        case.service.replace_preferences(
            case.a.id, expected_revision=1, value=MixinPreferences()
        )
    elif change == "source":
        case.scope.source = object()
    elif change == "database":
        case.scope.database = object()
    else:
        case.a.incarnation_id = "replacement"
    case.scope.release.set()
    with pytest.raises((MixinPreferenceConflictError, MixinSourceChangedError)):
        await task
    assert case.a.draft == "keep this draft"


@pytest.mark.asyncio
async def test_changed_library_version_and_placement_affect_next_capture_only(case):
    case.scope.release.set()
    original = await case.service.capture(case.a.id, expected_revision=1)
    case.scope.records[A.record_uuid].update(
        version=2,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "edited",
            "placement": "after",
        },
    )
    updated = await case.service.capture(case.a.id, expected_revision=1)
    assert original.items[0].record_version == 1
    assert updated.items[0].record_version == 2
    assert updated.items[0].definition.body == "edited"
    assert tuple(row.source for row in case.service.preferences(case.a.id).entries) == (
        A,
        B,
    )
    assert case.service.preferences(case.a.id).entries[0].last_used_version is None
    case.service.publish_accepted_usage(case.a.id, updated)
    assert tuple(row.source for row in case.service.preferences(case.a.id).entries) == (
        A,
        B,
    )


@pytest.mark.asyncio
async def test_enabled_missing_record_refuses(case):
    del case.scope.records[A.record_uuid]
    case.scope.release.set()
    with pytest.raises(MixinSnapshotUnavailableError):
        await case.service.capture(case.a.id, expected_revision=1)


def test_live_cas_rejects_stale_intent_and_toggle_keeps_positions(case):
    prefs = case.service.preferences(case.a.id)
    updated = case.service.replace_preferences(
        case.a.id,
        expected_revision=1,
        value=replace(
            prefs, entries=(replace(prefs.entries[0], enabled=False), prefs.entries[1])
        ),
    )
    assert tuple(row.source for row in updated.entries) == (A, B)
    with pytest.raises(MixinPreferenceConflictError):
        case.service.replace_preferences(case.a.id, expected_revision=1, value=prefs)


def test_received_input_and_claim_fence_preference_before_any_worker(case):
    inputs = case.store.session_input_snapshot(case.a.id)
    claim = case.store.claim_received_turn(case.a.id, "first")
    assert claim is not None
    assert case.store.claim_received_turn(case.a.id, "second") is None
    case.service.replace_preferences(
        case.a.id, expected_revision=1, value=MixinPreferences()
    )
    assert inputs.prompt_mixin_preferences.entries[0].source == A
    assert not case.store.session_inputs_are_current(inputs)
    assert not case.store.received_turn_is_current(claim)
    assert case.store.release_received_turn(claim)


@pytest.mark.asyncio
async def test_save_failure_retry_is_exact_and_new_choice_supersedes_it(case):
    from tldw_chatbook.Chat.console_chat_store import ConsoleSettingsComponent

    component = ConsoleSettingsComponent.PROMPT_MIXINS

    class Persistence:
        def __init__(self):
            self.fail = True
            self.revision = 0
            self.values = []

        def save_conversation_mixin_preferences(
            self, *, conversation_id, expected_revision, value
        ):
            assert conversation_id == "saved-a"
            assert expected_revision == self.revision
            self.values.append(value)
            if self.fail:
                raise OSError("injected")
            self.revision += 1
            return replace(value, revision=self.revision)

    persistence = Persistence()
    case.store.persistence = persistence
    case.a.ephemeral = False
    case.a.persisted_conversation_id = "saved-a"
    result = await case.store.persist_prompt_mixin_preferences(case.a.id)
    assert result.failed_components == frozenset({component})
    assert case.a.settings_persistence_failures[component].revision == 1
    assert case.service.preferences(case.a.id).entries[0].enabled
    persistence.fail = False
    assert await case.store.retry_console_settings_persistence(
        session_id=case.a.id, component=component, revision=1
    )
    assert not case.a.settings_persistence_failures
    persistence.fail = True
    case.service.replace_preferences(
        case.a.id, expected_revision=1, value=MixinPreferences()
    )
    await case.store.persist_prompt_mixin_preferences(case.a.id)
    assert not await case.store.retry_console_settings_persistence(
        session_id=case.a.id, component=component, revision=1
    )
    assert case.service.preferences(case.a.id).entries == ()


@pytest.mark.asyncio
async def test_unknown_preferences_preserved_and_unavailable(case):
    from tldw_chatbook.Chat.console_prompt_mixins import decode_mixin_preferences

    raw = '{"schema_version":47,"future":"keep"}'
    case.a.prompt_mixin_preferences = decode_mixin_preferences(raw)
    with pytest.raises(MixinPreferenceConflictError):
        case.service.replace_preferences(
            case.a.id, expected_revision=0, value=MixinPreferences()
        )
    with pytest.raises(MixinPreferenceConflictError):
        await case.service.capture(case.a.id, expected_revision=0)
    assert case.a.prompt_mixin_preferences.raw == raw


def test_source_identity_survives_reopen_but_fences_db_replacement(tmp_path):
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.prompt_scope_service import (
        LocalPromptService,
        PromptScopeService,
    )

    path = tmp_path / "credential@private.db"
    first = PromptsDatabase(path, client_id="mixin-first")
    second = None
    try:
        scope = PromptScopeService(LocalPromptService(first), None)
        original = scope.mixin_source_identity(mode="local")
        first.close_connection()
        second = PromptsDatabase(path, client_id="mixin-reopen")
        scope.local_service.prompt_db = second
        reopened = scope.mixin_source_identity(mode="local")
        assert original[0] == reopened[0]
        assert original[2] is not reopened[2]
        assert "credential" not in reopened[0] and "@" not in reopened[0]
        second.db_path = tmp_path / "retarget.db"
        assert scope.mixin_source_identity(mode="local")[0] != reopened[0]
        second.db_path = path
        with pytest.raises(ValueError):
            scope.mixin_source_identity(mode="server")
    finally:
        first.close_connection()
        if second is not None:
            second.close_connection()


def test_private_session_state_preserves_preferences_without_selection_activation(case):
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

    state = ConsoleSessionController._console_session_to_state(case.a)
    owner = object.__new__(ConsoleSessionController)
    owner._current_chat_store_accessor = lambda: case.store
    restored = owner._console_session_from_state(state)
    assert restored.prompt_mixin_preferences.preferences == case.service.preferences(
        case.a.id
    )
    assert restored.incarnation_id != case.a.incarnation_id


@pytest.mark.asyncio
async def test_native_preferences_save_and_reopen_are_chat_local(tmp_path):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "selections.db", "mixins-admission")
    store = None
    try:
        persistence = ChatPersistenceService(database)
        conversation = persistence.create_conversation(
            conversation_title="Saved mix-ins"
        )
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session()
        store.publish_first_persisted_conversation(session.id, conversation)
        prefs = store.replace_prompt_mixin_preferences(
            session.id,
            expected_revision=0,
            value=MixinPreferences(entries=(MixinSelection(A, True, "A", "before"),)),
        )
        result = await store.persist_prompt_mixin_preferences(session.id)
        assert not result.failed_components
        reopened = persistence.get_conversation_mixin_preferences(conversation)
        assert reopened.preferences.entries == prefs.entries
        assert reopened.preferences.revision == 1
        other = store.create_session()
        assert other.prompt_mixin_preferences.preferences.entries == ()
        assert (
            store.session_input_snapshot(other.id).prompt_mixin_preferences.entries
            == ()
        )
    finally:
        if store is not None:
            for live in store.sessions():
                store.close_session(live.id)
            store.end_app_runtime()
        database.close_connection()


@pytest.mark.asyncio
async def test_capture_queue_custody_and_final_context_keep_original_bodies(case):
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_prompt_queue import ConsolePromptQueueRegistry
    from tldw_chatbook.Chat.console_turn_context import (
        ConsoleTurnConfigurationSnapshot,
        ConsoleTurnCustodyRequest,
        _detached_configuration,
    )

    case.scope.release.set()
    frozen = await case.service.capture(case.a.id, expected_revision=1)
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=case.a.id,
        provider_selection=ConsoleProviderSelection(provider="llama_cpp"),
        prompt_mixins=frozen,
    )
    request = ConsoleTurnCustodyRequest(
        "queued-mixins", case.a.id, "raw draft", configuration
    )
    registry = ConsolePromptQueueRegistry()
    registry.begin_chain(case.a.id, context_epoch=0, expected_revision=0)
    admitted = registry.admit(
        case.a.id,
        text=request.draft,
        expected_revision=registry.snapshot(case.a.id).revision,
        custody_request=request,
    )
    assert admitted.applied
    case.service.replace_preferences(
        case.a.id, expected_revision=1, value=MixinPreferences()
    )
    case.scope.records.clear()
    claimed = registry.claim_next(
        case.a.id, expected_revision=registry.snapshot(case.a.id).revision
    )
    assert claimed.applied
    held = claimed.claim.prompt.custody_request
    assert held.configuration.prompt_mixins == frozen
    assert _detached_configuration(held.configuration).prompt_mixins == frozen
    assert "body-" not in repr(registry.snapshot(case.a.id))
    assert "body-" not in repr(configuration)
    assert case.scope.calls == [(A.record_uuid,)]


@pytest.mark.asyncio
async def test_app_settings_owner_drains_coalesced_selection_saves(case):
    from tldw_chatbook.Chat.console_settings_durability import (
        ConsoleSettingsDurabilityOwner,
    )

    owner = ConsoleSettingsDurabilityOwner()
    calls = []

    class Persistence:
        def save_conversation_mixin_preferences(
            self, *, conversation_id, expected_revision, value
        ):
            calls.append(value)
            return replace(value, revision=expected_revision + 1)

    case.store.persistence = Persistence()
    case.a.ephemeral = False
    case.a.persisted_conversation_id = "saved-owner"
    case.service.settings_durability_owner = owner
    for revision in range(1, 8):
        case.service.replace_preferences(
            case.a.id, expected_revision=revision, value=MixinPreferences()
        )
    assert len(owner.tasks) == 1
    await owner.close_and_drain()
    assert len(calls) == 1 and calls[0].revision == 8
    with pytest.raises(
        MixinPreferenceConflictError, match="mixin_save_admission_closed"
    ):
        case.service.replace_preferences(
            case.a.id, expected_revision=8, value=MixinPreferences()
        )
    assert case.service.preferences(case.a.id).revision == 8


@pytest.mark.asyncio
async def test_native_source_worker_retires_before_cancel_delivery(case, tmp_path):
    import threading

    from tldw_chatbook.Chat.console_preparation_reads import preparation_reads_for
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.prompt_scope_service import (
        LocalPromptService,
        PromptScopeService,
    )

    database = PromptsDatabase(tmp_path / "source.db", client_id="mixins-finite")
    scope = PromptScopeService(LocalPromptService(database), None)
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    original = database.fetch_mixin_snapshot

    def read(uuids):
        entered.set()
        assert release.wait(5)
        try:
            return original(uuids)
        finally:
            retired.set()

    database.fetch_mixin_snapshot = read
    try:
        saved = await scope.save_prompt(
            mode="local",
            name="Finite",
            artifact_type="mixin",
            prompt_format="structured",
            prompt_schema_version=3,
            prompt_definition={
                "kind": "text_mixin",
                "schema_version": 3,
                "body": "literal",
                "placement": "before",
            },
        )
        authority = scope.mixin_source_identity(mode="local")[0]
        case.service.prompt_scope_service = scope
        owner = SimpleNamespace(_preparation_reads=set())
        case.service.preparation_owner = owner
        preferences = MixinPreferences(
            entries=(
                MixinSelection(
                    MixinSourceRef(authority, "local", saved["uuid"]),
                    True,
                    "Finite",
                    "before",
                ),
            )
        )
        case.service.replace_preferences(
            case.a.id, expected_revision=1, value=preferences
        )
        task = asyncio.create_task(case.service.capture(case.a.id, expected_revision=2))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert preparation_reads_for(owner._preparation_reads, case.a.id)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert retired.is_set()
            assert not preparation_reads_for(owner._preparation_reads, case.a.id)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_original_retry_uses_memory_snapshot_without_library_reads(case):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    case.scope.release.set()
    frozen = await case.service.capture(case.a.id, expected_revision=1)
    message = case.store.append_message(
        case.a.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        metadata=MessageMetadata(
            mixin_attempt_id="original", mixin_snapshot_schema_version=1
        ),
    )
    from tldw_chatbook.Chat.console_prompt_mixins import MixinAttemptWrite

    case.a.prompt_mixin_attempts["original"] = MixinAttemptWrite(
        "original", "user", message.id, frozen
    )
    case.scope.records.clear()
    assert (
        await case.service.retry_snapshot(case.a.id, assistant_message_id=message.id)
        == frozen
    )
    del case.a.prompt_mixin_attempts["original"]
    with pytest.raises(MixinSnapshotUnavailableError):
        await case.service.retry_snapshot(case.a.id, assistant_message_id=message.id)
    legacy = case.store.append_message(
        case.a.id, role=ConsoleMessageRole.ASSISTANT, content="legacy"
    )
    assert (
        await case.service.retry_snapshot(case.a.id, assistant_message_id=legacy.id)
    ).items == ()
    assert case.scope.calls == [(A.record_uuid,)]


@pytest.mark.asyncio
async def test_coordinator_drain_keeps_captured_mixins_after_delete(case):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleProviderSelection,
        ConsoleRunStatus,
        ConsoleSubmissionOrigin,
    )
    from tldw_chatbook.Chat.console_prompt_queue import (
        ConsolePromptQueueRegistry,
        PromptQueuePauseReason,
    )
    from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
        ConsolePromptQueueCoordinator,
    )
    from tldw_chatbook.Chat.console_turn_context import (
        ConsoleTurnConfigurationSnapshot,
        ConsoleTurnCustodyRequest,
    )

    case.scope.release.set()
    captured = await case.service.capture(case.a.id, expected_revision=1)
    request = ConsoleTurnCustodyRequest(
        "queued-mixin-drain",
        case.a.id,
        "draft",
        ConsoleTurnConfigurationSnapshot.capture(
            session_id=case.a.id,
            provider_selection=ConsoleProviderSelection(provider="llama_cpp"),
            prompt_mixins=captured,
        ),
    )
    registry = ConsolePromptQueueRegistry()
    registry.begin_chain(case.a.id, context_epoch=0, expected_revision=0)
    assert registry.admit(
        case.a.id,
        text="draft",
        expected_revision=registry.snapshot(case.a.id).revision,
        custody_request=request,
    ).applied
    registry.pause(
        case.a.id,
        reason=PromptQueuePauseReason.MANUAL,
        expected_revision=registry.snapshot(case.a.id).revision,
    )
    seen = []

    async def submit(prompt, *, session_id, entry_id, authorization):
        seen.append(prompt.custody_request.configuration.prompt_mixins)
        coordinator.turn_accepted(
            session_id,
            origin=ConsoleSubmissionOrigin.QUEUED,
            context_epoch=0,
            entry_id=entry_id,
        )
        return ConsoleSubmitResult(
            True, True, terminal_status=ConsoleRunStatus.COMPLETED
        )

    coordinator = ConsolePromptQueueCoordinator(
        registry=registry,
        context_epoch=lambda _: 0,
        run_status=lambda _: ConsoleRunStatus.COMPLETED,
        submit_queued=submit,
    )
    case.service.replace_preferences(
        case.a.id, expected_revision=1, value=MixinPreferences()
    )
    queued_id = registry.snapshot(case.a.id).entries[0].entry_id
    replacement_request = replace(
        request,
        draft="edited draft",
        configuration=replace(
            request.configuration, prompt_mixins=MixinAttemptSnapshot()
        ),
    )
    assert registry.edit(
        case.a.id,
        entry_id=queued_id,
        text="edited draft",
        expected_revision=registry.snapshot(case.a.id).revision,
        custody_request=replacement_request,
    ).applied
    case.scope.records.clear()
    await coordinator.resume_and_drain(case.a.id)
    assert seen == [captured]
    assert registry.snapshot(case.a.id).waiting_count == 0
    assert case.scope.calls == [(A.record_uuid,)]


@pytest.mark.asyncio
async def test_temporary_promotion_seeds_preferences_atomically(tmp_path):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "promotion.db", "mixins-promotion")
    store = ConsoleChatStore(persistence=ChatPersistenceService(database))
    try:
        session = store.create_session(ephemeral=True)
        preferences = store.replace_prompt_mixin_preferences(
            session.id,
            expected_revision=0,
            value=MixinPreferences(entries=(MixinSelection(A, True, "A", "before"),)),
        )
        conversation = await asyncio.to_thread(
            store.promote_ephemeral_session, session.id
        )
        assert conversation and not session.ephemeral
        saved = store.persistence.get_conversation_mixin_preferences(conversation)
        assert saved.preferences == preferences
        reopened = store.restore_persisted_session(
            title="Reopened",
            workspace_id=None,
            persisted_conversation_id=conversation,
            all_nodes=[],
        )
        assert reopened.prompt_mixin_preferences.preferences == preferences
        assert reopened.incarnation_id != session.incarnation_id
    finally:
        for session in store.sessions():
            store.close_session(session.id)
        store.end_app_runtime()
        database.close_connection()


def test_old_selection_handle_refuses_same_id_recreation_and_rebinding(case):
    old = case.a
    case.store.close_session(old.id)
    replacement = case.store.create_session(session_id=old.id, ephemeral=True)
    with pytest.raises(MixinPreferenceConflictError, match="mixin_session_changed"):
        case.service.replace_preferences(
            old.id, expected_revision=0, value=MixinPreferences()
        )
    assert replacement.prompt_mixin_preferences.preferences.revision == 0


def test_bound_selection_handle_refuses_explicit_rebinding(case):
    case.store.rebind_persisted_conversation(case.a.id, "different-conversation")
    with pytest.raises(MixinPreferenceConflictError, match="mixin_session_changed"):
        case.service.replace_preferences(
            case.a.id, expected_revision=0, value=MixinPreferences()
        )
    assert case.a.prompt_mixin_preferences.preferences.entries == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["recorded", "empty", "missing", "corrupt", "rewritten-record", "legacy"]
)
async def test_native_retry_classification_survives_unparsed_metadata(
    case, tmp_path, state
):
    import json

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_prompt_mixins import MixinAttemptWrite
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.console_prompt_mixins import write_attempt

    database = CharactersRAGDB(tmp_path / f"{state}.db", "mixins-retry")
    try:
        persistence = ChatPersistenceService(database)
        conversation = persistence.create_conversation(conversation_title="Retry")
        user = database.add_message(
            {"conversation_id": conversation, "sender": "user", "content": "question"}
        )
        assistant = database.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "content": "answer",
                "parent_message_id": user,
            }
        )
        snapshot = MixinAttemptSnapshot(
            ()
            if state == "empty"
            else (
                MixinSnapshot(
                    A, 1, "A", LiteralMixinDefinition("original body", "before")
                ),
            )
        )
        if state not in ("legacy", "corrupt"):
            with database.transaction() as cursor:
                write_attempt(
                    cursor,
                    conversation,
                    batch=MixinAttemptWrite("original", user, assistant, snapshot),
                    native_message_ids={user: user, assistant: assistant},
                )
                if state == "missing":
                    cursor.execute(
                        "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id=?",
                        (conversation,),
                    )
                if state in ("corrupt", "rewritten-record"):
                    marker = {}
                    cursor.execute(
                        "UPDATE messages SET metadata_json=? WHERE id=?",
                        (json.dumps(marker), assistant),
                    )
        if state == "corrupt":
            with database.transaction() as cursor:
                cursor.execute(
                    "UPDATE messages SET metadata_json=? WHERE id=?",
                    (json.dumps({"mixin_attempt_id": "original"}), assistant),
                )
        expected_status = (
            "recorded"
            if state in ("recorded", "empty", "rewritten-record")
            else "legacy"
            if state == "legacy"
            else "unavailable"
        )
        durable_read = persistence.get_conversation_mixin_attempt(
            conversation, assistant_message_id=assistant
        )
        assert durable_read.status == expected_status
        case.store.persistence = persistence
        case.service.persistence = persistence
        from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

        message = ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="cached reply",
            persisted_message_id=assistant,
        )
        restored = case.store.restore_persisted_session(
            title="Retry restored",
            workspace_id=None,
            persisted_conversation_id=conversation,
            all_nodes=[message],
        )
        assert (
            case.store.persisted_message_id_for_session_node(restored.id, message.id)
            == assistant
        )
        if state in ("missing", "corrupt"):
            with pytest.raises(MixinSnapshotUnavailableError):
                await case.service.retry_snapshot(
                    restored.id, assistant_message_id=message.id
                )
        else:
            replay = await case.service.retry_snapshot(
                restored.id, assistant_message_id=message.id
            )
            assert replay == (MixinAttemptSnapshot() if state == "legacy" else snapshot)
        assert case.scope.calls == []
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_complete_supplied_empty_configuration_does_not_recapture(case):
    from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot

    gateway = SequencedGateway()
    gateway.release[0].set()
    controller = ConsoleChatController(
        store=case.store, provider_gateway=gateway, agent_runtime_enabled=False
    )
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=case.a.id,
        provider_selection=ConsoleProviderSelection(
            provider="llama_cpp", explicit_model="test-model"
        ),
        tool_configuration={"agent_runtime_enabled": False},
    )
    try:
        result = await controller.submit_draft(
            "complete request", session_id=case.a.id, configuration=configuration
        )
        assert result.accepted
        assert configuration.prompt_mixins.items == ()
        assert case.scope.calls == []
    finally:
        await controller.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["queue", "manual"])
@pytest.mark.parametrize("change", ["scope", "backend", "database", "path"])
async def test_direct_admission_retains_source_across_hook_await(
    case, tmp_path, monkeypatch, route, change
):
    from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.prompt_scope_service import (
        LocalPromptService,
        PromptScopeService,
    )

    database = PromptsDatabase(tmp_path / "source.db", client_id="direct-source")
    original_path = database.db_path
    replacement_database = None
    scope = PromptScopeService(LocalPromptService(database), None)
    saved = await scope.save_prompt(
        mode="local",
        name="A",
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "original",
            "placement": "before",
        },
    )
    selected = MixinSourceRef(
        scope.mixin_source_identity(mode="local")[0], "local", saved["uuid"]
    )
    case.service.prompt_scope_service = scope
    case.service.replace_preferences(
        case.a.id,
        expected_revision=1,
        value=MixinPreferences(
            entries=(MixinSelection(selected, True, "A", "before"),)
        ),
    )
    case.store.set_session_draft(case.a.id, "keep direct draft")

    class Gateway(SequencedGateway):
        resolutions = 0

        async def resolve_for_send(self, selection):
            self.resolutions += 1
            return await super().resolve_for_send(selection)

    gateway = Gateway()
    gateway.release[0].set()
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=case.a.id,
        provider_selection=ConsoleProviderSelection(
            provider="llama_cpp", explicit_model="test-model"
        ),
        tool_configuration={"agent_runtime_enabled": False},
    )
    controller = ConsoleChatController(
        store=case.store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
        turn_context_provider=lambda _: configuration,
    )
    controller.app = SimpleNamespace(prompt_scope_service=scope)
    entered, release = asyncio.Event(), asyncio.Event()

    async def hook():
        entered.set()
        await release.wait()

    monkeypatch.setattr(controller, "hook_admission_reason", hook)
    if route == "queue":
        registry = controller.prompt_queue_registry
        assert registry.begin_chain(
            case.a.id,
            context_epoch=case.store.conversation_context_epoch(case.a.id),
            expected_revision=0,
        ).applied
        task = asyncio.create_task(
            controller.queue_prompt(
                case.a.id,
                text="keep direct draft",
                expected_revision=registry.snapshot(case.a.id).revision,
            )
        )
    else:
        task = asyncio.create_task(
            controller.submit_draft("keep direct draft", session_id=case.a.id)
        )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not task.done()
        if change == "scope":
            controller.app.prompt_scope_service = PromptScopeService(
                LocalPromptService(database), None
            )
        elif change == "backend":
            scope.local_service = LocalPromptService(database)
        elif change == "database":
            replacement_database = PromptsDatabase(
                original_path, client_id="direct-replacement"
            )
            scope.local_service.prompt_db = replacement_database
        else:
            database.db_path = tmp_path / "retarget.db"
        release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        assert isinstance(result, MixinSourceChangedError)
        assert gateway.resolutions == 0 and gateway.user_turns == []
        assert controller.prompt_queue_registry.snapshot(case.a.id).total_count == 0
        assert case.a.draft == "keep direct draft"
        assert case.store.preparation_for_session(case.a.id) is None
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        database.db_path = original_path
        await controller.shutdown()
        if replacement_database is not None:
            replacement_database.close_connection()
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["provider", "native_drain"])
async def test_direct_source_fence_survives_later_durable_awaits(
    tmp_path, monkeypatch, phase
):
    from Tests.Chat.test_console_first_send_atomicity import _controller
    from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState

    database, store, controller, gateway = _controller(tmp_path)
    session = store.sessions()[0]
    scope = Scope()
    scope.release.set()
    from tldw_chatbook.Chat.attachment_core import PendingAttachment

    attachment = PendingAttachment(
        file_path="image.png",
        display_name="image.png",
        file_type="image",
        insert_mode="attachment",
        data=b"png",
        mime_type="image/png",
    )
    assert store.add_pending_attachment(session.id, attachment)
    configuration = replace(
        controller.resolve_turn_configuration_snapshot(session.id),
        capabilities={"vision": True, "max_history_images": 1},
    )
    controller._turn_context_provider = lambda _: configuration
    controller._agent_runtime_enabled = False
    controller.app = SimpleNamespace(prompt_scope_service=scope)
    service = controller.prompt_mixin_service()
    service.replace_preferences(
        session.id,
        expected_revision=0,
        value=MixinPreferences(entries=(MixinSelection(A, True, "A", "before"),)),
    )
    store.set_session_draft(session.id, "keep durable draft")
    entered, release = asyncio.Event(), asyncio.Event()
    if phase == "provider":
        original = gateway.resolve_for_send

        async def resolve(selection):
            result = await original(selection)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(gateway, "resolve_for_send", resolve)
    else:
        original = store.drain_agent_handoff

        async def drain(session_id):
            result = await original(session_id)
            preparation = store.preparation_for_session(session_id)
            if (
                not entered.is_set()
                and preparation is not None
                and preparation.state is ConsoleTurnPreparationState.COMMITTING
            ):
                assert (
                    controller._ordinary_native_commit_owner(preparation.preparation_id)
                    is not None
                )
                entered.set()
                await release.wait()
            return result

        monkeypatch.setattr(store, "drain_agent_handoff", drain)
    task = asyncio.create_task(
        controller.submit_draft("keep durable draft", session_id=session.id)
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not task.done()
        scope.database = object()
        release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        assert isinstance(result, MixinSourceChangedError)
        assert gateway.calls == 0
        assert store.session_draft(session.id) == "keep durable draft"
        assert store.pending_attachments(session.id) == [attachment]
        assert session.persisted_conversation_id is None
        assert store.preparation_for_session(session.id) is None
        assert controller._ordinary_native_commit_tasks(session.id) == ()
        assert all(
            row.status == "failed" for row in store.messages_for_session(session.id)
        )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await controller.shutdown()
        for live in store.sessions():
            store.close_session(live.id)
        store.end_app_runtime()
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["paused", "resumed_hook"])
async def test_pending_direct_continuation_keeps_source_fence(
    tmp_path, monkeypatch, phase
):
    from Tests.Chat.test_console_first_send_atomicity import _controller
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsolePreparationPauseKind,
        ConsoleTurnPreparationState,
    )

    database, store, controller, gateway = _controller(tmp_path)
    session = store.sessions()[0]
    scope = Scope()
    scope.release.set()
    configuration = controller.resolve_turn_configuration_snapshot(session.id)
    controller._turn_context_provider = lambda _: configuration
    controller._agent_runtime_enabled = False
    controller.app = SimpleNamespace(prompt_scope_service=scope)
    controller.prompt_mixin_service().replace_preferences(
        session.id,
        expected_revision=0,
        value=MixinPreferences(entries=(MixinSelection(A, True, "A", "before"),)),
    )
    store.set_session_draft(session.id, "keep paused draft")

    async def hold(**kwargs):
        preparation = kwargs["preparation"]
        paused = controller._pause_prepared_commit(
            preparation.preparation_id, ConsolePreparationPauseKind.PERSISTENCE
        )
        assert paused.state is ConsoleTurnPreparationState.PAUSED
        return ConsoleSubmitResult(False, False, preparation_id=paused.preparation_id)

    monkeypatch.setattr(controller, "_compaction_admission_check", hold)
    resumed = None
    release = asyncio.Event()
    try:
        first = await controller.submit_draft(
            "keep paused draft", session_id=session.id
        )
        assert not first.accepted
        preparation_id = first.preparation_id
        pending = controller._prepared_send_continuations[preparation_id]
        assert pending.prompt_mixin_capture is not None
        assert "prompt_mixin_capture" not in repr(pending)
        if phase == "paused":
            scope.database = object()
            result = await controller._continue_prepared_submission(preparation_id)
            assert not result.accepted
        else:
            entered = asyncio.Event()

            async def hook():
                entered.set()
                await release.wait()

            monkeypatch.setattr(controller, "hook_admission_reason", hook)
            resumed = asyncio.create_task(
                controller._continue_prepared_submission(preparation_id)
            )
            await asyncio.wait_for(entered.wait(), 5)
            assert not resumed.done()
            scope.database = object()
            release.set()
            result = (await asyncio.gather(resumed, return_exceptions=True))[0]
            assert isinstance(result, ConsoleSubmitResult) and not result.accepted
        assert gateway.calls == 0
        assert store.session_draft(session.id) == "keep paused draft"
        assert store.preparation_for_session(session.id) is None
        assert scope.calls == [(A.record_uuid,)]
        assert all(
            row.status == "failed" for row in store.messages_for_session(session.id)
        )
    finally:
        release.set()
        if resumed is not None:
            await asyncio.gather(resumed, return_exceptions=True)
        await controller.shutdown()
        for live in store.sessions():
            store.close_session(live.id)
        store.end_app_runtime()
        database.close_connection()


@pytest.mark.asyncio
async def test_delayed_accepted_layout_never_overwrites_newer_user_intent(case):
    case.scope.release.set()
    before = case.service.preferences(case.a.id)
    case.scope.records[A.record_uuid]["prompt_definition"]["placement"] = "after"
    snapshot = await case.service.capture(case.a.id, expected_revision=before.revision)
    newer = case.service.replace_preferences(
        case.a.id,
        expected_revision=before.revision,
        value=replace(
            before, entries=tuple(row for row in before.entries if row.source != A)
        ),
    )
    case.service.publish_accepted_usage(
        case.a.id, snapshot, observed_revision=before.revision
    )
    facts = case.service.preferences(case.a.id)
    assert facts.revision == newer.revision
    assert tuple(
        (row.source, row.placement, row.enabled) for row in facts.entries
    ) == tuple((row.source, row.placement, row.enabled) for row in newer.entries)
