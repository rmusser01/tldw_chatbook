"""Exact-target Persona assignment commits before publishing live identity."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)


@pytest.fixture
def context(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat.sqlite", "assignment-test")
    workspace_db = WorkspaceDB(tmp_path / "workspace.sqlite", "assignment-test")
    registry = LocalWorkspaceRegistryService(workspace_db)
    personas = LocalCharacterPersonaService(
        db, persona_store_path=tmp_path / "personas.json"
    )
    for persona_id in ("guide", "other"):
        personas.create_persona_profile(
            {
                "id": persona_id,
                "name": persona_id.title(),
                "system_prompt": f"Be {persona_id}.",
            }
        )
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=controller),
        local_character_persona_service=personas,
        workspace_registry_service=registry,
    )
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="test-model", temperature=0.42
        ),
        assistant_kind="generic",
    )
    store.persist_session_if_needed(session.id)
    yield SimpleNamespace(
        app=app,
        db=db,
        store=store,
        controller=controller,
        session=session,
        personas=personas,
        registry=registry,
    )
    db.close_connection()
    workspace_db.close()


def prepare(ctx, choice="guide", *, target=None, binding=None):
    from tldw_chatbook.Chat.console_persona_assignment import (
        prepare_buddy_persona_assignment,
    )

    target = target or ctx.session
    return prepare_buddy_persona_assignment(
        ctx.app, binding or BuddyBinding.for_session(target), target, choice
    )


@pytest.mark.asyncio
async def test_prepare_is_read_only_and_apply_round_trips_exact_identity(context):
    ctx = context
    original = replace(ctx.session)
    before = ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id)
    epoch = ctx.store.conversation_context_epoch(ctx.session.id)
    speech = ctx.store.speech_preference_epoch(ctx.session.id)
    assignment = prepare(ctx)
    assert ctx.session == original
    assert not ctx.store._fork_source_transitions
    assert ctx.db.get_conversation_by_id(original.persisted_conversation_id) == before
    sibling = ctx.store.create_session(
        settings=original.settings, assistant_kind="generic"
    )
    await assignment.apply()
    assert ctx.store.active_session_id == sibling.id
    assert sibling.assistant_kind == "generic"
    row = ctx.db.get_conversation_by_id(original.persisted_conversation_id)
    assert row["assistant_kind"] == ctx.session.assistant_kind == "persona"
    assert row["assistant_id"] == ctx.session.assistant_id == "guide"
    assert (
        row["persona_memory_mode"]
        == ctx.session.settings.persona_memory_mode
        == "read_only"
    )
    assert row["system_prompt"] == ctx.session.settings.system_prompt
    assert "Be guide." in row["system_prompt"]
    metadata = json.loads(row["metadata"])
    assert metadata["console_session_settings"]["character_label"] == "Guide"
    assert metadata["console_session_settings"]["temperature"] == 0.42
    assert row["version"] == before["version"] + 1
    assert ctx.session.identity_revision == original.identity_revision + 1
    assert (
        ctx.session.generation_settings_revision
        == original.generation_settings_revision + 1
    )
    assert ctx.store.conversation_context_epoch(ctx.session.id) > epoch
    assert ctx.store.speech_preference_epoch(ctx.session.id) > speech
    assert ctx.store.messages_for_session(ctx.session.id) == []


@pytest.mark.asyncio
async def test_sqlite_failure_leaves_prompt_settings_identity_and_live_memory_unchanged(
    context,
):
    ctx = context
    assignment = prepare(ctx)
    before = ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id)
    original = replace(ctx.session)
    with ctx.db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER reject_persona BEFORE UPDATE OF assistant_id ON conversations BEGIN SELECT RAISE(ABORT, 'assignment blocked'); END"
        )
    with pytest.raises(CharactersRAGDBError, match="assignment blocked"):
        await assignment.apply()
    assert (
        ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id) == before
    )
    assert ctx.session == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "revised",
        "deleted",
        "inactive",
        "settings",
        "identity",
        "repurposed",
        "busy",
        "pending",
        "durable",
    ],
)
async def test_apply_rejects_stale_persona_or_target_without_overwriting(
    context, change
):
    ctx = context
    assignment = prepare(ctx)
    if change == "revised":
        ctx.personas.update_persona_profile(
            "guide", {"system_prompt": "New instructions."}
        )
    elif change == "deleted":
        ctx.personas.delete_persona_profile("guide")
    elif change == "inactive":
        ctx.personas.update_persona_profile("guide", {"is_active": False})
    elif change == "settings":
        ctx.store.replace_session_settings(
            ctx.session.id, replace(ctx.session.settings, temperature=0.8)
        )
    elif change == "identity":
        ctx.session.identity_revision += 1
    elif change == "repurposed":
        ctx.session.conversation_binding_revision += 1
    elif change == "busy":
        ctx.controller._run_states[ctx.session.id] = ConsoleRunState(
            ConsoleRunStatus.STREAMING
        )
    elif change == "pending":
        ctx.controller.add_pending_round(ctx.session.id, "decision")
    elif change == "durable":
        row = ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id)
        ctx.db.update_conversation(row["id"], {"title": "New title"}, row["version"])
    before = ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id)
    original = replace(ctx.session)
    with pytest.raises(ValueError):
        await assignment.apply()
    assert ctx.session == original
    assert (
        ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id) == before
    )


@pytest.mark.asyncio
async def test_explicit_none_clears_persona_without_changing_generation_choices(
    context,
):
    ctx = context
    await prepare(ctx).apply()
    await prepare(ctx, "#none").apply()
    assert ctx.session.assistant_kind == "generic"
    assert ctx.session.assistant_id == "console"
    assert ctx.session.settings.system_prompt is None
    assert ctx.session.settings.character_label is None
    assert ctx.session.settings.persona_memory_mode is None
    assert ctx.session.settings.temperature == 0.42
    row = ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id)
    assert row["assistant_kind"] is None
    assert row["system_prompt"] is None
    assert row["persona_memory_mode"] is None


@pytest.mark.asyncio
async def test_workspace_default_preserves_policy_and_explicit_none(context):
    ctx = context
    workspace = ctx.registry.create_workspace(
        workspace_id="workspace",
        name="Workspace",
        assistant_defaults=WorkspaceAssistantDefaults(
            assistant_id="other", tool_policy_profile_id="existing-profile"
        ),
    )
    binding = BuddyBinding(kind="workspace", target_id=workspace.workspace_id)
    assignment = prepare(ctx, target=workspace, binding=binding)
    assert ctx.registry.get_workspace(workspace.workspace_id) == workspace
    await assignment.apply()
    changed = ctx.registry.get_workspace(workspace.workspace_id)
    assert changed.assistant_defaults == replace(
        workspace.assistant_defaults, assistant_id="guide"
    )
    await prepare(ctx, "#none", target=changed, binding=binding).apply()
    cleared = ctx.registry.get_workspace(workspace.workspace_id)
    assert cleared.assistant_defaults is None
    assert cleared.assistant_defaults_explicit_none
    assert ctx.session.assistant_kind == "generic"


@pytest.mark.asyncio
async def test_unchanged_and_same_persona_do_not_write(context):
    ctx = context
    await prepare(ctx).apply()
    before = ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id)
    original = replace(ctx.session)
    await prepare(ctx, "#unchanged").apply()
    await prepare(ctx).apply()
    assert ctx.session == original
    assert (
        ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id) == before
    )


@pytest.mark.parametrize("operation", ["set", "clear"])
def test_workspace_registry_compare_and_set_rejects_stale_default(context, operation):
    ctx = context
    original = ctx.registry.create_workspace(
        workspace_id="workspace", name="Workspace", assistant_defaults=None
    )
    newer = WorkspaceAssistantDefaults(assistant_id="other")
    ctx.registry.set_assistant_defaults(original.workspace_id, newer)
    with pytest.raises(WorkspaceRegistryServiceError, match="changed"):
        if operation == "set":
            ctx.registry.set_assistant_defaults(
                original.workspace_id,
                WorkspaceAssistantDefaults(assistant_id="guide"),
                expected_record=original,
            )
        else:
            ctx.registry.clear_assistant_defaults(
                original.workspace_id, expected_record=original
            )
    assert ctx.registry.get_workspace(original.workspace_id).assistant_defaults == newer


@pytest.mark.asyncio
async def test_queued_prompt_refuses_assignment(context):
    ctx = context
    queue = ctx.controller.prompt_queue_registry
    chain = queue.begin_chain(
        ctx.session.id,
        context_epoch=ctx.store.conversation_context_epoch(ctx.session.id),
        expected_revision=0,
    )
    added = queue.admit(
        ctx.session.id, text="Queued work", expected_revision=chain.snapshot.revision
    )
    assert added.applied
    with pytest.raises(ValueError, match="queued"):
        prepare(ctx)
    assert queue.snapshot(ctx.session.id).total_count == 1


@pytest.mark.parametrize(
    "kind", ["approval", "skill_install", "skill_script", "question", "worktree_merge"]
)
def test_parked_decision_without_run_badge_refuses_assignment(context, kind):
    ctx = context
    ctx.controller._interrupt_host.park_round_payload(
        kind, "decision", {"request_id": "decision", "session_id": ctx.session.id}
    )
    with pytest.raises(ValueError, match="decision"):
        prepare(ctx)


@pytest.mark.parametrize("ownership", ["character", "server"])
def test_character_and_remote_ownership_cannot_be_reassigned(context, ownership):
    ctx = context
    binding = BuddyBinding.for_session(ctx.session)
    if ownership == "character":
        ctx.session.assistant_kind = "character"
        ctx.session.character_id = 1
    else:
        ctx.session.runtime_backend = "server"
    with pytest.raises(ValueError):
        prepare(ctx, binding=binding)


@pytest.mark.asyncio
async def test_unsaved_temporary_assignment_never_creates_durable_conversation(context):
    ctx = context
    session = ctx.store.create_session(
        settings=ctx.session.settings, ephemeral=True, assistant_kind="generic"
    )
    before = (
        ctx.db.get_connection()
        .execute("SELECT COUNT(*) FROM conversations")
        .fetchone()[0]
    )
    await prepare(ctx, target=session).apply()
    assert session.assistant_id == "guide"
    assert session.persisted_conversation_id is None
    assert (
        ctx.db.get_connection()
        .execute("SELECT COUNT(*) FROM conversations")
        .fetchone()[0]
        == before
    )


@pytest.mark.asyncio
async def test_different_persona_never_inherits_unconfirmed_memory_writes(context):
    ctx = context
    await prepare(ctx).apply()
    ctx.session.persona_memory_mode = "read_write"
    ctx.session.settings = replace(
        ctx.session.settings, persona_memory_mode="read_write"
    )
    with pytest.raises(ValueError, match="memory"):
        prepare(ctx, "other")
    workspace = ctx.registry.create_workspace(
        workspace_id="rw",
        name="RW",
        assistant_defaults=WorkspaceAssistantDefaults(
            assistant_id="guide", persona_memory_mode="read_write"
        ),
        confirm_read_write=True,
    )
    with pytest.raises(ValueError, match="memory"):
        prepare(
            ctx,
            "other",
            target=workspace,
            binding=BuddyBinding(kind="workspace", target_id="rw"),
        )


@pytest.mark.asyncio
async def test_persona_change_does_not_persist_a_live_only_endpoint(context):
    from tldw_chatbook.Chat.console_session_endpoint_policy import (
        ConsoleEphemeralEndpointPolicy,
    )

    ctx = context
    endpoint = "http://ephemeral-only.example:12345"
    ctx.session.settings = replace(ctx.session.settings, base_url=endpoint)
    policy = ConsoleEphemeralEndpointPolicy(
        provider="llama_cpp", model="test-model", base_url=endpoint
    )
    ctx.session.ephemeral_endpoint_policy = policy
    await prepare(ctx).apply()
    assert ctx.session.settings.base_url == endpoint
    assert ctx.session.ephemeral_endpoint_policy is policy
    row = ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id)
    assert endpoint not in row["metadata"]


def test_already_stale_durable_identity_refuses_preparation(context):
    ctx = context
    row = ctx.db.get_conversation_by_id(ctx.session.persisted_conversation_id)
    ctx.db.update_conversation(
        row["id"],
        {
            "assistant_kind": "persona",
            "assistant_id": "other",
            "persona_memory_mode": "read_only",
        },
        row["version"],
    )
    with pytest.raises(ValueError, match="saved|Saved"):
        prepare(ctx)
    assert ctx.session.assistant_kind == "generic"


def test_live_preparation_refuses_assignment_even_before_run_badge(context):
    from Tests.Chat.test_console_turn_preparation import _preparation_values
    from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparation

    ctx = context
    values = _preparation_values(session_id=ctx.session.id)
    ctx.store.begin_preparation(ConsoleTurnPreparation(**values))
    with pytest.raises(ValueError, match="run"):
        prepare(ctx)


@pytest.mark.asyncio
async def test_pending_settings_writer_refuses_assignment(context):
    ctx = context
    lifecycle = ctx.store._settings_persistence_lifecycles[ctx.session.id]
    async with lifecycle.lock:
        with pytest.raises(ValueError, match="saving"):
            prepare(ctx)


@pytest.mark.asyncio
async def test_fork_snapshot_is_fenced_until_atomic_persona_publication(
    context, monkeypatch
):
    ctx = context
    message = ctx.store.append_message(
        ctx.session.id, role=ConsoleMessageRole.USER, content="Keep this transcript"
    )
    ctx.store.persist_message_if_needed(message.id)
    assert ctx.store.fork_eligibility(message.id).eligible
    writer = ctx.db.update_conversation

    def write_with_boundary_observation(*args, **kwargs):
        assert ctx.session.assistant_kind == "generic"
        assert ctx.session.settings.system_prompt is None
        assert not ctx.store.fork_eligibility(message.id).eligible
        return writer(*args, **kwargs)

    monkeypatch.setattr(ctx.db, "update_conversation", write_with_boundary_observation)
    await prepare(ctx).apply()
    assert ctx.store.fork_eligibility(message.id).eligible
    assert (
        ctx.store.messages_for_session(ctx.session.id)[0].content
        == "Keep this transcript"
    )
    assert ctx.store._fork_configuration_snapshot(ctx.session).assistant_id == "guide"


@pytest.mark.asyncio
async def test_workspace_stale_or_deleted_persona_cannot_change_defaults(context):
    ctx = context
    workspace = ctx.registry.create_workspace(
        workspace_id="workspace", name="Workspace", assistant_defaults=None
    )
    assignment = prepare(
        ctx,
        target=workspace,
        binding=BuddyBinding(kind="workspace", target_id="workspace"),
    )
    ctx.personas.delete_persona_profile("guide")
    with pytest.raises(ValueError, match="unavailable"):
        await assignment.apply()
    assert ctx.registry.get_workspace("workspace") == workspace


@pytest.mark.asyncio
async def test_prepared_workspace_assignment_rejects_replaced_profile_service(context):
    ctx = context
    workspace = ctx.registry.create_workspace(
        workspace_id="workspace", name="Workspace", assistant_defaults=None
    )
    assignment = prepare(
        ctx,
        target=workspace,
        binding=BuddyBinding(kind="workspace", target_id="workspace"),
    )
    ctx.app.workspace_registry_service = object()
    with pytest.raises(ValueError, match="Workspace"):
        await assignment.apply()
    assert ctx.registry.get_workspace("workspace") == workspace
